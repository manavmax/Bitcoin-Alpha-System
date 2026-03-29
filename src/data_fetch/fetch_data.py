#!/usr/bin/env python3
"""
Robust data fetcher with fallback logic for on-chain metrics.

Place at: src/data_fetch/fetch_data.py
Run with: python src/data_fetch/fetch_data.py

Environment variables (optional):
 - COINMETRICS_API_KEY
 - CHARTINSPECT_KEY
 - DUNE_API_KEY
 - BITQUERY_KEY
"""

import os
import time
import json
import math
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# APIs & keys from env (optional)
COINMETRICS_KEY = os.getenv("COINMETRICS_API_KEY", "")
CHARTINSPECT_KEY = os.getenv("CHARTINSPECT_KEY", "")
DUNE_KEY = os.getenv("DUNE_API_KEY", "")
BITQUERY_KEY = os.getenv("BITQUERY_KEY", "")

# Default date range for timeseries requests (last ~4 years)
END_DATE = datetime.utcnow().date()
START_DATE = END_DATE - timedelta(days=4*365)

# Which onchain metrics we require (tune this list for your models)
REQUIRED_ONCHAIN_METRICS = [
    "mvrv-data",
    "sopr",
    "nupl",
    "active-addresses",
    "transaction-count",
    "hashrate",
    "miner-revenue",
    "tx_volume_usd",
    "difficulty",
    "tx_count",
    "tx_revenue_usd"
]

# Map metrics to preferred provider(s). Order matters (attempt in order).
# Valid providers: "coinmetrics", "chartinspect", "blockchair", "bitquery", "dune", "local"
METRIC_SOURCE_MAP = {
    # default: try CoinMetrics first then ChartInspect then Blockchair etc.
    # If a metric has a special mapping name on CoinMetrics specify here.
    "mvrv-data": ["chartinspect", "coinmetrics", "blockchair"],
    "sopr": ["chartinspect", "coinmetrics"],
    "nupl": ["chartinspect", "coinmetrics"],
    "active-addresses": ["coinmetrics", "chartinspect", "bitquery", "blockchair"],
    "transaction-count": ["coinmetrics", "chartinspect", "bitquery", "blockchair"],
    "hashrate": ["coinmetrics", "blockchair", "chartinspect"],
    "miner-revenue": ["coinmetrics", "chartinspect"],
    "tx_volume_usd": ["coinmetrics", "chartinspect"],
    "difficulty": ["coinmetrics", "blockchair"],
    "tx_count": ["coinmetrics", "blockchair"],
    "tx_revenue_usd": ["coinmetrics"],
}

# CoinMetrics metric name map (if differs). Use empty to attempt the same name.
COINMETRICS_NAME_MAP = {
    "tx_volume_usd": "TxTfrValUSD",
    "tx_count": "TxCnt",
    "difficulty": "Difficulty",
    "hashrate": "HashRate",
    "miner-revenue": "MinerRevenueUSD",
    "tx_revenue_usd": "TxTfrValUSD",
    "active-addresses": "ActiveAddresses",
    # others left blank: coinmetrics may not have direct mapping for 'mvrv-data' etc
}

# ChartInspect endpoints: uses path `/api/v1/onchain/{metric}`
CHARTINSPECT_BASE = "https://chartinspect.com/api/v1/onchain"

# CoinMetrics API (free endpoints exist; some require API key)
COINMETRICS_BASE = "https://api.coinmetrics.io/v4"

# Blockchair snapshot endpoints (limited) - used as fallback where available
# Bitquery GraphQL used as optional fallback (requires API key)
BITQUERY_GRAPHQL = "https://graphql.bitquery.io/"

# Local fallback folder (you can save your own backups here)
LOCAL_BACKUP_DIR = os.path.join("data", "backup")
os.makedirs(LOCAL_BACKUP_DIR, exist_ok=True)

# -------------------------
# HELPERS
# -------------------------
def save_csv(df: pd.DataFrame, name: str):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    try:
        df.to_csv(path, index=False)
    except Exception:
        # try safer (no index)
        df.to_csv(path, index=False)
    print(f"✅ Saved: {path}")

def safe_request(url, params=None, headers=None, timeout=10, verb="GET"):
    tries = 0
    while tries < 4:
        try:
            if verb == "GET":
                r = requests.get(url, params=params, headers=headers, timeout=timeout)
            else:
                r = requests.post(url, json=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            msg = f"{code} {e}" if code else str(e)
            print(f"⚠️ Request failed ({msg}). attempt {tries+1}/4")
            # If 4xx likely won't change on retry (but sometimes a transient 429 or 400 with params)
            if code and 400 <= code < 500 and code != 429:
                # don't retry many times for deterministic client errors
                return e.response
            tries += 1
            time.sleep(1.5 * tries)
        except requests.RequestException as e:
            tries += 1
            print(f"⚠️ Request exception {e}. retrying {tries}/4")
            time.sleep(1.5 * tries)
    return None

def parse_timeseries_json(resp_json, time_keys=("time", "date", "timestamp"), value_keys=("value","val","metric_value","v")):
    """
    Try to robustly parse a JSON payload into DataFrame with columns ['date','value'] or
    a wide frame if multiple series present.
    """
    # If already a DataFrame convertible structure
    if isinstance(resp_json, list):
        try:
            df = pd.DataFrame(resp_json)
            # try to detect time/value
            for tk in time_keys:
                if tk in df.columns:
                    df = df.rename(columns={tk: "date"})
                    break
            for vk in value_keys:
                if vk in df.columns:
                    df = df.rename(columns={vk: "value"})
                    break
            # normalize date
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], utc=True)
            return df
        except Exception:
            return None
    if isinstance(resp_json, dict):
        # Some APIs package into {"data": [...]}
        if "data" in resp_json and isinstance(resp_json["data"], (list, dict)):
            return parse_timeseries_json(resp_json["data"], time_keys, value_keys)
        # coinmetrics format maybe { "data": { "series": [...] } } etc
        # If it's nested mapping of metric->list:
        # e.g., { metric: [{"time":..., "value":...}, ...], ...}
        # convert to wide form
        # Try to find lists inside
        lists = {k:v for k,v in resp_json.items() if isinstance(v, list)}
        if lists:
            # try to concat along date
            frames = []
            for k,v in lists.items():
                try:
                    dfk = pd.DataFrame(v)
                    # detect date column
                    for tk in time_keys:
                        if tk in dfk.columns:
                            dfk = dfk.rename(columns={tk: "date"})
                            break
                    # detect value column
                    for vk in value_keys:
                        if vk in dfk.columns:
                            dfk = dfk.rename(columns={vk: "value"})
                            break
                    if "date" in dfk.columns:
                        dfk["date"] = pd.to_datetime(dfk["date"], utc=True)
                        dfk = dfk[["date", "value"]]
                        dfk = dfk.rename(columns={"value": k})
                        frames.append(dfk)
                except Exception:
                    continue
            if frames:
                merged = frames[0]
                for f in frames[1:]:
                    merged = pd.merge(merged, f, on="date", how="outer")
                return merged
        # fallback: try to interpret single mapping of 'time'->'value' etc
        # e.g., {"time": [...], "value": [...]}
        if all(isinstance(resp_json.get(k), list) for k in ("time","value")):
            df = pd.DataFrame({"date": resp_json["time"], "value": resp_json["value"]})
            df["date"] = pd.to_datetime(df["date"], utc=True)
            return df
    return None

def ensure_timeseries_df(df):
    # ensure date column and sort
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","value"])
    # try common name guesses
    if "date" not in df.columns:
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower():
                df = df.rename(columns={c: "date"})
                break
    # collapse to first numeric column as value if 'value' not present
    if "value" not in df.columns:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 1:
            df = df[["date", numeric_cols[0]]].rename(columns={numeric_cols[0]: "value"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)
    return df

# -------------------------
# PRIMARY FETCHERS
# -------------------------
def fetch_ohlcv_binance(symbol="BTCUSDT", interval="1d", limit=2000):
    print("\n📥 Fetching OHLCV from Binance...")
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = safe_request(url, params=params)
    if not r:
        print("⚠️ Binance request failed.")
        return pd.DataFrame()
    data = r.json()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time","quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df = df[["open_time","open","high","low","close","volume"]]
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df[["date","open","high","low","close","volume"]].copy()
    save_csv(out.rename(columns={"date":"open_time"}), "btc_price_daily")  # legacy name used in your repo
    return out

def compute_volatility_from_ohlcv(ohlcv_df):
    print("⚙️ Computing volatility...")
    if ohlcv_df.empty:
        return pd.DataFrame()
    df = ohlcv_df.copy()
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std()
    out = df[["open_time","volatility"]].dropna() if "open_time" in df.columns else df[["date","volatility"]].dropna()
    # Save volatility in expected file name
    save_csv(out.rename(columns={"open_time":"date"}).reset_index(drop=True), "volatility")
    return out

def fetch_derivatives_binance_funding(symbol="BTCUSDT", limit=1000):
    print("📥 Fetching derivatives (funding rate)...")
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}
    r = safe_request(url, params=params)
    if not r:
        print("⚠️ Derivatives fetch failed.")
        return pd.DataFrame()
    df = pd.DataFrame(r.json())
    if df.empty:
        print("⚠️ Empty funding response.")
        return df
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    out = df[["date","funding_rate"]].dropna()
    save_csv(out, "derivatives")
    return out

# -------------------------
# On-chain provider implementations
# -------------------------
def fetch_onchain_coinmetrics(metrics, start=START_DATE, end=END_DATE, asset="btc"):
    """
    Attempt to fetch multiple metrics from CoinMetrics community timeseries endpoint.
    Returns a DataFrame if successful (wide format).
    """
    print(" → Trying CoinMetrics...")
    # Build metrics list with coinmetrics names if provided
    cm_metrics = []
    for m in metrics:
        cm_name = COINMETRICS_NAME_MAP.get(m, "")
        cm_metrics.append(cm_name if cm_name else m)
    cm_metrics = [c for c in cm_metrics if c]
    params = {
        "assets": asset,
        "metrics": ",".join(cm_metrics),
        "start_time": start.strftime("%Y-%m-%d"),
        "end_time": end.strftime("%Y-%m-%d"),
        "frequency": "1d"
    }
    url = f"{COINMETRICS_BASE}/timeseries/asset-metrics"
    headers = {}
    if COINMETRICS_KEY:
        headers["Authorization"] = f"Bearer {COINMETRICS_KEY}"
    r = safe_request(url, params=params, headers=headers)
    if not r:
        return None
    try:
        payload = r.json()
    except Exception as e:
        print("⚠️ CoinMetrics parse error:", e)
        return None
    # coinmetrics returns {"data": [{"time":.., "asset":.., "metric":.., "value":..}, ...]} or timeseries mapping
    df = parse_timeseries_json(payload)
    if df is None:
        # try specialized handling: payload may have 'data' as list of rows (time, metric, value)
        if isinstance(payload, dict) and "data" in payload:
            try:
                rows = payload["data"]
                df = pd.DataFrame(rows)
                # pivot metric->columns if metric present
                if "metric" in df.columns and "value" in df.columns and "time" in df.columns:
                    df_piv = df.pivot_table(index="time", columns="metric", values="value").reset_index().rename(columns={"time":"date"})
                    df_piv["date"] = pd.to_datetime(df_piv["date"], utc=True)
                    return df_piv
            except Exception:
                return None
    return df

def fetch_onchain_chartinspect(metric, days=1095):
    """Fetch single metric from ChartInspect. Needs CHARTINSPECT_KEY for authenticated endpoints."""
    if not CHARTINSPECT_KEY:
        # still attempt (some endpoints might be public)
        headers = {}
    else:
        headers = {"X-API-Key": CHARTINSPECT_KEY}
    url = f"{CHARTINSPECT_BASE}/{metric}"
    params = {"days": days}
    r = safe_request(url, params=params, headers=headers)
    if not r:
        return None
    try:
        payload = r.json()
    except Exception as e:
        print("⚠️ ChartInspect parse error:", e)
        return None
    df = parse_timeseries_json(payload)
    return df

def fetch_onchain_blockchair(metric, asset="bitcoin"):
    """
    Blockchair provides some snapshots and statistics; not a perfect substitute but used as fallback
    Example: https://api.blockchair.com/bitcoin/stats
    """
    try:
        if metric in ("hashrate", "difficulty", "miner-revenue"):
            url = f"https://api.blockchair.com/{asset}/stats"
            r = safe_request(url)
            if not r:
                return None
            payload = r.json()
            # stats response contains a mapping; create a single-row dataframe with today's date and value
            val = None
            # heuristics
            if metric == "hashrate":
                val = payload.get("data", {}).get("hashrate", None)
            elif metric == "difficulty":
                val = payload.get("data", {}).get("difficulty", None)
            elif metric == "miner-revenue":
                val = payload.get("data", {}).get("average_miner_reward", None)  # heuristic
            if val is not None:
                df = pd.DataFrame([{"date": datetime.utcnow().date(), metric: val}])
                return df
    except Exception as e:
        print(f"⚠️ Blockchair fetch error for {metric}: {e}")
    return None

def fetch_onchain_bitquery(query_str, variables=None):
    """Minimal Bitquery GraphQL runner - requires BITQUERY_KEY"""
    if not BITQUERY_KEY:
        return None
    headers = {"X-API-KEY": BITQUERY_KEY, "Content-Type": "application/json"}
    payload = {"query": query_str, "variables": variables or {}}
    r = safe_request(BITQUERY_GRAPHQL, params=payload, headers=headers, verb="POST")
    if not r:
        return None
    try:
        return r.json()
    except Exception:
        return None

def fetch_onchain_dune(query_id):
    """Ask Dune API to run a saved query (requires DUNE_KEY). Returns DataFrame if successful.
       Note: Dune rate limits; also requires you to create / provide query IDs for specific metrics.
    """
    if not DUNE_KEY or not query_id:
        return None
    headers = {"x-dune-api-key": DUNE_KEY, "Content-Type": "application/json"}
    url = f"https://api.dune.com/api/v1/query/{query_id}/results"
    r = safe_request(url, headers=headers)
    if not r:
        return None
    try:
        payload = r.json()
        # Dune payload structure is nested; attempt to parse into rows
        if "result" in payload and "rows" in payload["result"]:
            df = pd.DataFrame(payload["result"]["rows"])
            # try to normalize date column
            return df
    except Exception as e:
        print("⚠️ Dune parse error:", e)
    return None

def read_local_backup(metric):
    backup_path = os.path.join(LOCAL_BACKUP_DIR, f"{metric}.csv")
    if os.path.exists(backup_path):
        try:
            df = pd.read_csv(backup_path)
            df = ensure_timeseries_df(df)
            print(f" ↳ Loaded local backup for {metric}")
            return df
        except Exception:
            return None
    return None

# -------------------------
# Orchestrator: try providers in order
# -------------------------
def fetch_metric_with_fallback(metric):
    """
    Try providers listed in METRIC_SOURCE_MAP[metric] in order until data found.
    Returns DataFrame or empty DataFrame.
    """
    providers = METRIC_SOURCE_MAP.get(metric, ["coinmetrics","chartinspect","blockchair","bitquery","local"])
    combined_df = None

    for provider in providers:
        try:
            if provider == "coinmetrics":
                df = fetch_onchain_coinmetrics([metric])
                if df is None:
                    continue
                df = ensure_timeseries_df(df)
                if df.empty:
                    continue
                # rename column to 'value' or metric
                if "value" in df.columns:
                    df = df.rename(columns={"value": metric})
                elif metric in df.columns:
                    # coinmetrics pivot case -> df[metric]
                    pass
                # Save and return
                save_csv(df.rename(columns={"date":"date"}), f"onchain_{metric}")
                return df
            elif provider == "chartinspect":
                df = fetch_onchain_chartinspect(metric)
                if df is None:
                    continue
                df = ensure_timeseries_df(df)
                if df.empty:
                    continue
                df = df.rename(columns={"value": metric}) if "value" in df.columns else df
                save_csv(df.rename(columns={"date":"date"}), f"onchain_{metric}")
                return df
            elif provider == "blockchair":
                df = fetch_onchain_blockchair(metric)
                if df is None:
                    continue
                df = ensure_timeseries_df(df)
                if df.empty:
                    continue
                save_csv(df.rename(columns={"date":"date"}), f"onchain_{metric}")
                return df
            elif provider == "bitquery":
                # Noting: bitquery usage requires custom query mapping per metric - skip unless BITQUERY_KEY provided
                if not BITQUERY_KEY:
                    continue
                # For simplicity we skip implementing per-metric query templates here. User can expand.
                payload = fetch_onchain_bitquery(None)
                if payload:
                    df = parse_timeseries_json(payload)
                    if df is not None and not df.empty:
                        save_csv(df.rename(columns={"date":"date"}), f"onchain_{metric}")
                        return df
                continue
            elif provider == "dune":
                # User must populate DUNE_QUERIES mapping externally to point metric -> dune query id
                DUNE_QUERIES = {}  # user fill
                qid = DUNE_QUERIES.get(metric)
                if not qid or not DUNE_KEY:
                    continue
                df = fetch_onchain_dune(qid)
                if df is not None and not df.empty:
                    df = ensure_timeseries_df(df)
                    save_csv(df.rename(columns={"date":"date"}), f"onchain_{metric}")
                    return df
                continue
            elif provider == "local":
                df = read_local_backup(metric)
                if df is not None and not df.empty:
                    save_csv(df.rename(columns={"date":"date"}), f"onchain_{metric}")
                    return df
                continue
        except Exception as e:
            print(f"⚠️ Failed provider {provider} for {metric}: {e}")
            continue

    # If we reached here, all providers failed -> save empty template file and warn
    empty_df = pd.DataFrame(columns=["date", metric])
    save_csv(empty_df, f"onchain_{metric}")
    print(f"⚠️ {metric} could not be fetched from any provider (empty file saved)")
    return empty_df

# -------------------------
# HIGH LEVEL: fetch all required onchain metrics
# -------------------------
def fetch_all_onchain_metrics():
    print("\n📥 Fetching on-chain data (multi-provider fallback)...")
    collected = {}
    for m in REQUIRED_ONCHAIN_METRICS:
        print(f" - fetching {m} ...")
        df = fetch_metric_with_fallback(m)
        # ensure standardized columns
        df = ensure_timeseries_df(df)
        if "date" in df.columns and (m not in df.columns or df.columns.tolist() == ["date","value"]):
            # rename 'value' column to metric name
            if "value" in df.columns:
                df = df.rename(columns={"value": m})
        collected[m] = df

    # Attempt to align/merge into a single wide DataFrame by date
    merged = None
    for m, df in collected.items():
        if df is None or df.empty:
            continue
        # if df has metric column name instead of 'value', keep as-is
        # normalize
        if "date" not in df.columns:
            # try to detect a date-like column
            df = ensure_timeseries_df(df)
        if merged is None:
            merged = df.copy()
        else:
            merged = pd.merge(merged, df, on="date", how="outer")
    if merged is None:
        merged = pd.DataFrame(columns=["date"] + REQUIRED_ONCHAIN_METRICS)
    save_csv(merged, "onchain_all")
    return collected, merged

# -------------------------
# Macro fetching (unchanged, uses yfinance)
# -------------------------
def fetch_macro_yfinance():
    print("\n📥 Fetching macro data (yfinance)...")
    # Add the tickers your macro features require
    tickers = {
        "^TNX": "10Y",       # US 10Y yield
        "^VIX": "VIX",
        "DX-Y.NYB": "DXY"
    }
    dfs = []
    for tk, label in tickers.items():
        print(f"  - {tk}")
        try:
            df = yf.download(tk, period="5y", interval="1d", progress=False)
        except Exception:
            df = yf.download(tk, period="5y", interval="1d", progress=False)
        if df is None or df.empty:
            continue
        df = df.reset_index()[["Date","Close"]].rename(columns={"Date":"date","Close":label})
        df["date"] = pd.to_datetime(df["date"], utc=True)
        dfs.append(df)
    if not dfs:
        out = pd.DataFrame()
    else:
        macro = dfs[0]
        for d in dfs[1:]:
            macro = pd.merge(macro, d, on="date", how="outer")
        macro = macro.sort_values("date").reset_index(drop=True)
        save_csv(macro, "macro")
        return macro
    save_csv(pd.DataFrame(), "macro")
    return pd.DataFrame()

# -------------------------
# MAIN
# -------------------------
def main():
    print("📡 Fetching live data...")
    # 1. OHLCV
    ohlcv = fetch_ohlcv_binance()
    # compatibility with some of your scripts that expect 'open_time'
    if not ohlcv.empty:
        # rename 'date' -> 'open_time' to match older code
        ohlcv_out = ohlcv.rename(columns={"date":"open_time"})
        save_csv(ohlcv_out, "btc_price_daily")

    # 2. Volatility features
    compute_volatility_from_ohlcv(ohlcv)

    # 3. Derivatives / funding
    fetch_derivatives_binance_funding()

    # 4. On-chain metrics (multi-provider fallback)
    collected_onchain, merged_onchain = fetch_all_onchain_metrics()

    # 5. Macro
    fetch_macro_yfinance()

    print("\n✅ Data fetching pipeline complete.")
    print("Files saved in:", os.path.abspath(DATA_DIR))
    print("If any metric required by your model is missing, edit REQUIRED_ONCHAIN_METRICS or METRIC_SOURCE_MAP and re-run.")
    return

if __name__ == "__main__":
    main()
