"""
CoinMetrics derivatives data for Model 3 (supplementary raw inputs).

Tier notes (verified against the CoinMetrics v4 API):
  - Derivatives market data (open interest / funding rates / liquidations) is
    Pro (paid) tier only: the free Community API returns 403 for every
    derivatives market, so this script runs only when COINMETRICS_API_KEY is
    set. Without a key it exits cleanly and Model 3 keeps using
    Coinalyze + Binance (its primary sources anyway).
  - Uses the modern v4 per-market endpoints, NOT the old asset-metrics names
    (OpenInterest/FundingRate/... no longer exist and return HTTP 400):
      /timeseries/market-openinterest   -> value_usd (fallback: contract_count)
      /timeseries/market-funding-rates  -> rate (decimal fraction per interval)
      /timeseries/market-liquidations   -> amount * price, side=buy|sell
        (side 'sell' closes longs, side 'buy' closes shorts)

Output columns match what merge_derivatives_sources.py expects:
date, open_interest, funding_rate, long_liq, short_liq
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


COINMETRICS_KEY = os.getenv("COINMETRICS_API_KEY", "")

# Derivatives endpoints are paid-tier market data; always the Pro host.
COINMETRICS_BASE = "https://api.coinmetrics.io/v4"

OUT_DIR = Path("model_3_derivatives_flow/raw/coinmetrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKETS = ["binance-BTCUSDT-future"]
START_TIME = "2017-01-01"
PAGE_SIZE = 10000


def fetch_market_rows(endpoint: str, extra_params: dict | None = None) -> pd.DataFrame:
    """Fetch all rows for MARKETS from a market-* timeseries endpoint (paginated)."""
    url = f"{COINMETRICS_BASE}/timeseries/{endpoint}"
    params = {
        "markets": ",".join(MARKETS),
        "start_time": START_TIME,
        "end_time": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "page_size": PAGE_SIZE,
        "api_key": COINMETRICS_KEY,
    }
    if extra_params:
        params.update(extra_params)

    rows: list = []
    token = None
    for _ in range(200):  # safety cap: 200 × 10k rows
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        page = payload.get("data", [])
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected response shape from {endpoint}: keys={list(payload.keys())}")
        rows.extend(page)
        token = payload.get("next_page_token")
        if not token:
            break
        params["next_page_token"] = token

    if token:
        raise RuntimeError(
            f"{endpoint} returned more than {200 * PAGE_SIZE} rows — "
            "server-side daily downsampling (frequency=1d) appears unsupported"
        )
    if not rows:
        raise RuntimeError(f"{endpoint} returned no rows")
    return pd.DataFrame(rows)


def daily_series(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    """Collapse event-level rows to one mean value per UTC day."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["time"], utc=True, errors="coerce").dt.date
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["date", value_col])
    return (
        out.groupby("date")[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: out_col})
    )


def main():
    if not COINMETRICS_KEY:
        print(
            "ℹ️ CoinMetrics derivatives market data (open interest / funding / liquidations)\n"
            "   is Pro (paid) tier — the free Community API does not serve it.\n"
            "   Skipping CoinMetrics derivatives; Model 3 uses Coinalyze + Binance."
        )
        return

    frames = []

    # 1. Open interest — prefer USD notional for comparability with Coinalyze;
    #    the server is asked to downsample to daily (raw cadence is ~1 minute).
    try:
        oi = fetch_market_rows("market-openinterest", {"frequency": "1d"})
        value_col = "value_usd" if "value_usd" in oi.columns else "contract_count"
        frame = daily_series(oi, value_col, "open_interest")
        frames.append(frame)
        print(f"✅ CoinMetrics open interest saved | rows={len(frame)}")
    except Exception as e:
        print(f"⚠️ CoinMetrics open interest unavailable: {e}")

    # 2. Funding rate (event-driven, ~8h) — mean per day.
    try:
        fr = fetch_market_rows("market-funding-rates")
        frame = daily_series(fr, "rate", "funding_rate")
        frames.append(frame)
        print(f"✅ CoinMetrics funding rate saved | rows={len(frame)}")
    except Exception as e:
        print(f"⚠️ CoinMetrics funding rate unavailable: {e}")

    # 3. Liquidations (USD value). side='sell' closes longs, side='buy' closes shorts.
    try:
        liq = fetch_market_rows("market-liquidations", {"frequency": "1d"})
        liq["usd"] = (
            pd.to_numeric(liq["amount"], errors="coerce")
            * pd.to_numeric(liq["price"], errors="coerce")
        )
        liq["date"] = pd.to_datetime(liq["time"], utc=True, errors="coerce").dt.date
        liq = liq.dropna(subset=["date"])
        sides = liq.groupby(["date", "side"])["usd"].sum().unstack()
        frame = pd.DataFrame(
            {
                "date": sides.index,
                "long_liq": sides["sell"] if "sell" in sides.columns else float("nan"),
                "short_liq": sides["buy"] if "buy" in sides.columns else float("nan"),
            }
        )
        frames.append(frame)
        print(f"✅ CoinMetrics liquidations saved | rows={len(frame)}")
    except Exception as e:
        print(f"⚠️ CoinMetrics liquidations unavailable: {e}")

    if not frames:
        print("⚠️ No CoinMetrics derivatives series available — nothing saved.")
        return

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)

    out_path = OUT_DIR / "coinmetrics_derivatives_daily.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ CoinMetrics derivatives daily saved → {out_path} | rows={len(out)}")
    print(f"   columns: {list(out.columns)}")
    print(f"   range: {out['date'].min()} → {out['date'].max()}")


if __name__ == "__main__":
    main()

