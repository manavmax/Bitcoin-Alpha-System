"""
CoinMetrics on-chain data for Model 4 (raw inputs).

Tier notes (verified against the CoinMetrics v4 API):
  - The free Community API (community-api.coinmetrics.io) needs NO API key and
    serves network metrics such as HashRate, TxCnt, AdrActCnt and CapMVRVCur.
  - USD-denominated Pro metrics (DiffMean / TxTfrValUSD / RevUSD / FeeTotUSD)
    return 403 without paid credentials, so they are only requested when
    COINMETRICS_API_KEY is set.

The output keeps the 5-column schema that prepare_model4_features.py expects;
columns unavailable for the current tier are simply omitted.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


COINMETRICS_KEY = os.getenv("COINMETRICS_API_KEY", "")

# Free Community API by default; the Pro host only when a key is configured.
COINMETRICS_BASE = (
    "https://api.coinmetrics.io/v4"
    if COINMETRICS_KEY
    else "https://community-api.coinmetrics.io/v4"
)

OUT_DIR = Path("model_4_onchain_fundamentals/raw/coinmetrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CoinMetrics metric name -> model4 raw column name.
# Community (free) tier — always fetched:
COMMUNITY_METRICS = {
    "HashRate": "hash_rate",
    "TxCnt": "tx_count",
}
# Paid (Pro) tier — only fetched when COINMETRICS_API_KEY is set:
PAID_METRICS = {
    "DiffMean": "difficulty",
    "TxTfrValUSD": "tx_volume_usd",
    "RevUSD": "miner_revenue",
}


def fetch_asset_metrics_daily(metrics: list[str], start_time: str = "2010-07-01") -> pd.DataFrame:
    params = {
        "assets": "btc",
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        # API default page size is 100 rows — force a large page and still
        # follow next_page_token in case the range ever exceeds it.
        "page_size": 10000,
    }
    headers = {}
    if COINMETRICS_KEY:
        headers["Authorization"] = f"Bearer {COINMETRICS_KEY}"

    url = f"{COINMETRICS_BASE}/timeseries/asset-metrics"

    rows: list = []
    for _ in range(20):  # safety cap
        r = requests.get(url, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        payload = r.json()
        page = payload.get("data", [])
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected CoinMetrics response shape: keys={list(payload.keys())}")
        rows.extend(page)
        token = payload.get("next_page_token")
        if not token:
            break
        params["next_page_token"] = token

    if not rows:
        raise RuntimeError("CoinMetrics returned no rows")

    df = pd.DataFrame(rows)
    if "time" not in df.columns:
        raise RuntimeError(f"CoinMetrics response missing 'time' column: cols={list(df.columns)}")

    df["date"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.date
    df = df.dropna(subset=["date"]).sort_values("date")

    keep = ["date"]
    for m in metrics:
        if m in df.columns:
            keep.append(m)
    df = df[keep]

    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def main():
    frames = []

    # 1. Community-tier metrics (no key needed).
    df_community = fetch_asset_metrics_daily(metrics=list(COMMUNITY_METRICS))
    frames.append(df_community.rename(columns=COMMUNITY_METRICS))

    # 2. Paid-tier metrics (only with an API key).
    if COINMETRICS_KEY:
        try:
            df_paid = fetch_asset_metrics_daily(metrics=list(PAID_METRICS))
            frames.append(df_paid.rename(columns=PAID_METRICS))
        except Exception as e:
            print(f"⚠️ Paid CoinMetrics metrics unavailable ({e}) — continuing with community metrics only")
    else:
        print(
            "ℹ️ COINMETRICS_API_KEY not set — skipping paid-tier metrics "
            f"({', '.join(PAID_METRICS)}) and using free Community data only"
        )

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)

    out_path = OUT_DIR / "coinmetrics_onchain_daily.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ CoinMetrics on-chain daily saved → {out_path} | rows={len(out)}")
    print(f"   columns: {list(out.columns)}")
    print(f"   range: {out['date'].min()} → {out['date'].max()}")


if __name__ == "__main__":
    main()

