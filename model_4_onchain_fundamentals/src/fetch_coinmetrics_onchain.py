import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


COINMETRICS_BASE = "https://api.coinmetrics.io/v4"
COINMETRICS_KEY = os.getenv("COINMETRICS_API_KEY", "")

OUT_DIR = Path("model_4_onchain_fundamentals/raw/coinmetrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_asset_metrics_daily(metrics: list[str], start_time: str = "2010-07-01") -> pd.DataFrame:
    params = {
        "assets": "btc",
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    headers = {}
    if COINMETRICS_KEY:
        headers["Authorization"] = f"Bearer {COINMETRICS_KEY}"

    url = f"{COINMETRICS_BASE}/timeseries/asset-metrics"
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    payload = r.json()

    rows = payload.get("data", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Unexpected CoinMetrics response shape: keys={list(payload.keys())}")

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
    # Map CoinMetrics metric names → model4 raw names
    # These are common CM names; adjust if needed depending on your plan.
    metrics = [
        "HashRate",
        "Difficulty",
        "TxCnt",
        "TxTfrValUSD",
        "MinerRevenueUSD",
    ]

    df = fetch_asset_metrics_daily(metrics=metrics)

    out = df.rename(
        columns={
            "HashRate": "hash_rate",
            "Difficulty": "difficulty",
            "TxCnt": "tx_count",
            "TxTfrValUSD": "tx_volume_usd",
            "MinerRevenueUSD": "miner_revenue",
        }
    )

    out_path = OUT_DIR / "coinmetrics_onchain_daily.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ CoinMetrics on-chain daily saved → {out_path} | rows={len(out)}")
    print(f"   range: {out['date'].min()} → {out['date'].max()}")


if __name__ == "__main__":
    main()

