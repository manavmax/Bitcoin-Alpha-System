import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


COINMETRICS_BASE = "https://api.coinmetrics.io/v4"
COINMETRICS_KEY = os.getenv("COINMETRICS_API_KEY", "")

OUT_DIR = Path("model_3_derivatives_flow/raw/coinmetrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_asset_metrics_daily(metrics: list[str], start_time: str = "2015-01-01") -> pd.DataFrame:
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
    # Metrics per your spec. These may require a CoinMetrics plan; if the endpoint
    # rejects them, the pipeline will fall back to Coinalyze/Binance-only.
    metrics = [
        "OpenInterest",
        "FundingRate",
        "Volume",
        "TradeCount",
        "LongLiquidationsUSD",
        "ShortLiquidationsUSD",
    ]

    df = fetch_asset_metrics_daily(metrics=metrics, start_time="2017-01-01")

    out = df.rename(
        columns={
            "OpenInterest": "open_interest",
            "FundingRate": "funding_rate",
            "Volume": "volume",
            "TradeCount": "trade_count",
            "LongLiquidationsUSD": "long_liq",
            "ShortLiquidationsUSD": "short_liq",
        }
    )

    out_path = OUT_DIR / "coinmetrics_derivatives_daily.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ CoinMetrics derivatives daily saved → {out_path} | rows={len(out)}")
    print(f"   range: {out['date'].min()} → {out['date'].max()}")


if __name__ == "__main__":
    main()

