import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path


OUT_DIR = Path("model_3_derivatives_flow/raw/binance")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch(symbol="BTCUSDT", period="1d", limit=90) -> pd.DataFrame:
    url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
    params = {"symbol": symbol, "period": period, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
    df["long_short_ratio"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
    out = df[["date", "long_short_ratio"]].dropna().sort_values("date")
    return out


def main():
    df = fetch()
    out_path = OUT_DIR / "binance_long_short_ratio_1d.csv"
    df.to_csv(out_path, index=False)
    if not df.empty:
        print(f"✅ Binance long/short ratio saved → {out_path} | rows={len(df)}")
        print(f"   range: {df['date'].min()} → {df['date'].max()}")
    else:
        print(f"⚠️ Binance long/short ratio returned empty dataset → {out_path}")


if __name__ == "__main__":
    main()

