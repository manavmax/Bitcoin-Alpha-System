import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

PRICE_FILE = DATA_DIR / "btc_price_daily.csv"

BINANCE_URL = "https://api.binance.com/api/v3/klines"


def fetch_binance_daily(limit=1500):
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": limit
    }
    r = requests.get(BINANCE_URL, params=params, timeout=10)
    r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "qav", "trades",
        "taker_base", "taker_quote", "ignore"
    ])

    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["date", "open", "high", "low", "close", "volume"]]

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    return df


def load_btc_price():
    if PRICE_FILE.exists():
        df = pd.read_csv(PRICE_FILE, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"], utc=True)

        # If data is stale, refresh
        if df["date"].iloc[-1].date() < datetime.now(timezone.utc).date():
            fresh = fetch_binance_daily()
            fresh.to_csv(PRICE_FILE, index=False)
            return fresh

        return df

    df = fetch_binance_daily()
    df.to_csv(PRICE_FILE, index=False)
    return df

