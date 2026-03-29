import requests
import pandas as pd

BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_btc_daily(limit=2000):
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": limit
    }
    r = requests.get(BINANCE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore"
    ])

    df["date"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df[["date","open","high","low","close","volume"]].astype(float, errors="ignore")

    return df.sort_values("date")
