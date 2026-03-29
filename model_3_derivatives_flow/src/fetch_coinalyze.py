import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

API_KEY = os.getenv("COINALYZE_API_KEY")
if not API_KEY:
    raise RuntimeError("COINALYZE_API_KEY not set")

BASE_URL = "https://api.coinalyze.net/v1"
HEADERS = {"api_key": API_KEY}

SYMBOLS = "BTCUSDT_PERP.A"   # valid per docs
INTERVAL = "daily"

FROM_TS = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp())
TO_TS   = int(datetime.now(tz=timezone.utc).timestamp())

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "raw" / "coinalyze"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def fetch_history(endpoint, filename):
    params = {
        "symbols": SYMBOLS,
        "interval": INTERVAL,
        "from": FROM_TS,
        "to": TO_TS
    }

    r = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()

    payload = r.json()
    rows = []

    for block in payload:
        symbol = block["symbol"]
        for h in block["history"]:
            h["symbol"] = symbol
            rows.append(h)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"{endpoint} returned NO DATA — this should not happen")

    df.to_csv(RAW_DIR / filename, index=False)
    print(f"✅ {filename} saved | rows={len(df)}")

print("🚀 Fetching Coinalyze historical daily data (DOC-CORRECT)")

fetch_history("liquidation-history", "liquidations.csv")
fetch_history("open-interest-history", "open_interest.csv")
fetch_history("funding-rate-history", "funding_rate.csv")
fetch_history("long-short-ratio-history", "long_short_ratio.csv")
fetch_history("ohlcv-history", "ohlcv.csv")

print("🎯 Coinalyze raw historical data fetched successfully")
