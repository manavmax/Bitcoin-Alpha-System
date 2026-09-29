import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_KEY = os.getenv("COINALYZE_API_KEY")
if not API_KEY:
    raise RuntimeError("COINALYZE_API_KEY not set (add it to .env)")

BASE_URL = "https://api.coinalyze.net/v1"
HEADERS = {"api_key": API_KEY}   # header name per docs: api_key

SYMBOLS = "BTCUSDT_PERP.A"   # aggregate BTC perpetual, valid per docs
INTERVAL = "daily"

# Only CLOSED daily candles are fetched: `to` is one second before today's
# 00:00 UTC, so the current day's forming candle is never pulled (same
# closed-candle discipline as the Model 8 dataset cutoff). The API keeps
# full daily history (no rolling deletion for daily granularity), so a
# single call back to 2019 returns everything available (~2370+ points;
# oldest BTCUSDT_PERP.A daily candle is 2019-09-12).
FROM_TS = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp())
_MIDNIGHT = datetime.now(tz=timezone.utc).replace(
    hour=0, minute=0, second=0, microsecond=0
)
TO_TS = int(_MIDNIGHT.timestamp()) - 1

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "raw" / "coinalyze"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT_PAUSE_S = 1.6  # free tier = 40 calls/min per key; stay well below


def fetch_history(endpoint, filename):
    params = {
        "symbols": SYMBOLS,
        "interval": INTERVAL,
        "from": FROM_TS,
        "to": TO_TS
    }

    # Free tier returns 429 with a Retry-After header when exceeded.
    for attempt in range(6):
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS,
                         params=params, timeout=30)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "60"))
            print(f"⏳ {endpoint}: rate limited (429), waiting {wait}s "
                  f"[attempt {attempt + 1}/6]")
            time.sleep(wait)
            continue
        r.raise_for_status()
        break
    else:
        raise RuntimeError(f"{endpoint}: rate limit persisted after retries")

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
    d_min = pd.to_datetime(df["t"].min(), unit="s", utc=True).date()
    d_max = pd.to_datetime(df["t"].max(), unit="s", utc=True).date()
    print(f"✅ {filename} saved | rows={len(df)} | {d_min} → {d_max}")
    time.sleep(RATE_LIMIT_PAUSE_S)


print("🚀 Fetching Coinalyze historical daily data (closed candles only)")

fetch_history("liquidation-history", "liquidations.csv")
fetch_history("open-interest-history", "open_interest.csv")
fetch_history("funding-rate-history", "funding_rate.csv")
fetch_history("long-short-ratio-history", "long_short_ratio.csv")
fetch_history("ohlcv-history", "ohlcv.csv")

print("🎯 Coinalyze raw historical data fetched successfully")
