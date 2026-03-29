import os
import requests
import pandas as pd
from datetime import datetime, timezone

# ==========================
# CONFIG
# ==========================
START_DATE = "2018-01-01"
END_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

RAW_DIR = "model_6_macro_liquidity/raw/fred"
os.makedirs(RAW_DIR, exist_ok=True)

FRED_SERIES = {
    "fed_funds_rate": "DFF",
    "m2_money_supply": "M2SL",
    "cpi": "CPIAUCSL",
    "yield_curve": "T10Y2Y",
    "vix": "VIXCLS",
    "sp500": "SP500",
}

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


# ==========================
# FETCH FUNCTION
# ==========================
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "file_type": "json",
        "observation_start": START_DATE,
        "observation_end": END_DATE,
    }

    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()["observations"]

    df = pd.DataFrame(data)
    df = df[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print("🚀 Fetching Model 6 macro & liquidity data (FRED)...\n")

    for name, series_id in FRED_SERIES.items():
        try:
            df = fetch_fred_series(series_id)

            out_path = f"{RAW_DIR}/{name}.csv"
            df.to_csv(out_path, index=False)

            print(
                f"✅ {name} ({series_id}) saved | "
                f"rows={len(df)} | "
                f"{df['date'].min().date()} → {df['date'].max().date()}"
            )

        except Exception as e:
            print(f"❌ Failed fetching {name} ({series_id}): {e}")

    print("\n🎯 Model 6 macro data ingestion completed")
import os
import requests
import pandas as pd
from datetime import datetime, timezone

# ==========================
# CONFIG
# ==========================
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise RuntimeError("❌ FRED_API_KEY not found in environment variables")

START_DATE = "2018-01-01"
END_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

RAW_DIR = "model_6_macro_liquidity/raw/fred"
os.makedirs(RAW_DIR, exist_ok=True)

FRED_SERIES = {
    "fed_funds_rate": "DFF",
    "m2_money_supply": "M2SL",
    "cpi": "CPIAUCSL",
    "yield_curve": "T10Y2Y",
    "vix": "VIXCLS",
    "sp500": "SP500",
}

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


# ==========================
# FETCH FUNCTION
# ==========================
def fetch_fred_series(series_id: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": START_DATE,
        "observation_end": END_DATE,
    }

    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()["observations"]

    df = pd.DataFrame(data)
    df = df[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print("🚀 Fetching Model 6 macro & liquidity data (FRED)...\n")

    for name, series_id in FRED_SERIES.items():
        try:
            df = fetch_fred_series(series_id)

            out_path = f"{RAW_DIR}/{name}.csv"
            df.to_csv(out_path, index=False)

            print(
                f"✅ {name} ({series_id}) saved | "
                f"rows={len(df)} | "
                f"{df['date'].min().date()} → {df['date'].max().date()}"
            )

        except Exception as e:
            print(f"❌ Failed fetching {name} ({series_id}): {e}")

    print("\n🎯 Model 6 macro data ingestion completed")
