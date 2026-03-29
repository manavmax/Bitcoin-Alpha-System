import pandas as pd
import numpy as np
from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL3_FILE = BASE_DIR / "processed" / "model3_daily_features.csv"

# ⚠️ BTC DAILY PRICE FILE (BINANCE / KAGGLE)
PRICE_FILE = Path("model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv")

OUT_TRAIN = BASE_DIR / "processed" / "model3_train.csv"
OUT_TEST  = BASE_DIR / "processed" / "model3_test.csv"

# ==================== LOAD MODEL 3 FEATURES ====================
print("📥 Loading Model 3 features...")
m3 = pd.read_csv(MODEL3_FILE)
m3["date"] = pd.to_datetime(m3["date"])
m3["date"] = m3["date"].dt.tz_localize(None)  # ensure tz-naive

# ==================== LOAD BTC PRICE DATA ====================
print("📥 Loading BTC price data...")
price = pd.read_csv(PRICE_FILE)
price.columns = [c.strip().lower() for c in price.columns]

# ==================== FIND TIME COLUMN ====================
time_col = None
for c in ["open time", "close time", "timestamp", "date", "time"]:
    if c in price.columns:
        time_col = c
        break

if time_col is None:
    raise RuntimeError(
        f"❌ No usable time column found in BTC price file.\n"
        f"Columns: {list(price.columns)}"
    )

# ==================== PARSE DATE (BULLETPROOF) ====================
if pd.api.types.is_numeric_dtype(price[time_col]):
    # numeric timestamps
    if price[time_col].max() > 1e12:
        price["date"] = pd.to_datetime(price[time_col], unit="ms", utc=True)
    else:
        price["date"] = pd.to_datetime(price[time_col], unit="s", utc=True)
else:
    # string timestamps like "2018-01-01 00:00:00.000000 UTC"
    price["date"] = pd.to_datetime(price[time_col], utc=True)

# REMOVE TIMEZONE (CRITICAL FOR MERGE)
price["date"] = price["date"].dt.tz_localize(None)

# ==================== CLOSE PRICE ====================
if "close" not in price.columns:
    raise RuntimeError("❌ BTC price file must contain a 'close' column")

price = price.sort_values("date")

# ==================== TARGET CREATION ====================
price["target_return"] = np.log(price["close"]).diff().shift(-1)
price = price[["date", "target_return"]].dropna()

# ==================== MERGE ====================
df = m3.merge(price, on="date", how="inner")
df = df.dropna().reset_index(drop=True)

# ==================== MODEL 3 FEATURES ====================
FEATURES = [
    "long_liq",
    "short_liq",
    "liq_imbalance",
    "open_interest",
    "funding_rate",
    "long_short_ratio",
    "volume",
    "trade_count"
]

missing = set(FEATURES) - set(df.columns)
if missing:
    raise RuntimeError(f"❌ Missing Model 3 features: {missing}")

# ==================== TRAIN / TEST SPLIT ====================
split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

train.to_csv(OUT_TRAIN, index=False)
test.to_csv(OUT_TEST, index=False)

# ==================== SUMMARY ====================
print("✅ Model 3 dataset prepared successfully")
print(f"Train samples: {len(train)}")
print(f"Test samples:  {len(test)}")
print(f"Saved → {OUT_TRAIN}")
print(f"Saved → {OUT_TEST}")
