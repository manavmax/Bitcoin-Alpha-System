# model_3/src/prepare_data.py

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = BASE_DIR / "data_raw" / "btc_1d_2018_2025.csv"
OUTPUT_PARQUET = BASE_DIR / "data_processed" / "features.parquet"

OUTPUT_PARQUET.parent.mkdir(exist_ok=True)

print("📥 Loading BTC 1D data...")
df = pd.read_csv(INPUT_CSV)

# -----------------------------
# Normalize column names
# -----------------------------
df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

df = df.rename(columns={
    "open_time": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume"
})

# -----------------------------
# Timestamp parsing (ROBUST)
# -----------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
df = df.dropna(subset=["timestamp"])
df["timestamp"] = df["timestamp"].dt.tz_localize(None)
df = df.sort_values("timestamp").reset_index(drop=True)

print("📅 Date range:", df["timestamp"].min(), "→", df["timestamp"].max())

# -----------------------------
# PRICE ACTION FEATURES
# -----------------------------
df["candle_body"] = df["close"] - df["open"]
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
df["range"] = df["high"] - df["low"]

# -----------------------------
# RETURNS & MOMENTUM
# -----------------------------
df["return"] = df["close"].pct_change()
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["momentum_7"] = df["close"] - df["close"].shift(7)
df["momentum_14"] = df["close"] - df["close"].shift(14)

# -----------------------------
# VOLATILITY
# -----------------------------
df["volatility_14"] = df["return"].rolling(14).std()
df["volatility_30"] = df["return"].rolling(30).std()

# -----------------------------
# TECHNICAL INDICATORS
# -----------------------------
df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

# RSI
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["rsi_14"] = 100 - (100 / (1 + rs))

# Bollinger Bands
mid = df["close"].rolling(20).mean()
std = df["close"].rolling(20).std()
df["bb_upper"] = mid + 2 * std
df["bb_lower"] = mid - 2 * std

# -----------------------------
# TARGET: NEXT-DAY LOG RETURN
# -----------------------------
df["target_return"] = df["log_return"].shift(-1)

# -----------------------------
# DROP NON-NUMERIC COLUMNS
# -----------------------------
non_numeric_cols = df.select_dtypes(include=["object"]).columns
if len(non_numeric_cols) > 0:
    print("🧹 Dropping non-numeric columns:", list(non_numeric_cols))
    df = df.drop(columns=non_numeric_cols)

# -----------------------------
# FINAL CLEAN & SAVE
# -----------------------------
df = df.dropna().reset_index(drop=True)

df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"✅ Features saved to {OUTPUT_PARQUET}")
print(f"📊 Total rows: {len(df)}")
