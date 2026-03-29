# model_2_volatility_risk/src/compute_volatility_features.py

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
IN_FILE = BASE_DIR / "processed" / "garch_output.csv"
OUT_FILE = BASE_DIR / "processed" / "model2_volatility_features.csv"

# =========================
# Load
# =========================
df = pd.read_csv(IN_FILE, parse_dates=["date"])
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
df = df.sort_values("date").reset_index(drop=True)

# =========================
# Realized Volatility (14d)
# =========================
df["realized_vol"] = (
    df["log_return"]
    .rolling(14)
    .std()
)

# =========================
# True Range & ATR (14d)
# =========================
price = pd.read_csv(
    BASE_DIR / "data" / "btc_daily.csv"
)

price["date"] = pd.to_datetime(price["Open time"], utc=True, errors="coerce").dt.tz_localize(None)
price = price.sort_values("date")

price["prev_close"] = price["Close"].shift(1)
price["tr"] = np.maximum.reduce([
    price["High"] - price["Low"],
    abs(price["High"] - price["prev_close"]),
    abs(price["Low"] - price["prev_close"])
])

price["atr"] = price["tr"].rolling(14).mean()

df = df.merge(
    price[["date", "tr", "atr"]],
    on="date",
    how="left"
)

# =========================
# 1️⃣ Volatility of Volatility (VoV)
# =========================
df["vov"] = df["garch_vol"].rolling(14).std()

# =========================
# 2️⃣ Downside Volatility (14d)
# =========================
def downside_std(x):
    neg = x[x < 0]
    return np.sqrt(np.mean(neg ** 2)) if len(neg) > 0 else 0.0

df["downside_vol"] = (
    df["log_return"]
    .rolling(14)
    .apply(downside_std, raw=False)
)

# =========================
# 3️⃣ Volatility Momentum
# =========================
df["vol_momentum"] = df["garch_vol"] - df["garch_vol"].shift(7)

# =========================
# 4️⃣ Volatility Percentile Rank (252d)
# =========================
def vol_rank(series):
    return series.rank(pct=True).iloc[-1]

df["vol_rank"] = (
    df["garch_vol"]
    .rolling(252)
    .apply(vol_rank, raw=False)
)

# =========================
# 5️⃣ Volatility Regime (0/1/2)
# =========================
q1 = df["garch_vol"].quantile(0.33)
q2 = df["garch_vol"].quantile(0.66)

def regime(v):
    if v <= q1:
        return 0
    elif v <= q2:
        return 1
    else:
        return 2

df["vol_regime"] = df["garch_vol"].apply(regime)

# =========================
# 6️⃣ Crash Flag
# =========================
down_q = df["downside_vol"].quantile(0.9)

df["crash_flag"] = (
    (df["downside_vol"] > down_q) &
    (df["vov"] > df["vov"].shift(1))
).astype(int)

# =========================
# Cleanup
# =========================
df = df.dropna().reset_index(drop=True)

# =========================
# Save
# =========================
df.to_csv(OUT_FILE, index=False)

print("✅ Model 2 volatility features computed")
print(f"Saved → {OUT_FILE}")
print("Columns:", list(df.columns))
