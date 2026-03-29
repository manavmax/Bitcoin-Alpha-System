import numpy as np
import pandas as pd
from arch import arch_model
from pathlib import Path
import pickle

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "btc_daily.csv"
OUT_FEATURES = BASE_DIR / "processed" / "model2_features.csv"
OUT_GARCH = BASE_DIR / "models" / "garch.pkl"

OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
OUT_GARCH.parent.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_FILE)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# -------------------------
# Date handling (Binance)
# -------------------------
if "open time" in df.columns:
    df["date"] = pd.to_datetime(df["open time"], errors="coerce")
elif "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
else:
    raise RuntimeError(f"No valid date column found. Columns: {df.columns}")

df = df.sort_values("date").reset_index(drop=True)

# Ensure numeric
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# RETURNS
# =========================
df["return"] = np.log(df["close"]).diff()

# =========================
# REALIZED VOLATILITY (target)
# =========================
df["realized_vol"] = df["return"].rolling(14).std()

# =========================
# TRUE RANGE & ATR
# =========================
df["tr"] = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ),
)
df["atr"] = df["tr"].rolling(14).mean()

# =========================
# GARCH(1,1)
# =========================
returns = df["return"].dropna() * 100  # scale for stability

garch = arch_model(
    returns,
    vol="Garch",
    p=1,
    q=1,
    dist="normal"
)

garch_res = garch.fit(disp="off")

df.loc[returns.index, "garch_vol"] = garch_res.conditional_volatility / 100

# Save GARCH model
with open(OUT_GARCH, "wb") as f:
    pickle.dump(garch_res, f)

# =========================
# ADVANCED RISK FEATURES
# =========================

# 1️⃣ Volatility of Volatility
df["vov"] = df["garch_vol"].rolling(14).std()

# 2️⃣ Downside Volatility (Semi-variance)
df["downside_vol"] = (
    df["return"]
    .where(df["return"] < 0)
    .rolling(14)
    .std()
)

# 3️⃣ Volatility Momentum
df["vol_momentum"] = df["garch_vol"] - df["garch_vol"].shift(7)

# 4️⃣ Volatility Percentile Rank (252d)
df["vol_rank"] = (
    df["garch_vol"]
    .rolling(252)
    .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
)

# 5️⃣ Volatility Regime (Low / Mid / High)
df["vol_regime"] = pd.qcut(
    df["garch_vol"],
    q=3,
    labels=[0, 1, 2]
)

# 6️⃣ Crash Regime Flag
downside_threshold = df["downside_vol"].quantile(0.9)
df["crash_flag"] = (
    (df["downside_vol"] > downside_threshold) &
    (df["vov"].diff() > 0)
).astype(int)

# =========================
# FINAL CLEANUP
# =========================
FEATURE_COLS = [
    "date",
    "garch_vol",
    "realized_vol",
    "atr",
    "vov",
    "downside_vol",
    "vol_momentum",
    "vol_rank",
    "vol_regime",
    "crash_flag",
]

final_df = df[FEATURE_COLS].dropna().reset_index(drop=True)

final_df.to_csv(OUT_FEATURES, index=False)

# =========================
# SUMMARY
# =========================
print("✅ Model 2 features computed successfully")
print(f"Saved features → {OUT_FEATURES}")
print(f"Saved GARCH model → {OUT_GARCH}")
print("Columns:", list(final_df.columns))
print("Rows:", len(final_df))
