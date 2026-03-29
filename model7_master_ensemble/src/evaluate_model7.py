import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# CONFIG
# =========================
MODEL7_FILE = "model7_master_ensemble/results/model7_final_signal.csv"
PRICE_FILE = "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"
VOL_FILE = "model_2_volatility_risk/results/model2_final_volatility.csv"

TARGET_COL = "target_return"

# =========================
# DATE NORMALIZER
# =========================
def normalize_date(df):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    elif "Open time" in df.columns:
        df["date"] = pd.to_datetime(df["Open time"], errors="coerce", utc=True)
    else:
        raise RuntimeError(f"❌ No date column found: {df.columns}")

    df["date"] = df["date"].dt.tz_localize(None)
    return df

# =========================
# LOAD DATA
# =========================
print("📥 Loading Model 7 output...")
m7 = pd.read_csv(MODEL7_FILE)
m7 = normalize_date(m7)

# -------------------------
# AUTO-DETECT SIGNAL COLUMN
# -------------------------
POSSIBLE_SIGNAL_COLS = [
    "final_signal",
    "model7_signal",
    "ensemble_signal",
    "signal",
    "weighted_signal",
]

signal_col = None
for c in POSSIBLE_SIGNAL_COLS:
    if c in m7.columns:
        signal_col = c
        break

if signal_col is None:
    raise RuntimeError(
        f"❌ No signal column found in Model 7 output.\nColumns = {m7.columns.tolist()}"
    )

print(f"✅ Using signal column: {signal_col}")

# =========================
# LOAD BTC PRICE
# =========================
print("📥 Loading BTC price...")
price = pd.read_csv(PRICE_FILE)
price = normalize_date(price)

price = price.sort_values("date")
price[TARGET_COL] = price["Close"].pct_change().shift(-1)

# =========================
# LOAD VOLATILITY REGIMES
# =========================
print("📥 Loading volatility regimes...")
vol = pd.read_csv(VOL_FILE)
vol = normalize_date(vol)

# =========================
# MERGE DATA
# =========================
df = (
    m7.merge(price[["date", TARGET_COL]], on="date", how="inner")
      .merge(vol[["date", "vol_regime"]], on="date", how="left")
      .dropna()
)

print(f"📊 Evaluation samples: {len(df)}")

# =========================
# METRICS
# =========================
y_true = df[TARGET_COL]
y_pred = df[signal_col]

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

directional_accuracy = (np.sign(y_true) == np.sign(y_pred)).mean()
corr = np.corrcoef(y_true, y_pred)[0, 1]

print("\n📊 MODEL 7 — FINAL EVALUATION")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"Directional Accuracy: {directional_accuracy:.4f}")
print(f"Signal–Return Correlation: {corr:.4f}")

# =========================
# REGIME ANALYSIS
# =========================
print("\n📊 Directional Accuracy by Volatility Regime")
for r in sorted(df["vol_regime"].dropna().unique()):
    sub = df[df["vol_regime"] == r]
    da = (np.sign(sub[TARGET_COL]) == np.sign(sub[signal_col])).mean()
    print(f"Regime {int(r)} | samples={len(sub):4d} | DA={da:.4f}")

print("\n✅ MODEL 7 EVALUATION COMPLETE")
