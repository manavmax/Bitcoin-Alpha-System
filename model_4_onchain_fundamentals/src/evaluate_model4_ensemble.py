import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# PATHS
# =========================
ENSEMBLE_FILE = (
    "model_4_onchain_fundamentals/results/model4_final_signal.csv"
)

PRICE_FILE = (
    "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"
)

ONCHAIN_FEATURES_FILE = (
    "model_4_onchain_fundamentals/processed/model4_onchain_features.csv"
)

# =========================
# LOAD FILES
# =========================
print("📥 Loading Model 4 ensemble output...")
ens = pd.read_csv(ENSEMBLE_FILE)
ens["date"] = pd.to_datetime(ens["date"], utc=True)

print("📥 Loading BTC price data...")
price = pd.read_csv(PRICE_FILE)

# detect date column automatically
for col in ["date", "Date", "open time", "Open time"]:
    if col in price.columns:
        price["date"] = pd.to_datetime(price[col], utc=True, errors="coerce")
        break
else:
    raise RuntimeError("❌ No date column found in price file")

price = price.sort_values("date")
price["target_return"] = price["Close"].pct_change().shift(-1)

print("📥 Loading on-chain features (for regime analysis)...")
onchain = pd.read_csv(ONCHAIN_FEATURES_FILE)
onchain["date"] = pd.to_datetime(onchain["date"], utc=True)

# =========================
# MERGE
# =========================
df = (
    ens.merge(
        price[["date", "target_return"]],
        on="date",
        how="inner"
    )
    .merge(
        onchain[["date", "onchain_regime", "miner_stress_flag"]],
        on="date",
        how="left"
    )
)

df = df.dropna(subset=["model4_signal", "target_return"])

if len(df) < 200:
    raise RuntimeError("❌ Not enough data to evaluate Model 4")

print(f"📊 Evaluation samples: {len(df)}")

# =========================
# METRICS
# =========================
y_true = df["target_return"].values
y_pred = df["model4_signal"].values

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

directional_accuracy = np.mean(
    np.sign(y_true) == np.sign(y_pred)
)

corr = np.corrcoef(y_true, y_pred)[0, 1]

# =========================
# RESULTS
# =========================
print("\n📊 MODEL 4 — FINAL ENSEMBLE EVALUATION")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"Directional Accuracy: {directional_accuracy:.4f}")
print(f"Signal–Return Correlation: {corr:.4f}")

# =========================
# REGIME ANALYSIS
# =========================
print("\n📊 Directional Accuracy by On-Chain Regime")

for regime in sorted(df["onchain_regime"].dropna().unique()):
    sub = df[df["onchain_regime"] == regime]
    if len(sub) < 100:
        continue

    da = np.mean(
        np.sign(sub["model4_signal"]) == np.sign(sub["target_return"])
    )

    print(f"Regime {int(regime)} | samples={len(sub):4d} | DA={da:.4f}")

# =========================
# MINER STRESS ANALYSIS
# =========================
print("\n⛏️ Miner Stress Conditional Performance")

for flag in [0, 1]:
    sub = df[df["miner_stress_flag"] == flag]
    if len(sub) < 100:
        continue

    da = np.mean(
        np.sign(sub["model4_signal"]) == np.sign(sub["target_return"])
    )

    label = "Stress" if flag == 1 else "No Stress"
    print(f"{label:10s} | samples={len(sub):4d} | DA={da:.4f}")

# =========================
# FINAL VERDICT
# =========================
print("\n✅ MODEL 4 EVALUATION COMPLETE")

if directional_accuracy >= 0.60:
    print("🟢 STRONG on-chain signal — excellent filter")
elif directional_accuracy >= 0.55:
    print("🟡 MODERATE signal — good confirmation layer")
else:
    print("🔴 WEAK standalone — use only as soft bias")
