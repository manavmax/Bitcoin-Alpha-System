import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===============================
# CONFIG
# ===============================
ENSEMBLE_FILE = "model_6_macro_liquidity/results/model6_final_signal.csv"

# ===============================
# LOAD DATA
# ===============================
print("📥 Loading Model 6 ensemble output...")
df = pd.read_csv(ENSEMBLE_FILE, parse_dates=["date"])

REQUIRED_COLS = [
    "macro_lstm_signal",
    "macro_tree_signal",
    "macro_final_signal",
    "target_return"
]

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise RuntimeError(f"❌ Missing required columns: {missing}")

df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)

print(f"📊 Evaluation samples: {len(df)}")

# ===============================
# METRICS
# ===============================
y_true = df["target_return"].values
y_pred = df["macro_final_signal"].values

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

directional_accuracy = np.mean(
    np.sign(y_true) == np.sign(y_pred)
)

correlation = np.corrcoef(y_true, y_pred)[0, 1]

# ===============================
# OUTPUT
# ===============================
print("\n📊 MODEL 6 — MACRO ENSEMBLE EVALUATION")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"Directional Accuracy: {directional_accuracy:.4f}")
print(f"Signal–Return Correlation: {correlation:.4f}")

# ===============================
# REGIME DIAGNOSTIC (optional but useful)
# ===============================
if "macro_regime" in df.columns:
    print("\n📊 Directional Accuracy by Macro Regime")
    for r in sorted(df["macro_regime"].unique()):
        sub = df[df["macro_regime"] == r]
        if len(sub) > 50:
            da = np.mean(
                np.sign(sub["target_return"]) ==
                np.sign(sub["macro_final_signal"])
            )
            print(f"Regime {r} | samples={len(sub):4d} | DA={da:.4f}")

print("\n✅ MODEL 6 EVALUATION COMPLETE")
print("🟡 EXPECTED: moderate DA, low correlation → risk filter, not alpha")
