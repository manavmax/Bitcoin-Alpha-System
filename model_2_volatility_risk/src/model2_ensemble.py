# model_2_volatility_risk/src/model2_ensemble.py

import pandas as pd
import numpy as np
import torch
from volatility_lstm import VolatilityLSTM
from pathlib import Path

# =====================
# Paths
# =====================
BASE = Path(__file__).resolve().parents[1]

FEATURE_FILE = BASE / "processed" / "model2_volatility_features.csv"
LSTM_PRED_FILE = BASE / "results" / "volatility_lstm_preds.csv"
MODEL_FILE = BASE / "models" / "volatility_lstm.pth"
OUT_FILE = BASE / "results" / "model2_final_volatility.csv"

# =====================
# Load data
# =====================
df = pd.read_csv(FEATURE_FILE)
lstm_df = pd.read_csv(LSTM_PRED_FILE)

# Align lengths
df = df.iloc[-len(lstm_df):].reset_index(drop=True)
lstm_vol = lstm_df["lstm_volatility"].values

# =====================
# Ensemble
# =====================
ALPHA = 0.6  # GARCH weight

df["model2_volatility"] = (
    ALPHA * df["garch_vol"].values +
    (1 - ALPHA) * lstm_vol
)

# =====================
# Evaluation vs realized volatility
# =====================
true = df["realized_vol"].values
pred = df["model2_volatility"].values
mask = ~np.isnan(true) & ~np.isnan(pred)

mse = np.mean((pred[mask] - true[mask]) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(pred[mask] - true[mask]))
corr = np.corrcoef(pred[mask], true[mask])[0, 1]

print("\n📊 MODEL 2 — FINAL ENSEMBLE EVALUATION")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"Correlation: {corr:.4f}")
print(f"alpha (GARCH weight): {ALPHA}")

# =====================
# Save
# =====================
df.to_csv(OUT_FILE, index=False)

print(f"\n✅ Model 2 FINAL volatility saved → {OUT_FILE}")
