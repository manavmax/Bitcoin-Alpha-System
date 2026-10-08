import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.prediction_store import publish_append_only  # noqa: E402

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parents[1]

TCN_FILE = BASE_DIR / "results" / "model3_tcn_predictions.csv"
CNN_LSTM_FILE = BASE_DIR / "results" / "model3_cnn_lstm_predictions.csv"
OUT_FILE = BASE_DIR / "results" / "model3_ensemble_predictions.csv"

# ================= LOAD =================
tcn = pd.read_csv(TCN_FILE)
cnn = pd.read_csv(CNN_LSTM_FILE)

# Align by date (safety)
df = tcn.merge(
    cnn[["date", "cnn_lstm_pred"]],
    on="date",
    how="inner"
)

# ================= ENSEMBLE =================
df["model3_ensemble"] = (
    0.55 * df["tcn_pred"] +
    0.45 * df["cnn_lstm_pred"]
)

# ================= EVALUATION =================
# The newest published row still has an unknown next-day return (its ground
# truth is backfilled into the store a day later), so evaluate only on rows
# whose outcome is already known.
eval_df = df.dropna(subset=["target_return"])
y_true = eval_df["target_return"]
y_pred = eval_df["model3_ensemble"]

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
directional_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

print("\n📊 MODEL 3 — FINAL ENSEMBLE EVALUATION")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Directional Accuracy: {directional_acc:.4f}")

# ================= SAVE =================
# Append-only publication: previously published predictions are immutable.
# Only new dates are appended; `target_return` (ground truth, unknown on
# publication day) is backfilled once it becomes known.
publish_append_only(
    df,
    OUT_FILE,
    backfill_cols=("target_return",),
    refresh_env="MODEL3_REFRESH_DATES",
)
print("✅ Model 3 Ensemble completed successfully")
