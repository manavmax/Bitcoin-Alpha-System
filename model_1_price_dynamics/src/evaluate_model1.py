# model_1_price_dynamics/src/evaluate_model1.py

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
INPUT_PATH = RESULTS_DIR / "model1_final_predictions.csv"
OUTPUT_PATH = RESULTS_DIR / "model1_evaluation.json"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(INPUT_PATH)

preds = df["model1_return_prediction"].values
targets = df["true_return"].values

# ---------------- METRICS ----------------
mse = np.mean((preds - targets) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(preds - targets))
directional_accuracy = np.mean(np.sign(preds) == np.sign(targets))

metrics = {
    "MSE": float(mse),
    "RMSE": float(rmse),
    "MAE": float(mae),
    "Directional_Accuracy": float(directional_accuracy)
}

# ---------------- SAVE ----------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# ---------------- PRINT ----------------
print("📊 MODEL 1 – PRICE DYNAMICS EVALUATION")
for k, v in metrics.items():
    print(f"{k}: {v:.6f}")

print(f"✅ Evaluation saved to {OUTPUT_PATH}")
