# model_3/src/ensemble.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json

from dataset import BTCSequenceDataset
from model import LSTMPricePredictor
from tcn_model import TCN

# ---------------- CONFIG ----------------
SEQ_LEN = 30
BATCH_SIZE = 32
ALPHA = 0.6   # weight for LSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

LSTM_PATH = BASE_DIR / "models" / "lstm.pt"
TCN_PATH = BASE_DIR / "models" / "tcn.pt"

RESULTS_DIR.mkdir(exist_ok=True)

# ---------------- DATA ----------------
dataset = BTCSequenceDataset(seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- LOAD MODELS ----------------
input_size = dataset.features.shape[1]

lstm = LSTMPricePredictor(input_size=input_size).to(DEVICE)
lstm.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
lstm.eval()

tcn = TCN(
    input_size=input_size,
    num_channels=[32, 32, 32],
    kernel_size=3,
    dropout=0.2
).to(DEVICE)

tcn.load_state_dict(torch.load(TCN_PATH, map_location=DEVICE))
tcn.eval()

# ---------------- INFERENCE ----------------
ensemble_preds = []
targets = []

with torch.no_grad():
    for X, y in loader:
        X = X.to(DEVICE)

        lstm_pred = lstm(X)
        tcn_pred = tcn(X)

        combined = ALPHA * lstm_pred + (1 - ALPHA) * tcn_pred

        ensemble_preds.extend(combined.cpu().numpy())
        targets.extend(y.numpy())

ensemble_preds = np.array(ensemble_preds)
targets = np.array(targets)

# ---------------- METRICS ----------------
mse = np.mean((ensemble_preds - targets) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(ensemble_preds - targets))
directional_accuracy = np.mean(np.sign(ensemble_preds) == np.sign(targets))

metrics = {
    "MSE": float(mse),
    "RMSE": float(rmse),
    "MAE": float(mae),
    "Directional_Accuracy": float(directional_accuracy),
    "alpha": ALPHA
}

# ---------------- SAVE ----------------
with open(RESULTS_DIR / "ensemble_evaluation.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("📊 ENSEMBLE EVALUATION METRICS")
for k, v in metrics.items():
    print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

print("✅ Ensemble evaluation completed successfully")
