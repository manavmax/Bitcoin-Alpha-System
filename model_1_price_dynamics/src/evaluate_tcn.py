# model_3/src/evaluate_tcn.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json

from dataset import BTCSequenceDataset
from tcn_model import TCN

# ---------------- CONFIG ----------------
SEQ_LEN = 30
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "tcn.pt"
RESULTS_DIR = BASE_DIR / "results"

# ---------------- DATA ----------------
dataset = BTCSequenceDataset(seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
input_size = dataset.features.shape[1]

model = TCN(
    input_size=input_size,
    num_channels=[32, 32, 32],
    kernel_size=3,
    dropout=0.2
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- INFERENCE ----------------
preds, targets = [], []

with torch.no_grad():
    for X, y in loader:
        X = X.to(DEVICE)
        out = model(X)

        preds.extend(out.cpu().numpy())
        targets.extend(y.numpy())

preds = np.array(preds)
targets = np.array(targets)

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

with open(RESULTS_DIR / "tcn_evaluation.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("📊 TCN Evaluation Metrics")
for k, v in metrics.items():
    print(f"{k}: {v:.6f}")

print("✅ TCN evaluation completed successfully")
