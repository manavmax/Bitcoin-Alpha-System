# model_3/src/evaluate_nbeats.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json

from dataset import BTCSequenceDataset
from nbeats_model import NBeats

SEQ_LEN = 30
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "nbeats.pt"
RESULTS_DIR = BASE_DIR / "results"

dataset = BTCSequenceDataset(seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

input_size = SEQ_LEN * dataset.features.shape[1]

model = NBeats(input_size=input_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

preds, targets = [], []

with torch.no_grad():
    for X, y in loader:
        X = X.to(DEVICE)
        out = model(X)
        preds.extend(out.cpu().numpy())
        targets.extend(y.numpy())

preds = np.array(preds)
targets = np.array(targets)

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

with open(RESULTS_DIR / "nbeats_evaluation.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("📊 N-BEATS Evaluation")
for k, v in metrics.items():
    print(f"{k}: {v:.6f}")
