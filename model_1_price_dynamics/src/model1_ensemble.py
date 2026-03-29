# model_3/src/model1_ensemble.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

from dataset import BTCSequenceDataset
from model import LSTMPricePredictor
from tcn_model import TCN
from nbeats_model import NBeats

# ---------------- CONFIG ----------------
SEQ_LEN = 30
BATCH_SIZE = 32

W_LSTM = 0.40
W_TCN = 0.35
W_NBEATS = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
OUTPUT_PATH = BASE_DIR / "results" / "model1_final_predictions.csv"

# ---------------- DATA ----------------
dataset = BTCSequenceDataset(seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- LOAD MODELS ----------------
input_size = dataset.features.shape[1]
flat_input_size = SEQ_LEN * input_size

# LSTM
lstm = LSTMPricePredictor(input_size=input_size).to(DEVICE)
lstm.load_state_dict(torch.load(MODELS_DIR / "lstm.pt", map_location=DEVICE))
lstm.eval()

# TCN
tcn = TCN(
    input_size=input_size,
    num_channels=[32, 32, 32],
    kernel_size=3,
    dropout=0.2
).to(DEVICE)
tcn.load_state_dict(torch.load(MODELS_DIR / "tcn.pt", map_location=DEVICE))
tcn.eval()

# N-BEATS
nbeats = NBeats(input_size=flat_input_size).to(DEVICE)
nbeats.load_state_dict(torch.load(MODELS_DIR / "nbeats.pt", map_location=DEVICE))
nbeats.eval()

# ---------------- INFERENCE ----------------
final_preds = []
targets = []

with torch.no_grad():
    for X, y in loader:
        X = X.to(DEVICE)

        p_lstm = lstm(X)
        p_tcn = tcn(X)
        p_nbeats = nbeats(X)

        ensemble_pred = (
            W_LSTM * p_lstm
            + W_TCN * p_tcn
            + W_NBEATS * p_nbeats
        )

        final_preds.extend(ensemble_pred.cpu().numpy())
        targets.extend(y.numpy())

# ---------------- SAVE OUTPUT ----------------
df = pd.DataFrame({
    "model1_return_prediction": np.array(final_preds),
    "true_return": np.array(targets)
})

df.to_csv(OUTPUT_PATH, index=False)

print("✅ Model 1 ensemble completed")
print(f"Saved final Model 1 predictions → {OUTPUT_PATH}")
print("Weights used:",
      f"LSTM={W_LSTM}, TCN={W_TCN}, NBEATS={W_NBEATS}")
