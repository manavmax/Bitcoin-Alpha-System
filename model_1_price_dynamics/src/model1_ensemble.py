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
# drop_na_target=False keeps the most recent candle (whose next-day return is not
# known yet) so we can also emit a live signal for the current day.
dataset = BTCSequenceDataset(seq_len=SEQ_LEN, drop_na_target=False)
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
# Prediction i (sequence starting at row i) targets row i + SEQ_LEN, so attach
# that row's timestamp explicitly. build_dataset.py then merges Model 1 by date
# instead of aligning it positionally from the end — positional alignment
# silently shifted every prediction whenever the history length changed.
pred_dates = pd.to_datetime(dataset.timestamps[SEQ_LEN:])

df = pd.DataFrame({
    "date": pred_dates,
    "model1_return_prediction": np.asarray(final_preds).reshape(-1),
    "true_return": np.asarray(targets).reshape(-1),
})

df.to_csv(OUTPUT_PATH, index=False)

print("✅ Model 1 ensemble completed")
print(f"Saved final Model 1 predictions → {OUTPUT_PATH}")
print(f"Rows: {len(df)} | date range: {df['date'].min()} → {df['date'].max()}")
print("Weights used:",
      f"LSTM={W_LSTM}, TCN={W_TCN}, NBEATS={W_NBEATS}")
