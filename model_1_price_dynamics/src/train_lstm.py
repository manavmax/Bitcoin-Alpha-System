# model_3/src/train_lstm.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json

from dataset import BTCSequenceDataset
from model import LSTMPricePredictor

# ---------------- CONFIG ----------------
SEQ_LEN = 30
BATCH_SIZE = 16
EPOCHS = 100   # 🔥 increased
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------- DATA ----------------
dataset = BTCSequenceDataset(seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------- MODEL ----------------
input_size = dataset.features.shape[1]

model = LSTMPricePredictor(input_size=input_size).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN ----------------
losses = []

print("🚀 Training LSTM (return prediction, 100 epochs)")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    losses.append(epoch_loss)

    print(f"Epoch {epoch}/{EPOCHS} | MSE: {epoch_loss:.6f}")

# ---------------- SAVE ----------------
torch.save(model.state_dict(), MODEL_DIR / "lstm.pt")

with open(RESULTS_DIR / "lstm_metrics.json", "w") as f:
    json.dump({"loss": losses}, f, indent=4)

print("✅ LSTM training completed successfully")
