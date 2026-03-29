# model_2_volatility_risk/src/train_volatility_lstm.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from volatility_dataset import VolatilitySequenceDataset
from volatility_lstm import VolatilityLSTM
from pathlib import Path

# =====================
# Paths
# =====================
BASE = Path(__file__).resolve().parents[1]
DATA_FILE = BASE / "processed" / "model2_volatility_features.csv"
MODEL_FILE = BASE / "models" / "volatility_lstm.pth"
PRED_FILE = BASE / "results" / "volatility_lstm_preds.csv"

MODEL_FILE.parent.mkdir(exist_ok=True)
PRED_FILE.parent.mkdir(exist_ok=True)

# =====================
# Load data
# =====================
df = pd.read_csv(DATA_FILE)

FEATURES = [
    "garch_vol",
    "atr",
    "vov",
    "downside_vol",
    "vol_momentum",
    "vol_rank",
    "vol_regime",
    "crash_flag"
]

TARGET = "realized_vol"

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

# =====================
# Train / Test split
# =====================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================
# Dataset
# =====================
SEQ_LEN = 30

train_ds = VolatilitySequenceDataset(X_train, y_train, SEQ_LEN)
test_ds = VolatilitySequenceDataset(X_test, y_test, SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# =====================
# Model
# =====================
device = "cpu"
model = VolatilityLSTM(input_size=len(FEATURES)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# =====================
# Training
# =====================
EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()
    losses = []

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | MSE: {np.mean(losses):.6f}")

# =====================
# Evaluation
# =====================
model.eval()
preds, trues = [], []

with torch.no_grad():
    for Xb, yb in test_loader:
        p = model(Xb.to(device))
        preds.extend(p.cpu().numpy())
        trues.extend(yb.numpy())

preds = np.array(preds)
trues = np.array(trues)

mse = np.mean((preds - trues) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(preds - trues))

print("\n📊 Volatility LSTM Evaluation")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")

# =====================
# Save
# =====================
torch.save(model.state_dict(), MODEL_FILE)

out = df.iloc[-len(preds):].copy()
out["lstm_volatility"] = preds
out.to_csv(PRED_FILE, index=False)

print("✅ Volatility LSTM training completed")
print(f"Model saved → {MODEL_FILE}")
