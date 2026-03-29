import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# =========================
# Paths
# =========================
DATA_FILE = "model_4_onchain_fundamentals/processed/model4_onchain_features.csv"
MODEL_DIR = Path("model_4_onchain_fundamentals/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "onchain_lstm.pth"

# =========================
# Config
# =========================
SEQ_LEN = 30
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3

FEATURE_COLS = [
    "hash_rate_z",
    "difficulty_z",
    "tx_count_z",
    "tx_volume_usd_z",
    "miner_revenue_z",
    "network_activity_index"
]

TARGET_COL = "network_activity_index"  # ✅ internal on-chain target

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Dataset
# =========================
class OnChainDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx+self.seq_len], dtype=torch.float32)
        )

# =========================
# Model
# =========================
class OnChainLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

# =========================
# Load & Prepare Data
# =========================
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Fill slow-moving on-chain data
df = df.ffill()

# Drop rows still invalid
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)

print(f"📊 Samples available: {len(X)}")

if len(X) <= SEQ_LEN + 10:
    raise RuntimeError(
        f"❌ Too little data for LSTM. seq_len={SEQ_LEN}, samples={len(X)}"
    )

dataset = OnChainDataset(X, y, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# Train
# =========================
model = OnChainLSTM(input_size=len(FEATURE_COLS)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print("🚀 Training On-Chain LSTM...")

for epoch in range(EPOCHS):
    model.train()
    losses = []

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | MSE: {np.mean(losses):.6f}")

# =========================
# Save
# =========================
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ On-Chain LSTM trained & saved → {MODEL_PATH}")
