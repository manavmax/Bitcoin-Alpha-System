# model_4_onchain_fundamentals/src/train_onchain_transformer.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
ONCHAIN_FILE = "model_4_onchain_fundamentals/processed/model4_onchain_features.csv"
MODEL_OUT   = "model_4_onchain_fundamentals/models/onchain_transformer.pth"

SEQ_LEN = 90
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3

FEATURE_COLS = [
    "hash_rate",
    "difficulty",
    "tx_count",
    "tx_volume_usd",
    "miner_revenue",
    "network_activity_index",
    "onchain_regime"
]

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(ONCHAIN_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# =========================
# FIX MISSING DATA (CRITICAL)
# =========================
df[FEATURE_COLS] = df[FEATURE_COLS].ffill()

# =========================
# TARGET (ON-CHAIN DYNAMICS)
# =========================
df["target"] = df["network_activity_index"].shift(-1)

# =========================
# FINAL CLEAN
# =========================
df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

print(f"📊 Rows after clean: {len(df)}")

if len(df) <= SEQ_LEN:
    raise RuntimeError(
        f"❌ Not enough data. rows={len(df)}, seq_len={SEQ_LEN}"
    )

# =========================
# DATASET
# =========================
class OnChainTransformerDataset(Dataset):
    def __init__(self, df, features, target, seq_len):
        self.X = df[features].values.astype(np.float32)
        self.y = df[target].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len]),
            torch.tensor(self.y[idx+self.seq_len])
        )

dataset = OnChainTransformerDataset(
    df, FEATURE_COLS, "target", SEQ_LEN
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# MODEL
# =========================
class OnChainTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embed = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.fc(x[:, -1]).squeeze()

model = OnChainTransformer(len(FEATURE_COLS))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# =========================
# TRAIN
# =========================
print("🚀 Training On-Chain Transformer...")

for epoch in range(1, EPOCHS + 1):
    losses = []
    for X, y in loader:
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{EPOCHS} | MSE: {np.mean(losses):.6f}")

torch.save(model.state_dict(), MODEL_OUT)
print(f"✅ On-Chain Transformer trained & saved → {MODEL_OUT}")
