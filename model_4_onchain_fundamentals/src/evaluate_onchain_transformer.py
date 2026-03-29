# model_4_onchain_fundamentals/src/evaluate_onchain_transformer.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# =========================
# CONFIG
# =========================
FEATURE_FILE = "model_4_onchain_fundamentals/processed/model4_onchain_features.csv"
MODEL_FILE   = "model_4_onchain_fundamentals/models/onchain_transformer.pth"

SEQ_LEN = 90

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
df = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

df[FEATURE_COLS] = df[FEATURE_COLS].ffill()
df["target"] = df["network_activity_index"].shift(-1)
df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)

# =========================
# MODEL (MUST MATCH TRAINING)
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

# =========================
# LOAD MODEL
# =========================
model = OnChainTransformer(len(FEATURE_COLS))
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()

# =========================
# EVALUATION
# =========================
X = df[FEATURE_COLS].values.astype(np.float32)
y = df["target"].values.astype(np.float32)

preds = []
targets = []

with torch.no_grad():
    for i in range(len(X) - SEQ_LEN):
        seq = torch.tensor(X[i:i+SEQ_LEN]).unsqueeze(0)
        pred = model(seq).item()
        preds.append(pred)
        targets.append(y[i+SEQ_LEN])

preds = np.array(preds)
targets = np.array(targets)

mse = np.mean((preds - targets) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(preds - targets))

print("\n📊 MODEL 4 — ON-CHAIN TRANSFORMER EVALUATION")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
