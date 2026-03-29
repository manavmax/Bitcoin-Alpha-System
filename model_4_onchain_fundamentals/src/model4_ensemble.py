import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path

# =============================
# CONFIG
# =============================
BASE_DIR = Path("model_4_onchain_fundamentals")

DATA_FILE = BASE_DIR / "processed/model4_onchain_features.csv"
PRICE_FILE = "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"

LSTM_MODEL = BASE_DIR / "models/onchain_lstm.pth"
TR_MODEL   = BASE_DIR / "models/onchain_transformer.pth"

OUT_FILE = BASE_DIR / "results/model4_final_signal.csv"

DEVICE = "cpu"
SEQ_LEN = 60

# ⚠️ MUST MATCH TRAINING EXACTLY
LSTM_FEATURES = [
    "hash_rate_z",
    "difficulty_z",
    "tx_count_z",
    "tx_volume_usd_z",
    "miner_revenue_z",
    "network_activity_index",
]

TR_FEATURES = LSTM_FEATURES + ["onchain_regime"]

TARGET_COL = "Close"

# =============================
# MODELS (MATCH TRAINING)
# =============================
class OnchainLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, 1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class OnChainTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.fc(x[:, -1])

# =============================
# LOAD DATA
# =============================
print("📥 Loading data...")

onchain = pd.read_csv(DATA_FILE, parse_dates=["date"])
onchain["date"] = onchain["date"].dt.tz_localize(None)

price = pd.read_csv(PRICE_FILE)
price["date"] = pd.to_datetime(price["Open time"]).dt.tz_localize(None)
price[TARGET_COL] = price["Close"]

df = onchain.merge(
    price[["date", TARGET_COL]],
    on="date",
    how="inner"
)

print(f"📊 Rows after merge: {len(df)}")

# =============================
# CLEANING
# =============================
df["onchain_regime"] = df["onchain_regime"].astype(float)
df[LSTM_FEATURES + ["onchain_regime"]] = df[LSTM_FEATURES + ["onchain_regime"]].ffill()
df = df.dropna(subset=[TARGET_COL])

print(f"📊 Rows after cleaning: {len(df)}")

if len(df) <= SEQ_LEN:
    raise RuntimeError("❌ Not enough data for Model 4 ensemble")

# =============================
# BUILD SEQUENCES
# =============================
def make_sequences(data, seq_len):
    return torch.tensor(
        np.array([data[i:i+seq_len] for i in range(len(data)-seq_len)]),
        dtype=torch.float32
    )

X_lstm = make_sequences(df[LSTM_FEATURES].values, SEQ_LEN)
X_tr   = make_sequences(df[TR_FEATURES].values, SEQ_LEN)

# =============================
# LOAD MODELS
# =============================
lstm = OnchainLSTM(len(LSTM_FEATURES))
lstm.load_state_dict(torch.load(LSTM_MODEL, map_location=DEVICE))
lstm.eval()

transformer = OnChainTransformer(len(TR_FEATURES))
transformer.load_state_dict(torch.load(TR_MODEL, map_location=DEVICE))
transformer.eval()

# =============================
# INFERENCE
# =============================
with torch.no_grad():
    lstm_pred = lstm(X_lstm).squeeze().numpy()
    tr_pred   = transformer(X_tr).squeeze().numpy()

# =============================
# ENSEMBLE
# =============================
alpha = 0.6
final_pred = alpha * lstm_pred + (1 - alpha) * tr_pred

# =============================
# SAVE
# =============================
out = pd.DataFrame({
    "date": df["date"].iloc[SEQ_LEN:].values,
    "onchain_lstm": lstm_pred,
    "onchain_transformer": tr_pred,
    "model4_signal": final_pred,
})

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_FILE, index=False)

print("✅ MODEL 4 ENSEMBLE COMPLETED")
print(f"Saved → {OUT_FILE}")
