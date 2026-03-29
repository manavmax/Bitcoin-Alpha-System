import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ===============================
# CONFIG
# ===============================
DATA_FILE = "model_6_macro_liquidity/processed/model6_macro_features.csv"
PRICE_FILE = "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"
MODEL_OUT = "model_6_macro_liquidity/models/macro_lstm.pth"

FEATURE_COLS = [
    "DFF",
    "m2_yoy",
    "real_rate",
    "yield_inversion",
    "vix_z",
    "sp_trend"
]

SEQ_LEN = 90
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Path("model_6_macro_liquidity/models").mkdir(parents=True, exist_ok=True)

# ===============================
# DATASET
# ===============================
class MacroDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X, self.y = [], []
        for i in range(seq_len, len(X)):
            self.X.append(X[i - seq_len:i])
            self.y.append(y[i])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===============================
# MODEL
# ===============================
class MacroLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ===============================
# LOAD DATA
# ===============================
print("📥 Loading macro features...")
df = pd.read_csv(DATA_FILE, parse_dates=["date"])

print("📥 Loading BTC price...")
price = pd.read_csv(PRICE_FILE)

# --- normalize price date column ---
if "date" in price.columns:
    price["date"] = pd.to_datetime(price["date"], utc=True).dt.tz_localize(None)
elif "Open time" in price.columns:
    price["date"] = pd.to_datetime(price["Open time"], utc=True).dt.tz_localize(None)

df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

# --- target: forward 7-day return ---
price = price.sort_values("date")
price["target_return"] = price["Close"].pct_change(7).shift(-7)

df = df.merge(
    price[["date", "target_return"]],
    on="date",
    how="inner"
)

df = df.dropna(subset=FEATURE_COLS + ["target_return"])

print(f"📊 Samples after merge & clean: {len(df)}")

# ===============================
# SCALE FEATURES
# ===============================
scaler = StandardScaler()
df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

X = df[FEATURE_COLS].values
y = df["target_return"].values

dataset = MacroDataset(X, y, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===============================
# TRAIN
# ===============================
model = MacroLSTM(input_size=len(FEATURE_COLS)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print("🚀 Training Macro LSTM...")

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{EPOCHS} | MSE: {total_loss / len(loader):.6f}")

torch.save(model.state_dict(), MODEL_OUT)
print(f"✅ Macro LSTM trained & saved → {MODEL_OUT}")
