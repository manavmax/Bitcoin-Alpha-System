import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE_DIR / "processed" / "model3_train.csv"
TEST_FILE  = BASE_DIR / "processed" / "model3_test.csv"

MODEL_OUT = BASE_DIR / "models"
RESULTS_OUT = BASE_DIR / "results"

MODEL_OUT.mkdir(exist_ok=True)
RESULTS_OUT.mkdir(exist_ok=True)

# ==================== CONFIG ====================
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3

FEATURES = [
    "long_liq",
    "short_liq",
    "liq_imbalance",
    "open_interest",
    "funding_rate",
    "long_short_ratio",
    "volume",
    "trade_count"
]

TARGET = "target_return"

# ==================== DATASET ====================
class SequenceDataset(Dataset):
    def __init__(self, df, features, target, seq_len, scaler=None):
        self.features = df[features].values.astype(np.float32)
        self.target = df[target].values.astype(np.float32)

        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)

        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.target[idx + self.seq_len]
        return torch.tensor(x), torch.tensor(y)

# ==================== TCN MODEL ====================
class TCN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.net(x)
        return y[:, :, -1].squeeze()

# ==================== LOAD DATA ====================
train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

train_ds = SequenceDataset(train_df, FEATURES, TARGET, SEQ_LEN)
test_ds  = SequenceDataset(test_df, FEATURES, TARGET, SEQ_LEN, scaler=train_ds.scaler)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ==================== TRAIN ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCN(len(FEATURES)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print("🚀 Training Model 3 TCN...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch}/{EPOCHS} | MSE: {np.mean(losses):.6f}")

# ==================== SAVE MODEL ====================
torch.save(model.state_dict(), MODEL_OUT / "model3_tcn.pt")

# ==================== EVALUATION (TEST SLICE) ====================
model.eval()
preds_test, actuals_test = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        p = model(x).cpu().numpy()
        preds_test.extend(p)
        actuals_test.extend(y.numpy())

mse = mean_squared_error(actuals_test, preds_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals_test, preds_test)

directional_acc = np.mean(np.sign(preds_test) == np.sign(actuals_test))

print("\n📊 MODEL 3 — TCN EVALUATION (TEST)")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Directional Accuracy: {directional_acc:.4f}")

# ==================== FULL-HISTORY INFERENCE (TRAIN + TEST) ====================
full_df = pd.concat([train_df, test_df], ignore_index=True)
full_ds = SequenceDataset(full_df, FEATURES, TARGET, SEQ_LEN, scaler=train_ds.scaler)
full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False)

full_preds = []
with torch.no_grad():
    for x, y in full_loader:
        x = x.to(device)
        p = model(x).cpu().numpy()
        full_preds.extend(p)

# Align predictions with dates: each prediction corresponds to index i+SEQ_LEN
full_out = full_df.iloc[SEQ_LEN:].copy()
full_out["tcn_pred"] = full_preds

RESULTS_OUT.mkdir(exist_ok=True)
full_out.to_csv(RESULTS_OUT / "model3_tcn_predictions.csv", index=False)

print("✅ Model 3 TCN training + full-history predictions completed")
