import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ===============================
# CONFIG
# ===============================
MACRO_FEATURES_FILE = "model_6_macro_liquidity/processed/model6_macro_features.csv"
PRICE_FILE = "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"

LSTM_MODEL_FILE = "model_6_macro_liquidity/models/macro_lstm.pth"
TREE_MODEL_FILE = "model_6_macro_liquidity/models/macro_tree.pkl"

OUTPUT_FILE = "model_6_macro_liquidity/results/model6_final_signal.csv"

FEATURE_COLS = [
    "DFF",
    "m2_yoy",
    "real_rate",
    "yield_inversion",
    "vix_z",
    "sp_trend"
]

SEQ_LEN = 90
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Path("model_6_macro_liquidity/results").mkdir(parents=True, exist_ok=True)

# ===============================
# MODEL DEFINITIONS
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
df = pd.read_csv(MACRO_FEATURES_FILE, parse_dates=["date"])

print("📥 Loading BTC price...")
price = pd.read_csv(PRICE_FILE)

# --- normalize date columns ---
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

if "date" in price.columns:
    price["date"] = pd.to_datetime(price["date"], utc=True).dt.tz_localize(None)
else:
    price["date"] = pd.to_datetime(price["Open time"], utc=True).dt.tz_localize(None)

price = price.sort_values("date")
price["target_return"] = price["Close"].pct_change(7).shift(-7)

df = df.merge(
    price[["date", "target_return"]],
    on="date",
    how="inner"
)

# Keep the most recent rows even when target_return is unavailable (latest 7 days).
# For live inference, we need macro signals up to the latest feature date.
df = df.dropna(subset=FEATURE_COLS)
df = df.sort_values("date").reset_index(drop=True)

print(f"📊 Samples after merge & clean: {len(df)}")

# ===============================
# SCALE FEATURES
# ===============================
scaler = StandardScaler()
df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

X = df[FEATURE_COLS].values
y = df["target_return"].values

# ===============================
# LOAD MODELS
# ===============================
print("📦 Loading Macro LSTM...")
lstm = MacroLSTM(input_size=len(FEATURE_COLS)).to(DEVICE)
lstm.load_state_dict(torch.load(LSTM_MODEL_FILE, map_location=DEVICE))
lstm.eval()

print("📦 Loading Macro Tree...")
tree = joblib.load(TREE_MODEL_FILE)

# ===============================
# LSTM PREDICTIONS
# ===============================
lstm_preds = np.full(len(df), np.nan)

with torch.no_grad():
    for i in range(SEQ_LEN, len(X)):
        seq = torch.tensor(
            X[i - SEQ_LEN:i],
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        lstm_preds[i] = lstm(seq).cpu().item()

# ===============================
# TREE PREDICTIONS
# ===============================
tree_preds = tree.predict(X)

# ===============================
# FINAL ENSEMBLE
# ===============================
# Macro logic:
# - LSTM = slow-cycle filter
# - Tree = regime-sensitive adjustment
df["macro_lstm_signal"] = lstm_preds
df["macro_tree_signal"] = tree_preds

df["macro_final_signal"] = (
    0.6 * df["macro_lstm_signal"] +
    0.4 * df["macro_tree_signal"]
)

# ===============================
# SAVE OUTPUT
# ===============================
out = df[[
    "date",
    "macro_lstm_signal",
    "macro_tree_signal",
    "macro_final_signal",
    "target_return"
]]

out.to_csv(OUTPUT_FILE, index=False)

print("✅ MODEL 6 ENSEMBLE COMPLETED")
print(f"Saved → {OUTPUT_FILE}")
