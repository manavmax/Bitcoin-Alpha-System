import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

MACRO_FILE = BASE_DIR / "model_6_macro_liquidity/processed/model6_macro_features.csv"
PRICE_FILE = BASE_DIR / "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"
MODEL_OUT = BASE_DIR / "model_6_macro_liquidity/models/macro_tree.pkl"

# =========================
# LOAD DATA
# =========================
print("📥 Loading macro features...")
macro = pd.read_csv(MACRO_FILE, parse_dates=["date"])

print("📥 Loading BTC price data...")
price = pd.read_csv(PRICE_FILE)

# --- Robust date handling ---
if "date" not in price.columns:
    if "Open time" in price.columns:
        price["date"] = pd.to_datetime(price["Open time"], errors="coerce")
    else:
        raise RuntimeError("❌ No valid date column in BTC price file")

price["date"] = pd.to_datetime(price["date"]).dt.tz_localize(None)
macro["date"] = macro["date"].dt.tz_localize(None)

# =========================
# TARGET CONSTRUCTION
# =========================
price = price.sort_values("date")
price["future_return_30d"] = price["Close"].pct_change(30).shift(-30)

df = macro.merge(
    price[["date", "future_return_30d"]],
    on="date",
    how="inner"
)

# =========================
# FEATURE SELECTION
# =========================
FEATURE_COLS = [
    c for c in df.columns
    if c not in ["date", "future_return_30d", "macro_regime"]
]

df = df.dropna(subset=FEATURE_COLS + ["future_return_30d"])

print(f"📊 Samples after cleaning: {len(df)}")

if len(df) < 500:
    raise RuntimeError("❌ Not enough samples to train Macro Tree")

X = df[FEATURE_COLS].values
y = df["future_return_30d"].values

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    shuffle=False
)

# =========================
# MODEL
# =========================
print("🌳 Training Macro Tree Model...")

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=30,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, pred)
corr = np.corrcoef(y_test, pred)[0, 1]

print("\n📊 MODEL 6B — MACRO TREE EVALUATION")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"Correlation: {corr:.4f}")

# =========================
# SAVE MODEL
# =========================
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_OUT)

print(f"\n✅ Macro Tree model saved → {MODEL_OUT}")
