# model_2_volatility_risk/src/train_garch.py

import pandas as pd
import numpy as np
from arch import arch_model
from pathlib import Path

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "btc_daily.csv"
OUT_FILE = BASE_DIR / "processed" / "garch_output.csv"

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# =========================
# Load price data
# =========================
df = pd.read_csv(DATA_FILE)

# ---- Date handling (robust)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
elif "Open time" in df.columns:
    df["date"] = pd.to_datetime(df["Open time"], utc=True, errors="coerce").dt.tz_localize(None)
else:
    raise RuntimeError("❌ No date column found")

df = df.sort_values("date").reset_index(drop=True)

# ---- Ensure Close exists
if "Close" not in df.columns:
    raise RuntimeError("❌ Close price column missing")

# =========================
# Returns
# =========================
df["log_return"] = np.log(df["Close"]).diff()
df = df.dropna(subset=["date", "Close", "log_return"]).reset_index(drop=True)

returns = df["log_return"] * 100  # GARCH expects %

# =========================
# GARCH(1,1) Model
# =========================
print("🚀 Training GARCH(1,1)...")

garch = arch_model(
    returns,
    mean="Zero",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal"
)

res = garch.fit(disp="off")

df["garch_vol"] = res.conditional_volatility / 100.0

# =========================
# Evaluation summary
# =========================
print("\n📊 GARCH VOLATILITY SUMMARY")
print(df["garch_vol"].describe())

# Volatility clustering sanity check
autocorr = df["garch_vol"].autocorr(lag=1)
print(f"\nVolatility autocorr (lag 1): {autocorr:.4f}")

# =========================
# Save
# =========================
df_out = df[
    ["date", "Close", "log_return", "garch_vol"]
]

df_out.to_csv(OUT_FILE, index=False)

print(f"\n✅ GARCH output saved → {OUT_FILE}")
