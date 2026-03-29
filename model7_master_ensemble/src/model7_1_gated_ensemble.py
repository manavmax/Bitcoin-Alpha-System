import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# Paths
# =========================
OUT_DIR = Path("model7_master_ensemble/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "model7_1_gated_signal.csv"

PRICE_FILE = "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"

# =========================
# Gating thresholds
# =========================
VOL_EXTREME = 2          # model 2
MACRO_RISK_OFF = 2       # model 6
DERIV_STRESS = 0.02      # model 3
TREND_MIN = 0.002        # model 1

# =========================
# Robust utilities
# =========================
def load_price_timeline(path):
    df = pd.read_csv(path)

    # Try all known BTC date formats
    for col in ["date", "Date", "timestamp", "time", "Open time"]:
        if col in df.columns:
            try:
                if "time" in col.lower() and df[col].dtype != object:
                    df["date"] = pd.to_datetime(df[col], unit="ms", utc=True)
                else:
                    df["date"] = pd.to_datetime(df[col], utc=True, errors="coerce")
                break
            except Exception:
                continue

    if "date" not in df.columns:
        raise RuntimeError("❌ Cannot infer date column from BTC price file")

    df = df.dropna(subset=["date"]).sort_values("date")
    return df[["date"]].reset_index(drop=True)

def load_signal(path, candidate_cols):
    if not Path(path).exists():
        return None

    df = pd.read_csv(path)

    # Infer date
    for col in ["date", "Date", "timestamp", "time"]:
        if col in df.columns:
            df["date"] = pd.to_datetime(df[col], utc=True, errors="coerce")
            break

    if "date" not in df.columns:
        return None

    # Infer signal column
    sig_col = None
    for c in candidate_cols:
        if c in df.columns:
            sig_col = c
            break

    if sig_col is None:
        return None

    return df[["date", sig_col]].rename(columns={sig_col: candidate_cols[0]})

# =========================
# Load master timeline
# =========================
print("📥 Loading BTC master timeline...")
timeline = load_price_timeline(PRICE_FILE)
df = timeline.copy()

# =========================
# Load all models safely
# =========================
models = {
    "signal_1": load_signal(
        "model_1_price_dynamics/results/model1_final_predictions.csv",
        ["model1_return_prediction"]
    ),
    "signal_3": load_signal(
        "model_3_derivatives_flow/results/model3_ensemble_predictions.csv",
        ["model3_ensemble"]
    ),
    "vol_regime": load_signal(
        "model_2_volatility/results/model2_final_volatility.csv",
        ["vol_regime"]
    ),
    "macro_regime": load_signal(
        "model_6_macro_liquidity/results/model6_final_signal.csv",
        ["macro_regime"]
    ),
}

# Merge available models
for name, m in models.items():
    if m is not None:
        df = df.merge(m, on="date", how="left")
        print(f"✅ {name} loaded")
    else:
        print(f"⚠️ {name} missing — gated out")

df.fillna(0.0, inplace=True)

# =========================
# HARD GATING LOGIC
# =========================
def gate(row):
    # 1️⃣ Volatility kill-switch
    if row.get("vol_regime", 0) >= VOL_EXTREME:
        return 0.0

    # 2️⃣ Macro risk-off
    if row.get("macro_regime", 0) >= MACRO_RISK_OFF:
        return 0.0

    # 3️⃣ Derivatives stress → Model 3
    if abs(row.get("signal_3", 0)) > DERIV_STRESS:
        return row.get("signal_3", 0)

    # 4️⃣ Trend-follow → Model 1
    if abs(row.get("signal_1", 0)) > TREND_MIN:
        return row.get("signal_1", 0)

    return 0.0

df["model7_1_signal"] = df.apply(gate, axis=1)

# =========================
# Save
# =========================
out = df[["date", "model7_1_signal"]]
out.to_csv(OUT_FILE, index=False)

print("\n✅ MODEL 7.1 HARD-GATED ENSEMBLE COMPLETE")
print(f"Saved → {OUT_FILE}")
print(f"Active trade ratio: {(out.model7_1_signal != 0).mean():.2%}")
