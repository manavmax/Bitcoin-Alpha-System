import pandas as pd
import numpy as np

# =============================
# CONFIG
# =============================

PRICE_FILE = "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"

MODEL_FILES = {
    "m1": "model_1_price_dynamics/results/model1_final_predictions.csv",
    "m2": "model_2_volatility_risk/results/model2_final_volatility.csv",
    "m3": "model_3_derivatives_flow/results/model3_ensemble_predictions.csv",
    "m4": "model_4_onchain_fundamentals/results/model4_final_signal.csv",
    "m6": "model_6_macro_liquidity/results/model6_final_signal.csv",
}

OUTPUT_FILE = "model7_master_ensemble/results/model7_final_signal.csv"

# =============================
# UTILITIES
# =============================

def load_price_timeline():
    df = pd.read_csv(PRICE_FILE)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "Open time" in df.columns:
        df["date"] = pd.to_datetime(df["Open time"])
    else:
        raise RuntimeError("❌ BTC price file has no date column")

    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "Close"]]

def safe_load_model(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Failed loading {path}: {e}")
        return None

def extract_signal(df):
    """
    Attempts to extract the most reasonable signal column.
    """
    if df is None:
        return None

    candidates = [
        "final_signal",
        "model_ensemble",
        "model3_ensemble",
        "signal",
        "prediction",
        "return_prediction",
        "model1_return_prediction",
    ]

    for c in candidates:
        if c in df.columns:
            return df[c].astype(float)

    # fallback: last numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return df[numeric_cols[-1]].astype(float)

    return None

def align_to_master(signal, master_len):
    """
    Aligns a signal array to master timeline safely.
    """
    if signal is None:
        return np.zeros(master_len)

    signal = signal.values if isinstance(signal, pd.Series) else np.array(signal)

    if len(signal) == master_len:
        return signal

    if len(signal) > master_len:
        return signal[-master_len:]

    # pad front with zeros
    pad = master_len - len(signal)
    return np.concatenate([np.zeros(pad), signal])

# =============================
# MAIN
# =============================

def main():
    print("🚀 Running Model 7 Master Ensemble (error-proof)")

    price = load_price_timeline()
    n = len(price)

    # -----------------------------
    # Load models safely
    # -----------------------------

    signals = {}

    for key, path in MODEL_FILES.items():
        df = safe_load_model(path)
        sig = extract_signal(df)
        signals[key] = align_to_master(sig, n)
        print(f"✅ {key} loaded | active = {np.any(signals[key] != 0)}")

    # -----------------------------
    # Volatility gating (Model 2)
    # -----------------------------

    vol = np.abs(signals["m2"])
    vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

    # High vol → suppress slow models
    w_fast = 1.0 - vol_norm
    w_slow = vol_norm

    # -----------------------------
    # Static base weights
    # -----------------------------

    weights = {
        "m1": 0.30,  # price
        "m3": 0.30,  # derivatives
        "m4": 0.15,  # on-chain (bias only)
        "m6": 0.15,  # macro (risk filter)
    }

    # -----------------------------
    # Final ensemble
    # -----------------------------

    ensemble = (
        weights["m1"] * signals["m1"] * w_fast +
        weights["m3"] * signals["m3"] * w_fast +
        weights["m4"] * signals["m4"] * w_slow +
        weights["m6"] * signals["m6"] * w_slow
    )

    out = price.copy()
    out["model7_signal"] = ensemble

    out.to_csv(OUTPUT_FILE, index=False)

    print("✅ MODEL 7 COMPLETE")
    print(f"Saved → {OUTPUT_FILE}")

# =============================
if __name__ == "__main__":
    main()
