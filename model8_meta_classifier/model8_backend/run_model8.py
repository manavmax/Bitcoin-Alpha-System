import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

DATA_MAIN = BASE / "data" / "model8_dataset.csv"
DATA_MACRO = BASE / "data" / "macro_features.csv"

OUT_FILE = BASE / "results" / "model8_final_signal.csv"

MODEL_8A = BASE / "models" / "model8A_regime_selector.pkl"
MODEL_8B = BASE / "models" / "model8B_directional.pkl"


# --------------------------------------------------
# Regime reconstruction (MATCH TRAINING)
# --------------------------------------------------
def build_vol_regime(df):
    # simple realized vol proxy (same logic used during training)
    vol = df["close"].pct_change().rolling(20).std()
    thresh = vol.quantile(0.6)
    return (vol > thresh).astype(int)


def build_macro_regime(macro_df):
    # macro_signal > 0 => Risk-On
    return (macro_df["macro_signal"] > 0).astype(int)


# --------------------------------------------------
def ensure_missing_flags(df, signals):
    for s in signals:
        miss = f"{s}_missing"
        if miss not in df.columns:
            df[miss] = df[s].isna().astype(int)
        df[s] = df[s].fillna(0.0)
    return df


# --------------------------------------------------
def main():
    print("📦 Loading datasets")
    df = pd.read_csv(DATA_MAIN, parse_dates=["date"])
    macro = pd.read_csv(DATA_MACRO, parse_dates=["date"])

    # ------------------------------------------------
    # Rebuild regimes (CRITICAL FIX)
    # ------------------------------------------------
    print("🧩 Reconstructing regime features")

    df["vol_regime"] = build_vol_regime(df)

    macro["macro_regime"] = build_macro_regime(macro)

    # merge macro regime
    df = df.merge(
        macro[["date", "macro_regime", "macro_signal"]],
        on="date",
        how="left"
    )

    df[["macro_regime", "macro_signal"]] = df[
        ["macro_regime", "macro_signal"]
    ].fillna(0)

    # ------------------------------------------------
    print("🧠 Loading trained models")
    model8A = joblib.load(MODEL_8A)
    model8B = joblib.load(MODEL_8B)

    # ------------------------------------------------
    # Model 8A — Tradability
    # ------------------------------------------------
    X_8A = df[model8A.feature_names_in_].copy()
    df["tradable"] = model8A.predict(X_8A)

    # ------------------------------------------------
    # Model 8B — Direction
    # ------------------------------------------------
    signal_cols = [
        "signal_1",
        "signal_2",
        "signal_3",
        "signal_4",
        "signal_6",
    ]

    X_8B = df[signal_cols].copy()
    X_8B = ensure_missing_flags(X_8B, signal_cols)

    probs = model8B.predict_proba(X_8B)
    df["confidence"] = np.max(probs, axis=1)
    df["direction"] = np.where(probs[:, 1] >= 0.5, "LONG", "SHORT")

    # ------------------------------------------------
    # FINAL SIGNAL (NEVER EMPTY)
    # ------------------------------------------------
    df["final_signal"] = "NO_TRADE"
    mask = (df["tradable"] == 1) & (df["confidence"] >= 0.70)

    df.loc[mask & (df["direction"] == "LONG"), "final_signal"] = "LONG"
    df.loc[mask & (df["direction"] == "SHORT"), "final_signal"] = "SHORT"

    # ------------------------------------------------
    out = df[
        ["date", "final_signal", "confidence", "vol_regime", "macro_regime"]
    ].copy()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print(f"✅ Model 8 inference complete → {OUT_FILE}")
    print(f"📊 Rows written: {len(out)}")


if __name__ == "__main__":
    main()
