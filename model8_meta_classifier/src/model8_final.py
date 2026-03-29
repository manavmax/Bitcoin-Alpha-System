import pandas as pd
import joblib
import numpy as np

TAU = 0.60  # confidence threshold

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def safe_date(df):
    for c in ["date", "Date", "timestamp", "time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True)
            df.rename(columns={c: "date"}, inplace=True)
            return df
    raise RuntimeError("❌ No date column found")

def detect_macro_signal(df):
    candidates = [
        c for c in df.columns
        if "macro" in c.lower() or "liquidity" in c.lower()
    ]
    if not candidates:
        raise RuntimeError("❌ No macro signal column found in Model 6 output")
    return candidates[0]

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("🚀 Running FINAL Model 8 (8A + 8B)")

    # --------------------------------------------------
    # Load base dataset
    # --------------------------------------------------
    df = pd.read_csv(
        "model8_meta_classifier/data/model8_dataset.csv",
        parse_dates=["date"]
    )
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # --------------------------------------------------
    # Load volatility regimes (Model 2)
    # --------------------------------------------------
    vol = pd.read_csv(
        "model_2_volatility_risk/results/model2_final_volatility.csv"
    )
    vol = safe_date(vol)

    if "vol_regime" not in vol.columns:
        raise RuntimeError("❌ vol_regime missing in Model 2 output")

    vol = vol[["date", "vol_regime"]]

    # --------------------------------------------------
    # Load macro signal (Model 6)
    # --------------------------------------------------
    macro = pd.read_csv(
        "model_6_macro_liquidity/results/model6_final_signal.csv"
    )
    macro = safe_date(macro)

    macro_signal_col = detect_macro_signal(macro)
    print(f"📌 Using macro signal column: {macro_signal_col}")

    macro = macro[["date", macro_signal_col]].rename(
        columns={macro_signal_col: "macro_signal"}
    )

    # --------------------------------------------------
    # Merge regimes
    # --------------------------------------------------
    df = df.merge(vol, on="date", how="left")
    df = df.merge(macro, on="date", how="left")

    df["vol_regime"] = df["vol_regime"].fillna(1)
    df["macro_signal"] = df["macro_signal"].fillna(0.0)

    # Derive macro regime
    df["macro_regime"] = (df["macro_signal"] > 0).astype(int)

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    model8A = joblib.load(
        "model8_meta_classifier/models/model8A_regime_selector.pkl"
    )
    model8B = joblib.load(
        "model8_meta_classifier/models/model8B_directional.pkl"
    )

    # --------------------------------------------------
    # Stage 1 — Tradability (Model 8A)
    # --------------------------------------------------
    FEATURES_8A = ["vol_regime", "macro_regime", "macro_signal"]
    df["tradable"] = model8A.predict(df[FEATURES_8A])

    # --------------------------------------------------
    # Stage 2 — Direction + Confidence (Model 8B)
    # 🔑 CRITICAL FIX: enforce training feature order
    # --------------------------------------------------
    FEATURES_8B = model8B.get_booster().feature_names

    X_8B = df[FEATURES_8B]

    probs = model8B.predict_proba(X_8B)
    df["direction"] = model8B.predict(X_8B)
    df["confidence"] = probs.max(axis=1)

    # --------------------------------------------------
    # Final Signal Logic
    # --------------------------------------------------
    df["final_signal"] = "NO_TRADE"

    active = (df["tradable"] == 1) & (df["confidence"] >= TAU)

    df.loc[active & (df["direction"] == 1), "final_signal"] = "LONG"
    df.loc[active & (df["direction"] == 0), "final_signal"] = "SHORT"

    # --------------------------------------------------
    # Save output
    # --------------------------------------------------
    out = df[
        ["date", "final_signal", "confidence",
         "tradable", "vol_regime", "macro_regime"]
    ]

    out.to_csv(
        "model8_meta_classifier/results/model8_final_signal.csv",
        index=False
    )

    print("✅ MODEL 8 FINAL COMPLETE")
    print(f"Coverage @ τ={TAU}: {(active.mean() * 100):.2f}%")
    print("Saved → model8_meta_classifier/results/model8_final_signal.csv")

if __name__ == "__main__":
    main()
