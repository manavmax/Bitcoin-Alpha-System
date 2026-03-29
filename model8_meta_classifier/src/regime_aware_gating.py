import pandas as pd
import numpy as np

# ===============================
# CONFIG
# ===============================

PRED_PATH = "model8_meta_classifier/results/model8_predictions.csv"
VOL_PATH = "model_2_volatility_risk/results/model2_final_volatility.csv"
MACRO_PATH = "model_6_macro_liquidity/results/model6_final_signal.csv"
OUT_PATH = "model8_meta_classifier/results/model8_regime_aware_final.csv"

# Regime-specific confidence thresholds
THRESHOLDS = {
    (0, 1): 0.55,  # low vol, risk-on
    (0, 0): 0.65,  # low vol, risk-off
    (1, 1): 0.60,  # high vol, risk-on
    (1, 0): 0.70,  # high vol, risk-off
}

# ===============================
# HELPERS
# ===============================

def normalize_date(df):
    for c in df.columns:
        if "date" in c.lower():
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
            df.rename(columns={c: "date"}, inplace=True)
            return df
    raise RuntimeError("❌ No date column found")

# ===============================
# MAIN
# ===============================

def main():
    print("📥 Loading Model 8 predictions...")
    df = pd.read_csv(PRED_PATH)
    df = normalize_date(df)

    REQUIRED = ["pred_class", "confidence", "future_ret"]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise RuntimeError(f"❌ Missing columns in predictions: {missing}")

    # ----------------------------------
    # Reconstruct TRUE LABEL
    # ----------------------------------
    df["true_label"] = (df["future_ret"] > 0).astype(int)

    # ----------------------------------
    # Volatility regime (Model 2)
    # ----------------------------------
    print("📥 Loading volatility regimes (Model 2)...")
    vol = pd.read_csv(VOL_PATH)
    vol = normalize_date(vol)

    if "vol_regime" not in vol.columns:
        raise RuntimeError("❌ vol_regime missing in Model 2 output")

    df = df.merge(vol[["date", "vol_regime"]], on="date", how="left")
    df["vol_regime"] = df["vol_regime"].fillna(1).astype(int)

    # ----------------------------------
    # Macro regime (Model 6)
    # ----------------------------------
    print("📥 Loading macro regimes (Model 6)...")
    macro = pd.read_csv(MACRO_PATH)
    macro = normalize_date(macro)

    macro_col = None
    for c in macro.columns:
        if "regime" in c.lower() or "macro" in c.lower():
            macro_col = c
            break

    if macro_col is None:
        raise RuntimeError("❌ No macro regime column found in Model 6 output")

    print(f"⚠️ Using macro column: {macro_col}")

    df = df.merge(macro[["date", macro_col]], on="date", how="left")
    df.rename(columns={macro_col: "macro_regime"}, inplace=True)

    # Convert macro signal → binary regime
    df["macro_regime"] = (df["macro_regime"] > 0).astype(int)

    # ----------------------------------
    # REGIME-AWARE CONFIDENCE GATING
    # ----------------------------------
    print("\n🚦 Applying regime-aware confidence gating...")

    trade_flags = []
    for _, r in df.iterrows():
        key = (r["vol_regime"], r["macro_regime"])
        tau = THRESHOLDS.get(key, 0.7)
        trade_flags.append(int(r["confidence"] >= tau))

    df["trade"] = trade_flags

    traded = df[df["trade"] == 1].copy()

    coverage = len(traded) / len(df)

    if len(traded) > 0:
        da = (traded["pred_class"] == traded["true_label"]).mean()
    else:
        da = np.nan

    # ----------------------------------
    # REPORT
    # ----------------------------------
    print("\n📊 MODEL 8 — REGIME-AWARE RESULTS")
    print(f"Coverage: {coverage*100:.2f}%")
    print(f"Directional Accuracy (on trades): {da:.4f}")

    print("\n📊 Regime Breakdown")
    for vr in [0, 1]:
        for mr in [0, 1]:
            sub = traded[(traded["vol_regime"] == vr) &
                          (traded["macro_regime"] == mr)]
            if len(sub) > 20:
                acc = (sub["pred_class"] == sub["true_label"]).mean()
                print(f"Vol={vr}, Macro={mr} | Trades={len(sub):4d} | DA={acc:.3f}")

    # ----------------------------------
    # SAVE
    # ----------------------------------
    df.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Saved → {OUT_PATH}")

# ===============================
if __name__ == "__main__":
    main()
