import pandas as pd
import os

OUT_PATH = "model8_meta_classifier/data/model8A_dataset.csv"

def normalize_date(df):
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
            df[c] = df[c].dt.tz_localize(None)
            df.rename(columns={c: "date"}, inplace=True)
            return df
    raise RuntimeError("❌ No date column found")

def main():
    print("📥 Building Model 8A dataset (TRUE FULL SAMPLE MODE)...")

    # =====================================================
    # 1. LOAD BTC PRICE — MASTER TIMELINE (2919 rows)
    # =====================================================
    price = pd.read_csv(
        "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"
    )
    price = normalize_date(price)

    price = price[["date"]].sort_values("date").drop_duplicates()
    print("📌 BTC master samples:", len(price))

    df = price.copy()

    # =====================================================
    # 2. LOAD VOLATILITY REGIME (Model 2)
    # =====================================================
    vol = pd.read_csv(
        "model_2_volatility_risk/results/model2_final_volatility.csv"
    )
    vol = normalize_date(vol)

    if "vol_regime" not in vol.columns:
        raise RuntimeError("❌ vol_regime missing in Model 2 output")

    vol = vol[["date", "vol_regime"]].sort_values("date")

    df = df.merge(vol, on="date", how="left")

    # Conservative default: medium volatility
    df["vol_regime"] = df["vol_regime"].fillna(1)

    # =====================================================
    # 3. LOAD MACRO SIGNAL (Model 6)
    # =====================================================
    macro = pd.read_csv(
        "model_6_macro_liquidity/results/model6_final_signal.csv"
    )
    macro = normalize_date(macro)

    macro_signal_col = None
    for c in macro.columns:
        if "macro" in c.lower() and "signal" in c.lower():
            macro_signal_col = c
            break

    if macro_signal_col is None:
        raise RuntimeError(
            f"❌ No macro signal column found in Model 6 output: {macro.columns.tolist()}"
        )

    print(f"⚠️ Using macro signal column: {macro_signal_col}")

    macro = macro[["date", macro_signal_col]].rename(
        columns={macro_signal_col: "macro_signal"}
    ).sort_values("date")

    df = df.merge(macro, on="date", how="left")

    # Macro is slow → forward fill
    df["macro_signal"] = df["macro_signal"].ffill()
    df["macro_signal"] = df["macro_signal"].fillna(0.0)

    # =====================================================
    # 4. DERIVE REGIMES
    # =====================================================
    df["macro_regime"] = (df["macro_signal"] < 0).astype(int)

    # =====================================================
    # 5. DEFINE TRADABLE REGIME (LABEL)
    # =====================================================
    df["tradable"] = (
        (df["vol_regime"] == 1) &
        (df["macro_regime"] == 0)
    ).astype(int)

    # =====================================================
    # 6. SAVE
    # =====================================================
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("✅ Model 8A dataset BUILT SUCCESSFULLY")
    print("Samples:", len(df))
    print("\nTradable ratio:")
    print(df["tradable"].value_counts(normalize=True))

if __name__ == "__main__":
    main()
