import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

PRICE_FILE = "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_price_timeline():
    price = pd.read_csv(PRICE_FILE)

    if "date" in price.columns:
        price["date"] = pd.to_datetime(price["date"], utc=True)
    elif "Open time" in price.columns:
        price["date"] = pd.to_datetime(price["Open time"], utc=True)
    else:
        raise RuntimeError("❌ No usable date column in BTC price file")

    price = price.sort_values("date").reset_index(drop=True)
    price["target_return"] = price["Close"].pct_change().shift(-1)
    return price[["date", "target_return"]].dropna().reset_index(drop=True)


def load_signal_with_date(path, value_col=None):
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise RuntimeError(f"❌ No date column in {path}")

    df["date"] = pd.to_datetime(df["date"], utc=True)

    # If value_col explicitly provided
    if value_col:
        if value_col not in df.columns:
            raise RuntimeError(f"❌ {value_col} missing in {path}")
        return df[["date", value_col]]

    # Otherwise auto-detect signal column
    candidates = [
        c for c in df.columns
        if c != "date" and np.issubdtype(df[c].dtype, np.number)
    ]

    if len(candidates) == 0:
        raise RuntimeError(f"❌ No numeric signal column found in {path}")

    if len(candidates) > 1:
        print(f"⚠️ Multiple candidates in {path}, using: {candidates[0]}")

    return df[["date", candidates[0]]]


def load_model1_signal(price_dates):
    """
    Model 1 has NO date → align to BTC timeline from the END
    """
    df = pd.read_csv(
        "model_1_price_dynamics/results/model1_final_predictions.csv"
    )

    if "model1_return_prediction" not in df.columns:
        raise RuntimeError("❌ model1_return_prediction missing")

    n = len(df)
    aligned_dates = price_dates.iloc[-n:].reset_index(drop=True)

    out = pd.DataFrame({
        "date": aligned_dates,
        "signal_1": df["model1_return_prediction"].values
    })

    return out


# --------------------------------------------------
# Dataset construction
# --------------------------------------------------

def build_dataset():
    price = load_price_timeline()

    # Model 1 (special handling)
    m1 = load_model1_signal(price["date"])

    # Other models (normal merge)
    m2 = load_signal_with_date(
        "model_2_volatility_risk/results/model2_final_volatility.csv",
        "model2_volatility"
    ).rename(columns={"model2_volatility": "signal_2"})

    m3 = load_signal_with_date(
        "model_3_derivatives_flow/results/model3_ensemble_predictions.csv",
        "model3_ensemble"
    ).rename(columns={"model3_ensemble": "signal_3"})

    m4 = load_signal_with_date(
        "model_4_onchain_fundamentals/results/model4_final_signal.csv",
        "model4_signal"
    ).rename(columns={"model4_signal": "signal_4"})

    # --- Model 6 (Macro) ---
    m6 = load_signal_with_date(
       "model_6_macro_liquidity/results/model6_final_signal.csv"
    )

# Rename detected signal column safely
    m6 = m6.rename(columns={m6.columns[1]: "signal_6"})



    df = price.merge(m1, on="date", how="inner")
    df = df.merge(m2, on="date", how="inner")
    df = df.merge(m3, on="date", how="inner")
    df = df.merge(m4, on="date", how="inner")
    df = df.merge(m6, on="date", how="inner")

    # 3-class target
    df["target"] = 0
    df.loc[df["target_return"] > 0.002, "target"] = 1
    df.loc[df["target_return"] < -0.002, "target"] = -1

    return df.dropna().reset_index(drop=True)


# --------------------------------------------------
# Training
# --------------------------------------------------

def main():
    print("📥 Building Model 7.3 dataset...")
    df = build_dataset()

    FEATURES = [
        "signal_1",
        "signal_2",
        "signal_3",
        "signal_4",
        "signal_6",
    ]

    X = df[FEATURES].values
    y = df["target"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    print("🚀 Training Model 7.3 (BUY / SELL / NEUTRAL)...")
    model.fit(Xtr, ytr)

    preds = model.predict(Xte)

    print("\n📊 VALIDATION REPORT\n")
    print(classification_report(yte, preds, digits=4))
    os.makedirs("model7_master_ensemble/models", exist_ok=True)
    joblib.dump(model, "model7_master_ensemble/models/model7_3_classifier.pkl")
    joblib.dump(scaler, "model7_master_ensemble/models/model7_3_scaler.pkl")

    print("✅ Model 7.3 trained & saved")


if __name__ == "__main__":
    main()
