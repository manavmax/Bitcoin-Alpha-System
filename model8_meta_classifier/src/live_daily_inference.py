import pandas as pd
import joblib
from datetime import timedelta

DATASET_PATH = "model8_meta_classifier/data/model8_dataset.csv"
MODEL8A_PATH = "model8_meta_classifier/models/model8A_regime_selector.pkl"
MODEL8B_PATH = "model8_meta_classifier/models/model8B_directional.pkl"
OUTPUT_PATH = "model8_meta_classifier/results/model8_live_signal.csv"

THRESHOLD = 0.6

def main():
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    df = df.sort_values("date")

    latest = df.iloc[-1:].copy()

    model8A = joblib.load(MODEL8A_PATH)
    model8B = joblib.load(MODEL8B_PATH)

    FEATURES_8A = ["vol_regime", "macro_signal"]
    FEATURES_8B = [
        "signal_1","signal_2","signal_3","signal_4","signal_6",
        "signal_1_missing","signal_2_missing","signal_3_missing",
        "signal_4_missing","signal_6_missing"
    ]

    tradable = model8A.predict(latest[FEATURES_8A])[0]

    final_signal = "NO_TRADE"
    confidence = 0.0

    if tradable == 1:
        probs = model8B.predict_proba(latest[FEATURES_8B])[0]
        confidence = probs.max()

        if confidence >= THRESHOLD:
            final_signal = "LONG" if probs[1] > probs[0] else "SHORT"

    output = {
        "prediction_for_date": latest["date"].iloc[0] + timedelta(days=1),
        "signal": final_signal,
        "confidence": round(float(confidence), 3),
        "tradable": int(tradable),
        "vol_regime": int(latest["vol_regime"].iloc[0]),
        "macro_regime": "Risk-On" if latest["macro_signal"].iloc[0] > 0 else "Risk-Off"
    }

    pd.DataFrame([output]).to_csv(OUTPUT_PATH, index=False)
    print("✅ LIVE SIGNAL GENERATED")

if __name__ == "__main__":
    main()
