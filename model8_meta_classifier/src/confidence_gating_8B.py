import pandas as pd
import joblib
import numpy as np

DATA = "model8_meta_classifier/data/model8B_dataset.csv"
MODEL = "model8_meta_classifier/models/model8B_directional.pkl"

def main():
    print("📊 Confidence Gating — Model 8B")

    df = pd.read_csv(DATA)
    model = joblib.load(MODEL)

    X = df[
        [
            "signal_1",
            "signal_2",
            "signal_3",
            "signal_4",
            "signal_6",
            "signal_1_missing",
            "signal_2_missing",
            "signal_3_missing",
            "signal_4_missing",
            "signal_6_missing",
        ]
    ]

    probs = model.predict_proba(X)
    confidence = np.max(probs, axis=1)

    df["pred"] = model.predict(X)
    df["confidence"] = confidence

    for tau in [0.55, 0.60, 0.65, 0.70]:
        filtered = df[df["confidence"] >= tau]
        if len(filtered) < 30:
            continue

        da = (filtered["pred"] == filtered["direction"]).mean()
        coverage = len(filtered) / len(df)

        print(
            f"τ={tau:.2f} | coverage={coverage:.2%} | DA={da:.3f}"
        )

if __name__ == "__main__":
    main()
