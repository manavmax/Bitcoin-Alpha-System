import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA = "model8_meta_classifier/data/model8B_dataset.csv"
MODEL_OUT = "model8_meta_classifier/models/model8B_directional.pkl"

def main():
    print("🚀 Training Model 8B — Directional Alpha")

    df = pd.read_csv(DATA)

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

    y = df["direction"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    print("\n📊 MODEL 8B — VALIDATION REPORT\n")
    print(classification_report(y_val, preds))
    print("Accuracy:", accuracy_score(y_val, preds))

    joblib.dump(model, MODEL_OUT)
    print(f"\n✅ Model 8B saved → {MODEL_OUT}")

if __name__ == "__main__":
    main()
