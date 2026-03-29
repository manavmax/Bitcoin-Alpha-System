import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import os

DATA_PATH = "model8_meta_classifier/data/model8A_dataset.csv"
MODEL_PATH = "model8_meta_classifier/models/model8A_regime_selector.pkl"

def main():
    print("🚀 Training Model 8A — Regime Selector")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date")

    FEATURES = ["vol_regime", "macro_regime", "macro_signal"]
    TARGET = "tradable"

    X = df[FEATURES]
    y = df[TARGET]

    # --------------------------------------------------
    # Time-aware split (VERY IMPORTANT)
    # --------------------------------------------------
    split = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    # --------------------------------------------------
    # Model (conservative, stable)
    # --------------------------------------------------
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    print("\n📊 MODEL 8A — VALIDATION REPORT\n")
    print(classification_report(y_val, preds, digits=4))

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"✅ Model 8A saved → {MODEL_PATH}")

if __name__ == "__main__":
    main()
