import pandas as pd
import joblib
import numpy as np

df = pd.read_csv("model8_meta_classifier/data/model8_dataset.csv")
model = joblib.load("model8_meta_classifier/models/model8_xgb.pkl")

features = [c for c in df.columns if c.startswith("signal_")]
probs = model.predict_proba(df[features])

df["pred_class"] = (probs[:, 1] > 0.5).astype(int)
df["confidence"] = np.max(probs, axis=1)

df.to_csv(
    "model8_meta_classifier/results/model8_predictions.csv",
    index=False
)

print("✅ Predictions saved")
