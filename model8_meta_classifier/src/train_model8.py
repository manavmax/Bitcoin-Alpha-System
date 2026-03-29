import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("model8_meta_classifier/data/model8_dataset.csv")

# Label: direction only when meaningful
df["target"] = np.where(df["future_ret"] > 0, 1, 0)

features = [c for c in df.columns if c.startswith("signal_")]
X = df[features]
y = df["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

model.fit(X_train, y_train)

pred = model.predict(X_val)

print("\n📊 MODEL 8 VALIDATION REPORT\n")
print(classification_report(y_val, pred))

joblib.dump(model, "model8_meta_classifier/models/model8_xgb.pkl")
print("✅ Model 8 trained & saved")
