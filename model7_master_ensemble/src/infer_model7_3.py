import pandas as pd
import joblib

MODEL_FILE = "model7_master_ensemble/models/model7_3_decision.pkl"
OUT_FILE = "model7_master_ensemble/results/model7_3_decisions.csv"

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
encoder = bundle["encoder"]
FEATURES = bundle["features"]

df = pd.read_csv(
    "model7_master_ensemble/results/model7_final_signal.csv",
    parse_dates=["date"]
)

for f in FEATURES:
    if f not in df:
        df[f] = 0.0

proba = model.predict_proba(df[FEATURES])
pred = encoder.inverse_transform(proba.argmax(axis=1))

df["P_SELL"] = proba[:, 0]
df["P_NEUTRAL"] = proba[:, 1]
df["P_BUY"] = proba[:, 2]
df["decision"] = pred

df.to_csv(OUT_FILE, index=False)
print(f"✅ Model 7.3 inference saved → {OUT_FILE}")
