import pandas as pd
import joblib

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.prediction_store import publish_append_only  # noqa: E402

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

# Append-only publication: previously published predictions are immutable.
publish_append_only(df, OUT_FILE, refresh_env="MODEL7_3_REFRESH_DATES")
print(f"✅ Model 7.3 inference published → {OUT_FILE}")
