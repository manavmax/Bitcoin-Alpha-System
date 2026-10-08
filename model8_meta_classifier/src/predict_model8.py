import sys
from pathlib import Path

import pandas as pd
import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.prediction_store import publish_append_only  # noqa: E402

df = pd.read_csv("model8_meta_classifier/data/model8_dataset.csv")
model = joblib.load("model8_meta_classifier/models/model8_xgb.pkl")

features = [c for c in df.columns if c.startswith("signal_")]
probs = model.predict_proba(df[features])

df["pred_class"] = (probs[:, 1] > 0.5).astype(int)
df["confidence"] = np.max(probs, axis=1)

# Append-only publication: previously published predictions are immutable.
# Only new dates are appended; `future_ret` (ground truth, unknown on
# publication day) is backfilled once the next candle closes.
publish_append_only(
    df,
    "model8_meta_classifier/results/model8_predictions.csv",
    backfill_cols=("future_ret",),
    refresh_env="MODEL8_XGB_REFRESH_DATES",
)

print("✅ Predictions saved")
