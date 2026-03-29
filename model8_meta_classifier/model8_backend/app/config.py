from pathlib import Path

# Go up to project root (bitcoin_forecast)
BASE_DIR = Path(__file__).resolve().parents[3]

MODEL8_DIR = BASE_DIR / "model8_meta_classifier"

DATASET_PATH = MODEL8_DIR / "data" / "model8_dataset.csv"
FINAL_SIGNAL_PATH = MODEL8_DIR / "results" / "model8_final_signal.csv"

PREDICTIONS_PATH = MODEL8_DIR / "model8_backend" / "data" / "model8_predictions.csv"
VOL_PATH = MODEL8_DIR / "model8_backend" / "data" / "model2_final_volatility.csv"
MACRO_PATH = MODEL8_DIR / "model8_backend" / "data" / "model6_final_signal.csv"

MODEL_DIR = BASE_DIR / "model8_meta_classifier" / "model8_backend" / "models"
MODEL_8A_PATH = MODEL_DIR / "model8A_regime_selector.pkl"
MODEL_8B_PATH = MODEL_DIR / "model8B_directional.pkl"

DEFAULT_THRESHOLD = 0.65
