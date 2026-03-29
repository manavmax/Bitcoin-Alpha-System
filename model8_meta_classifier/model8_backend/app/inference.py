import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_PATH / "results" / "model8_final_signal.csv"

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(str(DATA_PATH))
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date")

def latest_state(threshold=0.6):
    df = load_data()
    latest = df.iloc[-1]

    return {
        "date": str(latest["date"].date()),
        "signal": latest["final_signal"],
        "confidence": float(latest["confidence"]),
        "tradable": bool(latest["tradable"]),
        "vol_regime": int(latest["vol_regime"]),
        "macro_regime": "Risk-On" if int(latest["macro_regime"]) == 1 else "Risk-Off",
    }

def history():
    df = load_data()
    return df.tail(365).to_dict(orient="records")
