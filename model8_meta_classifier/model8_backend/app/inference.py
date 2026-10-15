import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_PATH / "results" / "model8_final_signal.csv"

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(str(DATA_PATH))
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date")

def latest_state(threshold=0.6, require_tradable=True):
    df = load_data()
    latest = df.iloc[-1]

    d = str(latest.get("direction", "") or "").upper()
    if d not in ("LONG", "SHORT"):
        f = str(latest.get("final_signal", "") or "").upper()
        d = f if f in ("LONG", "SHORT") else ""

    conf = float(latest["confidence"])
    tradable = bool(latest["tradable"])
    if not d:
        reason = "no directional call"
        eff = "NO_TRADE"
    elif not (conf >= threshold):
        reason = f"confidence {conf:.2f} < τ {threshold:.2f}"
        eff = "NO_TRADE"
    elif require_tradable and not tradable:
        reason = "blocked by 8A (tradable=0)"
        eff = "NO_TRADE"
    else:
        reason = "active"
        eff = d

    return {
        "date": str(latest["date"].date()),
        "signal": eff,
        "frozen_signal": str(latest["final_signal"]),
        "direction": d,
        "reason": reason,
        "confidence": conf,
        "tradable": tradable,
        "vol_regime": int(latest["vol_regime"]),
        "macro_regime": "Risk-On" if int(latest["macro_regime"]) == 1 else "Risk-Off",
    }

def history():
    df = load_data()
    return df.tail(365).to_dict(orient="records")
