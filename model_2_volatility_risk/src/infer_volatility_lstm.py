import numpy as np
import pandas as pd
import torch
from pathlib import Path

from volatility_lstm import VolatilityLSTM


BASE = Path(__file__).resolve().parents[1]

DATA_FILE = BASE / "processed" / "model2_volatility_features.csv"
MODEL_FILE = BASE / "models" / "volatility_lstm.pth"
OUT_FILE = BASE / "results" / "volatility_lstm_preds.csv"


FEATURES = [
    "garch_vol",
    "atr",
    "vov",
    "downside_vol",
    "vol_momentum",
    "vol_rank",
    "vol_regime",
    "crash_flag",
]

SEQ_LEN = 30


def main():
    df = pd.read_csv(DATA_FILE)
    if "date" not in df.columns:
        raise RuntimeError("model2_volatility_features.csv missing 'date' column")

    X = df[FEATURES].values.astype(np.float32)

    device = "cpu"
    model = VolatilityLSTM(input_size=len(FEATURES)).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    preds = np.full(len(df), np.nan, dtype=np.float32)

    with torch.no_grad():
        for i in range(SEQ_LEN, len(X)):
            seq = torch.tensor(X[i - SEQ_LEN : i], dtype=torch.float32).unsqueeze(0).to(device)
            preds[i] = float(model(seq).cpu().item())

    out = df.copy()
    out["lstm_volatility"] = preds

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print("✅ Volatility LSTM inference completed")
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()

