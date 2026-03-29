import pandas as pd
import numpy as np

df = pd.read_csv(
    "model7_master_ensemble/results/model7_3_decisions.csv",
    parse_dates=["date"]
)

price = pd.read_csv(
    "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"
)

price["date"] = pd.to_datetime(price.iloc[:, 0], utc=True).dt.tz_convert(None)
price["Close"] = price.iloc[:, 4]

df = df.merge(price[["date", "Close"]], on="date", how="inner")

df["future_return"] = df["Close"].shift(-3) / df["Close"] - 1

active = df[df["decision"] != 0]

direction_correct = (
    ((active["decision"] == 1) & (active["future_return"] > 0)) |
    ((active["decision"] == -1) & (active["future_return"] < 0))
)

DA = direction_correct.mean()
trade_rate = len(active) / len(df)

print("\n📊 MODEL 7.3 — FINAL EVALUATION (WITH MODEL 2)")
print(f"Trades taken: {trade_rate:.2%}")
print(f"Directional Accuracy: {DA:.4f}")
