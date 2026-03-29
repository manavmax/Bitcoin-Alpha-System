import pandas as pd
from pathlib import Path

FILE = Path("model_4_onchain_fundamentals/processed/model4_onchain_features.csv")

df = pd.read_csv(FILE)

print("\n📊 MODEL 4 — ON-CHAIN REGIME DISTRIBUTION\n")
print(df["onchain_regime"].value_counts(normalize=True))

print("\n📈 Average network activity per regime\n")
print(df.groupby("onchain_regime")["network_activity_index"].mean())

print("\n⛏️ Miner stress frequency\n")
print(df.groupby("onchain_regime")["miner_stress_flag"].mean())
