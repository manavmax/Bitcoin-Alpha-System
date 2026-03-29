import pandas as pd

ONCHAIN = "model_4_onchain_fundamentals/processed/model4_onchain_features.csv"
PRICE = "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"

on = pd.read_csv(ONCHAIN)
px = pd.read_csv(PRICE)

# Normalize dates
on["date"] = pd.to_datetime(on["date"], errors="coerce").dt.date
px["date"] = pd.to_datetime(px["Open time"], errors="coerce").dt.date

print("\n--- ON-CHAIN ---")
print(on["date"].min(), "→", on["date"].max(), "| rows:", len(on))

print("\n--- PRICE ---")
print(px["date"].min(), "→", px["date"].max(), "| rows:", len(px))

# Intersection test
common = set(on["date"]).intersection(set(px["date"]))
print("\n--- INTERSECTION ---")
print("Common dates:", len(common))

if len(common) > 0:
    print("Sample common dates:", list(common)[:5])
