# model_6_macro_liquidity/src/prepare_model6_features.py

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("model_6_macro_liquidity/raw/fred")
OUT = Path("model_6_macro_liquidity/processed")
OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Safe loader with renaming
# -----------------------------
def load_series(filename, colname):
    df = pd.read_csv(RAW / f"{filename}.csv", parse_dates=["date"])
    df = df.rename(columns={"value": colname})
    return df.set_index("date")

# -----------------------------
# Load all macro series
# -----------------------------
fed = load_series("fed_funds_rate", "DFF")
m2 = load_series("m2_money_supply", "M2SL")
cpi = load_series("cpi", "CPIAUCSL")
yc = load_series("yield_curve", "T10Y2Y")
vix = load_series("vix", "VIXCLS")
sp = load_series("sp500", "SP500")

# -----------------------------
# Merge safely (NO collisions)
# -----------------------------
df = pd.concat(
    [fed, m2, cpi, yc, vix, sp],
    axis=1
).sort_index()

# Build a true daily calendar (crypto trades 24/7; macro releases are sparse)
daily_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
df = df.reindex(daily_index)

# Forward-fill macro data (hold last published value until next release)
df = df.ffill()

# -----------------------------
# Feature Engineering
# -----------------------------

# Liquidity growth (YoY)
df["m2_yoy"] = df["M2SL"].pct_change(12) * 100

# Inflation YoY
df["cpi_yoy"] = df["CPIAUCSL"].pct_change(12) * 100

# Real interest rate
df["real_rate"] = df["DFF"] - df["cpi_yoy"]

# Yield curve inversion
df["yield_inversion"] = (df["T10Y2Y"] < 0).astype(int)

# VIX stress (1Y rolling z-score)
vix_mean = df["VIXCLS"].rolling(252).mean()
vix_std = df["VIXCLS"].rolling(252).std()
df["vix_z"] = (df["VIXCLS"] - vix_mean) / vix_std

# Equity trend
df["sp_ma50"] = df["SP500"].rolling(50).mean()
df["sp_ma200"] = df["SP500"].rolling(200).mean()
df["sp_trend"] = df["sp_ma50"] - df["sp_ma200"]

# -----------------------------
# Macro Regime Assignment
# -----------------------------
def assign_regime(r):
    if (
        r["real_rate"] > 0
        and r["yield_inversion"] == 1
        and r["vix_z"] > 1
    ):
        return 0  # RISK OFF
    elif (
        r["m2_yoy"] > 0
        and r["sp_trend"] > 0
    ):
        return 2  # RISK ON
    else:
        return 1  # NEUTRAL

df["macro_regime"] = df.apply(assign_regime, axis=1)

# -----------------------------
# Final dataset
# -----------------------------
final_cols = [
    "DFF",
    "m2_yoy",
    "real_rate",
    "yield_inversion",
    "vix_z",
    "sp_trend",
    "macro_regime",
]

df_out = df[final_cols].dropna()

df_out.reset_index().rename(columns={"index": "date"}).to_csv(
    OUT / "model6_macro_features.csv",
    index=False,
)

print("✅ Model 6 macro features created")
print("\n📊 Regime distribution:")
print(df_out["macro_regime"].value_counts(normalize=True))
