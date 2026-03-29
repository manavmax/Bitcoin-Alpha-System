import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("model_4_onchain_fundamentals/raw/blockchain_com")
RAW_CM = Path("model_4_onchain_fundamentals/raw/coinmetrics/coinmetrics_onchain_daily.csv")
OUT_FILE = Path("model_4_onchain_fundamentals/processed/model4_onchain_features.csv")

# -----------------------------
# Helper: load blockchain.com CSV
# -----------------------------
def load_metric(path: Path, value_col: str):
    df = pd.read_csv(path)

    # Expect columns: date, <metric>
    if "date" not in df.columns or value_col not in df.columns:
        raise RuntimeError(
            f"❌ Invalid format in {path.name}. Columns: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    return df[["date", value_col]]


# -----------------------------
# Load all metrics
# -----------------------------
hash_rate = load_metric(RAW_DIR / "hash_rate.csv", "hash_rate")
difficulty = load_metric(RAW_DIR / "difficulty.csv", "difficulty")
tx_count = load_metric(RAW_DIR / "tx_count.csv", "tx_count")
tx_volume = load_metric(RAW_DIR / "tx_volume_usd.csv", "tx_volume_usd")
miner_rev = load_metric(RAW_DIR / "miner_revenue.csv", "miner_revenue")

# -----------------------------
# Merge (OUTER JOIN — critical)
# -----------------------------
df = hash_rate.merge(difficulty, on="date", how="outer")
df = df.merge(tx_count, on="date", how="outer")
df = df.merge(tx_volume, on="date", how="outer")
df = df.merge(miner_rev, on="date", how="outer")

df = df.sort_values("date").reset_index(drop=True)

# -----------------------------
# Fill gaps using CoinMetrics (real provider data)
# -----------------------------
if RAW_CM.exists():
    cm = pd.read_csv(RAW_CM, parse_dates=["date"])
    cm["date"] = pd.to_datetime(cm["date"])
    cm = cm.sort_values("date")
    df = df.merge(cm, on="date", how="left", suffixes=("", "_cm"))

    for col in ["hash_rate", "difficulty", "tx_count", "tx_volume_usd", "miner_revenue"]:
        cm_col = f"{col}_cm"
        if cm_col in df.columns:
            df[col] = df[col].combine_first(df[cm_col])

    drop_cols = [c for c in df.columns if c.endswith("_cm")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

# -----------------------------
# Feature engineering
# -----------------------------
base_cols = [
    "hash_rate",
    "difficulty",
    "tx_count",
    "tx_volume_usd",
    "miner_revenue",
]

for col in base_cols:
    if col not in df.columns:
        continue

    df[f"{col}_7d_pct"] = df[col].pct_change(7, fill_method=None)
    df[f"{col}_30d_pct"] = df[col].pct_change(30, fill_method=None)

    rolling_mean = df[col].rolling(90, min_periods=30).mean()
    rolling_std = df[col].rolling(90, min_periods=30).std()

    df[f"{col}_z"] = (df[col] - rolling_mean) / rolling_std


# -----------------------------
# Miner stress
# -----------------------------
df["miner_rev_ma30"] = df["miner_revenue"].rolling(30, min_periods=10).mean()
df["miner_stress_flag"] = (
    (df["miner_revenue"] < df["miner_rev_ma30"]) &
    (df["difficulty_7d_pct"] > 0)
).astype(int)


# -----------------------------
# Network Activity Index (CRITICAL FIX)
# -----------------------------
z_cols = [
    "hash_rate_z",
    "difficulty_z",
    "tx_count_z",
    "tx_volume_usd_z",
    "miner_revenue_z",
]

df["network_activity_index"] = df[z_cols].mean(axis=1, skipna=True)

# -----------------------------
# On-chain regime classification
# -----------------------------
valid = df["network_activity_index"].dropna()

low_q = valid.quantile(0.25)
high_q = valid.quantile(0.75)

df["onchain_regime"] = 1
df.loc[df["network_activity_index"] < low_q, "onchain_regime"] = 0
df.loc[df["network_activity_index"] > high_q, "onchain_regime"] = 2

# -----------------------------
# Final cleanup
# -----------------------------
df = df.dropna(subset=["network_activity_index"])

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_FILE, index=False)

print("✅ Model 4 on-chain features created (FINAL)")
print(f"Saved → {OUT_FILE}")
print("\n📊 Regime distribution:")
print(df["onchain_regime"].value_counts(normalize=True))
