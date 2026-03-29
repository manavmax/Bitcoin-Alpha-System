import requests
import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_URL = "https://api.blockchain.info/charts"
OUTPUT_DIR = Path("model_4_onchain_fundamentals/raw/blockchain_com")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {
    "hash_rate": "hash-rate",
    "difficulty": "difficulty",
    "tx_count": "n-transactions",
    "tx_volume_usd": "estimated-transaction-volume-usd",
    "miner_revenue": "miners-revenue"
}

# ---------------- FETCH FUNCTION ----------------
def fetch_metric(name, endpoint):
    print(f"📥 Fetching {name}...")

    url = f"{BASE_URL}/{endpoint}"
    params = {
        "timespan": "all",          # full history
        "sampled": "false",         # disable downsampling → every daily datapoint (no skipped days)
        "format": "json"
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    values = r.json()["values"]

    df = pd.DataFrame(values)
    df["date"] = pd.to_datetime(df["x"], unit="s", utc=True).dt.date
    df = df[["date", "y"]].rename(columns={"y": name})

    out_file = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(out_file, index=False)

    print(f"✅ Saved {name} → {out_file} | rows={len(df)}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    for name, endpoint in METRICS.items():
        fetch_metric(name, endpoint)

    print("\n🎯 Blockchain.com daily on-chain data ingestion completed")
