import pandas as pd

DATA_8A = "model8_meta_classifier/data/model8A_dataset.csv"
DATA_8 = "model8_meta_classifier/data/model8_dataset.csv"
OUT = "model8_meta_classifier/data/model8B_dataset.csv"

def main():
    print("📥 Building Model 8B dataset (TRADABLE ONLY)...")

    reg = pd.read_csv(DATA_8A, parse_dates=["date"])
    base = pd.read_csv(DATA_8, parse_dates=["date"])

    # Keep only tradable regimes
    tradable_dates = reg.loc[reg["tradable"] == 1, "date"]

    df = base[base["date"].isin(tradable_dates)].copy()

    # Drop samples without labels (latest day has no future return yet)
    df = df.dropna(subset=["future_ret"]).copy()

    # Binary directional label
    df["direction"] = (df["future_ret"] > 0).astype(int)

    df.to_csv(OUT, index=False)

    print("✅ Model 8B dataset ready")
    print("Samples:", len(df))
    print(df["direction"].value_counts(normalize=True))

if __name__ == "__main__":
    main()
