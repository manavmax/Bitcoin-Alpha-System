import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

COINALYZE_RAW = BASE_DIR / "raw" / "coinalyze"
BINANCE_LS = BASE_DIR / "raw" / "binance" / "binance_long_short_ratio_1d.csv"
CM_DERIV = BASE_DIR / "raw" / "coinmetrics" / "coinmetrics_derivatives_daily.csv"

OUT_DIR = BASE_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "model3_daily_features.csv"


def _load_coinalyze(name: str) -> pd.DataFrame:
    df = pd.read_csv(COINALYZE_RAW / name)
    df["date"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.date
    return df


def build_from_coinalyze() -> pd.DataFrame:
    liq = _load_coinalyze("liquidations.csv")
    oi = _load_coinalyze("open_interest.csv")
    fr = _load_coinalyze("funding_rate.csv")
    ls = _load_coinalyze("long_short_ratio.csv")
    ohl = _load_coinalyze("ohlcv.csv")

    liq_d = liq.groupby("date").agg(long_liq=("l", "sum"), short_liq=("s", "sum")).reset_index()
    liq_d["liq_imbalance"] = liq_d["long_liq"] - liq_d["short_liq"]

    oi_d = oi.groupby("date").agg(open_interest=("c", "mean")).reset_index()
    fr_d = fr.groupby("date").agg(funding_rate=("c", "mean")).reset_index()

    if {"long", "short"}.issubset(ls.columns):
        ls["ls_ratio"] = ls["long"] / ls["short"]
    elif {"l", "s"}.issubset(ls.columns):
        ls["ls_ratio"] = ls["l"] / ls["s"]
    elif {"r"}.issubset(ls.columns):
        ls["ls_ratio"] = ls["r"]
    else:
        raise RuntimeError(f"Unknown long/short schema: {ls.columns}")

    ls_d = ls.groupby("date").agg(long_short_ratio=("ls_ratio", "mean")).reset_index()

    ohl_d = ohl.groupby("date").agg(volume=("v", "sum"), trade_count=("tx", "sum")).reset_index()

    # Outer merges to keep all days (we'll fill gaps via other providers)
    df = liq_d.merge(oi_d, on="date", how="outer")
    df = df.merge(fr_d, on="date", how="outer")
    df = df.merge(ls_d, on="date", how="outer")
    df = df.merge(ohl_d, on="date", how="outer")

    return df.sort_values("date")


def apply_coinmetrics_fallback(df: pd.DataFrame) -> pd.DataFrame:
    if not CM_DERIV.exists():
        return df
    cm = pd.read_csv(CM_DERIV)
    cm["date"] = pd.to_datetime(cm["date"]).dt.date
    df = df.merge(cm, on="date", how="left", suffixes=("", "_cm"))

    for col in ["open_interest", "funding_rate", "volume", "trade_count", "long_liq", "short_liq"]:
        cm_col = f"{col}_cm"
        if cm_col in df.columns:
            df[col] = df[col].combine_first(df[cm_col])

    drop_cols = [c for c in df.columns if c.endswith("_cm")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if {"long_liq", "short_liq"}.issubset(df.columns) and "liq_imbalance" in df.columns:
        df["liq_imbalance"] = df["liq_imbalance"].combine_first(df["long_liq"] - df["short_liq"])

    return df


def apply_binance_long_short_fallback(df: pd.DataFrame) -> pd.DataFrame:
    if not BINANCE_LS.exists():
        return df
    b = pd.read_csv(BINANCE_LS)
    if b.empty:
        return df
    b["date"] = pd.to_datetime(b["date"]).dt.date
    df = df.merge(b, on="date", how="left", suffixes=("", "_bn"))
    if "long_short_ratio_bn" in df.columns:
        df["long_short_ratio"] = df["long_short_ratio"].combine_first(df["long_short_ratio_bn"])
        df = df.drop(columns=["long_short_ratio_bn"])
    return df


def main():
    df = build_from_coinalyze()
    df = apply_coinmetrics_fallback(df)
    df = apply_binance_long_short_fallback(df)

    # Final sorting and save; we do NOT forward-fill: if providers don't have it, it's NaN.
    df = df.sort_values("date")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print("✅ Model 3 daily feature table created (merged providers)")
    print(f"Saved → {OUT_FILE} | rows={len(df)}")
    print(f"Range: {df['date'].min()} → {df['date'].max()}")


if __name__ == "__main__":
    main()

