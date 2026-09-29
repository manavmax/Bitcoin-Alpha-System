import pandas as pd
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "data" / "raw" / "btc_price_daily.csv"

TARGETS = [
    ROOT / "model_1_price_dynamics" / "data_raw" / "btc_1d_2018_2025.csv",
    ROOT / "model_2_volatility_risk" / "data" / "btc_daily.csv",
]


def _canonicalize_new(df_new: pd.DataFrame) -> pd.DataFrame:
    df = df_new.copy()
    df["open_dt"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["open_dt"]).sort_values("open_dt")
    return df


def _canonicalize_old(df_old: pd.DataFrame) -> pd.DataFrame:
    df = df_old.copy()
    if "Open time" not in df.columns:
        raise RuntimeError("Target file missing 'Open time' column")
    df["open_dt"] = pd.to_datetime(df["Open time"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["open_dt"]).sort_values("open_dt")
    return df


def _format_open_time(dt_series: pd.Series) -> pd.Series:
    # Match existing files like: "2018-01-01 00:00:00.000000 UTC"
    return dt_series.dt.strftime("%Y-%m-%d %H:%M:%S.%f UTC")


def sync_one(target_path: Path, df_new: pd.DataFrame) -> None:
    if target_path.exists():
        df_old = pd.read_csv(target_path)
        df_old = _canonicalize_old(df_old)
        cols = [c for c in df_old.columns if c != "open_dt"]
    else:
        cols = [
            "Open time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
        df_old = pd.DataFrame(columns=cols + ["open_dt"])

    # Build new rows with the target schema
    df_append = pd.DataFrame({c: np.nan for c in cols}, index=range(len(df_new)))
    df_append["Open time"] = _format_open_time(df_new["open_dt"])

    # Map core OHLCV columns (case-insensitive safety)
    df_append["Open"] = pd.to_numeric(df_new["open"], errors="coerce")
    df_append["High"] = pd.to_numeric(df_new["high"], errors="coerce")
    df_append["Low"] = pd.to_numeric(df_new["low"], errors="coerce")
    df_append["Close"] = pd.to_numeric(df_new["close"], errors="coerce")
    df_append["Volume"] = pd.to_numeric(df_new["volume"], errors="coerce")

    df_append["open_dt"] = df_new["open_dt"].values

    # Carry the auxiliary kline fields through as well. Model 1's LSTM/TCN/N-BEATS
    # were trained with these as inputs; leaving them NaN truncated the feature
    # history at 2023-06-15 and broke inference.
    AUX_MAP = {
        "close_time": "Close time",
        "quote_asset_volume": "Quote asset volume",
        "num_trades": "Number of trades",
        "taker_buy_base": "Taker buy base asset volume",
        "taker_buy_quote": "Taker buy quote asset volume",
        "ignore": "Ignore",
    }
    for src_col, tgt_col in AUX_MAP.items():
        if tgt_col not in df_append.columns or src_col not in df_new.columns:
            continue
        if src_col == "close_time":
            ct = pd.to_datetime(df_new[src_col], utc=True, errors="coerce")
            df_append[tgt_col] = ct.dt.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
        else:
            df_append[tgt_col] = pd.to_numeric(df_new[src_col], errors="coerce")

    merged = pd.concat(
        [df_old[cols + ["open_dt"]], df_append[cols + ["open_dt"]]],
        ignore_index=True,
    )
    # Newly fetched rows must win for any date they share with the existing file.
    # Sort stably (existing rows first, new rows last) before de-duplicating —
    # the default quicksort is not stable, so keep="last" could otherwise keep the
    # stale row (which is how ~600 rows ended up with NaN auxiliary columns).
    merged = merged.sort_values("open_dt", kind="stable")
    merged = merged.drop_duplicates(subset=["open_dt"], keep="last")
    merged = merged.drop(columns=["open_dt"])

    target_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(target_path, index=False)

    print(f"✅ Synced BTC daily into {target_path} | rows={len(merged)}")
    print(f"   last_date={merged['Open time'].iloc[-1]}")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(str(SRC))

    df_new = pd.read_csv(SRC)
    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = required - set(df_new.columns)
    if missing:
        raise RuntimeError(f"btc_price_daily.csv missing columns: {sorted(missing)}")

    df_new = _canonicalize_new(df_new)

    for tgt in TARGETS:
        sync_one(tgt, df_new)


if __name__ == "__main__":
    main()

