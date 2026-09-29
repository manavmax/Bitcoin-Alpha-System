import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

# ===============================
# CONFIG
# ===============================
PRICE_FILE = "model_1_price_dynamics/data_raw/btc_1d_2018_2025.csv"

SIGNALS = {
    "signal_1": "model_1_price_dynamics/results/model1_final_predictions.csv",
    "signal_2": "model_2_volatility_risk/results/model2_final_volatility.csv",
    "signal_3": "model_3_derivatives_flow/results/model3_ensemble_predictions.csv",
    "signal_4": "model_4_onchain_fundamentals/results/model4_final_signal.csv",
    "signal_6": "model_6_macro_liquidity/results/model6_final_signal.csv",
}

OUT_FILE = "model8_meta_classifier/data/model8_dataset.csv"
KEEP_LIVE_LAST_ROW = True  # keep latest row even if future_ret is NaN (needed for live inference)

# Which candles make it onto the master timeline:
#   "closed" (default) -> strictly candles that have already closed
#                         (date < today UTC). A row dated D is only produced
#                         once day D has ended, so every published prediction
#                         is computed from a complete candle. On day T the
#                         newest row is T-1 (actionable on T); today's row
#                         appears only at the start of tomorrow.
#   "latest"           -> also includes the current, still-forming UTC day.
#                         DEBUGGING ONLY - never publish with this: the row
#                         mutates intraday, base models 4/6 publish closed
#                         candles only (so signal_4/signal_6 go missing on
#                         that row), and the premature signal can flip.
# Override at runtime with e.g.  MODEL8_CUTOFF=latest python run_daily_pipeline.py
CUTOFF_MODE = os.getenv("MODEL8_CUTOFF", "closed").strip().lower()

# ===============================
# DATE NORMALIZATION (CRITICAL)
# ===============================
def normalize_date(df):
    for c in ["date", "Date", "timestamp", "time", "Open time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = df[c].dt.tz_convert(None)  # 🔥 FORCE timezone-naive
            return df.rename(columns={c: "date"})
    return None

# ===============================
# LOAD BTC MASTER TIMELINE
# ===============================
def load_price():
    price = pd.read_csv(PRICE_FILE)
    price = normalize_date(price)
    if price is None:
        raise RuntimeError("❌ BTC price file has no usable date column")

    price = price.sort_values("date").reset_index(drop=True)

    # IMPORTANT: only fully closed candles are used by default - a prediction
    # may only be produced for a day once that day has ended (candle closed at
    # 00:00 UTC). Set MODEL8_CUTOFF=latest to include today's in-progress
    # candle (debugging only - produces a premature, mutating signal).
    today_utc = datetime.now(timezone.utc).date()
    if CUTOFF_MODE == "closed":
        price = price[price["date"].dt.date < today_utc].reset_index(drop=True)
    else:
        price = price[price["date"].dt.date <= today_utc].reset_index(drop=True)
        print(f"⚠️ MODEL8_CUTOFF=latest - including today's forming candle "
              f"({today_utc}); do NOT use for published signals")
    price["future_ret"] = price["Close"].pct_change().shift(-1)
    return price[["date", "future_ret"]]

# ===============================
# LOAD SIGNAL (WITH OR WITHOUT DATE)
# ===============================
def load_signal(path, name, master_dates):
    df = pd.read_csv(path)

    # Try date-based merge first
    dated = normalize_date(df)

    if dated is not None:
        # Auto-detect numeric signal column
        candidates = dated.select_dtypes(include=[np.number]).columns.tolist()
        candidates = [c for c in candidates if c not in ["future_ret", "target_return"]]

        if len(candidates) == 0:
            raise RuntimeError(f"❌ No numeric signal in {path}")

        col = candidates[0]
        dated = dated[["date", col]].rename(columns={col: name})
        return dated

    # Fallback: NO DATE → ALIGN FROM END
    print(f"⚠️ {path} has no date — aligning from end")
    sig = df.select_dtypes(include=[np.number]).iloc[:, 0].values

    out = pd.DataFrame({"date": master_dates})
    out[name] = np.nan
    out.loc[len(out) - len(sig):, name] = sig
    return out

# ===============================
# MAIN BUILD
# ===============================
def main():
    print("📥 Building Model 8 dataset (FULL SAMPLE MODE)...")

    price = load_price()
    base = price.copy()
    master_dates = base["date"].values

    for name, path in SIGNALS.items():
        print(f"🔗 Loading {name}...")
        sig = load_signal(path, name, master_dates)

        # Normalize date again (safety)
        sig["date"] = pd.to_datetime(sig["date"])
        base["date"] = pd.to_datetime(base["date"])

        base = base.merge(sig, on="date", how="left")

        # Missing flag (VERY IMPORTANT FEATURE)
        base[f"{name}_missing"] = base[name].isna().astype(int)

        # Fill missing signal with 0 (neutral)
        base[name] = base[name].fillna(0.0)

    # --------------------------------------------------------------
    # Coverage diagnostics - flag base signals whose source has gone
    # stale relative to the master timeline. Recent rows for a stale
    # signal are zero-filled with *_missing = 1, which degrades the
    # Model 8B inputs, so surface the gap loudly at build time.
    # --------------------------------------------------------------
    master_max = base["date"].max().date()
    for name, path in SIGNALS.items():
        flag = f"{name}_missing"
        with_data = base.loc[base[flag] == 0, "date"]
        if with_data.empty:
            print(f"⚠️ {name}: no usable rows loaded from {path}")
            continue
        last_ok = with_data.max().date()
        if last_ok < master_max:
            stale_days = (master_max - last_ok).days
            hint = (" - set COINALYZE_API_KEY (.env) and re-run to refresh "
                    "Model 3" if name == "signal_3" else "")
            print(f"⚠️ {name}: source STALE - last data {last_ok}, "
                  f"{stale_days} day(s) behind {master_max}; recent rows "
                  f"zero-filled with {flag}=1{hint}")

    # Keep the final row for live inference (future_ret is unknown for the most recent day).
    # Training scripts should explicitly drop NaN labels when needed.
    if not KEEP_LIVE_LAST_ROW:
        base = base.dropna(subset=["future_ret"])
    base = base.reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    base.to_csv(OUT_FILE, index=False)

    print("✅ Model 8 dataset built SUCCESSFULLY")
    print(f"Saved → {OUT_FILE}")
    print(f"Samples: {len(base)}")
    print("Columns:", base.columns.tolist())

# ===============================
if __name__ == "__main__":
    main()
