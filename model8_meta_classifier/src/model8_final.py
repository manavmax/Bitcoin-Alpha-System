import os
from datetime import datetime, timezone

import pandas as pd
import joblib
import numpy as np

TAU = 0.60  # confidence threshold

# Append-only publication target. Rows already written here are immutable:
# reruns never recompute or rewrite previously published predictions - each
# run only appends dates that have newly closed.
OUT_CSV = "model8_meta_classifier/results/model8_final_signal.csv"
OUT_COLUMNS = ["date", "final_signal", "confidence",
               "tradable", "vol_regime", "macro_regime"]

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def safe_date(df):
    for c in ["date", "Date", "timestamp", "time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True)
            df.rename(columns={c: "date"}, inplace=True)
            return df
    raise RuntimeError("❌ No date column found")

def detect_macro_signal(df):
    candidates = [
        c for c in df.columns
        if "macro" in c.lower() or "liquidity" in c.lower()
    ]
    if not candidates:
        raise RuntimeError("❌ No macro signal column found in Model 6 output")
    return candidates[0]

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("🚀 Running FINAL Model 8 (8A + 8B)")

    # --------------------------------------------------
    # Load base dataset
    # --------------------------------------------------
    df = pd.read_csv(
        "model8_meta_classifier/data/model8_dataset.csv",
        parse_dates=["date"]
    )
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # --------------------------------------------------
    # Load volatility regimes (Model 2)
    # --------------------------------------------------
    vol = pd.read_csv(
        "model_2_volatility_risk/results/model2_final_volatility.csv"
    )
    vol = safe_date(vol)

    if "vol_regime" not in vol.columns:
        raise RuntimeError("❌ vol_regime missing in Model 2 output")

    vol = vol[["date", "vol_regime"]]

    # --------------------------------------------------
    # Load macro signal (Model 6)
    # --------------------------------------------------
    macro = pd.read_csv(
        "model_6_macro_liquidity/results/model6_final_signal.csv"
    )
    macro = safe_date(macro)

    macro_signal_col = detect_macro_signal(macro)
    print(f"📌 Using macro signal column: {macro_signal_col}")

    macro = macro[["date", macro_signal_col]].rename(
        columns={macro_signal_col: "macro_signal"}
    )

    # --------------------------------------------------
    # Merge regimes
    # --------------------------------------------------
    df = df.merge(vol, on="date", how="left")
    df = df.merge(macro, on="date", how="left")

    df["vol_regime"] = df["vol_regime"].fillna(1)
    df["macro_signal"] = df["macro_signal"].fillna(0.0)

    # Derive macro regime.
    # NOTE: Model 8A was trained (see build_model8A_dataset.py) with
    # macro_regime = 1  <=>  macro_signal < 0  (negative/bearish macro outlook).
    # Inference MUST use the same convention, otherwise the 8A tradability gate
    # receives an inverted feature and flips its decision.
    df["macro_regime"] = (df["macro_signal"] < 0).astype(int)

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------
    model8A = joblib.load(
        "model8_meta_classifier/models/model8A_regime_selector.pkl"
    )
    model8B = joblib.load(
        "model8_meta_classifier/models/model8B_directional.pkl"
    )

    # --------------------------------------------------
    # Stage 1 — Tradability (Model 8A)
    # --------------------------------------------------
    FEATURES_8A = ["vol_regime", "macro_regime", "macro_signal"]
    df["tradable"] = model8A.predict(df[FEATURES_8A])

    # --------------------------------------------------
    # Stage 2 — Direction + Confidence (Model 8B)
    # 🔑 CRITICAL FIX: enforce training feature order
    # --------------------------------------------------
    FEATURES_8B = model8B.get_booster().feature_names

    X_8B = df[FEATURES_8B]

    probs = model8B.predict_proba(X_8B)
    df["direction"] = model8B.predict(X_8B)
    df["confidence"] = probs.max(axis=1)

    # --------------------------------------------------
    # Final Signal Logic
    # --------------------------------------------------
    df["final_signal"] = "NO_TRADE"

    active = (df["tradable"] == 1) & (df["confidence"] >= TAU)

    df.loc[active & (df["direction"] == 1), "final_signal"] = "LONG"
    df.loc[active & (df["direction"] == 0), "final_signal"] = "SHORT"

    # --------------------------------------------------
    # Save output — APPEND-ONLY publication
    # --------------------------------------------------
    # The block above computes a fresh full-history frame (needed for the
    # coverage diagnostic), but only *newly closed* dates may be published:
    #   * every row already present in OUT_CSV is kept byte-for-byte —
    #     previously published predictions are immutable, upstream refits
    #     (GARCH, base models, etc.) must never rewrite them;
    #   * rows for candles that have not closed yet (date >= today UTC)
    #     are never published (and defensively purged if found);
    #   * only dates missing from OUT_CSV are appended.
    # Re-running the pipeline on the same day is therefore byte-idempotent.
    fresh = df[OUT_COLUMNS].copy()
    fresh["date"] = pd.to_datetime(fresh["date"], utc=True)
    fresh = fresh.sort_values("date").reset_index(drop=True)
    fresh = fresh[fresh["date"].dt.date < datetime.now(timezone.utc).date()]

    if not os.path.exists(OUT_CSV):
        fresh.to_csv(OUT_CSV, index=False)
        print("✅ MODEL 8 FINAL COMPLETE")
        print(f"Coverage @ τ={TAU}: {(active.mean() * 100):.2f}%")
        print(f"Created → {OUT_CSV} | rows={len(fresh)}")
        return

    published = pd.read_csv(OUT_CSV)
    if not set(OUT_COLUMNS).issubset(published.columns):
        raise RuntimeError(
            f"❌ {OUT_CSV} has an unexpected schema {list(published.columns)}; "
            "refusing to touch published predictions. Restore the 6-column "
            "file before re-running."
        )
    published = published[OUT_COLUMNS]
    published["date"] = pd.to_datetime(published["date"], utc=True)

    today_utc = datetime.now(timezone.utc).date()

    # Defensive purge: a prediction must never exist for an unclosed candle.
    unclosed = published["date"].dt.date >= today_utc
    n_unclosed = int(unclosed.sum())
    if n_unclosed:
        dropped = (published.loc[unclosed, "date"]
                   .dt.strftime("%Y-%m-%d").tolist())
        print(f"🗑️ Removing non-closed prediction row(s) {dropped} "
              "(candle had not closed when they were published)")
        published = published.loc[~unclosed]

    # Optional, explicit override: MODEL8_REFRESH_DATES=YYYY-MM-DD[,..]
    # recomputes ONLY the listed already-published dates from the current
    # dataset (they are dropped from the frozen set and re-appended below).
    # Use when a base signal was unavailable at first publish (e.g. signal_4
    # arriving late from Blockchain.com). All other rows stay immutable.
    refresh_raw = [d.strip() for d in os.getenv("MODEL8_REFRESH_DATES", "").split(",") if d.strip()]
    if refresh_raw:
        refresh_set = {pd.to_datetime(d, utc=True) for d in refresh_raw}
        n_refresh = int(published["date"].isin(refresh_set).sum())
        published = published[~published["date"].isin(refresh_set)]
        print(f"⚠️ MODEL8_REFRESH_DATES={refresh_raw}: re-publishing "
              f"{n_refresh} previously-published row(s) from the current dataset")

    published_dates = set(published["date"])
    new_rows = fresh[~fresh["date"].isin(published_dates)]

    combined = (
        pd.concat([published, new_rows], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    # Keep published dtypes/formatting stable (int 0/1 vs float 0.0/1.0).
    combined["tradable"] = combined["tradable"].astype("int64")
    combined["macro_regime"] = combined["macro_regime"].astype("int64")
    combined["vol_regime"] = combined["vol_regime"].astype("float64")
    combined["confidence"] = combined["confidence"].astype("float64")
    combined = combined[OUT_COLUMNS]

    combined.to_csv(OUT_CSV, index=False)

    print("✅ MODEL 8 FINAL COMPLETE")
    print(f"Coverage @ τ={TAU}: {(active.mean() * 100):.2f}%")
    print(f"🔒 Append-only publish: {len(published)} frozen row(s), "
          f"{len(new_rows)} appended, {n_unclosed} purged")
    if len(new_rows):
        print("   new rows: "
              + ", ".join(new_rows["date"].dt.strftime("%Y-%m-%d")))
    print(f"Saved → {OUT_CSV} | rows={len(combined)}")

if __name__ == "__main__":
    main()
