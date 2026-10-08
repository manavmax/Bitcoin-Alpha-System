"""
Shared append-only prediction store for the Bitcoin Alpha System.

Guarantee (applies to every model's published prediction file, models 1-8):

  * A prediction row, once published for a UTC date, is FROZEN forever.
    Later runs may only APPEND rows for newly closed dates - they never
    recompute or rewrite previously published prediction values, even if
    the underlying model retrains or its inputs/features drift.
  * Rows for candles that have not closed yet (date >= today, UTC) are never
    published. If such a row ever slipped into the store, it is purged on
    the next run.
  * Ground-truth label columns that are unknown at publish time (e.g.
    true_return / target_return) may be backfilled for already-published
    dates. Outcomes are not predictions, so backfilling them never changes
    a published prediction.
  * Explicit escape hatch: pass refresh_env="<MODEL>_REFRESH_DATES" and set
    that env var to a comma-separated list of dates to re-publish ONLY those
    listed dates. Every other published row stays immutable.

Semantics mirror model8_meta_classifier/src/model8_final.py, which already
implemented this guarantee for the final published signal.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def publish_append_only(
    fresh: pd.DataFrame,
    out_path,
    date_col: str = "date",
    backfill_cols=(),
    refresh_env: str | None = None,
) -> pd.DataFrame:
    """
    Merge `fresh` predictions into the append-only store at `out_path`.

    - Existing published dates are kept verbatim (predictions are frozen).
    - Only dates not yet published are appended.
    - NaN cells in `backfill_cols` (ground-truth outcomes) of published dates
      are filled from `fresh` where the outcome has since become known.
    - Rows dated >= today (UTC) are never published (candle not closed yet).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    today_utc = datetime.now(timezone.utc).date()

    # ---------------- normalize fresh predictions ----------------
    fresh = fresh.copy()
    if date_col not in fresh.columns:
        raise RuntimeError(f"❌ fresh predictions have no '{date_col}' column")

    fresh[date_col] = pd.to_datetime(fresh[date_col], errors="coerce", utc=True)
    fresh = fresh.dropna(subset=[date_col])

    unclosed = fresh[date_col].dt.date >= today_utc
    n_unclosed_fresh = int(unclosed.sum())
    if n_unclosed_fresh:
        dropped = fresh.loc[unclosed, date_col].dt.strftime("%Y-%m-%d").tolist()
        print(f"🗑️  Skipping {n_unclosed_fresh} unclosed-candle row(s) {dropped} "
              "(a prediction for an unclosed candle must never be published)")
        fresh = fresh.loc[~unclosed]

    fresh = fresh.sort_values(date_col).reset_index(drop=True)
    fresh["_pub_date"] = fresh[date_col].dt.strftime("%Y-%m-%d")
    fresh = fresh.drop_duplicates(subset="_pub_date", keep="last")

    # ---------------- first publication ----------------
    if not out_path.exists():
        out = fresh.drop(columns=["_pub_date"])
        out[date_col] = out[date_col].dt.strftime("%Y-%m-%d")
        out.to_csv(out_path, index=False)
        print(f"🔒 Append-only store created → {out_path} | rows={len(out)}")
        return out


    # ---------------- load published (immutable) rows ----------------
    published = pd.read_csv(out_path)
    if date_col not in published.columns:
        raise RuntimeError(
            f"❌ {out_path} has no '{date_col}' column; refusing to touch it."
        )

    # Schema check: the published file must be a subset of what this run
    # produces. A schema change must be a deliberate manual migration,
    # never a silent side effect of a rerun.
    fresh_cols = [c for c in fresh.columns if c != "_pub_date"]
    extra = [c for c in published.columns if c not in fresh_cols]
    if extra:
        raise RuntimeError(
            f"❌ {out_path} contains column(s) {extra} that the current run "
            "does not produce. Refusing to publish: migrating or restoring "
            "the published file must be a deliberate manual step."
        )
    published = published[fresh_cols]

    published[date_col] = pd.to_datetime(
        published[date_col], errors="coerce", utc=True
    )
    published = published.dropna(subset=[date_col])

    # Defensive purge: a prediction must never exist for an unclosed candle.
    unclosed_pub = published[date_col].dt.date >= today_utc
    n_unclosed_pub = int(unclosed_pub.sum())
    if n_unclosed_pub:
        dropped = (
            published.loc[unclosed_pub, date_col].dt.strftime("%Y-%m-%d").tolist()
        )
        print(f"🗑️ Removing unclosed-candle prediction row(s) {dropped} "
              "(candle had not closed when they were published)")
        published = published.loc[~unclosed_pub]

    published["_pub_date"] = published[date_col].dt.strftime("%Y-%m-%d")
    # Frozen semantics: the FIRST published value for a date always wins.
    published = published.drop_duplicates(subset="_pub_date", keep="first")

    # Explicit, opt-in refresh of specific already-published dates.
    n_refresh = 0
    if refresh_env:
        raw = [d.strip() for d in os.getenv(refresh_env, "").split(",") if d.strip()]
        if raw:
            keys = {pd.to_datetime(d, utc=True).strftime("%Y-%m-%d") for d in raw}
            mask = published["_pub_date"].isin(keys)
            n_refresh = int(mask.sum())
            published = published.loc[~mask]
            print(f"⚠️ {refresh_env}={raw}: re-publishing {n_refresh} "
                  "previously published date(s); all other rows stay immutable")

    # Backfill ground-truth labels that were unknown at publish time.
    n_backfill = 0
    if backfill_cols:
        fl = fresh.drop_duplicates("_pub_date", keep="last").set_index("_pub_date")
        for col in backfill_cols:
            if col not in published.columns or col not in fl.columns:
                continue
            mapped = published["_pub_date"].map(fl[col])
            need = published[col].isna() & mapped.notna()
            n_backfill += int(need.sum())
            published.loc[need, col] = mapped[need]

    # ---------------- merge: frozen rows + new dates only ----------------
    published_dates = set(published["_pub_date"])
    new_rows = fresh.loc[~fresh["_pub_date"].isin(published_dates)]

    published = published.drop(columns=["_pub_date"])
    new_rows = new_rows.drop(columns=["_pub_date"])

    combined = pd.concat([published, new_rows], ignore_index=True)
    combined = combined.sort_values(date_col).reset_index(drop=True)
    combined = combined.drop_duplicates(subset=[date_col], keep="first")

    # Keep published dtypes stable (int vs float flips change CSV formatting).
    for col in published.columns:
        if combined[col].dtype != published[col].dtype:
            try:
                combined[col] = combined[col].astype(published[col].dtype)
            except (TypeError, ValueError):
                pass

    combined[date_col] = pd.to_datetime(
        combined[date_col], errors="coerce", utc=True
    )
    combined = combined.dropna(subset=[date_col])
    combined[date_col] = combined[date_col].dt.strftime("%Y-%m-%d")

    combined.to_csv(out_path, index=False)

    print(f"🔒 Append-only publish → {out_path}")
    print(f"   frozen: {len(published)} | appended: {len(new_rows)} | "
          f"purged (unclosed): {n_unclosed_fresh + n_unclosed_pub} | "
          f"labels backfilled: {n_backfill} | refreshed: {n_refresh}")
    if len(new_rows):
        print("   new dates: "
              + ", ".join(new_rows[date_col].dt.strftime("%Y-%m-%d")))
    return combined
