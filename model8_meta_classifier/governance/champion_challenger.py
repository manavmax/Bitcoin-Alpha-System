"""
Champion–Challenger governance hooks for Model 8.

Goal:
- Avoid retraining 8A / 8B on every run.
- Instead, periodically train "challenger" models on a more recent window,
  compare against the current "champion", and only promote if performance
  improves according to predefined metrics.

This module defines the structure; you can fill in the exact metric
thresholds and scheduling logic as you harden the system.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass
class ModelPaths:
    """Paths for champion / challenger artifacts."""

    model8a_champion: Path = BASE_DIR / "models" / "model8A_regime_selector.pkl"
    model8b_champion: Path = BASE_DIR / "models" / "model8B_directional.pkl"

    model8a_challenger: Path = BASE_DIR / "models" / "model8A_regime_selector_challenger.pkl"
    model8b_challenger: Path = BASE_DIR / "models" / "model8B_directional_challenger.pkl"


def load_model8A_training_data() -> pd.DataFrame:
    """
    Load full 8A dataset; governance logic can later subset to
    'recent window' vs 'older window' as needed.
    """
    path = BASE_DIR / "data" / "model8A_dataset.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_model8B_training_data() -> pd.DataFrame:
    """
    Load full 8B dataset of tradable-only samples.
    """
    path = BASE_DIR / "data" / "model8B_dataset.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def train_challenger_8A(window_years: int = 2) -> XGBClassifier:
    """
    Train a challenger 8A regime-selector on the most recent N years.
    """
    df = load_model8A_training_data()
    cutoff = df["date"].max() - pd.DateOffset(years=window_years)
    df_recent = df[df["date"] >= cutoff].copy()

    features = ["vol_regime", "macro_regime", "macro_signal"]
    target = "tradable"

    X = df_recent[features]
    y = df_recent[target]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    print("\n📊 8A Challenger (in-sample, recent window)")
    print(classification_report(y, preds, digits=4))
    print("AUC:", roc_auc_score(y, proba))

    paths = ModelPaths()
    paths.model8a_challenger.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths.model8a_challenger)

    return model


def train_challenger_8B(window_years: int = 2) -> XGBClassifier:
    """
    Train a challenger 8B directional classifier on tradable-only samples
    from the most recent N years.
    """
    df = load_model8B_training_data()
    cutoff = df["date"].max() - pd.DateOffset(years=window_years)
    df_recent = df[df["date"] >= cutoff].copy()

    features = [
        "signal_1",
        "signal_2",
        "signal_3",
        "signal_4",
        "signal_6",
        "signal_1_missing",
        "signal_2_missing",
        "signal_3_missing",
        "signal_4_missing",
        "signal_6_missing",
    ]
    target = "direction"

    X = df_recent[features]
    y = df_recent[target]

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    print("\n📊 8B Challenger (in-sample, recent window)")
    print(classification_report(y, preds))
    print("Accuracy:", accuracy_score(y, preds))
    print("AUC:", roc_auc_score(y, proba))

    paths = ModelPaths()
    paths.model8b_challenger.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths.model8b_challenger)

    return model


def compare_challenger_vs_champion(
    model_name: str,
    y_true: np.ndarray,
    y_champion: np.ndarray,
    y_challenger: np.ndarray,
) -> Dict[str, Any]:
    """
    Simple comparison helper to decide if challenger is better than champion.
    You can extend this with more sophisticated metrics and thresholds.
    """
    acc_champion = accuracy_score(y_true, y_champion)
    acc_challenger = accuracy_score(y_true, y_challenger)

    print(f"\n📈 {model_name} Champion vs Challenger")
    print(f"Champion accuracy : {acc_champion:.4f}")
    print(f"Challenger accuracy: {acc_challenger:.4f}")

    return {
        "model": model_name,
        "acc_champion": acc_champion,
        "acc_challenger": acc_challenger,
        "improved": acc_challenger > acc_champion,
    }


def promote_if_better() -> Tuple[bool, bool]:
    """
    High-level hook:
    - Load champion models.
    - Train challengers on recent window.
    - Evaluate on a held-out validation slice.
    - If challenger outperforms, overwrite champion artifact.

    Current simple policy (can be hardened later):
    - Use the last 180 calendar days as a validation slice.
    - Compare champion vs challenger on plain accuracy.
    - Promote if challenger accuracy exceeds champion accuracy by a
      small margin (min_improvement).
    """
    paths = ModelPaths()

    # ----------------------------------------
    # 8A: Regime selector (tradable vs not)
    # ----------------------------------------
    df8a = load_model8A_training_data()
    if df8a.empty:
        return False, False

    cutoff = df8a["date"].max() - pd.DateOffset(days=180)
    val = df8a[df8a["date"] >= cutoff].copy()
    if len(val) < 100:
        # Not enough fresh data to make a robust decision.
        return False, False

    X_val_8a = val[["vol_regime", "macro_regime", "macro_signal"]]
    y_val_8a = val["tradable"].values

    if not paths.model8a_champion.exists() or not paths.model8a_challenger.exists():
        promoted_8a = False
    else:
        champion_8a: XGBClassifier = joblib.load(paths.model8a_champion)
        challenger_8a: XGBClassifier = joblib.load(paths.model8a_challenger)

        y_champ_8a = champion_8a.predict(X_val_8a)
        y_chall_8a = challenger_8a.predict(X_val_8a)

        stats_8a = compare_challenger_vs_champion(
            "Model 8A",
            y_true=y_val_8a,
            y_champion=y_champ_8a,
            y_challenger=y_chall_8a,
        )

        min_improvement = 0.005  # 0.5pp accuracy improvement
        improved = (
            stats_8a["acc_challenger"] - stats_8a["acc_champion"] >= min_improvement
        )
        promoted_8a = bool(improved)
        if promoted_8a:
            joblib.dump(challenger_8a, paths.model8a_champion)

    # ----------------------------------------
    # 8B: Directional model (long vs short)
    # ----------------------------------------
    df8b = load_model8B_training_data()
    if df8b.empty:
        return promoted_8a, False

    cutoff_b = df8b["date"].max() - pd.DateOffset(days=180)
    val_b = df8b[df8b["date"] >= cutoff_b].copy()
    if len(val_b) < 100:
        return promoted_8a, False

    features_b = [
        "signal_1",
        "signal_2",
        "signal_3",
        "signal_4",
        "signal_6",
        "signal_1_missing",
        "signal_2_missing",
        "signal_3_missing",
        "signal_4_missing",
        "signal_6_missing",
    ]
    X_val_8b = val_b[features_b]
    y_val_8b = val_b["direction"].values

    if not paths.model8b_champion.exists() or not paths.model8b_challenger.exists():
        promoted_8b = False
    else:
        champion_8b: XGBClassifier = joblib.load(paths.model8b_champion)
        challenger_8b: XGBClassifier = joblib.load(paths.model8b_challenger)

        y_champ_8b = champion_8b.predict(X_val_8b)
        y_chall_8b = challenger_8b.predict(X_val_8b)

        stats_8b = compare_challenger_vs_champion(
            "Model 8B",
            y_true=y_val_8b,
            y_champion=y_champ_8b,
            y_challenger=y_chall_8b,
        )

        min_improvement_b = 0.005
        improved_b = (
            stats_8b["acc_challenger"] - stats_8b["acc_champion"] >= min_improvement_b
        )
        promoted_8b = bool(improved_b)
        if promoted_8b:
            joblib.dump(challenger_8b, paths.model8b_champion)

    return promoted_8a, promoted_8b

