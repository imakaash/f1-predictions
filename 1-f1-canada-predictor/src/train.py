"""Train and compare four models for podium prediction.

Models:
    1. Logistic Regression (scaled, class-balanced) - simple baseline
    2. Random Forest                                 - non-linear baseline
    3. XGBoost                                       - state-of-the-art GBM
    4. LightGBM                                      - faster GBM

Evaluation:
    Time-aware leave-one-year-out CV. We never train on a Canadian GP that
    happened AFTER the one we're scoring. Metrics:
        - ROC-AUC               (rank quality)
        - Average Precision     (precision-recall area, better for imbalance)
        - Top-3 hit rate        (how many of the actual podium finishers
                                 the model put in its top-3 predictions)

The best model by mean Top-3 hit rate is saved to models/best.joblib.

Run:
    python -m src.train
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_DIR, MODEL_DIR, RANDOM_SEED

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


FEATURE_COLS = [
    "grid",
    "qual_position",
    "form_avg_finish",
    "form_avg_qual",
    "form_dnf_rate",
    "form_avg_points",
    "team_form_avg_finish",
    "team_form_avg_points",
    "track_driver_starts",
    "track_driver_avg_finish",
    "track_driver_podiums",
    "track_team_avg_finish",
    "track_team_podiums",
]


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], object]


def _build_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            "LogReg",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "RandomForest",
            lambda: RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "XGBoost",
            lambda: xgb.XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "LightGBM",
            lambda: lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=-1,
                num_leaves=15,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
    ]


def _top3_hit_rate(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Of the 3 drivers the model ranks highest, how many actually podiumed?

    Returned as a fraction (0..1). Random would be ~3/20 = 0.15.
    """
    order = np.argsort(-y_prob)  # highest prob first
    top3_idx = order[:3]
    return float(y_true[top3_idx].sum()) / 3.0


def _evaluate(df: pd.DataFrame, spec: ModelSpec) -> dict:
    """Leave-one-year-out CV. For each held-out year, train on all OTHER years."""
    years = sorted(df["year"].unique())
    aucs, aps, hits = [], [], []

    for held_out in years:
        train = df[df["year"] != held_out]
        test = df[df["year"] == held_out]
        if len(test) == 0 or test["podium"].sum() == 0:
            continue

        X_train = train[FEATURE_COLS].fillna(train[FEATURE_COLS].median())
        y_train = train["podium"].values
        X_test = test[FEATURE_COLS].fillna(train[FEATURE_COLS].median())
        y_test = test["podium"].values

        model = spec.factory()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        aucs.append(roc_auc_score(y_test, proba))
        aps.append(average_precision_score(y_test, proba))
        hits.append(_top3_hit_rate(y_test, proba))

    return {
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "ap_mean": float(np.mean(aps)) if aps else float("nan"),
        "top3_hit_mean": float(np.mean(hits)) if hits else float("nan"),
        "top3_hit_std": float(np.std(hits)) if hits else float("nan"),
        "n_folds": len(aucs),
    }


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "training.parquet")
    log.info("Training set: %d rows, %d positives", len(df), df["podium"].sum())

    results = {}
    for spec in _build_models():
        metrics = _evaluate(df, spec)
        results[spec.name] = metrics
        log.info(
            "%-12s  AUC=%.3f±%.3f   AP=%.3f   Top3-hit=%.3f±%.3f   (folds=%d)",
            spec.name,
            metrics["auc_mean"],
            metrics["auc_std"],
            metrics["ap_mean"],
            metrics["top3_hit_mean"],
            metrics["top3_hit_std"],
            metrics["n_folds"],
        )

    # Choose best by Top-3 hit rate (the metric we actually care about).
    best_name = max(results, key=lambda k: results[k]["top3_hit_mean"])
    log.info("Best model: %s", best_name)

    # Refit best model on ALL data for final use at prediction time.
    best_spec = next(s for s in _build_models() if s.name == best_name)
    X_all = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y_all = df["podium"].values
    final_model = best_spec.factory()
    final_model.fit(X_all, y_all)

    artifact = {
        "model": final_model,
        "feature_cols": FEATURE_COLS,
        "feature_medians": df[FEATURE_COLS].median().to_dict(),
        "best_name": best_name,
    }
    joblib.dump(artifact, MODEL_DIR / "best.joblib")
    log.info("Saved %s", MODEL_DIR / "best.joblib")

    # Save the comparison table for the README / reports.
    with open(MODEL_DIR / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
