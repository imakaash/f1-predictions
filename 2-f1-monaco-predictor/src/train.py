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
    "form_avg_fp_delta",       # FP long-run pace trend across recent races
    "team_form_avg_finish",
    "team_form_avg_points",
    "track_driver_starts",
    "track_driver_avg_finish",
    "track_driver_podiums",
    "track_team_avg_finish",
    "track_team_podiums",
    "fp_long_run_delta",       # FP1/FP2 long-run pace at this race vs field
    "top_speed_delta",         # qualifying top speed vs field median (km/h)
    "is_wet",                  # 1 if race-day rainfall recorded
    "air_temp",                # race-day air temperature (°C)
    "track_temp",              # race-day track temperature (°C)
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


def _evaluate_ranker(df: pd.DataFrame) -> dict:
    """Leave-one-year-out CV for LightGBM LambdaRank (full finishing order)."""
    n_drivers = 20
    df = df.copy()
    df["relevance"] = (
        (n_drivers + 1 - df["finish_position"].fillna(n_drivers + 1))
        .clip(lower=0)
        .astype(int)
    )

    years = sorted(df["year"].unique())
    hits: list[float] = []

    for held_out in years:
        train = df[df["year"] != held_out].sort_values("year")
        test = df[df["year"] == held_out]
        if len(test) == 0 or test["podium"].sum() == 0:
            continue

        col_medians = train[FEATURE_COLS].median()
        X_train = train[FEATURE_COLS].fillna(col_medians)
        y_train = train["relevance"].values
        train_groups = train.groupby("year").size().values

        X_test = test[FEATURE_COLS].fillna(col_medians)

        ranker = lgb.LGBMRanker(
            n_estimators=400,
            num_leaves=15,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            verbosity=-1,
            n_jobs=-1,
        )
        ranker.fit(X_train, y_train, group=train_groups)
        scores = ranker.predict(X_test)
        hits.append(_top3_hit_rate(test["podium"].values, scores))

    return {
        "auc_mean": float("nan"),
        "auc_std": float("nan"),
        "ap_mean": float("nan"),
        "top3_hit_mean": float(np.mean(hits)) if hits else float("nan"),
        "top3_hit_std": float(np.std(hits)) if hits else float("nan"),
        "n_folds": len(hits),
    }


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

    ranker_metrics = _evaluate_ranker(df)
    results["LambdaRank"] = ranker_metrics
    log.info(
        "%-12s  AUC=%-9s  AP=%-9s  Top3-hit=%.3f±%.3f   (folds=%d)",
        "LambdaRank", "n/a", "n/a",
        ranker_metrics["top3_hit_mean"],
        ranker_metrics["top3_hit_std"],
        ranker_metrics["n_folds"],
    )

    # Choose best by Top-3 hit rate (the metric we actually care about).
    best_name = max(results, key=lambda k: results[k]["top3_hit_mean"])
    log.info("Best model: %s", best_name)

    # Refit best model on ALL data for final use at prediction time.
    col_medians = df[FEATURE_COLS].median()
    if best_name == "LambdaRank":
        n_drivers = 20
        df_r = df.copy()
        df_r["relevance"] = (
            (n_drivers + 1 - df_r["finish_position"].fillna(n_drivers + 1))
            .clip(lower=0)
            .astype(int)
        )
        df_r = df_r.sort_values("year")
        X_all = df_r[FEATURE_COLS].fillna(col_medians)
        y_all = df_r["relevance"].values
        groups_all = df_r.groupby("year").size().values
        final_model = lgb.LGBMRanker(
            n_estimators=400, num_leaves=15, learning_rate=0.05,
            random_state=RANDOM_SEED, verbosity=-1, n_jobs=-1,
        )
        final_model.fit(X_all, y_all, group=groups_all)
    else:
        best_spec = next(s for s in _build_models() if s.name == best_name)
        X_all = df[FEATURE_COLS].fillna(col_medians)
        y_all = df["podium"].values
        final_model = best_spec.factory()
        final_model.fit(X_all, y_all)

    artifact = {
        "model": final_model,
        "feature_cols": FEATURE_COLS,
        "feature_medians": col_medians.to_dict(),
        "best_name": best_name,
        "is_ranker": best_name == "LambdaRank",
    }
    joblib.dump(artifact, MODEL_DIR / "best.joblib")
    log.info("Saved %s", MODEL_DIR / "best.joblib")

    # Save the comparison table for the README / reports.
    with open(MODEL_DIR / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
