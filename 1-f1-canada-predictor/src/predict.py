"""Predict the upcoming Canadian GP podium.

Outputs each driver's podium probability and the model's top-3 picks.

If qualifying has already happened, you can pass --qual-csv with columns
'driver,qual_position' to use real qualifying instead of the season-average
proxy.

Run:
    python -m src.predict
    python -m src.predict --qual-csv data/canada_2026_qual.csv
"""
from __future__ import annotations

import argparse
import logging

import joblib
import pandas as pd

from src.config import DATA_DIR, MODEL_DIR, TARGET_RACE, TARGET_YEAR

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qual-csv",
        type=str,
        default=None,
        help="CSV with columns 'driver,qual_position' from real qualifying.",
    )
    args = parser.parse_args()

    artifact = joblib.load(MODEL_DIR / "best.joblib")
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    feature_medians = artifact["feature_medians"]
    best_name = artifact["best_name"]

    df = pd.read_parquet(DATA_DIR / "predict_input.parquet")

    # Optionally splice in real qualifying results.
    if args.qual_csv:
        qual = pd.read_csv(args.qual_csv)
        df = df.drop(columns=["grid", "qual_position"]).merge(
            qual.rename(columns={"qual_position": "qual_position"}),
            on="driver",
            how="left",
        )
        df["grid"] = df["qual_position"]
        log.info("Merged real qualifying from %s", args.qual_csv)

    # Fill missing features with the same medians used at training time.
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = X[c].fillna(feature_medians.get(c))

    proba = model.predict_proba(X)[:, 1]
    df = df.assign(podium_prob=proba).sort_values("podium_prob", ascending=False)

    log.info("Predictions for %s %s GP using %s", TARGET_YEAR, TARGET_RACE, best_name)
    print()
    print(f"=== {TARGET_YEAR} {TARGET_RACE} GP — Podium probabilities ({best_name}) ===")
    print()
    show = df[["driver", "team", "qual_position", "podium_prob"]].copy()
    show["podium_prob"] = (show["podium_prob"] * 100).round(1).astype(str) + "%"
    print(show.to_string(index=False))
    print()
    podium = df.head(3)
    print("🏆 Predicted podium:")
    for i, (_, row) in enumerate(podium.iterrows(), start=1):
        print(f"  P{i}: {row['driver']:>4}  ({row['team']})")
    print()

    # Save for downstream use.
    df.to_parquet(DATA_DIR / "predictions.parquet")
    df.to_csv(DATA_DIR / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
