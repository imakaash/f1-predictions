"""Feature engineering.

We produce two artifacts:

    data/training.parquet  - one row per (driver, historical Canadian GP)
                             with features computed using ONLY data from
                             before that race. Label = top-3 finish.

    data/predict_input.parquet - one row per driver entered for the upcoming
                                 race, with features computed from the
                                 full available history (past Canada GPs +
                                 season form so far).

Why a separate predict_input? Because at prediction time we don't have the
race's grid yet (qualifying hasn't happened) — we use the average qualifying
slot the driver/team has held this season as a proxy. This is overridden in
predict.py when actual qualifying results become available.

Run:
    python -m src.build_features
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    CURRENT_SEASON_ROUNDS_COMPLETED,
    DATA_DIR,
    PODIUM_THRESHOLD,
    ROLLING_FORM_WINDOW,
    TARGET_RACE,
    TARGET_YEAR,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _form_features_before(
    season_results: pd.DataFrame,
    driver: str,
    team: str,
    cutoff_date: pd.Timestamp,
    n: int = ROLLING_FORM_WINDOW,
) -> dict:
    """Compute recent-form features using only races strictly before cutoff_date.

    season_results is expected to be sorted oldest->newest already.
    """
    prior = season_results[season_results["event_date"] < cutoff_date]

    # Driver's last n races
    driver_recent = prior[prior["driver"] == driver].tail(n)
    if not driver_recent.empty:
        avg_finish = driver_recent["finish_position"].mean()
        avg_qual = driver_recent["qual_position"].mean()
        dnf_rate = driver_recent["dnf"].mean()
        avg_points = driver_recent["points"].mean()
    else:
        avg_finish = avg_qual = avg_points = np.nan
        dnf_rate = 0.0

    # Team's last n races (both cars combined). Strong signal for car pace.
    team_recent = prior[prior["team"] == team].tail(n * 2)
    team_avg_finish = team_recent["finish_position"].mean() if not team_recent.empty else np.nan
    team_avg_points = team_recent["points"].mean() if not team_recent.empty else np.nan

    return {
        "form_avg_finish": avg_finish,
        "form_avg_qual": avg_qual,
        "form_dnf_rate": dnf_rate,
        "form_avg_points": avg_points,
        "team_form_avg_finish": team_avg_finish,
        "team_form_avg_points": team_avg_points,
    }


def _track_history_before(
    canada_history: pd.DataFrame,
    driver: str,
    team: str,
    year: int,
) -> dict:
    """How has this driver/team done at Canada in PREVIOUS years?"""
    prior = canada_history[canada_history["year"] < year]

    drv = prior[prior["driver"] == driver]
    tm = prior[prior["team"] == team]

    return {
        "track_driver_starts": len(drv),
        "track_driver_avg_finish": drv["finish_position"].mean() if not drv.empty else np.nan,
        "track_driver_podiums": (drv["finish_position"] <= PODIUM_THRESHOLD).sum(),
        "track_team_avg_finish": tm["finish_position"].mean() if not tm.empty else np.nan,
        "track_team_podiums": (tm["finish_position"] <= PODIUM_THRESHOLD).sum(),
    }


# ---------------------------------------------------------------------------
# Build the training set
# ---------------------------------------------------------------------------
def build_training() -> pd.DataFrame:
    canada = pd.read_parquet(DATA_DIR / "canada_history.parquet")
    canada = canada.sort_values("event_date").reset_index(drop=True)

    rows = []
    for _, race_row in canada.iterrows():
        # For each driver in this past Canada GP, compute features using only
        # races before this event.
        cutoff = race_row["event_date"]

        # "Season form going into Canada that year": all races in the same
        # season earlier than this Canada GP. Easiest: filter canada history
        # to that season -- but we only have Canada races. We approximate
        # using prior-year Canada GPs as track history, and use the ENTIRE
        # canada history before cutoff as the recency proxy. A richer build
        # would also pull every race of every season; that's the obvious
        # next step (see README).
        feats = _form_features_before(canada, race_row["driver"], race_row["team"], cutoff)
        feats.update(
            _track_history_before(canada, race_row["driver"], race_row["team"], race_row["year"])
        )
        feats.update(
            {
                "driver": race_row["driver"],
                "team": race_row["team"],
                "year": race_row["year"],
                "grid": race_row["grid"],
                "qual_position": race_row["qual_position"],
                "finish_position": race_row["finish_position"],
                "podium": int(
                    (race_row["finish_position"] <= PODIUM_THRESHOLD)
                    and not race_row["dnf"]
                ),
            }
        )
        rows.append(feats)

    df = pd.DataFrame(rows)
    path = DATA_DIR / "training.parquet"
    df.to_parquet(path)
    log.info("Wrote %s (%d rows, %d positives)", path, len(df), df["podium"].sum())
    return df


# ---------------------------------------------------------------------------
# Build the prediction-time feature row
# ---------------------------------------------------------------------------
def build_predict_input() -> pd.DataFrame:
    canada = pd.read_parquet(DATA_DIR / "canada_history.parquet")
    season = pd.read_parquet(DATA_DIR / "season_form.parquet")
    season = season.sort_values("event_date").reset_index(drop=True)

    # Use the most-recent race in the season as the entry list.
    last_race = season["race"].iloc[-1]
    entry = season[season["race"] == last_race][["driver", "team"]].drop_duplicates()
    log.info("Using entry list from %s %s: %d drivers", TARGET_YEAR, last_race, len(entry))

    # Cutoff = "now" (after the most recent race)
    cutoff = season["event_date"].max() + pd.Timedelta(days=1)

    rows = []
    for _, drv_row in entry.iterrows():
        driver, team = drv_row["driver"], drv_row["team"]

        feats = _form_features_before(season, driver, team, cutoff)
        feats.update(_track_history_before(canada, driver, team, TARGET_YEAR))

        # No qualifying yet — use average qual position this season as a proxy.
        # predict.py will overwrite this once real quali results exist.
        drv_season = season[season["driver"] == driver]
        avg_qual = drv_season["qual_position"].mean()
        feats.update(
            {
                "driver": driver,
                "team": team,
                "year": TARGET_YEAR,
                "grid": avg_qual,             # proxy
                "qual_position": avg_qual,    # proxy
            }
        )
        rows.append(feats)

    df = pd.DataFrame(rows)
    path = DATA_DIR / "predict_input.parquet"
    df.to_parquet(path)
    log.info("Wrote %s (%d rows)", path, len(df))
    return df


def main() -> None:
    build_training()
    build_predict_input()


if __name__ == "__main__":
    main()
