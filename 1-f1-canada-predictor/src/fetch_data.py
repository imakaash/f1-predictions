"""Pull race + qualifying data using FastF1, save to parquet.

Two datasets are produced:
    data/canada_history.parquet  - past Canadian GPs (track-specific signal)
    data/season_form.parquet     - current-season races so far (form signal)

Run:
    python -m src.fetch_data
"""
from __future__ import annotations

import logging
from typing import Optional

import fastf1
import pandas as pd

from src.config import (
    CACHE_DIR,
    CURRENT_SEASON_ROUNDS_COMPLETED,
    DATA_DIR,
    TARGET_RACE,
    TARGET_YEAR,
    TRAINING_YEARS,
)

# FastF1 is chatty by default; quiet it down.
fastf1.Cache.enable_cache(str(CACHE_DIR))
logging.getLogger("fastf1").setLevel(logging.WARNING)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------
def _load_session(year: int, race: str, session_type: str):
    """Load a FastF1 session, returning None on failure (cancelled race, etc)."""
    try:
        session = fastf1.get_session(year, race, session_type)
        # Results are enough for our features; skip telemetry to keep it fast.
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        return session
    except Exception as e:  # noqa: BLE001
        log.warning("Could not load %s %s %s: %s", year, race, session_type, e)
        return None


def _race_results(year: int, race: str) -> Optional[pd.DataFrame]:
    """Return a DataFrame of one race's results with qualifying merged in."""
    race_session = _load_session(year, race, "R")
    qual_session = _load_session(year, race, "Q")
    if race_session is None or race_session.results is None or race_session.results.empty:
        return None

    race_df = race_session.results.copy()
    # Standardize columns we care about.
    race_df = race_df.rename(
        columns={
            "Abbreviation": "driver",
            "TeamName": "team",
            "GridPosition": "grid",
            "Position": "finish_position",
            "Status": "status",
            "Points": "points",
        }
    )
    keep_cols = ["driver", "team", "grid", "finish_position", "status", "points"]
    race_df = race_df[[c for c in keep_cols if c in race_df.columns]].copy()

    # Add qualifying position separately - more reliable than 'grid' (penalties shift grid).
    if qual_session is not None and qual_session.results is not None and not qual_session.results.empty:
        q = qual_session.results.rename(
            columns={"Abbreviation": "driver", "Position": "qual_position"}
        )[["driver", "qual_position"]]
        race_df = race_df.merge(q, on="driver", how="left")
    else:
        race_df["qual_position"] = pd.NA

    race_df["year"] = year
    race_df["race"] = race
    race_df["round"] = race_session.event.get("RoundNumber")
    race_df["event_date"] = pd.to_datetime(race_session.event.get("EventDate"))

    # Numeric coercion + DNF flag.
    for col in ("grid", "finish_position", "qual_position", "points"):
        race_df[col] = pd.to_numeric(race_df[col], errors="coerce")
    race_df["dnf"] = ~race_df["status"].astype(str).str.contains(
        "Finished|\\+", regex=True, na=False
    )

    return race_df


# ---------------------------------------------------------------------------
# Build the two datasets
# ---------------------------------------------------------------------------
def fetch_canada_history() -> pd.DataFrame:
    log.info("Fetching historical Canadian GP results for years %s", TRAINING_YEARS)
    frames = []
    for year in TRAINING_YEARS:
        df = _race_results(year, TARGET_RACE)
        if df is not None:
            log.info("  %s: %d drivers", year, len(df))
            frames.append(df)
    if not frames:
        raise RuntimeError("No historical Canadian GP data fetched.")
    out = pd.concat(frames, ignore_index=True)
    path = DATA_DIR / "canada_history.parquet"
    out.to_parquet(path)
    log.info("Wrote %s (%d rows)", path, len(out))
    return out


def fetch_season_form() -> pd.DataFrame:
    log.info(
        "Fetching %s season form (%d races so far)",
        TARGET_YEAR,
        len(CURRENT_SEASON_ROUNDS_COMPLETED),
    )
    frames = []
    for race in CURRENT_SEASON_ROUNDS_COMPLETED:
        df = _race_results(TARGET_YEAR, race)
        if df is not None:
            log.info("  %s %s: %d drivers", TARGET_YEAR, race, len(df))
            frames.append(df)
    if not frames:
        raise RuntimeError("No current-season data fetched.")
    out = pd.concat(frames, ignore_index=True)
    path = DATA_DIR / "season_form.parquet"
    out.to_parquet(path)
    log.info("Wrote %s (%d rows)", path, len(out))
    return out


def main() -> None:
    fetch_canada_history()
    fetch_season_form()


if __name__ == "__main__":
    main()
