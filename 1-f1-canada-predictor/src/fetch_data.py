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
import numpy as np
import pandas as pd

from src.config import (
    CACHE_DIR,
    CURRENT_SEASON_ROUNDS_COMPLETED,
    DATA_DIR,
    MIN_LONG_RUN_LAPS,
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
# New enrichment helpers
# ---------------------------------------------------------------------------
def _practice_pace(year: int, race: str) -> pd.DataFrame | None:
    """Long-run median pace from FP1 + FP2, normalised to field median.

    fp_long_run_delta: (driver_median - field_median) / field_median * 100.
    Negative = faster than field.
    """
    driver_times: dict[str, list[float]] = {}
    for session_type in ("FP1", "FP2"):
        try:
            sess = fastf1.get_session(year, race, session_type)
            sess.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception as e:
            log.warning("Practice load %s %s %s: %s", year, race, session_type, e)
            continue

        laps = sess.laps
        if laps is None or laps.empty:
            continue
        if "IsAccurate" in laps.columns:
            laps = laps[laps["IsAccurate"]]
        laps = laps.dropna(subset=["Driver", "Stint"])

        for (driver, _stint), grp in laps.groupby(["Driver", "Stint"]):
            if len(grp) < MIN_LONG_RUN_LAPS:
                continue
            times = grp["LapTime"].dt.total_seconds().dropna()
            if not times.empty:
                driver_times.setdefault(str(driver), []).extend(times.tolist())

    if not driver_times:
        return None

    driver_median = {d: float(np.median(ts)) for d, ts in driver_times.items()}
    field_median = float(np.median(list(driver_median.values())))
    return pd.DataFrame([
        {"driver": d, "fp_long_run_delta": round((t - field_median) / field_median * 100, 4)}
        for d, t in driver_median.items()
    ])


def _race_weather(year: int, race: str) -> dict:
    """Rain flag and temperatures from the race session weather data."""
    try:
        sess = fastf1.get_session(year, race, "R")
        sess.load(laps=False, telemetry=False, weather=True, messages=False)
        w = sess.weather_data
        if w is None or w.empty:
            return {}
        return {
            "is_wet": int(bool(w["Rainfall"].any())),
            "air_temp": round(float(w["AirTemp"].mean()), 1),
            "track_temp": round(float(w["TrackTemp"].mean()), 1),
        }
    except Exception as e:
        log.warning("Weather load %s %s: %s", year, race, e)
        return {}


def _qualifying_top_speed(year: int, race: str) -> pd.DataFrame | None:
    """Per-driver max speed from qualifying fastest-lap telemetry.

    top_speed_delta: driver top speed minus field median (km/h).
    Positive = faster straight-line speed than the field median.
    """
    try:
        sess = fastf1.get_session(year, race, "Q")
        sess.load(laps=True, telemetry=True, weather=False, messages=False)
    except Exception as e:
        log.warning("Quali telemetry load %s %s: %s", year, race, e)
        return None

    rows = []
    for driver_num in sess.drivers:
        try:
            drv_laps = sess.laps[sess.laps["DriverNumber"] == driver_num]
            if drv_laps.empty:
                continue
            fastest = drv_laps.loc[drv_laps["LapTime"].idxmin()]
            tel = fastest.get_car_data()
            if tel.empty:
                continue
            rows.append({"driver": str(fastest["Driver"]), "top_speed": float(tel["Speed"].max())})
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    field_med = float(df["top_speed"].median())
    df["top_speed_delta"] = (df["top_speed"] - field_med).round(2)
    return df[["driver", "top_speed_delta"]]


def _merge_enrichments(df: pd.DataFrame, year: int, race: str) -> pd.DataFrame:
    """Attach weather, practice pace, and top-speed columns to a race DataFrame."""
    for col, val in _race_weather(year, race).items():
        df[col] = val
    for col in ("is_wet", "air_temp", "track_temp"):
        if col not in df.columns:
            df[col] = np.nan

    pace = _practice_pace(year, race)
    df = df.merge(pace, on="driver", how="left") if pace is not None else df.assign(fp_long_run_delta=np.nan)

    spd = _qualifying_top_speed(year, race)
    df = df.merge(spd, on="driver", how="left") if spd is not None else df.assign(top_speed_delta=np.nan)

    return df


# ---------------------------------------------------------------------------
# Build the two datasets
# ---------------------------------------------------------------------------
def fetch_canada_history() -> pd.DataFrame:
    log.info("Fetching historical Canadian GP results for years %s", TRAINING_YEARS)
    frames = []
    for year in TRAINING_YEARS:
        df = _race_results(year, TARGET_RACE)
        if df is None:
            continue
        df = _merge_enrichments(df, year, TARGET_RACE)
        log.info("  %s: %d drivers", year, len(df))
        frames.append(df)
    if not frames:
        raise RuntimeError("No historical Canadian GP data fetched.")
    out = pd.concat(frames, ignore_index=True)
    path = DATA_DIR / "canada_history.parquet"
    out.to_parquet(path)
    out.to_csv(DATA_DIR / "canada_history.csv", index=False)
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
        if df is None:
            continue
        df = _merge_enrichments(df, TARGET_YEAR, race)
        log.info("  %s %s: %d drivers", TARGET_YEAR, race, len(df))
        frames.append(df)
    if not frames:
        raise RuntimeError("No current-season data fetched.")
    out = pd.concat(frames, ignore_index=True)
    path = DATA_DIR / "season_form.parquet"
    out.to_parquet(path)
    out.to_csv(DATA_DIR / "season_form.csv", index=False)
    log.info("Wrote %s (%d rows)", path, len(out))
    return out


def main() -> None:
    fetch_canada_history()
    fetch_season_form()


if __name__ == "__main__":
    main()
