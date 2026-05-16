"""Project-wide configuration.

Edit TARGET_YEAR / TARGET_ROUND when predicting a different race.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
CACHE_DIR = ROOT / "cache"

for d in (DATA_DIR, MODEL_DIR, CACHE_DIR):
    d.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# What we're predicting
# ---------------------------------------------------------------------------
TARGET_YEAR = 2026
TARGET_RACE = "Canada"  # FastF1 accepts the country name

# Years of historical data to use for training.
# 2018 is the earliest year FastF1 has full timing/telemetry data.
# Skipping 2020 (no Canadian GP - COVID) and 2021 (also cancelled).
TRAINING_YEARS = [2018, 2019, 2022, 2023, 2024, 2025]

# Current-season races to also pull (form data for the target prediction).
# Update this list as the season progresses.
CURRENT_SEASON_ROUNDS_COMPLETED = [
    "Australia",
    "China",
    "Japan",
    "Miami",
]

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
PODIUM_THRESHOLD = 3        # top-N finish counts as the positive class
RANDOM_SEED = 42
ROLLING_FORM_WINDOW = 3     # last N races for "recent form" features
