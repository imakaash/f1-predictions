# F1 ML Projects

A collection of Formula 1 machine learning projects using open-source data from the [FastF1](https://github.com/theOehrly/Fast-F1) library and the Jolpica/Ergast F1 API.

## Projects

| Project | Description |
|---|---|
| [1-f1-canada-predictor](1-f1-canada-predictor/README.md) | Predicts the top-3 finishers of the Canadian GP using XGBoost, LightGBM, Random Forest, and Logistic Regression |

## Repository structure

```
f1/
├── environment.yml               # shared conda environment (name: f1)
└── 1-f1-canada-predictor/        # Canadian GP podium prediction pipeline
    ├── src/                      # pipeline modules
    ├── notebooks/                # exploratory notebooks
    ├── data/                     # parquet files (git-ignored, created at runtime)
    ├── models/                   # trained model artifacts (git-ignored)
    ├── cache/                    # FastF1 API cache (git-ignored)
    └── requirements.txt          # pip fallback
```

## Setup

```bash
# Create and activate the conda environment (only needed once)
conda env create -f environment.yml
conda activate f1

# Run the Canadian GP prediction pipeline
cd 1-f1-canada-predictor
python -m src.run_pipeline
```

See each project's own README for a detailed usage guide.

## Data sources

- **FastF1** — wraps F1's official live-timing API; provides lap times, telemetry, session results, and weather
- **Jolpica-F1 / Ergast** — historical race results back to 2018 (auto-used by FastF1)

Data is fetched and cached locally on first run. No API key is required.