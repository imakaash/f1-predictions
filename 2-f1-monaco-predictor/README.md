# 🇲🇨 F1 Monaco GP Podium Predictor

Open-source machine learning pipeline that predicts the **top-3 finishers** of the Formula 1 Monaco Grand Prix at Circuit de Monaco, Monte Carlo.

Built on:
- **[FastF1](https://github.com/theOehrly/Fast-F1)** — open-source library that pulls from F1's official live timing API (results, laps, telemetry, weather)
- **scikit-learn**, **XGBoost**, **LightGBM** — gradient-boosted and linear models
- **Jolpica-F1** — Ergast-compatible historical results API (auto-used by FastF1)

## What it does

The pipeline:

1. **Pulls historical race data** for past Monaco GPs (2018 → present) plus the current season's races so far — including FP1/FP2 sessions, qualifying telemetry, and race-day weather.
2. **Engineers features**:
   - Driver form: rolling avg finish, qualifying pace, points, DNF rate
   - FP1/FP2 long-run pace delta vs field median
   - Race-day weather: rain flag, air temperature, track temperature
   - Qualifying telemetry: per-driver top speed vs field median
   - Track-specific history: past Monaco starts, avg finish, podiums — for driver and team
3. **Trains and compares 5 models**:
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost
   - LightGBM (binary podium classifier)
   - **LightGBM LambdaRank** — learning-to-rank on full finishing order
4. **Evaluates** with time-aware leave-one-year-out CV — no data leakage from future races.
5. **Predicts the 2026 Monaco GP** — outputs podium probabilities (classifiers) or full finishing-order scores (LambdaRank), picks the best model by top-3 hit rate.

## Sample output

![2026 Monaco GP podium prediction](assets/monaco_2026_prediction.png)

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Pull data + train + predict (one command)
python -m src.run_pipeline

# Or step-by-step:
python -m src.fetch_data        # downloads & caches race data
python -m src.build_features    # builds the training dataset
python -m src.train             # trains all 5 models, picks the best
python -m src.predict           # outputs the predicted podium

# After qualifying — pass real grid for a sharper prediction:
python -m src.predict --qual-csv data/monaco_2026_qual.csv
```

## Project structure

```
2-f1-monaco-predictor/
├── src/
│   ├── config.py          # constants: years, track name, season rounds
│   ├── fetch_data.py      # FastF1 data fetching + enrichment
│   ├── build_features.py  # feature engineering
│   ├── train.py           # model training + comparison
│   ├── predict.py         # produces the 2026 prediction
│   └── run_pipeline.py    # end-to-end runner
├── assets/                # screenshots / visuals
├── data/                  # parquet + csv files (git-ignored, created at runtime)
├── models/                # trained model artifacts (git-ignored)
├── cache/                 # FastF1's API cache (git-ignored)
└── requirements.txt
```

## Notes on the approach

- **Qualifying is everything at Monaco.** Circuit de Monaco is the narrowest circuit on the calendar — overtaking is almost impossible outside of pit-stop strategy. `qual_position` and `grid` carry far more weight here than at most other circuits. Passing real qualifying results via `--qual-csv` is especially important.
- **Weather matters more here.** Monaco can receive sudden rain during the race and the tunnel section means cars transition between wet and dry instantly. The `is_wet` feature carries strong signal for this circuit.
- **Track history is reliable.** Monaco is one of the few circuits where driver familiarity is a measurable advantage — the barriers are unforgiving and experience of the circuit layout shows up clearly in the historical data.
- **Time-aware split**: We never train on a race that happened *after* the race we're testing on. This is critical and easy to get wrong.
- **2026 caveat**: This is the first year of major regulation changes. Models trained primarily on 2026 data will be most reliable; use earlier seasons for *track*-specific signal only.

## Improving the model

1. Pull **full-season race history** for form features — currently form is approximated using Monaco-only history; pulling every race of every season would give much richer rolling-form signals.
2. Add a **Monte Carlo simulation** that samples DNF probability per driver to produce finishing-order distributions rather than point estimates.
3. **Ensemble** the binary classifier and LambdaRank scores — combining podium probability with rank score often outperforms either alone.
4. Add **FP3 / Sprint Qualifying pace** on sprint weekends where FP2 is replaced.
5. Feed in a **weather forecast** for race day before sessions run — Monaco is famously prone to sudden showers.
