# 🇨🇦 F1 Canadian GP Podium Predictor

Open-source machine learning pipeline that predicts the **top-3 finishers** of the Formula 1 Canadian Grand Prix at Circuit Gilles-Villeneuve in Montreal.

Built on:
- **[FastF1](https://github.com/theOehrly/Fast-F1)** — open-source library that pulls from F1's official live timing API (results, laps, telemetry, weather)
- **scikit-learn**, **XGBoost**, **LightGBM** — gradient-boosted and linear models
- **Jolpica-F1** — Ergast-compatible historical results API (auto-used by FastF1)

## What it does

The pipeline:

1. **Pulls historical race data** for past Canadian GPs (2018 → present) plus the current season's races so far.
2. **Engineers features** like driver form (rolling avg finish), qualifying pace, team form, track-specific historical performance, grid position, and DNF rate.
3. **Trains and compares 4 models** for podium prediction:
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost
   - LightGBM
4. **Evaluates** with a time-aware holdout (no data leakage from future races).
5. **Predicts the 2026 Canadian GP podium** — outputs each driver's probability of a top-3 finish.

## Sample output

![2026 Canadian GP podium prediction](assets/canada_2026_prediction.png)

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Pull data + train + predict (one command)
python -m src.run_pipeline

# Or step-by-step:
python -m src.fetch_data        # downloads & caches race data
python -m src.build_features    # builds the training dataset
python -m src.train             # trains all 4 models, picks the best
python -m src.predict           # outputs the predicted podium
```

## Project structure

```
f1-canada-predictor/
├── src/
│   ├── config.py          # constants: years, track name, drivers
│   ├── fetch_data.py      # uses FastF1 to pull session data
│   ├── build_features.py  # feature engineering
│   ├── train.py           # model training + comparison
│   ├── predict.py         # produces the 2026 prediction
│   └── run_pipeline.py    # end-to-end runner
├── data/                  # parquet files (created by fetch_data)
├── models/                # trained model artifacts
├── cache/                 # FastF1's API cache
└── requirements.txt
```

## Notes on the approach

- **Why podium classification?** Predicting an exact finishing order is extremely hard given DNFs, safety cars, and weather. Predicting a binary "did this driver finish top-3" is more tractable and what most public F1 ML projects target.
- **Why these features?** F1 races are mostly decided by car pace + qualifying position. Recent form (last 3 races) is the strongest signal — drivers in cars on a hot streak tend to keep producing.
- **Time-aware split**: We never train on a race that happened *after* the race we're testing on. This is critical and easy to get wrong.
- **2026 caveat**: This is the first year of major reg changes. Models trained primarily on 2026 data will be most reliable; use earlier seasons for *track*-specific signal only.

## Improving the model

Ideas, roughly in order of bang-for-buck:

1. Add **practice session pace** (FP1/FP2/FP3 long-run averages) — available via FastF1
2. Add **weather forecast** for race day (Montreal is famously rain-prone)
3. Use **telemetry-derived features**: top speed on the back straight, sector dominance
4. Add a **Monte Carlo simulation** that samples DNF probability per driver
5. Try a **listwise learning-to-rank** model (LightGBM `lambdarank`) on full finishing order, not just podium

## License

MIT. FastF1 and the underlying F1 data are unofficial and not associated with Formula 1 companies.
