# ML Surge Forecasting

Tabular time-series regression forecasting engine to predict granular demand surges and net deltas around key event locations.

## Project Structure

```text
├── data/
│   ├── raw/            # Untouched raw data files (ignored by Git)
│   ├── interim/        # Intermediary aggregated tables
│   └── processed/      # Cleaned feature tables ready for modeling
├── notebooks/          # Exploratory analysis & presentation figures
├── src/                # Modular pipeline scripts
│   ├── config.py       # Global paths, constants, and column names
│   ├── ingest.py       # Raw file acquisition
│   ├── clean.py        # Schema enforcement & parsing
│   ├── features.py     # Lags, rolling windows, calendar indicators
│   ├── baseline.py     # Naive median / control comparisons
│   ├── train.py        # Model training (LightGBM, XGBoost, Ridge)
│   └── evaluate.py     # Metrics (MAE, RMSE, R2) & SHAP attribution
├── tests/              # Sanity checks and unit tests
└── requirements.txt    # Pinned environment dependencies

