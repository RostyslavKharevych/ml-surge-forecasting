# SurgeEgress: ML Surge Forecasting

Forecasting post-event rideshare demand surges (Uber/Lyft) in New York City around Barclays Center (TLC Zone 189) and Citi Field (TLC Zone 253) using TLC High-Volume FHV records and scheduled event metadata.

---

## Project Overview

When major concerts or sporting events conclude, surrounding streets experience sharp rideshare demand surges that deviate significantly from typical traffic patterns. This project models the **Surge Delta** ($\Delta D_t$):

$$\Delta D_t = D_t - B_t$$

Where:
* $D_t$ is the spatially weighted 15-minute pickup demand across core and spillover taxi zones.
* $B_t$ is the historical clean-week counterfactual baseline for the identical zone, day of week, and 15-minute window.

---

## Repository Structure

```text
ml-surge-forecasting/
├── src/
│   ├── config.py             # Global constants, taxi zone IDs, spatial weights
│   ├── ingestion/            # Raw data collection (TLC parquet, Barclays crawler, APIs)
│   │   ├── tlc.py            # NYC TLC HVFHV batch downloader
│   │   ├── barclays_discovery.py
│   │   ├── barclays_ingest.py
│   │   ├── event_categories.json
│   │   ├── box_scores.py     # ESPN NBA/WNBA final buzzers & official attendance
│   │   ├── tour_boxscores.py # Concert tour box offices & attendance
│   │   └── weather.py        # Open-Meteo historical weather archive
│   ├── processing/           # Hygiene & harmonization
│   │   ├── clean_tlc.py      # TLC telemetry filtering & row-level audit
│   │   ├── calibration.py    # Empirical event runtimes from demand curves
│   │   ├── events.py         # Harmonized event schedule compiler
│   │   ├── event_overrides.csv
│   │   └── event_runtimes.csv
│   ├── features/             # Feature matrix & target formulation
│   │   ├── features.py       # Continuous 15m grid, spatial weighting, rolling lags
│   │   ├── baseline.py       # Clean-week baseline ($B_t$) and surge delta ($\Delta D_t$)
│   │   └── dataset.py        # Event x Prediction Window training matrix
│   └── models/               # Benchmark baselines & ML regressors
│       ├── train.py          # LightGBM, XGBoost, and Ridge regressors
│       └── evaluate.py       # Metrics (RMSE, MAE) and holdout validation
├── notebooks/                # EDA & step-by-step verification notebooks
├── tests/                    # Unit tests & offline fixtures
└── docs/                     # Technical documentation & ADRs
    ├── GUIDE.md              # Full pipeline reproduction guide
    ├── barclays_scraper.md   # Barclays Center crawler manual & edge cases
    └── merge-pipeline-summary.md
```
---

# Quickstart

## 1. Environment Setup

``` Bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run Test Suite

``` Bash
python -m pytest -q
```

## 3. Pipeline Execution

To reproduce the full pipeline from raw data to the final ML training matrix, follow the instructions in docs/GUIDE.md.

For detailed documentation on the Barclays event crawler and extraction audits, refer to docs/barclays_scraper.md.