# SurgeEgress: ML Surge Forecasting

Forecasting post-event rideshare demand surges (Uber/Lyft HVFHV) in New York City around **Barclays Center** (TLC Zone `189`, spillover `{49, 61, 62, 97, 181, 190}` at `0.25`) and **Citi Field** (TLC Zone `253`, spillover `{93}` at `0.75`) using NYC TLC trip records, event schedules, and Open-Meteo weather data.

---

## Project Overview

When major concerts or sporting events conclude, surrounding streets experience sharp rideshare demand surges above normal urban traffic. SurgeEgress models the **Surge Delta** ($\Delta y_t$):

$$\Delta y_t = y_t - y_{\text{baseline}}(t)$$

Where:
* $y_t$ (`y_actual`) is the spatially weighted 15-minute pickup demand anchored on `request_datetime` (with `pickup_datetime` fallback).
* $y_{\text{baseline}}(t)$ (`y_baseline`) is the clean-week counterfactual median for the same venue, day of week, and 15-minute slot across the 4 most recent non-event, non-holiday weeks (US federal + 2024–2025 NYC DOE school holidays).

---

## Repository Structure

```text
ml-surge-forecasting/
├── CONTRACTS.md                  # Locked handoff schemas (Steps 1-4)
├── scripts/
│   ├── run_pipeline.py           # One-command Step 1 data pipeline runner
│   └── dev/                      # Diagnostic inspection scripts
├── src/
│   ├── config.py                 # Canonical zones, weights, paths, timestamp anchors
│   ├── ingestion/
│   │   ├── tlc.py                # Monthly 2024-2025 TLC HVFHV downloader + PyArrow pushdown
│   │   ├── weather.py            # Open-Meteo hourly archive -> 15-min buckets
│   │   ├── citi_field_ingest.py  # 2024-2025 Mets games (MLB Stats API) + concerts
│   │   ├── barclays_discovery.py # Barclays Center event URL discovery
│   │   ├── barclays_ingest.py    # Barclays Center event scraper
│   │   ├── box_scores.py         # ESPN NBA/WNBA/NCAAM/WCBB buzzer ends & attendance
│   │   └── tour_boxscores.py     # Wikipedia + Touring Data concert attendance
│   ├── processing/
│   │   ├── clean_tlc.py          # Telematics/economic hygiene & timestamp fallback
│   │   ├── calibration.py        # Empirical category end-time calibration
│   │   └── events.py             # Dual-venue Event compiler & holiday flag helpers
│   ├── features/
│   │   ├── features.py           # Continuous 15-min grid, spatial weighting, lags
│   │   ├── baseline.py           # Clean-week baseline & surge_delta target
│   │   └── dataset.py            # Grouped 65/15/20 split -> event & model datasets
│   └── models/
│       ├── train.py              # Step 2: Ridge, LightGBM, XGBoost training
│       └── evaluate.py           # Step 2: RMSE, MAE, SHAP & artifact export
├── notebooks/                    # Exploratory & verification notebooks
├── tests/                        # Unit tests & offline HTML/JSON fixtures
└── docs/
    ├── PRD.md                    # Project requirements & validation strategy
    ├── GUIDE.md                  # Step-by-step execution walkthrough
    ├── barclays_scraper.md       # Barclays crawler manual & audit notes
    └── merge-pipeline-summary.md # Event-ridership merge decisions
```

---

## Quickstart

### 1. Environment Setup (One-Time)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Data Pipeline (Step 1)

**One-Command Full Pipeline** (ingests 2024–2025 TLC trips with `--delete-raw`, weather, Citi Field, Barclays, ESPN/tour box scores, cleans trips, and builds `data/processed/model_dataset.parquet`):
```bash
python -m scripts.run_pipeline
```

**Fast Rebuild from Cached Interim Files** (~5 seconds when raw/interim files are already present):
```bash
python -m scripts.run_pipeline --skip-tlc --skip-scrape
```

<details>
<summary>Modular individual commands (if running steps separately)</summary>

```bash
python -m src.ingestion.tlc --years 2024 2025 --delete-raw
python -m src.ingestion.weather
python -m src.ingestion.citi_field_ingest --years 2024 2025
python -m src.ingestion.barclays_discovery --year 2024
python -m src.ingestion.barclays_ingest --year 2024
python -m src.ingestion.barclays_discovery --year 2025
python -m src.ingestion.barclays_ingest --year 2025
python -m src.ingestion.box_scores
python -m src.ingestion.tour_boxscores --years 2024 2025
python -m src.processing.clean_tlc
python -m src.features.features
python -m src.features.dataset
```

</details>

### 3. Contract 1 Dataset Output

`src.features.dataset` produces both `data/processed/event_dataset.parquet` and the Contract 1 modeling handoff table `data/processed/model_dataset.parquet`:
- **Coverage:** **492 events** (**3,936 rows**, 8 buckets per event from `-30` to `+75` min relative to `event_end`) across **Barclays Center** (`venue_zone_id = 189`: 323 events / 2,584 rows) and **Citi Field** (`venue_zone_id = 253`: 169 events / 1,352 rows).
- **Validation Split:** Grouped chronologically by `event_start` (**~65% Train / 15% Validation / 20% Test** — 320 train / 74 val / 98 test events), keeping each event curve together. Forecast-time input limitations are documented in the modeling guide below.
- See [CONTRACTS.md](CONTRACTS.md) for exact column types and downstream interfaces.

### 4. Run Tests

```bash
python -m unittest discover -s tests -v
# or
python -m pytest -q
```

### 5. Modeling and inference

See [modeling-preparation.md](docs/modeling-preparation.md) for reproducible
training, frozen evaluation/export commands, the macOS OpenMP workaround, and
backend inference inputs. LightGBM is primary; Ridge is retained for comparison.
Contract 2 artifacts live in `data/artifacts/` and contain test-only predictions
for July–December 2025. The eight available buckets end at +75, so the locked
nine-bucket/+90 requirement remains an integration limitation.

---

## Roadmap (Steps 2–4, per [CONTRACTS.md](CONTRACTS.md))

1. **Step 1 (Complete):** Dual-venue ingestion, cleaning, clean-week baseline, and `data/processed/model_dataset.parquet`.
2. **Step 2 (Modeling implemented):** Control-median, Ridge and LightGBM comparison, frozen test evaluation, Contract 2 exports, and reusable inference. See the modeling guide above for results and remaining interface limitations.
3. **Step 3 (API — `src/api.py`):** Serve FastAPI endpoints `GET /events`, `GET /demands/{event_id}`, and `POST /simulate` (Contract 3).
4. **Step 4 (Demo — `app.py`):** Interactive Streamlit surge curve dashboard backed by Contract 3.
