# SurgeEgress — Locked Handoff Contracts

This document defines the three locked interfaces enabling parallel execution across Data Engineering (Person 1), Modeling (Person 2), Backend API (Person 3), and Frontend Demo (Person 4). Do not alter column names, types, or endpoint shapes without team sign-off.

---

## Contract 1: Person 1 -> Person 2 (`data/processed/model_dataset.parquet`)

Upstream intermediate event–ridership dataset (`src/features/dataset.py`) writes `data/processed/event_dataset.parquet`. The modeling handoff table `data/processed/model_dataset.parquet` guarantees these exact columns and types:

| Column | Type | Description |
| :--- | :--- | :--- |
| `event_id` | `str` | Unique event identifier (e.g., `CITI_FIELD_20240712_1910`, `BARCLAYS_20240322_1715`) |
| `venue_zone_id` | `int` | Primary TLC Taxi Zone ID (`253` for Citi Field, `189` for Barclays Center) |
| `time_bin` | `datetime64` | Start timestamp of the 15-minute bucket (naive `America/New_York`) |
| `mins_since_end` | `int` | Relative offset in minutes from `event_end` (`-30` through `+90`) |
| `event_type` | `str` | Event category (`baseball`, `basketball`, `concert`, etc.) |
| `attendance_est` | `int` | Verified or estimated event attendance |
| `precipitation_mm` | `float` | Hourly rainfall accumulation (`mm`) from Open-Meteo |
| `temp_f` | `float` | Outdoor ambient temperature (`°F`) |
| `y_actual` | `int` | Observed pickup demand in the bucket |
| `y_baseline` | `float` | Counterfactual non-event, non-holiday historical median pickup count |
| `surge_delta` | `float` | Target variable ($\Delta y_t = y_{\text{actual}} - y_{\text{baseline}}$) |

---

## Contract 2: Person 2 -> Person 3 & 4 (`data/artifacts/`)

All model outputs consumed by the API and dashboard live under `data/artifacts/`:

1. **`data/artifacts/model.joblib`**
   - Serialized trained LightGBM regressor paired with the exact ordered `feature_names` list expected at inference time.
2. **`data/artifacts/predictions_2025.parquet`**
   - Holdout predictions with required columns:
     - `event_id`, `time_bin`, `mins_since_end`, `y_actual`, `y_baseline`, `surge_delta_actual`, `surge_delta_pred`, `y_total_pred`
3. **`data/artifacts/metrics.json`**
   - Evaluation summary JSON with required keys:
     - `baseline_rmse`, `ridge_rmse`, `lgbm_rmse`, `rmse_reduction_pct`, `top_features`

---

## Contract 3: Person 3 <-> Person 4 (FastAPI JSON)

The FastAPI service (`src/api.py`) exposes three JSON endpoints consumed by the Streamlit demo (`app.py`):

1. **`GET /events`**
   - Returns a list of event metadata dictionaries (supporting filtering by venue, event type, and date).
2. **`GET /demands/{event_id}`**
   - Returns the 9-bucket egress curve (`mins_since_end` from `-30` to `+90`) for the requested `event_id`, where each bucket includes:
     - `mins_since_end`, `baseline`, `actual`, `predicted`
3. **`POST /simulate`**
   - Accepts scenario inputs: `attendance`, `is_raining`, `is_weekend`, `event_type`.
   - Returns the 9-bucket predicted egress surge curve across `-30` to `+90`.
