# Project Specification & Learning Plan: Urban Mobility & Event Surge Dynamics

**Project Name:** SurgeEgress  
**Team:** Jacky Ye, Rajit Ahmed, Rostyslav Kharevych, Thiri Myat Noe  
**Course:** Data Science (CUNY Tech Prep — Cohort 12)  
**Term:** Fall 2026  

---

## 1. Project Idea & Learning Goals

### High-Level Concept
When a major sporting event or concert ends in New York City, how much extra rideshare demand hits the surrounding streets, and how long does that surge persist? This project quantifies post-event rideshare demand spikes around major NYC venues, identifies which event characteristics drive the sharpest surges, and models the underlying exit dynamics.

NYC hosts large-scale events nearly every week on top of baseline urban traffic. Forecasting the magnitude and decay curve of post-event demand allows rideshare platforms and drivers to position supply proactively—reducing passenger wait times and localized street congestion.

### Core Data Science Objective
* **Task:** Supervised time-series regression to forecast net excess rideshare pickups across 15-minute intervals surrounding an event (targeting the sharp exit peak within the first 45–60 minutes post-event).
* **Target Variable ($\Delta y_t$):** **Surge Delta**, defined as the net excess High-Volume For-Hire Vehicle (HVFHV) pickups in a venue's primary TLC Taxi Zone above an expected counterfactual baseline:

$$\Delta y_t = y_t - y_{\text{baseline}}(t)$$

Where:
* $y_t$ is the observed HVFHV pickup count in 15-minute bucket $t$.
* $y_{\text{baseline}}(t)$ is the historical median pickup volume for the exact same zone, day of the week, and 15-minute time slot across verified non-event control weeks.

### Audience & Stakeholders
* **Engineering Team (Peers):** Collaborating via GitHub feature branches with shared utility modules for baseline calculation and data ingestion.
* **Course Instructors & Evaluators:** Evaluating methodological rigor, leakage prevention, feature engineering rationale, and model interpretability.
* **General Public / Commuters:** Interactive visualizations illustrating post-event surge curves (inspired by [subway.fyi/awfulness](https://subway.fyi/awfulness)).

---

## 2. Problem Statement & Motivation

### Real-World Scenario
A Tuesday afternoon baseball game at Citi Field and a sold-out Saturday night concert at Barclays Center generate fundamentally different street-level demand profiles, yet static historical averages treat them identically. Surge magnitude and duration depend heavily on event type, attendance, day of week, and weather.

### Why Simple Methods Fall Short
* **Event Hype & Attendance Variance:** A sold-out concert or playoff game produces a far sharper exit spike than a midweek regular-season game.
* **Non-Uniform Exit Behavior:** Crowds rarely exit at a single moment—blowout games trigger gradual early departures, whereas tight games or encores produce concentrated, delayed surges.
* **Exogenous Shocks:** Rain, overlapping local events, and transit alternatives distort normal pickup patterns in ways simple day-of-week averages cannot capture.

### Technical & Modeling Challenges
1. **Spatial Resolution & Spillover:** NYC TLC records are aggregated to polygon Taxi Zones rather than exact GPS coordinates. We must account for demand spilling over into adjacent zones (especially around dense transit hubs) without losing signal.
2. **Non-Linear Feature Engineering:** Engineering domain-specific features—such as attendance proxies, relative time-to-event-end ($t - t_{\text{end}}$), and weather interactions—so tree-based models learn true exit curves rather than memorizing calendar averages.

---

## 3. Project Scope

### In-Scope
* **Anchor Venues (2024–2025):**
  * **Citi Field / Flushing Meadows (Primary Zone `253` | Spillover `93`):** An outdoor stadium with dedicated parking and rideshare staging lots.
  * **Barclays Center / Atlantic Ave (Primary Zone `189` | Spillover `49, 61, 62, 97, 181, 190`):** A dense indoor arena atop a major subway hub with heavy background street traffic.
* **Time Granularity:** 15-minute intervals spanning $t_{\text{end}} - 30\text{ min}$ to $t_{\text{end}} + 90\text{ min}$ (capturing the pre-end trickle and the 45–60 minute post-event exit curve).
* **Candidate Models:**
  1. **Baseline 0 (Naive Persistence):** $`\hat{y}_t = y_{t-1}`$
  3. **Baseline 1 (Control Median):** Historical non-event day-of-week and time-slot median ($`y_{\text{baseline}}(t)`$)
  4. **Linear Benchmark:** Ridge Regression with lag, calendar, and weather features
  5. **Gradient Boosted Trees:** LightGBM and XGBoost Regressors to capture non-linear interactions

### Out-of-Scope
* **Real-Time Production Ingestion:** No live Ticketmaster scraping or streaming inference pipelines; all modeling runs on curated 2024–2025 historical batches.
* **Heavy Deep Learning:** Excluding LSTMs and Spatio-Temporal Graph Neural Networks (ST-GCNs) in favor of faster, interpretable tabular models.
* **City-Wide Modeling:** Scoped strictly to our two anchor venues and their adjacent spillover zones rather than all 260+ NYC Taxi Zones.
* **Subway Turnstile Modeling:** MTA turnstile data is logged on coarser intervals and deferred to future work.

---

## 4. Datasets & Data Preparation

### 4.1 Primary Trip Dataset
* **Source:** NYC TLC High-Volume For-Hire Vehicle (HVFHV) Trip Records (2024–2025 monthly Parquet files via NYC Open Data).
* **Key Fields:** `pickup_datetime`, `PULocationID`, `DOLocationID`, `trip_miles`, `trip_time`.
* **Preprocessing Pipeline:**
  * Stream/filter monthly Parquet files by target `PULocationID` using `pyarrow` predicate pushdown to prevent out-of-memory errors.
  * Remove invalid records (`trip_time <= 0`, `trip_miles <= 0`, or out-of-range timestamps).
  * Aggregate valid trips into 15-minute pickup counts per Taxi Zone.

### 4.2 Supplementary Contextual Data
* **Curated Event Schedule (CSV):** 30–50 verified events across Citi Field and Barclays Center (2024–2025) compiled from MLB box scores (Mets), NBA box scores (Nets), and Setlist.fm (concerts).
* **Historical Weather (Open-Meteo API):** Hourly precipitation (`mm`), ambient temperature (`°F`), and wind speed (`mph`) matched to venue coordinates.
* **Holiday Calendar:** Binary flags for US federal holidays and NYC public school holidays.

### 4.3 Feature Engineering & Baseline
* **Cyclical & Calendar Features:** `hour_of_day`, `day_of_week`, `is_weekend`, `is_holiday`, and cyclical encodings ($\sin/\cos$ transforms of hour).
* **Autoregressive & Spatial Signals:** Lagged target-zone pickups ($y_{t-1}, y_{t-2}$) and concurrent adjacent-zone pickup counts to capture crowd momentum and spillover.
* **Counterfactual Baseline ($y_{\text{baseline}}(t)$):** Median pickup volume for the exact zone, day of week, and 15-minute slot across the preceding 3–4 non-event, non-holiday control weeks.

### Table 1: Planned Feature Schema

| Feature Name | Type | Description / Role |
| :--- | :--- | :--- |
| `event_id` | `string` | Unique event key (e.g., `METS_20240712`, `CONCERT_BARC_20241005`) |
| `venue_zone_id` | `int` | Primary TLC Taxi Zone ID for the venue |
| `time_bin` | `timestamp` | Start timestamp of the 15-minute aggregation bucket |
| `event_type` | `category` | Event category (`baseball`, `basketball`, `concert`, `other`) |
| `start_time` | `timestamp` | Scheduled event start time |
| `end_time` | `timestamp` | Verified actual event end time (final out, buzzer, or lights-up) |
| `mins_since_end` | `int` | Relative offset in minutes ($t - t_{\text{end}}$) |
| `attendance_est` | `int` | Box score attendance or venue capacity proxy |
| `neighbor_pickups` | `int` | Aggregate pickups across adjacent spillover zones in bucket $t$ |
| `precipitation_mm` | `float` | Hourly rainfall accumulation (`mm`) from Open-Meteo |
| `temp_f` | `float` | Outdoor ambient temperature (`°F`) |
| `baseline_pickups` | `float` | Historical non-event median pickups ($y_{\text{baseline}}(t)$) |
| `surge_delta` *(Target)* | `float` | Net excess pickups above baseline ($\Delta y_t = y_t - y_{\text{baseline}}(t)$) |

---

## 5. Evaluation & Success Criteria

### Metrics
* **RMSE (Primary Optimization Metric):** Penalizes large errors during sharp post-event peak intervals.
* **MAE (Primary Interpretability Metric):** Measures average forecast error in raw pickup counts.
* **$R^2$ / MAPE:** Secondary metrics for variance explained and relative percentage error.

### Validation Strategy
* **Grouped Chronological Event Split (No Random Shuffling / No Data Leakage):**
  * Held events with valid baselines are sorted chronologically by `event_start` and partitioned at the **event** level (~**65% Train / 15% Validation / 20% Test**), keeping all 15-minute buckets of a given `event_id` in the same split.
  * This replaces fixed 2024/2025 calendar cutoffs so both MLB (spring–fall) and NBA/concert seasons are represented in training while strictly preserving temporal order.
* **Cross-Venue Generalization Check:** Evaluate how well a model trained primarily on Citi Field transfers learned parameters (attendance scaling, weather sensitivity) to Barclays Center.

### Success Benchmarks
* Complete an end-to-end reproducible pipeline with both baseline and gradient-boosted models.
* Achieve a **15%–20% reduction in RMSE** over the non-event historical baseline ($T - 7$ / control median).
* Surface top 3–5 predictive drivers using SHAP / feature importance analysis.

---

## 6. Project Milestones

### Table 2: Semester Roadmap

| Milestone | Key Deliverables |
| :--- | :--- |
| **1. Data Ingestion & Cleaning** | PyArrow TLC ingestion script, curated event CSV, weather join |
| **2. EDA & Feature Engineering** | Surge decay visualizations, lag/cyclical features, baseline benchmarks |
| **3. Model Training & Tuning** | Ridge & LightGBM/XGBoost pipelines, chronological validation, SHAP analysis |
| **4. Demo App & Final Report** | Interactive surge visualization app, final methodology report, presentation |

---

## 7. System Requirements & Demo API Design

### Functional Requirements
1. **Data Ingestion & Merging:** Automated scripts to ingest and join NYC TLC Parquet records, curated venue schedules, and Open-Meteo weather data.
2. **Temporal Bucketing:** Aggregate raw trip timestamps into 15-minute intervals aligned to relative event windows ($t_{\text{end}} - 30$ to $t_{\text{end}} + 90$).
3. **Surge Computation:** Compute counterfactual baselines and output $\Delta y_t$ actuals alongside model predictions.

### Demo API Endpoints
* `GET /events` — List all tracked events (supports filtering by `venue`, `event_type`, and `date_range`).
* `GET /demands` — Retrieve observed 15-minute pickup counts and historical baselines for a given `event_id` or `time_bin`.
* `GET /predictions` — Retrieve predicted $\Delta y_t$ surge curves for specified events or hypothetical scenario inputs.
