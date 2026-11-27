# Modeling and artifact handoff

Run from the repository root. Obtain the existing
`data/processed/model_dataset.parquet` from the data teammate; do not rebuild it
for modeling. Its SHA256 and package versions are recorded in
`data/artifacts/metadata.json`. Install `requirements.txt` in a Python 3.12
environment; match the metadata versions when reproducing/loading these bundles.
No modeling dependencies were added beyond the existing requirements.

```bash
conda activate surge-ml
python -m src.models.train     # training/validation search only
python -m src.models.holdout   # frozen test evaluation and local figures/CSVs
python -m src.models.export    # frozen training-only fit and Contract 2 exports
python -m pytest -q -rs
```

For the PRD's supplemental persistence/R² evaluation, run
`python -m src.models.evaluate`. This loads existing `model.joblib` and optional
`ridge.joblib`; it never fits, tunes or writes artifacts. Share the ignored Ridge
bundle separately to reproduce this comparison without retraining.

On macOS, if LightGBM cannot find `libomp.dylib`, this environment has a compatible
copy bundled with sklearn. After activating the environment, set its directory
before running the commands above (no hardcoded user path):

```bash
export DYLD_LIBRARY_PATH="$(python -c 'from pathlib import Path; import sklearn; print(Path(sklearn.__file__).parent / ".dylibs")')${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Check that this directory contains `libomp.dylib`. If it does not, install the
OpenMP runtime appropriate for your platform/environment. This workaround was
verified on this Mac; the system runtime may already work on other machines.

## Preparation and fixed models

- Reconstruct event starts from canonical IDs and use the upstream chronological
  65/15/20 rule: 320/74/98 events, 2,560/592/784 rows. Training ends April 18, 2025;
  validation ends July 19; test begins July 20. Curves stay together. Duplicate
  keys, incomplete curves, tied boundary starts and overlapping boundary windows
  are rejected. Training and validation never include test rows.
- Fit attendance medians on training only: venue/type, then venue, then global.
  Add `attendance_missing`. Fit scaling and one-hot categories on training only;
  unseen categories encode as zero indicators. No target/outlier clipping.
- Predictors: relative minutes, attendance and missingness, precipitation,
  temperature, rain (>0.1 mm), sine/cosine time of day, weekday, Friday–Sunday
  weekend, federal/NYC-school holiday, month, one-hot venue and event type.
  This yields 30 encoded columns. IDs, actual demand, baseline, target, observed
  lags and concurrent demand are excluded. Target is `surge_delta`.
- Ridge selected alpha=100 from 0.1/1/10/100. LightGBM selected seven leaves,
  minimum 64 rows/leaf and 250 trees from eight validation configurations.
  Fixed settings: learning_rate=0.03, max_depth=4, reg_alpha=1, reg_lambda=10,
  seed=42, deterministic single-thread execution. No train+validation refit.
  `holdout` and `export` use these frozen parameters without searching.

| Model | Train RMSE / MAE | Validation RMSE / MAE | Test RMSE / MAE | Test RMSE improvement |
| --- | ---: | ---: | ---: | ---: |
| Control median (zero delta) | 41.30 / 29.95 | 34.59 / 27.08 | 46.43 / 35.21 | — |
| Ridge | 30.95 / 22.37 | 26.24 / 20.98 | 33.29 / 23.93 | 28.29% |
| LightGBM | 26.52 / 18.45 | 22.24 / 16.69 | 33.98 / 23.63 | 26.81% |
| XGBoost (additional comparison) | 26.44 / 18.42 | 22.93 / 17.17 | 33.98 / 23.51 | 26.80% |

LightGBM remains primary for validation performance, peak reconstruction and
Contract 2. Ridge slightly leads test RMSE. Peak magnitude MAE is 33.83 for
LightGBM versus 36.49 for Ridge; peak timing MAE is 18.21 versus 46.53 minutes.
Ridge places peaks at -30 due to its decreasing linear curve. Peak metrics cover
all eight buckets; earliest maxima break ties and flat curves have no timing
score. Native LightGBM TreeSHAP passes an additivity check. Gain and mean absolute
SHAP are descriptive, not causal. Diagnostic plots select best/median/worst
LightGBM examples per venue, not a random sample.

## Artifacts and backend use

Track the three small Contract 2 files and `metadata.json` together in the PR.
Keep optional Ridge bundles, diagnostic CSVs/PNGs and cached data ignored; share
those separately or reproduce them with the commands above. No LFS is needed.

- `model.joblib`: dictionary containing `model` (LGBMRegressor), ordered
  `feature_names` (encoded names), and fitted `preprocessor` (sklearn Pipeline).
  Contract 2 does not specify a Python container; these are the backend keys.
  Load only trusted joblib files with matching package versions and this code.
- `predictions_2025.parquet`: exactly the eight Contract 2 columns; only 784 test
  rows for 98 events, July 20, 2025 14:00–December 29, 2025 23:15. The contract says
  holdout, not full-year coverage. No training/validation predictions are mixed in.
  `y_total_pred = y_baseline + surge_delta_pred`, with no clipping.
- `metrics.json`: exactly the five Contract 2 keys, using held-out test scores.
  `top_features` contains five encoded names ranked by training split gain:
  relative minutes, attendance, cosine time, temperature, month. The contract
  does not specify an item structure. Full gain/SHAP scores are optional CSVs.
- `metadata.json`: coverage, split provenance, frozen parameters, input SHA256,
  supported offsets, bundle conventions and package versions.

```python
from src.models.inference import load_bundle, predict_event_curve

bundle = load_bundle("data/artifacts/model.joblib")
curve = predict_event_curve(
    bundle, venue_zone_id=189, event_type="concert",
    estimated_event_end="2025-12-29 22:07", attendance_est=None,
    weather=weather_forecast,
)
```

Supply venue (189/253), event type, a naive America/New_York estimated end, expected
attendance or explicit None, and weather with exactly eight matching `time_bin`,
`precipitation_mm` (hourly mm), `temp_f` rows. The example floors end to 22:00 and
needs timestamps 21:30–23:15. Calendar/rain features are derived, never guessed.
An optional `baseline` table (`time_bin`, `y_baseline`) adds total demand; baseline
is not a predictor. `predict_rows` supports batch inputs. Both paths reuse fitted
preprocessing and validate feature order. Export checks exact reload/Parquet
prediction equality; `inference_examples.csv` contains two historical replays.

## Remaining limitations

- Eight buckets (-30 through +75), relative to floored end, differ from the locked
  +90/nine-bucket requirement. No +90 prediction is synthesized.
- No backend/simulator code exists in this checkout. Simulator fields cannot
  supply venue, dated end, temperature or rainfall amount; teammates need explicit
  context or an agreed interface change. No actual values exist for simulations.
- Archived weather, realized attendance and actual/calibrated ends make this a
  retrospective benchmark. Advance predictions need forecast-time inputs.
- 31/67 Barclays test events lack attendance (none at Citi). Missingness itself
  is influential and may encode category/source differences. Month-end 23:30
  near-zero demand is suspicious; October 31 is the worst plotted Barclays event.
  These anomalies remain unchanged. Both models underpredict large peaks.
- Splits differ seasonally; validation is baseball-heavy. Event buckets are
  correlated and overlapping events share outcomes. No uncertainty intervals are
  claimed. Naive timestamps have DST ambiguity; school holidays cover 2024–2025.

## Supplemental persistence and R² evaluation

Persistence is a rolling one-step benchmark: sort buckets within each event,
shift observed `y_actual` by one, and subtract the current `y_baseline` to predict
surge. No observations cross event boundaries. Exclude the first (-30) bucket
for every approach: validation has 518 eligible rows (74 events), test 686 (98).
Ridge and LightGBM still make advance predictions; only persistence can observe
previous demand during the curve, so its stronger scores reflect extra information.
The original eight-bucket results and locked artifact schemas remain unchanged.

| Split | Model | Eligible RMSE | Eligible MAE | RMSE improvement over eligible control |
| --- | --- | ---: | ---: | ---: |
| Validation | Control median | 34.94 | 27.36 | — |
| Validation | Ridge | 26.19 | 20.68 | 25.03% |
| Validation | LightGBM | 22.48 | 16.80 | 35.66% |
| Validation | XGBoost | 23.29 | 17.40 | 33.34% |
| Validation | Rolling persistence | 19.96 | 15.11 | 42.86% |
| Test | Control median | 47.68 | 36.22 | — |
| Test | Ridge | 33.99 | 24.08 | 28.71% |
| Test | LightGBM | 34.80 | 24.13 | 27.02% |
| Test | XGBoost | 34.93 | 24.10 | 26.74% |
| Test | Rolling persistence | 26.90 | 17.84 | 43.59% |

Full eight-bucket R² (validation/test): control median **−0.6881/−0.8482**,
Ridge **0.0280/0.0497**, LightGBM **0.3020/0.0100**. R² measures improvement over
the evaluated sample's mean, whereas our control predicts zero surge; beating
that control does not guarantee high R². R² is reported as undefined for constant
targets or fewer than two observations. The command also prints eligible-row R².
MAPE is unsuitable as a primary surge metric: division by zero is undefined,
near-zero targets inflate percentages, and negative excess demand makes percentage
interpretation misleading. RMSE/MAE remain primary; R² is supplementary.

## XGBoost comparison

Run `python -m src.models.train --xgboost` with the same environment/OpenMP setup.
This fits XGBoost only; it loads the existing Ridge/LightGBM bundles for comparison
and writes no artifacts. The ignored `ridge.joblib` must be supplied separately.
XGBoost is already in requirements.txt; this run used version 3.4.1.

Four validation candidates compare max_depth 2/3 and n_estimators 100/250, with
learning_rate=0.03, min_child_weight=20, reg_alpha=1, reg_lambda=10,
objective=reg:squarederror, tree_method=hist, seed=42 and one thread. Training-only
preprocessing is reused unchanged. Selected **depth 3 / 250 trees**, using
validation RMSE only; test inputs are scored after selection. No refit on validation.
Validation/test R² is **0.2581/0.0096**. Full-curve RMSE/MAE and the same seven-bucket
persistence comparison are included in the tables above.

| Peak metric | XGBoost validation | XGBoost test | LightGBM validation | LightGBM test |
| --- | ---: | ---: | ---: | ---: |
| Magnitude RMSE | 26.09 | 46.84 | 25.01 | 47.17 |
| Magnitude MAE | 19.68 | 33.54 | 18.57 | 33.83 |
| Timing MAE, minutes | 17.03 | 18.67 | 15.61 | 18.21 |

XGBoost is close to LightGBM on test, with slightly lower MAE and peak magnitude
error but weaker validation and peak timing. LightGBM remains primary. Existing
Ridge/LightGBM results, production bundles, inference and Contract 2 schemas are
unchanged. This comparison was requested after prior test results were known;
although selection uses validation only, the test set is no longer an untouched
basis for choosing a new primary model. No further search follows these results.

## Chronological cross-venue transfer and diagnostics

Run `python -m src.models.diagnostics` with the existing environment/OpenMP setup.
It trains source-only transfer models with the fixed selected parameters (no
search), loads existing pooled Ridge/LightGBM bundles for diagnostics, and reproduces
the fixed pooled XGBoost fit. CSVs go to ignored `reports/heldout/diagnostics/`.
No model artifacts, original evaluation files or observations are changed.

Use only each source venue's original training events; target validation/test
curves occur strictly after all source training curves. Preprocessing fits only
source data. Unseen venue/type indicators become zeros; missing target attendance
uses source-only medians. This is an exploratory transfer check with previously
selected pooled hyperparameters, not a target-blind tuning experiment: those
settings were originally selected using pooled validation. No target observations
are used to fit these transfer models or their preprocessing.

| Direction | Source events / rows | Source missing attendance | Target validation events / rows (missing) | Target test events / rows (missing) |
| --- | ---: | ---: | ---: | ---: |
| Barclays → Citi | 223 / 1,784 | 91 | 41 / 328 (0) | 31 / 248 (0) |
| Citi → Barclays | 97 / 776 | 2 | 33 / 264 (14) | 67 / 536 (31) |

Barclays source buckets end April 17, 2025; Citi source ends April 18. Target
validation starts April 19/26 respectively; test starts July 20. All 41/31 Citi
evaluation events have unseen types for the Barclays fit; 17/34 Barclays events
have unseen types for the Citi fit. Venue indicators are unseen in both directions.

Cells below show **RMSE / MAE / R²**:

| Direction | Model | Validation | Test |
| --- | --- | ---: | ---: |
| Barclays → Citi | Control median | 36.43 / 29.74 / −1.658 | 49.81 / 38.12 / −1.186 |
| Barclays → Citi | Ridge | 46.65 / 42.05 / −3.356 | 48.48 / 43.62 / −1.072 |
| Barclays → Citi | LightGBM | 21.77 / 16.87 / 0.051 | 31.91 / 22.75 / 0.103 |
| Barclays → Citi | XGBoost | 23.66 / 18.62 / −0.120 | 31.37 / 23.41 / 0.133 |
| Citi → Barclays | Control median | 32.14 / 23.77 / −0.225 | 44.78 / 33.86 / −0.725 |
| Citi → Barclays | Ridge | 31.80 / 24.58 / −0.199 | 34.55 / 24.36 / −0.027 |
| Citi → Barclays | LightGBM | 37.47 / 29.19 / −0.664 | 36.29 / 27.02 / −0.133 |
| Citi → Barclays | XGBoost | 38.52 / 30.17 / −0.759 | 36.97 / 27.57 / −0.176 |

Transfer is asymmetric and confounded by category, season, source sample size,
background demand and attendance scale. Source attendance medians are 16,090.5
(Barclays) and 30,596 (Citi); the latter is implausibly high for many missing Barclays
crowds but is intentionally not replaced using target data. Ridge extrapolates
poorly from arena to stadium crowds; trees cannot extrapolate beyond source
feature ranges. A single source venue cannot teach the model a venue effect.
These results do not establish venue-invariant demand relationships.

Frozen pooled-model diagnostics (all observations retained):

- Low R² reflects compressed predictions and a harder later period: actual mean/std
  rises from 22.08/26.62 on validation to 31.45/34.15 on test. LightGBM test prediction
  std is 16.22 (Ridge 11.93, XGBoost 16.09). LightGBM test bias is −6.64. Conservative
  trees, limited event-level signals and uncertain event ends are plausible
  contributors; this analysis does not establish causality or motivate test tuning.
- LightGBM test venue RMSE/R²: Barclays 35.27/−0.070, Citi 31.00/0.153. Concerts
  (33 events) have RMSE 42.49 and contribute 52.6% of its squared error, versus
  basketball (29) 27.68 and baseball (31) 31.00. Other types have one test event
  apiece, so their R² is unstable and should not support general conclusions.
- Missing-attendance events have LightGBM RMSE 39.64 versus 31.02 with attendance.
  Missingness occurs only at Barclays in test; the within-venue report avoids
  comparing it solely against Citi. Category/source quality can still confound it.
- The 79/784 buckets with actual surge ≥75 contribute 53.2% of LightGBM squared
  error; mean bias there is −72.63. For 16 events whose actual peak is ≥100,
  peak bias is −91.91, versus −16.20 for the other 82. Ridge/XGBoost show similar
  compression. Outcome-based bands are diagnostic only, never inference features.
- Flag the previously suspected pattern: month-end 23:30 with y_actual ≤2.
  Two test buckets (October 31 and November 30, 2025) are 0.26% of rows but 5.4%
  of LightGBM squared error. Their two full events contribute 11.7%. Remaining-row
  RMSE is 33.09 versus 33.98 overall; remaining-event R² is still only 0.020.
  These are descriptive subgroup scores, not cleaned/replacement evaluations.
  Anomalies matter but do not explain most error; upstream cause remains unverified.

The CSVs include counts, bias, target/prediction dispersion, group R², squared-error
shares and peak-band errors for all four approaches. No observations were removed,
hyperparameters changed, or primary-model decisions made from these diagnostics.
