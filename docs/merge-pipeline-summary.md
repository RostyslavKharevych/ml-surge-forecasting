# Event–Ridership Merge: Team Decision Summary

What we decided while building the training dataset (issues #3–#11), why, and what we found.
Terms (capitalised) are defined in `CONTEXT.md`. The full decision log, with every number, is in `DECISIONS.md`. The target trade-off has its own ADR in `docs/adr/0001-surge-delta-uses-clean-week-baseline.md`.

**Output:** `data/processed/event_dataset.parquet`. It has 310 Events and 2,480 rows (8 per Event). There are 99 Events in train, 49 in validation and 162 in test. Every Event so far is at Barclays.

## Decisions

### Zones
- Barclays uses core zone **189**, with spillover zones 49, 61, 62, 97, 181 and 190 at weight 0.25.
- Citi Field uses core zone **253**, with spillover zones 92 and 93 at weight 0.75.
- The PRD's original zones 17 and 93 were wrong and have been corrected. `src/config.py` `VENUE_CONFIG` is the source of truth.
- Weighted Demand = core-zone pickups + weight × spillover-zone pickups, per 15-minute bucket, at Venue level.

### Target: Clean-Week Baseline, not Week-Ago Demand (lag 672)
- **Surge Delta = Weighted Demand − Baseline.**
- **Baseline:** the median Weighted Demand of the 4 most recent Clean Weeks for the same venue, weekday and 15-minute slot, looking back up to 12 weeks. At least 3 Clean Weeks are required.
- **Why not last week (lag 672):** Barclays often has events on the same weekday in back-to-back weeks. Subtracting last week would compare one event night with another, and a real surge could show up as roughly 0.
- `demand_lag_672` is kept as a model **input** only. The old `target_delta_y` is retired.
- **Why median:** one unusual week (a storm or a transit outage) doesn't drag the Baseline.
- **Why 12 weeks, not 8:** with 8 weeks, busy stretches had no 3 quiet weeks. That dropped 55 Events, including 20 of the 51 validation Events, so validation was biased toward quiet weeks. With 12 weeks, 17 Events are dropped.
- A bucket with no Baseline is never filled in. An Event with such a bucket is dropped and logged.

### Affected Slot window
- An Affected Slot is any bucket in **[Event Start − 2h, Event End + 2h]** of an Event that was actually held.
- Cancelled Events affect nothing. Postponed Events affect only the date they were held. Duplicate listings affect nothing.
- We exclude these slots rather than whole days. That covers arrivals and slow exits without discarding the rest of the day.
- US federal holidays are excluded too (pandas `USFederalHolidayCalendar`). NYC school holidays are skipped.

### Event End and its End Time Basis
Every Event End records how it was obtained. The sources are listed from most to least trusted:

| Basis | Used for | Source |
|---|---|---|
| `box_score` | basketball | ESPN play-by-play "End Game" wallclock (the actual buzzer) |
| `broadcast_window` | wrestling, boxing | broadcast end, or published ringwalk time + fight length |
| `published_runtime` | family shows, circus | listed runtime (`src/event_runtimes.csv`) |
| `calibrated_category` | everything else | median time from start to the ridership peak, per category |

- **How calibration works:** for each category, take the median minutes from Event Start to the core-zone demand peak. Only training-period Events (Mar–Sep 2024) are used, so no future data leaks in. Results: concert 225 min, basketball 150 min, pooled fallback 195 min.
- **Checking calibration:** on 106 basketball games outside training, the calibrated end was off by a median of 9.5 min.
- **Order matters:** calibration runs on raw demand *before* the Baseline exists. The Baseline needs Event Ends, so this order avoids a circular dependency.
- We didn't use setlist.fm. Its times are rough, and it needs an API key.
- Human-verified facts live in `src/event_overrides.csv` and always win over automated sources.

### Attendance policy
- Attendance comes from **real sources only**:
  - Box scores (basketball)
  - Wikipedia and Touring Data tour tables (concerts)
  - Reported figures (wrestling, boxing)
- When no real figure exists, Attendance is left blank. It is **never filled with venue capacity**, because that just repeats `event_type` while looking like real data.
- A multi-night run reported as one figure is split evenly per night. Those values are averages, not counts.

### Row grain and Prediction Window
- **One row = one Event × one 15-minute bucket**, at Venue level. Core and spillover trips are kept as columns, not as separate targets.
- The Prediction Window is the 8 buckets from the bucket holding Event End − 30 min through +75 min. `minutes_since_end` takes the values −30 … +75.
- Back-to-back showings are all kept. `next_showing_gap` gives the minutes to the next Event Start that day, clipped at 0. `has_next_showing` flags whether one exists.
- **Split by Event Start:**
  - Train: Mar–Sep 2024
  - Validation: Oct–Dec 2024
  - Test: 2025

  All 8 rows of an Event share its split, so one Event can't land in both train and test.
- Fares and driver pay are dropped. They're measured in the same bucket as the target, so they would leak the answer.

### Weather
- We use one **recorded** hourly series per venue from the Open-Meteo archive. Each 15-minute bucket takes its hour's reading.
- Columns: precipitation, temperature, wind and `is_raining`.
- The weather grid is coarser than the taxi zones, so a per-zone series would add nothing.

### Event table contract
- One venue-agnostic Event table, with one adapter per venue. Only the Barclays adapter exists today. Citi Field plugs in without changing the merge.
- Timestamps are naive America/New_York local time.
- IDs look like `VENUE_YYYYMMDD_HHMM`.
- Every `event_end` has an `end_time_basis`, and every `attendance` has an `attendance_basis`. `load_events` raises an error otherwise.
- Statuses: `held`, `cancelled`, `postponed`, `duplicate`.
- Gaps in the scraped data are filled through reviewed files (`event_categories.json`, `event_overrides.csv`, `event_runtimes.csv`), not through code.

## Findings

**Coverage**, for the 310 Events in the dataset:

| End Time Basis | Events | Attendance Basis | Events |
|---|---|---|---|
| calibrated_category | 154 | box_score | 135 |
| box_score | 135 | (blank) | 126 |
| published_runtime | 15 | tour_boxscore | 44 |
| broadcast_window | 6 | reported | 5 |

**Dropped Events:** 35 of 345 held Events.
- 18 started in Jan–Feb 2024, which is outside every split.
- 15 started in Jan–Feb 2024 and also had no Baseline, because we have no Dec 2023 data.
- 2 had no Baseline alone: Iron Maiden and Kacey Musgraves, Nov 2024.

Dropped Events still count as Affected Slots and next showings for other Events.

**Exit curves:**
- Basketball peaks sharply about 15 minutes after the end.
- Concerts are flatter and stay elevated through +75.
- Box-score ends give the sharpest peak. Calibrated ends smear it, as expected from a per-category median.

**Baseline sanity:** on clean slots, the median Surge Delta is about 0 at both venues (Barclays 0.25, Citi Field 0.00). Around events it sits clearly above: median 6.4 across all Affected Slots, and a mean of 20–26 per weekday on Prediction Window rows.

## Known limitations
- **Recorded weather, not forecast.** A live system would only have the forecast.
- **Non-venue events in spillover zones**, such as the Prospect Park bandshell in zone 190, aren't modelled. They can make an Affected Slot look clean. Revisit if Baselines look noisy.
- **No Citi Field Events yet.** Citi Field serves only as a clean control, so the PRD's venue-transfer check can't run until a Citi Field adapter (Mets games and concerts) exists.
- **Short lags leak inside the window.** `demand_lag_1/2/4` and `rolling_*` at +15 … +75 include demand from the same Prediction Window. They only support one-bucket-ahead prediction. Drop them for a pre-event forecast.
- **Concert attendance is mostly blank** (56% of 2024 concerts, 73% of 2025), and not at random: festivals and smaller tours have no published figure. See #13.
- **Thin calibration for some categories.** Several categories rest on 1–2 training Events. Whether to pool categories under about 5 Events is still open.
- **Small train set:** 99 Events. This is enough for Ridge and shallow trees, not for much tuning.
