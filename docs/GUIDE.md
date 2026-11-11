# Guide: from nothing to a training dataset

This guide has two parts:

1. **Run it.** The exact commands, in order.
2. **Understand it.** What we built, why, how it maps to the PRD, where we are now and what comes next.

Use `CONTEXT.md` for word definitions and `DECISIONS.md` for every decision with its numbers.

---

## Part 1: Run it

Run everything from the repo root. You need Python 3.10 or newer.

The `data/` folder is gitignored, so a fresh clone has no data. Each step below downloads or builds its own piece. Every step only reads files that earlier steps wrote, so run them in this order.

```bash
# 0. Install
pip install -r requirements.txt

# 1. Ridership: download 24 monthly Uber/Lyft trip files (several hundred MB each, kept in data/raw/; ~10+ GB total) and keep only our zones
python -m src.ingestion.tlc             # -> data/interim/venue_trips_YYYY-MM.parquet
python -m src.processing.clean_tlc      # -> data/interim/cleaned_venue_trips_YYYY-MM.parquet
python -m src.features.features         # -> data/processed/venue_demand_15min_YYYY-MM.parquet (15-min counts per venue)

# 2. Events: scrape the Barclays calendar, one year at a time
python -m src.ingestion.barclays_discovery --year 2024
python -m src.ingestion.barclays_ingest --year 2024      
# -> data/interim/events/barclays_2024/events_all.csv
python -m src.ingestion.barclays_discovery --year 2025
python -m src.ingestion.barclays_ingest  --year 2025

# 3. Event facts: end times and attendance
python -m -m src.ingestion.box_scores     # basketball, from ESPN    -> data/interim/box_scores.csv
python -m src.ingestion.tour_boxscores    # concert attendance       -> data/interim/tour_attendance.csv
python -m src.ingestion.weather           # hourly weather per venue -> data/interim/weather/*.json

# 4. (optional) see the calibrated end-time durations and how accurate they are
python -m src.processing.calibration

# 5. Build the dataset
python -m src.features.dataset            # -> data/processed/event_dataset.parquet
```

**Check it worked.** Step 5 should print 2,480 rows: 99 train, 49 val and 162 test Events. The live websites may have changed since the original run, so small differences are possible. Then run:

```bash
python -m pytest -q                              # all tests should pass
jupyter notebook notebooks/05_event_ridership_merge.ipynb   # coverage, drops, exit curves, sanity plots
```

**If something breaks:**
- `ingest_events` exits with status 2: this is a data-quality *warning*, not a failure. Read `extraction_issues_all.csv` next to the output.
- `load_events` raises "unmatched override" or "Duplicate event_id": the website changed. Fix the matching row in `src/event_overrides.csv`.
- Box scores and tour data cache under `data/raw/`, weather under `data/interim/weather/`, so re-running step 3 is cheap.

Files you edit by hand are the human-reviewed facts. All of them are in git:

| File | What it holds |
|---|---|
| `src/event_overrides.csv` | Corrected starts, ends, attendance and statuses for specific showings. This file always wins over automated sources. |
| `src/event_runtimes.csv` | Published show lengths, such as Disney On Ice. |
| `src/event_categories.json` | The event type for pages the scraper couldn't classify. |
| `src/config.py` | Zones, weights, coordinates and the training period. |

---

## Part 2: Understand it

### The question (from the PRD)
When an event ends at a venue, how many *extra* Uber/Lyft rides appear in each 15-minute bucket afterward? The model predicts that "extra" number, called the **Surge Delta**.

### What we built, in plain words

1. **Count rides near the venue.** Every 15 minutes, we count rides requested in the venue's own taxi zone, plus a fraction of rides from the zones next to it. Neighbours count ¼ at Barclays and ¾ at Citi Field. This count is the **Weighted Demand**.
2. **List every event.** We scrape the Barclays calendar for 2024–2025. Each event gets a start, an end and, where possible, a real attendance figure.
3. **Find when each event really ended.** Basketball has the exact final-buzzer time from ESPN. Wrestling and boxing use broadcast times. Family shows use their published length. Everything else uses "typical length for this kind of event", measured from when ridership peaked after past events of that kind.
4. **Work out what a normal night looks like (the Baseline).** For an event on a Tuesday at 10pm, we look at the 4 most recent Tuesdays at 10pm that had *no* event and no holiday, and take the middle value.
5. **Calculate the extra:** Surge Delta = what actually happened − Baseline.
6. **Make rows.** For each event we take 8 buckets, from 30 minutes before the end to 75 minutes after. Each row gets the event info, the weather, calendar flags and recent demand. Events are split by date into train, validation and test.

### Why we made the big choices

| Choice | Why | How it relates to the PRD |
|---|---|---|
| Baseline = median of quiet weeks, not "same time last week" | Barclays often has events on back-to-back Tuesdays. Subtracting last week would subtract one event from another, so a real surge could look like zero. | This is the PRD's own definition: "non-event control weeks". The old code had drifted to last week's value. We fixed the code back to the PRD. |
| Look back 12 weeks, not 8 | With 8 weeks, busy weeks had no quiet comparison, so the busiest events got thrown out. Those are exactly the nights the model has to handle. | The PRD says "previous 3–4 non-event weeks". We still use 4 (minimum 3); we just allow looking further back to find them. |
| Exclude Start − 2h to End + 2h, not whole days | This covers arrivals and late leavers but keeps the rest of the day usable as a quiet comparison. | The PRD only says "non-event weeks". This makes that rule precise. |
| Real attendance only, blank otherwise | Filling with venue capacity just repeats the event type and looks like real data. | The PRD allowed "or estimated venue capacity". We chose not to. |
| No setlist.fm | Its times are rough, and it needs an API key. Calibrating from ridership peaks was accurate to about 10 minutes. | The PRD listed setlist.fm as a source. We dropped it. |
| One row per event × 15-min bucket, venue level | 8 buckets from end − 30 to end + 75; the last bucket covers up to +90. Per-zone modelling is a stretch goal. | Matches the PRD's [end − 30, end + 90] window. |
| Federal holidays only, no NYC school holidays | pandas has federal holidays built in; school calendars would need a new source for little gain. | The PRD asked for both. We skipped school holidays. |
| Recorded weather from Open-Meteo | It's free and matches the PRD's source. | Matches the PRD. A real forecast would be fuzzier, which is a known limitation. |
| Split train Mar–Sep 2024, val Oct–Dec 2024, test 2025 | Time order prevents peeking at the future. Splitting by event keeps one event from landing in both train and test. | Exactly the PRD's split. |
| Zones 189 and 253 | The PRD's first draft had zones 17 and 93, which was a mistake. | Corrected in the PRD. |

### Where we are now
- ✅ **Done:** the dataset. It has 310 events and 2,480 rows, every row has a valid Baseline and Surge Delta and all tests pass.
- ✅ **Sanity checked:** on quiet nights, the Surge Delta sits around 0, which means the Baseline works. After events it's clearly positive. Basketball spikes about 15 minutes after the buzzer. Concerts are flatter and longer.
- ⚠️ **Only Barclays has events.** Citi Field ridership is collected, but no Mets games or Citi Field concerts are in the event table yet.
- ⚠️ **Half the end times are estimates** (`calibrated_category`), and **attendance is blank for 126 of 310 events**, mostly concerts and family shows.
- ⚠️ **Short-term lag features** (demand 15/30/60 minutes ago) peek inside the Prediction Window. They're fine for "predict the next 15 minutes" but not for "predict tonight before it starts".
- ❌ **No model yet.** `src/train.py` and `src/evaluate.py` are empty.

### What's next
1. **Baselines from the PRD:** naive persistence (the value 15 minutes ago) and "T-7" (the same slot on a recent non-event week). These are the bars to beat.
2. **Ridge regression, then LightGBM/XGBoost,** on `event_dataset.parquet`. First decide whether to drop the short lags (pre-event forecast) or keep them (next-15-minutes forecast).
3. **Score with RMSE and MAE on validation.** Tune, then report on the 2025 test set once. PRD goal: at least 15–20% lower RMSE than the T-7 baseline.
4. **SHAP:** explain the top 3–5 features, as the PRD asks.
5. **Citi Field event adapter** (Mets games and concerts). This is needed for the PRD's "does it transfer between venues" check.
6. Open team questions: #13 (how to handle missing concert attendance) and #14 (small code clean-up).
