# Citi Field Event Data

This project builds a Citi Field event dataset for the TLC pickup EDA. It combines official Mets game information from MLB's public Stats API with the Citi Field TLC pickup data.

## What the ingestion pipeline does

The ingestion pipeline:

1. Requests the 2024 Mets schedule from MLB's Stats API.
2. Keeps completed games where the Mets were the home team at Citi Field.
3. Removes duplicate schedule records using the MLB game ID.
4. Retrieves the official start time and box-score duration for each game.
5. Converts start times to America/New_York local time.
6. Calculates each game's end time from its start time and duration.
7. Writes the complete event table to `citi_field_events_2024.csv`.

The pipeline includes every completed Citi Field game returned for the 2024 date range. The current file contains 85 games: 80 regular-season games and 5 postseason games.

## Event fields

The CSV contains:

| Column | Description |
|---|---|
| `event_id` | Project event identifier |
| `event_date` | Official game date |
| `venue` | Event venue, filtered to Citi Field |
| `venue_zone_id` | TLC zone ID for Citi Field, 93 |
| `opponent` | Visiting team |
| `start_time_local` | Official start time in New York time |
| `duration_minutes` | Official MLB game duration |
| `mlb_game_id` | MLB's unique game identifier |
| `end_time_local` | Calculated start time plus duration |

Attendance, final score, and weather are not included in this event table.

## Scripts

Run the ingestion pipeline from the project root with:

```bash
./.venv/bin/python scripts/ingest_citi_field_events.py \
  --year 2024 \
  --output citi_field_events_2024.csv
```

The ingestion script is located at:

```text
scripts/ingest_citi_field_events.py
```

The end-time calculation is reusable in:

```text
scripts/add_event_end_times.py
```

It calculates `end_time_local` from `start_time_local` and `duration_minutes`, so future event files can use the same logic without manually entering end times.

## EDA workflow

The event table is used with the monthly TLC parquet files in `high-vol-data/`. The EDA workflow:

1. Loads the TLC pickup timestamps and pickup zones.
2. Filters pickups to Citi Field's TLC zone, `PULocationID == 93`.
3. Combines the monthly files.
4. Groups pickups into 15-minute intervals.
5. Creates a game window from 30 minutes before the start through 90 minutes after the end.
6. Labels each pickup interval as `before_game`, `during_game`, `after_game`, or `no_event`.
7. Compares pickup counts across the event phases.

The notebook for this workflow is:

```text
notebooks/tlc_parquet.ipynb
```

## Tests

Tests are stored in:

```text
tests/test_event_times.py
```

Run them with:

```bash
./.venv/bin/python -m pytest -q
```

The tests verify end-time calculations, invalid inputs, complete CSV records, unique IDs, Citi Field filtering, date-range coverage, duration parsing, schedule deduplication, and exclusion of away or unfinished games.

