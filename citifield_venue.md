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
7. Labels Mets entries as `game`, adds the configured concert entries as `concert`, and writes the combined table to `data/interim/citi_field_events_2024.csv`.

The pipeline includes every completed Citi Field game returned for the 2024 date range, plus the configured Foo Fighters, blink-182, ATEEZ, and Green Day concert entries. The combined table contains 85 game rows and 4 concert rows. Concert end times and durations are intentionally blank for now.

## Event fields

The CSV contains:

| Column | Description |
|---|---|
| `event_id` | Project event identifier |
| `event_type` | `game` or `concert` |
| `event_name` | Mets matchup or concert headliner |
| `event_date` | Official game or concert date |
| `venue` | Event venue, filtered to Citi Field |
| `venue_zone_id` | TLC venue zone ID for Citi Field, 93 |
| `opponent` | Visiting team |
| `start_time_local` | Game first pitch or concert headliner start time in New York time |
| `attendance` | Provided concert attendance figure; blank for game rows |
| `duration_minutes` | Official MLB game duration |
| `mlb_game_id` | MLB's unique game identifier |
| `end_time_local` | Calculated start time plus duration |

Final score and weather are not included. The concert attendance values use the figures provided for this project. Green Day's 35,000 figure is the stated tour average, not a Citi Field-specific attendance count.

## Scripts

Run the ingestion pipeline from the project root with:

```bash
python -m src.ingestion.citi_field_ingest --year 2024
```

The ingestion script is located at:

```text
src/ingestion/citi_field_ingest.py
```

The end-time calculation is reusable in:

```text
src/processing/merge_events.py
```

It calculates `end_time_local` from `start_time_local` and `duration_minutes`, so future event files can use the same logic without manually entering end times.

## EDA workflow

The event table is used with monthly TLC parquet files in `data/raw/tlc/citi_field/`; the loader also accepts the legacy `high-vol-data/` folder. The EDA pickup catchment includes TLC pickup zones **93 and 253**: zone 93 for Citi Field and nearby zone 253 so pickups from the surrounding area are included. The event venue itself remains zone 93. The EDA workflow:

1. Loads the TLC pickup timestamps and pickup zones.
2. Filters pickups to `PULocationID` 93 or 253.
3. Combines the monthly files.
4. Groups pickups into combined 15-minute catchment intervals and also retains counts by zone.
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
tests/test_pipeline_structure.py
```

Run them with:

```bash
python -m pytest -q
```

The tests verify end-time calculations, invalid inputs, event CSV records when the local generated CSV is present, unique IDs, zone filtering and 15-minute aggregation, event-window boundaries, duration parsing, schedule deduplication, and exclusion of away or unfinished games.

