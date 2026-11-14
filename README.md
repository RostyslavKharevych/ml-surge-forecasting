# ML Surge Forecasting

Tools for studying pickup-demand changes around Citi Field.

## Project Structure

```text
data/
├── raw/                         # Local source data; ignored by Git
│   └── tlc/
│       └── citi_field/           # Monthly HVFHV parquet files
├── interim/                     # Cleaned pickups and extracted event CSVs
└── processed/                   # Merged feature matrices
src/
├── config.py                     # Shared paths, venue zones, time settings
├── ingestion/
│   ├── tlc_ingest.py             # Read/filter source HVFHV parquet fields
│   └── citi_field_ingest.py      # Fetch Mets games and assemble Citi events
├── processing/
│   ├── clean_tlc.py              # Clean pickup rows and create 15-minute counts
│   └── merge_events.py           # Calculate game ends and label event windows
└── processing/
    ├── clean_tlc.py              # Clean pickup rows and create 15-minute counts
    └── merge_events.py           # Calculate game ends and label event windows
notebooks/                         # EDA and verification
tests/                             # Unit and data-quality tests
requirements.txt
```

## Citi Field Pipeline

The local TLC input files are expected in `data/raw/tlc/citi_field/`. Citi Field pickup analysis includes zone 93 and spillover zone 253; the event venue ID remains 93. Shared paths and zone constants are defined in `src/config.py`.

Fetch all completed Mets home games for 2024, combine them with the configured concert rows, calculate game end times, and write the interim event CSV:

```bash
python -m src.ingestion.citi_field_ingest --year 2024
```

The command writes `data/interim/citi_field_events_2024.csv`. The notebook reads the parquet files, filters to zones 93 and 253, creates combined and per-zone 15-minute counts, prepares game windows, and compares pickup activity by phase.

Run tests from the project root:

```bash
python -m pytest -q
```

Raw and generated data files are ignored by Git. Scripts, notebooks, configuration, documentation, and tests are tracked.
