from pathlib import Path

# Anchors to project root regardless of whether run from root, notebooks/, or subpackages
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# TLC URL Base
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# Timestamp anchor constants (surge pricing is evaluated at request time; pickup is fallback)
PRIMARY_TIMESTAMP_COL = "request_datetime"
FALLBACK_TIMESTAMP_COL = "pickup_datetime"
TIME_INTERVAL = "15min"

# Venue zone configurations (aligned with docs/PRD.md Section 3)
VENUE_CONFIG = {
    "barclays": {
        "coordinates": (40.6826, -73.9754),  # arena (lat, lon), for Venue Weather
        "core_zone": 189,
        "spillover_zones": {
            49: 0.25,
            62: 0.25,
            61: 0.25,
            97: 0.25,
            181: 0.25,
            190: 0.25
        }
    },
    "citi_field": {
        "coordinates": (40.7571, -73.8458),
        "core_zone": 253,
        "spillover_zones": {
            93: 0.75
        }
    }
}

# Master list of all unique zones to ingest
ALL_INGESTION_ZONES = sorted({
    zone
    for cfg in VENUE_CONFIG.values()
    for zone in (cfg["core_zone"], *cfg["spillover_zones"].keys())
})

ANALYSIS_YEARS = [2024, 2025]
TRAINING_PERIOD = ("2024-03-01", "2024-09-30 23:59")  # inclusive; later events are validation/test
