from pathlib import Path

# Anchors to project root regardless of whether run from root, notebooks/, or subpackages
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# TLC URL Base
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# Venue zone configurations
VENUE_CONFIG = {
    "barclays": {
        "core_zone": 189,
        "spillover_zones": {
            49: 0.25,
            62: 0.25,
            69: 0.25,
            97: 0.25,
            181: 0.25,
            190: 0.25
        }
    },
    "citi_field": {
        "core_zone": 253,
        "spillover_zones": {
            93: 0.75
        }
    }
}

# Master list of all unique zones to ingest
ALL_INGESTION_ZONES = sorted(list({
    189, 49, 62, 69, 97, 181, 190, 253, 93
}))

ANALYSIS_YEARS = [2024, 2025]
