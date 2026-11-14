from pathlib import Path

# Shared project paths and venue zones.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_TLC_DIR = RAW_DATA_DIR / "tlc"
CITI_FIELD_TLC_DIR = RAW_TLC_DIR / "citi_field"
LEGACY_CITI_FIELD_TLC_DIR = BASE_DIR / "high-vol-data"
CITI_FIELD_EVENTS_PATH = INTERIM_DATA_DIR / "citi_field_events_2024.csv"
CITI_FIELD_PICKUPS_PATH = INTERIM_DATA_DIR / "citi_field_pickups_15min.parquet"

CITI_FIELD_CORE_ZONE = 93
CITI_FIELD_SPILLOVER_ZONES = frozenset({253})
CITI_FIELD_PICKUP_ZONES = frozenset(
    {CITI_FIELD_CORE_ZONE, *CITI_FIELD_SPILLOVER_ZONES}
)

# Pipeline Parameters
TIME_INTERVAL = "15min"  # 15-minute aggregation buckets
RANDOM_STATE = 42

# Column Standards (PRD Table 1 Schema)
COL_EVENT_ID = "item_or_event_id"
COL_CATEGORY = "category_type"
COL_START_TIME = "start_time"
COL_END_TIME = "end_time"
COL_ESTIMATED_SIZE = "estimated_size"
COL_TARGET = "trip_count"
COL_TARGET_SURGE = "surge_delta"


def get_citi_field_tlc_dir() -> Path:
    """Use the new raw-data layout, with a fallback for older local checkouts."""
    if CITI_FIELD_TLC_DIR.exists() and any(CITI_FIELD_TLC_DIR.glob("*.parquet")):
        return CITI_FIELD_TLC_DIR
    if LEGACY_CITI_FIELD_TLC_DIR.exists():
        return LEGACY_CITI_FIELD_TLC_DIR
    return CITI_FIELD_TLC_DIR
