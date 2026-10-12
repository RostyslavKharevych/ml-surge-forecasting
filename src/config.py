from pathlib import Path

# Project Root & Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

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
