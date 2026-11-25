import sys
from pathlib import Path
import pandas as pd

# Dynamic project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR

def inspect_january_ingestion():
    raw_file = RAW_DATA_DIR / "fhvhv_tripdata_2024-01.parquet"
    interim_file = INTERIM_DATA_DIR / "venue_trips_2024-01.parquet"

    print("=" * 60)
    print("STEP 1: INGESTION SANITY CHECK (JANUARY 2024)")
    print("=" * 60)

    print("\n[FILE SIZES ON DISK]")
    if raw_file.exists():
        print(f"  Raw Parquet:     {raw_file.name} ({raw_file.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"  Raw Parquet:     Not found at {raw_file}")

    if interim_file.exists():
        print(f"  Interim Parquet: {interim_file.name} ({interim_file.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"  Interim Parquet: Not found at {interim_file}")
        return

    df = pd.read_parquet(interim_file)
    print("\n[EXTRACTED INTERIM METRICS]")
    print(f"  Total records:      {len(df):,}")
    print(f"  Active columns:     {list(df.columns)}")
    print(f"  Earliest request:   {df['request_datetime'].min()}")
    print(f"  Latest request:     {df['request_datetime'].max()}")

    print("\n[PICKUP COUNTS BY TARGET ZONE]")
    zone_counts = df['PULocationID'].value_counts().sort_index()
    for zone, count in zone_counts.items():
        print(f"  Zone {zone:<3}: {count:>8,} trips")

if __name__ == "__main__":
    inspect_january_ingestion()
