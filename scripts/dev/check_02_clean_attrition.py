import sys
from pathlib import Path
import pandas as pd

# Dynamic project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INTERIM_DATA_DIR
from src.clean import clean_venue_dataframe

def inspect_january_cleaning():
    raw_interim_file = INTERIM_DATA_DIR / "venue_trips_2024-01.parquet"
    clean_interim_file = INTERIM_DATA_DIR / "cleaned_venue_trips_2024-01.parquet"

    print("=" * 60)
    print("STEP 2: CLEANING ATTRITION & SANITY AUDIT (JANUARY 2024)")
    print("=" * 60)

    if not raw_interim_file.exists():
        print(f"Raw interim file missing: {raw_interim_file}")
        return

    raw_df = pd.read_parquet(raw_interim_file)
    cleaned_df, report = clean_venue_dataframe(raw_df)

    print("\n[ROW ATTRITION BREAKDOWN]")
    print(f"  Initial records:        {report['initial_rows']:,}")
    print(f"  Exact duplicates:       -{report['dropped_duplicates']:,}")
    print(f"  Bad chronology:         -{report['dropped_bad_chronology']:,}")
    print(f"  Bad duration (<1m/>4h): -{report['dropped_bad_duration']:,}")
    print(f"  Bad distance (<0.1m):   -{report['dropped_bad_distance']:,}")
    print(f"  Excessive speed (>75m): -{report['dropped_excessive_speed']:,}")
    print(f"  Invalid fare/driver pay:-{report['dropped_invalid_economics']:,}")
    print(f"  --------------------------------------------------")
    print(f"  Final clean records:    {report['final_rows']:,}")
    print(f"  Retention rate:         {report['retention_rate_pct']}%")

    if clean_interim_file.exists():
        saved_df = pd.read_parquet(clean_interim_file)
        assert len(saved_df) == len(cleaned_df), "Mismatch between saved clean data and recomputed clean data!"
        print(f"\n[VERIFIED] Cleaned file matches on disk ({clean_interim_file.name}).")

if __name__ == "__main__":
    inspect_january_cleaning()
