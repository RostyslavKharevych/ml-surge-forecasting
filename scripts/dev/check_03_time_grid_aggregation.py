import sys
from pathlib import Path
import pandas as pd

# Dynamic project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR

def inspect_january_aggregation():
    agg_file = PROCESSED_DATA_DIR / "venue_demand_15min_2024-01.parquet"

    print("=" * 60)
    print("STEP 3: 15-MINUTE TIME GRID AGGREGATION (JANUARY 2024)")
    print("=" * 60)

    if not agg_file.exists():
        print(f"Processed file missing: {agg_file}")
        return

    df = pd.read_parquet(agg_file)
    print(f"\n[AGGREGATED DATASET SHAPE]: {df.shape}")
    print(f"Total 15-minute intervals: {len(df):,}")

    print("\n[INTERVAL METRICS BY VENUE]")
    metrics = df.groupby("venue")[["core_trips", "spillover_trips", "weighted_demand", "mean_base_fare"]].describe()
    print(metrics.round(2).T)

    print("\n[FIRST 5 INTERVALS FOR BARCLAYS CENTER]")
    cols = ["timestamp_15m", "core_trips", "spillover_trips", "weighted_demand", "median_base_fare"]
    print(df[df["venue"] == "barclays"][cols].head(5).to_string(index=False))

    print("\n[FIRST 5 INTERVALS FOR CITI FIELD]")
    print(df[df["venue"] == "citi_field"][cols].head(5).to_string(index=False))

if __name__ == "__main__":
    inspect_january_aggregation()
