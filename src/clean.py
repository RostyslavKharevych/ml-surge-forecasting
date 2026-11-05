"""
src/clean.py
============
Data hygiene and telematics validation pipeline for stadium catchment area trip microdata.

Domain Rationale for Cleaning Thresholds:
-----------------------------------------
1. Exact Duplicates: Drop redundant telemetry pings created during app sync retries.
2. Chronological Ordering: Enforces physics constraint:
   request_datetime <= pickup_datetime < dropoff_datetime.
3. Trip Duration:
   - MIN_TRIP_TIME_SEC = 60s: Discards accidental requests canceled after dispatch.
   - MAX_TRIP_TIME_SEC = 14,400s (4 hours): Discards app session tracking leaks.
4. Trip Distance:
   - MIN_TRIP_MILES = 0.1: Discards stationary cancellations where vehicle never moved.
   - MAX_TRIP_MILES = 100.0: Regional boundary cutoff for NYC Tri-State transit.
5. Implied Speed (Telematics Sanity):
   - Implied Speed = miles / (seconds / 3600).
   - Filter trips exceeding 75 mph to strip GPS multipath jumps in dense urban corridors.
6. Economic Integrity:
   - Base passenger fare must be positive (>= $0.01) to reflect real commercial demand.
   - Driver pay must be non-negative (>= $0.00).
"""

import sys
import re
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import pandas as pd

# Resolve project root dynamically and ensure visibility in sys.path
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INTERIM_DATA_DIR, ANALYSIS_YEARS

# Operational, Telematics & Economic Thresholds
MIN_TRIP_TIME_SEC = 60          # 1 minute minimum journey duration
MAX_TRIP_TIME_SEC = 14400       # 4 hours maximum journey duration
MIN_TRIP_MILES = 0.1            # 0.1 mile minimum (excludes stationary cancellations)
MAX_TRIP_MILES = 100.0          # 100 miles maximum (Tri-State regional transit cap)
MAX_IMPLIED_SPEED_MPH = 75.0    # 75 mph velocity threshold (eliminates urban GPS drift)
MIN_BASE_FARE = 0.01            # Valid positive passenger charge


def clean_venue_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Applies operational, temporal, and economic cleaning rules to raw venue trip records.
    Constructs an attrition dictionary documenting row-loss at each filter step.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw extracted venue trips dataframe from data/interim/.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, int]]:
        - Cleaned DataFrame satisfying all physical and economic constraints.
        - Detailed attrition report tracking dropped records per rule.
    """
    initial_rows = len(df)
    audit_report = {"initial_rows": initial_rows}

    # 1. Exact Deduplication
    df = df.drop_duplicates()
    audit_report["dropped_duplicates"] = initial_rows - len(df)

    # 2. Datetime Parsing & Chronological Ordering
    for col in ["request_datetime", "pickup_datetime", "dropoff_datetime"]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    valid_chronology = (
        (df["request_datetime"] <= df["pickup_datetime"]) &
        (df["pickup_datetime"] < df["dropoff_datetime"])
    )
    audit_report["dropped_bad_chronology"] = int((~valid_chronology).sum())
    df = df[valid_chronology].copy()

    # 3. Trip Duration Bounds (1 min to 4 hours)
    valid_duration = (
        (df["trip_time"] >= MIN_TRIP_TIME_SEC) & 
        (df["trip_time"] <= MAX_TRIP_TIME_SEC)
    )
    audit_report["dropped_bad_duration"] = int((~valid_duration).sum())
    df = df[valid_duration].copy()

    # 4. Trip Distance Bounds (0.1 miles to 100 miles)
    valid_distance = (
        (df["trip_miles"] >= MIN_TRIP_MILES) & 
        (df["trip_miles"] <= MAX_TRIP_MILES)
    )
    audit_report["dropped_bad_distance"] = int((~valid_distance).sum())
    df = df[valid_distance].copy()

    # 5. Implied Speed Sanity Check (Telematics)
    implied_speed = df["trip_miles"] / (df["trip_time"] / 3600.0)
    valid_speed = implied_speed <= MAX_IMPLIED_SPEED_MPH
    audit_report["dropped_excessive_speed"] = int((~valid_speed).sum())
    df = df[valid_speed].copy()

    # 6. Economic Validity (Positive fare & non-negative driver compensation)
    valid_fare = (df["base_passenger_fare"] >= MIN_BASE_FARE) & (df["driver_pay"] >= 0)
    audit_report["dropped_invalid_economics"] = int((~valid_fare).sum())
    df = df[valid_fare].copy()

    # Final summary statistics for auditing
    audit_report["final_rows"] = len(df)
    audit_report["total_rows_removed"] = initial_rows - len(df)
    audit_report["retention_rate_pct"] = round((len(df) / initial_rows) * 100, 2) if initial_rows > 0 else 0.0

    return df, audit_report


def clean_interim_month(year: int, month: int, overwrite: bool = False) -> Optional[Path]:
    """
    Cleans a single monthly interim dataset, generating a sanitized parquet partition.

    Parameters:
    -----------
    year : int
        Target calendar year.
    month : int
        Target calendar month.
    overwrite : bool, default=False
        If False, skips execution if cleaned output already exists.

    Returns:
    --------
    Optional[Path]:
        Location of the generated cleaned parquet file, or None if raw input is missing.
    """
    input_file = INTERIM_DATA_DIR / f"venue_trips_{year}-{month:02d}.parquet"
    output_file = INTERIM_DATA_DIR / f"cleaned_venue_trips_{year}-{month:02d}.parquet"

    if not input_file.exists():
        return None

    if output_file.exists() and not overwrite:
        print(f"[{year}-{month:02d}] Cleaned file already exists. Skipping.", flush=True)
        return output_file

    print(f"\n[{year}-{month:02d}] Cleaning interim data...", flush=True)
    raw_interim_df = pd.read_parquet(input_file)
    
    cleaned_df, report = clean_venue_dataframe(raw_interim_df)
    
    print(f"  Initial records:      {report['initial_rows']:,}", flush=True)
    print(f"  Exact duplicates:     -{report['dropped_duplicates']:,}", flush=True)
    print(f"  Bad chronology:       -{report['dropped_bad_chronology']:,}", flush=True)
    print(f"  Bad duration:         -{report['dropped_bad_duration']:,}", flush=True)
    print(f"  Bad distance:         -{report['dropped_bad_distance']:,}", flush=True)
    print(f"  Excessive speed:      -{report['dropped_excessive_speed']:,}", flush=True)
    print(f"  Invalid economics:    -{report['dropped_invalid_economics']:,}", flush=True)
    print(f"  Final clean records:  {report['final_rows']:,} ({report['retention_rate_pct']}% retained)", flush=True)

    cleaned_df.to_parquet(output_file, index=False)
    print(f"  [SAVED] {output_file.name}", flush=True)
    return output_file


def run_batch_clean(years: Optional[List[int]] = None, overwrite: bool = False):
    """
    Batch cleaning orchestrator: scans data/interim/ for uncleaned raw venue trip files 
    and systematically runs the cleaning rules across all partitions.

    Parameters:
    -----------
    years : Optional[List[int]], default=None
        Restricts cleaning to specific years if provided; otherwise cleans all files.
    overwrite : bool, default=False
        Policy on reprocessing existing cleaned parquet partitions.
    """
    print("=" * 60, flush=True)
    print("STARTING BATCH CLEANING PIPELINE", flush=True)
    print("=" * 60, flush=True)

    # Find all available interim raw files matching venue_trips_YYYY-MM.parquet
    interim_files = sorted(INTERIM_DATA_DIR.glob("venue_trips_*.parquet"))
    
    if not interim_files:
        print("No raw interim files found in data/interim/. Ingest data first.", flush=True)
        return

    pattern = re.compile(r"venue_trips_(\d{4})-(\d{2})\.parquet")
    processed_count = 0

    for file_path in interim_files:
        match = pattern.match(file_path.name)
        if not match:
            continue

        year, month = int(match.group(1)), int(match.group(2))
        
        # If specific years provided, filter accordingly
        if years and year not in years:
            continue

        result = clean_interim_month(year, month, overwrite=overwrite)
        if result:
            processed_count += 1

    print("\n" + "=" * 60, flush=True)
    print(f"BATCH CLEANING COMPLETE: Processed/Checked {processed_count} files.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    # Scans and processes all months currently available in data/interim/
    run_batch_clean(years=ANALYSIS_YEARS, overwrite=False)