"""
src/ingest.py
=============
Automated extraction and catchment-area filtering pipeline for NYC TLC High Volume 
For-Hire Vehicle (FHVHV) records (Uber & Lyft).

Design Principles:
------------------
1. Memory-Constrained Streaming: Raw monthly parquet files (~500 MB - 1 GB compressed, 
   representing 15M-25M citywide trips) are streamed directly from CloudFront in 1 MB 
   chunks to prevent memory exhaustion on local developer machines.
2. Predicate Pushdown Filtering: PyArrow dataset scanning filters pickups strictly 
   to the 9 target venue taxi zones before records enter Pandas RAM.
3. Idempotent Caching: Existing interim partitions are detected and preserved to 
   prevent redundant network downloads.
"""

import sys
from pathlib import Path
from typing import List, Optional
import requests
import pyarrow.dataset as ds
import pandas as pd

# Resolve project root dynamically and ensure visibility in sys.path
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    TLC_BASE_URL,
    ALL_INGESTION_ZONES,
    ANALYSIS_YEARS
)

# Standardized column schema required for downstream telematics and surge modeling
COLUMNS = [
    "request_datetime",     # Primary timestamp: when passenger requested the ride
    "pickup_datetime",      # Physical journey start
    "dropoff_datetime",     # Physical journey conclusion
    "PULocationID",         # TLC Taxi Zone ID for origin (venue filtering key)
    "DOLocationID",         # TLC Taxi Zone ID for destination
    "trip_miles",           # Total distance traveled (statute miles)
    "trip_time",            # Duration in seconds
    "base_passenger_fare",  # Unadjusted base fare (captures algorithm surge multiplier)
    "driver_pay"            # Supply-side compensation signal
]


def download_tlc_monthly_parquet(year: int, month: int, service: str = "fhvhv") -> Path:
    """
    Streams a single monthly TLC Parquet dataset from the official NYC CloudFront CDN.

    Parameters:
    -----------
    year : int
        Four-digit target year (e.g., 2024).
    month : int
        Target month (1 to 12).
    service : str, default="fhvhv"
        TLC vehicle category code ('fhvhv' = High-Volume For-Hire Vehicle: Uber & Lyft).

    Returns:
    --------
    Path:
        Local filesystem path to the downloaded raw Parquet file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{service}_tripdata_{year}-{month:02d}.parquet"
    target_path = RAW_DATA_DIR / filename
    
    # Skip download if the complete raw file is already cached locally
    if target_path.exists():
        print(f"File already exists locally: {target_path}")
        return target_path

    url = f"{TLC_BASE_URL}/{filename}"
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Stream in 1 MB chunks to bound transient memory allocation
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"Downloaded successfully: {target_path}")
    return target_path


def filter_venue_trips(
    parquet_path: Path,
    pickup_location_ids: List[int],
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extracts trips originating within venue catchment zones using PyArrow pushdown predicates.

    By pushing the zone filter into the Parquet reader level, only relevant rows are decoded, 
    reducing memory consumption by over 95% compared to reading full citywide datasets.

    Parameters:
    -----------
    parquet_path : Path
        Filesystem location of the raw TLC Parquet dataset.
    pickup_location_ids : List[int]
        Target TLC taxi zone IDs representing core and spillover stadium zones.
    columns : Optional[List[str]], default=None
        Subset of columns to read. Defaults to module COLUMNS list.

    Returns:
    --------
    pd.DataFrame:
        In-memory DataFrame filtered to pickups originating in the specified taxi zones.
    """
    if columns is None:
        columns = COLUMNS

    dataset = ds.dataset(parquet_path, format="parquet")
    predicate = ds.field("PULocationID").isin(pickup_location_ids)
    
    table = dataset.to_table(columns=columns, filter=predicate)
    df = table.to_pandas()
    print(f"Filtered {len(df):,} trips matching zones {pickup_location_ids}")
    return df


def ingest_month(year: int, month: int, keep_raw: bool = True) -> Path:
    """
    Orchestrates the ingestion, zone filtering, and interim caching for a single calendar month.

    Parameters:
    -----------
    year : int
        Target calendar year.
    month : int
        Target calendar month.
    keep_raw : bool, default=True
        Whether to retain the full multi-hundred MB raw citywide Parquet file on disk.

    Returns:
    --------
    Path:
        Location of the generated interim Parquet dataset.
    """
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    interim_path = INTERIM_DATA_DIR / f"venue_trips_{year}-{month:02d}.parquet"

    if interim_path.exists():
        print(f"[{year}-{month:02d}] Interim file already exists. Skipping.")
        return interim_path

    print(f"\n[{year}-{month:02d}] Processing...")
    raw_path = download_tlc_monthly_parquet(year, month)
    
    venue_df = filter_venue_trips(raw_path, ALL_INGESTION_ZONES)
    venue_df.to_parquet(interim_path, index=False)
    print(f"Filtered {len(venue_df):,} trips -> {interim_path.name}")

    if not keep_raw and raw_path.exists():
        raw_path.unlink()
        print(f"Deleted raw file: {raw_path.name}")

    return interim_path


def run_ingestion_pipeline(years: List[int] = ANALYSIS_YEARS, keep_raw: bool = True):
    """
    Batch orchestrator: loops across all target study years and months, downloading and 
    extracting venue-specific trips while handling unreleased future months gracefully.

    Parameters:
    -----------
    years : List[int], default=ANALYSIS_YEARS
        List of target years to process (e.g., [2024, 2025]).
    keep_raw : bool, default=True
        Disk preservation policy for raw TLC dumps.
    """
    print(f"Starting ingestion pipeline for years: {years}")
    print(f"Target Zones ({len(ALL_INGESTION_ZONES)}): {ALL_INGESTION_ZONES}")

    for year in years:
        for month in range(1, 13):
            try:
                ingest_month(year, month, keep_raw=keep_raw)
            except requests.exceptions.HTTPError as e:
                # Handle unreleased forward months on the TLC CDN (e.g., 403 Forbidden / 404 Not Found)
                if e.response.status_code in (403, 404):
                    print(f"  [NOTICE] {year}-{month:02d} not available on TLC CDN yet. Moving to next.")
                    continue
                else:
                    raise e
            except Exception as e:
                print(f"  [ERROR] Failed processing {year}-{month:02d}: {e}")
                raise e


if __name__ == "__main__":
    # Standard entry point: ingests full 2024-2025 batch
    print(f"Starting ingestion for zones: {ALL_INGESTION_ZONES}")
    run_ingestion_pipeline(years=[2024, 2025], keep_raw=True)
