"""
src/features.py
===============
15-Minute regular time-grid aggregation, spatial weighting, continuous multi-year 
consolidation, and lag feature engineering pipeline.

Architecture & Modeling Design:
-------------------------------
1. Primary Time Anchor (request_datetime):
   Trips are binned based on when the customer requested the ride. Algorithmic surge 
   pricing is evaluated at request time, not when the driver arrives.
2. Spatial Topologies & Weights:
   - Barclays Center (Dense transit hub): 1.0 * Zone 189 + 0.25 * (Zones 49, 62, 69, 97, 181, 190).
   - Citi Field (Isolated park/stadium): 1.0 * Zone 253 + 0.75 * Zone 93 (Flushing Meadows).
3. Zero-Gap Continuous Time Grid:
   Late-night stadium hours often have zero ride requests. Reindexing against a global 
   master DatetimeIndex guarantees zero-filled rows rather than missing timestamps.
4. Lag Architecture:
   - Short-term momentum: t-1 (15m), t-2 (30m), t-4 (1h).
   - Diurnal cyclicality: t-96 (24 hours ago).
   - Weekly seasonal baseline: t-672 (7 days ago).
   - Surge delta target: Δy_t = y_t - y_{t-672}.
5. Leakage Prevention:
   All rolling features use .shift(1) so the current interval is excluded from past statistics.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.config import (
    PROJECT_ROOT,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    VENUE_CONFIG,
    ANALYSIS_YEARS
)


def create_regular_time_grid(start_ts: pd.Timestamp, end_ts: pd.Timestamp, freq: str = "15min") -> pd.DatetimeIndex:
    """
    Generates a regular, contiguous DatetimeIndex bounded by normalized floor/ceil steps.

    Parameters:
    -----------
    start_ts : pd.Timestamp
        Earliest timestamp in the series.
    end_ts : pd.Timestamp
        Latest timestamp in the series.
    freq : str, default="15min"
        Time interval resolution.

    Returns:
    --------
    pd.DatetimeIndex:
        Uninterrupted timestamp sequence at the specified frequency.
    """
    start_floor = start_ts.floor(freq)
    end_floor = end_ts.floor(freq)
    return pd.date_range(start=start_floor, end=end_floor, freq=freq, name="timestamp_15m")


def aggregate_venue_demand(df: pd.DataFrame, venue_name: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Bins microdata trips into 15-minute intervals, applies spatial weighting to core 
    and spillover pickup zones, and aggregates fare and driver remuneration metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned trip microdata for a calendar month.
    venue_name : str
        Venue key ('barclays' or 'citi_field').
    config : Dict[str, Any]
        Venue zone topology dictionary containing 'core_zone' and 'spillover_zones'.

    Returns:
    --------
    pd.DataFrame:
        Interval-level aggregated demand metrics for the given venue.
    """
    core_zone = config["core_zone"]
    spillover_dict = config["spillover_zones"]
    all_venue_zones = [core_zone] + list(spillover_dict.keys())

    # Restrict to trips originating in this specific venue catchment area
    v_df = df[df["PULocationID"].isin(all_venue_zones)].copy()
    if v_df.empty:
        raise ValueError(f"No records found for venue: {venue_name}")

    # Bin rides to 15-minute intervals by request_datetime (surge pricing trigger)
    v_df["timestamp_15m"] = v_df["request_datetime"].dt.floor("15min")

    # Group by interval and zone to compute zone-level metrics
    grouped = v_df.groupby(["timestamp_15m", "PULocationID"]).agg(
        raw_trips=("request_datetime", "count"),
        mean_fare=("base_passenger_fare", "mean"),
        median_fare=("base_passenger_fare", "median"),
        mean_driver_pay=("driver_pay", "mean")
    ).reset_index()

    # Assign spatial weighting (1.0 for core; defined multiplier for spillovers)
    def get_spatial_weight(zone_id: int) -> float:
        return 1.0 if zone_id == core_zone else spillover_dict.get(zone_id, 0.0)

    grouped["spatial_weight"] = grouped["PULocationID"].apply(get_spatial_weight)
    grouped["weighted_trips"] = grouped["raw_trips"] * grouped["spatial_weight"]

    # Roll up zone metrics into overall venue catchment interval demand
    interval_summary = grouped.groupby("timestamp_15m").agg(
        core_trips=("raw_trips", lambda x: x[grouped.loc[x.index, "PULocationID"] == core_zone].sum()),
        spillover_trips=("raw_trips", lambda x: x[grouped.loc[x.index, "PULocationID"] != core_zone].sum()),
        total_raw_trips=("raw_trips", "sum"),
        weighted_demand=("weighted_trips", "sum"),
        mean_base_fare=("mean_fare", "mean"),
        median_base_fare=("median_fare", "median"),
        mean_driver_pay=("mean_driver_pay", "mean")
    )

    full_df = interval_summary.reset_index()
    full_df["venue"] = venue_name
    return full_df


def process_month_features(year: int, month: int, overwrite: bool = False) -> Optional[Path]:
    """
    Transforms cleaned microdata for a calendar month into 15-minute venue demand tables.

    Parameters:
    -----------
    year : int
        Target calendar year.
    month : int
        Target calendar month.
    overwrite : bool, default=False
        If False, skips existing processed parquet files.

    Returns:
    --------
    Optional[Path]:
        Location of the generated 15-minute processed dataset.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    input_file = INTERIM_DATA_DIR / f"cleaned_venue_trips_{year}-{month:02d}.parquet"
    output_file = PROCESSED_DATA_DIR / f"venue_demand_15min_{year}-{month:02d}.parquet"

    if not input_file.exists():
        return None

    if output_file.exists() and not overwrite:
        return output_file

    df = pd.read_parquet(input_file)
    venue_dfs = [aggregate_venue_demand(df, v_name, cfg) for v_name, cfg in VENUE_CONFIG.items()]
    combined_df = pd.concat(venue_dfs, ignore_index=True)
    combined_df.to_parquet(output_file, index=False)
    return output_file


def consolidate_demand_matrices(start_date: str = "2024-01-01 00:00:00", end_date: str = "2025-12-31 23:45:00") -> pd.DataFrame:
    """
    Stitches individual monthly partitions across the full 24-month horizon, resolves 
    boundary-seam collisions, and reindexes against a continuous master time grid per venue.

    Zero-fills inactive intervals (e.g., late-night stadium quiet hours) and forward/backward 
    fills pricing metrics to maintain uninterrupted continuity for lag features.

    Parameters:
    -----------
    start_date : str, default="2024-01-01 00:00:00"
        Global timeline anchor start.
    end_date : str, default="2025-12-31 23:45:00"
        Global timeline anchor end.

    Returns:
    --------
    pd.DataFrame:
        Continuous, zero-gap 2-year demand matrix for all target venues.
    """
    files = sorted(PROCESSED_DATA_DIR.glob("venue_demand_15min_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No processed files found in {PROCESSED_DATA_DIR}")

    dfs = [pd.read_parquet(f) for f in files]
    raw_consolidated = pd.concat(dfs, ignore_index=True)

    # Master regular grid across the entire multi-year timeline
    master_grid = pd.date_range(start=start_date, end=end_date, freq="15min", name="timestamp_15m")

    aligned_venues = []
    for venue_name in VENUE_CONFIG.keys():
        v_data = raw_consolidated[raw_consolidated["venue"] == venue_name].copy()
        
        # Deduplicate potential seam collisions at midnight across monthly files
        v_data = v_data.drop_duplicates(subset=["timestamp_15m"], keep="last")
        v_data = v_data.set_index("timestamp_15m")

        # Reindex to master regular grid to resolve unobserved intervals
        v_aligned = v_data.reindex(master_grid)
        v_aligned["venue"] = venue_name

        # Zero-fill intervals with no ride activity
        count_cols = ["core_trips", "spillover_trips", "total_raw_trips", "weighted_demand"]
        v_aligned[count_cols] = v_aligned[count_cols].fillna(0.0)

        # Forward and backward fill pricing signals across inactive intervals
        price_cols = ["mean_base_fare", "median_base_fare", "mean_driver_pay"]
        v_aligned[price_cols] = v_aligned[price_cols].ffill().bfill()

        aligned_venues.append(v_aligned.reset_index())

    final_df = pd.concat(aligned_venues, ignore_index=True)
    return final_df


def engineer_feature_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs cyclical calendar encodings, momentum lags, rolling statistical windows, 
    and the T-7 weekly baseline delta target (target_delta_y).

    Lag Specifications (15m step):
    -----------------------------
    - demand_lag_1:   t - 1   (15 min ago - immediate demand shock)
    - demand_lag_2:   t - 2   (30 min ago - surge velocity/slope)
    - demand_lag_4:   t - 4   (1 hour ago - immediate pre-event background)
    - demand_lag_96:  t - 96  (24 hours ago - daily diurnal cycle)
    - demand_lag_672: t - 672 (7 days ago - weekly seasonal baseline)

    Target Variable:
    ----------------
    target_delta_y = weighted_demand_t - weighted_demand_{t-672}
    Represents net demand elevation above the same 15-minute window last week.

    Parameters:
    -----------
    df : pd.DataFrame
        Continuous consolidated demand matrix from consolidate_demand_matrices().

    Returns:
    --------
    pd.DataFrame:
        DataFrame enriched with all engineered temporal and lag features.
    """
    df = df.copy()
    
    # 1. Cyclical Calendar Encodings (Harmonic Representations)
    ts = df["timestamp_15m"]
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    df["sin_time_of_day"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["cos_time_of_day"] = np.cos(2 * np.pi * minute_of_day / 1440)
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([4, 5, 6]).astype(int)

    # 2. Grouped Lag and Rolling Features per Venue
    venue_groups = []
    for venue_name, group in df.groupby("venue"):
        group = group.sort_values("timestamp_15m").copy()

        # Momentum & Seasonal Lags
        group["demand_lag_1"] = group["weighted_demand"].shift(1)
        group["demand_lag_2"] = group["weighted_demand"].shift(2)
        group["demand_lag_4"] = group["weighted_demand"].shift(4)
        group["demand_lag_96"] = group["weighted_demand"].shift(96)
        group["demand_lag_672"] = group["weighted_demand"].shift(672)

        # Rolling 1-Hour & 4-Hour Metrics (closed='left' via shift(1) to prevent lookahead leakage)
        group["rolling_mean_1h"] = group["weighted_demand"].shift(1).rolling(window=4, min_periods=1).mean()
        group["rolling_std_1h"] = group["weighted_demand"].shift(1).rolling(window=4, min_periods=1).std().fillna(0)
        group["rolling_mean_4h"] = group["weighted_demand"].shift(1).rolling(window=16, min_periods=1).mean()

        # Primary Modeling Target: Surge Elevation above Weekly Baseline
        group["target_delta_y"] = group["weighted_demand"] - group["demand_lag_672"]

        venue_groups.append(group)

    return pd.concat(venue_groups, ignore_index=True)


def run_feature_pipeline(save_master: bool = True) -> Path:
    """
    End-to-end feature pipeline execution: consolidates all monthly parquet files, 
    engineers all lags and cyclical signals, trims the initial 7-day warmup window 
    where lag_672 is undefined, and outputs the master ML matrix.

    Parameters:
    -----------
    save_master : bool, default=True
        Whether to serialize the final matrix to data/processed/.

    Returns:
    --------
    Path:
        Location of the final master training dataset.
    """
    consolidated_df = consolidate_demand_matrices()
    featured_df = engineer_feature_lags(consolidated_df)

    # Filter out initial 7-day warmup period where lag_672 (1-week lag) is NaN
    ml_ready_df = featured_df.dropna(subset=["demand_lag_672"]).reset_index(drop=True)
    master_file = PROCESSED_DATA_DIR / "master_feature_matrix_15min.parquet"
    
    if save_master:
        ml_ready_df.to_parquet(master_file, index=False)
        print(f"[MASTER MATRIX READY] Saved {master_file.name}")
        print(f"  Total records: {len(ml_ready_df):,}")

    return master_file


if __name__ == "__main__":
    run_feature_pipeline()