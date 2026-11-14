"""Clean TLC pickup rows and aggregate them into fixed time intervals."""

import pandas as pd

from src.config import CITI_FIELD_PICKUP_ZONES, TIME_INTERVAL


def clean_pickups(trips: pd.DataFrame, pickup_zones=CITI_FIELD_PICKUP_ZONES) -> pd.DataFrame:
    required = {"pickup_datetime", "PULocationID"}
    missing = required.difference(trips.columns)
    if missing:
        raise ValueError("Missing required TLC columns: {}".format(", ".join(sorted(missing))))

    cleaned = trips.copy()
    cleaned["pickup_datetime"] = pd.to_datetime(
        cleaned["pickup_datetime"], errors="coerce"
    )
    cleaned = cleaned[
        cleaned["PULocationID"].isin(pickup_zones)
        & cleaned["pickup_datetime"].notna()
    ].copy()
    return cleaned


def aggregate_pickups_15min(
    trips: pd.DataFrame, interval: str = TIME_INTERVAL
):
    """Return combined interval totals and a separate interval-by-zone table."""
    required = {"pickup_datetime", "PULocationID"}
    missing = required.difference(trips.columns)
    if missing:
        raise ValueError("Missing required TLC columns: {}".format(", ".join(sorted(missing))))

    binned = trips.copy()
    binned["time_bin"] = binned["pickup_datetime"].dt.floor(interval)
    combined = (
        binned.groupby("time_bin").size().reset_index(name="pickup_count")
        .sort_values("time_bin")
    )
    by_zone = (
        binned.groupby(["time_bin", "PULocationID"])
        .size().reset_index(name="pickup_count")
        .sort_values(["time_bin", "PULocationID"])
    )
    return combined, by_zone
