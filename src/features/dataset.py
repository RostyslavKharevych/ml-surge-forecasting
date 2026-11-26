"""Training dataset: one row per held Event × bucket of its Prediction Window, at Venue level.

Rows carry Surge Delta (the target) against the Clean-Week Baseline, Venue Weather, calendar
and holiday flags, the existing demand lags, Next Showing Gap and a chronological split.
"""

import argparse
import logging

import pandas as pd

from src.features.baseline import add_baseline, drop_events_without_baseline, is_holiday, prediction_buckets
from src.config import PROCESSED_DATA_DIR, VENUE_CONFIG
from src.processing.events import load_events
from src.features.features import consolidate_demand_matrices, engineer_feature_lags
from src.ingestion.weather import load_weather

log = logging.getLogger(__name__)

OUTPUT_PATH = PROCESSED_DATA_DIR / "event_dataset.parquet"
MODEL_OUTPUT_PATH = PROCESSED_DATA_DIR / "model_dataset.parquet"
MODEL_COLUMNS = [
    "event_id",
    "venue_zone_id",
    "time_bin",
    "mins_since_end",
    "event_type",
    "attendance_est",
    "precipitation_mm",
    "temp_f",
    "y_actual",
    "y_baseline",
    "surge_delta",
]
# Grouped chronological event split (~65% train / 15% val / 20% test sorted by event_start).
TRAIN_FRAC = 0.65
VAL_FRAC = 0.15
EVENT_FIELDS = ["event_id", "venue", "event_type", "event_start", "event_end", "end_time_basis",
                "attendance", "attendance_basis"]
DEMAND_FIELDS = ["core_trips", "spillover_trips", "weighted_demand", "baseline", "surge_delta",
                 "sin_time_of_day", "cos_time_of_day", "day_of_week", "is_weekend", "is_holiday",
                 "demand_lag_1", "demand_lag_2", "demand_lag_4", "demand_lag_96", "demand_lag_672",
                 "rolling_mean_1h", "rolling_std_1h", "rolling_mean_4h"]
WEATHER_FIELDS = ["precipitation_mm", "temp_f", "wind_mph", "is_raining"]


def assign_split(starts, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC):
    """Assign chronological train/val/test split (~65/15/20) sorted by Event Start."""
    split = pd.Series(None, index=starts.index, dtype=object)
    valid = starts.dropna().sort_values(kind="mergesort")
    n = len(valid)
    if n == 0:
        return split
    if n == 1:
        split.loc[valid.index] = "train"
        return split
    if n == 2:
        split.loc[valid.index[0]] = "train"
        split.loc[valid.index[1]] = "test"
        return split

    train_end = max(1, min(n - 2, round(n * train_frac)))
    val_end = max(train_end + 1, min(n - 1, round(n * (train_frac + val_frac))))
    split.loc[valid.index[:train_end]] = "train"
    split.loc[valid.index[train_end:val_end]] = "val"
    split.loc[valid.index[val_end:]] = "test"
    return split


def next_showing_starts(events, held):
    """Start of the next held Event at the same venue on the same day, per Event in `events`."""
    later = events.reset_index().merge(held[["venue", "event_start"]], on="venue", suffixes=("", "_next"))
    later = later[(later.event_start_next > later.event_start)
                  & (later.event_start_next.dt.normalize() == later.event_start.dt.normalize())]
    return later.groupby("index").event_start_next.min().reindex(events.index)


def build_dataset(events, demand, weather):
    """Prediction Window rows for held Events with a full Baseline, split ~65/15/20 chronologically.
    `demand` is the contiguous venue grid from consolidate_demand_matrices; `weather` as
    load_weather returns it."""
    demand = engineer_feature_lags(add_baseline(demand, events))
    demand["is_holiday"] = is_holiday(demand.timestamp_15m)
    held = events[(events.event_status == "held") & events.event_start.notna()]
    kept = drop_events_without_baseline(held, demand).copy()
    kept["split"] = assign_split(kept.event_start)
    kept["next_start"] = next_showing_starts(kept, held)

    rows = (kept[EVENT_FIELDS + ["split", "next_start"]]
            .assign(time_bin=kept.event_end.map(lambda end: list(prediction_buckets(end))))
            .explode("time_bin").astype({"time_bin": "datetime64[ns]"}))
    rows["minutes_since_end"] = (rows.time_bin - rows.event_end.dt.floor("15min")).dt.total_seconds().astype(int) // 60
    rows["next_showing_gap"] = ((rows.next_start - rows.time_bin).dt.total_seconds() / 60).clip(lower=0)  # 0 once the next showing has started
    rows["has_next_showing"] = rows.next_start.notna()
    rows = (rows.merge(demand[["venue", "timestamp_15m"] + DEMAND_FIELDS], how="left", validate="m:1",
                       left_on=["venue", "time_bin"], right_on=["venue", "timestamp_15m"])
            .merge(weather[["venue", "timestamp_15m"] + WEATHER_FIELDS], how="left", validate="m:1",
                   on=["venue", "timestamp_15m"]))
    return rows[["event_id", "venue", "time_bin", "minutes_since_end"] + EVENT_FIELDS[2:]
                + DEMAND_FIELDS + WEATHER_FIELDS + ["next_showing_gap", "has_next_showing", "split"]
                ].reset_index(drop=True)


def to_model_dataset(rows: pd.DataFrame) -> pd.DataFrame:
    """Project `build_dataset` output onto the locked Contract 1 `model_dataset.parquet` schema."""
    zone_map = {venue: int(cfg["core_zone"]) for venue, cfg in VENUE_CONFIG.items()}
    model_df = pd.DataFrame({
        "event_id": rows["event_id"].astype(str),
        "venue_zone_id": rows["venue"].map(zone_map).astype("int64"),
        "time_bin": pd.to_datetime(rows["time_bin"]),
        "mins_since_end": rows["minutes_since_end"].astype("int64"),
        "event_type": rows["event_type"].astype(str),
        "attendance_est": rows["attendance"].astype("Int64"),
        "precipitation_mm": rows["precipitation_mm"].astype("float64"),
        "temp_f": rows["temp_f"].astype("float64"),
        "y_actual": rows["weighted_demand"].astype("float64"),
        "y_baseline": rows["baseline"].astype("float64"),
        "surge_delta": rows["surge_delta"].astype("float64"),
    })
    return model_df[MODEL_COLUMNS].reset_index(drop=True)


def main():
    argparse.ArgumentParser(description=f"Build the training dataset into {OUTPUT_PATH} and {MODEL_OUTPUT_PATH}").parse_args()
    logging.basicConfig(level=logging.INFO)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_dataset(load_events(), consolidate_demand_matrices(), load_weather())
    rows.to_parquet(OUTPUT_PATH, index=False)
    model_rows = to_model_dataset(rows)
    model_rows.to_parquet(MODEL_OUTPUT_PATH, index=False)
    print(f"{OUTPUT_PATH}: {len(rows):,} rows")
    print(f"{MODEL_OUTPUT_PATH}: {len(model_rows):,} rows")
    print(rows.groupby(["venue", "split"]).agg(events=("event_id", "nunique"), rows=("event_id", "size")))


if __name__ == "__main__":
    main()
