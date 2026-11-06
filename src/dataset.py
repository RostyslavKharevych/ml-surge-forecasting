"""Training dataset: one row per held Event × bucket of its Prediction Window, at Venue level.

Rows carry Surge Delta (the target) against the Clean-Week Baseline, Venue Weather, calendar
and holiday flags, the existing demand lags, Next Showing Gap and a chronological split.
"""

import argparse
import logging

import pandas as pd

from src.baseline import add_baseline, drop_events_without_baseline, is_holiday, prediction_buckets
from src.config import PROCESSED_DATA_DIR, TRAINING_PERIOD
from src.events import load_events
from src.features import consolidate_demand_matrices, engineer_feature_lags
from src.weather import load_weather

log = logging.getLogger(__name__)

OUTPUT_PATH = PROCESSED_DATA_DIR / "event_dataset.parquet"
# Chronological split by Event Start (CTP_PRD.md); inclusive bounds.
SPLITS = {"train": TRAINING_PERIOD,
          "val": ("2024-10-01", "2024-12-31 23:59"),
          "test": ("2025-01-01", "2025-12-31 23:59")}
EVENT_FIELDS = ["event_id", "venue", "event_type", "event_start", "event_end", "end_time_basis",
                "attendance", "attendance_basis"]
DEMAND_FIELDS = ["core_trips", "spillover_trips", "weighted_demand", "baseline", "surge_delta",
                 "sin_time_of_day", "cos_time_of_day", "day_of_week", "is_weekend", "is_holiday",
                 "demand_lag_1", "demand_lag_2", "demand_lag_4", "demand_lag_96", "demand_lag_672",
                 "rolling_mean_1h", "rolling_std_1h", "rolling_mean_4h"]
WEATHER_FIELDS = ["precipitation_mm", "temp_f", "wind_mph", "is_raining"]


def assign_split(starts):
    """Split name per Event Start; None outside every split period."""
    split = pd.Series(None, index=starts.index, dtype=object)
    for name, (lo, hi) in SPLITS.items():
        split[starts.between(lo, hi)] = name
    return split


def next_showing_starts(events, held):
    """Start of the next held Event at the same venue on the same day, per Event in `events`."""
    later = events.reset_index().merge(held[["venue", "event_start"]], on="venue", suffixes=("", "_next"))
    later = later[(later.event_start_next > later.event_start)
                  & (later.event_start_next.dt.normalize() == later.event_start.dt.normalize())]
    return later.groupby("index").event_start_next.min().reindex(events.index)


def build_dataset(events, demand, weather):
    """Prediction Window rows for held Events with a full Baseline and an Event Start in a split.
    `demand` is the contiguous venue grid from consolidate_demand_matrices; `weather` as
    load_weather returns it."""
    demand = engineer_feature_lags(add_baseline(demand, events))
    demand["is_holiday"] = is_holiday(demand.timestamp_15m)
    held = events[(events.event_status == "held") & events.event_start.notna()]
    kept = drop_events_without_baseline(held, demand).copy()
    kept["split"] = assign_split(kept.event_start)
    outside = kept.split.isna()
    if outside.any():
        log.warning("%d events dropped, Event Start outside every split: %s", outside.sum(),
                    list(kept.event_id[outside]))
    kept = kept[~outside]
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


def main():
    argparse.ArgumentParser(description=f"Build the training dataset into {OUTPUT_PATH}").parse_args()
    logging.basicConfig(level=logging.INFO)
    rows = build_dataset(load_events(), consolidate_demand_matrices(), load_weather())
    rows.to_parquet(OUTPUT_PATH, index=False)
    print(f"{OUTPUT_PATH}: {len(rows):,} rows")
    print(rows.groupby("split").agg(events=("event_id", "nunique"), rows=("event_id", "size")))


if __name__ == "__main__":
    main()
