"""Baseline and Surge Delta: demand measured against Clean Weeks, not Week-Ago Demand (ADR 0001).

A bucket's Baseline is the median Weighted Demand of the 4 most recent Clean Weeks (same venue,
weekday and slot) within the previous 12 weeks, needing at least 3. A Clean Week is a past slot
that is neither an Affected Slot nor a federal holiday.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

log = logging.getLogger(__name__)

AFFECTED_MARGIN = pd.Timedelta(hours=2)
LOOKBACK_WEEKS = 12  # DECISIONS #27: 8 dropped 39% of validation Events
CLEAN_WEEKS = 4
MIN_CLEAN_WEEKS = 3
# Minutes Since End −30 … +75: the bucket holding Event End is offset 0.
PREDICTION_OFFSETS = pd.to_timedelta(range(-30, 90, 15), unit="min")


def affected_slots(demand, events):
    """Boolean per demand row: the bucket overlaps [Event Start − 2h, Event End + 2h] of a held
    Event at its venue."""
    held = events[(events.event_status == "held") & events.event_start.notna()]
    affected = pd.Series(False, index=demand.index)
    for e in held.itertuples():
        affected |= ((demand.venue == e.venue)
                     & demand.timestamp_15m.between((e.event_start - AFFECTED_MARGIN).floor("15min"),
                                                    e.event_end + AFFECTED_MARGIN))
    return affected


def is_holiday(timestamps):
    """Boolean per timestamp: its day is a US federal holiday."""
    days = timestamps.dt.normalize()
    return days.isin(USFederalHolidayCalendar().holidays(days.min(), days.max()))


def _venue_baseline(weighted, clean):
    """Baseline for one venue's day × slot tables; earlier same-weekday days are 7 rows up."""
    weeks = np.stack([weighted.where(clean).shift(7 * k).to_numpy()
                      for k in range(1, LOOKBACK_WEEKS + 1)])
    recent = weeks.copy()
    recent[np.cumsum(~np.isnan(weeks), axis=0) > CLEAN_WEEKS] = np.nan
    with warnings.catch_warnings():  # slots with no clean week stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        baseline = np.nanmedian(recent, axis=0)
    baseline[(~np.isnan(recent)).sum(axis=0) < MIN_CLEAN_WEEKS] = np.nan
    return pd.DataFrame(baseline, index=weighted.index, columns=weighted.columns)


def add_baseline(demand, events):
    """Demand with `baseline` and `surge_delta` (Weighted Demand − Baseline) per bucket.
    Expects a contiguous 15-min grid per venue, as consolidate_demand_matrices builds."""
    demand = demand.copy()
    clean = ~affected_slots(demand, events) & ~is_holiday(demand.timestamp_15m)
    frame = demand.assign(clean=clean, day=demand.timestamp_15m.dt.normalize(),
                          slot=demand.timestamp_15m - demand.timestamp_15m.dt.normalize())
    parts = []
    for venue, group in frame.groupby("venue"):
        tables = group.pivot(index="day", columns="slot", values=["weighted_demand", "clean"])
        days = pd.date_range(tables.index.min(), tables.index.max(), freq="D")  # keep 7 rows = 1 week
        tables = tables.reindex(days)
        baseline = _venue_baseline(tables["weighted_demand"].astype(float),
                                   tables["clean"].fillna(False).astype(bool))
        parts.append(baseline.rename_axis("day").stack().rename("baseline").reset_index()
                     .assign(venue=venue, timestamp_15m=lambda b: b.day + b.slot)
                     [["venue", "timestamp_15m", "baseline"]])
    demand = demand.merge(pd.concat(parts), on=["venue", "timestamp_15m"], how="left")
    demand["surge_delta"] = demand.weighted_demand - demand.baseline
    return demand


def prediction_buckets(event_end):
    """Bucket starts of the Prediction Window; the bucket holding Event End is offset 0."""
    return event_end.floor("15min") + PREDICTION_OFFSETS


def drop_events_without_baseline(events, demand):
    """Held Events with no Event End, or whose Prediction Window has any bucket lacking a
    Baseline, are dropped and logged; other Events pass through unchanged."""
    baseline = demand.set_index(["venue", "timestamp_15m"]).baseline
    held = events[events.event_status == "held"]
    lacking = held.apply(lambda e: pd.isna(e.event_end) or baseline.reindex(
        [(e.venue, ts) for ts in prediction_buckets(e.event_end)]).isna().any(), axis=1)
    dropped = lacking[lacking].index
    if len(dropped):
        log.warning("%d events dropped, Prediction Window lacks a Baseline:\n%s", len(dropped),
                    events.loc[dropped, ["event_id", "event_start", "event_end"]].to_string(index=False))
    return events.drop(dropped)
