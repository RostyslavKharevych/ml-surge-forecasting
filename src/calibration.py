"""Calibrated category durations: Event End for held Events with no specific end source.

A category's duration is the median minutes from Event Start to the raw core-zone demand peak
within Event Start + 5h, over training-period Events only. It needs no Baseline, so every
Event End is final before Affected Slots and Baselines are built.
"""

import argparse

import pandas as pd

from src.box_scores import fill_blanks
from src.config import TRAINING_PERIOD
from src.features import consolidate_demand_matrices

PEAK_SEARCH = pd.Timedelta(hours=5)
POOLED = "all"  # duration for categories with no training-period Events


def peak_minutes(events, demand):
    """Minutes from each Event's start to its venue's highest core-zone bucket within Start + 5h."""
    core = demand.set_index(["venue", "timestamp_15m"]).core_trips.sort_index()
    return events.apply(lambda e: (core[e.venue][e.event_start:e.event_start + PEAK_SEARCH].idxmax()
                                   - e.event_start) / pd.Timedelta(minutes=1), axis=1)


def training_events(events):
    return events[(events.event_status == "held") & events.event_start.between(*TRAINING_PERIOD)]


def category_durations(events, demand):
    """Median start-to-peak minutes per event_type over held training-period Events,
    plus a pooled median under POOLED."""
    train = training_events(events)
    minutes = peak_minutes(train, demand)
    return pd.concat([minutes.groupby(train.event_type).median(),
                      pd.Series({POOLED: minutes.median()})])


def apply_calibrated_ends(events, durations):
    """Event End = Event Start + its category's duration, for held Events whose end is blank."""
    events = events.copy()
    minutes = events.event_type.map(durations).fillna(durations[POOLED])
    ends = events.event_start + pd.to_timedelta(minutes, unit="min")
    fill_blanks(events, "event_end", "end_time_basis",
                ends.where(events.event_status == "held"), "calibrated_category")
    return events


def main():
    from src.events import load_events  # events reads this module
    argparse.ArgumentParser(description="Report calibrated category durations and validate "
                                        "them on basketball box-score ends").parse_args()
    demand = consolidate_demand_matrices()
    events = load_events(calibrate=False)
    durations = category_durations(events, demand)
    train = training_events(events)
    print("Minutes from Event Start to core-zone peak, training period (count, median):")
    print(pd.DataFrame({"events": train.event_type.value_counts(), "minutes": durations}))

    # Scored on games outside the training period, so the fit never sees them.
    games = events[(events.end_time_basis == "box_score") & events.event_start.notna()]
    games = games.drop(training_events(events).index, errors="ignore")
    true = (games.event_end - games.event_start) / pd.Timedelta(minutes=1)
    peak_error = peak_minutes(games, demand) - true
    applied_error = durations["basketball"] - true
    print(f"\nBasketball, {len(games)} box-score Events (minutes, + = later than the buzzer):")
    print(f"  raw peak vs end:          median {peak_error.median():.1f}, "
          f"median |error| {peak_error.abs().median():.1f}")
    print(f"  calibrated end vs end:    median {applied_error.median():.1f}, "
          f"median |error| {applied_error.abs().median():.1f}")
    print("  raw peak error by weekday:",
          peak_error.groupby(games.event_start.dt.day_name()).median().round().to_dict())


if __name__ == "__main__":
    main()
