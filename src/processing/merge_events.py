"""Calculate game end times and match pickup intervals to event windows."""

from __future__ import annotations

import pandas as pd


def add_event_end_times(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate end times for rows with durations; preserve concert blanks."""
    required = {"start_time_local", "duration_minutes"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(sorted(missing))))

    result = events.copy()
    if "event_type" in result.columns:
        needs_end_time = result["event_type"].ne("concert")
    else:
        needs_end_time = pd.Series(True, index=result.index)

    start_utc = pd.to_datetime(result["start_time_local"], utc=True, errors="coerce")
    duration = pd.to_numeric(result["duration_minutes"], errors="coerce")
    if start_utc[needs_end_time].isna().any():
        raise ValueError("One or more start_time_local values could not be parsed.")
    if duration[needs_end_time].isna().any():
        raise ValueError("One or more duration_minutes values are missing or invalid.")
    if (duration[needs_end_time] < 0).any():
        raise ValueError("duration_minutes cannot contain negative values.")

    result["end_time_local"] = pd.NA
    end_local = (
        start_utc[needs_end_time]
        + pd.to_timedelta(duration[needs_end_time], unit="m")
    ).dt.tz_convert("America/New_York")
    result.loc[needs_end_time, "end_time_local"] = end_local.dt.strftime(
        "%Y-%m-%d %H:%M:%S%z"
    )
    return result


def prepare_game_event_windows(
    events: pd.DataFrame, before_minutes: int = 30, after_minutes: int = 90
) -> pd.DataFrame:
    """Return validated game rows with UTC times and configured event windows."""
    games = events[events["event_type"] == "game"].copy()
    games["start_time_dt"] = pd.to_datetime(
        games["start_time_local"], utc=True, errors="coerce"
    )
    games["end_time_dt"] = pd.to_datetime(
        games["end_time_local"], utc=True, errors="coerce"
    )
    invalid = (
        games["start_time_dt"].isna()
        | games["end_time_dt"].isna()
        | (games["end_time_dt"] <= games["start_time_dt"])
    )
    if invalid.any():
        raise ValueError("Game events contain invalid or reversed time ranges.")
    games["window_start"] = games["start_time_dt"] - pd.Timedelta(minutes=before_minutes)
    games["window_end"] = games["end_time_dt"] + pd.Timedelta(minutes=after_minutes)
    return games


def label_pickup_intervals(
    pickup_counts: pd.DataFrame, game_events: pd.DataFrame
) -> pd.DataFrame:
    """Label each pickup interval before, during, after, or outside a game."""
    result = pickup_counts.copy()
    result["pickup_time_dt"] = (
        pd.to_datetime(result["time_bin"], errors="coerce")
        .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    )
    result["event_id"] = pd.NA
    result["event_date"] = pd.NA
    result["opponent"] = pd.NA
    result["event_phase"] = "no_event"

    phases = [
        ("during_game", "start_time_dt", "end_time_dt"),
        ("before_game", "window_start", "start_time_dt"),
        ("after_game", "end_time_dt", "window_end"),
    ]
    for phase, start_col, end_col in phases:
        for _, event in game_events.iterrows():
            match = (
                (result["pickup_time_dt"] >= event[start_col])
                & (result["pickup_time_dt"] < event[end_col])
                & result["event_phase"].eq("no_event")
            )
            result.loc[match, "event_id"] = event["event_id"]
            result.loc[match, "event_date"] = event["event_date"]
            result.loc[match, "opponent"] = event["opponent"]
            result.loc[match, "event_phase"] = phase
    return result
