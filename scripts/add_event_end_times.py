"""Calculate event end times from start times and durations.

The source event CSV intentionally stores only observed start times and
durations. This script derives the end time whenever the event data is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_event_end_times(events: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``events`` with a calculated ``end_time_local`` column."""
    required = {"start_time_local", "duration_minutes"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    result = events.copy()
    start_utc = pd.to_datetime(result["start_time_local"], utc=True, errors="coerce")
    duration = pd.to_numeric(result["duration_minutes"], errors="coerce")

    if start_utc.isna().any():
        raise ValueError("One or more start_time_local values could not be parsed.")
    if duration.isna().any():
        raise ValueError("One or more duration_minutes values are missing or invalid.")
    if (duration < 0).any():
        raise ValueError("duration_minutes cannot contain negative values.")

    end_local = (start_utc + pd.to_timedelta(duration, unit="m")).dt.tz_convert(
        "America/New_York"
    )
    result["end_time_local"] = end_local.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path("citi_field_events_2024.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("citi_field_events_2024_with_end_times.csv"),
    )
    args = parser.parse_args()

    events = pd.read_csv(args.input_csv)
    result = add_event_end_times(events)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} events to {args.output}")


if __name__ == "__main__":
    main()
