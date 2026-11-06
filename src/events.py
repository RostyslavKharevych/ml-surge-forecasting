"""The venue-agnostic Event table: one adapter per venue maps into it, reviewed overrides go on top."""

from pathlib import Path

import pandas as pd

from src.box_scores import GAMES_PATH, apply_box_scores, fill_blanks, read_games
from src.config import ANALYSIS_YEARS
from src.ingest_events import output_dir_for_year

EVENT_COLUMNS = ["event_id", "venue", "event_name", "event_type", "event_start", "event_end",
                 "end_time_basis", "attendance", "attendance_basis", "event_status"]
OVERRIDES_PATH = Path(__file__).with_name("event_overrides.csv")
OVERRIDE_KEY = ["venue", "event_date", "event_name"]
RUNTIMES_PATH = Path(__file__).with_name("event_runtimes.csv")
RUNTIME_KEY = ["venue", "event_name", "year"]
EVENT_STATUSES = {"held", "cancelled", "postponed"}
# A scraper "unknown" means the page carried no cancel/postpone notice; a rescheduled
# showing is listed on the date it was actually held.
OVERRIDE_FIELDS = {"event_start": pd.Timestamp, "event_end": pd.Timestamp, "end_time_basis": str,
                   "attendance": int, "attendance_basis": str, "event_status": str}
BARCLAYS_STATUS = {"unknown": "held", "rescheduled": "held",
                   "cancelled": "cancelled", "postponed": "postponed"}


def from_barclays_scrape(scrape):
    """Map rows of the Barclays scraper's events_all.csv into the Event table."""
    start = pd.to_datetime((scrape.event_date + " " + scrape.scheduled_start_time)
                           .where(scrape.scheduled_start_time != ""))
    events = pd.DataFrame({
        "venue": scrape.venue, "event_name": scrape.event_name,
        "event_type": scrape.category_type, "event_start": start,
        "event_end": pd.NaT, "end_time_basis": "", "attendance": pd.NA, "attendance_basis": "",
        "event_status": scrape.event_status.map(BARCLAYS_STATUS),
        "event_date": scrape.event_date,
    })
    if events.event_status.isna().any():
        raise ValueError(f"Unmapped scraper status: {set(scrape.event_status) - set(BARCLAYS_STATUS)}")
    return _finalise(events)


def barclays_events(years=ANALYSIS_YEARS):
    scrape = pd.concat([pd.read_csv(output_dir_for_year(year) / "events_all.csv",
                                    dtype=str, keep_default_na=False) for year in years])
    return from_barclays_scrape(scrape.reset_index(drop=True))


def apply_overrides(events, overrides):
    """Apply hand-verified facts. Each override key must match exactly one scraped showing;
    several override rows for one key split that showing into several Events."""
    overrides = overrides.fillna("").astype(str)
    # The id carries the showing date even when the start time is TBA.
    events = events.assign(event_date=pd.to_datetime(
        events.event_id.str.split("_").str[-2], format="%Y%m%d").dt.strftime("%Y-%m-%d"))
    keep, added = pd.Series(True, index=events.index), []
    for key, group in overrides.groupby(OVERRIDE_KEY, sort=False):
        match = (events[OVERRIDE_KEY] == key).all(axis=1)
        if match.sum() != 1:
            raise ValueError(f"Override {key} matches {match.sum()} events, expected 1")
        keep &= ~match
        base = events[match].iloc[0]
        for _, fix in group.iterrows():
            row = base.copy()
            for column, convert in OVERRIDE_FIELDS.items():
                if fix[column]:
                    row[column] = convert(fix[column])
            added.append(row)
    return _finalise(pd.concat([events[keep], pd.DataFrame(added)]).sort_values(
        "event_start", kind="stable").reset_index(drop=True))


def apply_runtimes(events, runtimes):
    """Event End = Event Start + a production's published runtime, for held showings whose
    end is still blank. A production is keyed by (venue, event_name, year); each runtime row
    must match at least one showing."""
    events = events.copy()
    key = events[["venue", "event_name"]].assign(year=events.event_start.dt.year.astype("Int64").astype(str))
    for _, row in runtimes.iterrows():
        match = (key[RUNTIME_KEY] == row[RUNTIME_KEY]).all(axis=1)
        if not match.any():
            raise ValueError(f"Runtime {list(row[RUNTIME_KEY])} matches no events")
        ends = events.event_start + pd.Timedelta(minutes=int(row.runtime_minutes))
        fill_blanks(events, "event_end", "end_time_basis",
                    ends.where(match & (events.event_status == "held")), "published_runtime")
    return events


def load_events(years=ANALYSIS_YEARS, overrides_path=OVERRIDES_PATH, box_scores_path=GAMES_PATH,
                runtimes_path=RUNTIMES_PATH):
    """The Event table the merge reads. Reviewed overrides win over box scores and runtimes;
    box_scores_path=None skips the box-score enrichment."""
    overrides = pd.read_csv(overrides_path, dtype=str, keep_default_na=False)
    events = apply_overrides(barclays_events(years), overrides)
    if box_scores_path is not None:
        events = apply_box_scores(events, read_games(box_scores_path))
    return apply_runtimes(events, pd.read_csv(runtimes_path, dtype=str, keep_default_na=False))


def _finalise(events):
    events = events.astype({"event_start": "datetime64[ns]", "event_end": "datetime64[ns]",
                            "attendance": "Int64"})
    # An overridden start may move the date; TBA showings keep their listed date.
    date = events.event_start.dt.strftime("%Y%m%d").fillna(events.event_date.str.replace("-", ""))
    events["event_id"] = (events.venue.str.upper() + "_" + date + "_"
                          + events.event_start.dt.strftime("%H%M").fillna("TBA"))
    if events.event_id.duplicated().any():
        raise ValueError(f"Duplicate event_id: {list(events.event_id[events.event_id.duplicated()])}")
    if ((events.event_end.notna()) != (events.end_time_basis != "")).any():
        raise ValueError("Every event_end needs an end_time_basis, and vice versa")
    if ((events.attendance.notna()) != (events.attendance_basis != "")).any():
        raise ValueError("Every attendance needs an attendance_basis, and vice versa")
    if not events.event_status.isin(EVENT_STATUSES).all():
        raise ValueError(f"Invalid event_status: {set(events.event_status)}")
    return events[EVENT_COLUMNS]
