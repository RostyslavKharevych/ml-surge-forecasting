"""The venue-agnostic Event table: one adapter per venue maps into it, reviewed overrides go on top."""

from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.ingestion.box_scores import GAMES_PATH, apply_box_scores, fill_blanks, read_games
from src.processing.calibration import apply_calibrated_ends, category_durations
from src.config import ANALYSIS_YEARS, INTERIM_DATA_DIR
from src.features.features import consolidate_demand_matrices
from src.ingestion.barclays_ingest import output_dir_for_year
from src.ingestion.citi_field_ingest import citi_field_events_path
from src.ingestion.tour_boxscores import ATTENDANCE_PATH, apply_tour_attendance, read_tour_attendance

EVENT_COLUMNS = ["event_id", "venue", "event_name", "event_type", "event_start", "event_end",
                 "end_time_basis", "attendance", "attendance_basis", "event_status"]
OVERRIDES_PATH = Path(__file__).with_name("event_overrides.csv")
OVERRIDE_KEY = ["venue", "event_date", "event_name"]
RUNTIMES_PATH = Path(__file__).with_name("event_runtimes.csv")
RUNTIME_KEY = ["venue", "event_name", "year"]
CITI_FIELD_EVENTS_PATH = INTERIM_DATA_DIR / "citi_field_events_2024.csv"
# `duplicate`: a listing for an Event already in the table (e.g. a suite package for a game);
# set only by override, and like cancelled it affects nothing.
EVENT_STATUSES = {"held", "cancelled", "postponed", "duplicate"}
# A scraper "unknown" means the page carried no cancel/postpone notice; a rescheduled
# showing is listed on the date it was actually held.
OVERRIDE_FIELDS = {"event_start": pd.Timestamp, "event_end": pd.Timestamp, "end_time_basis": str,
                   "attendance": int, "attendance_basis": str, "event_status": str}
BARCLAYS_STATUS = {"unknown": "held", "rescheduled": "held",
                   "cancelled": "cancelled", "postponed": "postponed"}
CITI_FIELD_EVENT_TYPES = {"game": "baseball"}

# 2024-2025 NYC DOE public school holidays and multi-day recesses (excluding staff-only
# Chancellor's Conference / Election Day closures).
_NYC_SCHOOL_HOLIDAY_RANGES = [
    ("2024-01-01", "2024-01-01"),  # Winter Recess / New Year's Day
    ("2024-01-15", "2024-01-15"),  # MLK Jr. Day
    ("2024-02-19", "2024-02-23"),  # Midwinter Recess
    ("2024-04-10", "2024-04-10"),  # Eid al-Fitr
    ("2024-04-22", "2024-04-30"),  # Spring Recess
    ("2024-05-27", "2024-05-27"),  # Memorial Day
    ("2024-06-19", "2024-06-19"),  # Juneteenth
    ("2024-10-03", "2024-10-04"),  # Rosh Hashanah
    ("2024-10-14", "2024-10-14"),  # Italian Heritage / Indigenous Peoples' Day
    ("2024-11-01", "2024-11-01"),  # Diwali
    ("2024-11-11", "2024-11-11"),  # Veterans Day
    ("2024-11-28", "2024-11-29"),  # Thanksgiving Recess
    ("2024-12-23", "2025-01-01"),  # Winter Recess
    ("2025-01-20", "2025-01-20"),  # MLK Jr. Day
    ("2025-01-29", "2025-01-29"),  # Lunar New Year
    ("2025-02-17", "2025-02-21"),  # Midwinter Recess
    ("2025-03-31", "2025-03-31"),  # Eid al-Fitr
    ("2025-04-14", "2025-04-18"),  # Spring Recess
    ("2025-05-26", "2025-05-26"),  # Memorial Day
    ("2025-06-19", "2025-06-19"),  # Juneteenth
    ("2025-09-23", "2025-09-24"),  # Rosh Hashanah
    ("2025-10-02", "2025-10-02"),  # Yom Kippur
    ("2025-10-13", "2025-10-13"),  # Italian Heritage / Indigenous Peoples' Day
    ("2025-10-20", "2025-10-20"),  # Diwali
    ("2025-11-11", "2025-11-11"),  # Veterans Day
    ("2025-11-27", "2025-11-28"),  # Thanksgiving Recess
    ("2025-12-24", "2025-12-31"),  # Winter Recess
]
NYC_SCHOOL_HOLIDAYS = pd.DatetimeIndex(
    sorted({day for start, end in _NYC_SCHOOL_HOLIDAY_RANGES for day in pd.date_range(start, end, freq="D")})
)


def is_federal_holiday(timestamps: pd.Series) -> pd.Series:
    """Boolean per timestamp: its calendar day is a US federal holiday."""
    if timestamps.empty:
        return pd.Series(dtype=bool, index=timestamps.index)
    days = pd.to_datetime(timestamps).dt.normalize()
    return days.isin(USFederalHolidayCalendar().holidays(days.min(), days.max()))


def is_nyc_school_holiday(timestamps: pd.Series) -> pd.Series:
    """Boolean per timestamp: its calendar day is a 2024-2025 NYC public school holiday/break."""
    if timestamps.empty:
        return pd.Series(dtype=bool, index=timestamps.index)
    days = pd.to_datetime(timestamps).dt.normalize()
    return days.isin(NYC_SCHOOL_HOLIDAYS)


def is_holiday(timestamps: pd.Series) -> pd.Series:
    """Boolean per timestamp: US federal holiday or NYC public school holiday."""
    return is_federal_holiday(timestamps) | is_nyc_school_holiday(timestamps)


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


def barclays_events(years=ANALYSIS_YEARS, missing_ok: bool = False):
    paths = [output_dir_for_year(year) / "events_all.csv" for year in years]
    existing = [p for p in paths if p.exists()]
    if not existing:
        if missing_ok:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        raise FileNotFoundError(f"No Barclays event CSVs found among: {paths}")
    scrape = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in existing])
    return from_barclays_scrape(scrape.reset_index(drop=True))


def _to_naive_ny(series: pd.Series) -> pd.Series:
    """Parse timestamps (with or without UTC offset) into naive America/New_York datetimes."""
    def _parse(val):
        if pd.isna(val) or str(val).strip() == "":
            return pd.NaT
        ts = pd.Timestamp(val)
        if ts.tzinfo is not None:
            return ts.tz_convert("America/New_York").tz_localize(None)
        return ts

    return pd.to_datetime(series.map(_parse))


def from_citi_field_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Map rows from the Citi Field ingestion CSV into the canonical Event table."""
    if raw.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    start = _to_naive_ny(raw["start_time_local"])
    if "end_time_local" in raw.columns:
        end = _to_naive_ny(raw["end_time_local"])
    else:
        end = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    if "duration_minutes" in raw.columns:
        duration = pd.to_numeric(raw["duration_minutes"], errors="coerce")
        end = end.fillna(start + pd.to_timedelta(duration, unit="m"))

    event_type = raw["event_type"].replace(CITI_FIELD_EVENT_TYPES)
    if "end_time_basis" in raw.columns:
        raw_end_basis = raw["end_time_basis"].fillna("").astype(str)
    else:
        raw_end_basis = pd.Series("", index=raw.index)
    default_end_basis = event_type.map({"baseball": "box_score"}).fillna("reported")
    end_time_basis = raw_end_basis.where(raw_end_basis != "", default_end_basis).where(end.notna(), "")

    attendance = pd.to_numeric(raw["attendance"], errors="coerce").round().astype("Int64") if "attendance" in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="Int64")
    if "attendance_basis" in raw.columns:
        raw_att_basis = raw["attendance_basis"].fillna("").astype(str)
    else:
        raw_att_basis = pd.Series("", index=raw.index)
    attendance_basis = raw_att_basis.where(raw_att_basis != "", "reported").where(attendance.notna(), "")

    if "event_status" in raw.columns:
        status = raw["event_status"].replace("", "held").fillna("held")
    else:
        status = pd.Series("held", index=raw.index)

    event_date = raw["event_date"] if "event_date" in raw.columns else start.dt.strftime("%Y-%m-%d")
    events = pd.DataFrame({
        "venue": "citi_field",
        "event_name": raw["event_name"],
        "event_type": event_type,
        "event_start": start,
        "event_end": end,
        "end_time_basis": end_time_basis,
        "attendance": attendance,
        "attendance_basis": attendance_basis,
        "event_status": status,
        "event_date": event_date,
    })
    return _finalise(events)


def citi_field_events(
    years=ANALYSIS_YEARS,
    path: Path | None = None,
    missing_ok: bool = False,
) -> pd.DataFrame:
    """Load and adapt Citi Field events for ``years`` (or ``path``) into the Event table."""
    paths = [Path(path)] if path is not None else [citi_field_events_path(year) for year in years]
    existing = [p for p in paths if p.exists()]
    if not existing:
        if missing_ok:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        raise FileNotFoundError(f"No Citi Field event CSVs found among: {paths}")
    raw = pd.concat(
        [pd.read_csv(p, dtype=str, keep_default_na=False) for p in existing],
        ignore_index=True,
    )
    return from_citi_field_events(raw)


def apply_overrides(events, overrides):
    """Apply hand-verified facts. Each override key must match exactly one scraped showing;
    several override rows for one key split that showing into several Events."""
    if events.empty:
        return events
    overrides = overrides.fillna("").astype(str)
    overrides = overrides[overrides["venue"].isin(events["venue"].unique())]
    if overrides.empty:
        return events
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
    if events.empty:
        return events
    runtimes = runtimes[runtimes["venue"].isin(events["venue"].unique())]
    if runtimes.empty:
        return events
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
                runtimes_path=RUNTIMES_PATH, tour_attendance_path=ATTENDANCE_PATH,
                citi_field_path=None, include_barclays=True, include_citi_field=True, calibrate=True):
    """The Event table the merge reads. Combines Barclays and (when present) Citi Field events;
    reviewed overrides win over box scores, tour box scores and runtimes; calibrated category
    durations fill the ends still blank. A None enrichment path skips that enrichment."""
    overrides = pd.read_csv(overrides_path, dtype=str, keep_default_na=False)
    frames = []
    if include_barclays:
        barclays_df = barclays_events(years, missing_ok=True)
        if not barclays_df.empty:
            frames.append(barclays_df)
    if citi_field_path is not None:
        frames.append(citi_field_events(path=citi_field_path))
    elif include_citi_field:
        citi_df = citi_field_events(years=years, missing_ok=True)
        if not citi_df.empty:
            frames.append(citi_df)
    if not frames:
        raise FileNotFoundError("No Barclays or Citi Field event CSVs found for the requested years.")
    base_events = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    events = apply_overrides(base_events, overrides)
    if box_scores_path is not None and Path(box_scores_path).exists():
        events = apply_box_scores(events, read_games(box_scores_path))
    if tour_attendance_path is not None and Path(tour_attendance_path).exists():
        events = apply_tour_attendance(events, read_tour_attendance(tour_attendance_path))
    events = apply_runtimes(events, pd.read_csv(runtimes_path, dtype=str, keep_default_na=False))
    if calibrate:
        events = apply_calibrated_ends(events, category_durations(events, consolidate_demand_matrices()))
    return events


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
