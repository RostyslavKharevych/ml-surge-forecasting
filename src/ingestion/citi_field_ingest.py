"""Fetch completed Mets home games at Citi Field and write per-year event CSVs.

Uses MLB's public Stats API for game dates, opponents, start times, game IDs,
and official box-score durations, combined with configured Citi Field concerts.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.config import ANALYSIS_YEARS, INTERIM_DATA_DIR, VENUE_CONFIG

API_ROOT = "https://statsapi.mlb.com/api/v1"
METS_TEAM_ID = 121
CITI_FIELD = "Citi Field"
CITI_FIELD_ZONE_ID = VENUE_CONFIG["citi_field"]["core_zone"]
NEW_YORK = ZoneInfo("America/New_York")
OUTPUT_COLUMNS = [
    "event_id",
    "event_type",
    "event_name",
    "event_date",
    "venue",
    "venue_zone_id",
    "opponent",
    "start_time_local",
    "attendance",
    "duration_minutes",
    "mlb_game_id",
    "end_time_local",
]

# Headliner start times and attendance figures supplied for this project.
# Green Day's number is a tour-average figure, not a Citi Field-specific count.
CONCERTS_2024 = [
    {
        "event_id": "CONCERT_FOO_FIGHTERS_20240719",
        "event_type": "concert",
        "event_name": "Foo Fighters",
        "event_date": "2024-07-19",
        "venue": CITI_FIELD,
        "venue_zone_id": CITI_FIELD_ZONE_ID,
        "opponent": None,
        "start_time_local": "2024-07-19 20:00:00-0400",
        "attendance": 40000,
        "duration_minutes": None,
        "mlb_game_id": None,
        "end_time_local": None,
    },
    {
        "event_id": "CONCERT_BLINK182_20240721",
        "event_type": "concert",
        "event_name": "blink-182",
        "event_date": "2024-07-21",
        "venue": CITI_FIELD,
        "venue_zone_id": CITI_FIELD_ZONE_ID,
        "opponent": None,
        "start_time_local": "2024-07-21 21:00:00-0400",
        "attendance": 34279,
        "duration_minutes": None,
        "mlb_game_id": None,
        "end_time_local": None,
    },
    {
        "event_id": "CONCERT_ATEEZ_20240803",
        "event_type": "concert",
        "event_name": "ATEEZ",
        "event_date": "2024-08-03",
        "venue": CITI_FIELD,
        "venue_zone_id": CITI_FIELD_ZONE_ID,
        "opponent": None,
        "start_time_local": "2024-08-03 21:30:00-0400",
        "attendance": 28467,
        "duration_minutes": None,
        "mlb_game_id": None,
        "end_time_local": None,
    },
    {
        "event_id": "CONCERT_GREEN_DAY_20240805",
        "event_type": "concert",
        "event_name": "Green Day",
        "event_date": "2024-08-05",
        "venue": CITI_FIELD,
        "venue_zone_id": CITI_FIELD_ZONE_ID,
        "opponent": None,
        "start_time_local": "2024-08-05 20:30:00-0400",
        "attendance": 35000,
        "duration_minutes": None,
        "mlb_game_id": None,
        "end_time_local": None,
    },
]

# Placeholder for curated 2025 Citi Field concerts; append dicts matching OUTPUT_COLUMNS.
CONCERTS_2025: list[dict] = []

CONCERTS_BY_YEAR = {
    2024: CONCERTS_2024,
    2025: CONCERTS_2025,
}


def citi_field_events_path(year: int, output_dir: Path = INTERIM_DATA_DIR) -> Path:
    """Return the canonical interim CSV path for a calendar year of Citi Field events."""
    return Path(output_dir) / f"citi_field_events_{year}.csv"


def _get_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    response = session.get(url, params=params, timeout=45)
    response.raise_for_status()
    return response.json()


def _duration_minutes(feed: dict) -> int:
    """Read the official game duration from the MLB box-score info field."""
    for item in feed.get("liveData", {}).get("boxscore", {}).get("info", []):
        if item.get("label") == "T":
            match = re.search(r"(\d+):(\d{2})", item.get("value", ""))
            if match:
                return int(match.group(1)) * 60 + int(match.group(2))
    raise ValueError("MLB box score did not include a parseable game duration.")


def _boxscore_attendance(feed: dict) -> Optional[int]:
    """Extract numeric game attendance from MLB live feed if available."""
    raw_att = feed.get("gameData", {}).get("gameInfo", {}).get("attendance")
    if isinstance(raw_att, (int, float)) and raw_att > 0:
        return int(raw_att)
    for item in feed.get("liveData", {}).get("boxscore", {}).get("info", []):
        if item.get("label") == "Att":
            digits = re.sub(r"[^\d]", "", str(item.get("value", "")))
            if digits:
                return int(digits)
    return None


def add_citi_field_end_times(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate local end times from start_time_local + duration_minutes for non-concerts."""
    required = {"start_time_local", "duration_minutes"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

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
    if needs_end_time.any():
        end_local = (
            start_utc[needs_end_time]
            + pd.to_timedelta(duration[needs_end_time], unit="m")
        ).dt.tz_convert("America/New_York")
        result.loc[needs_end_time, "end_time_local"] = end_local.dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
    return result


def fetch_games(year: int) -> pd.DataFrame:
    """Fetch every completed Mets home game at Citi Field for ``year``."""
    with requests.Session() as session:
        session.headers["User-Agent"] = "SurgeEgress/1.0 (citi-field-ingest)"
        schedule = _get_json(
            session,
            f"{API_ROOT}/schedule",
            {
                "sportId": 1,
                "teamId": METS_TEAM_ID,
                "startDate": f"{year}-01-01",
                "endDate": f"{year}-12-31",
                "hydrate": "venue,team",
            },
        )

        # Deduplicate gamePk because MLB schedule responses may list a game
        # under more than one date (for example, suspended/resumed games).
        unique_games = {}
        for date_entry in schedule.get("dates", []):
            for game in date_entry.get("games", []):
                if game.get("status", {}).get("abstractGameState") != "Final":
                    continue
                if game.get("venue", {}).get("name") != CITI_FIELD:
                    continue
                if game.get("teams", {}).get("home", {}).get("team", {}).get("id") != METS_TEAM_ID:
                    continue
                unique_games[game["gamePk"]] = game

        selected = sorted(
            unique_games.values(), key=lambda g: (g["officialDate"], g["gamePk"])
        )

        rows = []
        for game in selected:
            game_id = game["gamePk"]
            feed = _get_json(session, f"{API_ROOT}.1/game/{game_id}/feed/live")
            start_utc = datetime.fromisoformat(
                feed["gameData"]["datetime"]["dateTime"].replace("Z", "+00:00")
            )
            start_local = start_utc.astimezone(NEW_YORK)
            rows.append(
                {
                    "event_id": f"METS_{game['officialDate'].replace('-', '')}_{game_id}",
                    "event_type": "game",
                    "event_name": f"Mets vs. {game['teams']['away']['team']['name']}",
                    "event_date": game["officialDate"],
                    "venue": CITI_FIELD,
                    "venue_zone_id": CITI_FIELD_ZONE_ID,
                    "opponent": game["teams"]["away"]["team"]["name"],
                    "start_time_local": start_local.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "attendance": _boxscore_attendance(feed),
                    "duration_minutes": _duration_minutes(feed),
                    "mlb_game_id": game_id,
                }
            )

    events = pd.DataFrame(rows, columns=OUTPUT_COLUMNS[:-1])
    if events.empty:
        raise ValueError(f"No completed Citi Field Mets games found for {year}.")
    events = add_citi_field_end_times(events)
    return events[OUTPUT_COLUMNS]


def fetch_events(year: int) -> pd.DataFrame:
    """Return MLB games plus the configured Citi Field concerts for ``year``."""
    games = fetch_games(year)
    concerts_list = CONCERTS_BY_YEAR.get(year, [])
    concerts = pd.DataFrame(concerts_list, columns=OUTPUT_COLUMNS) if concerts_list else pd.DataFrame(columns=OUTPUT_COLUMNS)
    frames = [df for df in (games, concerts) if not df.empty]
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if events.empty:
        return events
    events["event_date"] = pd.to_datetime(events["event_date"])
    events = events.sort_values(["event_date", "start_time_local", "event_id"])
    events["event_date"] = events["event_date"].dt.strftime("%Y-%m-%d")
    return events[OUTPUT_COLUMNS].reset_index(drop=True)


def run_citi_field_ingestion(
    years: Iterable[int] = ANALYSIS_YEARS,
    output_dir: Path = INTERIM_DATA_DIR,
) -> list[Path]:
    """Fetch and write Citi Field event CSVs for each requested year."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for year in years:
        events = fetch_events(year)
        out_path = citi_field_events_path(year, output_dir=output_dir)
        events.to_csv(out_path, index=False)
        written.append(out_path)
        monthly = events.groupby(events["event_date"].str[:7]).size()
        print(f"Wrote {len(events)} events to {out_path}")
        print(monthly.to_string())
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=ANALYSIS_YEARS)
    parser.add_argument("--output-dir", type=Path, default=INTERIM_DATA_DIR)
    args = parser.parse_args()
    run_citi_field_ingestion(years=args.years, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
