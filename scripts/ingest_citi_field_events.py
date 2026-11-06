"""Fetch completed Mets home games at Citi Field and write the event CSV.

Uses MLB's public Stats API for game dates, opponents, start times, game IDs,
and official box-score durations. End times are calculated by the project's
``add_event_end_times`` function.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

try:
    # Package import used by tests and other Python modules.
    from .add_event_end_times import add_event_end_times
except ImportError:  # pragma: no cover - used when run as a file
    from add_event_end_times import add_event_end_times

API_ROOT = "https://statsapi.mlb.com/api/v1"
METS_TEAM_ID = 121
CITI_FIELD = "Citi Field"
CITI_FIELD_ZONE_ID = 93
NEW_YORK = ZoneInfo("America/New_York")
OUTPUT_COLUMNS = [
    "event_id",
    "event_date",
    "venue",
    "venue_zone_id",
    "opponent",
    "start_time_local",
    "duration_minutes",
    "mlb_game_id",
    "end_time_local",
]


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


def fetch_games(year: int) -> pd.DataFrame:
    """Fetch every completed Mets home game at Citi Field for ``year``."""

    with requests.Session() as session:
        session.headers["User-Agent"] = "citi-field-eda/1.0"
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
                    "event_date": game["officialDate"],
                    "venue": CITI_FIELD,
                    "venue_zone_id": CITI_FIELD_ZONE_ID,
                    "opponent": game["teams"]["away"]["team"]["name"],
                    "start_time_local": start_local.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "duration_minutes": _duration_minutes(feed),
                    "mlb_game_id": game_id,
                }
            )

    events = pd.DataFrame(rows, columns=OUTPUT_COLUMNS[:-1])
    if events.empty:
        raise ValueError(f"No completed Citi Field Mets games found for {year}.")
    events = add_event_end_times(events)
    return events[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--output", type=Path, default=Path("citi_field_events_2024.csv")
    )
    args = parser.parse_args()

    events = fetch_games(args.year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output, index=False)
    monthly = events.groupby(events["event_date"].str[:7]).size()
    print(f"Wrote {len(events)} events to {args.output}")
    print("Events by month:")
    print(monthly.to_string())


if __name__ == "__main__":
    main()
