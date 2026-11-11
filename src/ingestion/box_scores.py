"""Basketball enrichment: Event End and Attendance from ESPN box scores.

Games are matched to Events by venue and date, not team names, so multi-game sessions
(tournaments) work: each game joins the latest Event that started at or before its tip.
"""

import argparse
import json
import logging
import time
from urllib.request import Request, urlopen

import pandas as pd

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR

log = logging.getLogger(__name__)

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/{league}/{endpoint}"
# groups=50 (all of Division I) empties the pro scoreboards, so it is college-only.
LEAGUES = {"nba": "", "wnba": "", "mens-college-basketball": "&groups=50",
           "womens-college-basketball": "&groups=50"}
ESPN_VENUES = {"Barclays Center": "barclays"}
RAW_DIR = RAW_DATA_DIR / "box_scores"
GAMES_PATH = INTERIM_DATA_DIR / "box_scores.csv"
# Longer than any real game, overtimes included; beyond it the wallclock is corrupt.
MAX_GAME = pd.Timedelta("4h")
GAME_COLUMNS = ["venue", "game_id", "game_name", "tip", "end", "attendance"]


def _local(utc):
    return pd.Timestamp(utc).tz_convert("America/New_York").tz_localize(None)


def parse_summary(summary):
    """One game from an ESPN summary: actual tip and final-buzzer wallclock, attendance."""
    plays = [play for play in summary.get("plays", []) if play.get("wallclock")]
    ends = [play["wallclock"] for play in plays if play["type"]["text"] == "End Game"]
    tip = _local(plays[0]["wallclock"] if plays else summary["header"]["competitions"][0]["date"])
    end = _local(ends[-1]) if ends else pd.NaT
    return dict(venue=ESPN_VENUES[summary["gameInfo"]["venue"]["fullName"]],
                game_id=summary["header"]["id"], tip=tip,
                end=end if end - tip <= MAX_GAME else pd.NaT,
                # ESPN reports 0 when it has no figure.
                attendance=summary["gameInfo"].get("attendance") or None)


def held_basketball(events):
    return (events.event_type == "basketball") & (events.event_status == "held")


def fill_blanks(events, column, basis_column, values, basis):
    """Enrichment values go only into blank cells, so reviewed overrides win."""
    fill = events[column].isna() & values.notna()
    events.loc[fill, column] = values[fill]
    events.loc[fill, basis_column] = basis


def apply_box_scores(events, games):
    """Fill blank Event End / Attendance of held basketball Events from their games.
    A session's end is its last game's end, and only when every game has one;
    its attendance is the highest reported for any of its games."""
    basketball = held_basketball(events)
    held = events[basketball & events.event_start.notna()]
    games = games.astype({"tip": "datetime64[ns]", "end": "datetime64[ns]"})
    matched = pd.merge_asof(
        games.assign(date=games.tip.dt.date).sort_values("tip"),
        held[["event_id", "venue", "event_start"]].assign(date=held.event_start.dt.date)
        .sort_values("event_start"),
        left_on="tip", right_on="event_start", by=["venue", "date"])
    per_event = matched.dropna(subset=["event_id"]).groupby("event_id").agg(
        end=("end", lambda ends: ends.max() if ends.notna().all() else pd.NaT),
        attendance=("attendance", "max")).astype({"end": "datetime64[ns]", "attendance": "Int64"})

    events = events.copy()
    per_event = per_event.reindex(events.event_id).set_axis(events.index)
    fill_blanks(events, "event_end", "end_time_basis", per_event.end, "box_score")
    fill_blanks(events, "attendance", "attendance_basis", per_event.attendance, "box_score")

    unmatched = events[basketball & (events.event_end.isna() | events.attendance.isna())]
    if len(unmatched):
        log.warning("%d basketball events missing a box-score end or attendance:\n%s", len(unmatched),
                    unmatched[["event_id", "event_name", "event_end", "attendance"]].to_string(index=False))
    stray = matched[matched.event_id.isna()]
    if len(stray):
        log.warning("%d games matched no event:\n%s", len(stray),
                    stray[["game_id", "game_name", "tip"]].to_string(index=False))
    return events


def read_games(path=GAMES_PATH):
    return pd.read_csv(path, dtype={"game_id": str}, parse_dates=["tip", "end"]).astype(
        {"attendance": "Int64"})


def _fetch_json(url, path):
    # ponytail: a raw file is the cache; rerun after a failure to resume.
    if not path.exists():
        with urlopen(Request(url, headers={"User-Agent": "BarclaysBoxScores/1.0"}), timeout=30) as response:
            path.write_bytes(response.read())
        time.sleep(0.3)
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_games(dates):
    """Every ESPN game at a known venue on the given dates, all basketball leagues."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for date in sorted(set(dates)):
        for league, groups in LEAGUES.items():
            board = _fetch_json(ESPN_URL.format(league=league, endpoint="scoreboard")
                                + f"?dates={date:%Y%m%d}&limit=500{groups}",
                                RAW_DIR / f"{league}_{date:%Y%m%d}.json")
            for game in board["events"]:
                if game["competitions"][0].get("venue", {}).get("fullName") in ESPN_VENUES:
                    summary = _fetch_json(ESPN_URL.format(league=league, endpoint="summary")
                                          + f"?event={game['id']}", RAW_DIR / f"{league}_{game['id']}.json")
                    rows.append({**parse_summary(summary), "game_name": game["name"]})
    return pd.DataFrame(rows, columns=GAME_COLUMNS)


def main():
    from src.processing.events import load_events  # events reads this module's output

    argparse.ArgumentParser(description=f"Fetch basketball box scores into {GAMES_PATH}").parse_args()
    logging.basicConfig(format="%(message)s")
    events = load_events(box_scores_path=None, tour_attendance_path=None, calibrate=False)
    games = fetch_games(events[held_basketball(events)].event_start.dropna().dt.date)
    games.to_csv(GAMES_PATH, index=False)
    print(f"{len(games)} games -> {GAMES_PATH}")
    apply_box_scores(events, read_games())


if __name__ == "__main__":
    main()
