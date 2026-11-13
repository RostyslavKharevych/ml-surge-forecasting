from pathlib import Path

import pandas as pd
import pytest

from scripts.add_event_end_times import add_event_end_times


def test_calculates_end_time_from_start_and_duration():
    events = pd.DataFrame(
        {
            "start_time_local": ["2024-07-01 19:00:00-0400"],
            "duration_minutes": [120],
        }
    )

    result = add_event_end_times(events)

    assert result.loc[0, "end_time_local"] == "2024-07-01 21:00:00-0400"


def test_calculation_preserves_input_dataframe():
    events = pd.DataFrame(
        {
            "start_time_local": ["2024-07-01 19:00:00-0400"],
            "duration_minutes": [120],
        }
    )
    original = events.copy(deep=True)

    result = add_event_end_times(events)

    pd.testing.assert_frame_equal(events, original)
    assert "end_time_local" not in events.columns
    assert "end_time_local" in result.columns


@pytest.mark.parametrize(
    "events, message",
    [
        (pd.DataFrame({"duration_minutes": [120]}), "start_time_local"),
        (pd.DataFrame({"start_time_local": ["2024-07-01 19:00:00-0400"]}), "duration_minutes"),
        (
            pd.DataFrame(
                {
                    "start_time_local": ["not-a-time"],
                    "duration_minutes": [120],
                }
            ),
            "start_time_local",
        ),
        (
            pd.DataFrame(
                {
                    "start_time_local": ["2024-07-01 19:00:00-0400"],
                    "duration_minutes": [None],
                }
            ),
            "duration_minutes",
        ),
        (
            pd.DataFrame(
                {
                    "start_time_local": ["2024-07-01 19:00:00-0400"],
                    "duration_minutes": [-1],
                }
            ),
            "negative",
        ),
    ],
)
def test_rejects_invalid_event_values(events, message):
    with pytest.raises(ValueError, match=message):
        add_event_end_times(events)


def test_event_csv_has_required_complete_unique_records():
    csv_path = Path(__file__).parents[1] / "citi_field_events_2024.csv"
    events = pd.read_csv(csv_path)

    required_columns = {
        "event_id",
        "event_type",
        "event_name",
        "event_date",
        "venue",
        "venue_zone_id",
        "opponent",
        "start_time_local",
        "duration_minutes",
        "mlb_game_id",
        "end_time_local",
    }

    assert required_columns.issubset(events.columns)
    assert events["event_id"].is_unique
    assert events["venue"].eq("Citi Field").all()
    assert events["event_type"].isin(["game", "concert"]).all()
    assert events["event_name"].notna().all()

    games = events[events["event_type"] == "game"]
    concerts = events[events["event_type"] == "concert"]
    assert games["duration_minutes"].gt(0).all()
    assert games["end_time_local"].notna().all()
    assert games["mlb_game_id"].notna().all()
    assert concerts["attendance"].notna().all()
    assert concerts["duration_minutes"].isna().all()
    assert concerts["end_time_local"].isna().all()


def test_event_csv_end_times_match_duration():
    csv_path = Path(__file__).parents[1] / "citi_field_events_2024.csv"
    events = pd.read_csv(csv_path)
    games = events[events["event_type"] == "game"]
    start = pd.to_datetime(games["start_time_local"], utc=True)
    end = pd.to_datetime(games["end_time_local"], utc=True)
    calculated_end = start + pd.to_timedelta(games["duration_minutes"], unit="m")

    pd.testing.assert_series_equal(
        end.reset_index(drop=True),
        calculated_end.reset_index(drop=True),
        check_names=False,
    )


def test_event_csv_is_within_tlc_year_and_sorted():
    csv_path = Path(__file__).parents[1] / "citi_field_events_2024.csv"
    events = pd.read_csv(csv_path)
    dates = pd.to_datetime(events["event_date"])

    assert dates.min() >= pd.Timestamp("2024-01-01")
    assert dates.max() <= pd.Timestamp("2024-12-31")
    assert dates.is_monotonic_increasing
    assert events.loc[events.event_type == "game", "mlb_game_id"].is_unique


def test_event_csv_has_expected_citi_field_home_game_count():
    """The 2024 source should contain 80 regular-season and 5 postseason games."""
    csv_path = Path(__file__).parents[1] / "citi_field_events_2024.csv"
    events = pd.read_csv(csv_path)
    events = events[events.event_type == "game"]

    assert len(events) == 85
    assert (events["event_date"] < "2024-10-01").sum() == 80
    assert (events["event_date"] >= "2024-10-01").sum() == 5


def test_event_csv_contains_requested_concerts_and_attendance():
    csv_path = Path(__file__).parents[1] / "citi_field_events_2024.csv"
    events = pd.read_csv(csv_path)
    concerts = events[events.event_type == "concert"].set_index("event_name")

    assert set(concerts.index) == {"Foo Fighters", "blink-182", "ATEEZ", "Green Day"}
    assert concerts.loc["Foo Fighters", "event_date"] == "2024-07-19"
    assert concerts.loc["Foo Fighters", "attendance"] == 40000
    assert concerts.loc["blink-182", "attendance"] == 34279
    assert concerts.loc["ATEEZ", "attendance"] == 28467
    assert concerts.loc["Green Day", "attendance"] == 35000
    assert concerts["venue_zone_id"].eq(93).all()
    assert concerts["venue"].eq("Citi Field").all()


def test_event_csv_concert_start_times_are_local_and_end_times_are_blank():
    csv_path = Path(__file__).parents[1] / "citi_field_events_2024.csv"
    events = pd.read_csv(csv_path)
    concerts = events[events.event_type == "concert"].set_index("event_name")

    assert concerts.loc["Foo Fighters", "start_time_local"] == "2024-07-19 20:00:00-0400"
    assert concerts.loc["blink-182", "start_time_local"] == "2024-07-21 21:00:00-0400"
    assert concerts.loc["ATEEZ", "start_time_local"] == "2024-08-03 21:30:00-0400"
    assert concerts.loc["Green Day", "start_time_local"] == "2024-08-05 20:30:00-0400"
    assert concerts["end_time_local"].isna().all()


def test_duration_parser_handles_delay_note():
    from scripts.ingest_citi_field_events import _duration_minutes

    feed = {"liveData": {"boxscore": {"info": [{"label": "T", "value": "2:27 (:55 delay)."}]}}}

    assert _duration_minutes(feed) == 147


def test_duration_parser_requires_duration():
    from scripts.ingest_citi_field_events import _duration_minutes

    with pytest.raises(ValueError, match="duration"):
        _duration_minutes({"liveData": {"boxscore": {"info": []}}})


def test_fetch_games_filters_deduplicates_and_keeps_all_games(monkeypatch):
    from scripts import ingest_citi_field_events as ingest

    games = [
        {
            "gamePk": 1,
            "officialDate": "2024-04-01",
            "gameDate": "2024-04-01T23:10:00Z",
            "gameType": "R",
            "venue": {"name": "Citi Field"},
            "status": {"abstractGameState": "Final"},
            "teams": {
                "home": {"team": {"id": 121}},
                "away": {"team": {"name": "Team A"}},
            },
        },
        {
            "gamePk": 1,  # duplicate schedule record
            "officialDate": "2024-04-01",
            "gameDate": "2024-04-01T23:10:00Z",
            "venue": {"name": "Citi Field"},
            "status": {"abstractGameState": "Final"},
            "teams": {
                "home": {"team": {"id": 121}},
                "away": {"team": {"name": "Team A"}},
            },
        },
        {
            "gamePk": 2,
            "officialDate": "2024-04-02",
            "gameDate": "2024-04-02T23:10:00Z",
            "venue": {"name": "Away Park"},
            "status": {"abstractGameState": "Final"},
            "teams": {
                "home": {"team": {"id": 121}},
                "away": {"team": {"name": "Team B"}},
            },
        },
        {
            "gamePk": 3,
            "officialDate": "2024-04-03",
            "gameDate": "2024-04-03T23:10:00Z",
            "venue": {"name": "Citi Field"},
            "status": {"abstractGameState": "Preview"},
            "teams": {
                "home": {"team": {"id": 121}},
                "away": {"team": {"name": "Team C"}},
            },
        },
    ]

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, params=None, timeout=None):
            if url.endswith("/schedule"):
                return FakeResponse({"dates": [{"games": games}]})
            return FakeResponse(
                {
                    "gameData": {"datetime": {"dateTime": "2024-04-01T23:10:00Z"}},
                    "liveData": {"boxscore": {"info": [{"label": "T", "value": "2:30"}]}},
                }
            )

    monkeypatch.setattr(ingest.requests, "Session", FakeSession)
    result = ingest.fetch_games(2024)

    assert len(result) == 1
    assert result.loc[0, "mlb_game_id"] == 1
    assert result.loc[0, "venue"] == "Citi Field"
    assert result.loc[0, "duration_minutes"] == 150
    assert result.loc[0, "event_type"] == "game"
