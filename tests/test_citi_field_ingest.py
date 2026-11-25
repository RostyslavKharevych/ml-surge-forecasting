from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import src.ingestion.citi_field_ingest as ingest


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class CitiFieldEndTimeTests(unittest.TestCase):
    def test_calculates_end_time_and_preserves_concerts(self):
        events = pd.DataFrame([
            {
                "event_type": "game",
                "start_time_local": "2024-07-01 19:10:00-0400",
                "duration_minutes": 150,
            },
            {
                "event_type": "concert",
                "start_time_local": "2024-07-19 20:00:00-0400",
                "duration_minutes": None,
            },
        ])
        out = ingest.add_citi_field_end_times(events)
        self.assertEqual(out.loc[0, "end_time_local"], "2024-07-01 21:40:00-0400")
        self.assertTrue(pd.isna(out.loc[1, "end_time_local"]))

    def test_rejects_invalid_or_negative_durations(self):
        for bad in [
            pd.DataFrame({"event_type": ["game"], "start_time_local": ["bad"], "duration_minutes": [120]}),
            pd.DataFrame({"event_type": ["game"], "start_time_local": ["2024-07-01 19:00:00-0400"], "duration_minutes": [None]}),
            pd.DataFrame({"event_type": ["game"], "start_time_local": ["2024-07-01 19:00:00-0400"], "duration_minutes": [-5]}),
        ]:
            with self.assertRaises(ValueError):
                ingest.add_citi_field_end_times(bad)


class CitiFieldIngestTests(unittest.TestCase):
    def test_fetch_games_deduplicates_filters_and_parses_feed(self):
        games = [
            {
                "gamePk": 1001,
                "officialDate": "2024-04-01",
                "status": {"abstractGameState": "Final"},
                "venue": {"name": "Citi Field"},
                "teams": {
                    "home": {"team": {"id": 121}},
                    "away": {"team": {"name": "Detroit Tigers"}},
                },
            },
            {
                "gamePk": 1001,  # duplicate suspended/resumed listing
                "officialDate": "2024-04-01",
                "status": {"abstractGameState": "Final"},
                "venue": {"name": "Citi Field"},
                "teams": {
                    "home": {"team": {"id": 121}},
                    "away": {"team": {"name": "Detroit Tigers"}},
                },
            },
            {
                "gamePk": 1002,  # postponed, should be excluded
                "officialDate": "2024-04-02",
                "status": {"abstractGameState": "Preview"},
                "venue": {"name": "Citi Field"},
                "teams": {
                    "home": {"team": {"id": 121}},
                    "away": {"team": {"name": "Detroit Tigers"}},
                },
            },
            {
                "gamePk": 1003,  # away game, should be excluded
                "officialDate": "2024-04-05",
                "status": {"abstractGameState": "Final"},
                "venue": {"name": "Great American Ball Park"},
                "teams": {
                    "home": {"team": {"id": 113}},
                    "away": {"team": {"name": "New York Mets"}},
                },
            },
        ]

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url, params=None, timeout=None):
                if url.endswith("/schedule"):
                    return _FakeResponse({"dates": [{"games": games}]})
                return _FakeResponse({
                    "gameData": {
                        "datetime": {"dateTime": "2024-04-01T17:10:00Z"},
                        "gameInfo": {"attendance": 33123},
                    },
                    "liveData": {"boxscore": {"info": [{"label": "T", "value": "2:35"}]}},
                })

        with patch.object(ingest.requests, "Session", FakeSession):
            result = ingest.fetch_games(2024)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["event_id"], "METS_20240401_1001")
        self.assertEqual(row["venue_zone_id"], 253)
        self.assertEqual(row["duration_minutes"], 155)
        self.assertEqual(row["attendance"], 33123)
        self.assertEqual(row["start_time_local"], "2024-04-01 13:10:00-0400")
        self.assertEqual(row["end_time_local"], "2024-04-01 15:45:00-0400")

    def test_fetch_events_and_write_2024_and_2025(self):
        def fake_games(year):
            return pd.DataFrame([{
                "event_id": f"METS_{year}0701_1",
                "event_type": "game",
                "event_name": "Mets vs. Braves",
                "event_date": f"{year}-07-01",
                "venue": "Citi Field",
                "venue_zone_id": 253,
                "opponent": "Braves",
                "start_time_local": f"{year}-07-01 19:10:00-0400",
                "attendance": 35000,
                "duration_minutes": 160,
                "mlb_game_id": 1,
                "end_time_local": f"{year}-07-01 21:50:00-0400",
            }])

        with patch.object(ingest, "fetch_games", side_effect=fake_games):
            events_2024 = ingest.fetch_events(2024)
            events_2025 = ingest.fetch_events(2025)
            self.assertEqual(events_2024["event_type"].value_counts().to_dict(), {"game": 1, "concert": 4})
            self.assertTrue((events_2024["venue_zone_id"] == 253).all())
            self.assertEqual(events_2025["event_type"].value_counts().to_dict(), {"game": 1})

            with tempfile.TemporaryDirectory() as tmpdir:
                paths = ingest.run_citi_field_ingestion(years=[2024, 2025], output_dir=Path(tmpdir))
                self.assertEqual([p.name for p in paths], ["citi_field_events_2024.csv", "citi_field_events_2025.csv"])
                self.assertTrue(all(p.exists() for p in paths))
