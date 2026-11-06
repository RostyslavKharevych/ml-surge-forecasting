import unittest

import pandas as pd

from src.box_scores import GAMES_PATH
from src.config import INTERIM_DATA_DIR
from src.events import EVENT_COLUMNS, apply_overrides, from_barclays_scrape, load_events


def scraped(**overrides):
    row = dict(venue="barclays", event_name="NCAA March Madness", event_date="2024-03-22",
               scheduled_start_time="", category_type="basketball", event_status="unknown")
    return pd.DataFrame([{**row, **overrides}])


def override(**fields):
    row = dict(venue="barclays", event_date="2024-03-22", event_name="NCAA March Madness",
               event_start="", event_end="", end_time_basis="", attendance="",
               attendance_basis="", event_status="", source="test")
    return {**row, **fields}


class BarclaysAdapterTests(unittest.TestCase):
    def test_maps_scrape_into_event_table(self):
        events = from_barclays_scrape(scraped(scheduled_start_time="19:30:00"))
        self.assertEqual(list(events.columns), EVENT_COLUMNS)
        event = events.iloc[0]
        self.assertEqual(event.event_id, "BARCLAYS_20240322_1930")
        self.assertEqual(event.event_start, pd.Timestamp("2024-03-22 19:30"))
        self.assertIsNone(event.event_start.tz)
        self.assertEqual(event.event_type, "basketball")
        self.assertTrue(pd.isna(event.event_end))

    def test_status_mapping(self):
        for raw, expected in [("unknown", "held"), ("rescheduled", "held"),
                              ("cancelled", "cancelled"), ("postponed", "postponed")]:
            with self.subTest(raw=raw):
                events = from_barclays_scrape(scraped(event_status=raw))
                self.assertEqual(events.iloc[0].event_status, expected)

    def test_tba_start_gets_tba_id(self):
        self.assertEqual(from_barclays_scrape(scraped()).iloc[0].event_id, "BARCLAYS_20240322_TBA")


class OverrideTests(unittest.TestCase):
    def test_one_row_overrides_fields(self):
        events = apply_overrides(from_barclays_scrape(scraped()), pd.DataFrame([override(
            event_start="2024-03-22 17:15", event_end="2024-03-22 22:05",
            end_time_basis="box_score", attendance="17732", attendance_basis="box_score")]))
        event = events.iloc[0]
        self.assertEqual(event.event_id, "BARCLAYS_20240322_1715")
        self.assertEqual(event.event_end, pd.Timestamp("2024-03-22 22:05"))
        self.assertEqual(event.attendance, 17732)
        self.assertEqual(event.event_status, "held")

    def test_several_rows_split_one_showing(self):
        events = apply_overrides(from_barclays_scrape(scraped()), pd.DataFrame([
            override(event_start="2024-03-22 12:15"), override(event_start="2024-03-22 19:10")]))
        self.assertEqual(list(events.event_id), ["BARCLAYS_20240322_1215", "BARCLAYS_20240322_1910"])
        self.assertEqual(set(events.event_name), {"NCAA March Madness"})

    def test_unmatched_override_fails(self):
        with self.assertRaisesRegex(ValueError, "matches 0"):
            apply_overrides(from_barclays_scrape(scraped()),
                            pd.DataFrame([override(event_name="Nope")]))

    def test_end_without_basis_fails(self):
        with self.assertRaisesRegex(ValueError, "basis"):
            apply_overrides(from_barclays_scrape(scraped()),
                            pd.DataFrame([override(event_end="2024-03-22 22:05")]))


@unittest.skipUnless((INTERIM_DATA_DIR / "events/barclays_2025/events_all.csv").exists()
                     and GAMES_PATH.exists(), "scraped events or box scores not present")
class RealEventTableTests(unittest.TestCase):
    def test_reviewed_ncaa_sessions_and_full_categories(self):
        events = load_events()
        ncaa = events[events.event_name == "NCAA March Madness"]
        self.assertEqual(list(ncaa.event_start.dt.strftime("%Y-%m-%d %H:%M")),
                         ["2024-03-22 12:15", "2024-03-22 19:10", "2024-03-24 17:15"])
        self.assertEqual(set(ncaa.end_time_basis), {"box_score"})
        self.assertFalse((events.event_type == "").any())
        self.assertEqual(set(events.event_start.dt.year.dropna()), {2024, 2025})
