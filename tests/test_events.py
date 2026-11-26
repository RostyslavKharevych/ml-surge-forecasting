from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.ingestion.box_scores import GAMES_PATH
from src.config import (
    ALL_INGESTION_ZONES,
    FALLBACK_TIMESTAMP_COL,
    INTERIM_DATA_DIR,
    PRIMARY_TIMESTAMP_COL,
    VENUE_CONFIG,
)
from src.features.baseline import is_holiday as baseline_is_holiday
from src.features.features import aggregate_venue_demand
from src.processing.clean_tlc import clean_venue_dataframe
from src.processing.events import (
    EVENT_COLUMNS,
    apply_overrides,
    apply_runtimes,
    citi_field_events,
    from_barclays_scrape,
    from_citi_field_events,
    is_federal_holiday,
    is_holiday,
    is_nyc_school_holiday,
    load_events,
)
from src.ingestion.tour_boxscores import ATTENDANCE_PATH


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


def runtime(**fields):
    row = dict(venue="barclays", event_name="Disney On Ice", year="2024", runtime_minutes="105",
               source="test")
    return pd.DataFrame([{**row, **fields}])


class RuntimeTests(unittest.TestCase):
    def test_fills_end_from_start_plus_runtime(self):
        events = apply_runtimes(from_barclays_scrape(scraped(
            event_name="Disney On Ice", event_date="2024-01-18", scheduled_start_time="19:00:00",
            category_type="family_show")), runtime())
        self.assertEqual(events.iloc[0].event_end, pd.Timestamp("2024-01-18 20:45"))
        self.assertEqual(events.iloc[0].end_time_basis, "published_runtime")

    def test_existing_end_and_other_years_untouched(self):
        events = apply_overrides(from_barclays_scrape(pd.concat([
            scraped(event_name="Disney On Ice", event_date="2024-01-18", scheduled_start_time="19:00:00"),
            scraped(event_name="Disney On Ice", event_date="2025-01-16", scheduled_start_time="19:00:00")])),
            pd.DataFrame([override(event_name="Disney On Ice", event_date="2024-01-18",
                                   event_end="2024-01-18 21:00", end_time_basis="broadcast_window")]))
        events = apply_runtimes(events, runtime())
        self.assertEqual(list(events.end_time_basis), ["broadcast_window", ""])

    def test_unmatched_runtime_fails(self):
        with self.assertRaisesRegex(ValueError, "matches no"):
            apply_runtimes(from_barclays_scrape(scraped()), runtime())


@unittest.skipUnless((INTERIM_DATA_DIR / "events/barclays_2025/events_all.csv").exists()
                     and GAMES_PATH.exists() and ATTENDANCE_PATH.exists(),
                     "scraped events, box scores or tour attendance not present")
class RealEventTableTests(unittest.TestCase):
    def test_reviewed_ncaa_sessions_and_full_categories(self):
        events = load_events()
        ncaa = events[events.event_name == "NCAA March Madness"]
        self.assertEqual(list(ncaa.event_start.dt.strftime("%Y-%m-%d %H:%M")),
                         ["2024-03-22 12:15", "2024-03-22 19:10", "2024-03-24 17:15"])
        self.assertEqual(set(ncaa.end_time_basis), {"box_score"})
        self.assertFalse((events.event_type == "").any())
        self.assertEqual(set(events.event_start.dt.year.dropna()), {2024, 2025})

    def test_non_basketball_ends_from_specific_sources(self):
        events = load_events()
        held = events[events.event_status == "held"]
        for types, basis in [({"wrestling", "boxing"}, "broadcast_window"),
                             ({"family_show", "circus"}, "published_runtime")]:
            with self.subTest(basis=basis):
                self.assertEqual(set(held[held.event_type.isin(types)].end_time_basis), {basis})
        self.assertTrue(set(events.attendance_basis) <= {"", "box_score", "reported", "tour_boxscore"})


class CitiFieldAdapterTests(unittest.TestCase):
    def test_maps_games_and_concerts_into_event_table(self):
        raw = pd.DataFrame([
            {
                "event_id": "METS_20240712_745800",
                "event_type": "game",
                "event_name": "Mets vs. Rockies",
                "event_date": "2024-07-12",
                "venue": "Citi Field",
                "start_time_local": "2024-07-12 19:10:00-0400",
                "attendance": "",
                "duration_minutes": "165",
                "end_time_local": "",
            },
            {
                "event_id": "CONCERT_FOO_FIGHTERS_20240719",
                "event_type": "concert",
                "event_name": "Foo Fighters",
                "event_date": "2024-07-19",
                "venue": "Citi Field",
                "start_time_local": "2024-07-19 20:00:00-0400",
                "attendance": "40000",
                "duration_minutes": "",
                "end_time_local": "",
            },
        ])

        events = from_citi_field_events(raw)

        self.assertEqual(list(events.columns), EVENT_COLUMNS)
        self.assertEqual(list(events.event_id), ["CITI_FIELD_20240712_1910", "CITI_FIELD_20240719_2000"])
        self.assertEqual(list(events.venue), ["citi_field", "citi_field"])
        self.assertEqual(list(events.event_type), ["baseball", "concert"])
        self.assertEqual(events.loc[0, "event_start"], pd.Timestamp("2024-07-12 19:10"))
        self.assertIsNone(events.loc[0, "event_start"].tz)
        self.assertEqual(events.loc[0, "event_end"], pd.Timestamp("2024-07-12 21:55"))
        self.assertEqual(events.loc[0, "end_time_basis"], "box_score")
        self.assertTrue(pd.isna(events.loc[0, "attendance"]))
        self.assertEqual(events.loc[0, "attendance_basis"], "")
        self.assertTrue(pd.isna(events.loc[1, "event_end"]))
        self.assertEqual(events.loc[1, "end_time_basis"], "")
        self.assertEqual(events.loc[1, "attendance"], 40000)
        self.assertEqual(events.loc[1, "attendance_basis"], "reported")
        self.assertEqual(list(events.event_status), ["held", "held"])

    def test_reads_csv_and_handles_empty_input(self):
        empty = from_citi_field_events(pd.DataFrame())
        self.assertEqual(list(empty.columns), EVENT_COLUMNS)
        self.assertTrue(empty.empty)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "citi.csv"
            pd.DataFrame([{
                "event_type": "game",
                "event_name": "Mets vs. Braves",
                "event_date": "2024-05-10",
                "start_time_local": "2024-05-10 19:10:00-0400",
                "end_time_local": "2024-05-10 21:45:00-0400",
                "attendance": "32500",
            }]).to_csv(csv_path, index=False)
            loaded = citi_field_events(path=csv_path)
            self.assertEqual(loaded.loc[0, "event_id"], "CITI_FIELD_20240510_1910")
            self.assertEqual(loaded.loc[0, "event_end"], pd.Timestamp("2024-05-10 21:45"))
            self.assertEqual(loaded.loc[0, "attendance"], 32500)

    def test_load_events_combines_barclays_and_citi_field_with_tz_conversion(self):
        barclays = from_barclays_scrape(scraped(scheduled_start_time="19:30:00"))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            overrides = tmp / "overrides.csv"
            runtimes = tmp / "runtimes.csv"
            citi_csv = tmp / "citi_field_events_2024.csv"
            pd.DataFrame(columns=[
                "venue", "event_date", "event_name", "event_start", "event_end",
                "end_time_basis", "attendance", "attendance_basis", "event_status", "source",
            ]).to_csv(overrides, index=False)
            pd.DataFrame(columns=["venue", "event_name", "year", "runtime_minutes", "source"]).to_csv(
                runtimes, index=False
            )
            pd.DataFrame([
                {
                    "event_type": "game",
                    "event_name": "Mets vs. Phillies",
                    "event_date": "2024-03-29",
                    "start_time_local": "2024-03-29 13:40:00-0400",
                    "duration_minutes": "155",
                    "end_time_local": "",
                    "attendance": "41000",
                },
                {
                    "event_type": "concert",
                    "event_name": "Winter Special",
                    "event_date": "2024-11-15",
                    "start_time_local": "2024-11-15 20:00:00-0500",
                    "duration_minutes": "",
                    "end_time_local": "",
                    "attendance": "30000",
                },
            ]).to_csv(citi_csv, index=False)

            with patch("src.processing.events.barclays_events", return_value=barclays):
                combined = load_events(
                    years=[2024],
                    overrides_path=overrides,
                    box_scores_path=None,
                    runtimes_path=runtimes,
                    tour_attendance_path=None,
                    citi_field_path=citi_csv,
                    calibrate=False,
                )

        self.assertEqual(set(combined.venue), {"barclays", "citi_field"})
        self.assertEqual(len(combined), 3)
        citi_game = combined[combined.event_name == "Mets vs. Phillies"].iloc[0]
        self.assertEqual(citi_game.event_start, pd.Timestamp("2024-03-29 13:40"))
        self.assertIsNone(citi_game.event_start.tz)
        self.assertEqual(citi_game.event_end, pd.Timestamp("2024-03-29 16:15"))
        self.assertEqual(citi_game.end_time_basis, "box_score")
        citi_concert = combined[combined.event_name == "Winter Special"].iloc[0]
        self.assertEqual(citi_concert.event_start, pd.Timestamp("2024-11-15 20:00"))
        self.assertTrue(pd.isna(citi_concert.event_end))

    def test_load_events_falls_back_to_citi_field_when_barclays_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            citi_csv = tmp / "citi_field_events_2024.csv"
            pd.DataFrame([{
                "event_type": "game",
                "event_name": "Mets vs. Braves",
                "event_date": "2024-05-10",
                "start_time_local": "2024-05-10 19:10:00-0400",
                "duration_minutes": "150",
                "end_time_local": "",
                "attendance": "32500",
            }]).to_csv(citi_csv, index=False)

            with patch("src.processing.events.output_dir_for_year", return_value=tmp / "missing_barclays"):
                events = load_events(
                    years=[2024],
                    citi_field_path=citi_csv,
                    box_scores_path=tmp / "missing_box.csv",
                    tour_attendance_path=tmp / "missing_tour.csv",
                    calibrate=False,
                )

        self.assertEqual(list(events.venue), ["citi_field"])
        self.assertEqual(list(events.event_id), ["CITI_FIELD_20240510_1910"])


class HolidayHelperTests(unittest.TestCase):
    def test_federal_and_nyc_school_holiday_flags(self):
        ts = pd.Series(pd.to_datetime([
            "2024-11-11 19:00",  # Veterans Day (federal + NYC school)
            "2024-02-20 19:00",  # Midwinter Recess (NYC school only)
            "2024-11-01 19:00",  # Diwali (NYC school only)
            "2025-01-29 19:00",  # Lunar New Year (NYC school only)
            "2025-04-16 19:00",  # Spring Recess 2025 (NYC school only)
            "2025-10-20 19:00",  # Diwali 2025 (NYC school only)
            "2024-03-26 19:00",  # Regular Tuesday
            "2024-11-05 19:00",  # Non-holiday Tuesday
        ]))

        self.assertEqual(list(is_federal_holiday(ts)), [True, False, False, False, False, False, False, False])
        self.assertEqual(list(is_nyc_school_holiday(ts)), [True, True, True, True, True, True, False, False])
        self.assertEqual(list(is_holiday(ts)), [True, True, True, True, True, True, False, False])
        self.assertEqual(list(baseline_is_holiday(ts)), list(is_holiday(ts)))
        empty = pd.Series([], dtype="datetime64[ns]")
        self.assertTrue(is_holiday(empty).empty)


class ConfigAndTimestampFallbackTests(unittest.TestCase):
    def test_canonical_prd_zones_and_timestamp_fallback(self):
        self.assertEqual(VENUE_CONFIG["citi_field"]["core_zone"], 253)
        self.assertEqual(VENUE_CONFIG["citi_field"]["spillover_zones"], {93: 0.75})
        self.assertEqual(VENUE_CONFIG["barclays"]["core_zone"], 189)
        self.assertEqual(ALL_INGESTION_ZONES, [49, 61, 62, 93, 97, 181, 189, 190, 253])
        self.assertEqual(PRIMARY_TIMESTAMP_COL, "request_datetime")
        self.assertEqual(FALLBACK_TIMESTAMP_COL, "pickup_datetime")

        raw = pd.DataFrame({
            "request_datetime": [pd.NaT, "2024-07-01 19:00:00"],
            "pickup_datetime": ["2024-07-01 19:05:00", "2024-07-01 19:05:00"],
            "dropoff_datetime": ["2024-07-01 19:20:00", "2024-07-01 19:20:00"],
            "PULocationID": [253, 93],
            "trip_miles": [3.0, 4.0],
            "trip_time": [900, 900],
            "base_passenger_fare": [15.0, 20.0],
            "driver_pay": [10.0, 12.0],
        })
        cleaned, report = clean_venue_dataframe(raw)
        self.assertEqual(report["final_rows"], 2)
        agg = aggregate_venue_demand(
            cleaned.drop(columns=["request_datetime"]),
            "citi_field",
            VENUE_CONFIG["citi_field"],
        )
        self.assertAlmostEqual(agg["weighted_demand"].sum(), 1.75)
