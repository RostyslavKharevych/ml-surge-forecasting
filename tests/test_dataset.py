import unittest

import pandas as pd

from src.dataset import build_dataset


def event(start, end, status="held", event_type="concert"):
    return dict(event_id=f"BARCLAYS_{start}", venue="barclays", event_name="x", event_type=event_type,
                event_start=pd.Timestamp(start), event_end=pd.Timestamp(end) if end else pd.NaT,
                end_time_basis="box_score" if end else "", attendance=pd.NA, attendance_basis="",
                event_status=status)


GRID = pd.date_range("2024-01-01", "2025-01-31 23:45", freq="15min")


def demand():
    return pd.DataFrame({"venue": "barclays", "timestamp_15m": GRID, "core_trips": 8.0,
                         "spillover_trips": 8.0, "weighted_demand": 10.0})


def weather():
    return pd.DataFrame({"venue": "barclays", "timestamp_15m": GRID, "precipitation_mm": 0.0,
                         "temp_f": 50.0, "wind_mph": 5.0, "is_raining": False})


class BuildDatasetTests(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame([
            event("2024-01-17 19:00", "2024-01-17 21:10"),  # no Baseline yet → dropped
            event("2024-02-28 19:00", "2024-02-28 21:10"),  # before the training period
            event("2024-03-26 14:00", "2024-03-26 18:40", event_type="family"),  # long matinee
            event("2024-03-26 19:00", "2024-03-26 21:40"),
            event("2024-03-27 19:00", "2024-03-27 21:40", status="cancelled"),
            event("2024-11-05 19:00", "2024-11-05 21:40"),
            event("2024-11-11 19:00", "2024-11-11 21:40"),  # Veterans Day
            event("2025-01-14 19:00", "2025-01-14 21:40"),
        ])
        with self.assertLogs("src", "WARNING"):
            self.rows = build_dataset(self.events, demand(), weather())

    def test_one_row_per_event_and_prediction_bucket(self):
        self.assertEqual(self.rows.groupby("event_id").size().to_dict(),
                         {f"BARCLAYS_{s}": 8 for s in ["2024-03-26 14:00", "2024-03-26 19:00",
                                                       "2024-11-05 19:00", "2024-11-11 19:00",
                                                       "2025-01-14 19:00"]})
        evening = self.rows[self.rows.event_id == "BARCLAYS_2024-03-26 19:00"]
        self.assertEqual(list(evening.minutes_since_end), list(range(-30, 90, 15)))
        self.assertEqual(evening.time_bin.min(), pd.Timestamp("2024-03-26 21:00"))

    def test_split_by_event_start(self):
        splits = self.rows.groupby("event_id").split.first()
        self.assertEqual(list(splits), ["train", "train", "val", "val", "test"])

    def test_next_showing_gap(self):
        matinee = self.rows[self.rows.event_id == "BARCLAYS_2024-03-26 14:00"]
        # Window 18:00 … 19:45; evening show starts 19:00, so the gap clips at 0 from then.
        self.assertEqual(list(matinee.next_showing_gap), [60, 45, 30, 15, 0, 0, 0, 0])
        self.assertTrue(matinee.has_next_showing.all())
        evening = self.rows[self.rows.event_id == "BARCLAYS_2024-03-26 19:00"]
        self.assertTrue(evening.next_showing_gap.isna().all())
        self.assertFalse(evening.has_next_showing.any())

    def test_columns_joined(self):
        row = self.rows.iloc[0]
        self.assertEqual(row.baseline, 10)
        self.assertEqual(row.surge_delta, 0)
        self.assertEqual(row.temp_f, 50)
        self.assertEqual(row.demand_lag_672, 10)
        self.assertFalse(self.rows[["baseline", "surge_delta", "temp_f", "is_holiday",
                                    "day_of_week", "demand_lag_1"]].isna().any().any())
        holiday = self.rows.groupby("event_id").is_holiday.all()
        self.assertEqual(list(holiday[holiday].index), ["BARCLAYS_2024-11-11 19:00"])


if __name__ == "__main__":
    unittest.main()
