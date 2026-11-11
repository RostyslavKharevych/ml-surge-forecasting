import unittest

import pandas as pd

from src.processing.calibration import apply_calibrated_ends, category_durations


def event(start, event_type="concert", end=None, basis="", status="held"):
    return dict(venue="barclays", event_type=event_type, event_start=pd.Timestamp(start),
                event_end=pd.Timestamp(end) if end else pd.NaT, end_time_basis=basis,
                event_status=status)


def demand(peaks):
    """Flat core-zone demand with one spike per given timestamp."""
    grid = pd.date_range("2024-03-01", "2025-12-31 23:45", freq="15min")
    core = pd.Series(5, index=grid)
    core[pd.DatetimeIndex(peaks)] = 50
    return pd.DataFrame({"venue": "barclays", "timestamp_15m": grid, "core_trips": core.values})


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame([
            event("2024-04-05 20:00"),                       # training, peak +180
            event("2024-05-10 19:30"),                       # training, peak +210
            event("2024-06-01 19:00", "basketball",          # training, real end kept
                  end="2024-06-01 21:00", basis="box_score"),
            event("2024-11-01 20:00"),                       # validation: no calibration weight
            event("2025-03-01 19:00"),                       # to calibrate
            event("2025-03-02 19:00", "gymnastics"),         # no training events: pooled
            event("2025-03-03 19:00", status="cancelled"),
        ])
        self.demand = demand(["2024-04-05 23:00", "2024-05-10 23:00", "2024-06-01 21:15",
                              "2024-11-01 20:15",
                              # a later Fri/Sat-style nightlife spike beyond Start + 5h
                              "2024-04-06 01:30"])

    def test_durations_are_training_period_medians_of_start_to_peak(self):
        durations = category_durations(self.events, self.demand)
        self.assertEqual(durations["concert"], 195)
        self.assertEqual(durations["basketball"], 135)
        self.assertEqual(durations["all"], 180)

    def test_blank_held_ends_get_start_plus_category_duration(self):
        events = apply_calibrated_ends(self.events, category_durations(self.events, self.demand))
        self.assertEqual(list(events.event_end.dt.strftime("%Y-%m-%d %H:%M").fillna("")), [
            "2024-04-05 23:15", "2024-05-10 22:45", "2024-06-01 21:00", "2024-11-01 23:15",
            "2025-03-01 22:15", "2025-03-02 22:00", ""])
        self.assertEqual(list(events.end_time_basis), ["calibrated_category"] * 2 + ["box_score"]
                         + ["calibrated_category"] * 3 + [""])


if __name__ == "__main__":
    unittest.main()
