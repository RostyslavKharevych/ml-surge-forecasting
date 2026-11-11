import unittest

import pandas as pd

from src.features.baseline import add_baseline, affected_slots, drop_events_without_baseline


def event(start, end, status="held", venue="barclays"):
    return dict(event_id=f"{venue}_{start}", venue=venue, event_start=pd.Timestamp(start),
                event_end=pd.Timestamp(end), event_status=status)


def demand(start="2024-01-01", end="2024-03-31 23:45", spikes=(), values=None):
    """Weighted Demand 10 everywhere, 100 at the spike timestamps, or the given {ts: value}."""
    grid = pd.date_range(start, end, freq="15min")
    weighted = pd.Series(10.0, index=grid)
    weighted[pd.DatetimeIndex(list(spikes))] = 100.0
    for ts, value in (values or {}).items():
        weighted[pd.Timestamp(ts)] = value
    return pd.DataFrame({"venue": "barclays", "timestamp_15m": grid, "weighted_demand": weighted.values})


def at(frame, ts, column):
    return frame.set_index("timestamp_15m").at[pd.Timestamp(ts), column]


class AffectedSlotTests(unittest.TestCase):
    def test_slots_within_two_hours_of_held_events_only(self):
        grid = demand("2024-03-05", "2024-03-06 23:45")
        events = pd.DataFrame([event("2024-03-05 19:00", "2024-03-05 21:40"),
                               event("2024-03-06 19:00", "2024-03-06 22:00", status="cancelled"),
                               event("2024-03-06 12:00", "2024-03-06 15:00", venue="citi_field")])
        affected = grid.timestamp_15m[affected_slots(grid, events)]
        self.assertEqual(affected.min(), pd.Timestamp("2024-03-05 17:00"))
        self.assertEqual(affected.max(), pd.Timestamp("2024-03-05 23:30"))  # contains 23:40
        self.assertEqual(len(affected), 27)


class BaselineTests(unittest.TestCase):
    # Tuesday 2024-03-26 19:00; previous Tuesdays: 03-19, 03-12, 03-05, 02-27, 02-20, 02-13, 02-06, 01-30, ...
    TARGET = "2024-03-26 19:00"

    def baseline(self, events=(), **grid):
        events = pd.DataFrame(list(events), columns=["event_id", "venue", "event_start", "event_end",
                                                     "event_status"])
        return add_baseline(demand(**grid), events)

    def test_median_of_four_most_recent_clean_weeks(self):
        # 03-19 is an event week; 02-13 is the 5th clean week.
        weeks = {"2024-03-26": 7, "2024-03-19": 0, "2024-03-12": 1, "2024-03-05": 2,
                 "2024-02-27": 3, "2024-02-20": 4, "2024-02-13": 5}
        out = self.baseline(events=[event("2024-03-19 18:00", "2024-03-19 21:00")],
                            values={f"{d} 19:00": v for d, v in weeks.items()})
        self.assertEqual(at(out, self.TARGET, "baseline"), 2.5)
        self.assertEqual(at(out, self.TARGET, "surge_delta"), 4.5)

    def test_back_to_back_event_weeks_reach_further_back(self):
        weeks = ["2024-03-19", "2024-03-12", "2024-03-05", "2024-02-27"]
        out = self.baseline(events=[event(f"{d} 18:00", f"{d} 21:00") for d in weeks],
                            spikes=[f"{d} 19:00" for d in weeks] + ["2024-02-06 19:00", "2024-02-13 19:00"])
        # Clean weeks: 02-20 (10), 02-13 (100), 02-06 (100), 01-30 (10) → median 55
        self.assertEqual(at(out, self.TARGET, "baseline"), 55)

    def test_lookback_reaches_twelve_weeks_and_no_further(self):
        # Every Tuesday 02-13 … 04-16 has an event; 02-06, 01-30, 01-23 … are clean.
        weeks = pd.date_range("2024-02-13", "2024-04-16", freq="7D").strftime("%Y-%m-%d")
        out = self.baseline(events=[event(f"{d} 18:00", f"{d} 21:00") for d in weeks],
                            end="2024-04-30 23:45", values={"2024-01-23 19:00": 40})
        # 04-16: 12 weeks back reach 01-23 → 02-06, 01-30, 01-23 → median of 10, 10, 40
        self.assertEqual(at(out, "2024-04-16 19:00", "baseline"), 10)
        # 04-23: 01-23 is week 13 → only 02-06, 01-30 → no Baseline
        self.assertTrue(pd.isna(at(out, "2024-04-23 19:00", "baseline")))
        self.assertTrue(pd.isna(at(out, "2024-04-23 19:00", "surge_delta")))

    def test_holidays_are_not_clean(self):
        # Mon 2024-02-26: previous Mondays 02-19 (Presidents Day), 02-12, 02-05, 01-29, 01-22,
        # 01-15 (MLK Day), 01-08, 01-01 (New Year)
        out = self.baseline(spikes=["2024-02-19 19:00", "2024-01-15 19:00", "2024-01-01 19:00"])
        self.assertEqual(at(out, "2024-02-26 19:00", "baseline"), 10)

    def test_early_january_shortfall(self):
        out = self.baseline()
        # Wed 01-17: clean 01-10, 01-03 only. Wed 01-24: 01-17, 01-10, 01-03 → baseline.
        self.assertTrue(pd.isna(at(out, "2024-01-17 19:00", "baseline")))
        self.assertEqual(at(out, "2024-01-24 19:00", "baseline"), 10)


class DropEventsTests(unittest.TestCase):
    def test_events_with_any_prediction_bucket_lacking_baseline_are_dropped(self):
        events = pd.DataFrame([event("2024-01-17 19:00", "2024-01-17 21:10"),  # no Baseline yet
                               event("2024-03-26 19:00", "2024-03-26 21:10"),
                               event("2024-03-27 19:00", "2024-03-27 21:10", status="cancelled")])
        out = add_baseline(demand(), events)
        with self.assertLogs("src.features.baseline", "WARNING") as logs:
            kept = drop_events_without_baseline(events, out)
        self.assertEqual(list(kept.event_start.dt.strftime("%m-%d")), ["03-26", "03-27"])
        self.assertIn("barclays_2024-01-17 19:00", logs.output[0])

    def test_prediction_window_reaching_past_baselines_drops_event(self):
        # End 23:50 → window runs to 01:05 next day, beyond the demand grid.
        events = pd.DataFrame([event("2024-03-31 21:00", "2024-03-31 23:50"),
                               event("2024-03-26 19:00", None)])  # held, no Event End
        with self.assertLogs("src.features.baseline", "WARNING"):
            kept = drop_events_without_baseline(events, add_baseline(demand(), events))
        self.assertTrue(kept.empty)


if __name__ == "__main__":
    unittest.main()
