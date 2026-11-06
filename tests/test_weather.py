import unittest

import pandas as pd

from src.weather import to_buckets


class ToBucketsTests(unittest.TestCase):
    def test_each_15min_bucket_takes_its_hours_reading(self):
        archive = {"hourly": {"time": ["2024-07-01T19:00", "2024-07-01T20:00"],
                              "precipitation": [0.1, 2.5],
                              "temperature_2m": [80.1, 78.4],
                              "wind_speed_10m": [5.0, 12.3]}}
        buckets = to_buckets(archive, "barclays")

        self.assertEqual(len(buckets), 8)
        self.assertEqual(buckets.iloc[4].to_dict(), dict(
            venue="barclays", timestamp_15m=pd.Timestamp("2024-07-01 20:00"),
            precipitation_mm=2.5, temp_f=78.4, wind_mph=12.3, is_raining=True))
        self.assertEqual(list(buckets.timestamp_15m.dt.strftime("%H:%M")),
                         ["19:00", "19:15", "19:30", "19:45", "20:00", "20:15", "20:30", "20:45"])
        # 0.1 mm is a trace, not rain.
        self.assertEqual(list(buckets.is_raining), [False] * 4 + [True] * 4)
