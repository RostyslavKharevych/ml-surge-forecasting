import io
import unittest

import joblib
import numpy as np
import pandas as pd

from src.features.modeling import (
    BUCKETS, FEATURES, TARGET, CurveFeatures, chronological_split,
    make_preprocessor, predictor_frame,
)


def curve(day, venue=189, event_type="concert", attendance=100):
    start = pd.Timestamp(day) + pd.Timedelta(hours=19)
    prefix = "BARCLAYS" if venue == 189 else "CITI_FIELD"
    return pd.DataFrame({
        "event_id": f"{prefix}_{start:%Y%m%d_%H%M}",
        "venue_zone_id": venue, "event_type": event_type,
        "time_bin": [start + pd.Timedelta(hours=3, minutes=m) for m in BUCKETS],
        "mins_since_end": BUCKETS, "attendance_est": attendance,
        "precipitation_mm": 0.0, "temp_f": 60.0,
        "y_actual": 1.0, "y_baseline": 10.0, "surge_delta": -9.0,
    })


class ModelingFeaturesTests(unittest.TestCase):
    def test_split_is_grouped_chronological_and_order_independent(self):
        data = pd.concat([curve(day) for day in pd.date_range("2024-01-01", periods=492)], ignore_index=True)
        result = chronological_split(data.sample(frac=1, random_state=4))
        events = result.drop_duplicates("event_id").sort_values("event_start")
        self.assertEqual(events.split.tolist(), ["train"] * 320 + ["val"] * 74 + ["test"] * 98)
        self.assertTrue(result.groupby("event_id").split.nunique().eq(1).all())
        pd.testing.assert_series_equal(result[TARGET], data.loc[result.index, TARGET])

    def test_bad_curves_and_overlapping_boundaries_rejected(self):
        data = pd.concat([curve(day) for day in pd.date_range("2024-01-01", periods=3)], ignore_index=True)
        with self.assertRaises(ValueError):
            chronological_split(data.iloc[1:])
        with self.assertRaises(ValueError):
            chronological_split(pd.concat([data, data.iloc[:1]]))
        bad = data.copy()
        bad.loc[:7, "event_id"] = "unknown"
        with self.assertRaises(ValueError):
            chronological_split(bad)
        overlap = data.copy()
        overlap.loc[:7, "time_bin"] += pd.Timedelta(days=2)
        with self.assertRaises(ValueError):
            chronological_split(overlap)

    def test_training_only_hierarchical_imputation(self):
        train = pd.concat([
            curve("2024-01-01", attendance=100),
            curve("2024-01-02", attendance=300),
            curve("2024-01-03", event_type="basketball", attendance=900),
            curve("2024-01-04", venue=253, attendance=2000),
        ], ignore_index=True)
        transform = CurveFeatures().fit(train)
        future = pd.concat([
            curve("2025-01-01", attendance=pd.NA),
            curve("2025-01-02", event_type="new", attendance=pd.NA),
            curve("2025-01-03", venue=999, attendance=pd.NA),
            curve("2025-01-04", attendance=99999),
        ], ignore_index=True)
        result = transform.transform(future)
        self.assertEqual(result.attendance_est.iloc[::8].tolist(), [200, 300, 600, 99999])
        self.assertEqual(result.attendance_missing.iloc[::8].tolist(), [1, 1, 1, 0])
        self.assertEqual(transform.global_median_, 600)
        with self.assertRaises(ValueError):
            CurveFeatures().fit(curve("2024-01-01", attendance=pd.NA))

    def test_no_outcome_leakage_and_inference_round_trip(self):
        train = curve("2024-01-05")  # Friday follows the repository weekend convention.
        prep = make_preprocessor().fit(predictor_frame(train))
        expected = prep.transform(train)
        changed = train.assign(event_id="ignored", y_actual=999999, y_baseline=-9999,
                               surge_delta=123456, demand_lag_1=99999)
        np.testing.assert_array_equal(expected, prep.transform(changed))
        self.assertEqual(CurveFeatures().fit(train).transform(train).is_weekend.tolist(), [1] * 8)
        self.assertEqual(set(CurveFeatures().fit(train).transform(train).columns), set(FEATURES))
        names = prep.get_feature_names_out().tolist()
        self.assertFalse(any(name in names for name in ["event_id", "y_actual", "y_baseline", TARGET, "demand_lag_1"]))
        buffer = io.BytesIO()
        joblib.dump(prep, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        inference = predictor_frame(curve("2025-01-01", venue=253, event_type="unseen", attendance=pd.NA))
        self.assertTrue(np.isfinite(restored.transform(inference)).all())
        self.assertEqual(restored.transform(inference).shape[1], expected.shape[1])
        np.testing.assert_array_equal(expected, restored.transform(predictor_frame(train)))

    def test_unavailable_inference_inputs_are_not_assumed(self):
        data = curve("2024-01-01")
        prep = make_preprocessor().fit(data)
        for name in ["venue_zone_id", "event_type", "time_bin", "temp_f", "precipitation_mm"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                prep.transform(data.assign(**{name: None}))
        with self.assertRaises(ValueError):
            prep.transform(data.assign(mins_since_end=90))
        with self.assertRaises(ValueError):
            prep.transform(data.assign(venue_zone_id=189.5))


if __name__ == "__main__":
    unittest.main()
