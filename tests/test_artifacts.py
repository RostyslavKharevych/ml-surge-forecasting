import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.features.modeling import BUCKETS
from src.models.export import METRIC_KEYS, PREDICTION_COLUMNS, export_artifacts
from src.models.inference import load_bundle, predict_event_curve, predict_rows
from tests.test_modeling_features import curve


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.data = pd.concat([curve(day, venue=189 if i % 2 == 0 else 253,
                                    attendance=100 + i).assign(surge_delta=float(i - 10))
                              for i, day in enumerate(pd.date_range("2025-01-01", periods=20))], ignore_index=True)
        cls.dataset = cls.folder / "dataset.parquet"
        cls.data.to_parquet(cls.dataset, index=False)
        cls.output = cls.folder / "artifacts"
        export_artifacts(cls.dataset, cls.output)
        cls.bundle = load_bundle(cls.output / "model.joblib")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_schemas_and_holdout_coverage(self):
        metrics = json.loads((self.output / "metrics.json").read_text())
        self.assertEqual(list(metrics), METRIC_KEYS)
        self.assertTrue(all(isinstance(name, str) for name in metrics["top_features"]))
        self.assertEqual(set(self.bundle), {"model", "feature_names", "preprocessor"})
        predictions = pd.read_parquet(self.output / "predictions_2025.parquet")
        self.assertEqual(predictions.columns.tolist(), PREDICTION_COLUMNS)
        self.assertEqual(predictions.event_id.nunique(), 4)
        self.assertEqual(set(predictions.event_id), set(self.data.iloc[16 * 8:].event_id))
        self.assertTrue(predictions.time_bin.dt.year.eq(2025).all())
        np.testing.assert_array_equal(predictions.y_total_pred,
                                      predictions.y_baseline + predictions.surge_delta_pred)
        source = self.data.set_index(["event_id", "time_bin"]).loc[
            pd.MultiIndex.from_frame(predictions[["event_id", "time_bin"]])].reset_index()
        np.testing.assert_array_equal(predictions.surge_delta_pred, predict_rows(self.bundle, source))
        self.assertTrue(np.isfinite(list(metrics.values())[:4]).all())

    def test_serialization_ordering_and_complete_inference(self):
        event = self.data.iloc[-8:].copy()
        end = event.time_bin.iloc[0] + pd.Timedelta(minutes=30)
        kwargs = dict(venue_zone_id=253, event_type="concert", estimated_event_end=end,
                      attendance_est=119, weather=event[["time_bin", "temp_f", "precipitation_mm"]])
        inferred = predict_event_curve(self.bundle, **kwargs)
        self.assertEqual(inferred.mins_since_end.tolist(), list(BUCKETS))
        self.assertNotIn("y_total_pred", inferred)
        np.testing.assert_array_equal(inferred.surge_delta_pred, predict_rows(self.bundle, event))
        restored = load_bundle(self.output / "model.joblib")
        np.testing.assert_array_equal(predict_rows(restored, event[event.columns[::-1]]), inferred.surge_delta_pred)
        np.testing.assert_array_equal(predict_rows(restored, event.assign(venue_zone_id=253.0)),
                                      inferred.surge_delta_pred)
        broken = dict(self.bundle, feature_names=self.bundle["feature_names"][::-1])
        with self.assertRaises(ValueError):
            predict_rows(broken, event)
        # Explicit missing attendance is supported; no weather defaults or +90 bucket.
        self.assertTrue(np.isfinite(predict_event_curve(self.bundle, **dict(kwargs, attendance_est=None)).surge_delta_pred).all())
        with self.assertRaises(ValueError):
            predict_event_curve(self.bundle, **dict(kwargs, weather=kwargs["weather"].iloc[:-1]))
        with self.assertRaises(ValueError):
            predict_event_curve(self.bundle, **dict(kwargs, estimated_event_end=end.tz_localize("America/New_York")))
        with self.assertRaises(ValueError):
            predict_event_curve(self.bundle, **dict(kwargs, venue_zone_id=None))


if __name__ == "__main__":
    unittest.main()
