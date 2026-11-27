import unittest

import numpy as np
import pandas as pd

from src.models.evaluate import peak_errors, summarize_peaks
from src.models.holdout import LGBM_PARAMS, RIDGE_PARAMS, evaluate_holdout
from tests.test_modeling_features import curve


class HoldoutTests(unittest.TestCase):
    def test_peak_magnitude_timing_ties_and_flat_baseline(self):
        rows = curve("2024-01-01")
        rows["surge_delta"] = [-5, -4, -1, -1, -2, -3, -4, -5]
        predictions = {"model": [-5, -4, -3, -2, 1, 0, -1, -2], "zero": [0] * 8}
        errors = peak_errors(rows, predictions).set_index("model")
        self.assertEqual(errors.loc["model", "magnitude_error"], 2)
        self.assertEqual(errors.loc["model", "timing_error_minutes"], 30)
        self.assertTrue(errors.loc["model", "actual_peak_tied"])
        self.assertTrue(np.isnan(errors.loc["zero", "timing_error_minutes"]))
        summary = summarize_peaks(errors.reset_index()).set_index("model")
        self.assertEqual(summary.loc["model", "magnitude_mae"], 2)
        self.assertEqual(summary.loc["zero", "timing_events"], 0)

    def test_frozen_parameters_and_test_inputs_cannot_change_fitted_models(self):
        self.assertEqual(RIDGE_PARAMS["alpha"], 100)
        self.assertEqual({k: LGBM_PARAMS[k] for k in ["num_leaves", "min_child_samples", "n_estimators"]},
                         dict(num_leaves=7, min_child_samples=64, n_estimators=250))
        data = pd.concat([curve(day, attendance=100 + i).assign(surge_delta=float(i - 10))
                          for i, day in enumerate(pd.date_range("2024-01-01", periods=20))], ignore_index=True)
        before = evaluate_holdout(data)
        changed = data.copy()
        # Last four events are test; altering their targets must not alter any fitted predictions.
        changed.loc[changed.index >= 16 * 8, "surge_delta"] = 100000
        after = evaluate_holdout(changed)
        np.testing.assert_array_equal(before["predictions"]["lightgbm"], after["predictions"]["lightgbm"])
        np.testing.assert_array_equal(before["predictions"]["ridge"], after["predictions"]["ridge"])
        self.assertEqual(set(before["summary"].split), {"train", "val", "test"})
        self.assertEqual(before["test"].event_id.nunique(), 4)
        self.assertTrue(np.isfinite(before["feature_importance"].mean_abs_shap).all())


if __name__ == "__main__":
    unittest.main()
