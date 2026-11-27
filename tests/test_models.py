import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features.modeling import chronological_split
from src.models.evaluate import (
    compute_metrics, compute_r2, grouped_metrics, persistence_predictions,
    rmse_improvement, supplemental_evaluation,
)
from src.models.train import compare_xgboost, select_model, train_validate
from src.models.diagnostics import anomaly_flags, cross_venue_transfer, score_groups
from src.features.modeling import make_preprocessor
from tests.test_modeling_features import curve


class ConstantModel:
    def __init__(self, value):
        self.value = value
        self.fit_targets = None

    def fit(self, X, y):
        self.fit_targets = np.array(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


class EvaluationTests(unittest.TestCase):
    def test_r2_and_undefined_cases(self):
        self.assertEqual(compute_r2([-2, 0, 2], [-2, 0, 2]), 1)
        self.assertEqual(compute_r2([-2, 0, 2], [0, 0, 0]), 0)
        self.assertLess(compute_r2([-2, 0, 2], [10, 10, 10]), 0)
        self.assertTrue(np.isnan(compute_r2([1, 1], [1, 1])))
        self.assertTrue(np.isnan(compute_r2([1], [1])))

    def test_persistence_sorts_and_never_crosses_events(self):
        rows = pd.DataFrame({
            "event_id": ["b", "a", "b", "a"],
            "time_bin": pd.to_datetime(["2025-01-02 12:15", "2025-01-01 12:00",
                                        "2025-01-02 12:00", "2025-01-01 12:15"]),
            "y_actual": [400, 10, 300, 20], "y_baseline": [50, 1, 40, 2],
        })
        original = rows.copy(deep=True)
        result = persistence_predictions(rows)
        self.assertTrue(result.groupby("event_id").head(1).persistence_surge_delta_pred.isna().all())
        np.testing.assert_array_equal(result.groupby("event_id").tail(1).persistence_surge_delta_pred, [8, 250])
        pd.testing.assert_frame_equal(rows, original)
        changed = rows.copy()
        changed.loc[changed.event_id == "a", "y_actual"] = 99999
        pd.testing.assert_series_equal(
            result[result.event_id == "b"].persistence_surge_delta_pred,
            persistence_predictions(changed).query("event_id == 'b'").persistence_surge_delta_pred)

    def test_rolling_comparison_uses_identical_eligible_rows(self):
        rows = pd.concat([curve("2025-01-01"), curve("2025-01-02")], ignore_index=True)
        rows["split"] = ["val"] * 8 + ["test"] * 8
        rows.loc[[0, 8], "surge_delta"] = 1000  # Must not enter any rolling score.
        rows = rows.sample(frac=1, random_state=5)
        predictions = {"control_median": np.zeros(len(rows)), "ridge": np.full(len(rows), -9),
                       "lightgbm": rows.surge_delta.to_numpy()}
        full, rolling = supplemental_evaluation(rows, predictions)
        self.assertEqual(len(full), 6)
        self.assertTrue(rolling.rows.eq(7).all())
        self.assertTrue(rolling.events.eq(1).all())
        self.assertTrue(rolling[rolling.model == "control_median"].rmse.eq(9).all())
        self.assertTrue(rolling[rolling.model == "ridge"].rmse.eq(0).all())
        self.assertTrue(rolling[rolling.model == "lightgbm"].rmse.eq(0).all())
        self.assertTrue(rolling[rolling.model == "persistence"].rmse.eq(0).all())
        with self.assertRaises(ValueError):
            supplemental_evaluation(rows.assign(split="train"), predictions)

    def test_metrics_are_positional_and_keep_negatives(self):
        self.assertEqual(compute_metrics(pd.Series([-4, 0], index=[8, 9]),
                                         pd.Series([-4, 2], index=[0, 1])),
                         {"rmse": np.sqrt(2), "mae": 1.0})
        self.assertEqual(rmse_improvement(10, 8), 20)
        self.assertEqual(rmse_improvement(10, 12), -20)
        self.assertIsNone(rmse_improvement(0, 0))
        for actual, predicted in [([], []), ([1], [1, 2]), ([np.nan], [0])]:
            with self.assertRaises(ValueError):
                compute_metrics(actual, predicted)

    def test_grouped_metrics_use_group_baselines(self):
        rows = pd.DataFrame({"event_id": ["a", "b"], "venue_zone_id": [189, 253],
                             "surge_delta": [-4, 10]})
        report = grouped_metrics(rows, {"zero": [0, 0], "model": [-4, 5]}, "venue_zone_id")
        model = report[report.model == "model"].set_index("venue_zone_id")
        self.assertEqual(model.loc[189, "rmse"], 0)
        self.assertEqual(model.loc[253, "rmse_improvement_pct"], 50)


class TrainingTests(unittest.TestCase):
    def test_transfer_fits_source_only_and_uses_later_target_events(self):
        data = pd.concat([curve(day, venue=189 if i % 2 == 0 else 253,
                                attendance=(100 + i if i % 2 == 0 else 10000 + i),
                                event_type="concert" if i % 2 == 0 else "baseball")
                          for i, day in enumerate(pd.date_range("2025-01-01", periods=20))], ignore_index=True)
        fitted = []

        def preprocessor():
            prep = make_preprocessor()
            fitted.append(prep)
            return prep

        with patch("src.models.diagnostics.make_preprocessor", side_effect=preprocessor), \
             patch("src.models.diagnostics.fixed_models", side_effect=lambda: {
                 name: ConstantModel(0) for name in ["ridge", "lightgbm", "xgboost"]}):
            result = cross_venue_transfer(data)
        samples = result["transfer_samples"]
        self.assertEqual(samples.train_events.tolist(), [7, 7, 6, 6])
        self.assertEqual(samples.eval_events.tolist(), [2, 2, 1, 2])
        self.assertTrue((samples.train_end < samples.evaluation_start).all())
        self.assertTrue(samples.unseen_type_events.eq(samples.eval_events).all())
        self.assertEqual(fitted[0].named_steps["features"].global_median_, 106)
        self.assertEqual(fitted[1].named_steps["features"].global_median_, 10006)
        for prep, venue, median in zip(fitted, [189, 253], [106, 10006]):
            categories = prep.named_steps["encode"].named_transformers_["categorical"].categories_
            self.assertEqual(categories[0].tolist(), [str(venue)])
            target = data[data.venue_zone_id != venue]
            self.assertTrue(np.isfinite(prep.transform(target)).all())
            # Target attendance never contributes to source imputation.
            features = prep.named_steps["features"].transform(target.assign(attendance_est=None))
            self.assertTrue(features.attendance_est.eq(median).all())
        self.assertEqual(len(result["transfer"]), 16)

    def test_anomaly_diagnostics_attribute_errors_without_dropping_rows(self):
        data = curve("2025-10-31")
        data["time_bin"] += pd.Timedelta(minutes=90)
        original = data.copy(deep=True)
        buckets, events = anomaly_flags(data)
        self.assertEqual(buckets.sum(), 1)
        self.assertTrue(events.all())
        data["flag"] = buckets
        report = score_groups(data, {"zero": np.zeros(8)}, "flag")
        self.assertEqual(report.rows.sum(), 8)
        self.assertEqual(report.squared_error.sum(), (data.surge_delta ** 2).sum())
        pd.testing.assert_frame_equal(data.drop(columns="flag"), original)

    def test_xgboost_training_only_fit_selection_and_test_isolation(self):
        data = pd.concat([curve(day, attendance=100 + i).assign(surge_delta=float(i - 8))
                          for i, day in enumerate(pd.date_range("2025-01-01", periods=20))], ignore_index=True)
        splits = chronological_split(data).split
        data.loc[splits == "val", "attendance_est"] = 99999
        changed = data.copy()
        changed.loc[splits == "test", "surge_delta"] = 9999
        small_search = [dict(max_depth=2, n_estimators=3), dict(max_depth=3, n_estimators=5)]
        # Existing bundles supply predictions only and are never fitted/rewritten.
        with patch("src.models.train.XGB_SEARCH", small_search), \
             patch("src.models.inference.load_bundle", return_value={}), \
             patch("src.models.inference.predict_rows", side_effect=lambda bundle, rows: np.zeros(len(rows))):
            before = compare_xgboost(data, "unused")
            after = compare_xgboost(changed, "unused")
        search = before["xgboost_search"]
        selected = search.loc[search.val_rmse.idxmin(), "params"]
        self.assertEqual(before["model"].max_depth, selected["max_depth"])
        self.assertEqual(before["model"].n_estimators, selected["n_estimators"])
        self.assertEqual(before["preprocessor"].named_steps["features"].global_median_, 106)
        self.assertEqual(before["model"].get_booster().save_raw(), after["model"].get_booster().save_raw())
        pd.testing.assert_frame_equal(search, after["xgboost_search"])
        self.assertEqual(set(before["summary"].model), {"control_median", "ridge", "lightgbm", "xgboost"})
        self.assertEqual(set(before["peak_summary"].split), {"val", "test"})
        rolling = before["persistence_comparison"]
        self.assertEqual(set(rolling.model), {"control_median", "ridge", "lightgbm", "xgboost", "persistence"})
        self.assertTrue(rolling[rolling.split == "val"].rows.eq(21).all())
        self.assertTrue(rolling[rolling.split == "test"].rows.eq(28).all())

    def test_selection_uses_validation_and_never_fits_validation(self):
        first, second = ConstantModel(-9), ConstantModel(5)
        best, report = select_model([({"value": -9}, first), ({"value": 5}, second)],
                                    np.zeros((3, 1)), np.full(3, -9),
                                    np.zeros((2, 1)), np.full(2, 5))
        self.assertIs(best, second)
        np.testing.assert_array_equal(second.fit_targets, [-9] * 3)
        self.assertEqual(report.iloc[0].train_rmse, 0)
        self.assertEqual(report.iloc[1].val_rmse, 0)

    def test_pipeline_excludes_test_and_fits_preprocessing_on_train(self):
        data = pd.concat([curve(day, attendance=100 + i)
                          for i, day in enumerate(pd.date_range("2024-01-01", periods=20))],
                         ignore_index=True)
        split = chronological_split(data)
        # Deliberately invalid test predictors/targets must never reach preprocessing/scoring.
        data.loc[split.split == "test", ["temp_f", "surge_delta"]] = np.nan
        data.loc[split.split == "val", "attendance_est"] = 99999
        small_search = [{"num_leaves": 3, "min_child_samples": 16, "n_estimators": 5}]
        with patch("src.models.train.LGBM_SEARCH", small_search):
            result = train_validate(data)
        self.assertEqual(result["split_sizes"].events.to_dict(), {"train": 13, "val": 3, "test": 4})
        self.assertEqual(result["preprocessor"].named_steps["features"].global_median_, 106)
        self.assertEqual(set(result["summary"].split), {"train", "val"})
        self.assertEqual(len(result["ridge_search"]), 4)
        summary = result["summary"]
        self.assertTrue((summary[summary.model == "control_median"].rmse == 9).all())
        self.assertTrue((summary[summary.model == "ridge"].rmse < 1e-8).all())
        # A negative constant target remains negative after training and prediction.
        X_val = result["preprocessor"].transform(data[split.split == "val"])
        self.assertTrue((result["models"]["ridge"].predict(X_val) < 0).all())


if __name__ == "__main__":
    unittest.main()
