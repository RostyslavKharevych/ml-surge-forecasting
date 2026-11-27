"""Chronological venue transfer and frozen-model diagnostics; no tuning/artifact writes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.features.modeling import TARGET, chronological_split, make_preprocessor, predictor_frame
from src.models.evaluate import compute_metrics, compute_r2, peak_errors, rmse_improvement
from src.models.holdout import LGBM_PARAMS, RIDGE_PARAMS
from src.models.inference import load_bundle, predict_rows
from src.models.train import XGB_FIXED_PARAMS


def fixed_models():
    return {"ridge": Ridge(**RIDGE_PARAMS), "lightgbm": LGBMRegressor(**LGBM_PARAMS),
            "xgboost": XGBRegressor(**XGB_FIXED_PARAMS, max_depth=3, n_estimators=250)}


def score_groups(rows, predictions, group):
    records = []
    for name, values in predictions.items():
        frame = rows.copy()
        frame["prediction"] = np.asarray(values)
        for key, part in frame.groupby(group, observed=True, dropna=False):
            residual = part.prediction - part[TARGET]
            baseline = compute_metrics(part[TARGET], np.zeros(len(part)))["rmse"]
            metrics = compute_metrics(part[TARGET], part.prediction)
            records.append({group: key, "model": name, "events": part.event_id.nunique(),
                            "rows": len(part), **metrics,
                            "r2": compute_r2(part[TARGET], part.prediction), "bias": residual.mean(),
                            "actual_mean": part[TARGET].mean(), "actual_std": part[TARGET].std(ddof=0),
                            "predicted_std": part.prediction.std(ddof=0),
                            "squared_error": (residual ** 2).sum(),
                            "rmse_improvement_pct": rmse_improvement(baseline, metrics["rmse"])})
    return pd.DataFrame(records)


def cross_venue_transfer(data):
    rows = chronological_split(data)
    summary, samples = [], []
    for source, target in [(189, 253), (253, 189)]:
        train = rows[(rows.split == "train") & (rows.venue_zone_id == source)]
        prep = make_preprocessor()
        X_train = prep.fit_transform(predictor_frame(train))
        models = fixed_models()
        for model in models.values():
            model.fit(X_train, train[TARGET].to_numpy(dtype=float))
        for split in ["val", "test"]:
            evaluation = rows[(rows.split == split) & (rows.venue_zone_id == target)]
            if train.time_bin.max() >= evaluation.time_bin.min() or train.event_start.max() >= evaluation.event_start.min():
                raise ValueError("Transfer training must precede target evaluation")
            X = prep.transform(predictor_frame(evaluation))
            predictions = {"control_median": np.zeros(len(evaluation)),
                           **{name: model.predict(X) for name, model in models.items()}}
            report = score_groups(evaluation, predictions, "split")
            report.insert(0, "source_venue", source)
            report.insert(1, "target_venue", target)
            summary.append(report)
            train_events, eval_events = train.drop_duplicates("event_id"), evaluation.drop_duplicates("event_id")
            samples.append(dict(source_venue=source, target_venue=target, split=split,
                                train_events=len(train_events), train_rows=len(train),
                                eval_events=len(eval_events), eval_rows=len(evaluation),
                                train_attendance_missing=int(train_events.attendance_est.isna().sum()),
                                eval_attendance_missing=int(eval_events.attendance_est.isna().sum()),
                                training_attendance_median=prep.named_steps["features"].global_median_,
                                unseen_type_events=int((~eval_events.event_type.isin(train.event_type)).sum()),
                                train_end=train.time_bin.max(), evaluation_start=evaluation.time_bin.min()))
    return {"transfer": pd.concat(summary, ignore_index=True), "transfer_samples": pd.DataFrame(samples)}


def anomaly_flags(rows):
    """Previously suspected pattern, for attribution only; never used as a predictor/filter."""
    times = rows.time_bin
    candidate = times.dt.is_month_end & times.dt.hour.eq(23) & times.dt.minute.eq(30) & rows.y_actual.le(2)
    return candidate, rows.event_id.isin(rows.loc[candidate, "event_id"])


def model_diagnostics(data, artifacts):
    rows = chronological_split(data)
    train = rows[rows.split == "train"]
    # Reproduce the already-selected XGBoost fit, with no validation search.
    prep = make_preprocessor()
    X_train = prep.fit_transform(predictor_frame(train))
    xgb = fixed_models()["xgboost"].fit(X_train, train[TARGET].to_numpy(dtype=float))
    evaluation = rows[rows.split.isin(["val", "test"])].copy()
    predictions = {"control_median": np.zeros(len(evaluation)),
                   **{name: predict_rows(load_bundle(Path(artifacts) / filename), evaluation)
                      for name, filename in [("ridge", "ridge.joblib"), ("lightgbm", "model.joblib")]},
                   "xgboost": xgb.predict(prep.transform(predictor_frame(evaluation)))}
    summary = score_groups(evaluation, predictions, "split")
    test_mask = evaluation.split.eq("test").to_numpy()
    test = evaluation[evaluation.split == "test"].copy()
    test_predictions = {name: values[test_mask] for name, values in predictions.items()}
    test["attendance_missing"] = test.attendance_est.isna()
    test["venue_attendance"] = test.venue_zone_id.astype(str) + "_missing_" + test.attendance_missing.astype(str)
    test["surge_band"] = pd.cut(test[TARGET], [-np.inf, 0, 25, 75, np.inf],
                                 labels=["negative", "0_to_25", "25_to_75", "at_least_75"], right=False)
    test["anomaly_bucket"], test["anomaly_event"] = anomaly_flags(test)
    results = {"distribution": summary}
    for group in ["venue_zone_id", "event_type", "attendance_missing", "venue_attendance",
                  "surge_band", "anomaly_bucket", "anomaly_event"]:
        report = score_groups(test, test_predictions, group)
        report["squared_error_share_pct"] = report.squared_error / report.groupby("model").squared_error.transform("sum") * 100
        results[f"test_by_{group}"] = report
    peaks = peak_errors(test, test_predictions)
    peaks["peak_band"] = np.where(peaks.peak_actual >= 100, "at_least_100", "below_100")
    results["peak_bias"] = peaks.groupby(["model", "peak_band"]).agg(
        events=("event_id", "size"), actual_peak_mean=("peak_actual", "mean"),
        peak_bias=("magnitude_error", "mean"), peak_mae=("magnitude_error", lambda s: s.abs().mean())).reset_index()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/model_dataset.parquet"))
    parser.add_argument("--artifacts", type=Path, default=Path("data/artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/heldout/diagnostics"))
    args = parser.parse_args()
    data = pd.read_parquet(args.dataset)
    results = {**cross_venue_transfer(data), **model_diagnostics(data, args.artifacts)}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, report in results.items():
        report.to_csv(args.output / f"{name}.csv", index=False)
        print(f"\n{name}\n{report.to_string(index=False, float_format=lambda v: f'{v:.3f}')}")


if __name__ == "__main__":
    main()
