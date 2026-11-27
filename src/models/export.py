"""Contract 2 export from frozen training-only fits; no tuning."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.modeling import chronological_split
from src.models.holdout import LGBM_PARAMS, RIDGE_PARAMS, evaluate_holdout
from src.models.inference import load_bundle, predict_event_curve, predict_rows

PREDICTION_COLUMNS = ["event_id", "time_bin", "mins_since_end", "y_actual", "y_baseline",
                      "surge_delta_actual", "surge_delta_pred", "y_total_pred"]
METRIC_KEYS = ["baseline_rmse", "ridge_rmse", "lgbm_rmse", "rmse_reduction_pct", "top_features"]


def export_artifacts(dataset, output):
    dataset, output = Path(dataset), Path(output)
    data = pd.read_parquet(dataset)
    result = evaluate_holdout(data)
    training = chronological_split(data).query("split == 'train'")
    output.mkdir(parents=True, exist_ok=True)
    names = result["preprocessor"].get_feature_names_out().tolist()
    test = result["test"]
    for name, filename in [("lightgbm", "model.joblib"), ("ridge", "ridge.joblib")]:
        bundle = {"model": result["models"][name], "feature_names": names,
                  "preprocessor": result["preprocessor"]}
        joblib.dump(bundle, output / filename)
        restored = load_bundle(output / filename)
        np.testing.assert_allclose(predict_rows(restored, test), result["predictions"][name], rtol=0, atol=0)
    predictions = test[["event_id", "time_bin", "mins_since_end", "y_actual", "y_baseline"]].copy()
    predictions["surge_delta_actual"] = test.surge_delta
    predictions["surge_delta_pred"] = result["predictions"]["lightgbm"]
    predictions["y_total_pred"] = predictions.y_baseline + predictions.surge_delta_pred
    predictions = predictions[PREDICTION_COLUMNS].sort_values(["time_bin", "event_id"]).reset_index(drop=True)
    if not predictions.time_bin.dt.year.eq(2025).all():
        raise ValueError("predictions_2025 must contain only 2025 test buckets")
    predictions.to_parquet(output / "predictions_2025.parquet", index=False)
    restored_predictions = pd.read_parquet(output / "predictions_2025.parquet")
    pd.testing.assert_frame_equal(predictions, restored_predictions)
    metrics_by_model = result["summary"].query("split == 'test'").set_index("model")
    # Contract does not prescribe item structure: top_features is ordered feature-name strings.
    importance = result["feature_importance"].sort_values("gain", ascending=False)
    metrics = dict(baseline_rmse=float(metrics_by_model.loc["control_median", "rmse"]),
                   ridge_rmse=float(metrics_by_model.loc["ridge", "rmse"]),
                   lgbm_rmse=float(metrics_by_model.loc["lightgbm", "rmse"]),
                   rmse_reduction_pct=float(metrics_by_model.loc["lightgbm", "rmse_improvement_pct"]),
                   top_features=importance.feature.head(5).tolist())
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    metadata = dict(primary_model="lightgbm", comparison_model="ridge",
                    selection_reason="Validation RMSE, peak reconstruction, and Contract 2; Ridge has slightly lower test RMSE.",
                    fit_split=f"train only: {training.event_id.nunique()} events / {len(training)} rows",
                    prediction_split=f"test only: {test.event_id.nunique()} events / {len(test)} rows",
                    prediction_start=str(predictions.time_bin.min()), prediction_end=str(predictions.time_bin.max()),
                    coverage_note="Not full-year 2025. Training and validation predictions are excluded.",
                    supported_offsets=list(range(-30, 90, 15)),
                    contract_discrepancy="Eight buckets through +75; +90 is unsupported and not synthesized.",
                    model_bundle_keys=["model", "feature_names", "preprocessor"],
                    top_features_format="Five encoded feature-name strings ordered by descending training split gain; full scores in feature_importance.csv.",
                    ridge_parameters=RIDGE_PARAMS, lightgbm_parameters=LGBM_PARAMS,
                    dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    versions={package: version(package) for package in
                              ["numpy", "scipy", "pandas", "pyarrow", "scikit-learn",
                               "lightgbm", "joblib", "matplotlib", "pytest"]})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    importance.to_csv(output / "feature_importance.csv", index=False)
    # Explicit retrospective replay, not an assertion that historical weather was forecast available.
    examples = []
    bundle = load_bundle(output / "model.joblib")
    for venue in [189, 253]:
        event_id = test[test.venue_zone_id == venue].event_id.iloc[0]
        event = test[test.event_id == event_id].sort_values("mins_since_end")
        end_anchor = event.time_bin.iloc[0] + pd.Timedelta(minutes=30)
        attendance = event.attendance_est.iloc[0]
        curve = predict_event_curve(bundle, venue_zone_id=venue, event_type=event.event_type.iloc[0],
                                    estimated_event_end=end_anchor,
                                    attendance_est=None if pd.isna(attendance) else int(attendance),
                                    weather=event[["time_bin", "precipitation_mm", "temp_f"]],
                                    baseline=event[["time_bin", "y_baseline"]])
        np.testing.assert_allclose(curve.surge_delta_pred, predict_rows(bundle, event), rtol=0, atol=0)
        curve.insert(0, "event_id", event_id)
        examples.append(curve)
    pd.concat(examples).to_csv(output / "inference_examples.csv", index=False)
    return metadata, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/model_dataset.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/artifacts"))
    args = parser.parse_args()
    metadata, metrics = export_artifacts(args.dataset, args.output)
    print(json.dumps({"coverage": metadata, "metrics": metrics}, indent=2))
    for path in sorted(args.output.iterdir()):
        print(f"{path}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
