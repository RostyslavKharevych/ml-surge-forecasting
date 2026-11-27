"""Regression metrics and grouped validation reports without prediction clipping."""

import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred):
    """Calculate standard continuous regression metrics."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or not y_true.size:
        raise ValueError("Expected nonempty matching one-dimensional arrays")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics require finite targets and predictions")
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"rmse": float(rmse), "mae": float(mae)}


def rmse_improvement(baseline_rmse, model_rmse):
    """Percent reduction; undefined for a perfect baseline."""
    return 100 * (baseline_rmse - model_rmse) / baseline_rmse if baseline_rmse else None


def grouped_metrics(rows, predictions, group_column):
    """Score approaches within the same venue or relative-time group."""
    records = []
    for model, values in predictions.items():
        frame = rows[[group_column, "event_id", "surge_delta"]].copy()
        frame["prediction"] = np.asarray(values)
        for group, part in frame.groupby(group_column):
            metrics = compute_metrics(part.surge_delta, part.prediction)
            baseline = compute_metrics(part.surge_delta, np.zeros(len(part)))
            records.append({group_column: group, "model": model,
                            "events": part.event_id.nunique(), "rows": len(part),
                            **metrics, "rmse_improvement_pct":
                            rmse_improvement(baseline["rmse"], metrics["rmse"])})
    return pd.DataFrame(records)


def peak_errors(rows, predictions):
    """Event peak errors across all eight buckets, without clipping negative peaks.

    Magnitude error compares max(prediction) with max(actual). Timing is signed
    predicted minus actual peak offset. Earliest bucket breaks ties; flat curves
    have no identifiable peak time and receive NaN timing errors.
    """
    records = []
    for model, values in predictions.items():
        frame = rows[["event_id", "venue_zone_id", "mins_since_end", "surge_delta"]].copy()
        frame["prediction"] = np.asarray(values)
        for event_id, part in frame.groupby("event_id"):
            part = part.sort_values("mins_since_end").reset_index(drop=True)
            actual, pred = part.surge_delta.to_numpy(), part.prediction.to_numpy()
            actual_time = part.mins_since_end.iloc[np.argmax(actual)]
            pred_time = part.mins_since_end.iloc[np.argmax(pred)]
            timing = float(pred_time - actual_time) if np.ptp(actual) > 0 and np.ptp(pred) > 0 else np.nan
            records.append({"event_id": event_id, "venue_zone_id": part.venue_zone_id.iloc[0],
                            "model": model, "peak_actual": actual.max(), "peak_pred": pred.max(),
                            "magnitude_error": pred.max() - actual.max(),
                            "timing_error_minutes": timing,
                            "actual_peak_tied": int((actual == actual.max()).sum()) > 1,
                            "pred_peak_tied": int((pred == pred.max()).sum()) > 1,
                            "curve_rmse": compute_metrics(actual, pred)["rmse"]})
    return pd.DataFrame(records)


def summarize_peaks(errors):
    records = []
    for model, part in errors.groupby("model"):
        magnitude = compute_metrics(np.zeros(len(part)), part.magnitude_error)
        records.append({"model": model, "events": len(part),
                        "magnitude_rmse": magnitude["rmse"], "magnitude_mae": magnitude["mae"],
                        "magnitude_bias": part.magnitude_error.mean(),
                        "timing_mae_minutes": part.timing_error_minutes.abs().mean(),
                        "timing_bias_minutes": part.timing_error_minutes.mean(),
                        "timing_events": part.timing_error_minutes.notna().sum()})
    return pd.DataFrame(records)


def compute_r2(y_true, y_pred):
    """R² is undefined for fewer than two observations or a constant target."""
    compute_metrics(y_true, y_pred)  # Reuse shape/finite-value validation.
    actual, predicted = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    total = np.sum((actual - actual.mean()) ** 2)
    return float(1 - np.sum((actual - predicted) ** 2) / total) if len(actual) > 1 and total > 0 else np.nan


def persistence_predictions(rows):
    """Rolling one-step total demand benchmark, using only this event's previous observation.

    First buckets retain NaN predictions and must be excluded from all comparison
    approaches. This benchmark observes demand during the event curve; the advance
    models do not have access to that information.
    """
    ordered = rows.sort_values(["event_id", "time_bin"], kind="stable").copy()
    if ordered.duplicated(["event_id", "time_bin"]).any():
        raise ValueError("Duplicate event/time bucket")
    previous = ordered.groupby("event_id", sort=False).y_actual.shift(1)
    ordered["persistence_surge_delta_pred"] = previous - ordered.y_baseline
    return ordered


def supplemental_evaluation(rows, predictions):
    """Full-curve R² and a fair seven-bucket rolling comparison; never fit models.

    predictions are positional arrays for the supplied rows. Only validation/test
    rows are accepted. Existing full-curve RMSE/MAE reports remain unchanged.
    """
    if not rows.split.isin(["val", "test"]).all():
        raise ValueError("Supplemental evaluation accepts validation/test rows only")
    full, rolling = [], []
    frame = rows.copy()
    for model, values in predictions.items():
        frame[f"pred_{model}"] = np.asarray(values)
    for split, part in frame.groupby("split", sort=False):
        for model in predictions:
            full.append({"split": split, "model": model, "rows": len(part),
                         "r2": compute_r2(part.surge_delta, part[f"pred_{model}"])})
        eligible = persistence_predictions(part).dropna(subset=["persistence_surge_delta_pred"])
        baseline = compute_metrics(eligible.surge_delta, np.zeros(len(eligible)))["rmse"]
        for model in [*predictions, "persistence"]:
            pred = eligible["persistence_surge_delta_pred" if model == "persistence" else f"pred_{model}"]
            metrics = compute_metrics(eligible.surge_delta, pred)
            rolling.append({"split": split, "model": model, "events": eligible.event_id.nunique(),
                            "rows": len(eligible), **metrics,
                            "r2": compute_r2(eligible.surge_delta, pred),
                            "rmse_improvement_pct": rmse_improvement(baseline, metrics["rmse"])})
    return pd.DataFrame(full), pd.DataFrame(rolling)


def main():
    """Evaluate existing fitted bundles; no training, artifact writes or tuning."""
    import argparse
    from pathlib import Path

    from src.features.modeling import chronological_split
    from src.models.inference import load_bundle, predict_rows

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/model_dataset.parquet"))
    parser.add_argument("--artifacts", type=Path, default=Path("data/artifacts"))
    args = parser.parse_args()
    rows = chronological_split(pd.read_parquet(args.dataset))
    rows = rows[rows.split.isin(["val", "test"])]
    predictions = {"control_median": np.zeros(len(rows))}
    for name, filename in [("ridge", "ridge.joblib"), ("lightgbm", "model.joblib")]:
        predictions[name] = predict_rows(load_bundle(args.artifacts / filename), rows)
    full, rolling = supplemental_evaluation(rows, predictions)
    print("Full-curve R² (all eight buckets)\n" + full.to_string(index=False))
    print("\nRolling persistence comparison (same seven eligible buckets per event)\n" + rolling.to_string(index=False))


if __name__ == "__main__":
    main()
