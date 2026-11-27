"""Inference from fitted artifacts; no fitting or hidden scenario defaults."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.modeling import BUCKETS, predictor_frame


def load_bundle(path=Path("data/artifacts/model.joblib")):
    """Load only trusted joblib files (pickle can execute code)."""
    bundle = joblib.load(path)
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    names = bundle["preprocessor"].get_feature_names_out().tolist()
    if names != bundle["feature_names"]:
        raise ValueError("Artifact feature_names differ from fitted preprocessing order")
    if bundle["model"].n_features_in_ != len(names):
        raise ValueError("Model feature count does not match preprocessing")


def predict_rows(bundle, rows):
    """Transform with serialized training statistics and predict without clipping."""
    validate_bundle(bundle)
    X = bundle["preprocessor"].transform(predictor_frame(rows))
    return bundle["model"].predict(X)


def predict_event_curve(bundle, *, venue_zone_id, event_type, estimated_event_end,
                        attendance_est, weather, baseline=None):
    """Predict eight buckets from explicit advance inputs.

    estimated_event_end: naive America/New_York datetime, floored to 15 minutes.
    attendance_est: expected attendance or explicitly None when unavailable.
    weather: DataFrame containing exactly those eight time_bin timestamps and
        forecast/scenario precipitation_mm (hourly mm) and temp_f values.
    baseline: optional DataFrame with matching time_bin and y_baseline. Required
        only for total demand, never used as a surge predictor.
    """
    if venue_zone_id not in (189, 253):
        raise ValueError("venue_zone_id must be 189 (Barclays) or 253 (Citi Field)")
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError("event_type is required")
    end = pd.Timestamp(estimated_event_end)
    if pd.isna(end) or end.tzinfo is not None:
        raise ValueError("Supply a naive America/New_York estimated_event_end")
    times = end.floor("15min") + pd.to_timedelta(BUCKETS, unit="min")
    rows = pd.DataFrame({"time_bin": times, "mins_since_end": BUCKETS})
    rows["venue_zone_id"], rows["event_type"] = venue_zone_id, event_type
    rows["attendance_est"] = attendance_est
    weather = weather[["time_bin", "precipitation_mm", "temp_f"]].copy()
    weather["time_bin"] = pd.to_datetime(weather.time_bin)
    if len(weather) != len(times) or set(weather.time_bin) != set(times):
        raise ValueError("Weather must cover exactly the eight supported bucket timestamps")
    rows = rows.merge(weather, on="time_bin", validate="1:1", how="left")
    output = rows[["time_bin", "mins_since_end"]].copy()
    output["surge_delta_pred"] = predict_rows(bundle, rows)
    if baseline is not None:
        base = baseline[["time_bin", "y_baseline"]].copy()
        base["time_bin"] = pd.to_datetime(base.time_bin)
        if len(base) != len(times) or set(base.time_bin) != set(times):
            raise ValueError("Baseline must cover exactly the eight supported bucket timestamps")
        output = output.merge(base, on="time_bin", validate="1:1", how="left")
        if not np.isfinite(output.y_baseline.to_numpy(dtype=float)).all():
            raise ValueError("Baseline must be finite")
        output["y_total_pred"] = output.y_baseline + output.surge_delta_pred
    return output
