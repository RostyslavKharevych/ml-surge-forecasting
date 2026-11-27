"""Advance-curve inputs and train-fitted preprocessing; no model training or file writes."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.features.dataset import assign_split
from src.processing.events import is_holiday

BUCKETS = tuple(range(-30, 90, 15))
TARGET = "surge_delta"
INPUT_FEATURES = ["venue_zone_id", "event_type", "time_bin", "mins_since_end",
                  "attendance_est", "precipitation_mm", "temp_f"]
CATEGORICAL_FEATURES = ["venue_zone_id", "event_type"]
NUMERIC_FEATURES = ["mins_since_end", "attendance_est", "attendance_missing",
                    "precipitation_mm", "temp_f", "is_raining", "sin_time_of_day",
                    "cos_time_of_day", "day_of_week", "is_weekend", "is_holiday", "month"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def chronological_split(data):
    """Recover event starts from canonical IDs and apply the upstream 65/15/20 rule.

    Return a copy with audit metadata, never predictor columns. Reject incomplete
    curves and boundary overlaps rather than silently creating a leaky split.
    """
    rows = data.copy()
    starts = rows.event_id.str.extract(r"^(?:BARCLAYS|CITI_FIELD)_(\d{8}_\d{4})$", expand=False)
    rows["event_start"] = pd.to_datetime(starts, format="%Y%m%d_%H%M", errors="raise")
    if rows.event_start.isna().any():
        raise ValueError("Cannot reconstruct event_start from event_id")
    if rows.duplicated(["event_id", "time_bin"]).any():
        raise ValueError("Duplicate event/time bucket")
    curves = rows.groupby("event_id").mins_since_end.apply(lambda s: tuple(sorted(s)))
    if not curves.map(lambda offsets: offsets == BUCKETS).all():
        raise ValueError("Every event must contain exactly the eight available buckets")
    times = pd.to_datetime(rows.time_bin)
    if times.isna().any() or times.dt.tz is not None:
        raise ValueError("time_bin must be naive local timestamps without missing values")
    rows["time_bin"] = times
    anchors = times - pd.to_timedelta(rows.mins_since_end, unit="min")
    if (anchors.groupby(rows.event_id).nunique() != 1).any():
        raise ValueError("Buckets must share a consistent event-end anchor")
    events = rows[["event_id", "event_start"]].drop_duplicates().set_index("event_id").sort_index()
    events["split"] = assign_split(events.event_start)
    rows["split"] = rows.event_id.map(events.split)
    for earlier, later in [("train", "val"), ("val", "test")]:
        left, right = rows[rows.split == earlier], rows[rows.split == later]
        if not left.empty and not right.empty:
            if left.event_start.max() >= right.event_start.min() or left.time_bin.max() >= right.time_bin.min():
                raise ValueError("Chronological split boundary has tied starts or overlapping windows")
    return rows


def predictor_frame(data):
    """Strict input allowlist; identifiers, outcomes and observed demand never enter X."""
    inputs = data.loc[:, INPUT_FEATURES].copy()
    required = [name for name in INPUT_FEATURES if name != "attendance_est"]
    if inputs[required].isna().any().any():
        raise ValueError("All inputs except attendance must be supplied explicitly")
    if not inputs.mins_since_end.isin(BUCKETS).all():
        raise ValueError("Only the eight available relative offsets are supported")
    zones = pd.to_numeric(inputs.venue_zone_id, errors="raise")
    if not np.isfinite(zones).all() or not zones.eq(zones.round()).all():
        raise ValueError("Venue zone IDs must be finite integers")
    inputs["venue_zone_id"] = zones.astype("int64")
    return inputs


class CurveFeatures(TransformerMixin, BaseEstimator):
    """Calendar features and attendance medians learned exclusively in fit().

    Fit on training rows only. All events have eight rows, so attendance medians
    weight events equally. transform() also accepts inference rows without IDs/targets.
    """

    def fit(self, X, y=None):
        data = predictor_frame(X)
        attendance = data.attendance_est.astype("Float64").astype(float)
        self.global_median_ = attendance.median()
        if pd.isna(self.global_median_):
            raise ValueError("Training attendance has no observed values")
        data["attendance_est"] = attendance
        self.group_medians_ = data.groupby(CATEGORICAL_FEATURES).attendance_est.median()
        self.venue_medians_ = data.groupby("venue_zone_id").attendance_est.median()
        return self

    def transform(self, X):
        check_is_fitted(self, "global_median_")
        data = predictor_frame(X)
        result = pd.DataFrame(index=data.index)
        result["mins_since_end"] = data.mins_since_end
        attendance = data.attendance_est.astype("Float64").astype(float)
        result["attendance_missing"] = attendance.isna().astype(int)
        groups = pd.MultiIndex.from_frame(data[CATEGORICAL_FEATURES])
        group_fill = pd.Series(self.group_medians_.reindex(groups).to_numpy(), index=data.index)
        result["attendance_est"] = (attendance.fillna(group_fill)
                                    .fillna(data.venue_zone_id.map(self.venue_medians_))
                                    .fillna(self.global_median_))
        result["precipitation_mm"] = data.precipitation_mm
        result["temp_f"] = data.temp_f
        result["is_raining"] = (data.precipitation_mm > 0.1).astype(int)
        times = pd.to_datetime(data.time_bin)
        if times.isna().any() or times.dt.tz is not None:
            raise ValueError("Supply naive America/New_York bucket timestamps")
        minute = times.dt.hour * 60 + times.dt.minute
        result["sin_time_of_day"] = np.sin(2 * np.pi * minute / 1440)
        result["cos_time_of_day"] = np.cos(2 * np.pi * minute / 1440)
        result["day_of_week"] = times.dt.day_of_week
        result["is_weekend"] = times.dt.day_of_week.isin([4, 5, 6]).astype(int)
        result["is_holiday"] = is_holiday(times).astype(int)
        result["month"] = times.dt.month
        result["venue_zone_id"] = data.venue_zone_id.astype(str)
        result["event_type"] = data.event_type.astype(str)
        result = result[FEATURES]
        if not np.isfinite(result[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite numeric inputs; only attendance may be missing")
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)


def make_preprocessor():
    """Serializable sklearn pipeline, usable unchanged before Ridge or LightGBM.

    Scaling supports Ridge; trees can consume the same dense numeric matrix.
    Unknown categories produce all-zero indicators, without refitting on holdout.
    """
    return Pipeline([
        ("features", CurveFeatures()),
        ("encode", ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ], remainder="drop", verbose_feature_names_out=False)),
    ])
