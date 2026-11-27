"""Train/validation search, or optional XGBoost held-out comparison; no artifact writes."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from src.features.modeling import TARGET, chronological_split, make_preprocessor, predictor_frame
from src.models.evaluate import (
    compute_metrics, compute_r2, grouped_metrics, peak_errors, rmse_improvement,
    summarize_peaks, supplemental_evaluation,
)

SEED = 42
LGBM_FIXED_PARAMS = dict(learning_rate=0.03, max_depth=4, reg_lambda=10.0,
                         reg_alpha=1.0, random_state=SEED, n_jobs=1,
                         deterministic=True, force_col_wise=True, verbosity=-1)
RIDGE_ALPHAS = (0.1, 1, 10, 100)
# Eight conservative configurations: shallow trees, large leaves, shrinkage and L2.
LGBM_SEARCH = [dict(num_leaves=leaves, min_child_samples=minimum, n_estimators=trees)
               for leaves, minimum, trees in product((7, 15), (64, 128), (100, 250))]
XGB_FIXED_PARAMS = dict(objective="reg:squarederror", learning_rate=0.03,
                        min_child_weight=20, reg_alpha=1.0, reg_lambda=10.0,
                        tree_method="hist", random_state=SEED, n_jobs=1)
XGB_SEARCH = [dict(max_depth=depth, n_estimators=trees)
              for depth, trees in product((2, 3), (100, 250))]


def select_model(candidates, X_train, y_train, X_val, y_val):
    """Fit candidates on training only; choose minimum validation RMSE."""
    records, best_model, best_rmse = [], None, float("inf")
    for params, model in candidates:
        model.fit(X_train, y_train)
        train = compute_metrics(y_train, model.predict(X_train))
        val = compute_metrics(y_val, model.predict(X_val))
        records.append({"params": params, "train_rmse": train["rmse"],
                        "train_mae": train["mae"], "val_rmse": val["rmse"],
                        "val_mae": val["mae"]})
        if val["rmse"] < best_rmse:
            best_model, best_rmse = model, val["rmse"]
    if best_model is None:
        raise ValueError("At least one candidate is required")
    return best_model, pd.DataFrame(records)


def train_validate(data):
    """Test rows participate only in split reconstruction, never transformation/scoring.

    Return in-memory fitted components and reports. Validation selects parameters;
    it is not an unbiased final holdout. No refitting on combined train/validation.
    """
    rows = chronological_split(data)
    train, val = rows[rows.split == "train"], rows[rows.split == "val"]
    if train.empty or val.empty:
        raise ValueError("Nonempty training and validation splits are required")
    prep = make_preprocessor()
    X_train = prep.fit_transform(predictor_frame(train))
    X_val = prep.transform(predictor_frame(val))
    y_train, y_val = train[TARGET].to_numpy(dtype=float), val[TARGET].to_numpy(dtype=float)
    ridge, ridge_search = select_model(
        [(dict(alpha=a), Ridge(alpha=a, random_state=SEED)) for a in RIDGE_ALPHAS],
        X_train, y_train, X_val, y_val)
    lgbm, lgbm_search = select_model(
        [(params, LGBMRegressor(**params, **LGBM_FIXED_PARAMS)) for params in LGBM_SEARCH],
        X_train, y_train, X_val, y_val)
    models = {"ridge": ridge, "lightgbm": lgbm}
    summary, validation_predictions = [], {}
    for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        predictions = {"control_median": np.zeros(len(y)),
                       **{name: model.predict(X) for name, model in models.items()}}
        baseline_rmse = compute_metrics(y, predictions["control_median"])["rmse"]
        for name, prediction in predictions.items():
            metrics = compute_metrics(y, prediction)
            summary.append({"split": split, "model": name, **metrics,
                            "rmse_improvement_pct": rmse_improvement(baseline_rmse, metrics["rmse"])})
        if split == "val":
            validation_predictions = predictions
    return {"preprocessor": prep, "models": models,
            "summary": pd.DataFrame(summary), "ridge_search": ridge_search,
            "lightgbm_search": lgbm_search,
            "validation_by_venue": grouped_metrics(val, validation_predictions, "venue_zone_id"),
            "validation_by_bucket": grouped_metrics(val, validation_predictions, "mins_since_end"),
            "split_sizes": rows.groupby("split").agg(events=("event_id", "nunique"), rows=("event_id", "size"))}


def train_xgboost(train, val):
    """Select XGBoost using training-only preprocessing and validation RMSE only."""
    from xgboost import XGBRegressor

    prep = make_preprocessor()
    X_train = prep.fit_transform(predictor_frame(train))
    X_val = prep.transform(predictor_frame(val))
    model, search = select_model(
        [(params, XGBRegressor(**params, **XGB_FIXED_PARAMS)) for params in XGB_SEARCH],
        X_train, train[TARGET].to_numpy(dtype=float), X_val, val[TARGET].to_numpy(dtype=float))
    return model, prep, search


def compare_xgboost(data, artifacts):
    """Compare a validation-selected XGBoost with existing bundles; never rewrite them.

    The test split is transformed/scored only after selecting XGBoost parameters.
    Prior test results are already known; this is an additional PRD comparison,
    not a new untouched experiment or a primary-model selection exercise.
    """
    from src.models.inference import load_bundle, predict_rows

    rows = chronological_split(data)
    model, prep, search = train_xgboost(rows[rows.split == "train"], rows[rows.split == "val"])
    bundles = {name: load_bundle(Path(artifacts) / filename)
               for name, filename in [("ridge", "ridge.joblib"), ("lightgbm", "model.joblib")]}
    summary, peaks, rolling = [], [], []
    for split in ["train", "val", "test"]:
        part = rows[rows.split == split]
        predictions = {"control_median": np.zeros(len(part)),
                       **{name: predict_rows(bundle, part) for name, bundle in bundles.items()},
                       "xgboost": model.predict(prep.transform(predictor_frame(part)))}
        baseline = compute_metrics(part[TARGET], predictions["control_median"])["rmse"]
        for name, pred in predictions.items():
            metrics = compute_metrics(part[TARGET], pred)
            summary.append({"split": split, "model": name, **metrics,
                            "r2": compute_r2(part[TARGET], pred),
                            "rmse_improvement_pct": rmse_improvement(baseline, metrics["rmse"])})
        if split != "train":
            peak = summarize_peaks(peak_errors(part, predictions))
            peak.insert(0, "split", split)
            peaks.append(peak)
            _, eligible = supplemental_evaluation(part, predictions)
            rolling.append(eligible)
    return {"model": model, "preprocessor": prep, "xgboost_search": search,
            "summary": pd.DataFrame(summary), "peak_summary": pd.concat(peaks, ignore_index=True),
            "persistence_comparison": pd.concat(rolling, ignore_index=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/model_dataset.parquet"))
    parser.add_argument("--xgboost", action="store_true", help="Train XGBoost only; compare with existing Ridge/LightGBM bundles")
    parser.add_argument("--artifacts", type=Path, default=Path("data/artifacts"))
    args = parser.parse_args()
    data = pd.read_parquet(args.dataset)
    if args.xgboost:
        result = compare_xgboost(data, args.artifacts)
        for name in ["xgboost_search", "summary", "peak_summary", "persistence_comparison"]:
            print(f"\n{name}\n{result[name].to_string(index=False)}")
        selected = {**XGB_FIXED_PARAMS, **{key: result["model"].get_params()[key]
                                         for key in ["max_depth", "n_estimators"]}}
        print(f"\nSelected XGBoost: {selected}")
        return
    result = train_validate(data)
    for name in ["split_sizes", "ridge_search", "lightgbm_search", "summary",
                 "validation_by_venue", "validation_by_bucket"]:
        print(f"\n{name}\n{result[name].to_string(index=name == 'split_sizes', float_format=lambda v: f'{v:.4f}')}")
    for name, model in result["models"].items():
        print(f"\nSelected {name}: {model.get_params()}")


if __name__ == "__main__":
    main()
