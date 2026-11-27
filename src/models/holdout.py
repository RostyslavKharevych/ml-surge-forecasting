"""Frozen held-out evaluation and diagnostic reports; no production artifact export."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from src.features.modeling import TARGET, chronological_split, make_preprocessor, predictor_frame
from src.models.evaluate import compute_metrics, grouped_metrics, peak_errors, rmse_improvement, summarize_peaks
from src.models.train import LGBM_FIXED_PARAMS, SEED

# Frozen before viewing test results. No search and no train+validation refit.
RIDGE_PARAMS = {"alpha": 100, "random_state": SEED}
LGBM_PARAMS = {**LGBM_FIXED_PARAMS, "num_leaves": 7,
               "min_child_samples": 64, "n_estimators": 250}


def evaluate_holdout(data):
    rows = chronological_split(data)
    train = rows[rows.split == "train"]
    prep = make_preprocessor()
    X_train = prep.fit_transform(predictor_frame(train))
    models = {"ridge": Ridge(**RIDGE_PARAMS), "lightgbm": LGBMRegressor(**LGBM_PARAMS)}
    for model in models.values():
        model.fit(X_train, train[TARGET].to_numpy(dtype=float))
    summary, test_predictions, X_test = [], {}, None
    for split in ["train", "val", "test"]:
        part = rows[rows.split == split]
        X = X_train if split == "train" else prep.transform(predictor_frame(part))
        predictions = {"control_median": np.zeros(len(part)),
                       **{name: model.predict(X) for name, model in models.items()}}
        baseline = compute_metrics(part[TARGET], predictions["control_median"])["rmse"]
        for name, pred in predictions.items():
            metrics = compute_metrics(part[TARGET], pred)
            summary.append({"split": split, "model": name, **metrics,
                            "rmse_improvement_pct": rmse_improvement(baseline, metrics["rmse"])})
        if split == "test":
            test_predictions, X_test = predictions, X
    test = rows[rows.split == "test"].reset_index(drop=True)
    errors = peak_errors(test, test_predictions)
    lgbm = models["lightgbm"]
    # LightGBM's built-in exact TreeSHAP implementation; final column is expected value.
    contributions = lgbm.booster_.predict(X_test, pred_contrib=True)
    if not np.allclose(contributions.sum(axis=1), test_predictions["lightgbm"]):
        raise ValueError("TreeSHAP additivity check failed")
    importance = pd.DataFrame({"feature": prep.get_feature_names_out(),
                               "gain": lgbm.booster_.feature_importance(importance_type="gain"),
                               "splits": lgbm.booster_.feature_importance(importance_type="split"),
                               "mean_abs_shap": np.abs(contributions[:, :-1]).mean(axis=0)})
    return dict(summary=pd.DataFrame(summary), test_by_venue=grouped_metrics(test, test_predictions, "venue_zone_id"),
                test_by_bucket=grouped_metrics(test, test_predictions, "mins_since_end"),
                peak_events=errors, peak_summary=summarize_peaks(errors), feature_importance=importance,
                test=test, predictions=test_predictions, preprocessor=prep, models=models)


def write_reports(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    for name in ["summary", "test_by_venue", "test_by_bucket", "peak_events", "peak_summary", "feature_importance"]:
        result[name].to_csv(output / f"{name}.csv", index=False)
    imp = result["feature_importance"].sort_values("mean_abs_shap", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for ax, column, title in [(axes[0], "gain", "LightGBM training split gain"),
                              (axes[1], "mean_abs_shap", "Test mean absolute TreeSHAP (pickups)")]:
        top = imp.nlargest(12, column).sort_values(column)
        ax.barh(top.feature, top[column])
        ax.set_title(title)
    fig.savefig(output / "feature_importance.png", dpi=160)
    plt.close(fig)
    # Diagnostic selection by LightGBM test curve RMSE: best, median, worst per venue.
    selected = []
    errors = result["peak_events"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained", sharex=True)
    for row, venue in enumerate([189, 253]):
        ranked = errors[(errors.model == "lightgbm") & (errors.venue_zone_id == venue)].sort_values(["curve_rmse", "event_id"])
        for column, (label, position) in enumerate([("best", 0), ("median", len(ranked) // 2), ("worst", -1)]):
            event = ranked.iloc[position]
            selected.append({"venue_zone_id": venue, "selection": label,
                             "event_id": event.event_id, "lightgbm_curve_rmse": event.curve_rmse})
            part = result["test"][result["test"].event_id == event.event_id].sort_values("mins_since_end")
            ax = axes[row, column]
            ax.plot(part.mins_since_end, part.surge_delta, "o-", color="black", label="Actual")
            for name, values in result["predictions"].items():
                ax.plot(part.mins_since_end, np.asarray(values)[part.index], "--", label=name)
            ax.set_title(f"{event.event_id}\n{label}: LGBM RMSE {event.curve_rmse:.1f}", fontsize=10)
            ax.set_xlabel("Minutes since floored event end")
            ax.set_ylabel("Surge delta (weighted pickups)")
            ax.set_xticks(range(-30, 90, 15))
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output / "representative_curves.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(selected).to_csv(output / "curve_selection.csv", index=False)
    sections = [f"Frozen Ridge parameters: {RIDGE_PARAMS}", f"Frozen LightGBM parameters: {LGBM_PARAMS}"]
    for name in ["summary", "test_by_venue", "test_by_bucket", "peak_summary"]:
        sections.append(name + "\n" + result[name].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sections.append("Feature importance\n" + imp.to_string(index=False))
    sections.append("Diagnostics only; frozen training-only fits. See docs/modeling-preparation.md "
                    "for methodology, input availability and limitations. Peak timing uses earliest "
                    "tied maxima and excludes flat curves. Figures select best/median/worst "
                    "LightGBM errors per venue. TreeSHAP additivity was verified.")
    test = result["test"]
    missing = test.drop_duplicates("event_id").groupby("venue_zone_id").attendance_est.agg(["size", "count"])
    sections.append("Test attendance observed counts\n" + missing.to_string())
    (output / "findings.txt").write_text("\n\n".join(sections) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/model_dataset.parquet"))
    parser.add_argument("--output", type=Path, default=Path("reports/heldout"))
    args = parser.parse_args()
    result = evaluate_holdout(pd.read_parquet(args.dataset))
    write_reports(result, args.output)
    for name in ["summary", "test_by_venue", "test_by_bucket", "peak_summary"]:
        print(f"\n{name}\n{result[name].to_string(index=False)}")
    print("\nTop SHAP features\n", result["feature_importance"].nlargest(10, "mean_abs_shap").to_string(index=False))


if __name__ == "__main__":
    main()
