"""Evaluation and holdout verification metrics (RMSE, MAE, R2)."""

import numpy as np


def compute_metrics(y_true, y_pred):
    """Calculate standard continuous regression metrics."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"rmse": float(rmse), "mae": float(mae)}