"""MODULE 4 — evaluation metrics for regression and direction models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of samples where the predicted return has the correct sign."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    # MAPE on returns is unstable around zero; report it on a floored denominator.
    denom = np.maximum(np.abs(y_true), 1e-3)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / denom)) * 100),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "directional_accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_proba, float)))
    else:
        out["roc_auc"] = float("nan")
    return out
