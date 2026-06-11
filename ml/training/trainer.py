"""MODULE 4 — training harness.

prepare_xy            : feature matrix + target extraction for one stock
walk_forward_evaluate : leakage-safe expanding-window evaluation
train_and_forecast    : fit on full history, forecast the next horizon
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from ml.evaluation.metrics import classification_metrics, regression_metrics
from ml.features import add_targets, build_feature_matrix, model_feature_columns
from ml.models.registry import make_classifier, make_regressor
from ml.training.splits import time_series_splits
from ml.utils import ARTIFACTS_DIR, get_logger

logger = get_logger(__name__)

Task = Literal["regression", "classification"]


@dataclass
class WalkForwardResult:
    model_name: str
    task: Task
    horizon: int
    metrics: dict[str, float]
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    predictions: pd.DataFrame | None = None  # Date, y_true, y_pred (out-of-fold)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "task": self.task,
            "horizon": self.horizon,
            "metrics": self.metrics,
            "fold_metrics": self.fold_metrics,
        }


def prepare_xy(
    df: pd.DataFrame, horizon: int, task: Task, sector_ret: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the model-ready matrix for one stock's OHLCV frame.

    Returns (X, y, dates) with warm-up NaNs and unlabeled tail rows dropped.
    """
    feats = add_targets(build_feature_matrix(df, sector_ret), horizons=(horizon,))
    target_col = f"target_{'ret' if task == 'regression' else 'dir'}_{horizon}"
    cols = model_feature_columns(feats)
    data = feats[["Date", target_col] + cols].dropna()
    return data[cols], data[target_col], data["Date"]


def _make_model(name: str, task: Task):
    return make_regressor(name) if task == "regression" else make_classifier(name)


def walk_forward_evaluate(
    df: pd.DataFrame,
    model_name: str,
    horizon: int = 1,
    task: Task = "regression",
    n_splits: int = 5,
    sector_ret: pd.Series | None = None,
) -> WalkForwardResult:
    """Expanding-window out-of-sample evaluation with a `horizon` gap so
    overlapping forward targets never leak across the split boundary."""
    X, y, dates = prepare_xy(df, horizon, task, sector_ret)
    rows, folds = [], []

    for train_idx, test_idx in time_series_splits(len(X), n_splits=n_splits, gap=horizon):
        model = _make_model(model_name, task)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = np.asarray(model.predict(X.iloc[test_idx])).ravel()
        y_true = y.iloc[test_idx].to_numpy()

        if task == "regression":
            folds.append(regression_metrics(y_true, pred))
            proba = None
        else:
            proba = (model.predict_proba(X.iloc[test_idx])[:, 1]
                     if hasattr(model, "predict_proba") else None)
            folds.append(classification_metrics(y_true, pred, proba))

        rows.append(pd.DataFrame({
            "Date": dates.iloc[test_idx].to_numpy(),
            "y_true": y_true,
            "y_pred": pred,
            **({"y_proba": proba} if proba is not None else {}),
        }))

    oof = pd.concat(rows, ignore_index=True)
    if task == "regression":
        overall = regression_metrics(oof["y_true"], oof["y_pred"])
    else:
        overall = classification_metrics(
            oof["y_true"], oof["y_pred"],
            oof["y_proba"] if "y_proba" in oof else None,
        )

    logger.info("walk-forward %s h=%d %s: %s", model_name, horizon, task,
                {k: round(v, 4) for k, v in overall.items()})
    return WalkForwardResult(model_name, task, horizon, overall, folds, oof)


def train_and_forecast(
    df: pd.DataFrame,
    model_name: str = "random_forest",
    horizon: int = 20,
    task: Task = "regression",
    sector_ret: pd.Series | None = None,
    save_as: str | None = None,
) -> dict:
    """Fit on the full labeled history and forecast from the latest row.

    Returns the forecast plus walk-forward quality metrics for honesty about
    out-of-sample performance (the forecast itself is in-sample-fitted).
    """
    X, y, dates = prepare_xy(df, horizon, task, sector_ret)
    evaluation = walk_forward_evaluate(df, model_name, horizon, task, sector_ret=sector_ret)

    model = _make_model(model_name, task)
    model.fit(X, y)

    # The latest feature row has no label yet — rebuild without target dropna.
    feats = build_feature_matrix(df, sector_ret)
    latest = feats[X.columns].dropna().iloc[[-1]]
    raw_pred = float(np.asarray(model.predict(latest)).ravel()[0])

    out = {
        "model": model_name,
        "task": task,
        "horizon": horizon,
        "as_of": str(feats["Date"].iloc[-1].date()),
        "last_close": float(df["Close"].iloc[-1]),
        "metrics": evaluation.metrics,
    }
    if task == "regression":
        out["predicted_return"] = raw_pred
        out["predicted_price"] = out["last_close"] * (1 + raw_pred)
    else:
        proba = (float(model.predict_proba(latest)[0, 1])
                 if hasattr(model, "predict_proba") else float(raw_pred))
        out["direction"] = "up" if raw_pred >= 0.5 else "down"
        out["probability_up"] = proba

    if save_as:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACTS_DIR / f"{save_as}.joblib"
        try:
            joblib.dump({"model": model, "columns": list(X.columns), "meta": out}, path)
            out["artifact"] = str(path)
        except Exception as exc:  # deep models are not picklable — skip silently
            logger.warning("could not persist %s: %s", save_as, exc)
    return out
