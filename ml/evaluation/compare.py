"""MODULE 4 — model comparison across models / horizons for one stock."""
from __future__ import annotations

from typing import Literal

import pandas as pd

from ml.models.registry import CLASSIFIERS, REGRESSORS
from ml.utils import get_logger

# Defined locally (not imported from trainer) to break a circular import:
# trainer -> ml.evaluation/__init__ -> compare -> trainer. walk_forward_evaluate
# is imported lazily inside compare_models for the same reason.
Task = Literal["regression", "classification"]

logger = get_logger(__name__)

# Deep models are excluded from the default sweep: they dominate runtime and
# need GPU to be competitive. Pass them explicitly to include them.
DEFAULT_REGRESSORS = [m for m in ("linear_regression", "random_forest", "xgboost", "lightgbm")
                      if m in REGRESSORS]
DEFAULT_CLASSIFIERS = [m for m in ("logistic_regression", "random_forest", "xgboost")
                       if m in CLASSIFIERS]


def compare_models(
    df: pd.DataFrame,
    task: Task = "regression",
    horizons: tuple[int, ...] = (1, 5, 20),
    models: list[str] | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Walk-forward every (model, horizon) pair; returns a tidy metric table
    sorted so the best model per horizon comes first."""
    from ml.training.trainer import walk_forward_evaluate  # lazy: see Task note

    models = models or (DEFAULT_REGRESSORS if task == "regression" else DEFAULT_CLASSIFIERS)
    rows = []
    for horizon in horizons:
        for name in models:
            try:
                res = walk_forward_evaluate(df, name, horizon, task, n_splits=n_splits)
            except Exception as exc:
                logger.error("comparison failed for %s h=%d: %s", name, horizon, exc)
                continue
            rows.append({"model": name, "horizon": horizon, "task": task, **res.metrics})

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    sort_key = "rmse" if task == "regression" else "f1"
    ascending = task == "regression"
    return (table.sort_values(["horizon", sort_key], ascending=[True, ascending])
                 .reset_index(drop=True))


def best_model(comparison: pd.DataFrame, horizon: int) -> str:
    """Name of the top-ranked model for a horizon in a compare_models table."""
    sub = comparison[comparison["horizon"] == horizon]
    if sub.empty:
        raise ValueError(f"no comparison rows for horizon {horizon}")
    return str(sub.iloc[0]["model"])
