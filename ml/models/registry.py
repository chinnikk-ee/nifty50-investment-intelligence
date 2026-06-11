"""MODULE 4 — model registry.

Classical models are always available; XGBoost / LightGBM / deep models are
registered only when their libraries import cleanly, so the platform degrades
gracefully on minimal installs.
"""
from __future__ import annotations

from typing import Callable

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.utils import get_logger

logger = get_logger(__name__)

REGRESSORS: dict[str, Callable] = {}
CLASSIFIERS: dict[str, Callable] = {}


def _register(target: dict, name: str):
    def deco(fn):
        target[name] = fn
        return fn
    return deco


# ------------------------------------------------------------- regression
@_register(REGRESSORS, "linear_regression")
def _linear():
    return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])


@_register(REGRESSORS, "random_forest")
def _rf_reg():
    return RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        n_jobs=-1, random_state=42,
    )


# ------------------------------------------------------------- classification
@_register(CLASSIFIERS, "logistic_regression")
def _logit():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, C=0.5)),
    ])


@_register(CLASSIFIERS, "random_forest")
def _rf_clf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        n_jobs=-1, random_state=42,
    )


# ------------------------------------------------------------- optional boosters
try:
    from xgboost import XGBClassifier, XGBRegressor

    @_register(REGRESSORS, "xgboost")
    def _xgb_reg():
        return XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=0,
        )

    @_register(CLASSIFIERS, "xgboost")
    def _xgb_clf():
        return XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=0, eval_metric="logloss",
        )
except ImportError:  # pragma: no cover
    logger.warning("xgboost not installed — XGBoost models unavailable")

try:
    from lightgbm import LGBMRegressor

    @_register(REGRESSORS, "lightgbm")
    def _lgbm_reg():
        return LGBMRegressor(
            n_estimators=500, num_leaves=31, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=-1,
        )
except ImportError:  # pragma: no cover
    logger.warning("lightgbm not installed — LightGBM models unavailable")

# ------------------------------------------------------------- optional deep models
try:
    from ml.models.deep import LSTMForecaster, TransformerForecaster

    REGRESSORS["lstm"] = lambda: LSTMForecaster()
    REGRESSORS["transformer"] = lambda: TransformerForecaster()
except ImportError:  # pragma: no cover
    logger.warning("tensorflow not installed — LSTM/Transformer models unavailable")


def make_regressor(name: str):
    if name not in REGRESSORS:
        raise KeyError(f"Unknown regressor '{name}'. Available: {sorted(REGRESSORS)}")
    return REGRESSORS[name]()


def make_classifier(name: str):
    if name not in CLASSIFIERS:
        raise KeyError(f"Unknown classifier '{name}'. Available: {sorted(CLASSIFIERS)}")
    return CLASSIFIERS[name]()


def available_models() -> dict[str, list[str]]:
    return {"regressors": sorted(REGRESSORS), "classifiers": sorted(CLASSIFIERS)}
