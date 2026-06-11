"""MODULE 8 — explainable AI.

For any fitted model and prediction the platform reports:
  * why the model decided what it decided (per-feature contributions)
  * the most influential features globally
  * a confidence score

SHAP is preferred (TreeExplainer for tree ensembles, LinearExplainer for
linear pipelines); LIME is available as an alternative lens; permutation /
impurity importances are the always-available fallback. Every public function
degrades gracefully when the optional library is missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from ml.utils import get_logger

logger = get_logger(__name__)


def _unwrap(model):
    """(final_estimator, transform_fn) for Pipelines, identity otherwise."""
    if isinstance(model, Pipeline):
        final = model.steps[-1][1]
        pre = Pipeline(model.steps[:-1]) if len(model.steps) > 1 else None
        return final, (pre.transform if pre else (lambda X: X))
    return model, (lambda X: X)


# ------------------------------------------------------------------ importance
def feature_importances(model, X: pd.DataFrame, y=None) -> pd.Series:
    """Global importances: impurity-based for trees, |coef| for linear models,
    permutation importance as the model-agnostic fallback (needs y)."""
    est, _ = _unwrap(model)
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        imp = np.abs(np.asarray(est.coef_, dtype=float)).ravel()[: X.shape[1]]
    elif y is not None:
        result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        imp = result.importances_mean
    else:
        raise ValueError("model exposes no importances and no y given for permutation")
    s = pd.Series(imp, index=X.columns)
    total = s.abs().sum()
    return (s / total if total > 0 else s).sort_values(ascending=False)


# ------------------------------------------------------------------ SHAP
def shap_explain(model, X_background: pd.DataFrame, X_explain: pd.DataFrame) -> pd.DataFrame | None:
    """Per-feature SHAP contributions for each row of X_explain (None if shap
    is not installed or the model is unsupported)."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP explanation")
        return None

    est, transform = _unwrap(model)
    bg = transform(X_background)
    xe = transform(X_explain)
    try:
        if hasattr(est, "feature_importances_"):          # tree ensembles
            explainer = shap.TreeExplainer(est)
            values = explainer.shap_values(xe)
        elif hasattr(est, "coef_"):                       # linear models
            explainer = shap.LinearExplainer(est, bg)
            values = explainer.shap_values(xe)
        else:                                             # model-agnostic, sampled
            explainer = shap.KernelExplainer(
                est.predict, shap.sample(bg, min(50, len(bg)), random_state=42)
            )
            values = explainer.shap_values(xe, nsamples=100, silent=True)
    except Exception as exc:
        logger.error("SHAP failed: %s", exc)
        return None

    if isinstance(values, list):  # binary classifiers return [neg, pos]
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:          # (n, features, classes)
        values = values[:, :, -1]
    return pd.DataFrame(values, columns=X_explain.columns, index=X_explain.index)


# ------------------------------------------------------------------ LIME
def lime_explain(model, X_train: pd.DataFrame, x_row: pd.Series,
                 task: str = "regression", num_features: int = 8) -> list[tuple[str, float]] | None:
    """LIME local explanation for a single row (None if lime not installed)."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        logger.warning("lime not installed — skipping LIME explanation")
        return None

    mode = "regression" if task == "regression" else "classification"
    explainer = LimeTabularExplainer(
        X_train.to_numpy(), feature_names=list(X_train.columns),
        mode=mode, discretize_continuous=True, random_state=42,
    )
    predict_fn = (model.predict if mode == "regression"
                  else getattr(model, "predict_proba", model.predict))
    exp = explainer.explain_instance(x_row.to_numpy(), predict_fn, num_features=num_features)
    return exp.as_list()


# ------------------------------------------------------------------ unified
def explain_prediction(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    x_row: pd.DataFrame,
    task: str = "regression",
    oos_metric: float | None = None,
    top_n: int = 8,
) -> dict:
    """One prediction, fully explained.

    Confidence blends model certainty (classifier probability margin, or the
    walk-forward directional accuracy for regressors) so it reflects honest
    out-of-sample skill rather than in-sample fit.
    """
    pred = float(np.asarray(model.predict(x_row)).ravel()[0])

    if task == "classification" and hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x_row)[0, 1])
        confidence = abs(proba - 0.5) * 2          # 0 = coin flip, 1 = certain
    else:
        proba = None
        # Map directional accuracy (0.5 = noise) onto [0, 1].
        confidence = max(0.0, min(1.0, ((oos_metric or 0.5) - 0.5) * 2 + 0.25))

    method = "shap"
    shap_vals = shap_explain(model, X_train.tail(250), x_row)
    if shap_vals is not None:
        contrib = shap_vals.iloc[0]
    else:
        method = "feature_importance"
        imp = feature_importances(model, X_train, y_train)
        # Sign the importance by the row's z-score so direction is meaningful.
        z = (x_row.iloc[0] - X_train.mean()) / (X_train.std() + 1e-12)
        contrib = imp * np.sign(z.reindex(imp.index).fillna(0))

    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
    return {
        "prediction": pred,
        "probability_up": proba,
        "confidence": round(float(confidence), 4),
        "method": method,
        "top_features": [
            {
                "feature": name,
                "contribution": round(float(val), 6),
                "value": round(float(x_row.iloc[0][name]), 6),
                "direction": "pushes_up" if val > 0 else "pushes_down",
            }
            for name, val in top.items()
        ],
    }
