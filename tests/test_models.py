import numpy as np

from ml.evaluation.metrics import classification_metrics, regression_metrics
from ml.models.registry import available_models, make_classifier, make_regressor
from ml.training.splits import time_series_splits, walk_forward_splits
from ml.training.trainer import prepare_xy, train_and_forecast, walk_forward_evaluate


def test_registry_core_models_present():
    models = available_models()
    assert "linear_regression" in models["regressors"]
    assert "random_forest" in models["regressors"]
    assert "logistic_regression" in models["classifiers"]
    make_regressor("linear_regression")
    make_classifier("random_forest")


def test_splits_are_temporal_and_gapped():
    for train, test in time_series_splits(1000, n_splits=5, gap=20):
        assert train.max() < test.min()
        assert test.min() - train.max() > 20  # leakage gap honored


def test_walk_forward_splits_cover_data():
    splits = list(walk_forward_splits(500, train_window=200, test_window=50))
    assert splits, "no splits generated"
    for train, test in splits:
        assert train.max() < test.min()


def test_prepare_xy_no_nans(stock_df):
    X, y, dates = prepare_xy(stock_df, horizon=5, task="regression")
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert len(X) == len(y) == len(dates)


def test_walk_forward_evaluate_regression(stock_df):
    res = walk_forward_evaluate(stock_df, "linear_regression", horizon=5, n_splits=3)
    for key in ("rmse", "mae", "mape", "r2", "directional_accuracy"):
        assert key in res.metrics
    assert len(res.fold_metrics) == 3
    assert res.predictions is not None and len(res.predictions) > 100


def test_walk_forward_evaluate_classification(stock_df):
    res = walk_forward_evaluate(stock_df, "logistic_regression", horizon=5,
                                task="classification", n_splits=3)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in res.metrics
    assert 0 <= res.metrics["accuracy"] <= 1


def test_train_and_forecast_outputs(stock_df):
    out = train_and_forecast(stock_df, "linear_regression", horizon=5)
    assert "predicted_return" in out and "predicted_price" in out
    expected = out["last_close"] * (1 + out["predicted_return"])
    assert np.isclose(out["predicted_price"], expected)


def test_metric_functions_known_values():
    reg = regression_metrics([0.01, -0.02, 0.03], [0.01, -0.02, 0.03])
    assert reg["rmse"] == 0 and reg["r2"] == 1 and reg["directional_accuracy"] == 1
    clf = classification_metrics([1, 0, 1, 0], [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert clf["accuracy"] == 1 and clf["roc_auc"] == 1
