import numpy as np
import pandas as pd

from ml.features import (
    add_targets,
    bollinger_bands,
    build_feature_matrix,
    macd,
    model_feature_columns,
    moving_average,
    rsi,
)


def test_moving_average_warmup_and_value(stock_df):
    ma = moving_average(stock_df["Close"], 20)
    assert ma.iloc[:19].isna().all()
    assert np.isclose(ma.iloc[25], stock_df["Close"].iloc[6:26].mean())


def test_rsi_bounded(stock_df):
    r = rsi(stock_df["Close"])
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_monotonic_series_extremes():
    up = pd.Series(np.linspace(100, 200, 60))
    assert rsi(up).iloc[-1] > 95
    down = pd.Series(np.linspace(200, 100, 60))
    assert rsi(down).iloc[-1] < 5


def test_macd_hist_is_line_minus_signal(stock_df):
    m = macd(stock_df["Close"]).dropna()
    assert np.allclose(m["macd_hist"], m["macd"] - m["macd_signal"])


def test_bollinger_band_ordering(stock_df):
    bb = bollinger_bands(stock_df["Close"]).dropna()
    assert (bb["bb_upper"] >= bb["bb_mid"]).all()
    assert (bb["bb_mid"] >= bb["bb_lower"]).all()


def test_feature_matrix_shape_and_targets(stock_df):
    feats = add_targets(build_feature_matrix(stock_df))
    assert len(feats) == len(stock_df)
    for h in (1, 5, 20):
        assert f"target_ret_{h}" in feats
        # Last h rows have no forward label.
        assert feats[f"target_ret_{h}"].iloc[-h:].isna().all()
    # Forward 1d return matches a manual shift computation.
    manual = stock_df["Close"].shift(-1) / stock_df["Close"] - 1
    assert np.allclose(feats["target_ret_1"].dropna(), manual.dropna())


def test_model_columns_exclude_price_levels(stock_df):
    feats = add_targets(build_feature_matrix(stock_df))
    cols = model_feature_columns(feats)
    assert "Close" not in cols and "ma200" not in cols
    assert not any(c.startswith("target_") for c in cols)
    assert "rsi14" in cols and "close_over_ma50" in cols
