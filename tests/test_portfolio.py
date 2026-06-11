import numpy as np
import pandas as pd
import pytest

from ml.portfolio.optimizers import (
    annualized_inputs,
    equal_weight,
    max_sharpe,
    min_volatility,
    portfolio_stats,
    risk_parity,
)
from ml.portfolio.profiles import build_portfolio


def test_equal_weight_sums_to_one():
    w = equal_weight(["A", "B", "C"])
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, 1 / 3)


def test_risk_parity_equalizes_risk_contributions(returns_wide):
    _, cov = annualized_inputs(returns_wide)
    w = risk_parity(cov)
    assert np.isclose(w.sum(), 1.0)
    sigma = cov.to_numpy()
    rc = w.to_numpy() * (sigma @ w.to_numpy())
    assert rc.std() / rc.mean() < 0.05  # near-equal contributions


def test_optimizers_respect_constraints(returns_wide):
    mu, cov = annualized_inputs(returns_wide)
    for w in (min_volatility(mu, cov, max_weight=0.5), max_sharpe(mu, cov, max_weight=0.5)):
        assert np.isclose(w.sum(), 1.0)
        assert (w >= 0).all()
        assert (w <= 0.5 + 1e-6).all()


def test_min_vol_is_lowest_vol(returns_wide):
    mu, cov = annualized_inputs(returns_wide)
    vol_min = portfolio_stats(min_volatility(mu, cov, 1.0), mu, cov)["volatility"]
    vol_eq = portfolio_stats(equal_weight(list(mu.index)), mu, cov)["volatility"]
    assert vol_min <= vol_eq + 1e-9


def test_build_portfolio_profiles(close_wide):
    for profile in ("conservative", "balanced", "aggressive"):
        out = build_portfolio(close_wide, profile, lookback_days=500)
        assert np.isclose(sum(out["allocation"].values()), 100.0, atol=0.1)
        assert out["volatility"] > 0
        assert "sharpe" in out


def test_build_portfolio_unknown_profile(close_wide):
    with pytest.raises(KeyError):
        build_portfolio(close_wide, "yolo")
