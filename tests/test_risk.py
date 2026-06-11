import numpy as np
import pandas as pd

from ml.risk.metrics import (
    alpha_beta,
    conditional_var,
    max_drawdown,
    portfolio_risk,
    sharpe_ratio,
    stock_risk,
    value_at_risk,
    volatility,
)


def test_volatility_annualization():
    rets = pd.Series(np.random.default_rng(0).normal(0, 0.01, 1000))
    assert np.isclose(volatility(rets), rets.std() * np.sqrt(252))


def test_sharpe_positive_for_strong_returns():
    rets = pd.Series(np.full(252, 0.002))  # ~65% annual, zero vol -> guarded
    assert sharpe_ratio(rets) == 0.0       # zero std edge case returns 0
    noisy = pd.Series(np.random.default_rng(1).normal(0.002, 0.01, 1000))
    assert sharpe_ratio(noisy) > 1.0


def test_max_drawdown_known_path():
    # 100 -> 200 -> 100: drawdown is -50%.
    prices = pd.Series([100, 150, 200, 150, 100], dtype=float)
    rets = prices.pct_change().dropna()
    assert np.isclose(max_drawdown(rets), -0.5)


def test_var_cvar_ordering(returns_wide):
    rets = returns_wide.iloc[:, 0]
    var = value_at_risk(rets)
    cvar = conditional_var(rets)
    assert cvar >= var > 0  # expected shortfall is at least the VaR threshold


def test_beta_of_market_is_one(close_wide):
    market = close_wide.pct_change().mean(axis=1)
    _, beta = alpha_beta(market, market)
    assert np.isclose(beta, 1.0)


def test_stock_risk_keys(close_wide):
    market = close_wide.pct_change().mean(axis=1)
    out = stock_risk(close_wide.iloc[:, 0].dropna(), market)
    for key in ("volatility", "sharpe", "sortino", "calmar", "max_drawdown",
                "var_95", "cvar_95", "alpha", "beta"):
        assert key in out


def test_portfolio_risk_diversification(returns_wide):
    weights = {s: 1 / returns_wide.shape[1] for s in returns_wide.columns}
    out = portfolio_risk(returns_wide, weights)
    # A diversified portfolio can't be more volatile than its riskiest member.
    assert out["volatility"] <= returns_wide.std().max() * np.sqrt(252) + 1e-9
    assert out["diversification_ratio"] >= 1.0
