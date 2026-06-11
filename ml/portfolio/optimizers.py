"""MODULE 5 — portfolio optimizers.

Implemented directly on numpy/scipy so the platform has no hard dependency on
PyPortfolioOpt/CVXPY (both remain optional accelerators). All optimizers take
annualized expected returns `mu` (Series) and an annualized covariance matrix
`cov` (DataFrame) over the same symbols, and return a weight Series that sums
to 1 with 0 <= w <= max_weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ml.risk.metrics import RISK_FREE_RATE
from ml.utils import TRADING_DAYS_PER_YEAR, get_logger

logger = get_logger(__name__)


def annualized_inputs(returns_wide: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Daily simple returns -> (annualized mean vector, annualized covariance).
    Means are shrunk 50% toward the cross-sectional average: raw historical
    means are the noisiest input to mean-variance and shrinkage materially
    stabilizes the weights."""
    mu_raw = returns_wide.mean() * TRADING_DAYS_PER_YEAR
    mu = 0.5 * mu_raw + 0.5 * mu_raw.mean()
    cov = returns_wide.cov() * TRADING_DAYS_PER_YEAR
    # Ridge for numerical stability on short/collinear histories.
    cov = cov + np.eye(len(cov)) * 1e-6
    return mu, cov


def portfolio_stats(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame) -> dict[str, float]:
    w = weights.reindex(mu.index).fillna(0).to_numpy()
    ret = float(w @ mu.to_numpy())
    vol = float(np.sqrt(w @ cov.to_numpy() @ w))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0
    return {"expected_return": ret, "volatility": vol, "sharpe": sharpe}


def _clean(w: np.ndarray, index: pd.Index) -> pd.Series:
    w = np.clip(w, 0, None)
    w[w < 1e-4] = 0.0
    w = w / w.sum()
    return pd.Series(np.round(w, 6), index=index)


# ------------------------------------------------------------------ methods
def equal_weight(symbols: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(symbols), index=pd.Index(symbols, name="Symbol"))


def risk_parity(cov: pd.DataFrame, max_iter: int = 200, tol: float = 1e-8) -> pd.Series:
    """Equal-risk-contribution weights via Spinu-style fixed-point iteration."""
    sigma = cov.to_numpy()
    n = len(sigma)
    w = 1 / np.sqrt(np.diag(sigma))
    w = w / w.sum()
    for _ in range(max_iter):
        marginal = sigma @ w
        rc = w * marginal              # risk contributions
        target = rc.mean()
        w_new = w * np.sqrt(target / np.maximum(rc, 1e-12))
        w_new = w_new / w_new.sum()
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w = w_new
    return _clean(w, cov.index)


def _slsqp(objective, mu: pd.Series, cov: pd.DataFrame, max_weight: float) -> pd.Series:
    n = len(mu)
    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    x0 = np.full(n, 1.0 / n)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
    if not res.success:
        logger.warning("SLSQP did not converge (%s) — falling back to inverse-vol", res.message)
        iv = 1 / np.sqrt(np.diag(cov.to_numpy()))
        return _clean(iv, mu.index)
    return _clean(res.x, mu.index)


def min_volatility(mu: pd.Series, cov: pd.DataFrame, max_weight: float = 0.15) -> pd.Series:
    sigma = cov.to_numpy()
    return _slsqp(lambda w: w @ sigma @ w, mu, cov, max_weight)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, max_weight: float = 0.25) -> pd.Series:
    m, sigma = mu.to_numpy(), cov.to_numpy()

    def neg_sharpe(w):
        vol = np.sqrt(w @ sigma @ w)
        return -(w @ m - RISK_FREE_RATE) / max(vol, 1e-9)

    return _slsqp(neg_sharpe, mu, cov, max_weight)


def mean_variance(mu: pd.Series, cov: pd.DataFrame, risk_aversion: float = 3.0,
                  max_weight: float = 0.25) -> pd.Series:
    """Classic Markowitz utility: maximize w'mu - (lambda/2) w'Σw."""
    m, sigma = mu.to_numpy(), cov.to_numpy()
    return _slsqp(lambda w: -(w @ m - 0.5 * risk_aversion * (w @ sigma @ w)),
                  mu, cov, max_weight)


METHODS = {
    "equal_weight": "Equal Weight",
    "risk_parity": "Risk Parity",
    "mean_variance": "Mean-Variance (Markowitz)",
    "max_sharpe": "Maximum Sharpe",
    "min_volatility": "Minimum Volatility",
}


def optimize(method: str, returns_wide: pd.DataFrame, max_weight: float = 0.25,
             risk_aversion: float = 3.0) -> tuple[pd.Series, dict[str, float]]:
    """Run one method by name; returns (weights, stats)."""
    if method not in METHODS:
        raise KeyError(f"Unknown method '{method}'. Available: {sorted(METHODS)}")
    mu, cov = annualized_inputs(returns_wide)
    if method == "equal_weight":
        weights = equal_weight(list(returns_wide.columns))
    elif method == "risk_parity":
        weights = risk_parity(cov)
    elif method == "min_volatility":
        weights = min_volatility(mu, cov, max_weight)
    elif method == "max_sharpe":
        weights = max_sharpe(mu, cov, max_weight)
    else:
        weights = mean_variance(mu, cov, risk_aversion, max_weight)
    return weights, portfolio_stats(weights, mu, cov)
