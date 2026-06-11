"""MODULE 6 — risk assessment engine.

All inputs are daily simple-return Series/DataFrames; all ratios are
annualized with 252 trading days. The market proxy is the equal-weight mean
return of the full NIFTY universe (the dataset has no index series).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ml.utils import TRADING_DAYS_PER_YEAR

# Indian 10y G-Sec yield ballpark; override via env for other assumptions.
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.06"))
_DAILY_RF = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR


# ------------------------------------------------------------------ core
def volatility(returns: pd.Series, annualize: bool = True) -> float:
    vol = float(returns.std())
    return vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else vol


def sharpe_ratio(returns: pd.Series) -> float:
    excess = returns - _DAILY_RF
    sd = excess.std()
    # Epsilon guard: constant series give a float-noise std, not exactly 0.
    if np.isnan(sd) or sd < 1e-12:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series) -> float:
    excess = returns - _DAILY_RF
    downside = excess[excess < 0]
    dd = downside.std()
    if np.isnan(dd) or dd < 1e-12 or len(downside) == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0)).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    annual_ret = float(returns.mean()) * TRADING_DAYS_PER_YEAR
    return float(annual_ret / mdd)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical one-day VaR, reported as a positive loss fraction."""
    return float(-np.nanpercentile(returns.dropna(), (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected shortfall beyond the VaR threshold (positive loss fraction)."""
    r = returns.dropna()
    cutoff = np.nanpercentile(r, (1 - confidence) * 100)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else value_at_risk(returns, confidence)


def alpha_beta(returns: pd.Series, market: pd.Series) -> tuple[float, float]:
    """Annualized CAPM alpha and beta versus the market proxy."""
    joined = pd.concat([returns, market], axis=1, join="inner").dropna()
    if len(joined) < 30:
        return 0.0, 1.0
    r, m = joined.iloc[:, 0] - _DAILY_RF, joined.iloc[:, 1] - _DAILY_RF
    var_m = m.var()
    beta = float(m.cov(r) / var_m) if var_m > 0 else 1.0
    alpha_daily = float(r.mean() - beta * m.mean())
    return alpha_daily * TRADING_DAYS_PER_YEAR, beta


def market_returns(close_wide: pd.DataFrame) -> pd.Series:
    """Equal-weight universe return used as the market proxy."""
    return close_wide.pct_change().mean(axis=1)


# ------------------------------------------------------------------ reports
def _profile(returns: pd.Series, market: pd.Series | None) -> dict[str, float]:
    out = {
        "annual_return": float(returns.mean()) * TRADING_DAYS_PER_YEAR,
        "volatility": volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "calmar": calmar_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "var_95": value_at_risk(returns),
        "cvar_95": conditional_var(returns),
    }
    if market is not None:
        alpha, beta = alpha_beta(returns, market)
        out["alpha"], out["beta"] = alpha, beta
    return out


def stock_risk(close: pd.Series, market: pd.Series | None = None,
               lookback_days: int | None = None) -> dict[str, float]:
    """Stock-level risk profile from a Date-indexed close-price series."""
    returns = close.pct_change().dropna()
    if lookback_days:
        returns = returns.tail(lookback_days)
        market = market.tail(lookback_days) if market is not None else None
    return _profile(returns, market)


def portfolio_risk(returns_wide: pd.DataFrame, weights: dict[str, float],
                   market: pd.Series | None = None) -> dict[str, float]:
    """Portfolio-level risk from per-stock daily returns and target weights."""
    symbols = [s for s in weights if s in returns_wide.columns]
    w = pd.Series({s: weights[s] for s in symbols})
    w = w / w.sum()
    port = (returns_wide[symbols] * w).sum(axis=1, min_count=1).dropna()
    out = _profile(port, market)
    # Diversification ratio: weighted-average stand-alone vol over portfolio vol.
    stand_alone = returns_wide[symbols].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if out["volatility"] > 0:
        out["diversification_ratio"] = float((stand_alone * w).sum() / out["volatility"])
    return out
