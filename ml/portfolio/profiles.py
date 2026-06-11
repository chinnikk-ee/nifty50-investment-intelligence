"""MODULE 5 — investor profiles -> concrete portfolio recipes.

Each profile filters the stock universe, picks an optimization method and a
concentration cap, producing materially different portfolios for the same
data. `build_portfolio` is the single entry point used by the API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.portfolio.optimizers import optimize, portfolio_stats, annualized_inputs
from ml.utils import TRADING_DAYS_PER_YEAR, get_logger

logger = get_logger(__name__)

INVESTOR_PROFILES: dict[str, dict] = {
    "conservative": {
        "label": "Conservative",
        "description": "Capital preservation first: low-volatility universe, "
                       "minimum-volatility optimization, tight position caps.",
        "method": "min_volatility",
        "max_weight": 0.12,
        "universe_filter": "low_vol",     # keep the calmest 60% of stocks
    },
    "balanced": {
        "label": "Balanced",
        "description": "Growth with guardrails: full universe, risk-parity "
                       "allocation so no stock dominates the risk budget.",
        "method": "risk_parity",
        "max_weight": 0.20,
        "universe_filter": None,
    },
    "aggressive": {
        "label": "Aggressive",
        "description": "Return maximization: momentum-tilted universe, "
                       "maximum-Sharpe optimization, larger position caps.",
        "method": "max_sharpe",
        "max_weight": 0.30,
        "universe_filter": "momentum",    # keep the strongest 60% by 6m return
    },
}


def _filter_universe(returns_wide: pd.DataFrame, mode: str | None) -> pd.DataFrame:
    if mode is None or returns_wide.shape[1] <= 8:
        return returns_wide
    keep = max(8, int(returns_wide.shape[1] * 0.6))
    if mode == "low_vol":
        ranked = returns_wide.std().nsmallest(keep).index
    elif mode == "momentum":
        ranked = (1 + returns_wide.tail(126)).prod().nlargest(keep).index
    else:
        raise ValueError(f"unknown universe filter '{mode}'")
    return returns_wide[sorted(ranked)]


def build_portfolio(
    close_wide: pd.DataFrame,
    profile: str = "balanced",
    method: str | None = None,
    lookback_days: int = 756,
    symbols: list[str] | None = None,
) -> dict:
    """Build a portfolio for an investor profile (or an explicit method).

    Returns allocation percentages plus expected return / volatility / Sharpe.
    """
    if profile not in INVESTOR_PROFILES:
        raise KeyError(f"Unknown profile '{profile}'. Available: {sorted(INVESTOR_PROFILES)}")
    spec = INVESTOR_PROFILES[profile]

    prices = close_wide[symbols] if symbols else close_wide
    returns = prices.pct_change().tail(lookback_days)
    # Require reasonable coverage in the lookback window.
    returns = returns.dropna(axis=1, thresh=int(len(returns) * 0.8)).dropna(how="all")
    if returns.shape[1] < 2:
        raise ValueError("not enough stocks with sufficient history")

    if symbols is None:
        returns = _filter_universe(returns, spec["universe_filter"])

    chosen_method = method or spec["method"]
    weights, stats = optimize(chosen_method, returns.fillna(0), max_weight=spec["max_weight"])
    weights = weights[weights > 0].sort_values(ascending=False)

    return {
        "profile": profile,
        "profile_label": spec["label"],
        "profile_description": spec["description"],
        "method": chosen_method,
        "lookback_days": int(returns.shape[0]),
        "allocation": {s: round(float(w) * 100, 2) for s, w in weights.items()},
        **{k: round(v, 4) for k, v in stats.items()},
    }


def efficient_frontier(close_wide: pd.DataFrame, n_points: int = 25,
                       lookback_days: int = 756) -> list[dict]:
    """Random + optimized portfolios tracing the risk/return cloud for the UI."""
    returns = close_wide.pct_change().tail(lookback_days).dropna(axis=1, thresh=50).fillna(0)
    mu, cov = annualized_inputs(returns)
    rng = np.random.default_rng(7)
    points = []
    for _ in range(n_points * 8):
        w = rng.dirichlet(np.ones(len(mu)))
        stats = portfolio_stats(pd.Series(w, index=mu.index), mu, cov)
        points.append({k: round(v, 4) for k, v in stats.items()})
    return points
