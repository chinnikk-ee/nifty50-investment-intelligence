"""BONUS — Monte-Carlo forecasting and scenario simulation.

monte_carlo_forecast : bootstrap or GBM simulation of future price paths
scenario_simulate    : instantaneous shock analysis (market / sector / vol)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.risk.metrics import alpha_beta, market_returns
from ml.utils import TRADING_DAYS_PER_YEAR, get_logger

logger = get_logger(__name__)


def monte_carlo_forecast(
    close: pd.Series,
    horizon_days: int = 252,
    n_sims: int = 2000,
    method: str = "bootstrap",
    seed: int = 42,
) -> dict:
    """Simulate future price paths from historical daily returns.

    bootstrap : resample historical returns (preserves fat tails)
    gbm       : geometric Brownian motion with historical drift/vol
    """
    rets = close.pct_change().dropna().to_numpy()
    last = float(close.iloc[-1])
    rng = np.random.default_rng(seed)

    if method == "gbm":
        mu, sigma = rets.mean(), rets.std()
        shocks = rng.normal(mu - 0.5 * sigma**2, sigma, size=(n_sims, horizon_days))
        paths = last * np.exp(np.cumsum(shocks, axis=1))
    elif method == "bootstrap":
        sampled = rng.choice(rets, size=(n_sims, horizon_days), replace=True)
        paths = last * np.cumprod(1 + sampled, axis=1)
    else:
        raise ValueError(f"unknown method '{method}' (use 'bootstrap' or 'gbm')")

    terminal = paths[:, -1]
    pct = lambda q: np.percentile(paths, q, axis=0)  # noqa: E731
    bands = {f"p{q}": pct(q) for q in (5, 25, 50, 75, 95)}
    return {
        "method": method,
        "last_close": last,
        "horizon_days": horizon_days,
        "n_sims": n_sims,
        "expected_terminal": round(float(terminal.mean()), 2),
        "prob_positive": round(float((terminal > last).mean()), 4),
        "var_95_terminal": round(float(1 - np.percentile(terminal, 5) / last), 4),
        "paths_summary": [
            {"day": d + 1, **{k: round(float(v[d]), 2) for k, v in bands.items()}}
            for d in range(0, horizon_days, max(1, horizon_days // 60))
        ],
    }


def scenario_simulate(
    close_wide: pd.DataFrame,
    weights: dict[str, float],
    sectors: pd.Series,
    market_shock: float = -0.10,
    sector_shocks: dict[str, float] | None = None,
    vol_multiplier: float = 1.0,
) -> dict:
    """Estimate the portfolio's instantaneous P&L under a stress scenario.

    Each stock moves beta * market_shock plus its sector-specific shock; the
    vol multiplier widens the uncertainty band around the point estimate.
    """
    sector_shocks = sector_shocks or {}
    market = market_returns(close_wide)
    rets = close_wide.pct_change()

    impacts, total = [], 0.0
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    for symbol, weight in w.items():
        if symbol not in close_wide.columns:
            continue
        _, beta = alpha_beta(rets[symbol].dropna(), market)
        sector = sectors.get(symbol, "Other")
        shock = beta * market_shock + sector_shocks.get(sector, 0.0)
        contribution = float(weight) * shock
        total += contribution
        impacts.append({
            "symbol": symbol, "sector": sector, "weight": round(float(weight), 4),
            "beta": round(beta, 3), "stock_impact": round(shock, 4),
            "portfolio_contribution": round(contribution, 4),
        })

    daily_vol = float(
        (rets[[s for s in w.index if s in rets.columns]] * w.reindex(rets.columns).dropna())
        .sum(axis=1).std()
    )
    band = 1.645 * daily_vol * np.sqrt(5) * vol_multiplier  # ~1-week 90% band
    return {
        "market_shock": market_shock,
        "sector_shocks": sector_shocks,
        "vol_multiplier": vol_multiplier,
        "portfolio_impact": round(total, 4),
        "impact_band_low": round(total - band, 4),
        "impact_band_high": round(total + band, 4),
        "positions": sorted(impacts, key=lambda r: r["portfolio_contribution"]),
    }
