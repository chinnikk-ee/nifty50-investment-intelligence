"""BONUS — portfolio backtesting engine.

Simulates holding a target-weight portfolio with periodic rebalancing and
proportional transaction costs, and benchmarks it against the equal-weight
universe. Returns the equity curve plus the full risk-metric suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.risk.metrics import market_returns, portfolio_risk
from ml.utils import get_logger

logger = get_logger(__name__)


def backtest_portfolio(
    close_wide: pd.DataFrame,
    weights: dict[str, float],
    start: str | None = None,
    rebalance_days: int = 21,
    transaction_cost_bps: float = 10.0,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """Walk the price history holding `weights`, rebalancing every
    `rebalance_days` trading days and paying `transaction_cost_bps` on
    traded notional."""
    symbols = [s for s in weights if s in close_wide.columns]
    if not symbols:
        raise ValueError("none of the requested symbols exist in the price panel")
    w_target = pd.Series({s: weights[s] for s in symbols}, dtype=float)
    w_target = w_target / w_target.sum()

    prices = close_wide[symbols].dropna(how="any")
    if start:
        prices = prices[prices.index >= pd.Timestamp(start)]
    if len(prices) < rebalance_days + 2:
        raise ValueError("not enough overlapping history for a backtest")

    rets = prices.pct_change().fillna(0)
    tc = transaction_cost_bps / 10_000.0

    equity = np.empty(len(prices))
    weights_drift = w_target.copy()
    capital = initial_capital * (1 - tc)        # initial buy-in cost
    equity[0] = capital
    turnover_total = 0.0

    for t in range(1, len(prices)):
        day_ret = float((weights_drift * rets.iloc[t]).sum())
        capital *= 1 + day_ret
        # Let weights drift with returns...
        grown = weights_drift * (1 + rets.iloc[t])
        weights_drift = grown / grown.sum()
        # ...and snap back to target on rebalance days, paying costs on turnover.
        if t % rebalance_days == 0:
            turnover = float((weights_drift - w_target).abs().sum()) / 2
            capital *= 1 - turnover * tc * 2    # sell + buy legs
            turnover_total += turnover
            weights_drift = w_target.copy()
        equity[t] = capital

    curve = pd.Series(equity, index=prices.index, name="equity")
    port_rets = curve.pct_change().dropna()

    # First market return is NaN (pct_change) — zero it or it poisons cumprod.
    bench = (1 + market_returns(close_wide.loc[prices.index]).fillna(0)).cumprod()
    bench = bench / bench.iloc[0] * initial_capital

    metrics = portfolio_risk(rets, dict(w_target), market_returns(close_wide.loc[prices.index]))
    years = max(len(port_rets) / 252, 1e-9)
    return {
        "start": str(prices.index[0].date()),
        "end": str(prices.index[-1].date()),
        "initial_capital": initial_capital,
        "final_value": round(float(curve.iloc[-1]), 2),
        "total_return": round(float(curve.iloc[-1] / initial_capital - 1), 4),
        "cagr": round(float((curve.iloc[-1] / initial_capital) ** (1 / years) - 1), 4),
        "benchmark_final_value": round(float(bench.iloc[-1]), 2),
        "total_turnover": round(turnover_total, 4),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "equity_curve": [
            {"date": str(d.date()), "portfolio": round(float(v), 2),
             "benchmark": round(float(b), 2)}
            for d, v, b in zip(curve.index[::5], curve.iloc[::5], bench.iloc[::5])
        ],
    }
