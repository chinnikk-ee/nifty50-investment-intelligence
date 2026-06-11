"""MODULE 9 — investment intelligence engine.

Produces explainable BUY / HOLD / SELL recommendations from five signals,
each normalized to [-1, 1]:

  forecast   : model-predicted 20-day forward return (walk-forward validated)
  momentum   : 21-day rate of change vs the stock's own volatility
  trend      : price vs MA50/MA200 plus MACD histogram sign
  risk_adj   : trailing Sharpe ratio (risk-adjusted performance)
  sector_rel : 21-day return minus the stock's sector average

The composite is a weighted sum; thresholds map it to an action and every
component is reported with a natural-language reason so the user always sees
*why*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import build_feature_matrix
from ml.risk.metrics import sharpe_ratio
from ml.utils import get_logger

logger = get_logger(__name__)

WEIGHTS = {"forecast": 0.30, "momentum": 0.20, "trend": 0.20, "risk_adj": 0.15, "sector_rel": 0.15}
BUY_THRESHOLD, SELL_THRESHOLD = 0.20, -0.20


def _squash(x: float, scale: float) -> float:
    """tanh squashing so extreme inputs saturate instead of dominating."""
    return float(np.tanh(x / scale)) if np.isfinite(x) else 0.0


def score_components(df: pd.DataFrame, predicted_return: float | None,
                     sector_return_21d: float | None) -> dict[str, float]:
    feats = build_feature_matrix(df)
    last = feats.iloc[-1]
    ret = feats["ret_1d"].dropna()

    momentum_signal = _squash(last["roc21"] / max(last["vol_21d"] * 100 / 16, 0.5), 5.0)

    trend_parts = [
        np.sign(last["close_over_ma50"]) if np.isfinite(last["close_over_ma50"]) else 0,
        np.sign(last["close_over_ma200"]) if np.isfinite(last["close_over_ma200"]) else 0,
        np.sign(last["macd_hist"]) if np.isfinite(last["macd_hist"]) else 0,
    ]
    trend_signal = float(np.mean(trend_parts))

    sharpe = sharpe_ratio(ret.tail(252))
    risk_signal = _squash(sharpe, 1.5)

    forecast_signal = _squash(predicted_return, 0.05) if predicted_return is not None else 0.0

    if sector_return_21d is not None:
        stock_21d = float(df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1) if len(df) > 22 else 0.0
        sector_signal = _squash(stock_21d - sector_return_21d, 0.04)
    else:
        sector_signal = 0.0

    return {
        "forecast": round(forecast_signal, 4),
        "momentum": round(momentum_signal, 4),
        "trend": round(trend_signal, 4),
        "risk_adj": round(risk_signal, 4),
        "sector_rel": round(sector_signal, 4),
    }


_REASONS = {
    "forecast": ("the model forecasts a positive {h}-day return",
                 "the model forecasts a negative {h}-day return"),
    "momentum": ("momentum is strong", "momentum is weak"),
    "trend": ("price is trading above its long-term trend", "price is below its long-term trend"),
    "risk_adj": ("risk-adjusted performance (Sharpe) is attractive",
                 "risk-adjusted performance (Sharpe) is poor"),
    "sector_rel": ("it is outperforming its sector", "it is lagging its sector"),
}


def _reasoning(symbol: str, action: str, components: dict[str, float], horizon: int) -> str:
    ranked = sorted(components.items(), key=lambda kv: abs(kv[1]), reverse=True)
    drivers = [(k, v) for k, v in ranked if abs(v) >= 0.1][:3] or ranked[:2]
    phrases = [
        _REASONS[k][0 if v > 0 else 1].format(h=horizon) for k, v in drivers
    ]
    if action == "HOLD":
        return (f"HOLD {symbol}: signals are mixed — " + "; ".join(phrases) +
                ". No edge strong enough to act on.")
    joined = ", ".join(phrases[:-1]) + (" and " + phrases[-1] if len(phrases) > 1 else phrases[0] if len(phrases) == 1 else "")
    return f"{action} {symbol} because {joined}."


def recommend_stock(
    symbol: str,
    df: pd.DataFrame,
    predicted_return: float | None = None,
    sector_return_21d: float | None = None,
    horizon: int = 20,
) -> dict:
    """Score one stock and return an explainable recommendation."""
    components = score_components(df, predicted_return, sector_return_21d)
    composite = float(sum(WEIGHTS[k] * v for k, v in components.items()))
    action = "BUY" if composite >= BUY_THRESHOLD else "SELL" if composite <= SELL_THRESHOLD else "HOLD"
    return {
        "symbol": symbol,
        "action": action,
        "score": round(composite, 4),
        "components": components,
        "weights": WEIGHTS,
        "predicted_return": predicted_return,
        "horizon": horizon,
        "last_close": float(df["Close"].iloc[-1]),
        "reasoning": _reasoning(symbol, action, components, horizon),
    }


def recommend_all(
    panel: pd.DataFrame,
    forecasts: dict[str, float] | None = None,
    horizon: int = 20,
) -> list[dict]:
    """Recommendations for every stock in the panel, BUY-first by score.

    `forecasts` maps symbol -> predicted forward return; omitted symbols are
    scored on technical/risk signals only.
    """
    forecasts = forecasts or {}
    wide = panel.pivot_table(index="Date", columns="Symbol", values="Close")
    sector_map = panel.drop_duplicates("Symbol").set_index("Symbol")["Sector"]
    rets_21 = wide.pct_change(21).iloc[-1]
    sector_avg = rets_21.groupby(sector_map).mean()

    out = []
    for symbol, df in panel.groupby("Symbol"):
        if len(df) < 260:
            continue
        try:
            rec = recommend_stock(
                symbol,
                df.sort_values("Date"),
                predicted_return=forecasts.get(symbol),
                sector_return_21d=float(sector_avg.get(sector_map.get(symbol), np.nan))
                if symbol in sector_map.index else None,
                horizon=horizon,
            )
            rec["sector"] = sector_map.get(symbol, "Other")
            out.append(rec)
        except Exception as exc:
            logger.error("recommendation failed for %s: %s", symbol, exc)
    return sorted(out, key=lambda r: r["score"], reverse=True)
