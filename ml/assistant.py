"""BONUS — AI chat assistant over generated insights.

A deterministic intent-matching assistant grounded ONLY in artifacts the
platform itself computed (recommendations, forecasts, risk metrics, sector
rotation, portfolios). No external LLM, no live data — every answer is
traceable to a number the user can see elsewhere in the dashboard, which is
exactly what the competition's "no external sources" rule requires.
"""
from __future__ import annotations

import re
from typing import Callable

from ml.utils import get_logger

logger = get_logger(__name__)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


class InsightAssistant:
    """`context` is a provider object exposing the platform's computed data:

    symbols() -> list[str]
    recommendation(symbol) -> dict | None        (Module 9 output)
    recommendations() -> list[dict]
    forecast(symbol) -> dict                     (train_and_forecast output)
    risk(symbol) -> dict                         (stock_risk output)
    portfolio(profile) -> dict                   (build_portfolio output)
    sector_rotation() -> dict
    """

    def __init__(self, context):
        self.ctx = context
        # (pattern, handler) pairs — first match wins.
        self._intents: list[tuple[re.Pattern, Callable]] = [
            (re.compile(r"\b(buy|sell|hold|recommend)", re.I), self._recommendation),
            (re.compile(r"\b(forecast|predict|price target|expect)", re.I), self._forecast),
            (re.compile(r"\b(risk|volatil|drawdown|sharpe|var\b)", re.I), self._risk),
            (re.compile(r"\b(portfolio|allocat|invest|diversif)", re.I), self._portfolio),
            (re.compile(r"\b(sector|rotation|industry)", re.I), self._sector),
            (re.compile(r"\b(best|top|strongest|winners?)", re.I), self._top),
            (re.compile(r"\b(worst|avoid|weakest|losers?)", re.I), self._bottom),
        ]

    # ---------------------------------------------------------------- intents
    def _find_symbol(self, q: str) -> str | None:
        tokens = set(re.findall(r"[A-Za-z&-]+", q.upper()))
        for s in self.ctx.symbols():
            if s.upper() in tokens:
                return s
        return None

    def _recommendation(self, q: str, symbol: str | None) -> str:
        if symbol:
            rec = self.ctx.recommendation(symbol)
            if not rec:
                return f"I don't have a recommendation for {symbol} yet."
            return (f"{rec['reasoning']} Composite score {rec['score']:+.2f} "
                    f"(forecast {rec['components']['forecast']:+.2f}, momentum "
                    f"{rec['components']['momentum']:+.2f}, trend {rec['components']['trend']:+.2f}).")
        recs = self.ctx.recommendations()
        buys = [r["symbol"] for r in recs if r["action"] == "BUY"][:5]
        sells = [r["symbol"] for r in recs if r["action"] == "SELL"][:5]
        return (f"Current signals — BUY: {', '.join(buys) or 'none'}. "
                f"SELL: {', '.join(sells) or 'none'}. "
                "Ask about a specific stock for the full reasoning.")

    def _forecast(self, q: str, symbol: str | None) -> str:
        if not symbol:
            return "Tell me which stock to forecast, e.g. 'forecast RELIANCE'."
        f = self.ctx.forecast(symbol)
        da = f["metrics"].get("directional_accuracy")
        return (f"{symbol}: the {f['model']} model projects a {f['horizon']}-day return of "
                f"{_fmt_pct(f['predicted_return'])} (price ≈ ₹{f['predicted_price']:.2f} from "
                f"₹{f['last_close']:.2f}). Walk-forward directional accuracy: {da:.0%}. "
                "Treat this as a decision input, not a guarantee.")

    def _risk(self, q: str, symbol: str | None) -> str:
        if not symbol:
            return "Tell me which stock's risk to assess, e.g. 'how risky is TCS'."
        r = self.ctx.risk(symbol)
        return (f"{symbol} risk profile: annualized volatility {r['volatility']:.1%}, "
                f"Sharpe {r['sharpe']:.2f}, Sortino {r['sortino']:.2f}, max drawdown "
                f"{r['max_drawdown']:.1%}, 1-day VaR(95%) {r['var_95']:.2%}, "
                f"CVaR {r['cvar_95']:.2%}, beta {r.get('beta', 1.0):.2f}.")

    def _portfolio(self, q: str, symbol: str | None) -> str:
        profile = ("conservative" if re.search(r"conservativ|safe|low.?risk", q, re.I)
                   else "aggressive" if re.search(r"aggressiv|high.?risk|growth", q, re.I)
                   else "balanced")
        p = self.ctx.portfolio(profile)
        top = sorted(p["allocation"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        alloc = ", ".join(f"{s} {w:.1f}%" for s, w in top)
        return (f"{p['profile_label']} portfolio ({p['method'].replace('_', ' ')}): {alloc}"
                f"{' …' if len(p['allocation']) > 5 else ''}. Expected return "
                f"{p['expected_return']:.1%}, volatility {p['volatility']:.1%}, "
                f"Sharpe {p['sharpe']:.2f}.")

    def _sector(self, q: str, symbol: str | None) -> str:
        rot = self.ctx.sector_rotation()
        lead = ", ".join(f"{d['sector']} ({_fmt_pct(d['momentum'])})" for d in rot["leaders"])
        lag = ", ".join(f"{d['sector']} ({_fmt_pct(d['momentum'])})" for d in rot["laggards"])
        return (f"Sector rotation over the last {rot['window_days']} trading days — "
                f"leading: {lead}. Lagging: {lag}.")

    def _top(self, q: str, symbol: str | None) -> str:
        recs = self.ctx.recommendations()[:5]
        lines = "; ".join(f"{r['symbol']} ({r['score']:+.2f}, {r['action']})" for r in recs)
        return f"Highest-scoring stocks right now: {lines}."

    def _bottom(self, q: str, symbol: str | None) -> str:
        recs = self.ctx.recommendations()[-5:][::-1]
        lines = "; ".join(f"{r['symbol']} ({r['score']:+.2f}, {r['action']})" for r in recs)
        return f"Lowest-scoring stocks right now: {lines}."

    # ---------------------------------------------------------------- entry
    def answer(self, question: str) -> dict:
        symbol = self._find_symbol(question)
        for pattern, handler in self._intents:
            if pattern.search(question):
                try:
                    return {"answer": handler(question, symbol), "symbol": symbol,
                            "intent": handler.__name__.lstrip("_")}
                except Exception as exc:
                    logger.error("assistant intent failed: %s", exc)
                    return {"answer": f"I hit an error computing that: {exc}",
                            "symbol": symbol, "intent": "error"}
        if symbol:
            return {"answer": self._recommendation(question, symbol),
                    "symbol": symbol, "intent": "recommendation"}
        return {
            "answer": ("I can answer questions about the platform's own analysis: "
                       "recommendations ('should I buy INFY?'), forecasts "
                       "('forecast RELIANCE'), risk ('how risky is TCS?'), portfolios "
                       "('build me a conservative portfolio') and sector rotation "
                       "('which sectors are leading?')."),
            "symbol": None, "intent": "help",
        }
