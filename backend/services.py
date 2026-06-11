"""MODULE 10 — service layer.

A single lazily-initialized Platform object owns the processed data and every
analytics engine, with in-memory caching so repeated dashboard calls are
instant. The FastAPI routers stay thin: parse request -> call Platform ->
shape response.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np
import pandas as pd

from backend.data_loader import load_processed
from ml.analytics import correlation_network, sector_rotation
from ml.anomaly.detectors import detect_anomalies
from ml.assistant import InsightAssistant
from ml.backtest import backtest_portfolio
from ml.evaluation.compare import compare_models
from ml.explainability.explain import explain_prediction, feature_importances
from ml.features import build_feature_matrix
from ml.models.registry import available_models
from ml.portfolio.profiles import INVESTOR_PROFILES, build_portfolio, efficient_frontier
from ml.recommendation.engine import recommend_all, recommend_stock
from ml.risk.metrics import market_returns, portfolio_risk, stock_risk
from ml.simulation import monte_carlo_forecast, scenario_simulate
from ml.training.trainer import _make_model, prepare_xy, train_and_forecast
from ml.utils import get_logger

logger = get_logger(__name__)

# Fast default for interactive API calls; heavier models remain selectable.
DEFAULT_MODEL = "random_forest"


class Platform:
    def __init__(self):
        # RLock: cache builders re-enter the lock (e.g. stock_list -> data,
        # recommendations -> forecast), a plain Lock would self-deadlock.
        self._lock = threading.RLock()
        self._data: dict | None = None
        self._cache: dict[tuple, object] = {}

    # ------------------------------------------------------------- data
    @property
    def data(self) -> dict:
        if self._data is None:
            with self._lock:
                if self._data is None:
                    logger.info("loading processed data (runs ingestion if needed)…")
                    self._data = load_processed()
        return self._data

    @property
    def panel(self) -> pd.DataFrame:
        return self.data["panel"]

    @property
    def close_wide(self) -> pd.DataFrame:
        return self.data["close_wide"]

    @property
    def metadata(self) -> pd.DataFrame:
        return self.data["metadata"]

    def symbols(self) -> list[str]:
        return sorted(self.panel["Symbol"].unique())

    def stock_frame(self, symbol: str) -> pd.DataFrame:
        df = self.panel[self.panel["Symbol"] == symbol.upper()].sort_values("Date")
        if df.empty:
            raise KeyError(f"unknown symbol '{symbol}'")
        return df.reset_index(drop=True)

    def _memo(self, key: tuple, fn):
        if key not in self._cache:
            with self._lock:
                if key not in self._cache:
                    self._cache[key] = fn()
        return self._cache[key]

    # ------------------------------------------------------------- stocks
    def stock_list(self) -> list[dict]:
        def build():
            meta = self.metadata.set_index("Symbol")
            rows = []
            for symbol in self.symbols():
                close = self.close_wide[symbol].dropna()
                rets = close.pct_change().tail(252)
                rows.append({
                    "symbol": symbol,
                    "company": str(meta["Company"].get(symbol, symbol.title())),
                    "sector": str(meta["Sector"].get(symbol, "Other")),
                    "last_close": round(float(close.iloc[-1]), 2),
                    "return_1y": round(float(close.iloc[-1] / close.iloc[-min(252, len(close))] - 1), 4),
                    "volatility_1y": round(float(rets.std() * np.sqrt(252)), 4),
                })
            return rows
        return self._memo(("stock_list",), build)

    def stock_detail(self, symbol: str, days: int = 756) -> dict:
        df = self.stock_frame(symbol).tail(days)
        feats = build_feature_matrix(self.stock_frame(symbol)).tail(days)
        meta = self.metadata.set_index("Symbol")
        indicator_cols = ["ma20", "ma50", "ma200", "rsi14", "macd", "macd_signal",
                          "bb_upper", "bb_lower", "vol_21d"]
        return {
            "symbol": symbol.upper(),
            "company": str(meta["Company"].get(symbol.upper(), symbol.title())),
            "sector": str(meta["Sector"].get(symbol.upper(), "Other")),
            "start": str(df["Date"].iloc[0].date()),
            "end": str(df["Date"].iloc[-1].date()),
            "prices": [
                {"date": str(d.date()), "open": float(o), "high": float(h),
                 "low": float(lo), "close": float(c), "volume": float(v)}
                for d, o, h, lo, c, v in zip(df["Date"], df["Open"], df["High"],
                                             df["Low"], df["Close"], df["Volume"])
            ],
            "indicators": {
                c: [None if pd.isna(x) else round(float(x), 4) for x in feats[c]]
                for c in indicator_cols if c in feats
            },
        }

    # ------------------------------------------------------------- forecast
    def forecast(self, symbol: str, horizon: int = 20, model: str = DEFAULT_MODEL,
                 task: str = "regression") -> dict:
        key = ("forecast", symbol.upper(), horizon, model, task)
        return self._memo(key, lambda: {
            "symbol": symbol.upper(),
            **train_and_forecast(self.stock_frame(symbol), model, horizon, task),
        })

    def model_comparison(self, symbol: str, task: str = "regression",
                         horizons: tuple[int, ...] = (1, 5, 20)) -> list[dict]:
        key = ("compare", symbol.upper(), task, horizons)

        def build():
            table = compare_models(self.stock_frame(symbol), task=task, horizons=horizons)
            return [
                {"model": r.pop("model"), "horizon": int(r.pop("horizon")),
                 "task": r.pop("task"),
                 "metrics": {k: (None if pd.isna(v) else round(float(v), 4))
                             for k, v in r.items()}}
                for r in table.to_dict("records")
            ]
        return self._memo(key, build)

    def available_models(self) -> dict:
        return available_models()

    # ------------------------------------------------------------- portfolio
    def portfolio(self, profile: str = "balanced", method: str | None = None,
                  symbols: list[str] | None = None, lookback_days: int = 756) -> dict:
        key = ("portfolio", profile, method, tuple(symbols or ()), lookback_days)
        return self._memo(key, lambda: build_portfolio(
            self.close_wide, profile, method, lookback_days, symbols))

    def frontier(self) -> list[dict]:
        return self._memo(("frontier",), lambda: efficient_frontier(self.close_wide))

    def profiles(self) -> dict:
        return INVESTOR_PROFILES

    @staticmethod
    def questionnaire_profile(answers: dict) -> dict:
        horizon_pts = 1 if answers["horizon_years"] < 3 else 2 if answers["horizon_years"] < 7 else 4
        score = (horizon_pts + answers["loss_tolerance"] + answers["experience"]
                 + answers["income_stability"] + answers["goal"])
        profile = "conservative" if score <= 9 else "balanced" if score <= 14 else "aggressive"
        return {
            "profile": profile,
            "score": int(score),
            "rationale": (
                f"Scored {score}/20 across horizon, loss tolerance, experience, "
                f"income stability and goals — mapping to the "
                f"{INVESTOR_PROFILES[profile]['label']} profile: "
                f"{INVESTOR_PROFILES[profile]['description']}"
            ),
        }

    # ------------------------------------------------------------- risk
    def market(self) -> pd.Series:
        return self._memo(("market",), lambda: market_returns(self.close_wide))

    def stock_risk(self, symbol: str, lookback_days: int | None = None) -> dict:
        key = ("risk", symbol.upper(), lookback_days)
        return self._memo(key, lambda: {
            k: round(v, 4) for k, v in stock_risk(
                self.close_wide[symbol.upper()].dropna(), self.market(), lookback_days
            ).items()
        })

    # Alias used by the chat assistant's context protocol.
    def risk(self, symbol: str) -> dict:
        return self.stock_risk(symbol)

    def portfolio_risk(self, weights: dict[str, float]) -> dict:
        rets = self.close_wide.pct_change()
        return {k: round(v, 4)
                for k, v in portfolio_risk(rets, weights, self.market()).items()}

    def risk_table(self) -> list[dict]:
        def build():
            return [{"symbol": s, **self.stock_risk(s)} for s in self.symbols()]
        return self._memo(("risk_table",), build)

    # ------------------------------------------------------------- anomalies
    def anomalies(self, symbol: str | None = None, limit: int = 50) -> list[dict]:
        def one(sym: str) -> list[dict]:
            key = ("anomalies", sym)

            def build():
                report = detect_anomalies(self.stock_frame(sym), sym)
                if report.empty:
                    return []
                return [
                    {"date": str(r["Date"].date()), "symbol": r["Symbol"],
                     "close": round(float(r["Close"]), 2),
                     "ret_1d": round(float(r["ret_1d"]), 4),
                     "volume_ratio": round(float(r["volume_ratio"]), 2),
                     "drawdown": round(float(r["drawdown"]), 4),
                     "votes": int(r["votes"]), "methods": r["methods"], "type": r["type"]}
                    for _, r in report.iterrows()
                ]
            return self._memo(key, build)

        if symbol:
            return one(symbol.upper())[:limit]
        out: list[dict] = []
        for sym in self.symbols():
            out.extend(one(sym)[:5])
        out.sort(key=lambda r: r["date"], reverse=True)
        return out[:limit]

    # ------------------------------------------------------------- intelligence
    def recommendations(self, with_forecasts: bool = False) -> list[dict]:
        """with_forecasts=True trains a per-symbol forecaster (slow on first
        call, cached after); False scores on technical/risk signals only."""
        key = ("recommendations", with_forecasts)

        def build():
            forecasts = {}
            if with_forecasts:
                for sym in self.symbols():
                    try:
                        forecasts[sym] = self.forecast(sym)["predicted_return"]
                    except Exception as exc:
                        logger.warning("forecast failed for %s: %s", sym, exc)
            return recommend_all(self.panel, forecasts)
        return self._memo(key, build)

    def recommendation(self, symbol: str) -> dict | None:
        for rec in self.recommendations():
            if rec["symbol"] == symbol.upper():
                return rec
        return None

    def explain(self, symbol: str, horizon: int = 20, model_name: str = DEFAULT_MODEL,
                task: str = "regression") -> dict:
        key = ("explain", symbol.upper(), horizon, model_name, task)

        def build():
            df = self.stock_frame(symbol)
            X, y, _ = prepare_xy(df, horizon, task)
            model = _make_model(model_name, task)
            model.fit(X, y)
            feats = build_feature_matrix(df)
            latest = feats[X.columns].dropna().iloc[[-1]]
            oos = self.forecast(symbol, horizon, model_name, task)["metrics"].get(
                "directional_accuracy")
            result = explain_prediction(model, X, y, latest, task, oos_metric=oos)
            importances = feature_importances(model, X, y).head(15)
            return {
                "symbol": symbol.upper(), "model": model_name, "horizon": horizon,
                **result,
                "global_importances": {k: round(float(v), 4) for k, v in importances.items()},
            }
        return self._memo(key, build)

    # ------------------------------------------------------------- bonus
    def sector_rotation(self) -> dict:
        return self._memo(("sector_rotation",), lambda: sector_rotation(self.panel))

    def network(self, threshold: float = 0.5) -> dict:
        return self._memo(("network", threshold),
                          lambda: correlation_network(self.panel, threshold))

    def backtest(self, **kwargs) -> dict:
        return backtest_portfolio(self.close_wide, **kwargs)

    def simulate(self, symbol: str, **kwargs) -> dict:
        return {"symbol": symbol.upper(),
                **monte_carlo_forecast(self.close_wide[symbol.upper()].dropna(), **kwargs)}

    def scenario(self, weights: dict[str, float], **kwargs) -> dict:
        sectors = self.metadata.set_index("Symbol")["Sector"]
        return scenario_simulate(self.close_wide, weights, sectors, **kwargs)

    def chat(self, question: str) -> dict:
        return InsightAssistant(self).answer(question)

    def dashboard_summary(self) -> dict:
        recs = self.recommendations(with_forecasts=False)
        rets = self.close_wide.pct_change(21).iloc[-1].dropna().sort_values()
        rot = self.sector_rotation()
        return {
            "n_stocks": len(self.symbols()),
            "date_range": {"start": str(self.panel["Date"].min().date()),
                           "end": str(self.panel["Date"].max().date())},
            "buy_count": sum(1 for r in recs if r["action"] == "BUY"),
            "sell_count": sum(1 for r in recs if r["action"] == "SELL"),
            "top_movers_21d": [{"symbol": s, "return": round(float(v), 4)}
                               for s, v in rets.tail(5)[::-1].items()],
            "bottom_movers_21d": [{"symbol": s, "return": round(float(v), 4)}
                                  for s, v in rets.head(5).items()],
            "leading_sectors": rot["leaders"],
            "lagging_sectors": rot["laggards"],
        }


@lru_cache(maxsize=1)
def get_platform() -> Platform:
    return Platform()
