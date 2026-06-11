# API Reference

Base URL: `http://localhost:8000` · Interactive docs: `/docs` (Swagger) and `/redoc`.

## Stocks
| Method | Path | Description |
|---|---|---|
| GET | `/stocks?sector=&search=` | Universe list with last close, 1y return/vol |
| GET | `/stocks/{ticker}?days=756` | OHLCV history + indicator series (MA, RSI, MACD, Bollinger) |
| GET | `/sectors` | Distinct sectors |

## Forecasting
| Method | Path | Description |
|---|---|---|
| POST | `/forecast` | `{symbol, horizon: 1\|5\|20, model, task}` → prediction + walk-forward metrics |
| GET | `/forecast/models` | Models available in this install |
| GET | `/forecast/compare/{ticker}?task=` | Model × horizon comparison table |

## Portfolio
| Method | Path | Description |
|---|---|---|
| POST | `/portfolio` | `{profile, method?, symbols?, lookback_days}` → allocation %, expected return, vol, Sharpe |
| GET | `/portfolio/profiles` | Profile definitions |
| GET | `/portfolio/frontier` | Risk/return cloud for the frontier chart |
| POST | `/portfolio/questionnaire` | 5-question risk quiz → recommended profile |

## Risk
| Method | Path | Description |
|---|---|---|
| GET | `/risk` | Full-universe stock-level risk table |
| GET | `/risk/{ticker}?lookback_days=` | One stock's metrics (vol, Sharpe, Sortino, Calmar, max DD, VaR, CVaR, alpha, beta) |
| POST | `/risk/portfolio` | `{weights}` → portfolio-level metrics + diversification ratio |

## Anomalies
| Method | Path | Description |
|---|---|---|
| GET | `/anomalies?symbol=&limit=` | Ensemble-detected anomalies with type labels |

## Intelligence
| Method | Path | Description |
|---|---|---|
| GET | `/recommendations?with_forecasts=` | BUY/HOLD/SELL for the universe with reasoning |
| GET | `/recommendations/{ticker}` | One stock's recommendation |
| POST | `/explainability` | `{symbol, horizon, model}` → SHAP/importance contributions + confidence |
| POST | `/chat` | `{question}` → grounded assistant answer |

## Analytics (bonus)
| Method | Path | Description |
|---|---|---|
| GET | `/analytics/summary` | Dashboard summary (signals, movers, sector leaders) |
| GET | `/analytics/sector-rotation` | Rolling sector momentum ranks |
| GET | `/analytics/network?threshold=` | Correlation network nodes/edges |
| POST | `/analytics/backtest` | `{weights, rebalance_days, transaction_cost_bps}` → equity curve + metrics |
| POST | `/analytics/montecarlo` | `{symbol, horizon_days, n_sims, method}` → percentile path fan |
| POST | `/analytics/scenario` | `{weights, market_shock, sector_shocks}` → stress-test impact |

## Reports & meta
| Method | Path | Description |
|---|---|---|
| POST | `/reports/generate?focus_symbol=&profile=` | Multi-page PDF download |
| GET | `/health` | Liveness probe |

Errors: `404` unknown symbol, `422` validation/insufficient-data, with `{"detail": "..."}` bodies.
