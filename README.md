# NIFTY-50 Investment Intelligence Platform

AI-powered decision-support system for the **"Data-Driven Investment Intelligence Using NIFTY-50 Market Data"** challenge. The platform transforms historical NIFTY-50 OHLCV data into actionable, *explainable* investment insights — forecasting, portfolio construction, risk analytics, anomaly detection and natural-language recommendations — with a FastAPI backend and a Next.js 15 dashboard.

> Uses **only** the supplied historical dataset. No live data, no external APIs, no sentiment feeds. Forecasts are walk-forward validated; nothing here is investment advice.

## Quick start (one command)

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- API + Swagger docs: http://localhost:8000/docs

First boot generates a reproducible **synthetic demo dataset** automatically if the Kaggle data isn't present, so everything works out of the box.

## Quick start (local dev)

```bash
# Backend (Python 3.11)
python -m venv .venv && .venv\Scripts\activate     # Windows; use source .venv/bin/activate on Unix
pip install -r requirements.txt                    # + requirements-deep.txt for LSTM/Transformer
python scripts/download_data.py                    # optional: real Kaggle data into data/raw
uvicorn backend.main:app --reload                  # http://localhost:8000

# Frontend
cd frontend && npm install && npm run dev          # http://localhost:3000

# Tests
pytest
```

> **Warm the cache for an instant dashboard.** The first forecast/recommendation/PDF-report call trains models on demand — leakage-safe walk-forward validation means ~10s per stock, so the full universe takes a few minutes cold. Run `python scripts/train_all.py` once after setup (or hit `/recommendations?with_forecasts=true`) to warm every cache the API uses; the dashboard is instant for the rest of the session. This is expected behavior, not a hang.

## Using the real dataset

Download [rohanrao/nifty50-stock-market-data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data) from Kaggle (or run `python scripts/download_data.py` with kagglehub installed) and place the per-stock CSVs + `stock_metadata.csv` in `data/raw/`. Set `ALLOW_SYNTHETIC_DATA=false` to forbid the synthetic fallback. The ingestion pipeline (`python -m backend.data_loader`) validates schema, cleans, normalizes dates and writes parquet artifacts to `data/processed/`.

## What's inside

| Capability | Where | Highlights |
|---|---|---|
| Data ingestion | `backend/data_loader.py` | schema validation, dedup, gap repair, parquet outputs |
| Feature engineering | `ml/features.py` | MA/EMA, RSI, MACD, Bollinger, ATR, momentum, ROC, returns, rolling vol, volume, sector metrics |
| EDA | `ml/eda.py` | price/volume/sector/correlation/distribution/volatility/drawdown charts |
| Forecasting | `ml/models/`, `ml/training/`, `ml/evaluation/` | Linear/RF/XGBoost/LightGBM/LSTM/Transformer; 1/5/20-day horizons; time-series CV + walk-forward; full metric suite |
| Portfolios | `ml/portfolio/` | equal-weight, risk parity, Markowitz, max-Sharpe, min-vol; Conservative/Balanced/Aggressive profiles |
| Risk engine | `ml/risk/` | vol, Sharpe, Sortino, Calmar, max DD, VaR, CVaR, alpha, beta — stock & portfolio level |
| Anomaly detection | `ml/anomaly/` | IsolationForest + DBSCAN + Autoencoder ensemble with typed anomaly reports |
| Explainability | `ml/explainability/` | SHAP, LIME, feature importance, per-prediction confidence |
| Recommendations | `ml/recommendation/` | BUY/HOLD/SELL with natural-language reasoning over 5 weighted signals |
| API | `backend/` | `/stocks`, `/forecast`, `/portfolio`, `/risk`, `/anomalies`, `/recommendations`, `/explainability` + bonus endpoints |
| Dashboard | `frontend/` | Next.js 15 + TypeScript + Tailwind + Recharts; dark mode; 8 pages |
| Reports | `ml/reports.py`, `scripts/build_report.py` | downloadable analytics PDF + the written **[Technical Report](reports/generated/TECHNICAL_REPORT.pdf)** (`docs/TECHNICAL_REPORT.md`) |
| Bonus | `ml/analytics.py`, `ml/backtest.py`, `ml/simulation.py`, `ml/assistant.py` | sector rotation, correlation network, backtesting, Monte-Carlo & scenario simulation, risk questionnaire, insight chat assistant |

## Repository layout

```
investment-intelligence/
├── backend/          FastAPI app: data loader, services, routers, schemas
├── ml/               feature/modeling/portfolio/risk/anomaly/XAI engines
├── frontend/         Next.js 15 dashboard
├── data/             raw/ (input CSVs) and processed/ (parquet artifacts)
├── scripts/          download_data.py, train_all.py
├── tests/            pytest unit + API integration suite
├── docs/             architecture, API, models, install guides
├── reports/          generated PDF reports & EDA charts
├── docker/           backend + frontend Dockerfiles
└── docker-compose.yml
```

## Reproducing results

Re-create every result artifact and report figure from scratch:

```bash
# 1. Environment + dependencies (see "Quick start" above)
pip install -r requirements.txt

# 2. Use the real dataset so numbers match the report (not the synthetic fallback)
python scripts/download_data.py            # or place the Kaggle CSVs in data/raw/ manually
export ALLOW_SYNTHETIC_DATA=false          # Windows: set ALLOW_SYNTHETIC_DATA=false

# 3. Ingest → clean → parquet
python -m backend.data_loader              # writes data/processed/

# 4. Train, evaluate, score and render the report
python scripts/train_all.py                # ~10 min for the full 50-stock universe
```

This writes the result artifacts to `ml/artifacts/`:

| Artifact | Contents |
|---|---|
| `model_comparison.csv` | every stock × model × horizon: RMSE, MAE, MAPE, R², directional accuracy (out-of-fold) |
| `risk_table.csv` | vol, Sharpe, Sortino, Calmar, max drawdown, VaR, CVaR, alpha, beta per stock |
| `recommendations.json` | BUY/HOLD/SELL + component scores + reasoning |
| `*_ret20.joblib` | the trained per-stock forecasters the API serves |

and a timestamped PDF to `reports/generated/`. Because every stochastic step is seeded
(`random_state=42`, seeded synthetic generator), a clean run reproduces the headline figures
in the technical report — e.g. **20-day best directional accuracy ≈ 0.64**, **mean Sharpe ≈ 0.27**,
**23 BUY / 20 HOLD / 6 SELL** (the recommendation split is the full five-signal engine; the
dashboard's fast default view omits the forecast signal — see [docs/MODELS.md](docs/MODELS.md)).

> **Why it's trustworthy.** Evaluation is **leakage-safe**: expanding-window walk-forward splits
> with a gap equal to the forecast horizon, so overlapping forward-return targets never cross the
> train/test boundary, and all reported metrics are out-of-fold only.

See `docs/` for the architecture diagram, API reference, model documentation and detailed install instructions.
