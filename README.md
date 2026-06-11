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
| Reports | `ml/reports.py` | downloadable multi-page PDF |
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

## Reproducibility

- All randomness is seeded (`random_state=42` everywhere, seeded synthetic generator).
- `python scripts/train_all.py` re-runs the entire offline pipeline (ingest → EDA → model comparison → forecasts → recommendations → PDF report) and writes artifacts to `ml/artifacts/`.
- Evaluation is **leakage-safe**: expanding-window splits with a gap equal to the forecast horizon so overlapping forward targets never cross the train/test boundary.

See `docs/` for the architecture diagram, API reference, model documentation and detailed install instructions.
