# NIFTY-50 Investment Intelligence — Complete Explanation

This document explains **everything** in the project: what every dashboard screen shows and how to read it, and what every backend / ML / data component does under the hood. It is written so a non-developer can follow the dashboard sections and a developer can follow the internals.

> **Golden rule of the whole project:** it uses **only the supplied historical NIFTY-50 price/volume data**. No live prices, no news, no sentiment, no external APIs. Every number on screen is *computed* from that history. Forecasts are validated honestly (walk-forward), and nothing here is investment advice.

---

## Part 1 — The big picture

### What the platform is
A full-stack decision-support tool that turns raw historical stock data (Open/High/Low/Close/Volume per day, per stock) into:

- **Forecasts** — where a stock's return may go over the next 1, 5, or 20 days.
- **Portfolios** — how to split money across stocks for a chosen risk appetite.
- **Risk analytics** — how dangerous each stock is (volatility, drawdowns, worst-case loss).
- **Anomaly detection** — which days were statistically "weird" (crashes, volume surges).
- **Recommendations** — a plain-English BUY / HOLD / SELL call with the reasons.
- **Explainability** — *why* the model said what it said (which indicators drove it).

### The three layers
```
Browser  →  Frontend (Next.js dashboard)  →  Backend API (FastAPI)  →  ML engines (Python)  →  Data
```

1. **Frontend** (`frontend/`) — what you see and click. Next.js 15 + TypeScript + Tailwind + Recharts. 8 pages, dark/light mode.
2. **Backend** (`backend/`) — a web API. Thin endpoints validate your request and hand it to the `Platform` service, which calls the ML engines and **caches** the result so the second click is instant.
3. **ML engines** (`ml/`) — the actual math: indicators, models, portfolio optimizers, risk formulas, anomaly detectors, explainers.
4. **Data** (`data/`) — `raw/` holds input CSVs; `processed/` holds cleaned `.parquet` files. If no real data is present, a **synthetic generator** creates realistic demo data so everything works instantly.

### How a click becomes a number (request lifecycle)
Example: you open the **Risk** page.
1. The browser calls `GET /risk`.
2. FastAPI routes it to `backend/routers/risk.py`.
3. That router asks `Platform.stock_risk(...)` (in `backend/services.py`).
4. The Platform loads the price data once, computes risk metrics via `ml/risk/metrics.py`, and **memoizes** the answer.
5. JSON comes back, Recharts/tables render it.
6. Click again → served from cache, instantly.

---

## Part 2 — The dashboard, screen by screen

The left **sidebar** (`components/sidebar.tsx`) lists all 8 pages, shows the active one, and has a sun/moon **dark-mode toggle**. The footer reads "Historical data only" as a constant reminder of the data policy. Every page fetches from the backend with the tiny `useApi`/`useApiPost` hooks (`lib/api.ts`); while loading you see gray **skeleton** placeholders, and if the backend is down you see a red **ErrorNote** telling you how to start it.

---

### 1. Dashboard (`/` — `app/page.tsx`)
The market-overview landing page. Data source: `GET /analytics/summary` and `GET /recommendations`.

**Four stat cards (top row):**
- **Universe** — number of stocks being analyzed.
- **Buy Signals** — how many stocks have a composite score ≥ +0.20.
- **Sell Signals** — how many have a composite score ≤ −0.20.
- **Leading Sector** — the sector with the strongest recent momentum (and its momentum %).

**21-Day Movers** (bar chart) — the best and worst performers over the last ~month (21 trading days). Green bars up, red bars down. A quick read of "what's hot, what's not."

**Sector Rotation** (list) — sectors ranked by **63-day momentum** (about a quarter). Green = leaders, red = laggards. "Details →" links to AI Insights. This is the classic "which industries are money flowing into" view.

**Top AI Recommendations** (table) — the 8 highest-conviction calls: Symbol, Sector, Action badge (BUY/HOLD/SELL), Score, and a one-line **reasoning** sentence. Each symbol links to its page in Stock Explorer.

> How to read it: the cards are the market's pulse, movers/sectors show momentum, and the table is the "if you only read one thing" shortlist.

---

### 2. Stock Explorer (`/stocks` — `app/stocks/page.tsx`)
Inspect any single stock. Data: `GET /stocks`, `GET /sectors`, `GET /stocks/{symbol}`.

- **Search + sector filter** narrow the scrollable stock list on the left. Each list item shows symbol, sector, and 1-year return (green/red).
- **Three stat cards:** last close price (₹), 1-year return, 1-year annualized volatility.
- **Price chart** (last 3 trading years) overlays:
  - **Close** (blue) — the actual price.
  - **MA50 / MA200** (orange/purple) — 50- and 200-day moving averages; their crossover is a classic trend signal (MA50 above MA200 = uptrend / "golden cross").
  - **Bollinger Bands** (gray upper/lower) — volatility envelope; price near the upper band = stretched high, near lower = stretched low.
- **RSI(14) chart** — Relative Strength Index, 0–100. Above 70 = overbought (may pull back); below 30 = oversold (may bounce).

> How to read it: price vs. its moving averages tells you the trend; Bollinger width tells you how volatile it is right now; RSI tells you if it's overextended.

---

### 3. Forecasting (`/forecasting` — `app/forecasting/page.tsx`)
Predict a stock's forward return and visualize the cone of outcomes. Data: `POST /forecast`, `POST /analytics/montecarlo`, `GET /forecast/compare/{symbol}`, `GET /forecast/models`.

**Controls:** pick a **stock**, a **horizon** (1, 5, or 20 days), and a **model** (random_forest by default; XGBoost/LightGBM/LSTM/Transformer if installed). Click **Run Forecast**. ("Training…" appears because it genuinely trains/validates the model on the fly.)

**Four result cards:**
- **Last close** — current price and the date.
- **Predicted Nd return** — the model's forecast %, and the implied target price.
- **Directional accuracy** — *out-of-sample, walk-forward*: how often the model got the up/down direction right on data it never trained on. This is the honesty metric — treat it as the trust level.
- **RMSE / MAE / R²** — error magnitudes (lower RMSE/MAE = better; R² closer to 1 = better fit).

**Monte-Carlo Simulation** (band chart, 126 days, 2000 paths, bootstrap) — instead of one number, it simulates thousands of possible price paths by resampling historical daily returns, then shows percentile bands (p5–p95). The subtitle gives **P(gain)**, **terminal VaR-95** (a worst-case-ish loss), and the **expected terminal price**. The widening fan = growing uncertainty over time.

**Model Comparison** (table) — every available model × every horizon, ranked by walk-forward RMSE, with MAE, MAPE, R², and directional accuracy. This proves no single model is cherry-picked — you can see which actually wins.

> How to read it: the predicted return is the headline; directional accuracy is whether to believe it; the Monte-Carlo fan is the realistic range of outcomes.

---

### 4. Portfolio Builder (`/portfolio` — `app/portfolio/page.tsx`)
Turn a risk appetite into an actual allocation, then test it. Data: `POST /portfolio/questionnaire`, `POST /portfolio`, `POST /analytics/backtest`.

**Risk Questionnaire** (left) — 5 questions (horizon, loss tolerance, experience, income stability, goal). "Find My Profile" scores your answers into **Conservative / Balanced / Aggressive** and auto-builds the matching portfolio, with a one-line rationale.

**Build Portfolio** (left) — or pick the profile and optimization **method** manually:
- *equal_weight* — same amount in each.
- *risk_parity* — each stock contributes equal *risk*.
- *mean_variance* (Markowitz) — classic risk/return trade-off.
- *max_sharpe* — best risk-adjusted return.
- *min_volatility* — smoothest ride.
- Blank = the profile's sensible default.

**Results (right):**
- **Three stat cards:** expected annual return (a *shrunk*, conservative estimate), volatility, and Sharpe ratio (risk-free rate assumed 6%).
- **Allocation pie** — how the money is split across stocks.
- **Backtest** button → simulates actually holding that portfolio with **monthly rebalancing** and **10 bps transaction costs**, starting from ₹10,00,000. The chart plots your **portfolio vs. an equal-weight benchmark**, and the subtitle gives final value, total return, CAGR, and max drawdown.

> How to read it: the profile sets the strategy, the pie is the recipe, and the backtest is the "would this actually have worked?" reality check against a dumb equal-weight baseline.

---

### 5. Risk Analytics (`/risk` — `app/risk/page.tsx`)
A sortable risk league table plus a stress tester. Data: `GET /risk`, `POST /analytics/scenario`.

**Scenario Simulator** — pick an instantaneous **market shock** (−30% … +10%) and run a stress test on an equal-weight portfolio of the whole universe. Each stock is hit in proportion to its **beta** (sensitivity to the market). You get an estimated portfolio impact plus an uncertainty **band**.

**Universe Risk Table** — every stock, every metric, sortable by any column:
- **Ann. Return** — annualized return.
- **Volatility** — annualized standard deviation (how bumpy).
- **Sharpe** — return per unit of total risk.
- **Sortino** — like Sharpe but only penalizes *downside* risk.
- **Calmar** — return divided by worst drawdown.
- **Max DD** — largest peak-to-trough fall.
- **VaR 95 / CVaR 95** — Value-at-Risk and Conditional VaR: a bad-day loss threshold and the average loss *beyond* that threshold.
- **Alpha / Beta** — performance vs. the market proxy (equal-weight universe) and sensitivity to it. Beta > 1 = moves more than the market.

> How to read it: sort by Sharpe/Sortino to find the best risk-adjusted names; sort by Max DD/VaR to find the scariest ones; use the stress test to preview a crash.

---

### 6. Anomaly Detection (`/anomalies` — `app/anomalies/page.tsx`)
Flags statistically unusual trading days. Data: `GET /anomalies`.

Three unsupervised detectors vote — **Isolation Forest + DBSCAN + Autoencoder** — and a day is flagged only when **≥2 of the 3 agree** (the "Votes x/3" column). Filter by a single stock or view the top anomalies across all stocks.

Each row: Date, Symbol, **Type** badge, 1-day return, volume multiple, drawdown, vote count, and which detectors fired. Types:
- **Volatility Spike** — abnormally wild swing.
- **Unusual Return** — return far outside its normal range.
- **Volume Surge** — trading volume far above its average.
- **Extreme Drawdown** — a deep fall from a recent peak.

> How to read it: high votes + extreme drawdown = a genuine stress event worth investigating; a lone volume surge may just be index rebalancing or news.

---

### 7. AI Insights (`/insights` — `app/insights/page.tsx`)
The "why" page — explainability and a chat assistant. Data: `GET /recommendations`, `POST /explainability`, `POST /chat`.

**Recommendations list** — every stock's BUY/HOLD/SELL with score and reasoning. Click one to open its explanation.

**"Why? — {symbol}"** panel:
- **Signal components** bar chart — the 5 weighted signals behind the call (forecast, momentum, trend, risk-adjusted, sector-relative), each shown positive/negative.
- **Top feature contributions** bar chart — **SHAP** (or LIME / permutation fallback) attribution: which technical indicators pushed the model's forecast up or down. The subtitle shows the attribution method, the 20-day predicted return, and a **confidence** score (derived from the model's out-of-sample directional accuracy).

**Insight Assistant** (right, chat) — ask in plain English ("should I buy INFY?", "how risky is TCS?", "build me a conservative portfolio"). It's a deterministic, intent-matching assistant **grounded only in this platform's computed numbers** — no external LLM, so it never makes things up.

> How to read it: the component chart says *what kind* of signal drove the call; the SHAP chart says *which indicators* did; the assistant lets you interrogate it conversationally.

---

### 8. Settings (`/settings` — `app/settings/page.tsx`)
- **Appearance** — theme: Dark / Light / System.
- **Backend** — shows the API URL, a live online/offline health indicator (`GET /health`), and the list of available regressor/classifier models (depends on which optional libraries are installed).
- **Reports** — **Export PDF Report**: generates a full multi-page PDF (EDA charts, forecasts, portfolio, risk, recommendations) via `POST /reports/generate` and downloads it. Can take a minute because it computes everything fresh.

---

### Shared frontend building blocks
- `components/ui.tsx` — reusable Card, StatCard, Table, Badge, Button, Input, Select, Skeleton, ErrorNote.
- `components/charts.tsx` — Recharts wrappers: `PriceChart` (multi-line), `SimpleBars` (sign-colored bars), `BandChart` (Monte-Carlo percentile fan), `AllocationPie`.
- `lib/api.ts` — `apiGet`/`apiPost` + the `useApi`/`useApiPost` hooks; base URL from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).
- `lib/utils.ts` — formatters: `fmtPct` (percentages), `fmtInr` (₹ currency), `cn` (class merging).

---

## Part 3 — The backend API

`backend/main.py` builds the FastAPI app, enables CORS (so the browser on port 3000 can call the API on port 8000), mounts all routers, and exposes `GET /` (metadata) and `GET /health`. Interactive Swagger docs live at `http://localhost:8000/docs`.

### The Platform service (`backend/services.py`)
The heart of the backend. A single `Platform` object that:
- **Loads data once** (lazy) — the long price panel, the wide close-price matrix, and stock metadata.
- **Memoizes everything** — results are cached keyed by their parameters, behind a re-entrant lock (so the dashboard is slow only on the *first* computation of each thing). `scripts/train_all.py` can pre-warm these caches offline.
- **Orchestrates the ML engines** — every page's data ultimately comes from a Platform method: `stock_list`, `stock_detail`, `forecast`, `model_comparison`, `portfolio`, `frontier`, `stock_risk`, `portfolio_risk`, `anomalies`, `recommendations`, `explain`, `sector_rotation`, `network`, `backtest`, `simulate`, `scenario`, `chat`, `dashboard_summary`.

### Endpoints by router (`backend/routers/`)
| Router | Endpoints | Purpose |
|---|---|---|
| `stocks.py` | `GET /stocks`, `GET /stocks/{ticker}`, `GET /sectors` | Universe list, per-stock OHLCV+indicators, sector list |
| `forecast.py` | `POST /forecast`, `GET /forecast/models`, `GET /forecast/compare/{ticker}` | Run a forecast, list models, full model×horizon table |
| `portfolio.py` | `POST /portfolio`, `GET /portfolio/profiles`, `GET /portfolio/frontier`, `POST /portfolio/questionnaire` | Build allocation, list profiles, efficient frontier, risk quiz |
| `risk.py` | `GET /risk`, `GET /risk/{ticker}`, `POST /risk/portfolio` | Universe risk table, single-stock risk, custom-portfolio risk |
| `anomalies.py` | `GET /anomalies` | Detected anomalies (optionally per symbol) |
| `recommendations.py` | `GET /recommendations`, `GET /recommendations/{ticker}` | BUY/HOLD/SELL for all or one |
| `explainability.py` | `POST /explainability` | SHAP/LIME attribution + confidence |
| `analytics.py` | `GET /analytics/summary`, `/sector-rotation`, `/network`, `POST /backtest`, `/montecarlo`, `/scenario` | Dashboard KPIs + bonus analytics |
| `chat.py` | `POST /chat` | Insight assistant |
| `reports.py` | `POST /reports/generate` | Downloadable PDF |

`backend/schemas.py` holds the Pydantic request/response models (validation + the shapes you see in Swagger).

### Data pipeline (`backend/data_loader.py`)
Turns messy CSVs into clean analytics-ready files:
1. **Discover** CSVs in `data/raw/`.
2. **Validate** schema (Date, Open, High, Low, Close, Volume).
3. **Clean** per stock — parse/sort dates, drop duplicates, coerce numerics, forward-fill small price gaps, zero-fill volume gaps, drop impossible rows (Close ≤ 0, High < Low), require ≥260 rows (~1 year).
4. **Attach metadata** — company + sector (from `stock_metadata.csv`, or a built-in fallback map).
5. **Write artifacts** to `data/processed/`: `{SYMBOL}.parquet`, `prices_long.parquet`, `close_wide.parquet`, `metadata.csv`.
6. **Synthetic fallback** — if no raw CSVs exist and `ALLOW_SYNTHETIC_DATA=true` (default), it generates reproducible demo data so the stack runs out of the box. Set `ALLOW_SYNTHETIC_DATA=false` to forbid this.

---

## Part 4 — The ML engines (`ml/`)

### Feature engineering (`ml/features.py`)
Computes 30+ technical indicators from OHLCV: moving averages (SMA/EMA), MACD, RSI, momentum, rate-of-change, Bollinger Bands, ATR, daily/log returns, rolling volatility, volume features (volume ratio, OBV z-score), and sector relative strength. `build_feature_matrix` assembles them; `add_targets` appends forward-return targets for 1/5/20-day horizons. **Key trick:** model inputs are *scale-free ratios/oscillators* (close/MA−1, %B, RSI, ATR%) not raw prices, so one model transfers across stocks priced ₹100 or ₹10,000.

### Models (`ml/models/`)
- `registry.py` — a factory that exposes models **only if their library is installed**. Always available: Linear/Logistic Regression, Random Forest. Optional: XGBoost, LightGBM (gradient boosting), and LSTM/Transformer deep nets (TensorFlow).
- `deep.py` — Keras sequence models (LSTM, Transformer) with a scikit-learn-style `.fit/.predict`, Huber loss, early stopping.

### Training & evaluation (`ml/training/`, `ml/evaluation/`)
- `splits.py` — **leakage-safe** time-series cross-validation. Because forward-return targets overlap in time, it inserts a **gap equal to the forecast horizon** between train and test so the model can't peek at the future.
- `trainer.py` — prepares X/y, runs **walk-forward evaluation** (expanding window, out-of-sample), and produces the final forecast. Reports honest OOS metrics.
- `evaluation/metrics.py` — RMSE, MAE, MAPE, R², directional accuracy (regression); accuracy, precision, recall, F1, ROC-AUC (classification).
- `evaluation/compare.py` — sweeps all models × horizons and returns the ranked comparison table.

### Portfolio (`ml/portfolio/`)
- `profiles.py` — the three investor profiles and how each filters the universe and picks a default method.
- `optimizers.py` — pure NumPy/SciPy implementations of equal-weight, risk parity, min-volatility, and max-Sharpe (SLSQP-constrained), plus Markowitz mean-variance. Expected returns are **shrunk** toward the average and the covariance is **ridge-regularized** for stability — that's why expected returns are labeled "shrunk estimate."

### Risk (`ml/risk/metrics.py`)
Volatility, Sharpe, Sortino, Calmar, max drawdown, historical VaR-95, CVaR-95, and CAPM alpha/beta — at both stock and portfolio level. The **market proxy** for alpha/beta is the equal-weight universe return (no external index needed). Risk-free rate defaults to 6% (`RISK_FREE_RATE` env var).

### Anomaly detection (`ml/anomaly/detectors.py`)
Builds anomaly features (return z-score, rolling vol, volume ratio, drawdown, gaps, intraday range), then runs an **ensemble**: Isolation Forest + DBSCAN + an Autoencoder (with a PCA fallback if TensorFlow is absent). Majority vote flags the day, and a rule labels its type (volatility spike / unusual return / volume surge / extreme drawdown).

### Explainability (`ml/explainability/explain.py`)
Per-prediction attribution with graceful degradation: **SHAP** first (TreeExplainer/LinearExplainer/KernelExplainer), then **LIME**, then **permutation importance** as a last resort. Also computes feature importances and a **confidence** score from out-of-sample directional accuracy.

### Recommendations (`ml/recommendation/engine.py`)
Combines 5 normalized signals into one score and a natural-language reason:
| Signal | Weight | What it measures |
|---|---|---|
| forecast | 30% | model-predicted 20-day return |
| momentum | 20% | recent return scaled by volatility |
| trend | 20% | MA50/MA200 crossover + MACD sign |
| risk_adj | 15% | Sharpe ratio |
| sector_rel | 15% | stock return minus its sector's return |

Score **> +0.20 → BUY**, **< −0.20 → SELL**, else **HOLD**.

### Bonus engines
- `analytics.py` — **sector rotation** (rolling momentum ranks) and the **correlation network** (graph of stocks linked when correlation exceeds a threshold).
- `backtest.py` — portfolio backtester with periodic rebalancing, transaction costs, and an equal-weight benchmark.
- `simulation.py` — **Monte-Carlo** price paths (bootstrap or geometric Brownian motion) and **scenario** stress testing (beta-scaled market + sector shocks).
- `assistant.py` — the deterministic, regex-intent chat assistant grounded in computed artifacts.
- `reports.py` — multi-page matplotlib **PDF** report.
- `eda.py` — exploratory charts (price/volume/sector/correlation/distribution/drawdown).
- `synthetic.py` — the seeded regime-switching GBM generator that mirrors the Kaggle schema (includes 2008 and 2020 crash regimes) so the platform works with zero real data.

---

## Part 5 — Supporting pieces

- **`scripts/`** — `download_data.py` (fetch real Kaggle data), `train_all.py` (run the whole offline pipeline and warm caches), `package_for_kaggle.py` (bundle for a Kaggle notebook).
- **`notebooks/`** — `01_eda.ipynb` (exploration) and `kaggle_train.ipynb` (GPU training of the heavy LSTM/Transformer models on the real dataset).
- **`tests/`** — ~50 pytest unit + API integration tests; all 52 pass. The forecasting tests *actually train models* to prove the engine is honest.
- **`docker/` + `docker-compose.yml`** — one-command deployment: `docker compose up --build` → frontend on `:3000`, API docs on `:8000/docs`.
- **`docs/`** — `ARCHITECTURE.md`, `API.md`, `MODELS.md`, `INSTALL.md`, `KAGGLE.md`, and this `EXPLAINED.md`.
- **Reproducibility** — every random seed is fixed (`random_state=42`), so the same inputs always give the same outputs.

---

## Part 6 — How to run it

```bash
# Everything (recommended)
docker compose up --build      # → localhost:3000 (UI), localhost:8000/docs (API)

# Or local dev
uvicorn backend.main:app --reload      # backend
cd frontend && npm install && npm run dev   # frontend
pytest                                  # tests
```

First boot auto-generates synthetic demo data if the real Kaggle CSVs aren't in `data/raw/`. Drop the real dataset in and it switches over automatically — no code changes.

---

### One-paragraph summary
A historical-data-only investment platform: clean the data → engineer scale-free indicators → train honestly-validated forecasting models → build risk-profiled portfolios → measure risk → flag anomalies → fuse it all into explainable BUY/HOLD/SELL calls — all exposed through a fast, cached FastAPI backend and an 8-page Next.js dashboard, with everything reproducible and runnable in one command.
