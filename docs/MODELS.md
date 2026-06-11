# Model Documentation

## Targets

For each stock and horizon *h* ∈ {1, 5, 20} trading days:

- **Regression** — forward simple return: `Close[t+h] / Close[t] − 1`
- **Classification** — direction: `1` if the forward return is positive

## Features (`ml/features.py`)

Scale-free by construction (ratios/oscillators, not price levels): `close_over_ma{5,10,20,50,200}`, RSI-14, MACD (line/signal/histogram), Bollinger %B and bandwidth, ATR%, momentum/ROC (10/21d), 1/5/21-day returns, log returns, 21/63-day rolling volatility, volume ratio/ROC, sector relative strength, day-of-week and month.

## Models (`ml/models/registry.py`)

| Model | Type | Notes |
|---|---|---|
| Linear / Logistic Regression | baseline | standardized pipeline |
| Random Forest | ensemble | 300 trees, depth 8, leaf 20 — default API model |
| XGBoost | boosting | 400 × depth 5, lr 0.03, subsampled |
| LightGBM | boosting | 500 × 31 leaves, lr 0.03 |
| LSTM | deep | 64→32 stacked, lookback 30, Huber loss (optional TF install) |
| Transformer | deep | 2 encoder blocks, d_model 32, learned positions (optional) |

Models register only if their library imports — the platform never crashes on a minimal install.

## Validation (`ml/training/`)

- **Expanding-window time-series CV** (default 5 folds) and classic **walk-forward** splits.
- A **gap of h rows** is removed at every boundary: overlapping forward-return labels would otherwise leak future prices into training.
- Reported metrics are **out-of-fold only**. Regression: RMSE, MAE, MAPE (floored denominator), R², directional accuracy. Classification: accuracy, precision, recall, F1, ROC-AUC, directional accuracy.

## Portfolio optimization (`ml/portfolio/`)

Inputs are annualized from daily returns; expected returns are **shrunk 50% toward the cross-sectional mean** (raw historical means are too noisy for Markowitz). Methods: equal weight, risk parity (fixed-point ERC), Markowitz utility, max Sharpe, min volatility — all long-only with per-position caps via SLSQP. Profiles: Conservative (low-vol universe, min-vol, 12% cap), Balanced (risk parity, 20% cap), Aggressive (momentum universe, max Sharpe, 30% cap).

## Risk (`ml/risk/`)

Annualized at 252 days, risk-free 6% (env `RISK_FREE_RATE`). VaR/CVaR are historical, 95%, one-day. Beta/alpha are CAPM versus the equal-weight universe proxy (the dataset contains no index series).

## Recommendation scoring (`ml/recommendation/`)

`score = 0.30·forecast + 0.20·momentum + 0.20·trend + 0.15·risk_adj + 0.15·sector_rel`, each component tanh-squashed to [−1, 1]. BUY ≥ +0.20, SELL ≤ −0.20. Reasoning text is assembled from the strongest components.

## Explainability (`ml/explainability/`)

SHAP TreeExplainer for tree ensembles, LinearExplainer for linear models, KernelExplainer (sampled) otherwise; LIME as a second lens; signed feature importances as the dependency-free fallback. Confidence = probability margin (classifiers) or a mapping of out-of-sample directional accuracy (regressors) — never in-sample fit.

## Anomaly detection (`ml/anomaly/`)

Isolation Forest (1% contamination), DBSCAN (auto-eps from median pairwise distance, noise = anomaly) and an autoencoder (Keras, or PCA reconstruction fallback) vote over 7 engineered features; ≥2 votes flags the day, then a rule labels it volatility spike / unusual return / volume surge / extreme drawdown.
