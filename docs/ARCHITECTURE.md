# Architecture

```
                 ┌────────────────────────────────────────────────────────┐
                 │                  Next.js 15 Frontend                    │
                 │  Dashboard · Stocks · Forecasting · Portfolio · Risk   │
                 │  Anomalies · AI Insights (chat + SHAP) · Settings      │
                 └───────────────────────────┬────────────────────────────┘
                                             │ REST (JSON)
                 ┌───────────────────────────▼────────────────────────────┐
                 │                    FastAPI Backend                      │
                 │  routers/  → thin endpoints, Pydantic validation        │
                 │  services.Platform → lazy data load + memoized engines  │
                 └───────────────────────────┬────────────────────────────┘
                                             │
      ┌───────────────┬───────────────┬──────┴──────┬───────────────┬──────────────┐
      ▼               ▼               ▼             ▼               ▼              ▼
 ml/features     ml/models       ml/portfolio    ml/risk       ml/anomaly    ml/explainability
 (indicators)    ml/training     (5 optimizers,  (vol/Sharpe/  (IForest/     (SHAP/LIME/
                 ml/evaluation    3 profiles)    VaR/CVaR/     DBSCAN/AE      importance)
                 (walk-forward)                  alpha/beta)    ensemble)
      │               │               │             │               │              │
      └───────────────┴───────────────┴──────┬──────┴───────────────┴──────────────┘
                                             ▼
                            ml/recommendation (BUY/HOLD/SELL + reasoning)
                            ml/analytics · ml/backtest · ml/simulation
                            ml/assistant (insight chat) · ml/reports (PDF)
                                             ▲
                 ┌───────────────────────────┴────────────────────────────┐
                 │                 Data layer (backend/data_loader)        │
                 │  data/raw/*.csv ── validate → clean → normalize ──►     │
                 │  data/processed/{SYMBOL.parquet, prices_long.parquet,   │
                 │                  close_wide.parquet, metadata.csv}      │
                 │  (synthetic fallback generator when raw data absent)    │
                 └─────────────────────────────────────────────────────────┘
```

## Design decisions

- **Single service object** (`backend/services.Platform`): every engine is pure-functional over DataFrames; the Platform memoizes results keyed by request parameters so the dashboard is fast after first computation. `scripts/train_all.py` warms these caches offline.
- **Leakage-safe evaluation**: `ml/training/splits.py` inserts a gap equal to the forecast horizon between train and test windows because forward-return targets overlap; metrics are reported out-of-fold only.
- **Graceful degradation**: XGBoost/LightGBM/TensorFlow/SHAP/LIME are optional. The registry only exposes models whose libraries import; the autoencoder detector falls back to PCA reconstruction; explainability falls back to signed feature importances.
- **Scale-free features**: model inputs are ratios/oscillators (close/MA − 1, %B, ATR%, RSI, returns) rather than price levels, so models transfer across stocks with very different price scales.
- **Synthetic fallback**: a seeded regime-switching GBM generator mirrors the Kaggle schema exactly, so the full stack (and CI) runs without the dataset; real data drops in with zero code changes.
- **No external data**: the market proxy for beta/alpha is the equal-weight universe return; the chat assistant answers only from computed artifacts.
