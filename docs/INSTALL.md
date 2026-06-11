# Installation & Reproducibility Guide

## Prerequisites
- Docker (easiest), or: Python 3.11 + Node.js 20
- Optional: Kaggle account for the real dataset

## Option A — Docker (recommended)

```bash
docker compose up --build
```
Backend on :8000, frontend on :3000. The backend volume-mounts `./data`, so dropping the Kaggle CSVs into `data/raw/` and restarting switches from synthetic to real data.

## Option B — Local

```bash
# 1. Backend
python -m venv .venv
.venv\Scripts\activate                  # Windows  (Unix: source .venv/bin/activate)
pip install -r requirements.txt
pip install -r requirements-deep.txt    # optional: LSTM/Transformer/autoencoder

# 2. Data (optional — synthetic fallback otherwise)
python scripts/download_data.py         # needs `pip install kagglehub` + Kaggle creds
# or manually extract the Kaggle zip into data/raw/

# 3. Ingestion + offline training (warms all caches, writes artifacts & PDF)
python -m backend.data_loader
python scripts/train_all.py             # add --symbols RELIANCE TCS to subset

# 4. Serve
uvicorn backend.main:app --reload       # http://localhost:8000/docs

# 5. Frontend
cd frontend
npm install
npm run dev                             # http://localhost:3000
```

## Environment variables (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `./data` | raw/processed data root |
| `ARTIFACTS_DIR` | `./ml/artifacts` | trained-model & comparison artifacts |
| `ALLOW_SYNTHETIC_DATA` | `true` | auto-generate demo data when raw CSVs absent |
| `RISK_FREE_RATE` | `0.06` | annual rate for Sharpe/Sortino/alpha |
| `CORS_ORIGINS` | `http://localhost:3000` | comma-separated allowed origins |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | API base for the frontend |

## Tests

```bash
pytest            # unit + integration; uses isolated synthetic data, no downloads
```

## Reproducibility checklist
- Seeds fixed everywhere (`random_state=42`, seeded generators, `tf.random.set_seed`).
- Deterministic synthetic dataset (same seed → identical CSVs).
- `scripts/train_all.py` is the single entry point that regenerates every artifact: EDA charts (`reports/generated/eda/`), model comparison (`ml/artifacts/model_comparison.csv`), risk table, recommendations JSON and the PDF report.
- Walk-forward metrics are deterministic given the same data and seeds.
