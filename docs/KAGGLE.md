# Running the Full Training on Kaggle GPU

The heavy one-time training job (all 50 real stocks, every model family including
GPU-accelerated LSTM/Transformer) is best run on a free Kaggle GPU notebook,
where the official dataset is attachable directly.

## Steps

1. **Get the code zip.** `dist/investment-intelligence-kaggle.zip` in this repo
   (regenerate any time with `python scripts/package_for_kaggle.py`).

2. **Create the notebook.** On kaggle.com → *Create → New Notebook*, then
   *File → Import Notebook* and upload `notebooks/kaggle_train.ipynb`.

3. **Attach inputs** (right sidebar → Input → Add Input):
   - Search: **`rohanrao/nifty50-stock-market-data`** (the primary provided dataset)
   - Optionally: **`stoicstatic/india-stock-data-nse-1990-2020`** (the second
     provided dataset — the notebook prepends pre-2000 history for overlapping
     NIFTY symbols, adding up to a decade of extra training data; skipped
     automatically if not attached)
   - Upload: `investment-intelligence-kaggle.zip` as a new private dataset
     (e.g. named `nifty-intel-code`) and attach it.

4. **Enable GPU.** Settings → Accelerator → *GPU T4 x2* (or P100).

5. **Run all cells.** Expected runtime: ~15–40 min depending on accelerator.

## What it produces (Output tab → `nifty_intel_artifacts.zip`)

| Artifact | Contents |
|---|---|
| `artifacts/model_comparison.csv` | every classical model × horizon × stock, walk-forward metrics |
| `artifacts/deep_model_comparison.csv` | LSTM / Transformer vs LightGBM on focus stocks |
| `artifacts/recommendations.json` | BUY/HOLD/SELL for the full universe with reasoning |
| `artifacts/risk_table.csv` | full risk-metric suite per stock |
| `artifacts/*.joblib` | trained forecast models |
| `reports/investment_report_*.pdf` | the multi-page PDF report |
| `reports/eda/*.png` | the EDA chart suite |

## Bringing results back home

Unzip `nifty_intel_artifacts.zip` over your local `ml/artifacts/` and
`reports/generated/` — the dashboard and API pick the artifacts up as warm
caches. Also drop the real CSVs into `data/raw/` locally (and set
`ALLOW_SYNTHETIC_DATA=false`) so the interactive platform serves real data.

## Why GPU here and not locally

scikit-learn / XGBoost / LightGBM run on CPU either way, but the LSTM and
Transformer forecasters train 10–50× faster on the Kaggle GPU, and the official
dataset attaches with zero download. Local verification (pytest) stays fast and
CPU-only by design.
