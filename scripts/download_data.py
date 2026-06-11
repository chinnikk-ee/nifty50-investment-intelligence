"""Download the official Kaggle NIFTY-50 dataset into data/raw.

Dataset: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data
(one CSV per stock + stock_metadata.csv).

Usage (either path works):
  1) kagglehub (pip install kagglehub) with Kaggle credentials configured:
        python scripts/download_data.py
  2) Manual: download the zip from the Kaggle page, extract all CSVs into
        data/raw/

Without the dataset the platform auto-generates synthetic demo data
(ALLOW_SYNTHETIC_DATA=true), so every module still runs out of the box.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.utils import RAW_DIR, ensure_dirs, get_logger  # noqa: E402

logger = get_logger("download_data")

DATASET = "rohanrao/nifty50-stock-market-data"


def main() -> int:
    ensure_dirs()
    existing = list(RAW_DIR.glob("*.csv"))
    if existing:
        logger.info("data/raw already has %d CSVs — nothing to do", len(existing))
        return 0

    try:
        import kagglehub
    except ImportError:
        logger.error(
            "kagglehub not installed. Either `pip install kagglehub` (and set up "
            "Kaggle API credentials), or download the dataset manually from "
            "https://www.kaggle.com/datasets/%s and extract the CSVs into %s",
            DATASET, RAW_DIR,
        )
        return 1

    logger.info("downloading %s via kagglehub…", DATASET)
    path = Path(kagglehub.dataset_download(DATASET))
    copied = 0
    for csv in path.rglob("*.csv"):
        shutil.copy2(csv, RAW_DIR / csv.name)
        copied += 1
    logger.info("copied %d CSVs into %s", copied, RAW_DIR)
    return 0 if copied else 1


if __name__ == "__main__":
    raise SystemExit(main())
