"""Shared utilities for the ML package."""
from __future__ import annotations

import logging
import os
from pathlib import Path

TRADING_DAYS_PER_YEAR = 252

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "ml" / "artifacts"))


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def annualize_return(daily_mean: float) -> float:
    return daily_mean * TRADING_DAYS_PER_YEAR


def annualize_volatility(daily_std: float) -> float:
    return daily_std * (TRADING_DAYS_PER_YEAR ** 0.5)
