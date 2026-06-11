"""Synthetic NIFTY-50-shaped data generator.

Used ONLY as a reproducible demo fallback when the Kaggle dataset has not been
downloaded yet (ALLOW_SYNTHETIC_DATA=true). The schema exactly mirrors the
official dataset so every downstream module works identically on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.utils import RAW_DIR, ensure_dirs, get_logger

logger = get_logger(__name__)

# Symbol -> (sector, drift, vol) — representative subset of the NIFTY-50 universe.
UNIVERSE: dict[str, tuple[str, float, float]] = {
    "RELIANCE":   ("Energy", 0.00060, 0.018),
    "TCS":        ("Information Technology", 0.00055, 0.016),
    "INFY":       ("Information Technology", 0.00050, 0.017),
    "WIPRO":      ("Information Technology", 0.00035, 0.018),
    "HCLTECH":    ("Information Technology", 0.00045, 0.018),
    "HDFCBANK":   ("Banking", 0.00058, 0.015),
    "ICICIBANK":  ("Banking", 0.00052, 0.019),
    "KOTAKBANK":  ("Banking", 0.00048, 0.017),
    "SBIN":       ("Banking", 0.00040, 0.022),
    "AXISBANK":   ("Banking", 0.00042, 0.021),
    "BAJFINANCE": ("Financial Services", 0.00075, 0.024),
    "HDFC":       ("Financial Services", 0.00050, 0.017),
    "ITC":        ("Consumer Goods", 0.00035, 0.014),
    "HINDUNILVR": ("Consumer Goods", 0.00045, 0.013),
    "NESTLEIND":  ("Consumer Goods", 0.00048, 0.012),
    "BRITANNIA":  ("Consumer Goods", 0.00050, 0.015),
    "SUNPHARMA":  ("Pharmaceuticals", 0.00038, 0.018),
    "DRREDDY":    ("Pharmaceuticals", 0.00036, 0.017),
    "CIPLA":      ("Pharmaceuticals", 0.00034, 0.017),
    "ONGC":       ("Energy", 0.00025, 0.021),
    "POWERGRID":  ("Energy", 0.00030, 0.014),
    "NTPC":       ("Energy", 0.00028, 0.015),
    "TATAMOTORS": ("Manufacturing", 0.00040, 0.027),
    "TATASTEEL":  ("Manufacturing", 0.00035, 0.026),
    "MARUTI":     ("Manufacturing", 0.00045, 0.019),
    "ULTRACEMCO": ("Manufacturing", 0.00042, 0.018),
    "LT":         ("Manufacturing", 0.00044, 0.018),
    "BHARTIARTL": ("Telecommunication", 0.00038, 0.020),
}


def generate_stock_history(
    symbol: str,
    sector: str,
    drift: float,
    vol: float,
    start: str = "2000-01-03",
    end: str = "2021-04-30",
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate one stock's OHLCV history via a regime-switching GBM with
    volatility clustering, crash episodes and volume-volatility coupling."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # Volatility clustering (slow AR(1) on log-vol) + crash regimes.
    log_vol = np.zeros(n)
    log_vol[0] = np.log(vol)
    for t in range(1, n):
        log_vol[t] = 0.985 * log_vol[t - 1] + 0.015 * np.log(vol) + 0.05 * rng.normal()
    sigma = np.exp(log_vol)

    # Inject 2008-like and 2020-like crash windows when in range.
    rets = rng.normal(drift, sigma)
    for crash_start, crash_len, crash_drift in (("2008-09-01", 90, -0.004), ("2020-02-20", 40, -0.009)):
        idx = dates.searchsorted(pd.Timestamp(crash_start))
        if 0 < idx < n:
            sl = slice(idx, min(idx + crash_len, n))
            rets[sl] += crash_drift + rng.normal(0, 0.012, size=len(range(*sl.indices(n))))

    base_price = rng.uniform(80, 1500)
    close = base_price * np.exp(np.cumsum(rets))

    intraday = np.abs(rng.normal(0, sigma)) + 0.002
    open_ = close * (1 + rng.normal(0, sigma / 2))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    prev_close = np.concatenate([[close[0]], close[:-1]])

    base_volume = rng.uniform(2e5, 5e6)
    volume = (base_volume * (1 + 25 * np.abs(rets)) * rng.lognormal(0, 0.4, n)).astype(np.int64)
    vwap = (high + low + close) / 3
    turnover = vwap * volume

    return pd.DataFrame(
        {
            "Date": dates,
            "Symbol": symbol,
            "Series": "EQ",
            "Prev Close": prev_close.round(2),
            "Open": open_.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Last": close.round(2),
            "Close": close.round(2),
            "VWAP": vwap.round(2),
            "Volume": volume,
            "Turnover": turnover.round(2),
            "Trades": (volume / rng.uniform(50, 200)).astype(np.int64),
            "Deliverable Volume": (volume * rng.uniform(0.3, 0.7, n)).astype(np.int64),
            "%Deliverble": rng.uniform(0.3, 0.7, n).round(4),
        }
    )


def generate_dataset(out_dir=None, seed: int = 42) -> None:
    """Write the full synthetic dataset (per-stock CSVs + metadata) to data/raw."""
    ensure_dirs()
    out_dir = out_dir or RAW_DIR
    meta_rows = []
    for i, (symbol, (sector, drift, vol)) in enumerate(UNIVERSE.items()):
        df = generate_stock_history(symbol, sector, drift, vol, seed=seed + i)
        df.to_csv(out_dir / f"{symbol}.csv", index=False)
        meta_rows.append(
            {"Company Name": symbol.title(), "Industry": sector, "Symbol": symbol, "Series": "EQ"}
        )
        logger.info("generated %s (%d rows)", symbol, len(df))
    pd.DataFrame(meta_rows).to_csv(out_dir / "stock_metadata.csv", index=False)
    logger.info("synthetic dataset written to %s", out_dir)


if __name__ == "__main__":
    generate_dataset()
