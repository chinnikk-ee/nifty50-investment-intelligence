"""Shared fixtures: a small synthetic universe written to a temp DATA_DIR so
tests never touch real data and run in seconds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.synthetic import generate_stock_history

SYMBOLS = {
    "ALPHA": ("Banking", 0.0006, 0.016),
    "BETA": ("Information Technology", 0.0005, 0.018),
    "GAMMA": ("Energy", 0.0004, 0.020),
    "DELTA": ("Consumer Goods", 0.0005, 0.014),
}


@pytest.fixture(scope="session")
def stock_df() -> pd.DataFrame:
    """One stock's clean OHLCV frame (~4 years)."""
    df = generate_stock_history("ALPHA", "Banking", 0.0006, 0.016,
                                start="2017-01-02", end="2020-12-31", seed=1)
    return df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    frames = []
    for i, (sym, (sector, drift, vol)) in enumerate(SYMBOLS.items()):
        df = generate_stock_history(sym, sector, drift, vol,
                                    start="2017-01-02", end="2020-12-31", seed=10 + i)
        df = df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]].copy()
        df["Sector"] = sector
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def close_wide(panel) -> pd.DataFrame:
    return panel.pivot_table(index="Date", columns="Symbol", values="Close")


@pytest.fixture(scope="session")
def returns_wide(close_wide) -> pd.DataFrame:
    return close_wide.pct_change().dropna()


@pytest.fixture(scope="session")
def api_client(tmp_path_factory):
    """TestClient against a platform whose data dir is an isolated tmp path;
    the ingestion pipeline auto-generates synthetic data there."""
    import os

    data_dir = tmp_path_factory.mktemp("data")
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["ALLOW_SYNTHETIC_DATA"] = "true"

    # Reset path constants already imported from ml.utils.
    import ml.utils as utils

    utils.DATA_DIR = data_dir
    utils.RAW_DIR = data_dir / "raw"
    utils.PROCESSED_DIR = data_dir / "processed"

    import backend.data_loader as dl

    dl.RAW_DIR = utils.RAW_DIR
    dl.PROCESSED_DIR = utils.PROCESSED_DIR

    # Trim the synthetic universe (6 symbols, ~4 years) so ingestion, model
    # training and analytics in the API tests run in seconds, not minutes.
    import ml.synthetic as synth

    full = dict(synth.UNIVERSE)
    synth.UNIVERSE = {k: full[k] for k in list(full)[:6]}
    orig_gen = synth.generate_stock_history

    def short_history(*args, **kwargs):
        kwargs.setdefault("start", "2017-01-02")
        return orig_gen(*args, **kwargs)

    synth.generate_stock_history = short_history

    from fastapi.testclient import TestClient

    from backend.main import app
    from backend.services import get_platform

    get_platform.cache_clear()
    with TestClient(app) as client:
        yield client
    get_platform.cache_clear()
    synth.UNIVERSE = full
    synth.generate_stock_history = orig_gen
