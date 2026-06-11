import pandas as pd
import pytest

from backend.data_loader import SchemaValidationError, clean_stock_frame, validate_schema


def _raw(rows: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    base = pd.Series(range(100, 100 + rows), dtype=float)
    return pd.DataFrame({
        "Date": dates.astype(str),
        "Open": base,
        "High": base + 2,
        "Low": base - 2,
        "Close": base + 1,
        "Volume": 1000,
    })


def test_validate_schema_rejects_missing_columns():
    with pytest.raises(SchemaValidationError):
        validate_schema(pd.DataFrame({"Date": [], "Close": []}), "x.csv")


def test_clean_drops_duplicate_dates():
    raw = pd.concat([_raw(), _raw().head(10)], ignore_index=True)
    cleaned = clean_stock_frame(raw, "TEST")
    assert cleaned["Date"].is_unique


def test_clean_forward_fills_missing_prices():
    raw = _raw()
    raw.loc[50, "Close"] = None
    cleaned = clean_stock_frame(raw, "TEST")
    assert cleaned["Close"].notna().all()


def test_clean_removes_impossible_rows():
    raw = _raw()
    raw.loc[10, "Close"] = -5          # non-positive price
    raw.loc[20, "High"] = 0.0          # High < Low
    cleaned = clean_stock_frame(raw, "TEST")
    assert len(cleaned) == len(raw) - 2
    assert (cleaned["Close"] > 0).all()
    assert (cleaned["High"] >= cleaned["Low"]).all()


def test_clean_sets_symbol_and_sorts():
    raw = _raw().sample(frac=1, random_state=0)  # shuffle
    cleaned = clean_stock_frame(raw, "TEST")
    assert (cleaned["Symbol"] == "TEST").all()
    assert cleaned["Date"].is_monotonic_increasing
