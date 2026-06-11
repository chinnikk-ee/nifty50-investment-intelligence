"""MODULE 1 — Data Ingestion Pipeline.

Loads the official Kaggle NIFTY-50 dataset (one CSV per stock + stock_metadata.csv),
validates the schema, cleans it, normalizes dates and produces:

    data/processed/<SYMBOL>.parquet      per-stock cleaned OHLCV
    data/processed/prices_long.parquet   combined long-format panel
    data/processed/close_wide.parquet    Date x Symbol close-price matrix
    data/processed/metadata.csv          symbol -> company / sector mapping

Run:  python -m backend.data_loader
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ml.utils import PROCESSED_DIR, RAW_DIR, ensure_dirs, get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
OPTIONAL_COLUMNS = ["Symbol", "Series", "Prev Close", "Last", "VWAP", "Turnover",
                    "Trades", "Deliverable Volume", "%Deliverble"]
EXCLUDED_FILES = {"stock_metadata.csv", "nifty50_all.csv"}

# Fallback sector map used when stock_metadata.csv is unavailable.
FALLBACK_SECTORS = {
    "RELIANCE": "Energy", "ONGC": "Energy", "POWERGRID": "Energy", "NTPC": "Energy",
    "IOC": "Energy", "BPCL": "Energy", "GAIL": "Energy", "COALINDIA": "Energy",
    "TCS": "Information Technology", "INFY": "Information Technology",
    "WIPRO": "Information Technology", "HCLTECH": "Information Technology",
    "TECHM": "Information Technology",
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "KOTAKBANK": "Banking",
    "SBIN": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "BAJFINANCE": "Financial Services", "BAJAJFINSV": "Financial Services",
    "HDFC": "Financial Services", "SBILIFE": "Financial Services",
    "HDFCLIFE": "Financial Services",
    "ITC": "Consumer Goods", "HINDUNILVR": "Consumer Goods", "NESTLEIND": "Consumer Goods",
    "BRITANNIA": "Consumer Goods", "TITAN": "Consumer Goods", "ASIANPAINT": "Consumer Goods",
    "SUNPHARMA": "Pharmaceuticals", "DRREDDY": "Pharmaceuticals", "CIPLA": "Pharmaceuticals",
    "DIVISLAB": "Pharmaceuticals",
    "TATAMOTORS": "Manufacturing", "TATASTEEL": "Manufacturing", "MARUTI": "Manufacturing",
    "ULTRACEMCO": "Manufacturing", "LT": "Manufacturing", "JSWSTEEL": "Manufacturing",
    "HINDALCO": "Manufacturing", "HEROMOTOCO": "Manufacturing", "BAJAJ-AUTO": "Manufacturing",
    "EICHERMOT": "Manufacturing", "M&M": "Manufacturing", "GRASIM": "Manufacturing",
    "SHREECEM": "Manufacturing", "UPL": "Manufacturing", "ADANIPORTS": "Infrastructure",
    "BHARTIARTL": "Telecommunication",
}


class SchemaValidationError(ValueError):
    """Raised when a raw CSV does not contain the mandatory OHLCV columns."""


def validate_schema(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"{source}: missing required columns {missing}")


def clean_stock_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize dates, drop duplicates, repair missing values, enforce dtypes."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")

    if "Symbol" not in df.columns:
        df["Symbol"] = symbol
    else:
        # Symbol renames over the years (e.g. INFOSYSTCH -> INFY): use file symbol.
        df["Symbol"] = symbol

    # Keep only the equity series when present.
    if "Series" in df.columns:
        eq = df[df["Series"] == "EQ"]
        if len(eq) > 0.5 * len(df):
            df = eq

    numeric_cols = [c for c in df.columns if c not in ("Date", "Symbol", "Series")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Prices: forward-fill small gaps; volume gaps become 0 (no trading).
    price_cols = [c for c in ("Open", "High", "Low", "Close", "Prev Close", "Last", "VWAP")
                  if c in df.columns]
    df[price_cols] = df[price_cols].ffill(limit=5)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    df = df.dropna(subset=["Close"])
    # Remove impossible rows (non-positive prices, High < Low).
    df = df[(df["Close"] > 0) & (df["High"] >= df["Low"])]
    return df.reset_index(drop=True)


def load_metadata(raw_dir: Path) -> pd.DataFrame:
    meta_path = raw_dir / "stock_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        meta.columns = [c.strip() for c in meta.columns]
        meta = meta.rename(columns={"Industry": "Sector", "Company Name": "Company"})
        if "Sector" not in meta.columns:
            meta["Sector"] = meta["Symbol"].map(FALLBACK_SECTORS).fillna("Other")
        return meta[["Symbol", "Company", "Sector"]].drop_duplicates("Symbol")
    logger.warning("stock_metadata.csv not found — using fallback sector map")
    return pd.DataFrame(
        [{"Symbol": s, "Company": s.title(), "Sector": sec} for s, sec in FALLBACK_SECTORS.items()]
    )


def discover_stock_files(raw_dir: Path) -> list[Path]:
    return sorted(
        p for p in raw_dir.glob("*.csv") if p.name.lower() not in EXCLUDED_FILES
    )


def run_pipeline(raw_dir: Path | None = None, processed_dir: Path | None = None) -> pd.DataFrame:
    """Execute the full ingestion pipeline. Returns the combined long panel."""
    ensure_dirs()
    raw_dir = raw_dir or RAW_DIR
    processed_dir = processed_dir or PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = discover_stock_files(raw_dir)
    if not files:
        if os.getenv("ALLOW_SYNTHETIC_DATA", "true").lower() == "true":
            logger.warning("no raw CSVs found in %s — generating synthetic demo data", raw_dir)
            from ml.synthetic import generate_dataset

            generate_dataset(out_dir=raw_dir)
            files = discover_stock_files(raw_dir)
        else:
            raise FileNotFoundError(
                f"No stock CSVs in {raw_dir}. Run scripts/download_data.py first."
            )

    meta = load_metadata(raw_dir)
    frames: list[pd.DataFrame] = []
    for path in files:
        symbol = path.stem.upper()
        try:
            raw = pd.read_csv(path)
            validate_schema(raw, path.name)
            df = clean_stock_frame(raw, symbol)
            if len(df) < 260:  # require at least ~1 year of history
                logger.warning("skipping %s: only %d usable rows", symbol, len(df))
                continue
            df.to_parquet(processed_dir / f"{symbol}.parquet", index=False)
            frames.append(df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
                             + (["Turnover"] if "Turnover" in df.columns else [])])
            logger.info("processed %-12s rows=%-6d %s -> %s",
                        symbol, len(df), df["Date"].min().date(), df["Date"].max().date())
        except (SchemaValidationError, pd.errors.ParserError) as exc:
            logger.error("skipping %s: %s", path.name, exc)

    if not frames:
        raise RuntimeError("Pipeline produced no usable stock data.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(meta[["Symbol", "Sector"]], on="Symbol", how="left")
    panel["Sector"] = panel["Sector"].fillna("Other")
    panel.to_parquet(processed_dir / "prices_long.parquet", index=False)

    close_wide = panel.pivot_table(index="Date", columns="Symbol", values="Close")
    close_wide.to_parquet(processed_dir / "close_wide.parquet")

    meta.to_csv(processed_dir / "metadata.csv", index=False)
    logger.info("pipeline complete: %d symbols, %d rows", panel["Symbol"].nunique(), len(panel))
    return panel


def load_processed(processed_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load processed artifacts; runs the pipeline automatically if absent."""
    processed_dir = processed_dir or PROCESSED_DIR
    panel_path = processed_dir / "prices_long.parquet"
    if not panel_path.exists():
        run_pipeline(processed_dir=processed_dir)
    return {
        "panel": pd.read_parquet(panel_path),
        "close_wide": pd.read_parquet(processed_dir / "close_wide.parquet"),
        "metadata": pd.read_csv(processed_dir / "metadata.csv"),
    }


if __name__ == "__main__":
    run_pipeline()
