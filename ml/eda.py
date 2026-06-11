"""MODULE 3 — EDA visual analytics.

Reusable plotting functions over the processed panel. Each function returns a
matplotlib Figure (for PDF reports / notebooks); `save_fig` persists to disk.
All functions accept the long panel produced by backend.data_loader.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")

FIGSIZE = (11, 5.5)


def _close_wide(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot_table(index="Date", columns="Symbol", values="Close")


def save_fig(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# 1 ---------------------------------------------------------------- prices
def plot_price_trends(panel: pd.DataFrame, symbols: list[str], normalize: bool = True) -> plt.Figure:
    wide = _close_wide(panel)[symbols].dropna(how="all")
    if normalize:
        wide = wide / wide.bfill().iloc[0] * 100
    fig, ax = plt.subplots(figsize=FIGSIZE)
    wide.plot(ax=ax, linewidth=1.2)
    ax.set_title("Price Trends" + (" (indexed to 100)" if normalize else ""))
    ax.set_ylabel("Indexed Price" if normalize else "Close (INR)")
    ax.legend(ncol=4, fontsize=8)
    return fig


# 2 ---------------------------------------------------------------- volume
def plot_volume_trends(panel: pd.DataFrame, symbol: str) -> plt.Figure:
    df = panel[panel["Symbol"] == symbol].set_index("Date")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(df.index, df["Close"], color="tab:blue", lw=1.1)
    ax1.set_title(f"{symbol} — Price & Volume")
    ax1.set_ylabel("Close")
    ax2.bar(df.index, df["Volume"], width=1.5, color="tab:gray")
    ax2.plot(df.index, df["Volume"].rolling(50).mean(), color="tab:red", lw=1.2,
             label="50d avg volume")
    ax2.set_ylabel("Volume")
    ax2.legend(fontsize=8)
    return fig


# 3 ---------------------------------------------------------------- sectors
def plot_sector_comparison(panel: pd.DataFrame) -> plt.Figure:
    wide = _close_wide(panel)
    rets = wide.pct_change()
    sector_map = panel.drop_duplicates("Symbol").set_index("Symbol")["Sector"]
    sector_rets = rets.T.groupby(sector_map).mean().T
    cum = (1 + sector_rets.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    cum.plot(ax=ax, linewidth=1.4)
    ax.set_title("Cumulative Sector Performance (equal-weight)")
    ax.set_ylabel("Growth of 1 INR")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    return fig


# 4 ---------------------------------------------------------------- correlation
def plot_correlation_heatmap(panel: pd.DataFrame, symbols: list[str] | None = None,
                             window_days: int = 756) -> plt.Figure:
    wide = _close_wide(panel)
    if symbols:
        wide = wide[symbols]
    corr = wide.tail(window_days).pct_change().corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, square=True,
                cbar_kws={"shrink": 0.7}, xticklabels=True, yticklabels=True)
    ax.set_title(f"Return Correlation (last {window_days} trading days)")
    ax.tick_params(labelsize=7)
    return fig


# 5 ---------------------------------------------------------------- distribution
def plot_return_distribution(panel: pd.DataFrame, symbol: str) -> plt.Figure:
    rets = panel[panel["Symbol"] == symbol].set_index("Date")["Close"].pct_change().dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(rets, bins=120, kde=True, ax=ax1, color="tab:blue")
    ax1.axvline(rets.mean(), color="red", ls="--", lw=1, label=f"mean={rets.mean():.4f}")
    ax1.set_title(f"{symbol} — Daily Return Distribution")
    ax1.legend(fontsize=8)
    # QQ-style tail check against normal
    from scipy import stats

    stats.probplot(rets, dist="norm", plot=ax2)
    ax2.set_title("Normal Q-Q (fat tails visible at extremes)")
    return fig


# 6 ---------------------------------------------------------------- volatility
def plot_volatility(panel: pd.DataFrame, symbols: list[str], window: int = 21) -> plt.Figure:
    wide = _close_wide(panel)[symbols]
    vol = wide.pct_change().rolling(window).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    vol.plot(ax=ax, linewidth=1.0)
    ax.set_title(f"Rolling {window}d Annualized Volatility")
    ax.set_ylabel("Volatility")
    ax.legend(ncol=4, fontsize=8)
    return fig


# 7 ---------------------------------------------------------------- drawdown
def compute_drawdown(close: pd.Series) -> pd.Series:
    peak = close.cummax()
    return close / peak - 1


def plot_drawdown(panel: pd.DataFrame, symbol: str) -> plt.Figure:
    close = panel[panel["Symbol"] == symbol].set_index("Date")["Close"]
    dd = compute_drawdown(close)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(dd.index, dd, 0, color="tab:red", alpha=0.45)
    ax.set_title(f"{symbol} — Drawdown (max {dd.min():.1%})")
    ax.set_ylabel("Drawdown")
    return fig


def generate_eda_suite(panel: pd.DataFrame, out_dir: str | Path,
                       focus_symbol: str | None = None) -> list[Path]:
    """Render the full EDA chart suite to out_dir; returns written paths."""
    out_dir = Path(out_dir)
    symbols = sorted(panel["Symbol"].unique())
    focus = focus_symbol or symbols[0]
    top = symbols[: min(8, len(symbols))]
    jobs = [
        ("price_trends.png", plot_price_trends(panel, top)),
        ("volume_trends.png", plot_volume_trends(panel, focus)),
        ("sector_comparison.png", plot_sector_comparison(panel)),
        ("correlation_heatmap.png", plot_correlation_heatmap(panel)),
        ("return_distribution.png", plot_return_distribution(panel, focus)),
        ("volatility.png", plot_volatility(panel, top)),
        ("drawdown.png", plot_drawdown(panel, focus)),
    ]
    return [save_fig(fig, out_dir / name) for name, fig in jobs]
