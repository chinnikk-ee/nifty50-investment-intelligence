"""BONUS — sector rotation analysis and correlation network graph."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.utils import get_logger

logger = get_logger(__name__)


def sector_rotation(panel: pd.DataFrame, window: int = 63, points: int = 24) -> dict:
    """Rolling sector momentum ranks: which sectors are gaining or losing
    leadership. Returns a time series of ranks plus current leaders/laggards."""
    wide = panel.pivot_table(index="Date", columns="Symbol", values="Close")
    sector_map = panel.drop_duplicates("Symbol").set_index("Symbol")["Sector"]
    sector_close = wide.T.groupby(sector_map).mean().T
    momentum = sector_close.pct_change(window).dropna(how="all")
    ranks = momentum.rank(axis=1, ascending=False)

    step = max(1, len(ranks) // points)
    series = [
        {"date": str(d.date()),
         **{s: int(r) for s, r in row.items() if np.isfinite(r)}}
        for d, row in ranks.iloc[::step].iterrows()
    ]
    latest = momentum.iloc[-1].dropna().sort_values(ascending=False)
    return {
        "window_days": window,
        "rank_series": series,
        "sectors": list(momentum.columns),
        "leaders": [{"sector": s, "momentum": round(float(v), 4)} for s, v in latest.head(3).items()],
        "laggards": [{"sector": s, "momentum": round(float(v), 4)} for s, v in latest.tail(3).items()],
    }


def correlation_network(panel: pd.DataFrame, threshold: float = 0.5,
                        window_days: int = 504) -> dict:
    """Graph of return correlations above `threshold` for the network view."""
    wide = panel.pivot_table(index="Date", columns="Symbol", values="Close")
    corr = wide.tail(window_days).pct_change().corr()
    sector_map = panel.drop_duplicates("Symbol").set_index("Symbol")["Sector"]

    edges = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            c = corr.loc[a, b]
            if np.isfinite(c) and abs(c) >= threshold:
                edges.append({"source": a, "target": b, "correlation": round(float(c), 3)})

    degree = pd.Series(0, index=cols, dtype=int)
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    nodes = [
        {"id": s, "sector": str(sector_map.get(s, "Other")), "degree": int(degree[s])}
        for s in cols
    ]
    return {"threshold": threshold, "window_days": window_days,
            "nodes": nodes, "edges": edges}
