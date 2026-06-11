"""MODULE 4 — temporal split generators.

Both generators yield (train_idx, test_idx) pairs of position indices over a
chronologically sorted dataset. `gap` rows are removed from the end of every
training window so overlapping forward-return targets never leak into the
test period.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def time_series_splits(
    n: int, n_splits: int = 5, gap: int = 0, min_train_frac: float = 0.3
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window time-series CV (sklearn TimeSeriesSplit semantics,
    plus a leakage gap and a guaranteed minimum first training window)."""
    first_train = max(int(n * min_train_frac), 1)
    test_size = (n - first_train) // n_splits
    if test_size < 1:
        raise ValueError(f"not enough rows ({n}) for {n_splits} splits")
    for i in range(n_splits):
        test_start = first_train + i * test_size
        test_end = test_start + test_size if i < n_splits - 1 else n
        train_end = max(test_start - gap, 1)
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def walk_forward_splits(
    n: int, train_window: int, test_window: int, gap: int = 0, expanding: bool = True
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Classic walk-forward validation: train on a (rolling or expanding)
    window, test on the next `test_window` rows, then step forward."""
    start = train_window
    while start + gap < n:
        test_start = start + gap
        test_end = min(test_start + test_window, n)
        if test_start >= n:
            break
        train_start = 0 if expanding else start - train_window
        yield np.arange(train_start, start), np.arange(test_start, test_end)
        start += test_window
