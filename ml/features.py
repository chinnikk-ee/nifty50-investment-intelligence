"""MODULE 2 — Feature Engineering.

Vectorized technical-indicator computation on a single stock's OHLCV frame
(columns: Date, Open, High, Low, Close, Volume). Every function returns a
Series/DataFrame aligned to the input index; `build_feature_matrix` assembles
the full model-ready matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.utils import TRADING_DAYS_PER_YEAR

MA_WINDOWS = (5, 10, 20, 50, 200)


# ---------------------------------------------------------------- trend
def moving_average(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def exponential_moving_average(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = exponential_moving_average(close, fast)
    ema_slow = exponential_moving_average(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line}
    )


# ---------------------------------------------------------------- momentum
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    # gain/0 -> inf -> RSI 100 (pure uptrend); 0/0 -> NaN -> neutral 50.
    rs = gain / loss
    return (100 - 100 / (1 + rs)).fillna(50.0)


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    return close - close.shift(window)


def rate_of_change(close: pd.Series, window: int = 10) -> pd.Series:
    return close.pct_change(window) * 100


# ---------------------------------------------------------------- volatility
def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper, lower = mid + num_std * std, mid - num_std * std
    width = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_pct_b": pct_b}
    )


def average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["Close"].shift()
    tr = pd.concat(
        [df["High"] - df["Low"],
         (df["High"] - prev_close).abs(),
         (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def rolling_volatility(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    vol = returns.rolling(window).std()
    return vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else vol


# ---------------------------------------------------------------- returns
def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift())


# ---------------------------------------------------------------- volume
def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    vol = df["Volume"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["volume_ma20"] = vol.rolling(20).mean()
    out["volume_ratio"] = vol / out["volume_ma20"].replace(0, np.nan)
    out["volume_roc"] = vol.pct_change(5)
    # On-balance volume, normalized by its own 200-day scale.
    direction = np.sign(df["Close"].diff()).fillna(0)
    obv = (direction * vol).cumsum()
    out["obv_z"] = (obv - obv.rolling(200).mean()) / obv.rolling(200).std()
    return out


# ---------------------------------------------------------------- sector
def sector_performance(panel: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Per-sector rolling mean return computed from the long panel
    (Date, Symbol, Close, Sector). Returns Date x Sector matrix."""
    wide = panel.pivot_table(index="Date", columns="Symbol", values="Close")
    rets = wide.pct_change()
    sector_map = panel.drop_duplicates("Symbol").set_index("Symbol")["Sector"]
    sector_rets = rets.T.groupby(sector_map).mean().T
    return sector_rets.rolling(window).mean()


def sector_relative_strength(df: pd.DataFrame, sector_ret: pd.Series) -> pd.Series:
    """Stock 21d return minus its sector's 21d rolling mean return."""
    stock_ret = df["Close"].pct_change(21)
    aligned = sector_ret.reindex(pd.Index(df["Date"])).to_numpy()
    return stock_ret - pd.Series(aligned, index=df.index) * 21


# ---------------------------------------------------------------- assembly
def build_feature_matrix(df: pd.DataFrame, sector_ret: pd.Series | None = None) -> pd.DataFrame:
    """Assemble the full feature matrix for one stock.

    Parameters
    ----------
    df         : per-stock OHLCV frame sorted by Date.
    sector_ret : optional Date-indexed rolling sector return for relative strength.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"]
    feats = pd.DataFrame({"Date": df["Date"], "Close": close})

    for w in MA_WINDOWS:
        feats[f"ma{w}"] = moving_average(close, w)
        feats[f"close_over_ma{w}"] = close / feats[f"ma{w}"] - 1
    feats["ema12"] = exponential_moving_average(close, 12)
    feats["ema26"] = exponential_moving_average(close, 26)

    feats["rsi14"] = rsi(close)
    feats = pd.concat([feats, macd(close), bollinger_bands(close)], axis=1)
    feats["atr14"] = average_true_range(df)
    feats["atr_pct"] = feats["atr14"] / close

    feats["momentum10"] = momentum(close)
    feats["roc10"] = rate_of_change(close)
    feats["roc21"] = rate_of_change(close, 21)

    feats["ret_1d"] = daily_returns(close)
    feats["log_ret_1d"] = log_returns(close)
    feats["ret_5d"] = close.pct_change(5)
    feats["ret_21d"] = close.pct_change(21)
    feats["vol_21d"] = rolling_volatility(feats["ret_1d"])
    feats["vol_63d"] = rolling_volatility(feats["ret_1d"], 63)

    feats = pd.concat([feats, volume_features(df)], axis=1)

    if sector_ret is not None:
        feats["sector_rel_strength"] = sector_relative_strength(df, sector_ret)

    # Calendar effects.
    feats["day_of_week"] = df["Date"].dt.dayofweek
    feats["month"] = df["Date"].dt.month

    return feats


def add_targets(feats: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20)) -> pd.DataFrame:
    """Append forward-return regression targets and direction labels.

    target_ret_{h}  : forward h-day simple return (regression)
    target_dir_{h}  : 1 if forward return > 0 else 0 (classification)
    """
    out = feats.copy()
    for h in horizons:
        fwd = out["Close"].shift(-h) / out["Close"] - 1
        out[f"target_ret_{h}"] = fwd
        out[f"target_dir_{h}"] = (fwd > 0).astype(int).where(fwd.notna())
    return out


FEATURE_COLUMNS_EXCLUDE = {"Date", "Close"} | {f"ma{w}" for w in MA_WINDOWS} | {
    "ema12", "ema26", "bb_mid", "bb_upper", "bb_lower", "atr14", "momentum10", "obv_z"
}
# Note: raw price-level columns are excluded from the model inputs to keep
# features scale-free and comparable across stocks; ratio/oscillator forms
# (close_over_ma*, bb_pct_b, atr_pct, rsi, returns) are used instead.
# obv_z is kept OUT by default because of its long warm-up; include explicitly
# if training on full histories.


def model_feature_columns(feats: pd.DataFrame) -> list[str]:
    return [
        c for c in feats.columns
        if c not in FEATURE_COLUMNS_EXCLUDE and not c.startswith("target_")
    ]
