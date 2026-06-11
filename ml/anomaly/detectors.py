"""MODULE 7 — market anomaly detection.

Three unsupervised detectors over the same engineered anomaly features:

  * Isolation Forest
  * DBSCAN (noise points = anomalies)
  * Autoencoder reconstruction error (Keras when available, otherwise a PCA
    reconstruction-error fallback so the method always works)

`detect_anomalies` ensembles them, labels each anomalous day with a
human-readable type (volatility spike / unusual return / volume surge /
extreme drawdown) and returns a tidy report frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ml.eda import compute_drawdown
from ml.utils import get_logger

logger = get_logger(__name__)

ANOMALY_FEATURES = ["ret_1d", "ret_z", "vol_21", "volume_ratio", "drawdown", "gap", "intraday_range"]


def anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer detector inputs from one stock's OHLCV frame (Date-sorted)."""
    out = pd.DataFrame({"Date": df["Date"].to_numpy()})
    close = df["Close"].reset_index(drop=True)
    ret = close.pct_change()
    out["Close"] = close
    out["ret_1d"] = ret
    out["ret_z"] = (ret - ret.rolling(63).mean()) / ret.rolling(63).std()
    out["vol_21"] = ret.rolling(21).std()
    vol_ma = df["Volume"].reset_index(drop=True).rolling(21).mean()
    out["volume_ratio"] = df["Volume"].reset_index(drop=True) / vol_ma.replace(0, np.nan)
    out["drawdown"] = compute_drawdown(close)
    prev_close = close.shift()
    out["gap"] = df["Open"].reset_index(drop=True) / prev_close - 1
    out["intraday_range"] = ((df["High"] - df["Low"]) / close).reset_index(drop=True)
    return out.dropna().reset_index(drop=True)


def _scaled(feats: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(feats[ANOMALY_FEATURES])


# ------------------------------------------------------------------ detectors
def isolation_forest_detect(feats: pd.DataFrame, contamination: float = 0.01) -> np.ndarray:
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    return model.fit_predict(_scaled(feats)) == -1


def dbscan_detect(feats: pd.DataFrame, eps: float | None = None, min_samples: int = 10) -> np.ndarray:
    X = _scaled(feats)
    if eps is None:
        # Heuristic: median pairwise distance of a sample, shrunk.
        rng = np.random.default_rng(42)
        sample = X[rng.choice(len(X), size=min(400, len(X)), replace=False)]
        dists = np.linalg.norm(sample[:, None] - sample[None, :], axis=-1)
        eps = float(np.median(dists) * 0.5) or 1.0
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return labels == -1


def autoencoder_detect(feats: pd.DataFrame, quantile: float = 0.99) -> np.ndarray:
    """Reconstruction-error detector. Keras AE when TF is installed, else a
    PCA bottleneck reconstruction — same principle, zero heavy deps."""
    X = _scaled(feats)
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        tf.random.set_seed(42)
        ae = models.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(8, activation="relu"),
            layers.Dense(3, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(X.shape[1]),
        ])
        ae.compile(optimizer="adam", loss="mse")
        ae.fit(X, X, epochs=30, batch_size=64, verbose=0, shuffle=True)
        recon = ae.predict(X, verbose=0)
    except ImportError:
        logger.info("tensorflow unavailable — using PCA reconstruction fallback")
        pca = PCA(n_components=3, random_state=42).fit(X)
        recon = pca.inverse_transform(pca.transform(X))
    errors = np.mean((X - recon) ** 2, axis=1)
    return errors > np.quantile(errors, quantile)


# ------------------------------------------------------------------ reporting
def classify_anomaly(row: pd.Series) -> str:
    """Label the dominant cause of an anomalous day."""
    if row["drawdown"] <= -0.25:
        return "extreme_drawdown"
    if row["volume_ratio"] >= 3.0:
        return "volume_surge"
    if row["intraday_range"] >= 0.06 or row["vol_21"] >= 0.035:
        return "volatility_spike"
    return "unusual_return"


def detect_anomalies(
    df: pd.DataFrame,
    symbol: str = "",
    methods: tuple[str, ...] = ("isolation_forest", "dbscan", "autoencoder"),
    min_votes: int = 2,
) -> pd.DataFrame:
    """Run the detector ensemble over one stock. A day is anomalous when at
    least `min_votes` detectors agree. Returns a report frame."""
    feats = anomaly_features(df)
    if len(feats) < 100:
        return pd.DataFrame()

    votes = pd.DataFrame(index=feats.index)
    runners = {
        "isolation_forest": isolation_forest_detect,
        "dbscan": dbscan_detect,
        "autoencoder": autoencoder_detect,
    }
    for name in methods:
        try:
            votes[name] = runners[name](feats)
        except Exception as exc:
            logger.error("%s failed for %s: %s", name, symbol, exc)
            votes[name] = False

    feats["votes"] = votes.sum(axis=1).astype(int)
    feats["methods"] = votes.apply(lambda r: ",".join(c for c in votes.columns if r[c]), axis=1)
    report = feats[feats["votes"] >= min_votes].copy()
    if report.empty:
        return report
    report["type"] = report.apply(classify_anomaly, axis=1)
    report["Symbol"] = symbol
    cols = ["Date", "Symbol", "Close", "ret_1d", "volume_ratio", "drawdown",
            "vol_21", "votes", "methods", "type"]
    return report[cols].sort_values("Date", ascending=False).reset_index(drop=True)
