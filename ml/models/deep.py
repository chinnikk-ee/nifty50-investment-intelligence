"""MODULE 4 — deep sequence forecasters (LSTM, Transformer).

Both expose a scikit-learn style fit/predict API over tabular feature rows.
Sequences are assembled internally from `lookback` consecutive rows, so the
models plug into the same walk-forward harness as the classical learners.
Importing this module requires TensorFlow; ml.models.registry catches the
ImportError and simply skips deep models on minimal installs.
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

tf.random.set_seed(42)


class _SequenceForecaster:
    """Shared windowing / scaling logic for sequence models."""

    def __init__(self, lookback: int = 30, epochs: int = 30, batch_size: int = 64,
                 verbose: int = 0):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_: tf.keras.Model | None = None
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._train_tail: np.ndarray | None = None

    # -- internals ---------------------------------------------------------
    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sigma

    def _windows(self, X: np.ndarray) -> np.ndarray:
        n = len(X) - self.lookback + 1
        idx = np.arange(self.lookback)[None, :] + np.arange(n)[:, None]
        return X[idx]

    def _build(self, n_features: int) -> tf.keras.Model:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- sklearn-ish API ----------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-9
        Xs = self._scale(X)
        # Each window of `lookback` rows predicts the target of its last row.
        Xw = self._windows(Xs)
        yw = y[self.lookback - 1:]
        self._train_tail = Xs[-(self.lookback - 1):] if self.lookback > 1 else Xs[:0]

        self.model_ = self._build(X.shape[1])
        self.model_.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
        self.model_.fit(
            Xw, yw,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=self.verbose,
            shuffle=False,
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("model is not fitted")
        Xs = self._scale(np.asarray(X, dtype=np.float32))
        # Prepend the training tail so the first test rows have full context.
        full = np.vstack([self._train_tail, Xs]) if len(self._train_tail) else Xs
        Xw = self._windows(full)
        preds = self.model_.predict(Xw, verbose=0).ravel()
        # One prediction per test row (the tail provides exactly lookback-1 rows).
        return preds[-len(Xs):]


class LSTMForecaster(_SequenceForecaster):
    def _build(self, n_features: int) -> tf.keras.Model:
        return models.Sequential([
            layers.Input(shape=(self.lookback, n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])


class TransformerForecaster(_SequenceForecaster):
    """Compact encoder-only transformer for time-series regression."""

    def __init__(self, lookback: int = 30, d_model: int = 32, n_heads: int = 4,
                 n_blocks: int = 2, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks

    def _build(self, n_features: int) -> tf.keras.Model:
        inp = layers.Input(shape=(self.lookback, n_features))
        x = layers.Dense(self.d_model)(inp)
        # Learned positional embedding added to the projected sequence.
        pos = layers.Embedding(self.lookback, self.d_model)(
            tf.range(start=0, limit=self.lookback, delta=1)
        )
        x = x + pos
        for _ in range(self.n_blocks):
            attn = layers.MultiHeadAttention(
                num_heads=self.n_heads, key_dim=self.d_model // self.n_heads
            )(x, x)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
            ffn = layers.Dense(self.d_model * 2, activation="relu")(x)
            ffn = layers.Dense(self.d_model)(ffn)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(1)(x)
        return models.Model(inp, out)
