"""Synthetic datasets generated with NumPy only — no downloads."""

from __future__ import annotations

import numpy as np


def make_spirals(n_per_class: int = 700, n_classes: int = 3,
                 noise: float = 0.1, winding: float = 4.0, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Two-dimensional spiral dataset (the classic non-linear toy problem).

    Returns X of shape (n_per_class * n_classes, 2) and integer labels y.
    Each class is an arm winding outward from the origin with Gaussian noise.
    ``winding`` is the total angle (radians) each arm sweeps; 4.0 keeps the
    arms well separated so the problem is learnable but non-linear.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for k in range(n_classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        # Each arm starts at a different angle and winds outward.
        t = np.linspace(k * 2 * np.pi / n_classes,
                        k * 2 * np.pi / n_classes + winding,
                        n_per_class)
        x = r * np.cos(t) + rng.normal(0.0, noise, n_per_class)
        y = r * np.sin(t) + rng.normal(0.0, noise, n_per_class)
        xs.append(np.stack([x, y], axis=1))
        ys.append(np.full(n_per_class, k, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def train_val_split(x: np.ndarray, y: np.ndarray, val_frac: float = 0.2,
                    seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(x.shape[0])
    n_val = int(x.shape[0] * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]
