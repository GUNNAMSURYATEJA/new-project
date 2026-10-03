"""Layers with hand-derived forward and backward passes.

All layers operate on batches: X has shape (N, D). Each layer caches what its
backward pass needs; nothing here relies on autograd.
"""

from __future__ import annotations

import numpy as np


class Module:
    """Base class: holds parameters and their gradients."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def zero_grad(self) -> None:
        # Fill in place (never rebind): optimizers hold references to these
        # arrays, so rebinding would leave them pointing at stale gradients.
        for k, p in self.params.items():
            if k not in self.grads:
                self.grads[k] = np.zeros_like(p)
            else:
                self.grads[k][:] = 0


class Linear(Module):
    """y = x @ W.T + b. He initialization by default."""

    def __init__(self, in_dim: int, out_dim: int, seed: int | None = None) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.params["W"] = rng.standard_normal((out_dim, in_dim)) * np.sqrt(2.0 / in_dim)
        self.params["b"] = np.zeros(out_dim)
        self.zero_grad()
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.params["W"].T + self.params["b"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self._x
        assert x is not None
        # dL/dW = dout^T @ x ; dL/db = sum over batch ; dL/dx = dout @ W
        # Accumulate (+=): zero_grad() is responsible for clearing.
        self.grads["W"] += dout.T @ x
        self.grads["b"] += dout.sum(axis=0)
        return dout @ self.params["W"]


class ReLU(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return np.where(self._mask, x, 0.0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask


class Sigmoid(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Stable formulation without overflow: branch on the sign so that
        # exp() is only ever evaluated on non-positive arguments.
        # (np.where would evaluate both branches eagerly and overflow.)
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        self._out = out
        return self._out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        s = self._out
        return dout * s * (1.0 - s)


class Tanh(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        t = self._out
        return dout * (1.0 - t * t)


class Softmax(Module):
    """Row-wise softmax with the max-subtraction trick for stability."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x - x.max(axis=1, keepdims=True)
        e = np.exp(z)
        self._out = e / e.sum(axis=1, keepdims=True)
        return self._out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # dL/dx_i = s_i * (dout_i - sum_j dout_j * s_j), applied per row.
        s = self._out
        return s * (dout - (dout * s).sum(axis=1, keepdims=True))
