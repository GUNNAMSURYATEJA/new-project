"""Loss functions with hand-derived gradients.

CrossEntropyLoss takes raw logits (not probabilities): it applies log-softmax
internally, which is the numerically stable formulation. Pair it with a model
whose final layer is Linear — no Softmax layer needed at the output.
"""

from __future__ import annotations

import numpy as np


class MSELoss:
    """Mean squared error: mean((pred - target)^2)."""

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._pred = pred
        self._target = target
        return float(np.mean((pred - target) ** 2))

    def backward(self) -> np.ndarray:
        # dL/dpred = 2 * (pred - target) / (N * D): mean is over ALL elements.
        return 2.0 * (self._pred - self._target) / self._pred.size


class CrossEntropyLoss:
    """Softmax cross-entropy over integer class labels. Input = logits."""

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        n = logits.shape[0]
        z = logits - logits.max(axis=1, keepdims=True)
        log_sum = np.log(np.exp(z).sum(axis=1, keepdims=True))
        log_softmax = z - log_sum
        self._probs = np.exp(log_softmax)
        self._target = target
        self._n = n
        return float(-log_softmax[np.arange(n), target].mean())

    def backward(self) -> np.ndarray:
        # dL/dlogits = (softmax(logits) - onehot(target)) / N
        dlogits = self._probs.copy()
        dlogits[np.arange(self._n), self._target] -= 1.0
        return dlogits / self._n
