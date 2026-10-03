"""Optimizers written from scratch: SGD (with momentum) and Adam."""

from __future__ import annotations

import numpy as np


class SGD:
    """Stochastic gradient descent with optional momentum.

    v <- mu * v - lr * grad ; p <- p + v   (v init 0)
    With mu=0 this is vanilla SGD: p <- p - lr * grad.
    """

    def __init__(self, params: list[np.ndarray], grads: list[np.ndarray],
                 lr: float = 1e-2, momentum: float = 0.0) -> None:
        self.params = params
        self.grads = grads
        self.lr = lr
        self.momentum = momentum
        self._v = [np.zeros_like(p) for p in params]

    def step(self) -> None:
        for p, g, v in zip(self.params, self.grads, self._v):
            v[:] = self.momentum * v - self.lr * g
            p += v

    def zero_grad(self, modules) -> None:
        for m in modules:
            m.zero_grad()


class Adam:
    """Adam (Kingma & Ba, 2014) with bias correction.

    m <- b1*m + (1-b1)*g ; v <- b2*v + (1-b2)*g^2
    mhat = m/(1-b1^t) ; vhat = v/(1-b2^t)
    p <- p - lr * mhat / (sqrt(vhat) + eps)
    """

    def __init__(self, params: list[np.ndarray], grads: list[np.ndarray],
                 lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8) -> None:
        self.params = params
        self.grads = grads
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = [np.zeros_like(p) for p in params]
        self._v = [np.zeros_like(p) for p in params]
        self._t = 0

    def step(self) -> None:
        self._t += 1
        b1t = 1.0 - self.beta1 ** self._t
        b2t = 1.0 - self.beta2 ** self._t
        for p, g, m, v in zip(self.params, self.grads, self._m, self._v):
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * g * g
            mhat = m / b1t
            vhat = v / b2t
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self, modules) -> None:
        for m in modules:
            m.zero_grad()
