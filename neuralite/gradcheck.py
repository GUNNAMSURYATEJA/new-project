"""Numerical gradient checking: the correctness backbone of NeuraLite.

Compares hand-derived analytic gradients against central finite differences.
Target: relative error below 1e-5. If a check fails, the backward pass is
wrong — do not loosen the tolerance, fix the math.
"""

from __future__ import annotations

import numpy as np


def rel_error(a: np.ndarray, n: np.ndarray) -> float:
    """Max relative error, robust to near-zero gradients."""
    denom = np.maximum(1.0, np.abs(a) + np.abs(n))
    return float(np.max(np.abs(a - n) / denom))


def numerical_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central differences: (f(x+e) - f(x-e)) / 2e, elementwise."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        ix = it.multi_index
        old = x[ix]
        x[ix] = old + eps
        fp = f(x)
        x[ix] = old - eps
        fm = f(x)
        x[ix] = old
        grad[ix] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return grad


def check_layer(layer, x: np.ndarray, eps: float = 1e-5,
                tol: float = 1e-5, seed: int = 0) -> dict[str, float]:
    """Grad-check a layer's parameter grads and input grads.

    Scalar objective: weighted sum of the layer's output with fixed random
    weights (a plain sum would be vacuous for Softmax, whose rows sum to 1).
    Returns the max relative error per checked quantity; raises
    AssertionError if any exceeds tol.
    """
    rng = np.random.default_rng(seed)
    x = x.astype(np.float64)
    out = layer.forward(x)
    w = rng.standard_normal(out.shape)  # fixed random projection
    dx_analytic = layer.backward(w).astype(np.float64)

    dx_numeric = numerical_grad(lambda xx: float((layer.forward(xx) * w).sum()),
                                x.copy(), eps)
    errors = {"dx": rel_error(dx_analytic, dx_numeric)}

    for name, p in layer.params.items():
        p64 = p.astype(np.float64)
        layer.params[name] = p64
        layer.zero_grad()
        layer.backward(w)
        ga = layer.grads[name].astype(np.float64)

        def f_param(pp, _layer=layer, _name=name, _x=x, _w=w):
            _layer.params[_name] = pp
            return float((_layer.forward(_x) * _w).sum())

        gn = numerical_grad(f_param, p64.copy(), eps)
        errors[f"d{name}"] = rel_error(ga, gn)
        layer.params[name] = p  # restore original dtype/values

    for name, err in errors.items():
        assert err < tol, f"{type(layer).__name__} grad check failed for {name}: rel_error={err:.2e}"
    return errors


def check_loss(loss, pred: np.ndarray, target: np.ndarray,
               eps: float = 1e-5, tol: float = 1e-5) -> float:
    """Grad-check a loss's gradient w.r.t. its predictions."""
    pred = pred.astype(np.float64)
    loss.forward(pred, target)
    g_analytic = loss.backward().astype(np.float64)
    g_numeric = numerical_grad(lambda pp: loss.forward(pp, target), pred.copy(), eps)
    err = rel_error(g_analytic, g_numeric)
    assert err < tol, f"{type(loss).__name__} grad check failed: rel_error={err:.2e}"
    return err
