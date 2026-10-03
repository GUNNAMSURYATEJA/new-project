"""Gradient checks for every layer: analytic vs. numerical (tol 1e-5)."""

import numpy as np
import pytest

from neuralite import Linear, ReLU, Sigmoid, Softmax, Tanh
from neuralite.gradcheck import check_layer

rng = np.random.default_rng(0)
X = rng.standard_normal((7, 5))  # (N, D); no exact zeros for ReLU


def test_linear_shapes():
    layer = Linear(5, 3, seed=0)
    out = layer.forward(X)
    assert out.shape == (7, 3)
    dx = layer.backward(np.ones((7, 3)))
    assert dx.shape == (7, 5)
    assert layer.grads["W"].shape == (3, 5)
    assert layer.grads["b"].shape == (3,)


def test_linear_grad():
    errors = check_layer(Linear(5, 3, seed=1), X)
    assert all(e < 1e-5 for e in errors.values())


def test_relu_grad():
    errors = check_layer(ReLU(), X)
    assert all(e < 1e-5 for e in errors.values())


def test_relu_forward():
    out = ReLU().forward(np.array([[-1.0, 0.5], [2.0, -0.1]]))
    np.testing.assert_allclose(out, [[0.0, 0.5], [2.0, 0.0]])


def test_sigmoid_grad():
    errors = check_layer(Sigmoid(), X)
    assert all(e < 1e-5 for e in errors.values())


def test_sigmoid_range():
    out = Sigmoid().forward(np.array([[-1000.0, 0.0, 1000.0]]))
    assert np.all((out >= 0) & (out <= 1))
    np.testing.assert_allclose(out[0, 1], 0.5)
    np.testing.assert_allclose(out[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[0, 2], 1.0, atol=1e-12)


def test_tanh_grad():
    errors = check_layer(Tanh(), X)
    assert all(e < 1e-5 for e in errors.values())


def test_softmax_grad():
    errors = check_layer(Softmax(), X)
    assert all(e < 1e-5 for e in errors.values())


def test_softmax_rows_sum_to_one():
    out = Softmax().forward(rng.standard_normal((10, 4)) * 5)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(10), rtol=1e-12)
    assert np.all(out > 0)
