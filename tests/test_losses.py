"""Gradient checks and sanity values for the losses."""

import numpy as np

from neuralite import CrossEntropyLoss, MSELoss
from neuralite.gradcheck import check_loss

rng = np.random.default_rng(1)


def test_mse_grad():
    pred = rng.standard_normal((8, 3))
    target = rng.standard_normal((8, 3))
    assert check_loss(MSELoss(), pred, target) < 1e-5


def test_mse_value():
    loss = MSELoss().forward(np.array([[1.0, 2.0]]), np.array([[1.0, 4.0]]))
    assert loss == 2.0  # mean([0, 4])


def test_crossentropy_grad():
    logits = rng.standard_normal((8, 4))
    target = rng.integers(0, 4, size=8)
    assert check_loss(CrossEntropyLoss(), logits, target) < 1e-5


def test_crossentropy_uniform_is_log_k():
    # Uniform logits -> each class prob 1/K -> loss = log(K).
    k = 5
    loss = CrossEntropyLoss().forward(
        np.zeros((10, k)), np.zeros(10, dtype=np.int64))
    assert abs(loss - np.log(k)) < 1e-12


def test_crossentropy_confident_is_small():
    logits = np.array([[10.0, -10.0, -10.0]])
    loss = CrossEntropyLoss().forward(logits, np.array([0]))
    assert loss < 1e-6
