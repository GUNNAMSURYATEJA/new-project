"""End-to-end model tests: shapes, decreasing loss, overfitting."""

import numpy as np

from neuralite import Adam, CrossEntropyLoss, Linear, ReLU, Sequential, SGD


def _blobs(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(-1.5, 0.5, (n // 2, 2))
    x1 = rng.normal(1.5, 0.5, (n - n // 2, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    return x, y


def _mlp(seed: int = 0) -> Sequential:
    return Sequential([Linear(2, 16, seed=seed), ReLU(), Linear(16, 2, seed=seed + 1)])


def test_forward_shape_and_predict():
    model = _mlp()
    x, y = _blobs()
    assert model.forward(x).shape == (60, 2)
    pred = model.predict(x)
    assert pred.shape == (60,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_loss_decreases():
    x, y = _blobs()
    model = _mlp()
    loss = CrossEntropyLoss()
    params, grads = model.parameters()
    opt = Adam(params, grads, lr=0.05)
    h = model.fit(x, y, loss=loss, optimizer=opt, epochs=30,
                  batch_size=16, seed=0, verbose=False)
    assert h["train_loss"][-1] < h["train_loss"][0]


def test_overfits_tiny_dataset():
    x, y = _blobs(n=24, seed=7)
    model = _mlp(seed=3)
    loss = CrossEntropyLoss()
    params, grads = model.parameters()
    opt = Adam(params, grads, lr=0.05)
    h = model.fit(x, y, loss=loss, optimizer=opt, epochs=300,
                  batch_size=24, seed=0, verbose=False)
    assert h["train_acc"][-1] > 0.99


def test_val_tracking():
    x, y = _blobs()
    model = _mlp()
    loss = CrossEntropyLoss()
    params, grads = model.parameters()
    opt = SGD(params, grads, lr=0.1, momentum=0.9)
    h = model.fit(x[:48], y[:48], x[48:], y[48:], loss=loss, optimizer=opt,
                  epochs=5, batch_size=16, seed=0, verbose=False)
    assert len(h["val_loss"]) == 5
    assert len(h["val_acc"]) == 5
    assert all(0.0 <= a <= 1.0 for a in h["val_acc"])
