"""Hand-computed single-step checks for SGD and Adam."""

import numpy as np

from neuralite import Adam, SGD


def test_sgd_no_momentum():
    p = [np.array([1.0, 2.0])]
    g = [np.array([0.1, -0.2])]
    SGD(p, g, lr=0.1, momentum=0.0).step()
    np.testing.assert_allclose(p[0], [0.99, 2.02], rtol=1e-12)


def test_sgd_momentum_two_steps():
    p = [np.array([1.0])]
    g = [np.array([1.0])]
    opt = SGD(p, g, lr=0.1, momentum=0.9)
    opt.step()  # v = -0.1 ; p = 0.9
    np.testing.assert_allclose(p[0], [0.9], rtol=1e-12)
    opt.step()  # v = 0.9*(-0.1) - 0.1 = -0.19 ; p = 0.71
    np.testing.assert_allclose(p[0], [0.71], rtol=1e-12)


def test_adam_single_step():
    p = [np.array([1.0])]
    g = [np.array([0.5])]
    Adam(p, g, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8).step()
    # m = 0.05, v = 0.00025, mhat = 0.5, vhat = 0.25
    expected = 1.0 - 0.001 * 0.5 / (np.sqrt(0.25) + 1e-8)
    np.testing.assert_allclose(p[0], [expected], rtol=1e-12)


def test_adam_moves_toward_minimum():
    # On f(p) = p^2 starting at p=3, Adam must decrease |p|.
    p = [np.array([3.0])]
    opt = Adam(p, [np.zeros(1)], lr=0.1)
    for _ in range(50):
        opt.grads[0][:] = 2 * p[0]
        opt.step()
    assert abs(p[0][0]) < 3.0
