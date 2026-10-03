"""NeuraLite: a neural network written from scratch in pure NumPy.

No autograd frameworks. Every forward and backward pass is hand-derived.
"""

from neuralite.layers import Linear, ReLU, Sigmoid, Softmax, Tanh
from neuralite.losses import CrossEntropyLoss, MSELoss
from neuralite.model import Sequential
from neuralite.optim import Adam, SGD

__version__ = "0.1.0"
__all__ = [
    "Linear", "ReLU", "Sigmoid", "Softmax", "Tanh",
    "CrossEntropyLoss", "MSELoss",
    "Sequential", "Adam", "SGD",
]
