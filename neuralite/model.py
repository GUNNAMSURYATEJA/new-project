"""Sequential model with a minibatch training loop."""

from __future__ import annotations

import numpy as np

from neuralite.layers import Module


class Sequential:
    def __init__(self, layers: list[Module]) -> None:
        self.layers = layers
        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

    # -- plumbing ------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def parameters(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        params, grads = [], []
        for layer in self.layers:
            for k in layer.params:
                params.append(layer.params[k])
                grads.append(layer.grads[k])
        return params, grads

    # -- evaluation ----------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)

    @staticmethod
    def accuracy(logits: np.ndarray, target: np.ndarray) -> float:
        return float((logits.argmax(axis=1) == target).mean())

    # -- training ------------------------------------------------------
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
            loss=None, optimizer=None, epochs: int = 100, batch_size: int = 64,
            seed: int = 0, verbose: bool = True) -> dict[str, list[float]]:
        if loss is None or optimizer is None:
            raise ValueError("fit() requires a loss and an optimizer")
        rng = np.random.default_rng(seed)
        n = x_train.shape[0]

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = x_train[idx], y_train[idx]
                self.zero_grad()
                logits = self.forward(xb)
                l = loss.forward(logits, yb)
                self.backward(loss.backward())
                optimizer.step()

            # End-of-epoch metrics on the full sets (no grad needed here).
            train_logits = self.forward(x_train)
            train_loss = loss.forward(train_logits, y_train)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(self.accuracy(train_logits, y_train))
            if x_val is not None:
                val_logits = self.forward(x_val)
                self.history["val_loss"].append(loss.forward(val_logits, y_val))
                self.history["val_acc"].append(self.accuracy(val_logits, y_val))

            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0
                            or epoch == epochs):
                msg = (f"epoch {epoch:4d}/{epochs}  train_loss={train_loss:.4f} "
                       f"train_acc={self.history['train_acc'][-1]:.4f}")
                if x_val is not None:
                    msg += (f"  val_loss={self.history['val_loss'][-1]:.4f} "
                            f"val_acc={self.history['val_acc'][-1]:.4f}")
                print(msg)
        return self.history
