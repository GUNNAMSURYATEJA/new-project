"""NeuraLite demo: train a 2-hidden-layer MLP on the spiral dataset from scratch.

Prints train/val accuracy and saves a decision-boundary plot + loss curve
to examples/. Target: under 5 minutes on CPU.
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuralite import Adam, CrossEntropyLoss, Linear, ReLU, Sequential
from neuralite.data import make_spirals, train_val_split


def main() -> None:
    t0 = time.time()
    print("=== NeuraLite demo: MLP on 3-arm spirals ===\n")

    x, y = make_spirals(n_per_class=700, n_classes=3, seed=0)
    x_train, y_train, x_val, y_val = train_val_split(x, y, val_frac=0.2, seed=0)
    print(f"train: {x_train.shape[0]} points | val: {x_val.shape[0]} points")

    model = Sequential([
        Linear(2, 64, seed=0), ReLU(),
        Linear(64, 64, seed=1), ReLU(),
        Linear(64, 3, seed=2),
    ])
    loss = CrossEntropyLoss()
    params, grads = model.parameters()
    opt = Adam(params, grads, lr=1e-2)

    history = model.fit(x_train, y_train, x_val, y_val, loss=loss,
                        optimizer=opt, epochs=400, batch_size=64,
                        seed=0, verbose=True)

    train_acc = history["train_acc"][-1]
    val_acc = history["val_acc"][-1]
    print(f"\nfinal train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    outdir = Path(__file__).resolve().parent.parent / "examples"
    outdir.mkdir(exist_ok=True)

    # Decision boundary on a grid.
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 200),
                         np.linspace(-1.2, 1.2, 200))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    zz = model.predict(grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].contourf(xx, yy, zz, alpha=0.35, cmap="viridis")
    axes[0].scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", s=8, edgecolors="none")
    axes[0].set_title("Decision boundary (3-arm spirals)")
    axes[0].set_aspect("equal")

    axes[1].plot(history["train_loss"], label="train loss")
    axes[1].plot(history["val_loss"], label="val loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy")
    axes[1].set_title("Loss curves")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / "spiral_results.png", dpi=120)
    print(f"saved {outdir / 'spiral_results.png'}")
    print(f"\ndemo wall-clock: {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
