"""Train the classifier.

    python train_classifier.py --data data/dataset.pth --model-dir model --figures figures/sign

The classifier is small and its shape is fixed: the picture's pixels go in, 32 numbers come
out of the middle layer, and those turn into 2 scores, one per class. The bigger score wins.

The middle layer is the one we care about, and later scripts read it by name -- see
``MIDDLE_LAYER`` below.

Training is the usual thing: show it a picture, compare its answer with the right answer,
and nudge its numbers until it is usually right. The epoch with the best validation accuracy
is the one we keep.

Writes:

* ``<model-dir>/classifier.pt`` -- the trained weights, with the config and the final numbers
* ``<model-dir>/history.json``   -- the loss and accuracy curves, one entry per epoch
* ``<figures>/training.png``     -- those curves as a picture
* ``<figures>/architecture.png`` -- what the network looks like
* ``<figures>/decoy.png``        -- the feature we missed, and what it costs the classifier
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# The shape of the network, fixed on purpose: 256 pixels in, 32 numbers in the middle,
# 2 scores out. The README describes exactly this and no other.
N_HIDDEN = 32
N_CLASSES = 2
# The middle layer, by name, so other scripts can read it out of the trained model.
MIDDLE_LAYER = "body.1"

TRAIN_COLOUR = "#4c72b0"
VAL_COLOUR = "#c44e52"
INPUT_COLOUR = "#4c72b0"
HIDDEN_COLOUR = "#dd8452"
MIDDLE_COLOUR = "#c44e52"
# Marks the two spots on the average pictures in the decoy figure: the one the label is
# about, and the one we missed.
ASKED_COLOUR = "#4c72b0"
MISSED_COLOUR = "#c44e52"
# How big to draw those rings, as a multiple of the spot width.
RING_SCALE = 1.9


class Classifier(nn.Module):
    """A picture in, two scores out, with one small layer in the middle.

    The middle layer is what everything later is about: it is the only place where the
    network holds a summary of the picture rather than the picture itself.
    """

    def __init__(self, n_pixels, n_hidden=N_HIDDEN, n_classes=N_CLASSES):
        super().__init__()
        # ``body`` is a Sequential so the middle layer has a stable name to hook.
        self.body = nn.Sequential(
            nn.Linear(n_pixels, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x):
        return self.body(x)


def load_split(dataset, name):
    """One split as standardised inputs and labels, both tensors."""
    index = dataset["splits"][name]
    x = dataset["x"][index]
    x = (x - dataset["mean"]) / dataset["std"]
    return x, dataset["y"][index]


def accuracy(model, x, y, batch_size=8192):
    """Fraction of examples the model gets right."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start:start + batch_size])
            correct += int((logits.argmax(dim=1) == y[start:start + batch_size]).sum())
    return correct / len(x)


def train(x, y, x_val, y_val, epochs, batch_size, lr, weight_decay, seed, device):
    """Train the classifier, keeping the epoch that did best on the validation split.

    The numbers are nudged with Adam on cross-entropy loss -- the standard way to say
    "make the score for the right class the biggest one".

    Returns the model holding the best epoch's weights, the per-epoch history, and which
    epoch that was.
    """
    torch.manual_seed(seed)
    model = Classifier(x.shape[1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    history = []
    best_state, best_val, best_epoch = None, -1.0, 1
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x))
        running = 0.0
        for start in range(0, len(x), batch_size):
            batch = order[start:start + batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimiser.step()
            running += loss.detach().item() * len(batch)
        schedule.step()

        val_accuracy = accuracy(model, x_val, y_val)
        history.append({
            "epoch": epoch + 1,
            "train_loss": running / len(x),
            "train_accuracy": accuracy(model, x, y),
            "val_accuracy": val_accuracy,
        })

        # Keep this epoch's weights if it is the best so far.
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, best_epoch


def training_figure(history, out_path, dpi=150):
    """Two panels: the loss coming down, and the accuracy going up."""
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train",
                 color=TRAIN_COLOUR, lw=1.8)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["train_accuracy"] for h in history], label="train",
                 color=TRAIN_COLOUR, lw=1.8)
    axes[1].plot(epochs, [h["val_accuracy"] for h in history], label="validation",
                 color=VAL_COLOUR, lw=1.8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.4, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _pixel_grid(ax, x, y, rows, cols, width, colour):
    """A ``rows x cols`` grid of small squares: one square per pixel."""
    cell = width / max(rows, cols)
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x + c * cell, y + (rows - 1 - r) * cell), cell * 0.88,
                                   cell * 0.88, facecolor=colour, edgecolor="#333333",
                                   linewidth=0.3))


def _number_stack(ax, x, y0, y1, n, width, colour):
    """``n`` thin rectangles stacked vertically: one rectangle per number."""
    step = (y1 - y0) / n
    for i in range(n):
        ax.add_patch(Rectangle((x, y0 + i * step + 0.08 * step), width, 0.84 * step,
                               facecolor=colour, edgecolor="white", linewidth=0.4))


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color="#333333"))


def architecture_figure(rows, cols, n_hidden, out_path, dpi=150):
    """What the classifier looks like: pixels in, the middle layer, two scores out.

    Drawn from the sizes actually used, so the counts on the picture are the counts in
    the trained model.
    """
    n_pixels = rows * cols
    fig, ax = plt.subplots(figsize=(11.0, 3.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.set_axis_off()

    # The picture going in, one square per pixel.
    _pixel_grid(ax, 0.35, 1.75, rows, cols, 1.55, INPUT_COLOUR)
    ax.text(1.12, 1.6, f"input  x\none {rows}x{cols} image\n({n_pixels} pixels)",
            ha="center", va="top", fontsize=9)

    _arrow(ax, 2.02, 2.53, 2.52, 2.53)

    # The middle layer, one rectangle per number.
    _number_stack(ax, 2.68, 1.62, 3.44, n_hidden, 0.5, HIDDEN_COLOUR)
    ax.add_patch(Rectangle((2.56, 1.5), 0.74, 2.06, fill=False, linestyle="--",
                           linewidth=1.6, edgecolor=MIDDLE_COLOUR))
    ax.text(2.93, 1.42, f"hidden activation  h\n{n_hidden} numbers", ha="center", va="top",
            fontsize=9)
    ax.text(2.93, 3.62, "layer analysed", ha="center", va="bottom", fontsize=9,
            color=MIDDLE_COLOUR)

    # Two scores out, and the bigger one wins.
    for y, name in ((3.0, "score for class 0"), (2.06, "score for class 1")):
        _arrow(ax, 3.42, y, 4.72, y)
        ax.add_patch(Rectangle((4.82, y - 0.3), 2.25, 0.6, facecolor=INPUT_COLOUR,
                               alpha=0.22, edgecolor=INPUT_COLOUR))
        ax.text(5.94, y, name, ha="center", va="center", fontsize=9)
    ax.text(7.3, 2.53, "argmax\n= prediction", ha="left", va="center", fontsize=9)

    fig.text(
        0.5, 0.03,
        f"Classifier with one hidden layer: {n_pixels} inputs, {n_hidden} hidden "
        f"units, two logits.\n"
        "Trained with cross-entropy; the prediction is the larger logit.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def decoy_figure(dataset, metrics, out_path, dpi=150):
    """The feature we missed, and what it costs the classifier.

    Left: the class-1 and class-0 average pictures. Two spots stand out -- the middle,
    which is the rule the label is written from, and the corner, which is the feature we
    missed -- because in this data the corner spot follows the answer too.

    Right: the cost of missing it. Accuracy on the test split, where the corner spot is
    present, against the counterfactual split, where it has been re-drawn independently.
    The gap is what the classifier was leaning on the corner spot for.
    """
    meta = dataset["meta"]
    rows, cols = meta["rows"], meta["cols"]
    positions = meta["spot_positions"]
    missed = meta["missed_spot_index"]
    # A spot of strength one is one standard deviation of ink, and should read as black,
    # the same convention the data figures use.
    limit = float(dataset["W_true"].max())

    index = dataset["splits"]["test"]
    x = dataset["x"][index].numpy()
    y = dataset["y"][index].numpy()
    averages = {label: x[y == label].mean(axis=0).reshape(rows, cols) for label in (1, 0)}

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))

    # The two averages, with the same two spots circled on both so they can be compared.
    for ax, label in zip(axes[:2], (1, 0)):
        ax.imshow(averages[label], cmap="Greys", vmin=0.0, vmax=limit,
                  interpolation="bilinear")
        for spot, colour in ((0, ASKED_COLOUR), (missed, MISSED_COLOUR)):
            spot_y, spot_x = positions[spot]
            ax.add_patch(Circle((spot_x, spot_y), RING_SCALE * meta["sigma"],
                                facecolor="none", edgecolor=colour, linewidth=1.8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"average of class {label}", fontsize=10)

    # The cost: accuracy with the corner spot, and without it.
    labels = ["correlated\ncorner spot", "decorrelated\ncorner spot"]
    values = [metrics["test"], metrics["cf"]]
    bars = axes[2].bar(labels, values, color=[ASKED_COLOUR, VAL_COLOUR], width=0.6)
    for bar, value in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                     ha="center", va="bottom", fontsize=10)
    axes[2].set_ylim(0.0, 1.06)
    axes[2].set_xlim(-0.65, 1.95)
    axes[2].set_ylabel("accuracy")
    axes[2].set_title("accuracy change", fontsize=10)
    axes[2].grid(axis="y", alpha=0.3)

    # Mark the drop: the old height, and an arrow down to the new one.
    drop = values[0] - values[1]
    axes[2].plot([-0.4, 1.5], [values[0], values[0]], linestyle="--", linewidth=1.0,
                 color="#999999")
    axes[2].annotate("", xy=(1.45, values[1]), xytext=(1.45, values[0]),
                     arrowprops=dict(arrowstyle="-|>", color=VAL_COLOUR, linewidth=1.6))
    axes[2].text(1.52, (values[0] + values[1]) / 2, f"drop\n{drop:.3f}", va="center",
                 ha="left", fontsize=9.5, color=VAL_COLOUR)

    fig.text(
        0.5, 0.02,
        "Left: class-conditional average images. Blue ring: spot 0, which defines the "
        "label. Red ring: spot 1, correlated with the label\nbut absent from the label "
        "rule. Right: accuracy on test images and on images where spot 1 is decorrelated.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Train the classifier.")
    parser.add_argument("--data", default="data/dataset.pth")
    parser.add_argument("--model-dir", default="model", help="where the trained model goes")
    parser.add_argument("--figures", default="figures/sign", help="where the figures go")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    dataset = torch.load(args.data)
    x, y = load_split(dataset, "train")
    x_val, y_val = load_split(dataset, "val")
    x, y = x.to(device), y.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    started = time.time()
    model, history, best_epoch = train(x, y, x_val, y_val, args.epochs, args.batch_size,
                                       args.lr, args.weight_decay, args.seed, device)

    # The model now holds the best epoch's weights; score those on every split.
    metrics = {}
    for name in ("train", "val", "test", "cf"):
        x_split, y_split = load_split(dataset, name)
        metrics[name] = accuracy(model, x_split.to(device), y_split.to(device))

    # The class-conditional mean strength of the two spots, the quantity the decoy figure
    # is about. Computed over the labelled splits, where the correlation is present.
    labelled = torch.cat([dataset["splits"][name]
                          for name in ("train", "val", "test")])
    a_labelled = dataset["a"][labelled]
    y_labelled = dataset["y"][labelled]
    missed = dataset["meta"]["missed_spot_index"]
    metrics["average_strength_gap_middle"] = float(
        a_labelled[y_labelled == 1, 0].mean()
        - a_labelled[y_labelled == 0, 0].mean())
    metrics["average_strength_gap_missed"] = float(
        a_labelled[y_labelled == 1, missed].mean()
        - a_labelled[y_labelled == 0, missed].mean())

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"n_pixels": int(x.shape[1]), "n_hidden": N_HIDDEN,
                   "n_classes": N_CLASSES, "middle_layer": MIDDLE_LAYER},
        "metrics": metrics,
        "best_epoch": best_epoch,
        "args": vars(args),
    }, model_dir / "classifier.pt")
    (model_dir / "history.json").write_text(json.dumps(history, indent=2))
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    training_figure(history, Path(args.figures) / "training.png")
    architecture_figure(dataset["meta"]["rows"], dataset["meta"]["cols"], N_HIDDEN,
                        Path(args.figures) / "architecture.png")
    decoy_figure(dataset, metrics, Path(args.figures) / "decoy.png")

    print(f"\ntrained {args.epochs} epochs in {time.time() - started:.1f}s")
    print(f"  kept epoch {best_epoch} (validation accuracy {metrics['val']:.4f})")
    print(f"\naccuracy")
    print(f"  {'split':<16s} {'accuracy':>10s}")
    for name in ("train", "val", "test", "cf"):
        print(f"  {name:<16s} {metrics[name]:10.4f}")
    print(f"\n  lost when the missed feature is decorrelated (test - cf): "
          f"{metrics['test'] - metrics['cf']:+.4f}")
    print(f"  class-conditional mean strength gap (spot 0 / spot {missed}): "
          f"{metrics['average_strength_gap_middle']:.3f} / "
          f"{metrics['average_strength_gap_missed']:.3f}")
    print(f"\nwrote {model_dir / 'classifier.pt'}")
    print(f"wrote {Path(args.figures) / 'training.png'}")
    print(f"wrote {Path(args.figures) / 'architecture.png'}")
    print(f"wrote {Path(args.figures) / 'decoy.png'}")


if __name__ == "__main__":
    main()
