"""Train the classifier, and measure what the shortcut costs it.

    ../../.venv/bin/python train_classifier.py --data data/dataset.pth \
                              --model-dir model --figures figures

The classifier is a small convolutional network. The picture's 3 x 28 x 28 numbers go
through two rounds of convolution and pooling, come out as **128 numbers** in the middle
(called **``h``**, and the only part everything later is about), and those turn into 2
scores, one per class. The bigger score wins.

Training is the usual thing: show it a picture, compare its answer with the right answer,
and nudge its numbers until it is usually right. The epoch with the best validation accuracy
is the one we keep.

Two numbers matter, and the whole project is about the distance between them:

* **test** -- accuracy on more pictures from the world as it was built, where colour points
  at the right answer 95% of the time;
* **shift** -- accuracy on the same world with the colour tie **cut**: same digits, same
  nuisances, same colours, but which colour goes with which answer is now a coin flip.

The classifier is never told about the colour. If it answered from the shape alone the two
numbers would be the same. The gap between them is the part of its accuracy that was
borrowed from the colour -- and it is the number the rest of the project has to repair.

Writes:

* ``<model-dir>/classifier.pt`` -- the trained weights, with the config and the final numbers
* ``<model-dir>/history.json``   -- the loss and accuracy curves, one entry per epoch
* ``<model-dir>/metrics.json``   -- the numbers quoted in the README
* ``<figures>/training.png``     -- those curves as a picture
* ``<figures>/architecture.png`` -- what the network looks like
* ``<figures>/shortcut.png``     -- the colour we did not ask for, and what it costs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import FancyArrowPatch, Rectangle

# The shape of the network, fixed on purpose: 3 x 28 x 28 in, 128 numbers in the middle,
# 2 scores out. The README describes exactly this and no other.
N_HIDDEN = 128
N_CLASSES = 2
# The middle layer, by name, so other scripts can read it out of the trained model: the GELU
# that ends the first half of ``body``.
MIDDLE_LAYER = "body.8"

TRAIN_COLOUR = "#4c72b0"
VAL_COLOUR = "#c44e52"
INPUT_COLOUR = "#4c72b0"
HIDDEN_COLOUR = "#dd8452"
MIDDLE_COLOUR = "#c44e52"
# Marks the two halves of the picture in the shortcut figure: the shape, and the ink colour.
SHAPE_COLOUR = "#4c72b0"
INK_COLOUR = "#c44e52"


class Classifier(nn.Module):
    """A picture in, two scores out, with one small layer of numbers in the middle.

    The middle layer is what everything later is about: it is the only place where the
    network holds a summary of the picture rather than the picture itself.
    """

    def __init__(self, n_channels, n_hidden=N_HIDDEN, n_classes=N_CLASSES):
        super().__init__()
        # ``body`` is a Sequential so the middle layer has a stable name to hook.
        self.body = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x):
        return self.body(x)


def standardise(x, dataset):
    """The pictures as the classifier sees them: centred on the training statistics."""
    return (x - dataset["mean"]) / dataset["std"]


def load_split(dataset, name):
    """One split as standardised pictures and labels, both tensors."""
    index = dataset["splits"][name]
    return standardise(dataset["x"][index], dataset), dataset["y"][index]


def grayscale_split(dataset, name):
    """One split with the colour taken away: the frame painted the one grey both hues share.

    The page inside is untouched, so this is the same pictures with nothing left to read
    from the frame at all -- no hue, and no difference in brightness either. The model was
    trained on coloured pictures, so this is not a split it was built for; it is here as a
    second, independent read on how much of the answer it can get from the digit alone.
    """
    index = dataset["splits"][name]
    gray = dataset["gray"][index].unsqueeze(1).repeat(1, 3, 1, 1)
    return standardise(gray, dataset), dataset["y"][index]


def accuracy(model, x, y, batch_size=1024):
    """Fraction of examples the model gets right."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start:start + batch_size])
            correct += int((logits.argmax(dim=1) == y[start:start + batch_size]).sum())
    return correct / len(x)


def train(x, y, x_val, y_val, epochs, batch_size, lr, weight_decay, seed):
    """Train the classifier, keeping the epoch that did best on the validation split.

    The numbers are nudged with Adam on cross-entropy loss -- the standard way to say "make
    the score for the right class the biggest one". Returns the model holding the best
    epoch's weights, the per-epoch history, and which epoch that was.
    """
    torch.manual_seed(seed)
    model = Classifier(x.shape[1])
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


# --------------------------------------------------------------------------- the figures


def _picture_panel(ax, rgb, title=None, fontsize=10):
    """One picture, drawn on a white page so a pale stroke is still visible."""
    ax.imshow(np.clip(rgb, 0.0, 1.0), interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fontsize)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color="#333333"))


def _number_stack(ax, x, y0, y1, n, width, colour):
    """``n`` thin rectangles stacked vertically: one rectangle per number.

    Only the first few are drawn in full; past a point the rectangles get thinner than the
    eye can separate, so the stack is drawn as a solid block with the count written beside
    it. Either way the height is the number of numbers.
    """
    drawn = min(n, 16)
    step = (y1 - y0) / drawn
    for i in range(drawn):
        ax.add_patch(Rectangle((x, y0 + i * step + 0.08 * step), width, 0.84 * step,
                               facecolor=colour, edgecolor="white", linewidth=0.4))


def architecture_figure(rows, cols, n_hidden, out_path, dpi=150):
    """What the classifier looks like: a picture in, the middle layer, two scores out.

    Drawn from the sizes actually used, so the counts on the picture are the counts in the
    trained model.
    """
    fig, ax = plt.subplots(figsize=(11.0, 3.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.set_axis_off()

    # The picture going in.
    ax.add_patch(Rectangle((0.35, 1.75), 1.55, 1.55, facecolor="#dfe6f2",
                           edgecolor="#333333", linewidth=0.8))
    ax.text(1.12, 1.62, f"the input  x\none {rows}x{cols} picture\n"
                        f"3 colour channels", ha="center", va="top", fontsize=9)

    _arrow(ax, 2.02, 2.53, 2.42, 2.53)

    # The two convolution rounds that do the seeing.
    for i, (label, x0) in enumerate((("convolution\n+ pooling", 2.5), ("again", 3.7))):
        ax.add_patch(Rectangle((x0, 1.95), 1.0, 1.16, facecolor="#e8e8e8",
                               edgecolor="#888888", linewidth=0.8))
        ax.text(x0 + 0.5, 2.53, label, ha="center", va="center", fontsize=8.5)
    ax.text(3.6, 1.72, "the network works out what is in the picture", ha="center",
            va="top", fontsize=9)

    _arrow(ax, 4.8, 2.53, 5.2, 2.53)

    # The middle layer, one rectangle per number.
    _number_stack(ax, 5.36, 1.62, 3.44, n_hidden, 0.5, HIDDEN_COLOUR)
    ax.add_patch(Rectangle((5.24, 1.5), 0.74, 2.06, fill=False, linestyle="--",
                           linewidth=1.6, edgecolor=MIDDLE_COLOUR))
    ax.text(5.61, 1.42, f"the middle  h\n{n_hidden} numbers", ha="center", va="top",
            fontsize=9)
    ax.text(5.61, 3.62, "what we look inside", ha="center", va="bottom", fontsize=9,
            color=MIDDLE_COLOUR)

    # Two scores out, and the bigger one wins.
    for y, name in ((3.0, "score for class 0"), (2.06, "score for class 1")):
        _arrow(ax, 6.10, y, 7.40, y)
        ax.add_patch(Rectangle((7.50, y - 0.3), 2.25, 0.6, facecolor=INPUT_COLOUR,
                               alpha=0.22, edgecolor=INPUT_COLOUR))
        ax.text(8.62, y, name, ha="center", va="center", fontsize=9)
    ax.text(9.95, 2.53, "bigger score wins\n= the prediction", ha="left", va="center",
            fontsize=9)

    fig.text(
        0.5, 0.03,
        f"the classifier: two rounds of convolution and pooling, then {n_hidden} numbers in "
        f"the middle, then two scores.\n"
        "we train it on examples, and count how often the bigger score is right.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


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


def shortcut_figure(dataset, metrics, out_path, dpi=150):
    """The colour we did not ask for, and what it costs the classifier.

    Left and middle: the class-1 and class-0 average pictures. The stroke is the shape, and
    the shape averages are much alike; what jumps out is the colour, because in this data
    the ink follows the answer too.

    Right: the cost. Accuracy on the test split, where the colour points at the answer, and
    on the shift split, where the colour tie has been cut. The gap is the part of the
    accuracy that was borrowed from the colour.
    """
    index = dataset["splits"]["test"]
    x = dataset["x"][index].numpy()
    y = dataset["y"][index].numpy()
    averages = {label: x[y == label].mean(axis=0).transpose(1, 2, 0) for label in (1, 0)}
    # Put the average on a common brightness scale, so the two can be compared directly.
    limit = max(averages[1].max(), averages[0].max())

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))

    for ax, label in zip(axes[:2], (1, 0)):
        _picture_panel(ax, averages[label] / limit, f"average of class {label}")

    labels = ["colour points\nat the answer", "colour tie\ncut (shift)"]
    values = [metrics["test"], metrics["shift"]]
    bars = axes[2].bar(labels, values, color=[SHAPE_COLOUR, VAL_COLOUR], width=0.6)
    for bar, value in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                     ha="center", va="bottom", fontsize=10)
    axes[2].set_ylim(0.0, 1.06)
    axes[2].set_xlim(-0.65, 1.95)
    axes[2].set_ylabel("accuracy")
    axes[2].set_title("what it costs", fontsize=10)
    axes[2].grid(axis="y", alpha=0.3)

    # Mark the drop: the old height, and an arrow down to the new one.
    drop = values[0] - values[1]
    axes[2].plot([-0.4, 1.5], [values[0], values[0]], linestyle="--", linewidth=1.0,
                 color="#999999")
    axes[2].annotate("", xy=(1.45, values[1]), xytext=(1.45, values[0]),
                     arrowprops=dict(arrowstyle="-|>", color=VAL_COLOUR, linewidth=1.6))
    axes[2].text(1.52, (values[0] + values[1]) / 2, f"lost\n{drop:.3f}", va="center",
                 ha="left", fontsize=9.5, color=VAL_COLOUR)

    fig.text(
        0.5, 0.02,
        "left: the class-1 and class-0 average pictures. the shape averages are much alike; "
        "the colour is what stands out, because the ink\nfollows the answer.   right: what "
        "is left of the accuracy when that tie is cut.",
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
    parser.add_argument("--model-dir", default="model", help="where the model goes")
    parser.add_argument("--figures", default="figures", help="where the figures go")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset = torch.load(args.data)
    meta = dataset["meta"]

    x_train, y_train = load_split(dataset, "train")
    x_val, y_val = load_split(dataset, "val")

    model, history, best_epoch = train(
        x_train, y_train, x_val, y_val, args.epochs, args.batch_size, args.lr,
        args.weight_decay, args.seed,
    )

    # The two numbers the project is about, plus the greyscale read on the shift split.
    metrics = {}
    for name in ("train", "val", "test", "shift"):
        x, y = load_split(dataset, name)
        metrics[name] = accuracy(model, x, y)
    x_gray, y_gray = grayscale_split(dataset, "shift")
    metrics["shift_gray"] = accuracy(model, x_gray, y_gray)
    metrics["gap"] = metrics["test"] - metrics["shift"]

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"n_channels": int(x_train.shape[1]),
                   "n_hidden": N_HIDDEN, "n_classes": N_CLASSES,
                   "rows": int(meta["rows"]), "cols": int(meta["cols"])},
        "metrics": metrics,
        "best_epoch": best_epoch,
    }, model_dir / "classifier.pt")
    (model_dir / "history.json").write_text(json.dumps(history, indent=2))
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    training_figure(history, Path(args.figures) / "training.png")
    architecture_figure(meta["rows"], meta["cols"], N_HIDDEN,
                        Path(args.figures) / "architecture.png")
    shortcut_figure(dataset, metrics, Path(args.figures) / "shortcut.png")

    print(f"\ntrained for {args.epochs} epochs; the best was epoch {best_epoch}")
    print(f"  train      {metrics['train']:.3f}")
    print(f"  validation {metrics['val']:.3f}")
    print(f"  test       {metrics['test']:.3f}   (colour points at the answer)")
    print(f"  shift      {metrics['shift']:.3f}   (colour tie cut)")
    print(f"  shift, frame greyed {metrics['shift_gray']:.3f}   (colour removed altogether)")
    print(f"\n  the shortcut costs it {metrics['gap']:.3f} of accuracy")

    print(f"\nwrote {model_dir / 'classifier.pt'}")
    print(f"wrote {model_dir / 'history.json'}")
    print(f"wrote {Path(args.figures) / 'shortcut.png'}")


if __name__ == "__main__":
    main()
