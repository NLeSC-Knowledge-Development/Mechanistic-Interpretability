"""Lay out the pictures the classifier sees: the first figure of this project.

    ../../.venv/bin/python make_fig1.py --data data/dataset.pth --out figures

Nothing is trained here. The point of the figure is to show, side by side, the two things
every picture carries: the **digit**, which is what the answer is about, and the **frame
colour**, which is not. The bottom two rows are the same eight pictures with the colour
taken away -- the world as it would look if the frame had never been painted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_data import render

COLUMNS = 4


def main():
    parser = argparse.ArgumentParser(description="Draw the data figure.")
    parser.add_argument("--data", default="data/dataset.pth")
    parser.add_argument("--out", default="figures", help="folder for the figure")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    dataset = torch.load(args.data)
    meta = dataset["meta"]
    y = dataset["y"]

    # Four examples of each class, taken from the split the world was built in.
    picks = {}
    for label in (1, 0):
        index = dataset["splits"]["test"]
        picks[label] = index[(y[index] == label)][:COLUMNS]

    # The same pictures with the colour taken away: the frame painted the one grey that both
    # hues share, and the page inside left exactly as it was.
    rows = []
    for label in (1, 0):
        index = picks[label]
        pictures = dataset["x"][index]
        # A grey frame of the same strength: same picture, no colour at all.
        a = dataset["a"][index].clone()
        a[:, 0] = 0.0
        source = dataset["source"][index].numpy()
        plain, _ = render(source, a.numpy(), meta["frame_width"])
        rows.append((f"class {label}", pictures.numpy()))
        rows.append((f"class {label}, colour removed", plain))

    fig, axes = plt.subplots(4, COLUMNS, figsize=(7.6, 8.2))
    for row, (title, pictures) in enumerate(rows):
        for column in range(COLUMNS):
            ax = axes[row, column]
            ax.imshow(np.clip(pictures[column].transpose(1, 2, 0), 0.0, 1.0),
                      interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(title, fontsize=9, rotation=0, ha="right", va="center",
                                labelpad=18)

    fig.suptitle("the pictures the classifier sees", fontsize=11)
    fig.text(
        0.5, 0.015,
        "the answer is about the digit: class 1 is 5-9, class 0 is 0-4. the frame is a "
        "feature we did not ask for, and in this data\nit follows the answer too. the "
        "bottom half of each pair of rows is the same picture with the colour taken away.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = Path(args.out) / "data.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
