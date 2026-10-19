"""Build figure 1: what we are classifying.

    python make_fig1.py --out figures/sign

Reads the per-example PNGs that ``generate_data.py`` wrote into the data folder and lays
them out in two rows:

* **class 1** -- the centre spot is on;
* **class 0** -- the centre spot is off.

On the first picture of each row a circle marks the centre spot, and the question the
whole task comes down to is written next to it: is it on? The label down the left-hand
side of each row is just its class.

Writes ``data.png`` into ``--out``.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle

CLASS_COLOUR = {1: "#c44e52", 0: "#4c72b0"}
# How big to draw the circle, as a multiple of the width of the spot it surrounds.
RING_SCALE = 1.7
# Red, so it shows up both against the white page and against the black spot it circles.
RING_COLOUR = "#c44e52"


def example_pngs(pngs_dir, label, split="test", n=4):
    """The first ``n`` example pictures of this class in the given split."""
    paths = sorted(glob.glob(str(Path(pngs_dir) / f"class{label}" / f"{split}_*.png")))
    if not paths:
        raise FileNotFoundError(
            f"no pictures for class {label} in {pngs_dir}/class{label} "
            f"-- run generate_data.py first, or pass --pngs"
        )
    return paths[:n]


def main():
    parser = argparse.ArgumentParser(description="Build figure 1: the data we classify.")
    parser.add_argument("--data", default="data/dataset.pth",
                        help="the saved dataset, for where the centre spot is")
    parser.add_argument("--pngs", default="data/pngs", help="folder of per-example PNGs")
    parser.add_argument("--out", default="figures/sign", help="folder to write data.png into")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "cf"])
    parser.add_argument("--n", type=int, default=4, help="examples per class")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    meta = torch.load(args.data)["meta"]
    # Spot 0 is always the centre of the picture; the positions are (y, x) pixels of the
    # 16x16 grid the pictures are built on.
    centre_y, centre_x = meta["spot_positions"][0]

    rows = [(1, "class 1"), (0, "class 0")]
    fig, axes = plt.subplots(len(rows), args.n, figsize=(10.2, 6.6))

    ring = None  # the centre spot as (x, y, radius) in the pixels of the picture
    for row, (label, name) in zip(axes, rows):
        for column, (ax, path) in enumerate(zip(row, example_pngs(args.pngs, label,
                                                                 args.split, args.n))):
            picture = mpimg.imread(path)
            ax.imshow(picture, interpolation="bilinear")
            # The pictures are stored larger than the grid they show, so the grid position
            # of the spot has to be scaled up to the pixels we are drawing on.
            height, width = picture.shape[:2]
            scale_x, scale_y = width / meta["cols"], height / meta["rows"]
            x_px = (centre_x + 0.5) * scale_x
            y_px = (centre_y + 0.5) * scale_y
            radius_px = RING_SCALE * meta["sigma"] * (scale_x + scale_y) / 2
            # Circle the centre spot on the first picture of the row, and leave the rest
            # unmarked so the pictures can be looked at as they are.
            if column == 0:
                ax.add_patch(Circle((x_px, y_px), radius_px, facecolor="none",
                                    edgecolor=RING_COLOUR, linewidth=1.6))
            if ring is None:
                ring = (x_px, y_px, radius_px)
            ax.set_xticks([])
            ax.set_yticks([])
        # One name per row, down the left-hand side.
        row[0].set_ylabel(name, fontsize=10.5, color=CLASS_COLOUR[label],
                          fontweight="bold", labelpad=10, va="center", linespacing=1.5)

    # Ask the question once, next to the spot it is about.
    ring_x, ring_y, ring_radius = ring
    axes[0][0].annotate(
        "spot 0 active?", xy=(ring_x, ring_y - ring_radius), xycoords="data",
        xytext=(0.5, 1.02), textcoords="axes fraction", ha="center", va="bottom",
        fontsize=9, color="#333333",
        arrowprops=dict(arrowstyle="-|>", color="#333333", linewidth=0.9),
    )

    fig.text(
        0.5, 0.012,
        "Label rule: class 1 if spot 0 (centre) is active, class 0 otherwise. The circle "
        "marks spot 0 on the first image of each row.\nDark pixels indicate ink; white "
        "indicates no ink. The other active spots are drawn from the 25-spot lattice.",
        ha="center", va="bottom", fontsize=9, color="#333333",
    )
    fig.tight_layout(rect=(0.105, 0.075, 1, 0.95))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "data.png", dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {out_dir / 'data.png'}")


if __name__ == "__main__":
    main()
