"""Generate the sandbox dataset: small pictures of spots of ink.

    python generate_data.py --out data

The whole point of this folder is that we know the answer. We do not start from a
mystery network: we build a world in which the "features" are ours, we write down which
one the label depends on, and only then do we look for it through a model.

The world is a picture. Every example is a small image, and every feature is a **spot of
ink** in a fixed place on the grid:

* there are 25 spots, laid out on a full 5x5 lattice inset from the border;
* an example is made by switching a few spots on, with random strengths, and adding them
  up. Every spot is ink: it is either off, and lays down nothing, or on with a positive
  strength. A picture is therefore white where nothing is happening, and dark where a spot
  is.

Two spots have a fixed meaning, so that index 0 and index 1 mean the same thing in every
run and in every figure:

* spot **0**, in the middle of the image, is the spot the label is about;
* spot **1**, in the top-left corner, is the **missed feature** for the ``sign`` task: a
  real spot in the data, correlated with the label, that the rule we wrote down does not
  mention. (For ``conjunction`` the corner is already part of the label, so the missed
  feature moves to the next spot, index 2.)

The correlation is ours to set. ``--correlation c`` makes that spot appear with the label
``0.5 + 0.5 * c`` of the time, so ``0.99`` gives agreement 99.5% of the time where chance
alone would give 50%. ``--correlation 0`` leaves its appearance independent of the label,
which is also how the counterfactual split is built.

Two tasks are provided:

``sign``
    class 1 iff spot 0 is on.
``conjunction``
    class 1 iff spot 0 **and** spot 1 are both on. No straight line through pixel
    space separates this.

Everything is written to ``--out``: one ``dataset.pth`` holding the pictures, the labels,
the *true* spot strengths and the spot templates, plus one PNG per example under
``pngs/`` so the data can be looked at directly. The pictures are stored **raw**, together
with the statistics needed to standardise them, so the PNGs show the real thing and a
later training step can standardise however it likes. Numbers go in as tensors and load
back with plain ``torch.load``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

DEFAULT_ROWS = 16
DEFAULT_COLS = 16
# Inset of the lattice from the image border, measured in pixels.
DEFAULT_MARGIN = 2.0
# The lattice is 5x5, so all 25 of the 25 possible spots are used.
DEFAULT_POINTS = 5
# Spot width as a fraction of the distance between neighbouring spots. Below 0.5 the
# spots read as separate objects; above it they start to merge.
DEFAULT_SIGMA_FRACTION = 0.52
# Probability that one of the spots the label is about is on at all, and so lays down ink.
DEFAULT_P_LABEL = 0.5
DEFAULT_P_ACTIVE = 0.2
DEFAULT_NOISE = 0.05
DEFAULT_CORRELATION = 0.99
TASKS = ("sign", "conjunction")


# --------------------------------------------------------------------------- the spots


def lattice(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, margin=DEFAULT_MARGIN,
            points=DEFAULT_POINTS):
    """The 5x5 grid of candidate spot positions, as ``(ys, xs)`` pixel coordinates."""
    ys = np.linspace(margin, rows - 1.0 - margin, points)
    xs = np.linspace(margin, cols - 1.0 - margin, points)
    return ys, xs


def default_sigma(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, margin=DEFAULT_MARGIN,
                  points=DEFAULT_POINTS, fraction=DEFAULT_SIGMA_FRACTION):
    """Spot width scaled to the lattice spacing, so a picture looks the same at any
    resolution."""
    ys, xs = lattice(rows, cols, margin, points)
    spacing = min(ys[1] - ys[0], xs[1] - xs[0])
    return float(spacing * fraction)


def blob(rows, cols, centre, sigma):
    """A smooth spot of ink, flattened into a vector of length ``rows * cols``."""
    yy, xx = np.mgrid[0:rows, 0:cols]
    y, x = centre
    spot = np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma**2)))
    return spot.ravel()


def spot_positions(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, margin=DEFAULT_MARGIN,
                   points=DEFAULT_POINTS):
    """The ordered list of all 25 spot positions.

    Index 0 is always the middle of the image and index 1 is always the top-left corner,
    so that "spot 0" and "spot 1" mean the same thing in every task and every run.
    """
    ys, xs = lattice(rows, cols, margin, points)
    centre = (float(ys[points // 2]), float(xs[points // 2]))
    corner = (float(ys[0]), float(xs[0]))
    everything = [(float(y), float(x)) for y in ys for x in xs]
    ordered = [centre, corner] + [p for p in everything if p != centre and p != corner]
    return ordered


def feature_directions(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, sigma=None,
                       margin=DEFAULT_MARGIN, points=DEFAULT_POINTS):
    """The spot templates, one row per spot, each unit-norm.

    Shape ``(points**2, rows * cols)`` -- 25 rows by default.
    """
    if sigma is None:
        sigma = default_sigma(rows, cols, margin, points)
    positions = spot_positions(rows, cols, margin, points)
    directions = np.stack([blob(rows, cols, p, sigma) for p in positions])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / norms


def n_label_spots(task):
    """How many spots the label is about."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    return 1 if task == "sign" else 2


def missed_spot_index(task):
    """Index of the missed feature: the spot right after the ones the label is about.

    For ``sign`` that is the corner spot (1). For ``conjunction`` the corner is already
    part of the label, so the missed feature is the next spot (2).
    """
    return n_label_spots(task)


def make_labels(a, task):
    """The ground-truth label function, applied to the true spot strengths."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    if task == "sign":
        return (a[:, 0] > 0).astype(np.int64)
    return ((a[:, 0] > 0) & (a[:, 1] > 0)).astype(np.int64)


# ------------------------------------------------------------------------ the pictures


def draw(n, W_true, task, rng, p_active=DEFAULT_P_ACTIVE, noise=DEFAULT_NOISE,
         correlation=DEFAULT_CORRELATION, p_label=DEFAULT_P_LABEL):
    """Draw ``n`` examples from the world.

    Parameters
    ----------
    n : int
        Number of examples.
    W_true : ndarray, shape (n_spots, rows*cols)
        The unit-norm spot templates.
    task : str
        ``"sign"`` or ``"conjunction"``.
    rng : numpy.random.Generator
        Random source.
    p_active : float
        Probability that a background spot is switched on.
    noise : float
        Pixel noise standard deviation.
    correlation : float
        How often the missed feature's appearance agrees with the label: it is on
        ``0.5 + 0.5 * correlation`` of the time. ``0`` leaves it independent of the label.
    p_label : float
        Probability that one of the spots the label is about is on, and lays down ink.

    Returns
    -------
    x : ndarray, shape (n, rows*cols)
        Pictures, flattened, before standardisation.
    y : ndarray, shape (n,)
        Labels.
    a : ndarray, shape (n, n_spots)
        True spot strengths.
    """
    n_spots, d_model = W_true.shape
    missed = missed_spot_index(task)
    n_label = n_label_spots(task)

    a = np.zeros((n, n_spots))
    # The spots the label is about: each one is on with probability ``p_label``, and lays
    # down a positive amount of ink.
    on = rng.random((n, n_label)) < p_label
    a[:, :n_label] = on * np.abs(rng.normal(size=(n, n_label)))

    y = make_labels(a, task)

    # Tie the missed feature's appearance to the label. The label is computed from the
    # label spots only, so this changes nothing causal about y -- it only makes that spot
    # predictive on this distribution. At ``correlation = 0`` its presence is independent
    # of the label, which is how the counterfactual split is built.
    p_missed = 0.5 + 0.5 * correlation * np.where(y > 0, 1.0, -1.0)
    a[:, missed] = (rng.random(n) < p_missed) * np.abs(rng.normal(size=n))

    # Sparse background spots, so a picture is a genuine mixture of several overlapping
    # spots rather than a single clean object.
    n_background = n_spots - missed - 1
    active = rng.random((n, n_background)) < p_active
    a[:, missed + 1:] = active * np.abs(rng.normal(size=(n, n_background)))

    x = a @ W_true + noise * rng.normal(size=(n, d_model))
    return x, y, a


def build_dataset(n_train, n_val, n_test, n_cf, rows, cols, task, sigma, p_active,
                  noise, correlation, seed, p_label=DEFAULT_P_LABEL):
    """Draw the four splits and standardise with the train statistics only.

    The counterfactual split ``cf`` keeps the same label function and the same
    distribution of everything else, but **breaks the correlation between the missed
    feature and the label**. A model that learned the spot the label is about keeps its
    accuracy there; a model that leaned on the missed feature does not. That gap is a
    direct, assumption-free read on what a classifier is using.
    """
    W_true = feature_directions(rows=rows, cols=cols, sigma=sigma)
    rng = np.random.default_rng(seed + 1)

    counts = {"train": n_train, "val": n_val, "test": n_test, "cf": n_cf}
    x_parts, y_parts, a_parts = [], [], []
    for name, n in counts.items():
        strength = correlation if name != "cf" else 0.0
        x, y, a = draw(n, W_true, task, rng, p_active, noise, strength, p_label)
        x_parts.append(x)
        y_parts.append(y)
        a_parts.append(a)

    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    a = np.concatenate(a_parts)

    # Split indices into the one big array, so every split lives side by side.
    splits, start = {}, 0
    for name, n in counts.items():
        splits[name] = np.arange(start, start + n)
        start += n

    mean = x[splits["train"]].mean(axis=0)
    std = x[splits["train"]].std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    meta = {
        "rows": int(rows),
        "cols": int(cols),
        "sigma": float(sigma if sigma is not None else default_sigma(rows, cols)),
        "task": task,
        "correlation": float(correlation),
        "p_label": float(p_label),
        "p_active": float(p_active),
        "noise": float(noise),
        "seed": int(seed),
        "n_label_spots": int(n_label_spots(task)),
        "missed_spot_index": int(missed_spot_index(task)),
        "spot_positions": [[float(yy), float(xx)]
                           for yy, xx in spot_positions(rows, cols)],
        "counts": {name: int(n) for name, n in counts.items()},
    }

    return {
        "x": x,
        "y": y,
        "a": a,
        "W_true": W_true,
        "splits": splits,
        "mean": mean,
        "std": std,
        "meta": meta,
    }


# --------------------------------------------------------------------------- the files


def to_tensors(dataset):
    """Package the dataset for ``torch.save``: numbers as tensors, metadata as plain Python.

    ``torch.load`` defaults to ``weights_only=True`` since torch 2.6, and that refuses
    numpy arrays. Converting here means the file loads with no special arguments, while
    the metadata stays readable without unpickling anything unusual.
    """
    return {
        "x": torch.as_tensor(dataset["x"], dtype=torch.float32),
        "y": torch.as_tensor(dataset["y"], dtype=torch.int64),
        "a": torch.as_tensor(dataset["a"], dtype=torch.float32),
        "W_true": torch.as_tensor(dataset["W_true"], dtype=torch.float32),
        "splits": {name: torch.as_tensor(index, dtype=torch.int64)
                   for name, index in dataset["splits"].items()},
        "mean": torch.as_tensor(dataset["mean"], dtype=torch.float32),
        "std": torch.as_tensor(dataset["std"], dtype=torch.float32),
        "meta": dataset["meta"],
    }


def save_pngs(dataset, out_dir, dpi=150):
    """Write one PNG per example, so the data can be looked at directly.

    The page is white and the spots are black, the same convention the figures in this
    folder use.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = dataset["x"], dataset["y"]
    rows, cols = dataset["meta"]["rows"], dataset["meta"]["cols"]
    # A spot of strength one is one standard deviation of ink, and should read as black.
    # Pinning the scale there, rather than to the rarest pixel in the whole set, keeps an
    # ordinary spot dark instead of washing every picture out.
    limit = float(dataset["W_true"].max())

    written = 0
    for split, index in dataset["splits"].items():
        for i in index:
            picture = x[i].reshape(rows, cols)
            folder = out_dir / f"class{int(y[i])}"
            folder.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.imshow(picture, cmap="Greys", vmin=0.0, vmax=limit,
                      interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.subplots_adjust(0, 0, 1, 1)
            fig.savefig(folder / f"{split}_{i:05d}.png", dpi=dpi)
            plt.close(fig)
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Generate the spots-of-ink dataset and one PNG per example."
    )
    parser.add_argument("--out", default="data", help="folder to write into")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-cf", type=int, default=200,
                        help="counterfactual examples, with the missed feature decorrelated")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--task", default="sign", choices=list(TASKS))
    parser.add_argument("--sigma", type=float, default=None,
                        help="spot width in pixels; defaults to the lattice spacing")
    parser.add_argument("--p-label", type=float, default=DEFAULT_P_LABEL,
                        help="probability that a spot the label is about is on")
    parser.add_argument("--p-active", type=float, default=DEFAULT_P_ACTIVE)
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    parser.add_argument("--correlation", type=float, default=DEFAULT_CORRELATION,
                        help="how strongly the missed feature agrees with the label; "
                             "0 leaves it independent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-png", action="store_true",
                        help="save only the dataset, not the pictures")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, n_cf=args.n_cf,
        rows=args.rows, cols=args.cols, task=args.task, sigma=args.sigma,
        p_active=args.p_active, noise=args.noise, correlation=args.correlation,
        seed=args.seed, p_label=args.p_label,
    )

    torch.save(to_tensors(dataset), out_dir / "dataset.pth")

    if not args.no_png:
        written = save_pngs(dataset, out_dir / "pngs", dpi=args.dpi)
    else:
        written = 0

    meta = dataset["meta"]
    missed = meta["missed_spot_index"]
    a, y = dataset["a"], dataset["y"]

    print(f"\nwrote {out_dir / 'dataset.pth'}")
    print(f"  {meta['rows']}x{meta['cols']} pictures, "
          f"{len(meta['spot_positions'])} spots, task '{meta['task']}'")
    for name, n in meta["counts"].items():
        print(f"  {name:<6s} {n:5d} examples")
    correlated = np.concatenate([dataset["splits"][n] for n in ("train", "val", "test")])
    cf = dataset["splits"]["cf"]
    # The missed feature is on exactly when the label is 1, ``0.5 + 0.5 * correlation``
    # of the time where the correlation was built in and half the time when it is not.
    agree = (a[:, missed] > 0) == (y > 0)
    print(f"\n  spot 0 (the middle) decides the answer")
    print(f"  spot {missed} (the missed feature) appears with the label "
          f"{agree[correlated].mean():.3f} of the time where the correlation was built in,")
    print(f"                                             "
          f"{agree[cf].mean():.3f} of the time in the counterfactual split")
    if written:
        print(f"\nwrote {written} pictures to {out_dir / 'pngs'}")


if __name__ == "__main__":
    main()
