"""Simulate Gray-Scott and build the prediction dataset for the emulator.

    ../../.venv/bin/python generate_data.py --out data
    ../../.venv/bin/python generate_data.py --explore --figures figures

The system
----------
Gray-Scott is a pair of chemicals on a periodic grid. Both diffuse, and they react::

    du/dt = Du * lap(u) - u*v^2 + F*(1 - u)
    dv/dt = Dv * lap(v) + u*v^2 - (F + k)*v

Four terms, and the behaviour is decided by how they balance.

* ``u*v^2`` -- the reaction, and the only source of new ``v``. It consumes ``u`` and produces
  ``v``, and it needs *two* copies of ``v`` to happen at all, so ``v`` catalyses its own
  production. Where a little ``v`` has appeared, it makes more of itself;
* ``F*(1 - u)`` -- the feed. Fresh ``u`` is supplied towards ``u = 1``, at a rate ``F``;
* ``-(F + k)*v`` -- the drain. ``v`` leaves at a rate slightly *faster* than it is fed, ``k``
  above ``F``, which is what stops the reaction from running away;
* ``Du = 0.16`` and ``Dv = 0.08`` -- the diffusion. ``v`` spreads at half the speed of ``u``,
  and that asymmetry is the whole reason there are patterns: a fast-spreading ``v`` would smear
  out before it could build up anywhere, so the reaction would never localise.

The uniform state ``u = 1, v = 0`` is a fixed point of the equations. It is stable to small
perturbations and unstable to large ones, and that threshold is the property this folder is
built on: a patch of a few cells with a few times the background ``v`` does not decay, it grows
into a self-sustaining **spot** of high ``v`` surrounded by low ``u`` -- a blob that lives on
its own, moves slowly, and can divide. A perturbation below the threshold disappears without a
trace. Whether a local nudge dies or changes the pattern forever is decided by its size, and
the whole causal experiment in this folder rests on that.

The parameters ``F`` and ``k`` choose which of those regimes the system is in, and they are not
interchangeable: the same equations give a featureless state, a stable lattice of spots, spots
that keep dividing, travelling worms, or spatiotemporal chaos. ``--explore`` walks the (F, k)
plane, reports what each pair does, and writes a contact sheet, because the choice is a
measurement rather than a preference.

This folder uses ``F = 0.022`` and ``k = 0.059``, which was picked from that sheet and then
checked over a run a hundred times longer: separate circular spots, dividing and dying, with
the coverage of the grid wandering between 0.03 and 0.15 and never filling in. The neighbours
were rejected for a reason worth writing down -- at ``(0.038, 0.066)`` the pattern reaches a
coverage of 0.155 and stays there, to three decimal places, for 60 000 steps. An emulator
trained on that would be trained to copy its input, and its sparse autoencoder would have been
asked to explain a lookup table.

The task
--------
The emulator is given the state now and the state one emulator call ago, and must produce the
state one call ahead. **One call advances the system by ``--stride`` model steps** (100 by
default). That number is the design decision that makes the project work: with a stride of 1
the network would only have to reproduce the local five-point stencil of the integrator, the
way ``--stride`` 1 would make it a learned time-stepper. With a stride of a few hundred steps,
several things happen to the pattern inside a single call, and the interesting one is that a
spot can be *born* between the input and the target. Deciding that requires more than the local
value of the two fields, which is exactly what leaves something for a sparse autoencoder to
find.

Writes ``<out>/gray_scott.npz``: the inputs, targets and raw targets of the three splits, the
standardisation statistics, the settings that produced them, and the measurements of the
pattern the dataset was drawn from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The system. Du = 2 * Dv is the standard Gray-Scott choice; Dv as large as Du leaves no
# pattern, because v has to be able to build up somewhere before it spreads away.
DU = 0.16
DV = 0.08
# Explicit Euler needs Du * dt / dx^2 <= 0.25 for the five-point stencil; 0.16 is inside that.
DT = 1.0

DEFAULT_N = 64
# The regime, picked by measurement rather than by preference: see ``--explore``. At
# (F, k) = (0.022, 0.059) the pattern is a population of separate circular spots that divide
# and die, and it stays that way -- coverage wanders between 0.03 and 0.15 of the grid over
# 60 000 steps while the number of spots moves between 20 and 39. One grid away, at
# (0.038, 0.066), the same experiment gives coverage 0.155 +- 0.000 with 23 blobs for 60 000
# steps: a frozen lattice, and an emulator whose task would be to copy its input.
DEFAULT_F = 0.022
DEFAULT_K = 0.059
# Model steps per emulator call: what the emulator has to produce in one go.
DEFAULT_STRIDE = 100
# How many consecutive recorded states the emulator is shown. Two is what the folder uses.
DEFAULT_HISTORY = 2
# How many spots the initial pattern is seeded with, on a jittered lattice.
DEFAULT_SPOTS = 25
SEED_RADIUS = 3.0

# What counts as "inside a spot" when measuring the pattern, and how close two bright cells can
# be before they are called the same blob.
SPOT_THRESHOLD = 0.25
SPOT_RADIUS = 4

# The (F, k) grid ``--explore`` walks. The corners are the well-known ones: low F is the
# chaotic corner, high F with high k is the featureless one.
EXPLORE_F = (0.014, 0.022, 0.030, 0.038, 0.046, 0.054)
EXPLORE_K = (0.045, 0.052, 0.059, 0.066)


# ------------------------------------------------------------------------------- the system


def laplacian(field):
    """The five-point stencil of the Laplacian, on a periodic grid with ``dx = 1``.

    The wrap-around at the edges is not a boundary condition to be chosen: the system is
    defined on a torus, and the integrator, the emulator and every perturbation experiment
    respect that.
    """
    return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
            - 4.0 * field)


def euler_step(u, v, F, k, dt=DT, Du=DU, Dv=DV):
    """One explicit Euler step of the Gray-Scott equations."""
    reaction = u * v * v
    u_next = u + dt * (Du * laplacian(u) - reaction + F * (1.0 - u))
    v_next = v + dt * (Dv * laplacian(v) + reaction - (F + k) * v)
    return u_next, v_next


def advance(u, v, n_steps, F, k, dt=DT):
    """Integrate forward, recording nothing."""
    for _ in range(n_steps):
        u, v = euler_step(u, v, F, k, dt)
    return u, v


def integrate(u, v, n_steps, F, k, dt=DT, sample_every=1):
    """Integrate, keeping the state every ``sample_every`` steps, the initial state first.

    Returns an array of shape ``(samples, 2, n, n)``: the two fields, stacked, so that the
    channel axis is the one thing that distinguishes ``u`` from ``v``.
    """
    if n_steps % sample_every != 0:
        raise ValueError(f"n_steps ({n_steps}) must be a multiple of sample_every "
                         f"({sample_every})")
    recorded = np.empty((n_steps // sample_every + 1, 2, *u.shape), dtype=np.float64)
    recorded[0, 0], recorded[0, 1] = u, v
    index = 1
    for step_index in range(1, n_steps + 1):
        u, v = euler_step(u, v, F, k, dt)
        if step_index % sample_every == 0:
            recorded[index, 0], recorded[index, 1] = u, v
            index += 1
    return recorded[:index]


def seed_pattern(n, n_spots, rng, radius=SEED_RADIUS):
    """A periodic grid holding ``u = 1``, ``v = 0``, with ``n_spots`` discs planted in it.

    The discs are placed on a jittered lattice rather than uniformly at random: uniform
    placement clumps, and two seeds that overlap or touch are one spot, not two. The values are
    the standard ones for turning the reaction on, ``u = 0.5`` and ``v = 0.25``.
    """
    u = np.ones((n, n))
    v = np.zeros((n, n))
    side = int(np.ceil(np.sqrt(n_spots)))
    spacing = n / side
    offset = 0.5 * spacing
    for row in range(side):
        for col in range(side):
            centre_row = (row + 0.5) * spacing + rng.uniform(-0.3, 0.3) * spacing
            centre_col = (col + 0.5) * spacing + rng.uniform(-0.3, 0.3) * spacing
            # Periodic distance from every cell to this centre.
            dy = np.minimum((np.arange(n) - centre_row) % n, (centre_row - np.arange(n)) % n)
            dx = np.minimum((np.arange(n) - centre_col) % n, (centre_col - np.arange(n)) % n)
            inside = dy[:, None] ** 2 + dx[None, :] ** 2 <= radius**2
            u[inside] = 0.50
            v[inside] = 0.25
    return u, v


# --------------------------------------------------------------------- what the pattern does


def coverage(v, threshold=SPOT_THRESHOLD):
    """The fraction of the grid that is inside a spot."""
    return float((v > threshold).mean())


def spot_centres(v, threshold=SPOT_THRESHOLD, radius=SPOT_RADIUS):
    """The middle of each blob: the brightest cell in its neighbourhood, above the threshold.

    Counting blobs exactly would mean connected components; this is a greedy non-maximum
    suppression instead -- brightest cell first, anything within ``radius`` of a centre already
    taken is the same blob -- which is less machinery and agrees with what the eye calls a
    spot.
    """
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
               if dy * dy + dx * dx <= radius * radius]
    candidates = np.argwhere(v > threshold)
    if len(candidates) == 0:
        return []
    order = np.argsort(-v[candidates[:, 0], candidates[:, 1]])
    taken = np.zeros(v.shape, dtype=bool)
    centres = []
    for index in order:
        row, col = candidates[index]
        if taken[row, col]:
            continue
        centres.append((int(row), int(col)))
        for dy, dx in offsets:
            taken[(row + dy) % v.shape[0], (col + dx) % v.shape[1]] = True
    return centres


def pattern_length(v):
    """The typical spacing between blobs, from the spatial autocorrelation.

    Radius of the first zero crossing of the radially-averaged autocorrelation, doubled: the
    autocorrelation of a lattice of blobs dips through zero half way to the next blob, so
    twice that radius is the spacing. It is the one length in the system that nothing in the
    script sets -- ``F`` and ``k`` decide it.
    """
    n = v.shape[0]
    centred = v - v.mean()
    spectrum = np.fft.fft2(centred)
    correlation = np.real(np.fft.ifft2(spectrum * np.conj(spectrum)))
    correlation = np.fft.fftshift(correlation)
    centre = n // 2
    rows, cols = np.mgrid[0:n, 0:n]
    radius = np.sqrt((rows - centre) ** 2 + (cols - centre) ** 2).astype(int)
    counts = np.bincount(radius.ravel())
    radial = np.bincount(radius.ravel(), correlation.ravel()) / np.maximum(counts, 1)
    below = np.flatnonzero(radial[1:] <= 0.0)
    return float(2.0 * (below[0] + 1)) if below.size else float(n)


def decorrelation_time(recorded, sample_every, threshold=0.5):
    """How many model steps it takes the pattern to half-forget itself, in model steps.

    The traffic that decides the stride. A pattern that is still perfectly correlated with
    itself after thousands of steps is one the emulator could predict by copying its input;
    this measures how fast the field actually turns over, one cell at a time, and returns NaN
    when it has not halved within the window at all.
    """
    signals = recorded[:, 1].reshape(len(recorded), -1)
    centred = signals - signals.mean(axis=0, keepdims=True)
    n = len(centred)
    spectrum = np.fft.rfft(centred, n=2 * n, axis=0)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), axis=0)[:n].mean(axis=1)
    correlation = correlation / correlation[0]
    below = np.flatnonzero(correlation < threshold)
    return float(below[0] * sample_every) if below.size else float("nan")


def verdict(coverage_early, coverage_late, spots_early, spots_late, half_life):
    """A one-word reading of a run: what happened to the pattern while it was integrated."""
    if coverage_late < 0.005:
        return "decayed"
    if coverage_late > 0.6:
        return "filled"
    if spots_late > 1.5 * max(spots_early, 1):
        return "multiplying"
    if np.isnan(half_life):
        return "frozen"
    return "changing"


def explore(n, n_steps, n_spots, seed, out_path, sample_every=20, dpi=150):
    """Walk the (F, k) plane, report what each pair does, and draw the contact sheet."""
    rows, final = [], {}
    for F in EXPLORE_F:
        for k in EXPLORE_K:
            u, v = seed_pattern(n, n_spots, np.random.default_rng(seed))
            recorded = integrate(u, v, n_steps, F, k, sample_every=sample_every)
            early = max(len(recorded) // 8, 1)
            coverage_early, coverage_late = coverage(recorded[early, 1]), coverage(recorded[-1, 1])
            spots_early = len(spot_centres(recorded[early, 1]))
            spots_late = len(spot_centres(recorded[-1, 1]))
            half_life = decorrelation_time(recorded, sample_every)
            rows.append({
                "F": F, "k": k,
                "coverage": coverage_late,
                "spots": spots_late,
                "spacing": pattern_length(recorded[-1, 1]),
                "half_life": half_life,
                "verdict": verdict(coverage_early, coverage_late, spots_early, spots_late,
                                   half_life),
            })
            final[(F, k)] = recorded[-1, 1]

    print(f"\n{n} x {n} cells, {n_steps} steps, {n_spots} seeds, "
          f"Du = {DU}, Dv = {DV}, dt = {DT}")
    print(f"\n  {'F':>7s}{'k':>8s}{'coverage':>10s}{'blobs':>7s}{'spacing':>9s}"
          f"{'half-life':>11s}   what happened")
    for row in rows:
        half_life = "static" if np.isnan(row["half_life"]) else f"{row['half_life']:.0f}"
        print(f"  {row['F']:7.4f}{row['k']:8.4f}{row['coverage']:10.3f}{row['spots']:7d}"
              f"{row['spacing']:9.1f}{half_life:>11s}   {row['verdict']}")

    fig, axes = plt.subplots(len(EXPLORE_F), len(EXPLORE_K),
                             figsize=(2.5 * len(EXPLORE_K), 2.5 * len(EXPLORE_F)))
    for row_index, F in enumerate(EXPLORE_F):
        for col_index, k in enumerate(EXPLORE_K):
            entry = next(row for row in rows if row["F"] == F and row["k"] == k)
            axis = axes[row_index, col_index]
            axis.imshow(final[(F, k)], cmap="inferno", vmin=0.0, vmax=0.4)
            axis.set_title(f"F={F:.3f}  k={k:.3f}\n{entry['verdict']}, {entry['spots']} blobs, "
                           f"spacing {entry['spacing']:.0f}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("which (F, k) does what: the field v after "
                 f"{n_steps} steps", fontsize=11)
    fig.text(0.5, 0.01, "every panel starts from the same seeded pattern. the regime this "
                        "folder uses is picked from this sheet, and its numbers go in the "
                        "README.", ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure_path = out_path / "regimes.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=dpi)
    plt.close(fig)

    # The table, stored: which regime this folder uses was a decision, and a decision that is
    # only in a terminal transcript is not one a reader can check.
    (out_path / "regimes.json").write_text(json.dumps(
        {"steps": int(n_steps), "cells": int(n), "seeds": int(n_spots),
         "F": list(EXPLORE_F), "k": list(EXPLORE_K), "rows": rows}, indent=2))
    print(f"\nwrote {figure_path}")
    print(f"wrote {out_path / 'regimes.json'}")
    return rows


# ------------------------------------------------------------------------------ the dataset


def build_pairs(trajectory, history=DEFAULT_HISTORY):
    """Turn a recorded trajectory into ``(inputs, targets)``.

    ``inputs[k] = trajectory[k : k + history]`` and ``targets[k] = trajectory[k + history]``,
    so the target is the state one emulator call after the last input and never inside the
    input window.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 4:
        raise ValueError(f"trajectory must be (time, 2, n, n), got {trajectory.shape}")
    n_pairs = len(trajectory) - history
    windows = np.lib.stride_tricks.sliding_window_view(trajectory, history, axis=0)
    inputs = np.moveaxis(windows[:n_pairs], -1, 1)
    return np.ascontiguousarray(inputs), trajectory[history:]


def make_dataset(n_steps, stride, n=DEFAULT_N, F=DEFAULT_F, k=DEFAULT_K, dt=DT,
                 history=DEFAULT_HISTORY, n_spots=DEFAULT_SPOTS, warm_up=2000, gap=20,
                 fractions=(0.7, 0.15, 0.15), seed=0):
    """Simulate once and split into train / validation / test blocks.

    The blocks are contiguous in time, with ``gap`` recorded samples dropped between them, so
    no validation or test state sits next to one the emulator was trained on. The gap is in
    *recorded* samples, each of which is ``stride`` model steps apart, so the default 20 is
    2000 model steps of separation.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    u, v = seed_pattern(n, n_spots, np.random.default_rng(seed))
    u, v = advance(u, v, warm_up, F, k, dt)
    trajectory = integrate(u, v, n_steps * stride, F, k, dt, sample_every=stride)

    usable = len(trajectory) - 2 * gap
    n_train = int(usable * fractions[0])
    n_val = int(usable * fractions[1])
    train_slice = trajectory[: n_train + history]
    val_slice = trajectory[n_train + gap : n_train + gap + n_val + history]
    test_slice = trajectory[n_train + gap + n_val + gap :]

    mean = np.zeros(2)
    std = np.ones(2)
    parts = {
        "train": build_pairs(train_slice, history),
        "val": build_pairs(val_slice, history),
        "test": build_pairs(test_slice, history),
    }
    # One mean and one standard deviation per field, not per cell: the system is the same
    # everywhere on the grid, so a per-cell statistic would be estimated from the very pattern
    # it is meant to summarise, and would only add noise.
    mean = parts["train"][0].mean(axis=(0, 1, 3, 4))
    std = parts["train"][0].std(axis=(0, 1, 3, 4))
    if np.any(std <= 0.0):
        raise ValueError("the training inputs are constant: the pattern decayed, and this "
                         "dataset has nothing to learn from")

    splits = {}
    for name, (raw_x, raw_y) in parts.items():
        shape = (1, 1, 2, 1, 1)
        splits[name] = {
            "inputs": ((raw_x - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32),
            "targets": ((raw_y - mean.reshape(1, 2, 1, 1))
                        / std.reshape(1, 2, 1, 1)).astype(np.float32),
            "raw_targets": raw_y.astype(np.float32),
        }

    final = trajectory[-1, 1]
    settings = {
        "N": int(n), "F": float(F), "k": float(k), "Du": float(DU), "Dv": float(DV),
        "dt": float(dt), "stride": int(stride), "history": int(history), "seed": int(seed),
        "n_samples": int(n_steps), "n_spots": int(n_spots), "warm_up": int(warm_up),
        "gap": int(gap),
        "coverage": coverage(final),
        "n_blobs": float(len(spot_centres(final))),
        "spacing": pattern_length(final),
        "half_life": decorrelation_time(trajectory, stride),
    }
    return {"splits": splits, "mean": mean, "std": std, "settings": settings}


def save_dataset(dataset, path):
    """Write the dataset as one ``.npz``, mirroring the structure of :func:`make_dataset`."""
    arrays = {}
    for name, block in dataset["splits"].items():
        for key, value in block.items():
            arrays[f"{name}_{key}"] = value
    arrays["mean"] = dataset["mean"]
    arrays["std"] = dataset["std"]
    for key, value in dataset["settings"].items():
        arrays[key] = value
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_dataset(path):
    """Read what :func:`save_dataset` wrote, as a plain dict of numpy arrays."""
    with np.load(path) as handle:
        arrays = {key: handle[key] for key in handle.files}

    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = {key: np.asarray(arrays.pop(f"{name}_{key}"), dtype=np.float32)
                        for key in ("inputs", "targets", "raw_targets")}
    integer_keys = ("N", "stride", "history", "seed", "n_samples", "n_spots", "warm_up", "gap")
    settings = {}
    for key in integer_keys + ("F", "k", "Du", "Dv", "dt", "coverage", "n_blobs", "spacing",
                               "half_life"):
        if key in arrays:
            settings[key] = int(arrays[key]) if key in integer_keys else float(arrays[key])
    return {"splits": splits,
            "mean": np.asarray(arrays["mean"], dtype=np.float64),
            "std": np.asarray(arrays["std"], dtype=np.float64),
            "settings": settings}


# ----------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(
        description="Simulate Gray-Scott and write the fixed-horizon prediction dataset.")
    parser.add_argument("--out", default="data", help="folder to write the dataset into")
    parser.add_argument("--figures", default="figures", help="folder for the contact sheet")
    parser.add_argument("--explore", action="store_true",
                        help="walk the (F, k) plane and write figures/regimes.png instead of "
                             "building the dataset")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="grid size")
    parser.add_argument("--f", type=float, default=DEFAULT_F, help="the feed rate F")
    parser.add_argument("--k", type=float, default=DEFAULT_K, help="the drain excess k")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="model steps per emulator call")
    parser.add_argument("--samples", type=int, default=1600,
                        help="recorded states in the one long run")
    parser.add_argument("--warm-up", type=int, default=2000,
                        help="steps discarded before the first recorded state")
    parser.add_argument("--gap", type=int, default=20,
                        help="recorded samples dropped between the splits")
    parser.add_argument("--spots", type=int, default=DEFAULT_SPOTS)
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--steps", type=int, default=8000, help="explore: steps per (F, k)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.explore:
        explore(args.n, args.steps, args.spots, args.seed, Path(args.figures))
        return

    print(f"\nintegrating {args.n} x {args.n} cells for "
          f"{args.warm_up + args.samples * args.stride} steps ...")
    dataset = make_dataset(args.samples, args.stride, n=args.n, F=args.f, k=args.k,
                           history=args.history, n_spots=args.spots, warm_up=args.warm_up,
                           gap=args.gap, seed=args.seed)
    path = save_dataset(dataset, Path(args.out) / "gray_scott.npz")
    settings = dataset["settings"]

    print(f"  one emulator call advances the system by {settings['stride']} steps\n")
    print(f"  {'split':<12s}{'pairs':>8s}{'shape of one input':>22s}")
    for name, block in dataset["splits"].items():
        print(f"  {name:<12s}{len(block['inputs']):8d}{str(block['inputs'].shape[2:]):>22s}")
    print(f"\n  the pattern the dataset was drawn from:")
    print(f"    coverage {settings['coverage']:.3f} of the grid, "
          f"{settings['n_blobs']:.0f} blobs, spacing about {settings['spacing']:.1f} cells")
    half_life = settings["half_life"]
    print(f"    half-forgotten after {half_life:.0f} steps, so one call at stride "
          f"{settings['stride']} is {settings['stride'] / half_life:.0%} of that")
    print(f"\n  standardisation: u {dataset['mean'][0]:+.3f} +- {dataset['std'][0]:.3f}, "
          f"v {dataset['mean'][1]:+.3f} +- {dataset['std'][1]:.3f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
