"""Draw the system: the pattern, and the threshold that decides everything.

    ../../.venv/bin/python make_fig1.py --data data/gray_scott.npz --out figures

Nothing is trained here. The figure is the introduction: what the pattern looks like, what the
two chemicals do, and the one property of this system that the rest of the folder depends on.

That property is the **threshold**. The uniform state -- no spots anywhere, ``u = 1`` and
``v = 0`` -- is a fixed point of the equations, and it is stable: a small enough local
disturbance is absorbed and leaves no trace. The same disturbance a little larger does not
decay at all. It grows into a spot, a blob of high ``v`` ringed by low ``u`` that feeds itself
and stays there. Which one happens is decided by the size of the nudge and by nothing else, and
the bottom panel measures it: every disc of each radius and amplitude is run for 3000 steps, and
what is drawn is the share of the grid that ends up covered by spots -- nothing at all for one
that was absorbed, a definite patch for one that survived.

Everything here is measured rather than asserted, and it is the reason the causal experiment in
``perturb.py`` can have a yes-or-no answer instead of a small number. A push into this system
either does nothing, or it changes the pattern permanently.

There is a second measurement in the same direction, and it explains why the seed is fixed and
checked below: the system is **bistable**. Seed the same arrangement of discs at seeds 0, 1, 2,
3 and 4 -- same recipe, same number of discs, only the jitter differs -- and seed 0 ends with
twenty-one spots on the grid while the other four die out completely. The trivial state is not
a transient on the way to the pattern; it is a second resting place, and which one an initial
condition reaches depends on how big the disturbance was.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_data import (advance, coverage, load_dataset, pattern_length, seed_pattern,
                           spot_centres)

# The scan of the threshold: discs of these radii, filled with these amplitudes of v.
SCAN_RADII = (2, 3, 4, 5, 6)
SCAN_AMPLITUDES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
# How long each trial is given to decide: a spot that is going to form has formed by now, and
# one that is going to die has died.
SCAN_STEPS = 3000
# What counts as "it survived": a spot covers more than this fraction of the grid.
SURVIVAL_COVERAGE = 0.01
# The seeds used to show that the same recipe can end in either resting place, and how long
# each is given to decide.
BISTABILITY_SEEDS = (0, 1, 2, 3, 4)
BISTABILITY_STEPS = 16000


def disc_field(n, radius, amplitude, centre=None):
    """The uniform state with one disc disturbance: ``v`` raised, ``u`` lowered to match.

    ``u`` goes down where ``v`` goes up because the reaction consumes the one to make the
    other, so a disturbance that only raised ``v`` and left the feed untouched would not be a
    perturbation the equations could actually produce.
    """
    rows, cols = np.mgrid[0:n, 0:n]
    centre_row, centre_col = (n // 2, n // 2) if centre is None else centre
    dy = np.minimum((rows - centre_row) % n, (centre_row - rows) % n)
    dx = np.minimum((cols - centre_col) % n, (centre_col - cols) % n)
    inside = dy**2 + dx**2 <= radius**2
    u = np.ones((n, n))
    v = np.zeros((n, n))
    u[inside] = 1.0 - amplitude
    v[inside] = amplitude
    return u, v


def threshold_scan(n, F, k, dt, radii=SCAN_RADII, amplitudes=SCAN_AMPLITUDES,
                   steps=SCAN_STEPS):
    """What every disc turns into after ``steps``: the share of the grid covered by spots.

    A quantity rather than a verdict. A trial that was absorbed ends at zero and one that became
    a spot ends with a definite share of the grid covered, so the outcome can be drawn as a
    colour rather than labelled per cell, and what counts as surviving is a threshold on this
    (``SURVIVAL_COVERAGE``) that can be moved without re-running anything.
    """
    final = np.zeros((len(radii), len(amplitudes)))
    for row, radius in enumerate(radii):
        for col, amplitude in enumerate(amplitudes):
            u, v = disc_field(n, radius, amplitude)
            u, v = advance(u, v, steps, F, k, dt)
            final[row, col] = coverage(v)
    return final


def bistability(n, F, k, dt, warm_up, n_spots, seeds=BISTABILITY_SEEDS,
                steps=BISTABILITY_STEPS):
    """Does the same seeding recipe end in the patterned state, or in the uniform one?

    Five arrangements of the same number of discs, differing only in where the jitter puts
    them. This is the measurement that says the uniform state is a resting place rather than a
    way-station, and it is the reason the seed above is fixed and checked.
    """
    outcome = {}
    for seed in seeds:
        u, v = seed_pattern(n, n_spots, np.random.default_rng(seed))
        u, v = advance(u, v, warm_up + steps, F, k, dt)
        outcome[int(seed)] = {"coverage": coverage(v),
                             "blobs": int(len(spot_centres(v)))}
    return outcome


def threshold_figure(snapshots, counts, outcome, radii, amplitudes, settings, steps,
                     out_path, dpi=150):
    """The pattern at four moments, and what a disc of each radius and amplitude becomes."""
    fig = plt.figure(figsize=(12.6, 6.0))
    grid = fig.add_gridspec(2, 12, height_ratios=(1.0, 0.9), hspace=0.55, wspace=0.6,
                            left=0.055, right=0.97, top=0.94, bottom=0.185)

    # --- the pattern, evolving ---
    for index, (moment, field) in enumerate(snapshots):
        axis = fig.add_subplot(grid[0, index * 3:(index + 1) * 3])
        image = axis.imshow(field, cmap="inferno", vmin=0.0, vmax=0.45)
        axis.set_title(f"{moment} steps later", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        if index == len(snapshots) - 1:
            fig.colorbar(image, ax=axis, fraction=0.046, label="$v$")

    # --- what the disc becomes, as a quantity and not a verdict ---
    axis = fig.add_subplot(grid[1, 0:12])
    image = axis.imshow(outcome, cmap="inferno", origin="lower", aspect="auto",
                        vmin=0.0, vmax=max(float(outcome.max()), 1e-9))
    axis.set_xticks(np.arange(len(amplitudes)))
    axis.set_xticklabels([f"{value:.2f}" for value in amplitudes], fontsize=9)
    axis.set_yticks(np.arange(len(radii)))
    axis.set_yticklabels([str(value) for value in radii], fontsize=9)
    axis.set_xlabel("amplitude of $v$ in the seeded disc")
    axis.set_ylabel("radius of the disc, cells")
    axis.set_title(f"share of the grid covered by spots after {steps} steps", fontsize=10)
    # Where the colour changes from nothing to something, one point per radius: the step in
    # amplitude, drawn on the map it is read from.
    first = [next((col for col, value in enumerate(row) if value > SURVIVAL_COVERAGE), None)
             for row in outcome]
    axis.plot([col for col in first if col is not None],
              [row for row, col in enumerate(first) if col is not None],
              color="white", marker="o", markersize=6, lw=1.4,
              label="smallest amplitude that survives")
    axis.legend(fontsize=8.5, loc="upper left")
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.01, label="share of the grid")

    fig.text(0.5, 0.02,
             f"Gray-Scott at $F$ = {settings['F']}, $k$ = {settings['k']}, "
             f"{settings['N']} by {settings['N']} cells, wrapping; the pattern holds "
             f"{counts['blobs']:.0f} spots covering {counts['coverage']:.0%} of the grid, about "
             f"{counts['spacing']:.0f} cells apart.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Draw the Gray-Scott introduction figure.")
    parser.add_argument("--data", default="data/gray_scott.npz",
                        help="used for the settings the system was built with")
    parser.add_argument("--out", default="figures", help="folder for the figure")
    parser.add_argument("--model-dir", default="model",
                        help="where to record the measurements this script makes")
    parser.add_argument("--steps-between", type=int, default=4000,
                        help="steps between the four snapshots")
    parser.add_argument("--seed", type=int, default=0,
                        help="which arrangement of discs to start from; some die out")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    settings = load_dataset(args.data)["settings"]
    n, F, k, dt = settings["N"], settings["F"], settings["k"], settings["dt"]

    # The pattern, from a fresh seed so the figure is not a picture of the training data. The
    # seed is fixed, and the pattern is checked rather than assumed: in this regime an
    # unfavourable seeding can die out altogether, and a figure of an empty grid would be a
    # figure of nothing.
    u, v = seed_pattern(n, settings["n_spots"], np.random.default_rng(args.seed))
    u, v = advance(u, v, settings["warm_up"], F, k, dt)
    snapshots = []
    for index in range(4):
        u, v = advance(u, v, args.steps_between, F, k, dt)
        snapshots.append(((index + 1) * args.steps_between, v.copy()))
    if coverage(v) < 0.005:
        raise ValueError(f"the pattern died out for seed {args.seed}; pick another seed rather "
                         f"than drawing an empty grid")

    centres = spot_centres(snapshots[-1][1])
    counts = {"coverage": coverage(v), "blobs": float(len(centres)),
              "spacing": pattern_length(v)}

    print(f"\nthreshold scan: {len(SCAN_RADII)} radii x {len(SCAN_AMPLITUDES)} amplitudes x "
          f"{SCAN_STEPS} steps")
    final_coverage = threshold_scan(n, F, k, dt)
    survived = final_coverage > SURVIVAL_COVERAGE
    print(f"\n  {'radius':>7s}  smallest surviving amplitude of v")
    for row, radius in enumerate(SCAN_RADII):
        alive = [amplitude for amplitude, ok in zip(SCAN_AMPLITUDES, survived[row]) if ok]
        smallest = f"{alive[0]:.2f}" if alive else "none of them"
        print(f"  {radius:7d}  {smallest}")
    print(f"\n  the pattern: {counts['blobs']:.0f} spots, coverage {counts['coverage']:.3f}, "
          f"spacing about {counts['spacing']:.0f} cells")

    print(f"\nhow often does the seeded pattern survive? {len(BISTABILITY_SEEDS)} seeds, "
          f"{BISTABILITY_STEPS} steps each")
    outcome = bistability(n, F, k, dt, settings["warm_up"], settings["n_spots"])
    for seed, result in outcome.items():
        print(f"  seed {seed}: coverage {result['coverage']:.3f}, {result['blobs']} blobs")

    out = threshold_figure(snapshots, counts, final_coverage, SCAN_RADII, SCAN_AMPLITUDES,
                           settings, SCAN_STEPS, Path(args.out) / "system.png")

    # Every measurement this script makes, stored: the figure is not the only thing it produces,
    # and the README quotes the numbers from here rather than from a terminal transcript.
    smallest = {}
    for row, radius in enumerate(SCAN_RADII):
        alive = [amplitude for amplitude, ok in zip(SCAN_AMPLITUDES, survived[row]) if ok]
        smallest[str(radius)] = float(alive[0]) if alive else None
    measurements = {
        "settings": {"N": int(n), "F": float(F), "k": float(k), "Du": float(settings["Du"]),
                     "Dv": float(settings["Dv"]), "dt": float(dt)},
        "pattern": counts,
        "threshold": {"radii": [int(value) for value in SCAN_RADII],
                      "amplitudes": [float(value) for value in SCAN_AMPLITUDES],
                      "steps": int(SCAN_STEPS),
                      "final_coverage": final_coverage.tolist(),
                      "survived": survived.tolist(),
                      "smallest_surviving": smallest},
        "bistability": {"steps": int(BISTABILITY_STEPS), "outcome": outcome},
    }
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "system.json").write_text(json.dumps(measurements, indent=2))

    print(f"\nwrote {out}")
    print(f"wrote {model_dir / 'system.json'}")


if __name__ == "__main__":
    main()
