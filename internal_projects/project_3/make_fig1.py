"""Draw the system: the ring, one run of it, and two runs that start 1e-8 apart.

    ../../.venv/bin/python make_fig1.py --out figures

Nothing is trained here. The figure is the introduction: what the 40 numbers look like as
they evolve, and what it means for the system to be chaotic.

Everything is computed from the same settings the dataset was built with, so the picture is
of the same system the emulator is trained on, and it needs no stored trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_data import (DEFAULT_DT, DEFAULT_F, DEFAULT_N, TIME_UNIT_IN_DAYS,
                           generate_trajectory, integrate, load_dataset, rk4_step)

TRAIN_COLOUR = "#4c72b0"
VAL_COLOUR = "#c44e52"
BASELINE_COLOUR = "#999999"


def main():
    parser = argparse.ArgumentParser(description="Draw the Lorenz-96 introduction figure.")
    parser.add_argument("--data", default="data/lorenz96.npz",
                        help="used only for the settings the system was built with")
    parser.add_argument("--out", default="figures", help="folder for the figure")
    parser.add_argument("--days", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    settings = load_dataset(args.data)["settings"]
    n_sites = settings["N"]
    forcing = settings["F"]
    dt = settings["dt"]
    dt_out = settings["dt_out"]

    n_samples = int(round(args.days / (dt_out * TIME_UNIT_IN_DAYS)))
    trajectory = generate_trajectory(n_samples, dt=dt, dt_out=dt_out, N=n_sites, F=forcing,
                                    seed=args.seed)
    days = np.arange(len(trajectory)) * dt_out * TIME_UNIT_IN_DAYS

    # A twin that starts 1e-8 away -- 100 million times smaller than the state itself.
    rng = np.random.default_rng(args.seed + 1)
    direction = rng.normal(size=n_sites)
    direction /= np.linalg.norm(direction)
    twin = integrate(trajectory[0] + 1e-8 * direction, dt, n_samples * int(round(dt_out / dt)),
                     F=forcing, sample_every=int(round(dt_out / dt)))
    distance = np.sqrt(np.mean((twin - trajectory) ** 2, axis=1))
    size_of_variability = float(np.std(trajectory))
    # Two states with nothing to do with each other sit this far apart in this metric; it is
    # the level at which a forecast has stopped carrying information.
    unrelated_level = float(np.sqrt(2.0) * size_of_variability)

    # When does the difference first become visible: 0.05% of the plotted range.
    visible = np.flatnonzero(distance > 0.0005 * (trajectory.max() - trajectory.min()))
    first_visible = days[visible[0]] if visible.size else float("nan")

    fig = plt.figure(figsize=(13.2, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.85), hspace=0.42, wspace=0.22)

    # --- the state at every site over time ---
    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(trajectory.T, aspect="auto", origin="lower", cmap="coolwarm",
                      extent=[days[0], days[-1], 0, n_sites])
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("site")
    ax.set_title("the state at every site, one run of the true system", fontsize=10)
    fig.colorbar(image, ax=ax, label="value")

    # --- a few sites traced out ---
    ax = fig.add_subplot(grid[0, 1])
    for site, colour in ((10, TRAIN_COLOUR), (20, "#dd8452"), (30, "#55a868")):
        ax.plot(days, trajectory[:, site], color=colour, lw=1.2, label=f"site {site}")
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("value")
    ax.set_title("a few sites on their own", fontsize=10)
    ax.legend(fontsize=8.5, ncols=3)
    ax.grid(alpha=0.3)

    # --- two runs, 1e-8 apart ---
    ax = fig.add_subplot(grid[1, 0])
    ax.plot(days, distance, color=VAL_COLOUR, lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("distance between the two runs")
    ax.grid(alpha=0.3, which="both")
    doubling = settings.get("system_doubling_days", 2.09)
    reference = distance[0] * 2 ** (days / doubling)
    ax.plot(days, reference, color=BASELINE_COLOUR, lw=1.3, linestyle="--",
            label=f"doubling every {doubling:.1f} days")
    ax.axhline(unrelated_level, color=TRAIN_COLOUR, lw=1.3, linestyle=":",
               label=f"two unrelated states ({unrelated_level:.1f} apart)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("two runs 1e-8 apart", fontsize=10)

    # --- what that means, in words, as the third panel ---
    ax = fig.add_subplot(grid[1, 1])
    ax.set_axis_off()
    ax.text(0.0, 0.98,
            "The same equations, run twice.\n\n"
            "The two runs differ by 1e-8 at the start -- a hundred\n"
            "million times smaller than the state itself. For the first\n"
            f"{first_visible:.0f} days the difference is invisible at this plotting\n"
            "precision. It then grows exponentially, doubling every\n"
            f"{doubling:.1f} days, until by the end of the run the two are simply\n"
            "uncorrelated: an error the size of the system's own\n"
            "variability means knowing nothing.\n\n"
            "Nothing random happened anywhere: this is the same\n"
            "equation, the same code, and a starting point moved\n"
            "by less than any measurement could resolve.",
            ha="left", va="top", fontsize=8.8, color="#333333", linespacing=1.4)

    fig.suptitle("Lorenz-96: 40 numbers on a ring, and the wall of chaos", fontsize=12)
    fig.text(0.5, 0.015,
             f"$dx_i/dt = (x_{{i+1}} - x_{{i-2}})\\,x_{{i-1}} - x_i + F$,  "
             f"{n_sites} sites, $F$ = {forcing:g}. completely deterministic, trivially cheap "
             f"to simulate, and not predictable in detail:\nsmall errors grow. this is the "
             f"property every number in this folder runs into.",
             ha="center", va="bottom", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    out = Path(args.out) / "system.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  the two runs first differ visibly at day {first_visible:.0f}, "
          f"and the difference doubles every {doubling:.1f} days")


if __name__ == "__main__":
    main()
