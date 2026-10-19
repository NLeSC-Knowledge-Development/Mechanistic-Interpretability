"""The moving pictures: what Gray-Scott does, and what the emulator does with it.

    ../../.venv/bin/python animate.py --data data/gray_scott.npz --model-dir model \
                                      --figures figures

A pattern that divides and dies is not well served by a still image, and neither is a threshold,
which is a story about two nudges that differ only in size. Four animations, each of them one
section of the README in motion:

* ``pattern.gif``    -- the system running: spots dividing, wandering and dying (section 1.1)
* ``disc_0.10.gif``  -- a disc of $v$, radius 4, amplitude 0.10: absorbed, leaves no trace
* ``disc_0.20.gif``  -- the same disc at amplitude 0.20: it grows into a spot (section 1.2)
* ``rollout.gif``   -- the emulator on its own, against the true system (Figure 5)
* ``push.gif``      -- one push at one cell, next to the same rollout without it (Figure 9)

Each animation names its own initial condition -- a seeded pattern, or one disc of a given radius
and amplitude -- and what became of that initial condition is written beside the gif, so a figure
and its measurement travel together.

Nothing here is a separate calculation. Each animation is drawn from the same models and the same
JSON as the still figures -- the fields come out of the same integrator and the same network -- and
this script exists so that they can be redrawn without retraining anything.

A GIF is a compromise: 256 colours and no compression worth the name. These are kept small by
drawing few cells at a modest figure size, and each one reports what it weighs.
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
from matplotlib.animation import FuncAnimation, PillowWriter

from generate_data import advance, coverage, load_dataset, seed_pattern, spot_centres
from make_fig1 import disc_field
from perturb import blob_near, paired_rollout
from train_emulator import free_rollout, load_emulator, truth_rollout
from train_sae import SparseAutoencoder

# One colour scale for every field drawn here, so that a frame from one animation can be compared
# with a frame from another: spots peak at about 0.45 and the background is zero.
V_MIN, V_MAX = 0.0, 0.45
MARKER_COLOUR = "#39c5f0"
# Enough resolution to see a spot and small enough that a hundred frames do not add up to a file
# nobody wants to download. A GIF stores every frame in full.
DPI = 80


def save(animation, path, fps):
    """Write an animation as a GIF, and say how large it came out."""
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    size = path.stat().st_size / 1e6
    print(f"  wrote {path} ({size:.1f} MB, {fps} frames per second)")
    return path


def subplot_fields(axes, frames, titles, step_labels):
    """A function that draws one frame of one or more fields, for FuncAnimation to call."""
    images = [axis.imshow(frames[0][index], cmap="inferno", vmin=V_MIN, vmax=V_MAX,
                          interpolation="nearest")
              for index, axis in enumerate(axes)]
    for axis, title in zip(axes, titles):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title, fontsize=10)

    def draw(frame):
        for image, stack in zip(images, frames):
            image.set_data(stack[frame])
        axes[0].set_xlabel(step_labels[frame], fontsize=9.5)
        return images

    return draw


# --------------------------------------------------------------------------- the system, running


def pattern_animation(settings, out_path, seed=0, warm_up=2000, steps=4000, every=50, fps=12):
    """The system left to itself: the animation the whole folder is a question about."""
    n, F, k, dt = settings["N"], settings["F"], settings["k"], settings["dt"]
    u, v = seed_pattern(n, settings["n_spots"], np.random.default_rng(seed))
    u, v = advance(u, v, warm_up, F, k, dt)
    frames = []
    for _ in range(steps // every):
        u, v = advance(u, v, every, F, k, dt)
        frames.append(v.copy())
    if max(float(field.max()) for field in frames) < 0.1:
        raise ValueError(f"the pattern died out for seed {seed}; pick another rather than "
                         f"animating an empty grid")

    fig, axis = plt.subplots(figsize=(4.4, 4.4))
    fig.set_dpi(DPI)
    labels = [f"step {(index + 1) * every:,}" for index in range(len(frames))]
    animation = FuncAnimation(fig, subplot_fields([axis], [frames], [""], labels),
                              frames=len(frames), interval=1000 / fps, blit=False)
    fig.text(0.5, 0.02, f"Gray-Scott at $F$ = {F}, $k$ = {k}, and nothing else happening",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(animation, out_path, fps)
    plt.close(fig)
    return {
        "gif": out_path.name, "grid": int(n), "F": float(F), "k": float(k), "dt": float(dt),
        "seed": int(seed), "warm_up_steps": int(warm_up), "animated_steps": int(steps),
        "steps_per_frame": int(every), "frames": len(frames), "fps": int(fps),
        "blobs": int(len(spot_centres(frames[-1]))), "coverage": float(coverage(frames[-1])),
    }


# ------------------------------------------------------------- the disc-shaped initial condition

# The initial conditions the README shows moving: one disc of ``v``, radius 4, put into the
# uniform state at two amplitudes that straddle the transition measured in Figure 1 -- for this
# radius the smallest amplitude that survives is 0.15. One condition, one gif, and what became of
# it stored beside the gif.
DISC_RADIUS = 4
DISC_AMPLITUDES = (0.10, 0.20)
DISC_STEPS = 900
DISC_EVERY = 15


def disc_animation(settings, radius, amplitude, out_path, steps=DISC_STEPS, every=DISC_EVERY,
                   fps=12):
    """One disc-shaped initial condition evolving, and the measurement that goes with it.

    Nothing is specified but the disc: the uniform state, one patch of ``v`` at this radius and
    amplitude, and the equations. What the animation shows is the fate of that one nudge, and
    whether it survived is returned as a number rather than left to the eye.
    """
    n, F, k, dt = settings["N"], settings["F"], settings["k"], settings["dt"]
    u, v = disc_field(n, radius, amplitude)
    frames = [v.copy()]
    for _ in range(int(steps / every)):
        u, v = advance(u, v, every, F, k, dt)
        frames.append(v.copy())

    fig, axis = plt.subplots(figsize=(3.4, 3.4))
    fig.set_dpi(DPI)
    labels = [f"step {index * every:,}" for index in range(len(frames))]
    animation = FuncAnimation(fig, subplot_fields([axis], [frames],
                                                 [f"a disc of $v$ = {amplitude:.2f}"], labels),
                              frames=len(frames), interval=1000 / fps, blit=False)
    fig.tight_layout()
    save(animation, out_path, fps)
    plt.close(fig)
    return {
        "gif": out_path.name, "radius": int(radius), "amplitude": float(amplitude),
        "steps": int(steps), "steps_per_frame": int(every), "frames": len(frames),
        "fps": int(fps), "survived": bool(float(frames[-1].max()) > 0.25),
        "final_peak": float(frames[-1].max()), "final_coverage": float(coverage(frames[-1])),
    }


def disc_animations(settings, figures, amplitudes=DISC_AMPLITUDES, radius=DISC_RADIUS, fps=12):
    """Every disc-shaped initial condition, one gif each, with the outcomes stored beside them."""
    results = {}
    for amplitude in amplitudes:
        result = disc_animation(settings, radius, amplitude,
                                figures / f"disc_{amplitude:.2f}.gif", fps=fps)
        results[f"{amplitude:.2f}"] = result
        print(f"  a disc of radius {radius} with v = {amplitude:.2f}: "
              f"{'a spot' if result['survived'] else 'absorbed'} after {result['steps']} steps")
    (figures / "disc.json").write_text(json.dumps(results, indent=2))
    print(f"  wrote {figures / 'disc.json'}")
    return results


# ------------------------------------------------------------------ the emulator on its own


def rollout_animation(model, settings, test_inputs, mean, std, out_path, calls=40, start=0, fps=8):
    """The true system and the emulator, side by side, from the same state.

    This is Figure 5 with the plots taken away: the error curves say the two have parted company,
    and the frames say what that looks like -- which is two populations of spots with the same
    statistics and no spot in the same place.
    """
    stride = settings["stride"]
    states = test_inputs[start:start + 1]
    physical = states * std.reshape(1, 1, 2, 1, 1) + mean.reshape(1, 1, 2, 1, 1)
    predicted = free_rollout(model, states, calls)
    truth = truth_rollout(physical, calls, stride, settings["F"], settings["k"], settings["dt"])
    emulator_v = predicted[:, 0, 1] * std[1] + mean[1]
    truth_v = truth[:, 0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.2))
    fig.set_dpi(DPI)
    labels = [f"{call} calls" for call in range(calls + 1)]
    animation = FuncAnimation(
        fig, subplot_fields(axes, [truth_v, emulator_v], ["the true system", "the emulator"],
                            labels),
        frames=calls + 1, interval=1000 / fps, blit=False)
    fig.text(0.5, 0.02, f"one call is {stride} model steps; the same start, {calls} calls of "
                        f"autoregression", ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    saved = save(animation, out_path, fps)
    plt.close(fig)
    return saved


# ------------------------------------------------------------------------------------- the push


def push_animation(model, sae, layer, settings, test_inputs, mean, std, out_path, calls=25,
                   push_fraction=0.25, site_radius=6, tries=40, seed=0, fps=8):
    """One push at one cell, once, beside the same rollout without it.

    The pair shown is the first one found where the push was followed by a blob the control
    rollout did not have -- chosen by running the experiment, not by looking at an animation and
    picking a pretty one. If no pair out of ``tries`` does that, the first pair is shown and said
    to be unremarkable, which is the honest thing to animate.
    """
    direction = _top_direction(sae.directions())
    rng = np.random.default_rng(seed)
    n = settings["N"]

    shown = None
    chosen_by = f"no pair out of {tries} worked, so this is the first one tried"
    for attempt in range(tries):
        index = int(rng.integers(0, len(test_inputs)))
        row, col = int(rng.integers(0, n)), int(rng.integers(0, n))
        pushed, control, _ = paired_rollout(model, layer, test_inputs[index], row, col,
                                            direction, calls, push_fraction)
        final_pushed = pushed[-1, 1] * std[1] + mean[1]
        final_control = control[-1, 1] * std[1] + mean[1]
        appeared = (blob_near(final_pushed, row, col, site_radius)
                    and not blob_near(final_control, row, col, site_radius))
        if shown is None:
            shown = (pushed, control, row, col)
        if appeared:
            shown = (pushed, control, row, col)
            chosen_by = f"a pair where the push changed the outcome (attempt {attempt + 1})"
            break
    pushed, control, row, col = shown
    pushed_v = pushed[:, 1] * std[1] + mean[1]
    control_v = control[:, 1] * std[1] + mean[1]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.4))
    fig.set_dpi(DPI)
    labels = [f"{call + 1} calls after the push" for call in range(calls)]
    animation = FuncAnimation(
        fig, subplot_fields(axes, [control_v, pushed_v],
                            ["without the push", "with the push"], labels),
        frames=calls, interval=1000 / fps, blit=False)
    for axis in axes:
        axis.plot([col], [row], marker="o", markersize=11, markerfacecolor="none",
                  markeredgecolor=MARKER_COLOUR, markeredgewidth=2.0)
    fig.text(0.5, 0.02, f"the marked cell is where the token was pushed, once, on the first call; "
                        f"the pair was picked by {chosen_by}",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    saved = save(animation, out_path, fps)
    plt.close(fig)
    return saved


def _top_direction(directions):
    """Which direction to push: the one the ranking in perturb.py puts first.

    Read from ``perturb.json`` when that run has happened -- the ranking costs about a minute to
    recompute and this script should not be slower than the thing it illustrates -- and falling
    back to the first direction of the dictionary when it has not.
    """
    path = Path("model/perturb.json")
    if path.exists():
        effects = np.asarray(json.loads(path.read_text())["effects"])
        return directions[int(np.argmax(effects))]
    return directions[0]


# ----------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="Draw the animations.")
    parser.add_argument("--data", default="data/gray_scott.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--model-name", default="emulator")
    parser.add_argument("--pattern-steps", type=int, default=4000)
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--push-calls", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--only", default=None,
                        help="draw just some of pattern, disc, rollout, push "
                             "(comma-separated for more than one)")
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    settings = dataset["settings"]
    mean, std = dataset["mean"], dataset["std"]
    figures = Path(args.figures)
    wanted = set(args.only.split(",")) if args.only else {"pattern", "disc", "rollout", "push"}

    print(f"\nanimations at {settings['N']} x {settings['N']} cells, "
          f"F = {settings['F']}, k = {settings['k']}\n")
    if "pattern" in wanted:
        result = pattern_animation(settings, figures / "pattern.gif", seed=args.seed,
                                   steps=args.pattern_steps, fps=args.fps)
        (figures / "pattern.json").write_text(json.dumps(result, indent=2))
        print(f"  wrote {figures / 'pattern.json'}")
    if "disc" in wanted:
        disc_animations(settings, figures, fps=args.fps)

    model = None
    if ({"rollout", "push"} & wanted) and (Path(args.model_dir) / f"{args.model_name}.pt").exists():
        model, _ = load_emulator(Path(args.model_dir) / f"{args.model_name}.pt")
    if "rollout" in wanted:
        if model is None:
            raise FileNotFoundError("train_emulator.py has not been run, so there is no rollout "
                                   "to animate")
        rollout_animation(model, settings, dataset["splits"]["test"]["inputs"], mean, std,
                          figures / "rollout.gif", calls=args.calls, fps=args.fps)
    if "push" in wanted:
        if model is None:
            raise FileNotFoundError("train_emulator.py has not been run, so there is no push to "
                                   "animate")
        sae_path = Path(args.model_dir) / "sae.pt"
        if not sae_path.exists():
            raise FileNotFoundError("train_sae.py has not been run, so there is no direction to "
                                    "push")
        checkpoint = torch.load(sae_path, weights_only=False)
        config = checkpoint["config"]
        sae = SparseAutoencoder(config["n_input"], config["n_latent"], config["k"])
        sae.load_state_dict(checkpoint["state_dict"])
        sae.eval()
        push_animation(model, sae, config["layer"], settings,
                       dataset["splits"]["test"]["inputs"], mean, std,
                       figures / "push.gif", calls=args.push_calls, fps=args.fps, seed=args.seed)


if __name__ == "__main__":
    main()
