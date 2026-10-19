"""Push one direction at one cell, and see whether the pattern is changed for good.

    ../../.venv/bin/python perturb.py --data data/gray_scott.npz --model-dir model \
                                      --figures figures

This is the experiment the folder exists for. Everything before it describes what the sparse
autoencoder found; this asks whether any of it is a **handle** on the model's behaviour, and
whether the handle turns something in the world the model is imitating.

**1. Which directions matter.** Every one of the 512 directions is pushed at every cell by the
same fraction of the token's own size, and the change in the predicted next state is recorded.
Most directions should do very little; the ranking says which few do something, and the
comparison against random directions of the same size says whether that means anything.

**2. One push, at one cell, once.** The push is applied on the first call and never again, and
from there on the model runs on its own. A pattern in this regime can do two things: a small
disturbance is absorbed and leaves no trace, and a large enough one becomes a spot that stays.
So the question has a yes-or-no answer rather than a number -- and it is asked **paired**: the
same state, the same cell, with and without the push, so that a blob which was going to appear
anyway is not counted as the push's doing.

**3. What the push is equivalent to, in the equations.** The push is a change to the emulator's
activations, which is not a thing the equations can be given. But the difference it makes to the
predicted field *is* a perturbation of the two fields, and that can be measured: its shape, its
size, and what the **true** system does when it is added to the real state and integrated with
the real equations. If the emulator's direction is a handle on something physical, this is where
it shows: the same nudge, put to the equations instead of to the network.

Writes ``<model-dir>/perturb.json`` and ``<figures>/perturb.png``.
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
import torch.nn.functional as F

from generate_data import SPOT_THRESHOLD, advance, load_dataset, spot_centres
from train_emulator import (BASELINE_COLOUR, HIDDEN_COLOUR, MIDDLE_COLOUR, TRAIN_COLOUR,
                            load_emulator, token_activations)
from train_sae import SparseAutoencoder

# How hard the push is, as a fraction of the token's own size.
PUSH_FRACTION = 0.25
# The ranking is measured over this many states.
N_RANK_STATES = 16
# The paired experiment: how many (state, cell) pairs, and how many calls to follow them.
N_PAIRS = 48
N_CALLS = 20
# A blob counts as having appeared at the push site if its middle is within this many cells.
SITE_RADIUS = 6
TOP_COLOUR = "#c44e52"


def patched_forward(model, inputs, layer, patch):
    """Run the model with ``patch`` applied to one layer's activations.

    A forward hook that returns a tensor replaces the module's output, which is the same
    mechanism used to read activations everywhere else in this folder -- used here to write
    instead of read.
    """
    handle = model.get_submodule(layer).register_forward_hook(
        lambda module, args, output: patch(output))
    try:
        with torch.no_grad():
            return model(inputs)
    finally:
        handle.remove()


def rank_directions(model, layer, states, directions, push_fraction=PUSH_FRACTION):
    """How much the predicted next state moves when each direction is pushed at every cell.

    The push is ``push_fraction`` times the size of the token it is applied to, so every
    direction gets the same relative shove and the comparison between them is fair.
    """
    inputs = torch.as_tensor(states, dtype=torch.float32)
    with torch.no_grad():
        base = model(inputs)
    tokens = token_activations(model, states, layer)
    scale = tokens.norm(dim=1).mean()

    effects = torch.zeros(len(directions))
    for index in range(len(directions)):
        step = F.normalize(directions[index].unsqueeze(0), dim=1).squeeze(0) \
            * push_fraction * scale
        pushed = patched_forward(model, inputs, layer,
                                 lambda output, s=step: output + s[None, :, None, None])
        effects[index] = (pushed - base).pow(2).mean().sqrt()
    return effects


def paired_rollout(model, layer, state, row, col, direction, n_calls,
                   push_fraction=PUSH_FRACTION):
    """Two rollouts from the same state, identical except for one push at one cell.

    The push happens on the first call only; after that neither rollout is touched and each is
    fed its own prediction. Returns the two paths through the fields, and the field difference
    the push made on the very first call -- which is the perturbation the equations could be
    given.
    """
    window = torch.as_tensor(state[None], dtype=torch.float32)
    tokens = token_activations(model, state[None], layer)
    scale = tokens.norm(dim=1).mean()
    step = F.normalize(direction.unsqueeze(0), dim=1).squeeze(0) * push_fraction * scale

    def patch(output):
        pushed = output.clone()
        pushed[0, :, row, col] = pushed[0, :, row, col] + step
        return pushed

    pushed_state = patched_forward(model, window, layer, patch)
    with torch.no_grad():
        plain_state = model(window)
    effective = (pushed_state - plain_state).numpy()[0]
    pushed_window = torch.cat([window[:, 1:], pushed_state[:, None]], dim=1)
    plain_window = torch.cat([window[:, 1:], plain_state[:, None]], dim=1)
    pushed_fields = [pushed_state.numpy()[0]]
    plain_fields = [plain_state.numpy()[0]]
    for _ in range(n_calls - 1):
        with torch.no_grad():
            pushed_state = model(pushed_window)
            plain_state = model(plain_window)
        pushed_window = torch.cat([pushed_window[:, 1:], pushed_state[:, None]], dim=1)
        plain_window = torch.cat([plain_window[:, 1:], plain_state[:, None]], dim=1)
        pushed_fields.append(pushed_state.numpy()[0])
        plain_fields.append(plain_state.numpy()[0])
    return np.stack(pushed_fields), np.stack(plain_fields), effective


def blob_near(field, row, col, radius=SITE_RADIUS):
    """Is there a blob whose middle is within ``radius`` cells of this cell?"""
    n = field.shape[0]
    for centre_row, centre_col in spot_centres(field):
        dy = min(abs(centre_row - row), n - abs(centre_row - row))
        dx = min(abs(centre_col - col), n - abs(centre_col - col))
        if np.hypot(dy, dx) <= radius:
            return True
    return False


def perturb_figure(effects, top, random_effects, rates, labels, example, truth_rates,
                   out_path, dpi=150):
    """Left: which directions matter. Centre: did a blob appear. Right: one example, up close."""
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))

    ranked = np.sort(effects)[::-1]
    axes[0].plot(np.arange(1, len(ranked) + 1), ranked, color=TRAIN_COLOUR, lw=1.8,
                 label="the SAE's directions")
    axes[0].axhline(float(np.mean(random_effects)), color=BASELINE_COLOUR, lw=1.4,
                    linestyle="--", label="random directions, on average")
    axes[0].scatter([1], [ranked[0]], color=TOP_COLOUR, zorder=3, s=28)
    axes[0].annotate(f"latent {top}", xy=(1, ranked[0]), xytext=(24, ranked[0] * 1.12),
                     fontsize=9, color=TOP_COLOUR,
                     arrowprops=dict(arrowstyle="-|>", color=TOP_COLOUR, linewidth=1.2))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("direction, ranked")
    axes[0].set_ylabel("change in the predicted next state")
    axes[0].set_title("which directions matter", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].grid(alpha=0.3, which="both")

    position = np.arange(len(labels))
    width = 0.34
    colours = [TOP_COLOUR, MIDDLE_COLOUR, BASELINE_COLOUR]
    equations = [truth_rates.get(label, 0.0) for label in labels]
    axes[1].bar(position - width / 2, rates, width=width, color=colours[:len(labels)],
                label="the emulator")
    axes[1].bar(position + width / 2, equations, width=width, color="#c9c9c9",
                edgecolor="#8a8a8a", linewidth=0.6,
                label="the equations, given the same nudge")
    axes[1].set_xticks(position)
    axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_ylabel(f"share followed by a blob ({N_PAIRS} pairs)")
    axes[1].set_ylim(0.0, max(0.35, float(np.max(rates + equations)) * 1.35))
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_title("the push, and what followed it", fontsize=10)
    for index, (rate, other) in enumerate(zip(rates, equations)):
        axes[1].text(index - width / 2, rate, f"{rate:.0%}", ha="center", va="bottom",
                     fontsize=8.5)
        axes[1].text(index + width / 2, other, f"{other:.0%}", ha="center", va="bottom",
                     fontsize=8.5, color="#666666")

    control, pushed = example
    difference = pushed - control
    shown = np.concatenate([control, pushed, difference], axis=1)
    image = axes[2].imshow(shown, cmap="inferno", vmin=0.0, vmax=0.45)
    axes[2].axvline(control.shape[1] - 0.5, color="white", lw=1.0)
    axes[2].axvline(2 * control.shape[1] - 0.5, color="white", lw=1.0)
    axes[2].set_xticks([control.shape[1] / 2, 1.5 * control.shape[1],
                        2.5 * control.shape[1]])
    axes[2].set_xticklabels(["without\nthe push", "with\nthe push", "the\ndifference"],
                            fontsize=8.5)
    axes[2].set_yticks([])
    axes[2].set_title(f"one pair, {N_CALLS} calls later", fontsize=10)
    fig.colorbar(image, ax=axes[2], fraction=0.046, label="$v$")

    fig.text(0.5, 0.01,
             "left: every direction pushed at every cell, and how far the predicted next state "
             "moves. centre: a push at one cell, once, then the\nmodel runs on its own -- the "
             "share of pairs where a blob appeared at the push site, and the same nudge given to "
             "the true equations. right: one\npair of rollouts, and the difference between them.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Push one direction at one cell and watch what follows.")
    parser.add_argument("--data", default="data/gray_scott.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--model-name", default="emulator")
    parser.add_argument("--push-fraction", type=float, default=PUSH_FRACTION)
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    parser.add_argument("--calls", type=int, default=N_CALLS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset = load_dataset(args.data)
    settings = dataset["settings"]
    mean, std = dataset["mean"], dataset["std"]
    n = settings["N"]
    model, checkpoint = load_emulator(Path(args.model_dir) / f"{args.model_name}.pt")
    sae_checkpoint = torch.load(Path(args.model_dir) / "sae.pt", weights_only=False)
    sae_config = sae_checkpoint["config"]
    sae = SparseAutoencoder(sae_config["n_input"], sae_config["n_latent"], sae_config["k"])
    sae.load_state_dict(sae_checkpoint["state_dict"])
    sae.eval()
    layer = sae_config["layer"]
    directions = sae.directions()

    # ------------------------------------------------------------- 1. which directions matter
    rank_states = dataset["splits"]["val"]["inputs"][:N_RANK_STATES]
    effects = rank_directions(model, layer, rank_states, directions, args.push_fraction)
    order = torch.argsort(effects, descending=True)
    top, inert = int(order[0]), int(order[-1])
    generator = torch.Generator().manual_seed(args.seed)
    random_set = F.normalize(torch.randn(64, sae_config["n_input"], generator=generator), dim=1)
    random_effects = rank_directions(model, layer, rank_states, random_set, args.push_fraction)

    # ------------------------------------------------- 2. one push, at one cell, once
    rng = np.random.default_rng(args.seed)
    test = dataset["splits"]["test"]["inputs"]
    stride = max(len(test) // args.pairs, 1)
    starts = test[::stride][:args.pairs]
    rows = rng.integers(0, n, size=len(starts))
    cols = rng.integers(0, n, size=len(starts))

    candidates = [
        (f"latent {top}", directions[top]),
        (f"latent {inert}", directions[inert]),
        ("a random direction", random_set[0]),
    ]
    rates, labels, effective_fields, examples = [], [], {}, {}
    for label, direction in candidates:
        hits, effective = 0, []
        for index, state in enumerate(starts):
            row, col = int(rows[index]), int(cols[index])
            pushed, control, difference = paired_rollout(
                model, layer, state, row, col, direction, args.calls, args.push_fraction)
            final_pushed = pushed[-1, 1] * std[1] + mean[1]
            final_control = control[-1, 1] * std[1] + mean[1]
            appeared = (blob_near(final_pushed, row, col)
                        and not blob_near(final_control, row, col))
            hits += int(appeared)
            # Rolled so the push site sits at the centre: the profile is a shape, and averaging
            # shapes without aligning them would average them away.
            effective.append(np.roll(np.roll(difference, -row, axis=-2), -col, axis=-1))
            if appeared and label not in examples:
                examples[label] = (final_control, final_pushed)
        rates.append(hits / len(starts))
        labels.append(label)
        effective_fields[label] = np.mean(effective, axis=0)

    # --------------------------------- 3. the same nudge, given to the equations instead
    truth_rates = {}
    for label, direction in candidates:
        profile = effective_fields[label]
        hits = 0
        for index, state in enumerate(starts):
            row, col = int(rows[index]), int(cols[index])
            u = state[-1, 0] * std[0] + mean[0]
            v = state[-1, 1] * std[1] + mean[1]
            # A blob that was already there is not this nudge's doing, so the baseline is taken
            # before the nudge rather than after it.
            before = blob_near(v, row, col)
            u = u + np.roll(np.roll(profile[0], row, axis=-2), col, axis=-1)
            v = v + np.roll(np.roll(profile[1], row, axis=-2), col, axis=-1)
            u, v = advance(u, v, args.calls * settings["stride"], settings["F"], settings["k"],
                           settings["dt"])
            hits += int(blob_near(v, row, col) and not before)
        truth_rates[label] = hits / len(starts)

    results = {
        "layer": layer,
        "n_latent": int(sae_config["n_latent"]),
        "push_fraction": args.push_fraction,
        "effect_top": float(effects[top]),
        "effect_median": float(effects.median()),
        "effect_inert": float(effects[inert]),
        "effects": effects.tolist(),
        "random_effect_mean": float(random_effects.mean()),
        "random_effect_max": float(random_effects.max()),
        "n_pairs": int(len(starts)),
        "n_calls": int(args.calls),
        "nucleation_rate": dict(zip(labels, [float(rate) for rate in rates])),
        "nucleation_rate_truth": {label: float(rate) for label, rate in truth_rates.items()},
        "effective_perturbation_peak": {
            label: [float(np.abs(field[0]).max()), float(np.abs(field[1]).max())]
            for label, field in effective_fields.items()},
        "effective_perturbation_mean": [float(effective_fields[labels[0]][0].mean()),
                                        float(effective_fields[labels[0]][1].mean())],
        "spot_threshold": SPOT_THRESHOLD,
    }
    model_dir = Path(args.model_dir)
    (model_dir / "perturb.json").write_text(json.dumps(results, indent=2))
    example = examples.get(labels[0], (np.zeros((n, n)), np.zeros((n, n))))
    perturb_figure(effects.numpy(), top, random_effects.numpy(), rates, labels, example,
                   truth_rates, Path(args.figures) / "perturb.png")

    # ---------------------------------------------------------------------- what we found
    print(f"\nlayer '{layer}', {sae_config['n_latent']} directions, push of "
          f"{args.push_fraction:.2f} x the token size")
    print(f"\n1. which directions matter")
    print(f"  the ten strongest: " + ", ".join(f"{int(i)} ({effects[i]:.4f})"
                                               for i in order[:10]))
    print(f"  median {effects.median():.4f}, weakest {effects[inert]:.4f}; random directions "
          f"average {random_effects.mean():.4f}, the best of the 64 {random_effects.max():.4f}")

    print(f"\n2. one push, at one cell, once, then the model runs on its own "
          f"({len(starts)} pairs, {args.calls} calls)")
    for label, rate in zip(labels, rates):
        print(f"  {label:<20s} a blob appeared at the push site in {rate:.0%} of pairs")

    print(f"\n3. the same nudge, given to the equations")
    for label, rate in truth_rates.items():
        peak_u, peak_v = results["effective_perturbation_peak"][label]
        print(f"  {label:<20s} peak |du| {peak_u:.4f}, |dv| {peak_v:.4f} -> the true system "
              f"grows a blob in {rate:.0%} of cases")

    print(f"\nwrote {model_dir / 'perturb.json'}")
    print(f"wrote {Path(args.figures) / 'perturb.png'}")


if __name__ == "__main__":
    main()
