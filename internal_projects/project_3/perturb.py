"""Push one feature inside the emulator, and measure what follows.

    ../../.venv/bin/python perturb.py --data data/lorenz96.npz --model-dir model \
                                      --figures figures

The SAE gives a dictionary of directions in the emulator's internal representation. Two
questions follow, and this script answers both.

**1. Which of the directions matter?** Every direction is pushed at every site by the same
fraction of the token's own size, the network is run forward, and the change in the predicted
next state is recorded. Most directions should do very little -- that is what a dictionary
trained on one layer of a small network ought to look like -- and the ranking says which few
do something. This is the cheap question, and it is answered for all latents at once.

**2. What does a push do in time and space?** This is the expensive one, and it is the reason
the folder exists. Push one direction at **one site of one state**, then let the model run on
its own from there, and watch the difference between the pushed and unpushed forecasts:

* **growth** -- the size of the difference against lead time, and the rate at which it doubles,
  against the system's own rate of 2.1 days;
* **spread** -- where on the ring the two forecasts differ, against lead time. A local push
  should stay local for a while and then travel.

The controls get exactly the same size of push: a random direction in the same token space,
and a direction belonging to a latent the ranking says is inert. If the named direction behaves
like those, then on this task the dictionary is a description and not a handle -- which is
worth knowing before pointing the method at a model where it cannot be checked.

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

from generate_data import TIME_UNIT_IN_DAYS, load_dataset
from train_emulator import (BASELINE_COLOUR, HIDDEN_COLOUR, MIDDLE_COLOUR, TRAIN_COLOUR,
                            load_emulator, token_activations)
from train_sae import SparseAutoencoder

# How hard the push is, as a fraction of the token's own size.
PUSH_FRACTION = 0.25
# The ranking is measured over this many states.
N_RANK_STATES = 400
# The propagation study: how many starts, and how long to follow them.
N_STARTS = 48
N_STEPS = 40
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
    """How much the predicted next state moves when each direction is pushed at every site.

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
        step = F.normalize(directions[index].unsqueeze(0), dim=1).squeeze(0) * push_fraction * scale
        pushed = patched_forward(model, inputs, layer, lambda output, s=step: output + s)
        effects[index] = (pushed - base).pow(2).mean().sqrt()
    return effects


def rollout_difference(model, layer, starts, direction, sites, n_steps=N_STEPS,
                       push_fraction=PUSH_FRACTION):
    """The difference between a pushed and an unpushed rollout, for a batch of starts.

    Each start gets its own push site, so the result averages over where the push happened.
    The push is applied once, on the first step; after that both rollouts are left alone and
    fed their own predictions back, so whatever difference appears from there on is the
    model's own dynamics amplifying what was changed.
    """
    n_sites = starts.shape[-1]
    inputs = torch.as_tensor(starts, dtype=torch.float32)
    with torch.no_grad():
        tokens = token_activations(model, starts, layer).reshape(len(starts), n_sites, -1)
        scale = tokens.norm(dim=2).mean()

        def patch(output):
            output = output.clone()
            index = torch.arange(len(output), device=output.device)
            output[index, sites, :] = (output[index, sites, :]
                                       + direction.unsqueeze(0) * push_fraction * scale)
            return output

        pushed_state = patched_forward(model, inputs, layer, patch)
        plain_state = model(inputs)
        pushed_window = torch.cat([inputs[:, 1:], pushed_state.unsqueeze(1)], dim=1)
        plain_window = torch.cat([inputs[:, 1:], plain_state.unsqueeze(1)], dim=1)

        difference = []
        for _ in range(n_steps):
            difference.append((pushed_window[:, -1] - plain_window[:, -1]).abs().numpy())
            pushed_state = model(pushed_window)
            plain_state = model(plain_window)
            pushed_window = torch.cat([pushed_window[:, 1:], pushed_state.unsqueeze(1)], dim=1)
            plain_window = torch.cat([plain_window[:, 1:], plain_state.unsqueeze(1)], dim=1)
    return np.stack(difference)


def doubling_time(lead_days, size):
    """Doubling time of a difference curve, over the stretch before it saturates."""
    above = np.flatnonzero(size > 2.0 * size[0])
    if above.size == 0:
        return float("nan")
    window = slice(0, max(int(above[0]), 2))
    slope = np.polyfit(lead_days[window], np.log(size[window]), 1)[0]
    return float(np.log(2.0) / slope) if slope > 0 else float("nan")


def _ranks(values):
    """Average ranks, largest first.

    Ties get the average of the positions they occupy, which matters here: half of the
    latents read the next state at essentially zero, and giving those an arbitrary order would
    manufacture agreement or disagreement between the two rankings out of nothing.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    ranked = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or ranked[index] != ranked[start]:
            ranks[order[start:index]] = 0.5 * (start + index - 1)
            start = index
    return ranks


def _rank_correlation(first, second):
    """Pearson correlation of two sets of ranks: the Spearman correlation."""
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.sqrt((first * first).sum() * (second * second).sum())
    return float((first * second).sum() / denominator) if denominator > 0 else float("nan")


# --------------------------------------------------------------------------------- figures


def perturb_figure(effects, top, curves, doubling, heat, lead_days, n_sites,
                   out_path, dpi=150):
    """Left: how much each direction matters. Centre: how fast a push grows. Right: where it
    has got to."""
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4))

    ranked = np.sort(effects)[::-1]
    axes[0].plot(np.arange(1, len(ranked) + 1), ranked, color=TRAIN_COLOUR, lw=1.8)
    axes[0].scatter([1], [ranked[0]], color=TOP_COLOUR, zorder=3, s=28)
    axes[0].annotate(f"latent {top}", xy=(1, ranked[0]), xytext=(20, ranked[0] * 0.75),
                     fontsize=9, color=TOP_COLOUR,
                     arrowprops=dict(arrowstyle="-|>", color=TOP_COLOUR, linewidth=1.2))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("latent, ranked")
    axes[0].set_ylabel("change in the predicted next state")
    axes[0].set_title("which directions matter", fontsize=10)
    axes[0].grid(alpha=0.3, which="both")

    for label, size, colour, style in curves:
        axes[1].plot(lead_days, size, color=colour, lw=1.9, linestyle=style, label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("lead time (days)")
    axes[1].set_ylabel("difference between the two forecasts")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3, which="both")
    axes[1].set_title(f"the strongest push doubles every {doubling[0]:.1f} days",
                      fontsize=10)

    image = axes[2].imshow(heat, aspect="auto", origin="lower", cmap="magma",
                           extent=[0, n_sites, lead_days[0], lead_days[-1]])
    axes[2].set_xlabel("site")
    axes[2].set_ylabel("lead time (days)")
    axes[2].set_title("where the difference is", fontsize=10)
    fig.colorbar(image, ax=axes[2], label="difference")

    fig.text(0.5, 0.01,
             "left: each of the SAE's directions pushed at every site, and how far the "
             "predicted next state moves. centre: one push at one site, then the\nmodel runs "
             "on its own -- the difference grows until it saturates. right: the same push, "
             "site by site; a local change does not stay local.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(
        description="Push one internal feature and measure the consequence.")
    parser.add_argument("--data", default="data/lorenz96.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--model-name", default="emulator")
    parser.add_argument("--push-fraction", type=float, default=PUSH_FRACTION)
    parser.add_argument("--starts", type=int, default=N_STARTS)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset = load_dataset(args.data)
    settings = dataset["settings"]
    n_sites = settings["N"]
    model, checkpoint = load_emulator(Path(args.model_dir) / f"{args.model_name}.pt")
    sae_checkpoint = torch.load(Path(args.model_dir) / "sae.pt", weights_only=False)
    sae_config = sae_checkpoint["config"]
    sae = SparseAutoencoder(sae_config["n_input"], sae_config["n_latent"], sae_config["k"])
    sae.load_state_dict(sae_checkpoint["state_dict"])
    sae.eval()
    layer = sae_config["layer"]
    directions = sae.directions()

    # --------------------------------------------------------- 1. which directions matter
    rank_states = dataset["splits"]["val"]["inputs"][:N_RANK_STATES]
    effects = rank_directions(model, layer, rank_states, directions, args.push_fraction)
    order = torch.argsort(effects, descending=True)
    top, inert = int(order[0]), int(order[-1])
    random_direction = F.normalize(
        torch.randn(1, sae_config["n_input"],
                    generator=torch.Generator().manual_seed(args.seed)), dim=1).squeeze(0)

    # What the directions we are about to push actually read, from interpret.py. Latent
    # activations are one number per (state, site), so each latent gets an honest held-out R²
    # for the state the emulator is trying to predict -- no selection, one number per
    # direction, so the two latents below can simply be looked up.
    reads = json.loads((Path(args.model_dir) / "interpret.json").read_text())
    predicting = np.asarray(reads["predicting_latent_r2"])
    best_reader = int(np.argmax(predicting))

    # How much the two orderings have to do with each other. "Moves the prediction" and
    # "reads the next state" are different operations on the same 512 directions -- one is a
    # large push used at every site at once, the other a linear read-out of a single token --
    # so whether a direction that matters is also a direction that means something is a
    # measurement, not a definition. Ranked largest first, so a small rank means "near the
    # top" for both, and the correlation is of the rankings rather than of the raw sizes.
    effect_ranks = _ranks(effects.numpy())
    reading_ranks = _ranks(predicting)
    effect_reading_correlation = _rank_correlation(effect_ranks, reading_ranks)
    by_effect = order[:10].tolist()
    by_reading = np.argsort(-predicting)[:10].tolist()
    overlap = len(set(by_effect) & set(by_reading))

    # --------------------------------------------------------- 2. what a push does
    rng = np.random.default_rng(args.seed)
    starts = dataset["splits"]["test"]["inputs"][:args.starts]
    sites = rng.integers(0, n_sites, size=len(starts))
    candidates = [
        (f"latent {top} (the strongest)", directions[top], HIDDEN_COLOUR, "-"),
        (f"latent {inert} (the weakest)", directions[inert], MIDDLE_COLOUR, "--"),
        ("a random direction", random_direction, BASELINE_COLOUR, ":"),
    ]

    lead_steps = np.arange(1, args.steps + 1)
    lead_days = lead_steps * settings["dt_out"] * TIME_UNIT_IN_DAYS
    curves, runs = [], {}
    for label, direction, colour, style in candidates:
        difference = rollout_difference(model, layer, starts, direction, sites,
                                        args.steps, args.push_fraction)
        size = np.sqrt((difference**2).mean(axis=(1, 2)))
        runs[label] = difference
        curves.append((label, size, colour, style))
    doubling = [doubling_time(lead_days, size) for _, size, _, _ in curves]
    saturated = [float(lead_days[int(np.argmax(size > 0.9 * size[-1]))])
                 for _, size, _, _ in curves]

    top_heat = runs[curves[0][0]].mean(axis=1).T          # (sites, lead steps)

    results = {
        "layer": layer,
        "n_latent": int(sae_config["n_latent"]),
        "push_fraction": args.push_fraction,
        "effect_top": float(effects[top]),
        "effect_median": float(effects.median()),
        "effect_inert": float(effects[inert]),
        "latents_above_half_of_top": int((effects > 0.5 * effects[top]).sum()),
        "top_latents": [[int(i), float(effects[i])] for i in order[:10]],
        "effects": effects.tolist(),
        "reads_next_state": {"the strongest": float(predicting[top]),
                             "the weakest": float(predicting[inert]),
                             "the best reading latent": float(predicting[best_reader]),
                             "the best reading latent index": best_reader,
                             "median": float(np.median(predicting))},
        "effect_vs_reading": {
            "rank correlation": effect_reading_correlation,
            "top ten by effect": [int(i) for i in by_effect],
            "top ten by reading": [int(i) for i in by_reading],
            "shared in both top tens": overlap,
            "effect of the top readers": float(np.median(effects[by_reading])),
            "reading of the top effects": float(np.median(predicting[by_effect])),
        },
        "lead_days": lead_days.tolist(),
        "system_doubling_days": settings.get("system_doubling_days"),
        "curves": {label: size.tolist() for label, size, _, _ in curves},
        "doubling_days": {label: float(value)
                          for (label, _, _, _), value in zip(curves, doubling)},
        "saturated_by_day": {label: value
                             for (label, _, _, _), value in zip(curves, saturated)},
        "initial_difference": {label: float(size[0]) for label, size, _, _ in curves},
        "final_difference": {label: float(size[-1]) for label, size, _, _ in curves},
    }

    model_dir = Path(args.model_dir)
    (model_dir / "perturb.json").write_text(json.dumps(results, indent=2))
    perturb_figure(effects.numpy(), top, curves, doubling, top_heat, lead_days, n_sites,
                   Path(args.figures) / "perturb.png")

    # ---------------------------------------------------------------------- what we found
    print(f"\nlayer '{layer}', {sae_config['n_latent']} directions, push of "
          f"{args.push_fraction:.2f} x the token size")
    print(f"\n1. which directions matter")
    print(f"  the ten strongest: " + ", ".join(
        f"{int(i)} ({effects[i]:.4f})" for i in order[:10]))
    print(f"  median {effects.median():.4f}, weakest {effects[inert]:.4f}; "
          f"{int((effects > 0.5 * effects[top]).sum())} directions above half of the strongest")
    print(f"  what they read, from interpret.py: how well one latent predicts the next state")
    print(f"    latent {top} reads it at R² {predicting[top]:.3f}, latent {inert} at "
          f"{predicting[inert]:.3f}; the best reader is latent {best_reader} at "
          f"{predicting[best_reader]:.3f}, the median {np.median(predicting):.3f}")
    print(f"  do the two orderings agree?")
    print(f"    rank correlation between them {effect_reading_correlation:+.3f}")
    print(f"    the ten strongest:            " + ", ".join(str(int(i)) for i in by_effect))
    print(f"    the ten best readers:         " + ", ".join(str(int(i)) for i in by_reading))
    print(f"    {overlap} of the 512 are in both top tens; the ten best readers push by a "
          f"median {np.median(effects[by_reading]):.4f}\n"
          f"    against {effects.median():.4f} for a median latent, and the ten strongest "
          f"read at a median R² {np.median(predicting[by_effect]):.4f}")

    print(f"\n2. one push, at one site, then the model runs on its own")
    print(f"  {'':<26s}" + "".join(f"{d:>8.1f}" for d in lead_days[::8]))
    for label, size, _, _ in curves:
        print(f"  {label:<26s}" + "".join(f"{v:8.4f}" for v in size[::8]))
    for index, (label, size, _, _) in enumerate(curves):
        reached = saturated[index]
        print(f"  {label:<26s} doubles every {doubling[index]:.2f} days, "
              f"reaches 90% of its final size by day {reached:.1f}")
    print(f"  the system's own errors double every "
          f"{settings.get('system_doubling_days', float('nan')):.2f} days")

    print(f"\nwrote {model_dir / 'perturb.json'}")
    print(f"wrote {Path(args.figures) / 'perturb.png'}")


if __name__ == "__main__":
    main()
