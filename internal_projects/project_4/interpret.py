"""Ask what the emulator's tokens mean, against answers the equations supply.

    ../../.venv/bin/python interpret.py --data data/gray_scott.npz --model-dir model \
                                        --figures figures

Three measurements, each against a ground truth that comes from the system rather than from the
sparse autoencoder.

**1. The receptive field, measured.** The architecture predicts that a token at ``blocks.2`` has
passed through blocks reaching 1, 2 and 4 cells out, so it can be affected by the seven cells to
either side and by no others. That is checkable: perturb the fields at **one cell** and watch the
token move, cell by cell. Outside the predicted window the response should be exactly zero, and
inside it the profile is the network's own picture of what a neighbour does to it.

**2. What each latent reads.** The quantities the equation is built from, one number per cell,
each paired with that cell's own token: the two fields, the reaction, the tendency of ``v``, the
curvature, how much spot is in the neighbourhood, how far the nearest blob is, and the grid's
average as a control that a global quantity is absent. For every one of them the same three
things are measured: the best *single* latent, all 512 read jointly, and the layer's own
numbers, which is the ceiling. The best single latent is chosen on one half of the states and
scored on the other: taking the largest of 512 values and reporting that same value would be
optimistic, and the optimism would be in the number quoted.

**3. The two events.** A spot appearing and a spot disappearing are what this system *does*, and
a token that knows one is coming is holding something with a name outside the model. Two labels,
taken from the future state and applied to the cell: *a spot is about to appear within a few
cells of here*, and *the spot here is about to die*. These are rare and binary, so a correlation
would be a poor description of how well they are read; they are scored by AUROC, the chance a
positive cell is ranked above a negative one. The reference is not zero but the thing an
observer would use: how much spot is already in the neighbourhood. A latent that cannot beat
looking at the patch is not telling us anything about the future.

Writes ``<model-dir>/interpret.json`` and ``<figures>/receptive_field.png`` and
``<figures>/features.png``.
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

from generate_data import SPOT_THRESHOLD, laplacian, load_dataset, spot_centres
from train_emulator import (BASELINE_COLOUR, HIDDEN_COLOUR, INPUT_COLOUR, MIDDLE_COLOUR,
                            TRAIN_COLOUR, VAL_COLOUR, load_emulator, token_activations)
from train_sae import SparseAutoencoder

# Perturbation applied to one input cell, in standardised units, and how far out to look.
PERTURBATION = 0.5
# How many cells beyond the predicted receptive field to probe.
MARGIN = 4
# Ridge penalty, on features put on a common scale first (see ``_fit_and_score``).
RIDGE = 1e-3
# The patch used for the "how much spot is around here" quantities, and the radius within which
# a new blob counts as having appeared next to a cell.
PATCH_RADIUS = 5
EVENT_RADIUS = 5
SHORTCUT_COLOUR = "#c44e52"


# ------------------------------------------------------------------- ground truth quantities


def box_mean(field, radius):
    """The average of ``field`` over a square patch, on the periodic grid.

    One roll per offset, which is slower than a cumulative sum and much easier to check: the
    patch wraps at the edges like everything else here.
    """
    total = np.zeros_like(field)
    for dy in range(-radius, radius + 1):
        rolled = np.roll(field, dy, axis=-2)
        for dx in range(-radius, radius + 1):
            total += np.roll(rolled, dx, axis=-1)
    return total / ((2 * radius + 1) ** 2)


def distance_to_nearest(field_shape, centres):
    """Distance from every cell to the nearest of ``centres``, on the periodic grid."""
    n = field_shape[0]
    rows, cols = np.mgrid[0:n, 0:n]
    best = np.full((n, n), np.inf)
    for centre_row, centre_col in centres:
        dy = np.minimum((rows - centre_row) % n, (centre_row - rows) % n)
        dx = np.minimum((cols - centre_col) % n, (centre_col - cols) % n)
        best = np.minimum(best, np.sqrt(dy**2 + dx**2))
    return best


def local_quantities(states, next_states, F, k):
    """The physically named numbers a token at one cell might be holding.

    Each is one number per (state, cell), so it pairs directly with that cell's token, with
    nothing averaged over the grid and no choice of direction made before the measurement. They
    are the quantities the equation is built from, plus the two structural ones -- how much
    spot is in the neighbourhood, and how far the middle of the nearest blob is -- plus the
    grid's average as a control.
    """
    u, v = states[:, 0], states[:, 1]
    next_v = next_states[:, 1]
    reaction = u * v * v
    centres = [spot_centres(field) for field in v]
    patch = np.stack([box_mean(field, PATCH_RADIUS) for field in v])
    patch_coverage = np.stack([box_mean(field > SPOT_THRESHOLD, PATCH_RADIUS) for field in v])
    nearest = np.stack([distance_to_nearest(field.shape, found)
                        for field, found in zip(v, centres)])
    grid_mean = v.mean(axis=(1, 2), keepdims=True) + np.zeros_like(v)
    return {
        "the field u": u,
        "the field v": v,
        "the reaction term u*v^2": reaction,
        "what v is doing right now": reaction - (F + k) * v,
        "the curvature of v": np.stack([laplacian(field) for field in v]),
        "how much spot is around here": patch,
        "how far the nearest blob is": nearest,
        "how worried the patch is (its coverage)": patch_coverage,
        "what v will do next": next_v - v,
        "the grid's average (control)": grid_mean,
    }, centres


def event_labels(centres, next_centres, shape, radius=EVENT_RADIUS):
    """Where a blob appears, and where one dies, in the next call.

    A blob is "new" when nothing at the previous state was within ``radius`` of its middle, and
    a cell is labelled when it is within ``radius`` of such a middle. Deaths are the same the
    other way round. Both return a boolean per (state, cell).
    """
    n = shape[0]
    born = np.zeros((len(centres), n, n), dtype=bool)
    died = np.zeros((len(centres), n, n), dtype=bool)
    for index in range(len(centres)):
        before = centres[index]
        after = next_centres[index]
        for new in after:
            if all(np.hypot(new[0] - old[0], new[1] - old[1]) > radius for old in before):
                born[index] |= distance_to_nearest((n, n), [new]) <= radius
        for old in before:
            if all(np.hypot(old[0] - new[0], old[1] - new[1]) > radius for new in after):
                died[index] |= distance_to_nearest((n, n), [old]) <= radius
    return {"a blob is about to be born here": born,
            "the blob here is about to die": died}


# -------------------------------------------------------------------------- honest read-outs


def _ranks(values):
    """Average ranks, largest first, so that ties do not get an arbitrary order.

    Ties matter here: a top-k latent is exactly zero for most cells, so nearly every cell shares
    its score, and a ranking that broke those ties by accident would invent structure. The ties
    are handled by finding the blocks of equal values and ranking each block at once: with a few
    hundred blocks among a million cells, that is the difference between a second and ten minutes.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    ranked = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    edges = np.flatnonzero(ranked[1:] != ranked[:-1]) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(values)]))
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks


def auroc(scores, labels):
    """The chance a positive cell is ranked above a negative one.

    Computed from ranks, which handles the huge tie groups a sparse latent produces without
    pretending they are distinguishable.
    """
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _ranks(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fit_and_score(x_fit, y_fit, x_test, y_test, ridge=RIDGE):
    """Least squares on one set of states, scored on another.

    Each column is put on its own scale first -- these are activations, and their sizes differ by
    orders of magnitude -- so that the ridge penalty means the same thing whether eight numbers
    are being fitted or five hundred on the same states.
    """
    x_fit = np.atleast_2d(np.asarray(x_fit, dtype=np.float64))
    x_test = np.atleast_2d(np.asarray(x_test, dtype=np.float64))
    centre = x_fit.mean(axis=0, keepdims=True)
    spread = x_fit.std(axis=0, keepdims=True)
    spread[spread == 0.0] = 1.0
    x_fit = torch.as_tensor((x_fit - centre) / spread)
    x_test = torch.as_tensor((x_test - centre) / spread)
    y_fit = torch.as_tensor(y_fit, dtype=torch.float64).reshape(-1, 1)
    y_test = torch.as_tensor(y_test, dtype=torch.float64).reshape(-1, 1)

    design_fit = torch.cat([x_fit, torch.ones(len(x_fit), 1, dtype=torch.float64)], dim=1)
    design_test = torch.cat([x_test, torch.ones(len(x_test), 1, dtype=torch.float64)], dim=1)
    gram = design_fit.T @ design_fit + ridge * torch.eye(design_fit.shape[1],
                                                        dtype=torch.float64)
    coefficients = torch.linalg.solve(gram, design_fit.T @ y_fit)
    prediction = design_test @ coefficients
    residual = ((y_test - prediction) ** 2).sum()
    total = ((y_test - y_test.mean()) ** 2).sum()
    return float(1.0 - residual / total)


def out_of_sample_r2(features, target, ridge=RIDGE):
    """R² on held-out cells, for a least-squares fit from ``features`` to ``target``."""
    half = len(features) // 2
    return _fit_and_score(features[:half], target[:half], features[half:], target[half:], ridge)


def best_latent_r2(features, target, ridge=RIDGE):
    """The best single latent for ``target``, chosen on one half and scored on the other.

    Returns the honest score and which latent it was. The choice is made with the scores each
    latent gets on the second half, and the number reported is that latent's score on the first:
    a value computed on states that had no part in choosing it.
    """
    half = len(features) // 2
    ranking = np.array([_fit_and_score(features[:half, index:index + 1], target[:half],
                                       features[half:, index:index + 1], target[half:], ridge)
                        for index in range(features.shape[1])])
    chosen = int(np.argmax(ranking))
    score = _fit_and_score(features[half:, chosen:chosen + 1], target[half:],
                           features[:half, chosen:chosen + 1], target[:half], ridge)
    return float(score), chosen


# --------------------------------------------------------------------------------- figures


def receptive_field_figure(offsets, sizes, radius, out_path, dpi=150):
    """How far a change at one cell is felt, cell by cell."""
    fig, axis = plt.subplots(figsize=(8.4, 4.0))
    axis.bar(offsets, sizes, color=[HIDDEN_COLOUR if abs(offset) <= radius else "#dddddd"
                                    for offset in offsets], width=0.7)
    axis.axvspan(-radius - 0.5, radius + 0.5, color=INPUT_COLOUR, alpha=0.07)
    axis.set_xlabel("offset from the perturbed cell, cells")
    axis.set_ylabel("how far the token moves")
    axis.set_title(f"the receptive field at this layer, measured against a prediction of "
                   f"{radius}", fontsize=10)
    axis.grid(axis="y", alpha=0.3)
    axis.text(0.02, 0.96, f"the architecture predicts |offset| <= {radius} (shaded); outside it\n"
                          f"the response is zero to machine precision",
              transform=axis.transAxes, ha="left", va="top", fontsize=8.5, color="#444444")
    fig.text(0.5, 0.02,
             "the fields are perturbed at one cell, and the token is read at every other cell. "
             "whether the model uses all of the window the\narchitecture gives it is a different "
             "question -- and the profile is the answer, not the cut-off.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def features_figure(names, local_latent, local_dictionary, local_layer, events, event_auroc,
                    event_reference, out_path, dpi=150):
    """Left: what one latent can hold. Right: whether one can see an event coming."""
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))

    width = 0.26
    position = np.arange(len(names))
    for offset, values, colour, label in (
            (-width, local_latent, SHORTCUT_COLOUR, "the best single latent"),
            (0.0, local_dictionary, MIDDLE_COLOUR, "all 512 latents together"),
            (width, local_layer, TRAIN_COLOUR, "the layer's own numbers")):
        axes[0].barh(position + offset, values, height=width, color=colour, label=label)
    axes[0].axvline(0.0, color="#888888", lw=0.9)
    axes[0].set_yticks(position)
    axes[0].set_yticklabels(names, fontsize=8.5)
    axes[0].set_xlabel("R² on held-out cells")
    axes[0].set_xlim(min(-0.1, float(np.min(local_layer)) - 0.05), 1.05)
    axes[0].legend(fontsize=8.5, loc="lower right")
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].set_title("what a token holds, and what one direction gets", fontsize=10)

    position = np.arange(len(events))
    width = 0.28
    axes[1].bar(position - width, event_auroc, width=width, color=SHORTCUT_COLOUR,
                label="the best single latent")
    axes[1].bar(position, event_reference, width=width, color=BASELINE_COLOUR,
                label="how much spot is already here")
    axes[1].axhline(0.5, color="#888888", lw=1.0, linestyle=":")
    axes[1].set_xticks(position)
    axes[1].set_xticklabels([name.replace(" here", "\nhere") for name in events], fontsize=9)
    axes[1].set_ylabel("AUROC on held-out cells")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend(fontsize=8.5, loc="upper right")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_title("can a direction see it coming?", fontsize=10)

    fig.text(0.5, 0.02,
             "left: the quantities the equation is made of, which a token has every reason to "
             "hold, plus the grid average as a control. the top bar is the ceiling -- what is\n"
             "in the layer at all -- and the bottom bar is what one of the 512 directions "
             "recovers of it. right: 0.5 is a coin toss; the grey bar is what an observer\n"
             "would use, namely how much spot is in the patch already.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="Ask what the emulator's tokens mean.")
    parser.add_argument("--data", default="data/gray_scott.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--model-name", default="emulator")
    parser.add_argument("--states", type=int, default=234,
                        help="validation states to measure over; they are never trained on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset = load_dataset(args.data)
    settings = dataset["settings"]
    mean, std = dataset["mean"], dataset["std"]
    model, checkpoint = load_emulator(Path(args.model_dir) / f"{args.model_name}.pt")
    sae_checkpoint = torch.load(Path(args.model_dir) / "sae.pt", weights_only=False)
    sae_config = sae_checkpoint["config"]
    sae = SparseAutoencoder(sae_config["n_input"], sae_config["n_latent"], sae_config["k"])
    sae.load_state_dict(sae_checkpoint["state_dict"])
    sae.eval()
    layer = sae_config["layer"]
    n_sites = settings["N"]

    # Held out from the SAE: tokens from the validation block.
    val = dataset["splits"]["val"]
    states = val["inputs"][:args.states]
    physical = states[:, -1] * std[None, :, None, None] + mean[None, :, None, None]
    next_physical = val["raw_targets"][:args.states]
    radius = model.receptive_field_of(layer)

    # -------------------------------------------------- 1. the receptive field, measured
    # The perturbation is applied in standardised units, which is the space the emulator works in:
    # half a standard deviation of each field, at one cell. In the fields' own units that is a
    # change of 0.5 * std, which is what the measurement is reported as.
    up = np.array(states, copy=True)
    down = np.array(states, copy=True)
    up[:, :, :, n_sites // 2, n_sites // 2] += PERTURBATION
    down[:, :, :, n_sites // 2, n_sites // 2] -= PERTURBATION
    h_up = token_activations(model, up, layer, batch_size=8)
    h_down = token_activations(model, down, layer, batch_size=8)
    change = (h_up - h_down).reshape(len(states), n_sites, n_sites, -1).mean(dim=0)
    span = radius + MARGIN
    offsets = list(range(-span, span + 1))
    sizes = [float(change[n_sites // 2, (n_sites // 2 + offset) % n_sites].norm())
             for offset in offsets]

    # -------------------------------------------------- 2. what each latent reads
    tokens = token_activations(model, states, layer, batch_size=8)
    with torch.no_grad():
        activations = sae.encode(tokens).reshape(len(states), n_sites, n_sites, -1)
    flat_activation = activations.reshape(-1, activations.shape[-1]).numpy()
    flat_token = tokens.numpy()

    quantities, centres = local_quantities(physical, next_physical, settings["F"], settings["k"])
    next_centres = [spot_centres(field) for field in next_physical[:, 1]]
    local_names, local_latent, local_dictionary, local_layer, local_chosen = [], [], [], [], {}
    for name, values in quantities.items():
        # Every quantity has to be exactly one number per cell, because that is what pairs it with
        # a token. A quantity with channels in it would flatten to a longer vector and silently
        # fit the wrong thing, so it is checked rather than broadcast.
        if values.size != flat_activation.shape[0]:
            raise ValueError(f"quantity {name!r} holds {values.size} numbers for "
                             f"{flat_activation.shape[0]} cells: it is not one per cell")
        target = values.reshape(-1)
        score, chosen = best_latent_r2(flat_activation, target)
        local_names.append(name)
        local_latent.append(score)
        local_dictionary.append(out_of_sample_r2(flat_activation, target))
        local_layer.append(out_of_sample_r2(flat_token, target))
        local_chosen[name] = chosen
    local_latent = np.array(local_latent)
    local_dictionary = np.array(local_dictionary)
    local_layer = np.array(local_layer)

    # -------------------------------------------------- 3. can a direction see an event coming?
    labels = event_labels(centres, next_centres, physical.shape[-2:])
    events, event_auroc, event_reference, event_base_rate = [], [], [], []
    reference = quantities["how much spot is around here"].reshape(-1)
    for name, binary in labels.items():
        target = binary.reshape(-1)
        per_latent = np.array([auroc(flat_activation[:, index], target)
                               for index in range(flat_activation.shape[1])])
        events.append(name)
        event_auroc.append(float(np.nanmax(per_latent)))
        event_reference.append(auroc(reference, target))
        event_base_rate.append(float(target.mean()))

    inside = [index for index, offset in enumerate(offsets) if abs(offset) <= radius]
    outside = [index for index, offset in enumerate(offsets) if abs(offset) > radius]
    results = {
        "layer": layer,
        "predicted_radius": int(radius),
        "offsets": offsets,
        "response_size": sizes,
        "response_inside": float(np.mean([sizes[index] for index in inside])),
        "response_outside": float(np.max([sizes[index] for index in outside])),
        "local_quantities": local_names,
        "local_quantity_latent_r2": local_latent.tolist(),
        "local_quantity_dictionary_r2": local_dictionary.tolist(),
        "local_quantity_layer_r2": local_layer.tolist(),
        "local_quantity_latent": local_chosen,
        "events": events,
        "event_auroc": event_auroc,
        "event_auroc_of_local_spot": event_reference,
        "event_base_rate": event_base_rate,
        "principal_room_95": sae_checkpoint["metrics"].get("principal_room_95"),
    }

    model_dir = Path(args.model_dir)
    (model_dir / "interpret.json").write_text(json.dumps(results, indent=2))
    receptive_field_figure(offsets, sizes, radius, Path(args.figures) / "receptive_field.png")
    features_figure(local_names, local_latent, local_dictionary, local_layer, events,
                    np.array(event_auroc), np.array(event_reference),
                    Path(args.figures) / "features.png")

    # ---------------------------------------------------------------------- what we found
    print(f"\nlayer '{layer}', {args.states} states from the validation block, which the SAE "
          f"never saw")
    print(f"\n1. the receptive field, measured (the architecture predicts {radius} cells)")
    print("     offset " + "".join(f"{offset:>7d}" for offset in offsets))
    print("     moves  " + "".join(f"{size:7.3f}" for size in sizes))
    print(f"  inside the window the token moves by "
          f"{np.mean([sizes[index] for index in inside]):.3f} on average; outside it, at most "
          f"{np.max([sizes[index] for index in outside]):.2e}")

    print(f"\n2. what each latent reads, one number per cell, paired with that cell's token:")
    print(f"    {'':<40s}{'latent':>8s}{'which':>7s}{'512':>8s}{'layer':>8s}")
    for index, name in enumerate(local_names):
        print(f"    {name:<40s}{local_latent[index]:8.3f}{local_chosen[name]:7d}"
              f"{local_dictionary[index]:8.3f}{local_layer[index]:8.3f}")

    print(f"\n3. the two events, scored by AUROC (0.5 is a coin toss):")
    print(f"    {'':<40s}{'latent':>8s}{'spot now':>10s}{'how often':>10s}")
    for index, name in enumerate(events):
        print(f"    {name:<40s}{event_auroc[index]:8.3f}{event_reference[index]:10.3f}"
              f"{event_base_rate[index]:10.4f}")

    print(f"\nwrote {model_dir / 'interpret.json'}")
    print(f"wrote {Path(args.figures) / 'receptive_field.png'}")
    print(f"wrote {Path(args.figures) / 'features.png'}")


if __name__ == "__main__":
    main()
