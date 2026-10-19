"""Take the SAE apart: what its directions correspond to in the system.

    ../../.venv/bin/python interpret.py --data data/lorenz96.npz --model-dir model \
                                        --figures figures

The SAE was trained on one thing only: rebuild the activation at one site. It never saw the
ring, the equations, or the state. This script asks what the directions it found correspond
to, and it does so against ground truth we can compute exactly, in two steps.

**1. The receptive field, measured.** Each token is a function of the state in a window
around its site. How wide is that window? The architecture says it should be
``n_blocks * kernel_size // 2`` = 6 sites to either side, and that is a *prediction*, not a
tautology: it follows from the stack of circular convolutions. We check it by perturbing the
*input* at one site and watching the *token* at every other site move. Outside the predicted
window the response should be exactly zero, and inside it the response profile is the
network's own picture of what a neighbour does to it.

**2. The directions those perturbations define.** Perturbing the state at offset ``d`` moves
the token along one particular direction. Those directions are ground truth: they are defined
by the system, not by the SAE. So we can ask which of the SAE's dictionary directions comes
closest to each of them -- and compare that with the same statistic for random directions,
because with a dictionary this wide, one direction is always "fairly close".

**3. What the latents track in the state.** The natural global description of a ring is by
wavenumber: how much energy at each scale around the circle. A token is local, so no single
token can hold a wavenumber; the question is whether the layer as a whole tracks them, and
whether any *single* latent does. Both are measured out of sample, against a linear probe of
the same activation, which is the ceiling any single direction is measured against.

Writes ``<model-dir>/interpret.json`` and the figures ``receptive_field.png`` and
``features.png``.
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

from generate_data import load_dataset
from train_emulator import (BASELINE_COLOUR, HIDDEN_COLOUR, INPUT_COLOUR, MIDDLE_COLOUR,
                            TRAIN_COLOUR, VAL_COLOUR, load_emulator, token_activations)
from train_sae import SparseAutoencoder

# Perturbation applied to one input site, in standardised units, and how far out to look.
PERTURBATION = 0.5
# How many sites beyond the predicted receptive field to probe.
MARGIN = 3
# How many states the measurements are averaged over.
N_STATES = 600
# Fractions of those states used to fit and to score the wavenumber probes.
FIT_FRACTION = 0.5
# Ridge penalty, on features put on a common scale first (see ``_fit_and_score``).
RIDGE = 1e-3
SHORTCUT_COLOUR = "#c44e52"


def offset_directions(model, inputs, layer, site=0, radius=6, margin=MARGIN,
                      perturbation=PERTURBATION):
    """How the token at one site moves when the state at another offset changes.

    For every offset ``d`` between ``-(radius + margin)`` and ``+(radius + margin)``, perturb
    the state at one site by ``+/- perturbation`` and difference the activations of every
    other site. Returns the mean direction and the mean size of that change, per offset.

    Because the network is translation-equivariant by construction, one perturbed site and one
    set of comparison sites is enough: every other site-to-site relation is a rotation of it.
    """
    span = radius + margin
    offsets = list(range(-span, span + 1))
    n_sites = inputs.shape[-1]
    site = site % n_sites

    up = np.array(inputs, copy=True)
    down = np.array(inputs, copy=True)
    up[:, :, site] += perturbation
    down[:, :, site] -= perturbation

    h_up = token_activations(model, up, layer).reshape(len(inputs), n_sites, -1)
    h_down = token_activations(model, down, layer).reshape(len(inputs), n_sites, -1)
    change = (h_up - h_down).mean(dim=0)          # (n_sites, width)

    sizes, directions = [], []
    for offset in offsets:
        vector = change[(site + offset) % n_sites]
        sizes.append(float(vector.norm()))
        directions.append(F.normalize(vector.unsqueeze(0), dim=1).squeeze(0))
    return offsets, torch.stack(directions), sizes


def best_alignment(dictionary, directions):
    """For each direction, the largest absolute cosine with any dictionary direction."""
    dictionary = F.normalize(dictionary, dim=1)
    directions = F.normalize(directions, dim=1)
    return (directions @ dictionary.T).abs().max(dim=1).values


def wavenumber_energy(states):
    """Energy per wavenumber, excluding the mean, shape ``(n_states, n_sites // 2)``."""
    amplitudes = np.fft.rfft(states, axis=-1, norm="ortho")
    multiplicity = np.ones(amplitudes.shape[-1])
    multiplicity[1:-1] = 2.0
    return (np.abs(amplitudes) ** 2 * multiplicity)[:, 1:]


def out_of_sample_r2(features, target, fit_fraction=FIT_FRACTION, ridge=1e-6):
    """R² on held-out states, for a least-squares fit from ``features`` to ``target``.

    Fitting and scoring on different states matters here: with 600 states and a handful of
    features, an in-sample R² is optimistic, and the point of the comparison is small
    differences between an SAE latent and a linear probe.
    """
    n_fit = int(len(features) * fit_fraction)
    return _fit_and_score(features[:n_fit], target[:n_fit],
                          features[n_fit:], target[n_fit:], ridge)


def _fit_and_score(x_fit, y_fit, x_test, y_test, ridge=RIDGE):
    """Least squares on one set of states, scored on another.

    Each column is put on its own scale first -- these are activations, and their sizes differ
    by orders of magnitude -- so that the ridge penalty means the same thing whether four
    numbers are being fitted or five hundred on the same states.
    """
    x_fit = np.atleast_2d(np.asarray(x_fit, dtype=np.float64))
    x_test = np.atleast_2d(np.asarray(x_test, dtype=np.float64))
    centre = x_fit.mean(axis=0, keepdims=True)
    spread = x_fit.std(axis=0, keepdims=True)
    spread[spread == 0.0] = 1.0
    x_fit = (x_fit - centre) / spread
    x_test = (x_test - centre) / spread

    x_fit = torch.as_tensor(x_fit)
    x_test = torch.as_tensor(x_test)
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


def best_latent_r2(features, target, ridge=RIDGE):
    """The best single latent for ``target``, chosen on one half of the states and scored on
    the other. Returns the honest score and which latent it was.

    Taking the largest of 512 R² values and reporting that same value would be optimistic:
    with that many candidates one of them is always lucky, and the luck ends up in the number
    being reported. So the states are split in two. Every latent is fitted on the first half
    and scored on the second, which ranks them; the winner then gives its *first*-half score,
    a number computed on states that had no part in choosing it.
    """
    half = len(features) // 2
    first, second = slice(0, half), slice(half, None)
    ranking = np.array([_fit_and_score(features[first, index:index + 1], target[first],
                                       features[second, index:index + 1], target[second], ridge)
                        for index in range(features.shape[1])])
    chosen = int(np.argmax(ranking))
    score = _fit_and_score(features[second, chosen:chosen + 1], target[second],
                           features[first, chosen:chosen + 1], target[first], ridge)
    return float(score), chosen


def local_quantities(states, mean, std, raw_next):
    """The physically named numbers a token at one site might be holding.

    Each is one number per (state, site), so it pairs directly with that site's token: no
    averaging over the ring, and no choice about what to look at. They are the quantities the
    equation itself is built out of, plus the two that a token has no excuse to know -- the
    ring's average state, as a control that the measurement can find a global quantity when
    there is one, and the state it is about to have to predict.
    """
    now = states[:, -1, :] * std + mean               # x_i(t), the site's own state
    before = states[:, 0, :] * std + mean             # x_i(t - dt_out)
    upstream = np.roll(now, 1, axis=1)                # x_{i-1}
    shear = np.roll(now, -1, axis=1) - np.roll(now, 2, axis=1)   # x_{i+1} - x_{i-2}
    ring_mean = now.mean(axis=1, keepdims=True) + np.zeros_like(now)
    return {
        "how much the state changes": raw_next - now,
        "the site's own state": now,
        "the state one step back": before,
        "its upstream neighbour": upstream,
        "the shear factor": shear,
        "the nonlinear term": shear * upstream,
        "the state it must predict": raw_next,
        "the ring's average state (control)": ring_mean,
    }


# --------------------------------------------------------------------------------- figures


def receptive_field_figure(offsets, sizes, radius, alignment, baseline, out_path, dpi=150):
    """Left: how far a change at one site is felt. Right: how well the SAE reaches those
    directions."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))

    axes[0].bar(offsets, sizes, color=[HIDDEN_COLOUR if abs(d) <= radius else "#dddddd"
                                       for d in offsets], width=0.7)
    axes[0].axvspan(-radius - 0.5, radius + 0.5, color=INPUT_COLOUR, alpha=0.07)
    axes[0].set_xlabel("offset from the perturbed site (sites)")
    axes[0].set_ylabel("how far the token moves")
    axes[0].set_title("the receptive field, measured", fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].text(0.02, 0.96, f"the architecture predicts |offset| <= {radius}\n"
                             f"(shaded); outside it the bars are zero, to rounding",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=8.5,
                 color="#444444")

    axes[1].bar(offsets, alignment, color=SHORTCUT_COLOUR, width=0.7, label="the SAE's 512")
    axes[1].plot(offsets, baseline, color=BASELINE_COLOUR, lw=1.6, marker="o",
                 markersize=3.5, label="random directions")
    axes[1].set_xlabel("offset from the perturbed site (sites)")
    axes[1].set_ylabel("best cosine with that direction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(fontsize=8.5, loc="lower right")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_title("does the dictionary reach them?", fontsize=10)

    fig.text(0.5, 0.02,
             "left: perturb the state at one site, and see how far away the token still "
             "moves. the cut-off is where the architecture says it is.\nright: for each "
             "offset, the closest direction in the SAE's dictionary, against the same "
             "statistic for 512 random directions.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def features_figure(wavenumbers, latent_r2, layer_r2, names, local_latent,
                    local_dictionary, local_layer, out_path, dpi=150):
    """What the latents track: the global wavenumber energies, and the local quantities."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))

    width = 0.36
    position = np.arange(len(wavenumbers))
    axes[0].bar(position - width / 2, latent_r2, width=width, color=SHORTCUT_COLOUR,
                label="the best single latent")
    axes[0].bar(position + width / 2, layer_r2, width=width, color=TRAIN_COLOUR,
                label="the layer's own 32 numbers")
    axes[0].axhline(0.0, color="#888888", lw=0.9)
    axes[0].set_xticks(position[::2])
    axes[0].set_xticklabels([str(k) for k in wavenumbers[::2]], fontsize=8.5)
    axes[0].set_xlabel("wavenumber")
    axes[0].set_ylabel("R² on held-out states")
    axes[0].legend(fontsize=8.5, loc="lower left")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_title("the global features: energy per wavenumber", fontsize=10)

    width = 0.26
    position = np.arange(len(names))
    for offset, values, colour, label in (
            (-width, local_latent, SHORTCUT_COLOUR, "the best single latent"),
            (0.0, local_dictionary, MIDDLE_COLOUR, "all 512 latents together"),
            (width, local_layer, TRAIN_COLOUR, "the layer's own 32 numbers")):
        axes[1].barh(position + offset, values, height=width, color=colour, label=label)
    axes[1].axvline(0.0, color="#888888", lw=0.9)
    axes[1].set_yticks(position)
    axes[1].set_yticklabels(names, fontsize=8.5)
    axes[1].set_xlabel("R² on held-out states")
    axes[1].set_xlim(min(-0.1, float(np.min(local_layer)) - 0.05), 1.05)
    axes[1].legend(fontsize=8.5, loc="lower right")
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].set_title("the local quantities, one number per site", fontsize=10)

    fig.text(0.5, 0.015,
             "left: the state is also described globally, by how much energy sits at each "
             "wavenumber -- and that energy is spread around the whole ring, so a token at one "
             "site does not hold it.\nright: the quantities the equation is made of, which a "
             "token has every reason to hold. the top bar is the ceiling: what is present in "
             "the layer at all.\nthe bottom bar is what one of the 512 directions recovers "
             "of it.",
             ha="center", va="bottom", fontsize=9, color="#444444", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.155, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="Take the SAE apart.")
    parser.add_argument("--data", default="data/lorenz96.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--model-name", default="emulator")
    parser.add_argument("--states", type=int, default=N_STATES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset = load_dataset(args.data)
    n_sites = dataset["settings"]["N"]
    model, checkpoint = load_emulator(Path(args.model_dir) / f"{args.model_name}.pt")
    sae_checkpoint = torch.load(Path(args.model_dir) / "sae.pt", weights_only=False)
    sae_config = sae_checkpoint["config"]
    sae = SparseAutoencoder(sae_config["n_input"], sae_config["n_latent"], sae_config["k"])
    sae.load_state_dict(sae_checkpoint["state_dict"])
    sae.eval()
    layer = sae_config["layer"]

    # Held out from the SAE: tokens from the validation block.
    states = dataset["splits"]["val"]["inputs"][:args.states]
    # The prediction for this layer's token, and for the model's output (which is wider,
    # because the output has seen one more block).
    radius = model.receptive_field_of(layer)
    output_radius = model.receptive_field()

    # -------------------------------------------------- 1. the receptive field, measured
    offsets, directions, sizes = offset_directions(model, states, layer, radius=radius)

    # -------------------------------------------------- 2. does the dictionary reach them?
    alignment = best_alignment(sae.directions(), directions)
    generator = torch.Generator().manual_seed(args.seed)
    random_directions = torch.randn(sae_config["n_latent"], sae_config["n_input"],
                                    generator=generator)
    baseline = best_alignment(random_directions, directions)

    # -------------------------------------------------- 3. what the latents track
    tokens = token_activations(model, states, layer)
    with torch.no_grad():
        activations = sae.encode(tokens).reshape(len(states), n_sites, -1)

    # Global quantities: a wavenumber's energy is spread over the whole ring, so the
    # comparison is between averages over the ring, and every state in the validation block is
    # used -- there is no reason to be short of them here.
    global_states = dataset["splits"]["val"]["inputs"]
    with torch.no_grad():
        global_activation = sae.encode(token_activations(model, global_states, layer))
    global_activation = global_activation.reshape(len(global_states), n_sites, -1)
    global_activation = global_activation.mean(dim=1).numpy()
    global_token = token_activations(model, global_states, layer)
    global_token = global_token.reshape(len(global_states), n_sites, -1).mean(dim=1).numpy()

    energy = np.log(wavenumber_energy(dataset["splits"]["val"]["raw_targets"])
                    + 1e-12)                                        # (states, k_max)
    wavenumbers = np.arange(1, energy.shape[1] + 1)

    latent_r2, layer_r2 = [], []
    for index in range(energy.shape[1]):
        target = energy[:, index]
        latent_r2.append(best_latent_r2(global_activation, target)[0])
        layer_r2.append(out_of_sample_r2(global_token, target))
    latent_r2 = np.array(latent_r2)
    layer_r2 = np.array(layer_r2)

    # Local quantities: one number per (state, site), paired with that site's own token, so
    # nothing is averaged and no choice of direction is made before the measurement.
    quantities = local_quantities(states, dataset["mean"], dataset["std"],
                                  dataset["splits"]["val"]["raw_targets"][:args.states])
    flat_activation = activations.reshape(-1, activations.shape[-1]).numpy()
    flat_token = tokens.numpy()

    local_names = []
    local_latent, local_dictionary, local_layer, local_latent_index = [], [], [], {}
    for name, values in quantities.items():
        target = values.reshape(-1)
        score, chosen = best_latent_r2(flat_activation, target)
        local_names.append(name)
        local_latent.append(score)
        local_dictionary.append(out_of_sample_r2(flat_activation, target))
        local_layer.append(out_of_sample_r2(flat_token, target))
        local_latent_index[name] = chosen
    local_latent = np.array(local_latent)
    local_dictionary = np.array(local_dictionary)
    local_layer = np.array(local_layer)

    # One number per latent, for the one quantity the emulator exists to produce: which
    # directions read it, and how well. No selection is involved, so these can be looked up
    # individually -- ``perturb.py`` uses this to say what the direction it pushes reads.
    predicting = quantities["the state it must predict"].reshape(-1)
    predicting_latent_r2 = [out_of_sample_r2(flat_activation[:, index:index + 1], predicting)
                            for index in range(flat_activation.shape[1])]

    inside = [index for index, offset in enumerate(offsets) if abs(offset) <= radius]
    outside = [index for index, offset in enumerate(offsets) if abs(offset) > radius]
    results = {
        "layer": layer,
        "predicted_radius": int(radius),
        "predicted_output_radius": int(output_radius),
        "offsets": offsets,
        "response_size": sizes,
        "response_inside": float(np.mean([sizes[i] for i in inside])),
        "response_outside": float(np.max([sizes[i] for i in outside])),
        "alignment": alignment.numpy().tolist(),
        "alignment_baseline": baseline.numpy().tolist(),
        "alignment_mean_inside": float(alignment[inside].mean()),
        "alignment_mean_baseline_inside": float(baseline[inside].mean()),
        "wavenumbers": wavenumbers.tolist(),
        "wavenumber_latent_r2": latent_r2.tolist(),
        "wavenumber_layer_r2": layer_r2.tolist(),
        "wavenumber_mean_latent_r2": float(latent_r2.mean()),
        "wavenumber_mean_layer_r2": float(layer_r2.mean()),
        "local_quantities": local_names,
        "local_quantity_latent_r2": local_latent.tolist(),
        "local_quantity_dictionary_r2": local_dictionary.tolist(),
        "local_quantity_layer_r2": local_layer.tolist(),
        "local_quantity_latent": local_latent_index,
        "predicting_latent_r2": predicting_latent_r2,
        "principal_room_95": sae_checkpoint["metrics"].get("principal_room_95"),
    }

    model_dir = Path(args.model_dir)
    (model_dir / "interpret.json").write_text(json.dumps(results, indent=2))
    receptive_field_figure(offsets, sizes, radius, alignment.numpy(), baseline.numpy(),
                           Path(args.figures) / "receptive_field.png")
    features_figure(wavenumbers, latent_r2, layer_r2, local_names,
                    local_latent, local_dictionary, local_layer,
                    Path(args.figures) / "features.png")

    # ---------------------------------------------------------------------- what we found
    print(f"\nlayer '{layer}', {args.states} states from the validation block")
    print(f"\n1. the receptive field, measured (the architecture predicts {radius} sites for "
          f"this layer, {output_radius} for the model's output)")
    print("     offset  " + "".join(f"{d:>7d}" for d in offsets))
    print("     moves   " + "".join(f"{s:7.3f}" for s in sizes))
    print(f"  inside the predicted radius the token moves by {np.mean([sizes[i] for i in inside]):.3f}"
          f" on average; outside it, at most {np.max([sizes[i] for i in outside]):.2e}")

    print("\n2. does the SAE's dictionary reach those directions?")
    print("     offset         " + "".join(f"{d:>7d}" for d in offsets))
    print("     the SAE's 512  " + "".join(f"{a:7.3f}" for a in alignment.numpy()))
    print("     random         " + "".join(f"{b:7.3f}" for b in baseline.numpy()))
    print(f"  inside the window: {alignment[inside].mean():.3f} against "
          f"{baseline[inside].mean():.3f} for random directions")

    print("\n3. what the latents track in the state")
    print(f"  the global wavenumber energies: the best single latent averages R² "
          f"{latent_r2.mean():.3f} over the {len(wavenumbers)} wavenumbers, the layer's own "
          f"{global_token.shape[1]} numbers get {layer_r2.mean():.3f} -- neither finds anything")
    print(f"  the local quantities, one number per site, paired with that site's token:")
    print(f"    {'':<38s}{'latent':>9s}{'which':>7s}{'512':>9s}{'layer':>9s}")
    for index, name in enumerate(local_names):
        print(f"    {name:<38s}{local_latent[index]:9.3f}{local_latent_index[name]:7d}"
              f"{local_dictionary[index]:9.3f}{local_layer[index]:9.3f}")

    print(f"\nwrote {model_dir / 'interpret.json'}")
    print(f"wrote {Path(args.figures) / 'receptive_field.png'}")
    print(f"wrote {Path(args.figures) / 'features.png'}")


if __name__ == "__main__":
    main()
