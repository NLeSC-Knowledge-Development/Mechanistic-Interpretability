"""Take the SAE apart: which direction is which, and what each one does.

    ../../.venv/bin/python interpret.py --data data/dataset.pth --model-dir model \
                                        --figures figures

``train_sae.py`` trained the SAE and asked whether its directions line up with anything we
know. This script answers the same question properly, and it is worth saying why the obvious
method is the wrong one.

Every picture was built by taking a real digit and doing a known set of things to it, and we
recorded the true value of every factor that went in. So the tempting move is to *fit* each
factor's direction: regress ``h`` on the table of true values and take the coefficients. That
is unreliable here. ``h`` is a small bunch of numbers dominated by one thing -- the answer --
and a factor whose true effect is weak will have its fitted direction swamped by whatever
else the fit can grab; the direction comes out pointing at the answer instead of at the
factor, and it comes out looking entirely reasonable. Adding a column of pure noise to the
table and fitting it the same way is a quick way to watch this happen.

So instead we **change the factor and watch what moves**:

    1. take the same digit, the same rotation, the same everything;
    2. draw the picture twice, differing only in one factor -- a warm frame and a cool frame,
       or a turn left and a turn right;
    3. the change in ``h`` between the two pictures is that factor's direction, and nothing
       else's.

Averaged over hundreds of digits, that gives each factor's own direction in ``h`` with no
fitting and no guessing. Drawing the same picture twice gives a change of exactly zero,
which is the check that the measurement is doing what it says.

With those directions in hand, this script then asks the SAE three questions:

* for every factor we know, does any of the SAE's 1024 directions point at it? -- and how
  does that compare with random directions, which is the number that means something;
* which single direction in ``h`` carries the answer, and how much of it is the shortcut?
* and is the shortcut its own direction at all, or has the network merged it with the shape?

Writes:

* ``<model-dir>/interpret.json`` -- every number below
* ``<figures>/recovery.png``     -- one bar per true direction: how well the SAE finds it
* ``<figures>/directions.png``   -- what the answer reads, and whether the shortcut is
                                    a direction of its own
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

from generate_data import COLOUR, FACTORS, render
from train_classifier import Classifier
from train_sae import SparseAutoencoder, factor_directions, middle_layer_activations

# How far each factor is moved, up and down, to measure its direction. Big enough to be seen
# in the picture, small enough to stay in the world the network was trained on.
STEPS = {
    "colour": 0.35,      # on the signed colour strength, which runs from -1 to 1
    "rotation": 8.0,     # degrees
    "zoom": 0.12,
    "shift_x": 0.08,     # in units of half the picture
    "shift_y": 0.08,
    "ink": 0.12,
}
# How many pictures the intervention is averaged over.
N_PAIRS = 1000
# The colours the two features of the story are drawn in: the shape (the task) and the
# shortcut (the thing we did not ask for).
TASK_COLOUR = "#4c72b0"
SHORTCUT_COLOUR = "#c44e52"
NUISANCE_COLOUR = "#b8b8b8"
BASELINE_COLOUR = "#666666"


def intervention_directions(model, dataset, device="cpu", seed=0, n_pairs=N_PAIRS):
    """Every recorded factor's own direction inside ``h``, by changing it and nothing else.

    Returns a tensor of shape ``(len(FACTORS), dim h)``, one unit direction per factor, and
    the size of the change each one makes -- a factor that moves ``h`` a lot is one the
    network holds on to.
    """
    index = dataset["splits"]["test"][:n_pairs]
    source = dataset["source"][index].numpy()
    a = dataset["a"][index].numpy()
    rows, cols = dataset["meta"]["rows"], dataset["meta"]["cols"]

    # The same noise in both members of every pair, so the only difference between them is
    # the factor we moved.
    rng = np.random.default_rng(seed)
    noise = dataset["meta"]["noise"] * rng.normal(size=(len(index), 3, rows, cols))

    directions, sizes = [], []
    for i, name in enumerate(FACTORS):
        up, down = a.copy(), a.copy()
        up[:, i] += STEPS[name]
        down[:, i] -= STEPS[name]
        with torch.no_grad():
            h_up = middle_layer_activations(model, standardise(render(source, up, dataset["meta"]["frame_width"], noise)[0], dataset))
            h_down = middle_layer_activations(model, standardise(render(source, down, dataset["meta"]["frame_width"], noise)[0], dataset))
        change = h_up - h_down
        sizes.append(float(change.norm(dim=1).mean()))
        directions.append(change.mean(dim=0))
    return F.normalize(torch.stack(directions), dim=1), sizes


def standardise(x, dataset):
    """The pictures as the classifier sees them."""
    return (torch.as_tensor(x, dtype=torch.float32) - dataset["mean"]) / dataset["std"]


def class_direction(h, y):
    """The direction in ``h`` that carries the answer: the difference of the class means.

    Measured without any SAE, it is what the classifier's answer looks like from inside the
    network. Everything it answers with has to be somewhere along this one direction.
    """
    return F.normalize(h[y > 0].mean(dim=0) - h[y <= 0].mean(dim=0), dim=0)


def best_alignment(directions, true_directions):
    """How well a list of directions matches a set of true directions.

    For every true direction, the direction that lines up with it best, averaged over them.
    The same measure is used for the random list, so the two numbers are directly
    comparable -- with 1024 arrows to choose from, one of them sits fairly close to anything
    by luck alone, and the baseline is what says how much of the number is luck.
    """
    directions = F.normalize(directions, dim=1)
    true_directions = F.normalize(true_directions, dim=1)
    return (true_directions @ directions.T).abs().max(dim=1).values


def recovery_figure(labels, values, baseline, out_path, dpi=150):
    """One bar per true direction: how well the SAE reaches it, against the luck level.

    The line across each bar is the best any of 1024 *random* directions manages for that
    same direction. A bar only says something where it clears its own line.
    """
    colours = [SHORTCUT_COLOUR if label == "colour" else
               TASK_COLOUR if label == "the answer" else NUISANCE_COLOUR
               for label in labels]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    position = np.arange(len(labels))
    bars = ax.bar(position, values, color=colours, width=0.62)
    for x, value, base in zip(position, values, baseline):
        ax.plot([x - 0.34, x + 0.34], [base, base], color=BASELINE_COLOUR, lw=1.6,
                zorder=3)
        ax.text(x, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(position)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("how well the closest of the\nSAE's directions lines up")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.99, 0.02, "the short line on each bar is what 1024 random\ndirections manage "
                        "-- the luck level", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=BASELINE_COLOUR)
    fig.text(0.5, 0.02,
             "the SAE was shown h, and nothing else. its directions still reach the things we "
             "know went into the picture --\nthe frame colour (red), the answer itself "
             "(blue), and the nuisances (grey) -- well past what random directions reach.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def directions_figure(names, vs_class, vs_colour, out_path, dpi=150):
    """Two panels: what the answer reads, and whether the shortcut is a direction of its own.

    Left: the cosine between the direction that carries the answer and each factor's own
    direction. A tall bar is a factor the classifier's answer is built out of.

    Right: the cosine between the shortcut's direction and every other factor's. A low bar
    there means the colour of the frame is its own thing inside the network -- which is
    exactly what has to be true for it to be removable later.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0), sharey=True)
    position = np.arange(len(names))

    for ax, values, title, highlight in (
        (axes[0], vs_class, "how much of the answer is this factor?", "colour"),
        (axes[1], vs_colour, "how much of the frame colour is this factor?", "colour"),
    ):
        colours = [SHORTCUT_COLOUR if name == highlight else TASK_COLOUR
                   if name == "the answer" else NUISANCE_COLOUR for name in names]
        signs = ["+" if value >= 0 else "\u2212" for value in values]
        ax.bar(position, values, color=colours, width=0.62)
        for x, value, sign in zip(position, values, signs):
            ax.text(x, value + (0.03 if value >= 0 else -0.05), f"{sign}{abs(value):.2f}",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
        ax.set_xticks(position)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.axhline(0.0, color="#999999", lw=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.25, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("cosine between the two directions")

    fig.text(0.5, 0.02,
             "left: the classifier's answer, broken into the factors we know went into the "
             "picture. the frame colour carries it.\nright: the same factors against the "
             "frame colour itself. the colour is a direction of its own, so it can be "
             "measured and pushed without dragging the answer along with it.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Take the SAE apart.")
    parser.add_argument("--data", default="data/dataset.pth")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    checkpoint = torch.load(Path(args.model_dir) / "classifier.pt")
    config = checkpoint["config"]
    model = Classifier(config["n_channels"], config["n_hidden"], config["n_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    sae_checkpoint = torch.load(Path(args.model_dir) / "sae.pt")
    sae_config = sae_checkpoint["config"]
    sae = SparseAutoencoder(sae_config["n_input"], sae_config["n_latent"], sae_config["k"])
    sae.load_state_dict(sae_checkpoint["state_dict"])
    sae.eval()

    dataset = torch.load(args.data)
    x_all = standardise(dataset["x"], dataset)
    with torch.no_grad():
        h_all = middle_layer_activations(model, x_all)

    index = torch.cat([dataset["splits"]["val"], dataset["splits"]["test"]])
    h, y, a = h_all[index], dataset["y"][index], dataset["a"][index]

    # ------------------------------------------------ the true directions, by intervention
    true_directions, sizes = intervention_directions(model, dataset, seed=args.seed)
    answer = class_direction(h_all[dataset["splits"]["train"]],
                             dataset["y"][dataset["splits"]["train"]])

    # The regression estimator, kept only as the comparison §3.3 discusses. ``w`` is the
    # classifier's decision direction, the difference of the two head rows.
    train_index = dataset["splits"]["train"]
    w = F.normalize(
        (model.body[-1].weight[1] - model.body[-1].weight[0]).detach().unsqueeze(0),
        dim=1).squeeze(0)
    fitted = factor_directions(h_all[train_index], dataset["a"][train_index])
    regression_colour_cosine = float((fitted[COLOUR] @ w).abs())
    noise_cosines = []
    for seed in range(3):
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(len(train_index), 1, generator=generator)
        augmented = torch.cat([dataset["a"][train_index], noise], dim=1)
        fitted_noise = factor_directions(h_all[train_index], augmented)
        noise_cosines.append(float((fitted_noise[-1] @ w).abs()))

    # ------------------------------------------------ what the SAE reaches, and the baseline
    named = torch.cat([true_directions, answer.unsqueeze(0)])
    reached = best_alignment(sae.directions(), named)
    generator = torch.Generator().manual_seed(args.seed)
    random_directions = torch.randn(sae_config["n_latent"], h_all.shape[1],
                                    generator=generator)
    baseline = best_alignment(random_directions, named)

    labels = list(FACTORS) + ["the answer"]
    values = [float(v) for v in reached]
    base_values = [float(v) for v in baseline]

    # ------------------------------------------------ the two cosine tables
    vs_answer = [float(answer @ true_directions[i]) for i in range(len(FACTORS))] + [1.0]
    vs_colour = [float(true_directions[COLOUR] @ true_directions[i])
                 for i in range(len(FACTORS))] + [float(true_directions[COLOUR] @ answer)]

    order = torch.argsort(reached, descending=True)
    with torch.no_grad():
        # Which single latent is closest to each true direction.
        cosine = (F.normalize(sae.directions(), dim=1) @ F.normalize(named, dim=1).T).abs()
    closest = {label: int(cosine[:, i].argmax()) for i, label in enumerate(labels)}

    results = {
        "factors": labels,
        "reached": dict(zip(labels, values)),
        "baseline": dict(zip(labels, base_values)),
        "reached_order": [labels[int(i)] for i in order],
        "closest_latent": closest,
        "cosine_with_answer": dict(zip(labels, vs_answer)),
        "cosine_with_colour": dict(zip(labels, vs_colour)),
        "change_size": dict(zip(FACTORS, sizes)),
        "regression_diagnostic": {
            "colour_cosine_with_decision_direction": regression_colour_cosine,
            "pure_noise_cosine_with_decision_direction": noise_cosines,
        },
    }

    model_dir = Path(args.model_dir)
    (model_dir / "interpret.json").write_text(json.dumps(results, indent=2))
    recovery_figure(labels, values, base_values, Path(args.figures) / "recovery.png")
    directions_figure(labels, vs_answer, vs_colour, Path(args.figures) / "directions.png")

    # ------------------------------------------------------------------------- what we found
    print(f"\neach factor's own direction in h, measured by changing it and nothing else "
          f"({N_PAIRS} pictures):")
    print(f"  {'factor':<9s} {'moves h by':>11s}   {'cos with the answer':>19s}"
          f"   {'cos with the frame colour':>26s}")
    for i, name in enumerate(FACTORS):
        print(f"  {name:<9s} {sizes[i]:11.2f}   {vs_answer[i]:+19.2f}"
              f"   {vs_colour[i]:+26.2f}")
    print(f"  {'answer':<9s} {'':>11s}   {1.0:+19.2f}   {vs_colour[-1]:+26.2f}")

    print(f"\nhow well the SAE's {sae_config['n_latent']} directions reach them:")
    for label, value, base in zip(labels, values, base_values):
        best_latent = closest[label]
        print(f"  {label:<10s} {value:.3f}   (random directions reach {base:.3f})   "
              f"closest latent {best_latent}")
    behind = float(np.mean(values) - np.mean(base_values))
    print(f"\n  on average the SAE reaches {np.mean(values):.3f} against the luck level "
          f"{np.mean(base_values):.3f}, a margin of {behind:.3f}")

    print(f"\nwrote {model_dir / 'interpret.json'}")
    print(f"wrote {Path(args.figures) / 'recovery.png'}")
    print(f"wrote {Path(args.figures) / 'directions.png'}")


if __name__ == "__main__":
    main()
