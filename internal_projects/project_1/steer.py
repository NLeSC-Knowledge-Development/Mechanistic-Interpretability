"""Use the trained SAE the way one actually uses one: find a direction, then turn it.

    ../.venv/bin/python steer.py --data data/dataset.pth --model-dir model \
                                 --figures figures/sign

``interpret.py`` answers the question *what did the SAE find*. This script is the other
half of the same pair: it is a worked example of *using* it, and it follows the recipe that
any SAE is used by, whatever the model underneath:

1. **Score every direction against something you care about.** The SAE never saw the
   labels, so its 256 directions come with no names. The cheapest way to name one is to ask
   how well its strength predicts the class -- ``auroc`` below -- and keep the best few.
2. **Look at what the winner follows.** A direction that predicts the class is worth
   knowing about, so its activation is lined up against the true spot strengths, the same
   way ``interpret.py`` does it.
3. **Turn it, and turn it the other way.** Scale that one direction's strength, rebuild
   ``h``, and watch the classifier's answer. Doing it both ways is what separates a dial
   from a coincidence.
4. **Push something else in exactly the same way.** A random direction, of the same size,
   and a direction with nothing to do with the class: if the answer moves just as much for
   those, the dial is not doing anything.
5. **Check whether one direction is the whole story.** A group of the most class-related
   directions, and a plain linear probe with no SAE at all, say how much of the effect is
   really spread out and how much is one clean axis.

The push itself is plain arithmetic, and it is worth writing down. The SAE says ``h`` is
``decoder(z)`` plus a small leftover. Scaling direction ``c`` by ``(1 + gamma)`` means
scaling its strength only, so

    h_new = decoder(z with z[c] scaled by (1 + gamma)) + leftover.

At ``gamma = 0`` that is the old ``h`` again, to rounding -- the check at the top of every
result below. The leftover is put back so the push is not mixed up with the SAE's own
rebuilding error.

Writes:

* ``<model-dir>/steering.json`` -- the numbers
* ``<figures>/steering.png``    -- the three panels: which directions matter, the push and
                                  its controls, and the group and the probe
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
import torch.nn as nn
import torch.nn.functional as F

from interpret import CONTROL_COLOUR, GAMMAS, auroc, rebuild_push
from train_classifier import MIDDLE_COLOUR, TRAIN_COLOUR, VAL_COLOUR, Classifier
from train_sae import SparseAutoencoder, middle_layer_activations

# How many of the best-scoring directions are pushed together as the group. It is the same
# number as ``k``, the directions allowed on at once, so "the group" is as big as a single
# example's own list of directions -- not a bigger hand than the SAE is dealt.
GROUP = 8
# A control has to be a direction that actually switches on: a direction that is zero on
# every picture is not a fair control, because pushing it is a no-op by construction rather
# than by showing that the direction does not matter. This is the smallest fraction of
# pictures a control direction has to fire on.
CONTROL_MIN_FIRING = 0.05


def fit_probe(h, y, steps=400, lr=0.5):
    """A plain linear probe: the best straight direction through ``h`` that separates the
    two classes, with no SAE anywhere.

    Trained by gradient descent on cross-entropy like everything else in this folder, and
    seeded, so the direction is the same every run. Returns the unit direction, so it can
    be pushed in exactly the same way as the SAE's own directions.
    """
    torch.manual_seed(0)
    layer = nn.Linear(h.shape[1], 2)
    optimiser = torch.optim.LBFGS(layer.parameters(), lr=lr, max_iter=steps,
                                 line_search_fn="strong_wolfe")
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(layer(h), y)
        loss.backward()
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        direction = F.normalize((layer.weight[1] - layer.weight[0]).unsqueeze(0), dim=1)
    return direction


@torch.no_grad()
def direction_curve(head, h, direction, strength, gammas=GAMMAS):
    """Change in the probability of class 1 as a fixed direction is pushed.

    ``strength`` is the size of the push on each picture, so a curve is ``gamma`` times
    one fixed step along ``direction``. Every curve in this script is measured this way,
    which is what makes them comparable with each other.
    """
    base = torch.softmax(head(h), dim=1)[:, 1]
    curve = []
    for gamma in gammas:
        pushed = h + gamma * strength.unsqueeze(1) * direction
        probability = torch.softmax(head(pushed), dim=1)[:, 1]
        curve.append(float((probability - base).mean()))
    return curve


@torch.no_grad()
def scale_curve(sae, head, h, z, latents, gammas=GAMMAS):
    """Change in the probability of class 1 as the strengths of ``latents`` are scaled.

    This is the SAE's own dial: instead of moving ``h`` along a fixed direction, the
    written strengths are multiplied and ``h`` is rebuilt from them. Scaling a set of
    latents all at once is the same arithmetic with several strengths moved together.
    """
    base = torch.softmax(head(h), dim=1)[:, 1]
    leftover = h - sae.decoder(z)
    curve = []
    for gamma in gammas:
        z_new = z.clone()
        z_new[:, latents] *= 1.0 + gamma
        pushed = sae.decoder(z_new) + leftover
        probability = torch.softmax(head(pushed), dim=1)[:, 1]
        curve.append(float((probability - base).mean()))
    return curve


def steering_figure(aurocs, best, group, curves, probe_auroc, gammas, out_path, dpi=150):
    """Three panels, in the order the recipe uses them.

    Left: every direction's score. One bar per direction, sorted, with the best one marked
    and the line at 0.5 where a direction says nothing at all -- the shortlist the recipe
    starts from.

    Middle: the push. The best direction turned up and down, against a random direction of
    exactly the same size and a direction with nothing to do with the class.

    Right: is one direction the whole story? The eight best directions pushed together,
    against the single best, against a probe that never saw the SAE.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3))

    # --- every direction, sorted ---------------------------------------------------
    ranked = np.sort(aurocs)[::-1]
    order = np.argsort(aurocs)[::-1]
    colours = ["#cccccc"] * len(ranked)
    for position, index in enumerate(order[:GROUP]):
        colours[position] = TRAIN_COLOUR if index != best else VAL_COLOUR
    axes[0].bar(np.arange(len(ranked)), ranked, color=colours, width=0.85)
    axes[0].axhline(0.5, color=CONTROL_COLOUR, linewidth=1.2, linestyle="--")
    axes[0].text(len(ranked) * 0.99, 0.505, "chance (0.5)", ha="right", va="bottom",
                 fontsize=8.5, color=CONTROL_COLOUR)
    axes[0].axvline(GROUP - 0.5, color=CONTROL_COLOUR, linewidth=1.0, linestyle=":")
    axes[0].annotate(f"direction {best}", xy=(0, ranked[0]), xytext=(14, ranked[0] - 0.06),
                     fontsize=9, color=VAL_COLOUR,
                     arrowprops=dict(arrowstyle="-|>", color=VAL_COLOUR, linewidth=1.2))
    axes[0].set_xlabel("SAE directions, sorted by score")
    axes[0].set_ylabel("class-prediction AUROC")
    axes[0].set_ylim(0.4, 0.88)
    axes[0].grid(axis="y", alpha=0.3)

    # --- the push, and the controls -------------------------------------------------
    for label, values, colour, style in curves["single"]:
        axes[1].plot(gammas, values, marker="o", markersize=4, linewidth=1.8, color=colour,
                     linestyle=style, label=label)
    axes[1].axhline(0.0, color="#dddddd", linewidth=1.0, zorder=0)
    axes[1].set_xlabel("steering coefficient gamma")
    axes[1].set_ylabel("change in probability of class 1")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)

    # --- group, single, and the probe -----------------------------------------------
    for label, values, colour, style in curves["compare"]:
        axes[2].plot(gammas, values, marker="o", markersize=4, linewidth=1.8, color=colour,
                     linestyle=style, label=label)
    axes[2].axhline(0.0, color="#dddddd", linewidth=1.0, zorder=0)
    axes[2].set_xlabel("steering coefficient gamma")
    axes[2].set_ylabel("change in probability of class 1")
    axes[2].legend(fontsize=8.5)
    axes[2].grid(alpha=0.3)
    axes[2].set_title(f"linear probe AUROC {probe_auroc:.3f}", fontsize=9.5,
                      color="#444444")

    fig.text(
        0.5, 0.01,
        "Left: AUROC of each SAE direction as a predictor of the class, sorted; the SAE "
        "never used labels. Middle: effect of scaling the best\ndirection, with random and "
        "unrelated controls of equal magnitude. Right: the eight best directions scaled "
        "together, compared with the single\nbest direction and with a linear probe "
        "trained on labels.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Use the trained SAE: find a direction, then turn it.")
    parser.add_argument("--data", default="data/dataset.pth")
    parser.add_argument("--model-dir", default="model",
                        help="where the trained classifier and SAE are")
    parser.add_argument("--figures", default="figures/sign", help="where the figures go")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    # The classifier, as trained, and the SAE that was trained on its middle layer.
    checkpoint = torch.load(Path(args.model_dir) / "classifier.pt")
    config = checkpoint["config"]
    model = Classifier(config["n_pixels"], config["n_hidden"], config["n_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    head = model.body[-1]

    sae_checkpoint = torch.load(Path(args.model_dir) / "sae.pt")
    sae_config = sae_checkpoint["config"]
    sae = SparseAutoencoder(sae_config["n_input"], sae_config["n_latent"], sae_config["k"])
    sae.load_state_dict(sae_checkpoint["state_dict"])
    sae.eval()

    dataset = torch.load(args.data)
    x_all = (dataset["x"] - dataset["mean"]) / dataset["std"]
    with torch.no_grad():
        h_all = middle_layer_activations(model, x_all)
        z_all = sae.encode(h_all)

    # Score on the validation and test pictures: a separate set from the one the SAE was
    # trained on, so the numbers are not just it quoting itself back.
    scored = torch.cat([dataset["splits"]["val"], dataset["splits"]["test"]])
    h, z = h_all[scored], z_all[scored]
    a, y = dataset["a"][scored], dataset["y"][scored]

    # --------------------------------------------------- 1. score every direction
    aurocs = np.array([auroc(z[:, i].numpy(), y.numpy()) for i in range(z.shape[1])])
    best = int(aurocs.argmax())
    group = [int(i) for i in np.argsort(aurocs)[::-1][:GROUP]]
    # How often each direction switches on at all. A direction that is exactly zero on
    # every picture has no dial to turn, so the control has to be picked from the ones that
    # do fire -- see ``CONTROL_MIN_FIRING``.
    firing = (z > 0).float().mean(dim=0).numpy()
    # A direction with no relation to the class, for the control: among the ones that
    # actually fire, the one whose score comes closest to saying nothing at all.
    silent = firing < CONTROL_MIN_FIRING
    unrelated = int(np.argmin(np.where(silent, np.inf, np.abs(aurocs - 0.5))))
    useful = int((aurocs > 0.7).sum())

    # --------------------------------------------------- 2. what the winner follows
    z_scored = z - z.mean(dim=0, keepdim=True)
    a_scored = a - a.mean(dim=0, keepdim=True)
    z_scored = z_scored / z_scored.norm(dim=0, keepdim=True).clamp_min(1e-12)
    a_scored = a_scored / a_scored.norm(dim=0, keepdim=True).clamp_min(1e-12)
    tracking = (z_scored[:, best].unsqueeze(0) @ a_scored).squeeze(0)

    # --------------------------------------------------- 3-5. the push and the controls
    generator = torch.Generator().manual_seed(args.seed)
    random_unit = F.normalize(torch.randn(1, h.shape[1], generator=generator), dim=1)
    test = dataset["splits"]["test"]
    h_test, z_test = h_all[test], z_all[test]
    size = z_test[:, best]  # the push is one unit of the direction itself, per picture

    # The probe: trained on the train split only, so the SAE and the probe see the same
    # amount of data and neither is scored on what it learned from.
    probe = fit_probe(h_all[dataset["splits"]["train"]], dataset["y"][dataset["splits"]["train"]])
    with torch.no_grad():
        probe_scores = (h_test @ probe.T).squeeze(1).numpy()
    probe_auroc = auroc(probe_scores, dataset["y"][test].numpy())

    # Every curve gets the same per-picture size, so the only thing that differs between
    # them is the direction being pushed. The direction that is not about the class is
    # pushed along its own axis rather than by its own strength, so that a control is not
    # quietly given a smaller push than the real thing.
    curves = {
        "single": [
            (f"SAE direction {best}", scale_curve(sae, head, h_test, z_test, [best]),
             VAL_COLOUR, "-"),
            (f"unrelated direction {unrelated} (fires {firing[unrelated]:.0%})",
             direction_curve(head, h_test, sae.directions()[unrelated].unsqueeze(0), size),
             CONTROL_COLOUR, "-."),
            ("random direction, same size",
             direction_curve(head, h_test, random_unit, size), CONTROL_COLOUR, ":"),
        ],
        "compare": [
            (f"{GROUP} best directions together",
             scale_curve(sae, head, h_test, z_test, group), VAL_COLOUR, "-"),
            (f"single best direction ({best})",
             scale_curve(sae, head, h_test, z_test, [best]), TRAIN_COLOUR, "--"),
            ("linear probe (uses labels)",
             direction_curve(head, h_test, probe, size), MIDDLE_COLOUR, "-."),
        ],
    }

    # The check before anything is believed: no push must mean no change.
    zero_shift = float((rebuild_push(sae, h_test, z_test, best, 0.0) - h_test).abs().max())

    # The honest note at the end: how much of the probe's better score is just the probe
    # being readable on every picture, where a direction is exactly zero most of the time.
    fires = z_test[:, best] > 0
    best_where_fires = auroc(z_test[fires, best].numpy(), dataset["y"][test][fires].numpy())
    probe_where_fires = auroc(probe_scores[fires.numpy()],
                              dataset["y"][test][fires].numpy())

    results = {
        "best_direction": best,
        "best_auroc": float(aurocs[best]),
        "best_firing_fraction": float(firing[best]),
        "group": group,
        "group_auroc": [float(aurocs[i]) for i in group],
        "useful_directions": useful,
        "unrelated_direction": unrelated,
        "unrelated_auroc": float(aurocs[unrelated]),
        "unrelated_firing_fraction": float(firing[unrelated]),
        "probe_auroc_where_best_fires": float(probe_where_fires),
        "best_auroc_where_best_fires": float(best_where_fires),
        "tracking": [float(v) for v in tracking],
        "probe_auroc": float(probe_auroc),
        "gammas": list(GAMMAS),
        "curves": {name: [values for _, values, _, _ in options]
                   for name, options in curves.items()},
        "zero_push_shift": zero_shift,
    }

    model_dir = Path(args.model_dir)
    (model_dir / "steering.json").write_text(json.dumps(results, indent=2))
    steering_figure(aurocs, best, group, curves, probe_auroc, GAMMAS,
                    Path(args.figures) / "steering.png")

    # --------------------------------------------------------------------- what we found
    print(f"\nscored {z.shape[1]} directions on {len(scored)} pictures the SAE was "
          f"not trained on")

    print("\n1. which directions predict the class")
    print(f"  the best is direction {best}, at AUROC {aurocs[best]:.3f}")
    print(f"  {useful} directions score above 0.700")
    print(f"  the 8 best: " + ", ".join(f"{i} ({aurocs[i]:.2f})" for i in group[:GROUP]))

    print(f"\n2. what direction {best} follows")
    for spot, label in ((0, "the middle spot (the rule)"),
                        (dataset["meta"]["missed_spot_index"], "the corner spot (the missed one)")):
        print(f"  {label:<34s} correlation {float(tracking[spot]):+.3f}")
    others = [j for j in range(len(tracking))
              if j not in (0, dataset["meta"]["missed_spot_index"])]
    background = max(others, key=lambda j: abs(float(tracking[j])))
    print(f"  {'the largest background correlation':<34s} correlation "
          f"{float(tracking[background]):+.3f}")

    print("\n3-5. change in the probability of class 1")
    for title, options in curves.items():
        print(f"  {title}")
        print(f"  {'gamma':>7s}" + "".join(f"{label[:26]:>29s}" for label, *_ in options))
        for row, gamma in enumerate(GAMMAS):
            print(f"  {gamma:7.1f}" + "".join(
                f"{values[row]:29.4f}" for _, values, *_ in options))
    print(f"\n  the best direction fires on {firing[best]:.2f} of the pictures")
    print(f"  the probe scores AUROC {probe_auroc:.3f} and the best direction "
          f"{aurocs[best]:.3f}; where the direction fires, {probe_where_fires:.3f} "
          f"against {best_where_fires:.3f}")
    print(f"  at gamma = 0 the new h is {zero_shift:.1e} away from the old one")

    print(f"\nwrote {model_dir / 'steering.json'}")
    print(f"wrote {Path(args.figures) / 'steering.png'}")


if __name__ == "__main__":
    main()
