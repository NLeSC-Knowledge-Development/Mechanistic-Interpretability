"""Take the trained SAE apart: which direction is which, and what it does to the answer.

    python interpret.py --data data/dataset.pth --model-dir model --figures figures/sign

The SAE was trained with one job: rebuild ``h``. It never saw a label, and it was never told
which spot the task is about. So the only way to find out whether it learned *our* features
is to go back to the answers we already have.

Two questions, in order.

**Does the SAE have a direction for the missed feature?** The corner spot is real, and the
classifier leans on it -- accuracy drops by 0.225 on pictures where it is re-drawn
independently. But it also rides along with the middle spot: the two appear together 99.5% of
the time. So before asking which direction is which, this measures how much of a direction
each spot shares with the middle spot, read straight off the data by fitting ``h`` against
the true spot strengths. If two spots are the same direction in ``h``, then nothing
downstream can tell them apart, and no SAE can have a direction for one and not the other.
Every direction is then lined up against each spot's own direction, and the best one is
compared against an SAE of the same shape that was never trained -- a bar has to clear that
baseline before it is saying anything.

**What does that direction do?** Turning it up and down and watching the classifier's answer
is the direct test of whether it is a handle on the answer, rather than a coincidence. The
same push is made along the two spots' own directions, along a random direction, and along a
direction with nothing to do with the class, so the effect has something to be compared
against.

Writes:

* ``<model-dir>/interpret.json``   -- the numbers
* ``<figures>/recovery.png``       -- the true spots against the directions
* ``<figures>/missed_feature.png`` -- why the two spots are one thing in ``h``, and what
                                      pushing does
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

from train_classifier import TRAIN_COLOUR, VAL_COLOUR, Classifier
from train_sae import SparseAutoencoder, middle_layer_activations, spot_directions

# index 0 is always the middle spot; the missed feature is whatever ``meta`` says, and is the
# corner (index 1) for the ``sign`` task.
MIDDLE_SPOT = 0
# What a control push is drawn in: grey, so it does not compete with the two real features.
CONTROL_COLOUR = "#999999"
# How the given direction is pushed. ``gamma = 0`` is no push at all, ``-1`` is as though the
# direction were switched off, ``+1`` is half again as much.
GAMMAS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def auroc(scores, labels):
    """Area under the ROC curve: 0.5 is useless, 1.0 is perfect.

    A direction is off for most pictures, so equal scores sit everywhere and the ties have
    to be shared out evenly -- otherwise the ranks, and the score, come out wrong.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())

    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start
        while end + 1 < len(scores) and ordered[end + 1] == ordered[start]:
            end += 1
        # Ranks are 1-based, and every tied score gets the average rank it covers.
        ranks[order[start:end + 1]] = (start + end + 2) / 2.0
        start = end + 1

    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def alignment(directions, spot_dirs):
    """How nearly each direction lines up with each spot's direction. Shape
    ``(n_directions, n_spots)``.

    Sign is thrown away -- a direction pointing backwards still says the same thing -- so
    these are sizes of cosines, between 0 and 1.
    """
    return (F.normalize(directions, dim=1) @ spot_dirs.T).abs()


def tracking(z, a):
    """How closely each direction's activation follows each true spot's strength.

    Plain correlation, both sides centred and scaled to unit length first, so the dot product
    is the correlation itself. Shape ``(n_directions, n_spots)``.

    ``alignment`` asks which direction *is* a spot; this asks which direction *answers to* a
    spot. They are not the same question: a direction can point almost exactly along a feature
    and still hardly ever switch on, in which case pushing it does nothing.
    """
    z = z - z.mean(dim=0, keepdim=True)
    a = a - a.mean(dim=0, keepdim=True)
    z = z / z.norm(dim=0, keepdim=True).clamp_min(1e-12)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return z.T @ a


@torch.no_grad()
def push_curve(head, h, direction, strength, gammas=GAMMAS):
    """Change in the probability of class 1 as one fixed direction is pushed.

    ``strength`` is the size of the push on each picture. Turning a direction's dial up by
    ``gamma`` changes ``h`` by ``gamma * strength * direction``, which is exactly the change
    this adds -- so the same function serves the real direction and every control.
    """
    base = torch.softmax(head(h), dim=1)[:, 1]
    curve = []
    for gamma in gammas:
        pushed = h + gamma * strength.unsqueeze(1) * direction
        probability = torch.softmax(head(pushed), dim=1)[:, 1]
        curve.append(float((probability - base).mean()))
    return curve


@torch.no_grad()
def rebuild_push(sae, h, z, direction, gamma):
    """The new ``h`` after one dial is turned, with the leftover the SAE cannot rebuild put back.

    At ``gamma = 0`` this must give back the old ``h`` exactly. If it does not, the push is
    being mixed up with the SAE's own error and the measurement means nothing.
    """
    rebuild = sae.decoder(z)
    leftover = h - rebuild
    z_new = z.clone()
    z_new[:, direction] *= 1.0 + gamma
    return sae.decoder(z_new) + leftover


def recovery_figure(best, untrained_best, missed_spot, out_path, dpi=150):
    """One bar per true spot: how well the best direction lines up with that spot.

    Blue is the middle spot, the rule we wrote down. Red is the corner spot, the feature we
    missed. The rest are the background spots, which nothing is supposed to track. The grey
    line is the same measurement on an SAE of the same shape that was never trained, so a bar
    has to clear that line before it is saying anything.
    """
    values = best.numpy()
    baseline = untrained_best.numpy()
    spots = np.arange(len(values))
    colours = ["#cccccc"] * len(values)
    colours[MIDDLE_SPOT] = TRAIN_COLOUR
    colours[missed_spot] = VAL_COLOUR

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(spots, values, color=colours, width=0.72)
    ax.plot(spots, baseline, marker="o", markersize=3.5, linewidth=1.3,
            color=CONTROL_COLOUR, label="untrained SAE, same architecture")

    ax.set_xticks(spots)
    ax.set_xticklabels([str(s) for s in spots], fontsize=7)
    ax.set_xlabel("true spot")
    ax.set_ylabel("best |cosine| with spot direction")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")

    fig.text(
        0.5, 0.02,
        "One bar per true spot: the largest |cosine| between any SAE direction and the "
        "spot's fitted direction in h.\nBlue: spot 0, the label rule. Red: spot 1, "
        "correlated with the label but absent from the label rule.\nGrey line: untrained "
        "SAE of the same architecture.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def missed_feature_figure(separability, separability_baseline, missed_spot, curves, gammas,
                          out_path, dpi=150):
    """Left: how much each spot shares a direction with the middle spot. Right: the push.

    The left panel is the reason the rest of the picture looks the way it does. If the corner
    spot points the same way as the middle spot inside ``h``, then nothing downstream -- not
    the SAE, not a push -- can tell the two apart, however good it is.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))

    spots = np.arange(len(separability))
    colours = ["#cccccc"] * len(spots)
    colours[missed_spot] = VAL_COLOUR
    axes[0].bar(spots, separability.numpy(), color=colours, width=0.72)
    axes[0].plot(spots, separability_baseline.numpy(), marker="o", markersize=3.5,
                 linewidth=1.3, color=CONTROL_COLOUR,
                 label="first-layer shortcut")
    axes[0].set_xticks(spots)
    axes[0].set_xticklabels([str(s) for s in spots], fontsize=7)
    axes[0].set_xlabel("true spot")
    axes[0].set_ylabel("|cosine| with spot 0 direction")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8.5, loc="upper right")

    for label, values, colour, style in curves:
        axes[1].plot(gammas, values, marker="o", markersize=4, linewidth=1.7, color=colour,
                     linestyle=style, label=label)
    axes[1].axhline(0.0, color="#dddddd", linewidth=1.0, zorder=0)
    axes[1].set_xlabel("steering coefficient gamma")
    axes[1].set_ylabel("change in probability of class 1")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)

    fig.text(
        0.5, 0.01,
        "Left: |cosine| between each fitted spot direction and spot 0's direction in h. "
        "Spot 0 is 1 against itself by definition.\nSpot 1 is nearly parallel to spot 0, so "
        "no downstream method can separate them. Right: effect of steering the SAE's "
        "direction\nand the two fitted spot directions on the classifier output.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Take the trained SAE apart and test what its directions do.")
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

    sae_checkpoint = torch.load(Path(args.model_dir) / "sae.pt")
    sae_config = sae_checkpoint["config"]
    sae = SparseAutoencoder(sae_config["n_input"], sae_config["n_latent"], sae_config["k"])
    sae.load_state_dict(sae_checkpoint["state_dict"])
    sae.eval()

    dataset = torch.load(args.data)
    missed_spot = dataset["meta"]["missed_spot_index"]
    x_all = (dataset["x"] - dataset["mean"]) / dataset["std"]
    with torch.no_grad():
        h_all = middle_layer_activations(model, x_all)
        z_all = sae.encode(h_all)

    train = dataset["splits"]["train"]
    # Score on the validation and test pictures: a separate set from the one the SAE was
    # trained on, so the numbers are not just it quoting itself back.
    scored = torch.cat([dataset["splits"]["val"], dataset["splits"]["test"]])
    h, z = h_all[scored], z_all[scored]
    a, y = dataset["a"][scored], dataset["y"][scored]

    # Each true spot's own direction inside h, read off the training pictures.
    true_directions = spot_directions(h_all[train], dataset["a"][train])
    directions = sae.directions()

    cosine = alignment(directions, true_directions)
    best = cosine.max(dim=0).values
    best_direction = int(cosine[:, missed_spot].argmax())
    middle_direction = int(cosine[:, MIDDLE_SPOT].argmax())

    # The reason the whole question is delicate: how much of a direction each other spot shares
    # with the middle spot. If a spot points the same way as the middle spot inside h, then no
    # direction can belong to one of them and not the other, however good the SAE is.
    separability = (true_directions @ true_directions[MIDDLE_SPOT]).abs()
    # The first-layer shortcut, kept only for comparison: push each spot's picture through the
    # first layer and call that its direction. This ignores the GELU between the input and h.
    with torch.no_grad():
        shortcut = F.normalize(dataset["W_true"] @ model.body[0].weight.T, dim=1)
        separability_baseline = (shortcut @ shortcut[MIDDLE_SPOT]).abs()

    # The baseline is an SAE of the same shape that was never trained: same architecture, same
    # sparsity, same number of directions, only random ones.
    torch.manual_seed(args.seed)
    untrained = SparseAutoencoder(sae_config["n_input"], sae_config["n_latent"],
                                  sae_config["k"])
    untrained.eval()
    with torch.no_grad():
        untrained_best = alignment(untrained.directions(), true_directions).max(dim=0).values

    aurocs = np.array([auroc(z[:, i].numpy(), y.numpy()) for i in range(z.shape[1])])
    best_auroc_direction = int(aurocs.argmax())
    # The direction whose activation actually answers to the missed feature. This is the one to
    # push: a direction that lines up but never switches on is not a handle on anything.
    tracking_matrix = tracking(z, a)
    tracking_direction = int(tracking_matrix[:, missed_spot].abs().argmax())
    # A direction with no relation to the class, for the control push: among the directions
    # that actually fire, the one whose score is closest to chance. A direction that is zero
    # on every picture is not a fair control -- pushing it is a no-op by construction rather
    # than evidence that the direction does not matter.
    firing = (z > 0).float().mean(dim=0)
    silent = firing < 0.05
    unrelated_direction = int(np.argmin(np.where(silent, np.inf, np.abs(aurocs - 0.5))))

    # What pushing does, on the test pictures. Every push is given the same size -- the size of
    # the tracking direction's own activation -- so the curves are comparable.
    generator = torch.Generator().manual_seed(args.seed)
    test = dataset["splits"]["test"]
    h_test, z_test = h_all[test], z_all[test]
    head = model.body[-1]
    size = z_test[:, tracking_direction]
    random_unit = F.normalize(torch.randn(1, h_test.shape[1], generator=generator), dim=1)

    curves = {
        "the SAE direction": push_curve(head, h_test,
                                        directions[tracking_direction].unsqueeze(0), size),
        "the corner's own direction":
            push_curve(head, h_test, true_directions[missed_spot].unsqueeze(0), size),
        "the middle's own direction":
            push_curve(head, h_test, true_directions[MIDDLE_SPOT].unsqueeze(0), size),
        "a random push": push_curve(head, h_test, random_unit, size),
        "a different direction": push_curve(head, h_test,
                                            directions[unrelated_direction].unsqueeze(0),
                                            size),
    }

    # The check before anything is believed: no push must mean no change.
    zero_shift = float((rebuild_push(sae, h_test, z_test, tracking_direction, 0.0)
                        - h_test).abs().max())

    results = {
        "missed_spot": int(missed_spot),
        "best_direction_for_middle": middle_direction,
        "best_direction_for_middle_cosine": float(cosine[middle_direction, MIDDLE_SPOT]),
        "best_direction_for_corner": best_direction,
        "best_direction_for_corner_cosine": float(cosine[best_direction, missed_spot]),
        "separate_directions": bool(middle_direction != best_direction),
        "tracking_direction": tracking_direction,
        "tracking_corner": float(tracking_matrix[tracking_direction, missed_spot]),
        "tracking_middle": float(tracking_matrix[tracking_direction, MIDDLE_SPOT]),
        "tracking_direction_cosine_with_corner":
            float(cosine[tracking_direction, missed_spot]),
        "best_auroc_direction": best_auroc_direction,
        "best_auroc": float(aurocs[best_auroc_direction]),
        "unrelated_direction": unrelated_direction,
        "unrelated_auroc": float(aurocs[unrelated_direction]),
        "unrelated_firing_fraction": float(firing[unrelated_direction]),
        "best_per_spot": [float(v) for v in best],
        "untrained_best_per_spot": [float(v) for v in untrained_best],
        "direction_shared_with_middle": [float(v) for v in separability],
        "direction_shared_with_middle_shortcut": [float(v) for v in separability_baseline],
        "gammas": list(GAMMAS),
        "curves": curves,
        "zero_push_shift": zero_shift,
    }

    model_dir = Path(args.model_dir)
    (model_dir / "interpret.json").write_text(json.dumps(results, indent=2))

    recovery_figure(best, untrained_best, missed_spot, Path(args.figures) / "recovery.png")
    missed_feature_figure(separability, separability_baseline, missed_spot, [
        ("the SAE's direction", curves["the SAE direction"], VAL_COLOUR, "-"),
        ("the corner spot's own direction", curves["the corner's own direction"],
         CONTROL_COLOUR, "--"),
        ("the middle spot's own direction", curves["the middle's own direction"],
         TRAIN_COLOUR, "--"),
        ("a random push, same size", curves["a random push"], CONTROL_COLOUR, ":"),
        ("a different direction", curves["a different direction"], CONTROL_COLOUR, "-."),
    ], GAMMAS, Path(args.figures) / "missed_feature.png")

    # --------------------------------------------------------------------- what we found
    print(f"\nscored on {len(scored)} pictures the SAE was not trained on")
    print(f"  the middle spot is spot {MIDDLE_SPOT}; the missed feature is spot {missed_spot}")

    print("\nis the missed feature a separate thing inside h?")
    print(f"  {'how much each spot shares a direction':<44s} {'from h':>7s} {'shortcut':>9s}")
    for label, spot in (("the corner spot (the one we missed)", missed_spot),
                        ("the best of the background spots",
                         max([j for j in range(len(separability))
                              if j not in (MIDDLE_SPOT, missed_spot)],
                             key=lambda j: float(separability[j])))):
        print(f"  {label:<44s} {float(separability[spot]):7.3f} "
              f"{float(separability_baseline[spot]):9.3f}")

    print("\nthe best direction for each feature")
    print(f"  {'feature':<34s} {'direction':>9s} {'lines up':>9s} {'untrained':>10s}")
    for label, spot in (("the middle spot (the rule)", MIDDLE_SPOT),
                        ("the corner spot (the missed one)", missed_spot)):
        print(f"  {label:<34s} {int(cosine[:, spot].argmax()):9d} {float(best[spot]):9.3f} "
              f"{float(untrained_best[spot]):10.3f}")
    others = [j for j in range(len(best)) if j not in (MIDDLE_SPOT, missed_spot)]
    best_other = max(others, key=lambda j: float(best[j]))
    print(f"  {'the best background spot (spot ' + str(best_other) + ')':<34s} "
          f"{int(cosine[:, best_other].argmax()):9d} {float(best[best_other]):9.3f} "
          f"{float(untrained_best[best_other]):10.3f}")
    if middle_direction == best_direction:
        print(f"\n  the same direction ({best_direction}) comes out best for both spots.")
    else:
        print(f"\n  direction {middle_direction} for the middle, {best_direction} for the "
              f"corner.")

    print("\nthe direction the SAE actually answers with")
    print(f"  direction {tracking_direction}: follows the corner spot at "
          f"{float(tracking_matrix[tracking_direction, missed_spot]):+.3f}, "
          f"the middle spot at "
          f"{float(tracking_matrix[tracking_direction, MIDDLE_SPOT]):+.3f}")
    print("  it lines up with both, because the two spots are one direction in h")

    print("\nthe effect of the missed feature (change in probability of class 1)")
    print(f"  {'gamma':>6s}" + "".join(f"{name:>30s}" for name in curves))
    for row, gamma in enumerate(GAMMAS):
        print(f"  {gamma:6.1f}" + "".join(
            f"{curves[name][row]:20.4f}" for name in curves))
    print(f"\n  at gamma = 0 the new h is {zero_shift:.1e} away from the old one")

    print(f"\nwrote {model_dir / 'interpret.json'}")
    print(f"wrote {Path(args.figures) / 'recovery.png'}")
    print(f"wrote {Path(args.figures) / 'missed_feature.png'}")


if __name__ == "__main__":
    main()
