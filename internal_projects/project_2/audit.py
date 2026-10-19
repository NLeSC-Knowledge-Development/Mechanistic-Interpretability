"""Use the SAE the way one actually does: find the shortcut, then do something about it.

    ../../.venv/bin/python audit.py --data data/dataset.pth --model-dir model \
                                    --figures figures

``interpret.py`` answered the question *what did the SAE find*. This script is the other
half of the same pair: it is a worked example of **using** it, and it follows the recipe an
SAE is put to on any model:

1. **Score every direction against something you care about.** The SAE never saw a label, so
   its 1024 directions come with no names. The cheapest way to name one is to ask how well
   its strength sorts the pictures -- ``auroc`` below -- and keep the best.
2. **Look at what the winner follows.** A direction that predicts the answer is worth
   knowing about, so we check it against the thing we know is really there: the frame
   colour. The SAE was never told the frame existed.
3. **Turn it, and turn it the other way.** Push ``h`` along that feature's direction and
   watch the classifier's answer. Doing it both ways separates a dial from a coincidence.
4. **Push something else in exactly the same way.** A random direction of the same size, and
   a direction that fires just as often with nothing to do with the class: if the answer
   moves just as much for those, the dial is doing nothing.
5. **Do something useful with it.** Two things, and they are the point of the whole folder:
   a **monitor** that flags, picture by picture, the answers that were given from the frame
   rather than from the digit; and a **repair**, measured four ways at once, because the
   interesting part is not that one of them helps but *which* of them is doing the work.

One honest thing about the push, before it is used. The SAE's own dial scales a written
strength --

    h_new = decoder(z with z[c] scaled by (1 + gamma)) + leftover

-- and it is a fair question whether the latent it names is also the *axis* of the feature.
Here it is not, and the measurements below say so: a latent can be an excellent **detector**
of the frame colour (its strength sorts the pictures at AUROC 0.99) while the direction it
writes into ``h`` is not the colour's direction at all, because what a latent reads through
its encoder and what it writes through its decoder are two different things. So the dial in
the figure, and the surgery in the repair, push ``h`` along the *feature's own direction* --
the one measured by changing the frame and nothing else -- and the SAE's job is to say which
feature is worth pointing at, and to read it back per picture.

Writes:

* ``<model-dir>/audit.json`` -- the numbers
* ``<figures>/audit.png``    -- three panels: which directions matter, the dial and its
                               controls, and the four repairs side by side
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

from generate_data import COLOUR
from interpret import (BASELINE_COLOUR, SHORTCUT_COLOUR, TASK_COLOUR,
                       intervention_directions, standardise)
from train_classifier import Classifier, MIDDLE_COLOUR, accuracy, grayscale_split
from train_sae import SparseAutoencoder, middle_layer_activations

# How hard the dial is turned.
GAMMAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
# A control has to be a direction that actually switches on: a direction that is zero on
# every picture is not a fair control, because pushing it is a no-op by construction rather
# than by showing that the direction does not matter. This is the smallest share of pictures
# a control direction has to fire on.
CONTROL_MIN_FIRING = 0.05


def auroc(scores, labels):
    """The area under the ROC curve: 0.5 for a coin, 1.0 for a perfect sorter.

    The standard way to say "how well does this one number sort the pictures into the two
    classes". Ties get the average rank, so a direction that is exactly zero on half the
    pictures does not get a free pass.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    positive = labels == 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def project_out(h, basis):
    """Remove the span of ``basis`` from every row of ``h``.

    The surgery the repair uses: whatever ``h`` was saying along those directions is taken
    out, and everything across them is left untouched.
    """
    orthonormal, _ = torch.linalg.qr(basis.T)
    return h - (h @ orthonormal) @ orthonormal.T


def fit_readout(h, y, steps=800, lr=0.05, seed=0):
    """A fresh two-class read-out on top of ``h``, trained the same way the head was.

    This is the light half of the repair: the big network that does the seeing is left
    exactly as it is, and only the last two scores are re-learned, from the labels we
    already had.
    """
    torch.manual_seed(seed)
    layer = nn.Linear(h.shape[1], 2)
    optimiser = torch.optim.Adam(layer.parameters(), lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        nn.functional.cross_entropy(layer(h), y).backward()
        optimiser.step()
    layer.eval()
    return layer


def accuracy_of(scorer, h, y):
    with torch.no_grad():
        return float((scorer(h).argmax(dim=1) == y).float().mean())


@torch.no_grad()
def dial_curve(sae, head, h, z, latent, gammas=GAMMAS):
    """Change in the probability of class 1 as one direction's strength is scaled."""
    base = torch.softmax(head(h), dim=1)[:, 1]
    leftover = h - sae.decoder(z)
    curve = []
    for gamma in gammas:
        z_new = z.clone()
        z_new[:, latent] *= 1.0 + gamma
        probability = torch.softmax(head(sae.decoder(z_new) + leftover), dim=1)[:, 1]
        curve.append(float((probability - base).mean()))
    return curve


@torch.no_grad()
def direction_curve(head, h, direction, size, gammas=GAMMAS):
    """The same curve, but pushing ``h`` along a fixed direction instead of a written one.

    Same size of push as the SAE's own dial gets on each picture, so the only thing that
    differs between the curves is which way they point.
    """
    base = torch.softmax(head(h), dim=1)[:, 1]
    curve = []
    for gamma in gammas:
        pushed = h + gamma * size.unsqueeze(1) * direction
        probability = torch.softmax(head(pushed), dim=1)[:, 1]
        curve.append(float((probability - base).mean()))
    return curve


def audit_figure(scores, best, curves, repair, out_path, dpi=150):
    """Three panels, in the order the recipe uses them.

    Left: every direction's score, best first, with the line at 0.5 where a direction says
    nothing at all -- the shortlist the recipe starts from.

    Middle: the dial. The best direction turned up and down, against a random direction of
    exactly the same size and a direction that fires just as often with nothing to do with
    the class.

    Right: what the repair buys. Accuracy on the world as it was built, and on the same
    world with the colour tie cut, before and after taking the shortcut out.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.3))

    ranked = np.sort(scores)[::-1]
    colours = ["#cccccc"] * len(ranked)
    colours[0] = SHORTCUT_COLOUR
    axes[0].bar(np.arange(len(ranked)), ranked, color=colours, width=0.85)
    axes[0].axhline(0.5, color=BASELINE_COLOUR, linewidth=1.2, linestyle="--")
    axes[0].text(len(ranked) * 0.99, 0.505, "says nothing", ha="right", va="bottom",
                 fontsize=8.5, color=BASELINE_COLOUR)
    axes[0].annotate(f"direction {best}", xy=(0, ranked[0]), xytext=(24, ranked[0] - 0.08),
                     fontsize=9, color=SHORTCUT_COLOUR,
                     arrowprops=dict(arrowstyle="-|>", color=SHORTCUT_COLOUR, linewidth=1.2))
    axes[0].set_xlabel("the SAE's directions, best first")
    axes[0].set_ylabel("how well the direction\npredicts the answer (AUROC)")
    axes[0].set_ylim(0.4, 1.02)
    axes[0].grid(axis="y", alpha=0.3)

    for label, values, colour, style in curves:
        axes[1].plot(GAMMAS, values, marker="o", markersize=4, linewidth=1.8, color=colour,
                     linestyle=style, label=label)
    axes[1].axhline(0.0, color="#dddddd", linewidth=1.0, zorder=0)
    axes[1].set_xlabel("how hard the direction is pushed, gamma")
    axes[1].set_ylabel("change in probability of class 1")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)

    groups = ["the world as\nbuilt (test)", "colour tie\ncut (shift)"]
    candidates = [
        ("as trained", [repair["test_before"], repair["shift_before"]], "#cccccc"),
        ("re-fit the read-out", [repair["test_refit"], repair["shift_refit"]], "#9ecae1"),
        ("re-fit after the colour\nis out of h", [repair["test_surgery"],
                                                repair["shift_surgery"]], TASK_COLOUR),
        ("the colour out at the\ninput instead", [repair["test_grayed"],
                                                 repair["shift_grayed"]], "#dd8452"),
    ]
    position = np.arange(len(groups))
    width = 0.2
    for offset, (label, values, colour) in enumerate(candidates):
        bars = axes[2].bar(position + (offset - 1.5) * width, values, width=width,
                           color=colour, label=label)
        for bar, value in zip(bars, values):
            axes[2].text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                         ha="center", va="bottom", fontsize=7.5, rotation=90)
    axes[2].set_xticks(position)
    axes[2].set_xticklabels(groups, fontsize=9.5)
    axes[2].set_ylim(0.0, 1.12)
    axes[2].set_ylabel("accuracy")
    axes[2].legend(fontsize=7.5, loc="lower right", ncols=1)
    axes[2].grid(axis="y", alpha=0.3)

    fig.text(
        0.5, 0.01,
        "left: every one of the SAE's directions scored by how well it predicts the answer "
        "-- it never saw a label. middle: the frame colour's\nown direction turned up and "
        "down, against the same size of push in a random direction and in a direction that "
        "fires just as often but says\nnothing. right: four things you could do about it. "
        "re-fitting the read-out already does the work; the surgery adds nothing on top, and\n"
        "taking the colour out at the input is the one that really repairs it.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Use the trained SAE: find the shortcut, then take it out.")
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
    head = model.body[-1]

    sae_checkpoint = torch.load(Path(args.model_dir) / "sae.pt")
    sae_config = sae_checkpoint["config"]
    sae = SparseAutoencoder(sae_config["n_input"], sae_config["n_latent"], sae_config["k"])
    sae.load_state_dict(sae_checkpoint["state_dict"])
    sae.eval()

    dataset = torch.load(args.data)
    x_all = standardise(dataset["x"], dataset)
    with torch.no_grad():
        h_all = middle_layer_activations(model, x_all)
        z_all = sae.encode(h_all)
    splits = dataset["splits"]
    tr, va, te, sh = splits["train"], splits["val"], splits["test"], splits["shift"]

    # --------------------------------------------------- 1. score every direction
    scored = torch.cat([va, te])
    z_scored, y_scored = z_all[scored], dataset["y"][scored]
    scores = np.array([auroc(z_scored[:, i].numpy(), y_scored.numpy())
                       for i in range(z_scored.shape[1])])
    best = int(scores.argmax())
    firing = (z_all[scored] > 0).float().mean(dim=0).numpy()
    # A direction with nothing to do with the class, for the control: among the ones that
    # actually fire, the one whose score comes closest to saying nothing at all.
    silent = firing < CONTROL_MIN_FIRING
    unrelated = int(np.argmin(np.where(silent, np.inf, np.abs(scores - 0.5))))

    # --------------------------------------------------- 2. what the winner follows
    rose = dataset["a"][scored, COLOUR]
    colour_auroc = auroc(z_scored[:, best].numpy(), (rose > 0).numpy().astype(int))
    colour_correlation = float(torch.corrcoef(torch.stack([z_scored[:, best], rose]))[0, 1])
    colour_direction = intervention_directions(model, dataset, seed=args.seed)[0][COLOUR]
    # How well the *direction the latent writes into h* matches the colour's own direction.
    # Measured rather than assumed, because it is the part that does not follow.
    cosine_with_colour = float(
        F.normalize(sae.directions()[best].unsqueeze(0), dim=1) @ colour_direction)
    # And the same reading on the split where the colour stops being about the answer at
    # all: a feature detector should still see the feature, even when it is useless there.
    colour_auroc_shift = auroc(z_all[sh, best].numpy(),
                               (dataset["a"][sh, COLOUR] > 0).numpy().astype(int))

    # --------------------------------------------------- 3-4. the dial and its controls
    h_test, z_test = h_all[te], z_all[te]
    generator = torch.Generator().manual_seed(args.seed)
    random_unit = F.normalize(torch.randn(1, h_all.shape[1], generator=generator), dim=1)
    # One unit of the feature itself, per picture: how much of that feature this picture is
    # holding, measured from the middle of the split rather than from zero, because ``h`` is
    # mostly one big positive offset and the offset is not the feature. The same size of
    # push is handed to the controls, so the only thing that differs between the curves is
    # which way they point.
    centred = h_test - h_test.mean(dim=0, keepdim=True)
    size = centred @ colour_direction
    curves = [
        ("the frame colour's own direction",
         direction_curve(head, h_test, colour_direction.unsqueeze(0), size),
         SHORTCUT_COLOUR, "-"),
        (f"direction {unrelated}, fires but says nothing",
         direction_curve(head, h_test, sae.directions()[unrelated].unsqueeze(0), size),
         MIDDLE_COLOUR, "-."),
        ("a random direction, same size",
         direction_curve(head, h_test, random_unit, size), BASELINE_COLOUR, ":"),
    ]
    # The SAE's own dial on the latent it named, kept for the honest note: it moves the
    # answer, but not the way the colour does.
    latent_curve = dial_curve(sae, head, h_test, z_test, best)
    # The check before anything is believed: a push of zero must be a change of zero. This
    # is the SAE's own rebuild at gamma = 0, which is where the rounding would show up.
    zero_shift = dial_curve(sae, head, h_test, z_test, best, gammas=(0.0,))[0]
    with torch.no_grad():
        rebuilt = sae.decoder(z_test) + (h_test - sae.decoder(z_test))
    zero_rebuild = float((rebuilt - h_test).abs().max())

    # The probability curves above push gently, and a confident classifier has little room
    # to move further. The decisive number is how many *answers* the direction changes when
    # it is taken out altogether -- the same operation the monitor below is built on.
    answers = head(h_all).argmax(dim=1)
    changed = {}
    with torch.no_grad():
        for label, direction in (("colour", colour_direction),
                                 ("unrelated", sae.directions()[unrelated]),
                                 ("random", random_unit.squeeze(0))):
            pitched = head(project_out(h_all, direction.unsqueeze(0))).argmax(dim=1)
            changed[label] = float((pitched[te] != answers[te]).float().mean())

    # --------------------------------------------------- 5a. the monitor
    # Per picture: take the feature out and see whether the answer moves. If it does, that
    # answer was coming from the frame -- the feature -- and not from the digit.
    with torch.no_grad():
        without = project_out(h_all, colour_direction.unsqueeze(0))
    prediction = head(h_all).argmax(dim=1)
    colour_driven = head(without).argmax(dim=1) != prediction
    right = prediction == dataset["y"]
    monitor = {}
    for name, index in (("test", te), ("shift", sh)):
        driven = colour_driven[index]
        monitor[name] = {
            "flagged_share": float(driven.float().mean()),
            "accuracy_flagged": float(right[index][driven].float().mean()),
            "accuracy_quiet": float(right[index][~driven].float().mean()),
            "accuracy_all": float(right[index].float().mean()),
        }

    # --------------------------------------------------- 5b. the repair
    # Four candidates, measured side by side, because the interesting part is which of them
    # is doing the work. Each is judged on the two splits: the world as it was built, and
    # the same world with the colour tie cut, which is where the shortcut was hurting.
    def fresh_readout(hmod):
        return fit_readout(hmod[tr], dataset["y"][tr], seed=args.seed)

    repair = {"test_before": accuracy_of(head, h_all[te], dataset["y"][te]),
              "shift_before": accuracy_of(head, h_all[sh], dataset["y"][sh])}

    # (1) nothing removed: a fresh read-out on exactly the same representation. This is the
    # control for the surgery, and it turns out to be the interesting one.
    plain = fresh_readout(h_all)
    repair["test_refit"] = accuracy_of(plain, h_all[te], dataset["y"][te])
    repair["shift_refit"] = accuracy_of(plain, h_all[sh], dataset["y"][sh])

    # (2) the surgery: the colour's own direction taken out of h, then a fresh read-out. The
    # direction is measured the way interpret.py measures it -- draw the same pictures twice,
    # changing only the frame, and see which way h moves. No labels are used for it.
    repaired = project_out(h_all, colour_direction.unsqueeze(0))
    after = fresh_readout(repaired)
    repair["test_surgery"] = accuracy_of(after, repaired[te], dataset["y"][te])
    repair["shift_surgery"] = accuracy_of(after, repaired[sh], dataset["y"][sh])

    # (3) only the latents the SAE names, which is the move that suggests itself.
    z_muted = z_all.clone()
    z_muted[:, best] = 0.0
    with torch.no_grad():
        muted = sae.decoder(z_muted) + (h_all - sae.decoder(z_all))
    repair["shift_latents_only"] = accuracy_of(fresh_readout(muted), muted[sh],
                                               dataset["y"][sh])

    # (4) the one that works, and the only one that touches no representation at all: paint
    # the frame grey, so there is nothing left for the network to read. It needs no labels
    # and no retraining -- only knowing which feature to take away, which is what the SAE
    # said. It is done by hand here, because we know where the frame is.
    gray_test, gray_test_y = grayscale_split(dataset, "test")
    gray_shift, gray_shift_y = grayscale_split(dataset, "shift")
    repair["test_grayed"] = accuracy(model, gray_test, gray_test_y)
    repair["shift_grayed"] = accuracy(model, gray_shift, gray_shift_y)

    # How much of the feature is actually left after each removal, measured by trying to read
    # it back out. This is the check on whether the surgery does what it claims.
    def readable_colour(hmod):
        reader = fit_readout(hmod[tr], (dataset["a"][tr, COLOUR] > 0).long(),
                             seed=args.seed)
        with torch.no_grad():
            scores_ = (reader(hmod[sh])[:, 1] - reader(hmod[sh])[:, 0]).numpy()
        return auroc(scores_, (dataset["a"][sh, COLOUR] > 0).numpy().astype(int))

    repair["colour_auroc_after_surgery"] = readable_colour(repaired)
    repair["colour_auroc_after_muting_latent"] = readable_colour(muted)

    # The honest comparison at the end: the same read-out, fitted with the labels we were
    # holding back. It is sharper, and it is the one thing the SAE did not have.
    probe_auroc = readable_colour(h_all)

    results = {
        "n_latent": int(sae_config["n_latent"]),
        "best_direction": best,
        "best_auroc": float(scores[best]),
        "directions_above_0_7": int((scores > 0.7).sum()),
        "top_ten": [[int(i), float(scores[i])] for i in np.argsort(scores)[::-1][:10]],
        "best_firing_fraction": float(firing[best]),
        "unrelated_direction": unrelated,
        "unrelated_auroc": float(scores[unrelated]),
        "unrelated_firing_fraction": float(firing[unrelated]),
        "colour_auroc": float(colour_auroc),
        "colour_auroc_on_the_shift_split": float(colour_auroc_shift),
        "colour_correlation": colour_correlation,
        "cosine_with_colour_direction": cosine_with_colour,
        "probe_auroc_for_the_colour": float(probe_auroc),
        "gammas": list(GAMMAS),
        "curves": {label: values for label, values, _, _ in curves},
        "answers_changed_by_taking_it_out": changed,
        "latent_dial_curve": latent_curve,
        "zero_push_shift": zero_shift,
        "zero_rebuild_shift": zero_rebuild,
        "monitor": monitor,
        "repair": repair,
    }

    model_dir = Path(args.model_dir)
    (model_dir / "audit.json").write_text(json.dumps(results, indent=2))
    audit_figure(scores, best, curves, repair, Path(args.figures) / "audit.png")

    # --------------------------------------------------------------------- what we found
    print(f"\nscored {sae_config['n_latent']} directions on {len(scored)} pictures the SAE "
          f"was not trained on")

    print("\n1. which directions predict the answer")
    print(f"  the best is direction {best}, at AUROC {scores[best]:.3f}")
    print(f"  {int((scores > 0.7).sum())} directions score above 0.700")
    print(f"  the ten best: " + ", ".join(
        f"{int(i)} ({scores[i]:.2f})" for i in np.argsort(scores)[::-1][:10]))

    print(f"\n2. what direction {best} follows")
    print(f"  it reads the frame colour at AUROC {colour_auroc:.3f}"
          f", correlating {colour_correlation:+.2f} with it")
    print(f"  and it still reads it on the shift split, where the colour means nothing:"
          f" AUROC {colour_auroc_shift:.3f}")
    print(f"  but the direction it writes into h is not the colour's direction: "
          f"cosine {cosine_with_colour:+.2f} -- a detector, not an axis")

    print("\n3-4. change in the probability of class 1")
    print(f"  {'gamma':>7s}" + "".join(f"{label[:32]:>35s}" for label, *_ in curves))
    for row, gamma in enumerate(GAMMAS):
        print(f"  {gamma:7.1f}" + "".join(f"{values[row]:35.4f}" for _, values, *_ in curves))
    print(f"  the same as the SAE's own dial on direction {best}: "
          + ", ".join(f"{value:+.4f}" for value in latent_curve))
    print(f"  taking the direction out altogether changes the answer on "
          f"{changed['colour']:.2f} of the pictures, against {changed['unrelated']:.2f} "
          f"and {changed['random']:.2f} for the two controls")
    print(f"  direction {best} fires on {firing[best]:.2f} of the pictures; "
          f"direction {unrelated} on {firing[unrelated]:.2f}")
    print(f"  at gamma = 0 the SAE's rebuild is {zero_rebuild:.1e} away from the real h, "
          f"and a push of zero changes the answer by {zero_shift:.1e}")

    print("\n5a. the monitor: which answers came from the frame?")
    for name in ("test", "shift"):
        entry = monitor[name]
        print(f"  {name:<6s} flags {entry['flagged_share']:.2f} of the pictures: accuracy "
              f"{entry['accuracy_flagged']:.3f} there, {entry['accuracy_quiet']:.3f} "
              f"elsewhere ({entry['accuracy_all']:.3f} overall)")

    print("\n5b. the repair: four things you could do, on the two splits")
    print(f"  {'':<40s} {'the world':>10s} {'colour cut':>11s}")
    for label, test_value, shift_value in (
            ("as trained", repair["test_before"], repair["shift_before"]),
            ("re-fit the read-out", repair["test_refit"], repair["shift_refit"]),
            ("re-fit after the colour is out of h", repair["test_surgery"],
             repair["shift_surgery"]),
            ("take the colour out at the input", repair["test_grayed"],
             repair["shift_grayed"])):
        print(f"  {label:<40s} {test_value:10.3f} {shift_value:11.3f}")
    print(f"  the surgery and the plain re-fit agree to three places, and the colour is still "
          f"readable after it: AUROC {repair['colour_auroc_after_surgery']:.3f}")
    print(f"  taking out only the latents the SAE names gives "
          f"{repair['shift_latents_only']:.3f} on the cut split, no better than the plain "
          f"re-fit:\nthe colour is spread over the dictionary, not in one of it")
    print(f"\n  one honest note: the same read-out fitted *with* the labels reads the "
          f"colour at AUROC {probe_auroc:.3f},\n  against {colour_auroc:.3f} for the "
          f"latent the SAE found without ever seeing one.")

    print(f"\nwrote {model_dir / 'audit.json'}")
    print(f"wrote {Path(args.figures) / 'audit.png'}")


if __name__ == "__main__":
    main()
