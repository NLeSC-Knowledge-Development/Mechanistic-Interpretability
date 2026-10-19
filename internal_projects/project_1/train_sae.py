"""Train a sparse autoencoder (SAE) on the classifier's middle layer.

    python train_sae.py --data data/dataset.pth --model-dir model --figures figures/sign

The SAE is a second, separate network. It never sees the pictures and it never sees the
labels. The only thing it is given is ``h`` -- the classifier's middle layer, 32 numbers --
and its only job is to rebuild it.

It runs in two halves:

* **encoder** -- given ``h``, decide which of its ``--n-latent`` directions are present, and
  how strongly;
* **decoder** -- add those directions up, with those strengths, to rebuild ``h``.

Two rules make it useful, and both are visible in the code below:

1. **Sparse.** Only the ``--k`` strongest directions are kept and the rest are set to
   exactly zero -- see ``SparseAutoencoder.encode``. Forcing almost everything to be off is
   what makes the list break the muddle into separate pieces, instead of smearing every
   direction over every picture.
2. **It is trained for one thing only:** make "rebuilt h" as close to the real ``h`` as
   possible -- the loss is the plain squared error against ``h``, with no labels anywhere.

Because we built our own world, we can do more than train it. We know the 25 true spots, so
we can ask whether the learned directions line up with them at all, by comparing each spot's
direction in ``h`` with the closest SAE direction -- and against a list of *random*
directions, to see what "line up" is worth. That check is the reason the numbers in the
README mean something.

Writes:

* ``<model-dir>/sae.pt``           -- the trained SAE, with the config and the final numbers
* ``<model-dir>/sae_history.json`` -- the reconstruction, one entry per epoch
* ``<model-dir>/sae_metrics.json`` -- the numbers quoted in the README
* ``<figures>/sae.png``            -- what the SAE looks like
* ``<figures>/sae_training.png``   -- the training curves
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle

# The classifier, its middle layer and the way it is read back are one story, so the pieces
# are imported rather than copied.
from train_classifier import (Classifier, HIDDEN_COLOUR, INPUT_COLOUR, MIDDLE_COLOUR,
                              MIDDLE_LAYER, TRAIN_COLOUR, VAL_COLOUR, _arrow, _number_stack)

# The SAE, fixed on purpose at the same shape the README describes: 256 directions, each as
# long as ``h``, of which only 8 may be on for any one picture.
N_LATENT = 256
K = 8
# The directions that are off are drawn light grey, so the few that are on stand out.
OFF_COLOUR = "#dcdcdc"


class SparseAutoencoder(nn.Module):
    """``h`` in, ``h`` out, through a short list of directions that is mostly off.

    The list of directions is the decoder: one direction per row, each as long as ``h``.
    The encoder says how much of each direction is present. The sparsity is the ``topk`` in
    ``encode`` -- it keeps the ``k`` strongest and sets the rest to exactly zero.
    """

    def __init__(self, n_input, n_latent=N_LATENT, k=K):
        super().__init__()
        self.n_latent = n_latent
        self.k = k
        self.encoder = nn.Linear(n_input, n_latent)
        self.decoder = nn.Linear(n_latent, n_input)
        self.normalise_decoder()

    def normalise_decoder(self):
        """Make every direction unit length.

        A direction should say *which way*, and the encoder should say *how strongly*; if
        the directions were free to stretch, the two would trade off and neither would mean
        anything on its own. Dividing the encoder input by the decoder bias, below, is the
        other half of the same idea.
        """
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, h):
        """Which directions are present, and how strongly: exactly ``k`` of them, the rest 0."""
        pre = F.relu(self.encoder(h - self.decoder.bias))
        values, indices = torch.topk(pre, self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, indices, values)

    def forward(self, h):
        z = self.encode(h)
        return self.decoder(z), z

    def directions(self):
        """The list of directions, one per latent, each as long as ``h``."""
        return self.decoder.weight.detach().T.contiguous()


def middle_layer_activations(model, x, layer=MIDDLE_LAYER, batch_size=8192):
    """``h`` for every example: the output of the classifier's middle layer.

    The classifier is only ever run forward and never changed -- the SAE is trained
    afterwards, from the ``h`` values this reads out.
    """
    captured = []
    handle = model.get_submodule(layer).register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach()))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            model(x[start:start + batch_size])
    handle.remove()
    return torch.cat(captured, dim=0)


def reconstruction_fraction(h, h_hat):
    """How much of ``h``'s variation the SAE rebuilds: 1 is perfect, 0 is no better than
    the average."""
    error = ((h_hat - h) ** 2).mean()
    total = h.var(dim=0, unbiased=False).mean()
    return float(1.0 - error / total)


def best_alignment(directions, true_directions):
    """How well a list of directions matches the true spots.

    For every true spot, the direction that lines up with it best (by the size of the
    cosine), averaged over the spots. The same measure is used for the random list, so the
    two numbers are directly comparable.
    """
    directions = F.normalize(directions, dim=1)
    true_directions = F.normalize(true_directions, dim=1)
    cosine = (true_directions @ directions.T).abs()
    return float(cosine.max(dim=1).values.mean())


def spot_directions(h, a):
    """Every true spot's own direction inside ``h``, fitted from the true spot strengths.

    One direction per spot: the way ``h`` moves when that spot is given more ink, fitted
    straight from the true spot strengths. Fitting matters, because the usual shortcut --
    push the spot's picture through the first layer -- steps straight over the GELU bend
    between the picture and ``h``.
    """
    design = torch.cat([torch.ones(len(a), 1), a], dim=1)
    coefficients = torch.linalg.lstsq(design, h).solution
    return F.normalize(coefficients[1:], dim=1)


def train(h, h_val, n_latent, k, epochs, batch_size, lr, seed, device):
    """Train the SAE to rebuild ``h``, and report the validation reconstruction per epoch.

    Only ``h`` is used: there is no label anywhere in this function.

    Directions that never win the top-k are given a fresh job at the end of every epoch --
    see ``_resample_dead``. Without that they stay off forever, not because there is
    nothing left to explain but because they never got to try, and a good part of the list
    goes to waste.
    """
    torch.manual_seed(seed)
    sae = SparseAutoencoder(h.shape[1], n_latent, k).to(device)
    optimiser = torch.optim.Adam(sae.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    rng = np.random.default_rng(seed)
    # Reproducible for the resampling, which does not go through the numpy source above.
    generator = torch.Generator(device=device).manual_seed(seed + 1)

    history = []
    for epoch in range(epochs):
        sae.train()
        order = rng.permutation(len(h))
        running = 0.0
        # Which directions fired at least once this epoch.
        fired = torch.zeros(n_latent, dtype=torch.bool, device=device)
        for start in range(0, len(h), batch_size):
            batch = order[start:start + batch_size]
            optimiser.zero_grad()
            h_hat, z = sae(h[batch])
            loss = F.mse_loss(h_hat, h[batch])
            loss.backward()
            optimiser.step()
            # Keep the directions unit length after every step, so the split between
            # "which direction" and "how strong" stays where the first rule put it.
            sae.normalise_decoder()
            with torch.no_grad():
                fired |= (z > 0).any(dim=0)
            running += loss.detach().item() * len(batch)
        schedule.step()

        # The directions that sat out the whole epoch get pointed at something new.
        _resample_dead(sae, optimiser, h, (~fired).nonzero(as_tuple=True)[0], generator)

        sae.eval()
        with torch.no_grad():
            h_hat_train, _ = sae(h)
            h_hat_val, _ = sae(h_val)
        history.append({
            "epoch": epoch + 1,
            "train_loss": running / len(h),
            "train_reconstruction": reconstruction_fraction(h, h_hat_train),
            "val_reconstruction": reconstruction_fraction(h_val, h_hat_val),
            # How many directions sat the whole epoch out, before they were re-drawn.
            "never_on": int((~fired).sum()),
        })

    sae.eval()
    return sae, history


def _resample_dead(sae, optimiser, h, dead, generator):
    """Give the directions that never fire a new job.

    A direction that never wins the top-k never gets a gradient either, so it stays off
    forever -- not because there is nothing left to explain, but because it never got to
    try. Hand each dead direction the leftover of one example: the part of ``h`` the other
    directions failed to rebuild. It then has something real to compete on, and the count
    of directions that never turn on falls.
    """
    if dead.numel() == 0:
        return
    index = torch.randint(0, len(h), (int(dead.numel()),), generator=generator, device=h.device)
    with torch.no_grad():
        batch = h[index]
        h_hat, _ = sae(batch)
        residual = F.normalize(batch - h_hat, dim=1)
        # Match the size of the directions that are already working, so a new one can
        # actually win the top-k on the example it was just handed.
        scale = sae.encoder.weight.data.norm(dim=1).median()
        sae.encoder.weight.data[dead] = residual * scale
        sae.encoder.bias.data[dead] = 0.0
        sae.decoder.weight.data[:, dead] = residual.T
    sae.normalise_decoder()
    # Forget the optimiser's history for what was just re-drawn, so the new values are not
    # undone by momentum on the very next step.
    for param, dim in ((sae.encoder.weight, 0), (sae.encoder.bias, 0),
                       (sae.decoder.weight, 1)):
        entry = optimiser.state.get(param)
        if not entry:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            if key in entry and entry[key].dim() > 0:
                entry[key].index_fill_(dim, dead, 0.0)


def training_figure(history, out_path, dpi=150):
    """Three panels: the error coming down, how much of h comes back, and the directions
    that were going unused.

    The right-hand panel is the interesting one. The SAE starts out leaving a good part of
    its list unused, because a direction that never wins the top-k never gets a gradient
    either -- it stays off for good. Re-drawing those directions every epoch (see
    ``_resample_dead``) brings the count steadily down, until none are left over.
    """
    epochs = [entry["epoch"] for entry in history]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9))

    axes[0].plot(epochs, [entry["train_loss"] for entry in history], color=TRAIN_COLOUR,
                 lw=1.6)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("reconstruction MSE")
    axes[0].set_yscale("log")
    axes[0].set_title("reconstruction error", fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [entry["train_reconstruction"] for entry in history],
                 label="train", color=TRAIN_COLOUR, lw=1.6)
    axes[1].plot(epochs, [entry["val_reconstruction"] for entry in history],
                 label="validation", color=VAL_COLOUR, lw=1.6)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("fraction of variance explained")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("variance of h explained", fontsize=10)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [entry["never_on"] for entry in history], color=MIDDLE_COLOUR,
                 lw=1.6)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("unused directions")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("directions never active", fontsize=10)
    axes[2].grid(alpha=0.3)

    fig.text(
        0.5, 0.01,
        "The SAE receives h only: no images and no labels. The objective is reconstruction "
        "of h.\nDirections that never activate are reinitialised each epoch, so the right "
        "panel reaches zero.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def sae_figure(n_input, n_latent, k, out_path, dpi=150):
    """What the SAE looks like.

    Left to right: the ``h`` the classifier holds, the encoder turning it into a list of
    directions that is almost all off, and the decoder adding the few that are on back up
    to rebuild ``h``. The column in the middle is the tall one -- it is the big part of the
    SAE, and only a few stripes on it are lit. Drawn from the sizes actually used, so the
    counts on the picture are the counts in the trained SAE.
    """
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.4)
    ax.set_axis_off()

    # The list of directions: one thin bar each, almost all of them off.
    _sparse_stack(ax, 1.64, 1.42, 4.06, n_latent, k, 0.5)
    ax.add_patch(Rectangle((1.52, 1.30), 0.74, 2.88, fill=False, linestyle="--",
                           linewidth=1.6, edgecolor=MIDDLE_COLOUR))
    ax.text(1.89, 4.18, "latent activations:\nwhich directions are present, and how strongly",
            ha="center", va="bottom", fontsize=9)
    ax.text(1.89, 1.22, f"{n_latent} directions\neach {n_input} numbers long", ha="center",
            va="top", fontsize=9)

    # A key for the column: the few bars that are on, and the many that are not.
    ax.add_patch(Rectangle((0.58, 0.60), 0.18, 0.18, facecolor=MIDDLE_COLOUR,
                           edgecolor="none"))
    ax.text(0.84, 0.69, f"{k} active", ha="left", va="center", fontsize=9,
            color=MIDDLE_COLOUR)
    ax.add_patch(Rectangle((2.02, 0.60), 0.18, 0.18, facecolor=OFF_COLOUR,
                           edgecolor="#bbbbbb", linewidth=0.5))
    ax.text(2.28, 0.69, f"the other {n_latent - k}, exactly zero", ha="left", va="center",
            fontsize=9, color="#777777")

    # The h going in on the left, and the h coming back out on the right.
    _number_stack(ax, 0.36, 2.06, 3.42, n_input, 0.42, HIDDEN_COLOUR)
    ax.add_patch(Rectangle((0.25, 1.95), 0.64, 1.58, fill=False, linestyle="--",
                           linewidth=1.6, edgecolor=MIDDLE_COLOUR))
    ax.text(0.56, 1.87, f"the middle  h\n{n_input} numbers", ha="center", va="top",
            fontsize=9)
    ax.text(0.56, 3.55, "classifier hidden activation", ha="center", va="bottom",
            fontsize=9, color=MIDDLE_COLOUR)

    _arrow(ax, 1.00, 2.74, 1.48, 2.74)
    ax.text(1.24, 2.87, "encoder", ha="center", va="bottom", fontsize=9)

    _arrow(ax, 2.32, 2.74, 2.80, 2.74)
    ax.text(2.56, 2.87, "decoder", ha="center", va="bottom", fontsize=9)

    _number_stack(ax, 2.96, 2.06, 3.42, n_input, 0.42, INPUT_COLOUR)
    ax.add_patch(Rectangle((2.85, 1.95), 0.64, 1.58, fill=False, linestyle="--",
                           linewidth=1.6, edgecolor=MIDDLE_COLOUR))
    ax.text(3.16, 1.87, f"rebuilt  h\n{n_input} numbers", ha="center", va="top", fontsize=9)
    ax.text(3.16, 3.55, "reconstruction of h", ha="center", va="bottom",
            fontsize=9, color=INPUT_COLOUR)

    # The two rules, written beside the column they are about.
    ax.text(4.35, 2.74,
            "One bar per direction.\n\n"
            f"Only {k} of the {n_latent} are allowed to be nonzero,\n"
            f"so the remaining {n_latent - k} are exactly zero.\n"
            "This constraint forces the representation\nto separate features.\n\n"
            "The only training objective is\n"
            "reconstruction of h; labels are not used.",
            ha="left", va="center", fontsize=9.5, color="#333333", linespacing=1.5)

    fig.text(
        0.5, 0.02,
        f"Sparse autoencoder: a separate network. It maps h to {n_latent} latent strengths, "
        f"keeps the {k} largest\nand sets the rest to zero, then reconstructs h from the "
        f"active directions.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _sparse_stack(ax, x, y0, y1, n, k, width, on_colour=MIDDLE_COLOUR, off_colour=OFF_COLOUR):
    """``n`` bars stacked vertically, of which ``k`` are switched on.

    The sparse list at a glance: a tall column that is almost entirely off, with a handful
    of stripes that are on. There really are ``n`` bars -- one per direction -- so the
    column is as tall as the list is long; the ``k`` that are lit are drawn a little taller
    than a single bar and spread evenly, so that a few among hundreds can be seen at all.
    """
    step = (y1 - y0) / n
    height = step * 3.0
    for i in range(n):
        ax.add_patch(Rectangle((x, y0 + i * step), width, step, facecolor=off_colour,
                               edgecolor="none"))
    for j in range(k):
        centre = y0 + (y1 - y0) * (j + 0.5) / k
        ax.add_patch(Rectangle((x - 0.02, centre - height / 2), width + 0.04, height,
                               facecolor=on_colour, edgecolor="none"))


def main():
    parser = argparse.ArgumentParser(description="Train a sparse autoencoder on h.")
    parser.add_argument("--data", default="data/dataset.pth")
    parser.add_argument("--model-dir", default="model",
                        help="where the trained classifier is, and the SAE goes")
    parser.add_argument("--figures", default="figures/sign", help="where the figures go")
    parser.add_argument("--n-latent", type=int, default=N_LATENT)
    parser.add_argument("--k", type=int, default=K, help="directions allowed on at once")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    # The classifier, as trained, and h read out of its middle layer.
    checkpoint = torch.load(Path(args.model_dir) / "classifier.pt")
    config = checkpoint["config"]
    model = Classifier(config["n_pixels"], config["n_hidden"], config["n_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    dataset = torch.load(args.data)
    x_all = ((dataset["x"] - dataset["mean"]) / dataset["std"]).to(device)
    h_all = middle_layer_activations(model, x_all).cpu()
    h_by_split = {name: h_all[index] for name, index in dataset["splits"].items()}

    started = time.time()
    sae, history = train(h_by_split["train"], h_by_split["val"], args.n_latent, args.k,
                         args.epochs, args.batch_size, args.lr, args.seed, device)

    # How much of h it rebuilds, on every split.
    with torch.no_grad():
        reconstruction = {}
        for name, h in h_by_split.items():
            h_hat, _ = sae(h)
            reconstruction[name] = reconstruction_fraction(h, h_hat)

        # How many directions are on, and how many never turn on at all.
        z_train = sae.encode(h_by_split["train"])
        active_per_example = float((z_train > 0).sum(dim=1).float().mean())
        never_on = int((z_train > 0).sum(dim=0).eq(0).sum())

    # Each true spot's direction inside h, fitted from the true spot strengths. This is
    # the same definition interpret.py uses, so the number here and the recovery figure
    # are the same measurement.
    with torch.no_grad():
        train_index = dataset["splits"]["train"]
        true_directions = spot_directions(h_all[train_index], dataset["a"][train_index])
        match = best_alignment(sae.directions(), true_directions)
        generator = torch.Generator().manual_seed(args.seed)
        random_directions = torch.randn(args.n_latent, true_directions.shape[1],
                                        generator=generator)
        random_match = best_alignment(random_directions, true_directions)

    metrics = {
        "reconstruction": reconstruction,
        "active_per_example": active_per_example,
        "never_on": never_on,
        "n_latent": int(args.n_latent),
        "k": int(args.k),
        "match_true_spots": match,
        "match_random": random_match,
        "n_spots": int(true_directions.shape[0]),
    }

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": sae.state_dict(),
        "config": {"n_input": int(h_all.shape[1]), "n_latent": int(args.n_latent),
                   "k": int(args.k), "middle_layer": MIDDLE_LAYER},
        "metrics": metrics,
        "args": vars(args),
    }, model_dir / "sae.pt")
    (model_dir / "sae_history.json").write_text(json.dumps(history, indent=2))
    (model_dir / "sae_metrics.json").write_text(json.dumps(metrics, indent=2))

    sae_figure(int(h_all.shape[1]), args.n_latent, args.k, Path(args.figures) / "sae.png")
    training_figure(history, Path(args.figures) / "sae_training.png")

    print(f"\ntrained the SAE for {args.epochs} epochs in {time.time() - started:.1f}s")
    print(f"\n  {'split':<6s} {'rebuilt':>9s}")
    for name in ("train", "val", "test", "cf"):
        print(f"  {name:<6s} {reconstruction[name]:9.4f}")
    print(f"\n  directions on per picture   {active_per_example:.0f}"
          f"  (forced to {args.k})")
    print(f"  directions never on         {never_on} of {args.n_latent}")
    print(f"\n  how well its directions match the true spots: {match:.3f}")
    print(f"  the same score for random directions:         {random_match:.3f}")
    print(f"\nwrote {model_dir / 'sae.pt'}")
    print(f"wrote {model_dir / 'sae_history.json'}")
    print(f"wrote {model_dir / 'sae_metrics.json'}")
    print(f"wrote {Path(args.figures) / 'sae.png'}")
    print(f"wrote {Path(args.figures) / 'sae_training.png'}")

if __name__ == "__main__":
    main()
