"""Train a sparse autoencoder (SAE) on the emulator's internal activations.

    ../../.venv/bin/python train_sae.py --data data/lorenz96.npz --model-dir model \
                                        --figures figures --layer blocks.1

The SAE is a second, separate network. It never sees the ring, it never sees the equations,
and it never sees the true state: the only thing it is given is one layer's activations -- one
token per site, a handful of numbers long -- and its only job is to rebuild them.

It runs in two halves:

* **encoder** -- given a token, decide which of its ``--n-latent`` directions are present, and
  how strongly;
* **decoder** -- add those directions up, with those strengths, to rebuild the token.

Two rules make it useful, and both are visible in the code:

1. **Sparse.** Only the ``--k`` strongest directions are kept and the rest are set to exactly
   zero. Forcing almost everything to be off is what makes the list break the muddle into
   separate pieces, instead of smearing every direction over every token.
2. **It is trained for one thing only:** make the rebuilt token as close to the real one as
   possible -- plain squared error, with no labels and no physics anywhere.

Two things are reported before anything is interpreted.

* **Reconstruction.** How much of the token's variation comes back. A dictionary that cannot
  rebuild its own input is not describing anything.
* **How much room the token has.** The number of principal directions of the token
  distribution that hold 95% of its variance. This is the cheap check worth doing on any model
  before running an SAE: if a handful of numbers already carry everything, there is nothing
  hidden for a dictionary to find, and a sparse autoencoder will mostly rediscover ordinary
  principal components.

Writes:

* ``<model-dir>/sae.pt``           -- weights, config and metrics
* ``<model-dir>/sae_history.json`` -- reconstruction per epoch
* ``<model-dir>/sae_metrics.json`` -- the numbers quoted in the README
* ``<figures>/sae.png``            -- what the SAE is
* ``<figures>/sae_training.png``   -- reconstruction, and the directions going unused
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
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch import nn

from generate_data import load_dataset
from train_emulator import (HIDDEN_COLOUR, INPUT_COLOUR, MIDDLE_COLOUR, TRAIN_COLOUR,
                            VAL_COLOUR, _arrow, load_emulator, token_activations)

# The dictionary is wider than the token by a good margin, with a small number of latents
# allowed on at once -- the same shape the first two projects used.
N_LATENT = 512
K = 16
OFF_COLOUR = "#dcdcdc"


class SparseAutoencoder(nn.Module):
    """A token in, the same token out, through a short list of directions that is mostly off.

    The list of directions is the decoder: one direction per latent, each as long as the
    token. The encoder says how much of each direction is present. The sparsity is the
    ``topk`` in :meth:`encode`, which keeps the ``k`` strongest and sets the rest to zero.
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

        A direction should say *which way*, and the encoder should say *how strongly*. If the
        directions were free to stretch, the two would trade off and neither would mean
        anything on its own.
        """
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, tokens):
        """Which directions are present, and how strongly: exactly ``k`` of them."""
        pre = F.relu(self.encoder(tokens - self.decoder.bias))
        values, indices = torch.topk(pre, self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, indices, values)

    def forward(self, tokens):
        z = self.encode(tokens)
        return self.decoder(z), z

    def directions(self):
        """The dictionary, one unit direction per row."""
        return self.decoder.weight.detach().T.contiguous()


def reconstruction_fraction(tokens, rebuilt):
    """How much of the token's variation the SAE rebuilds: 1 is perfect, 0 is no better than
    the average."""
    error = ((rebuilt - tokens) ** 2).mean()
    total = tokens.var(dim=0, unbiased=False).mean()
    return float(1.0 - error.detach() / total)


def principal_room(tokens, threshold=0.95):
    """How many directions of the token distribution hold ``threshold`` of its variance.

    The token's *effective width*, measured rather than assumed. Fewer than a handful means
    the layer has already squeezed its state into a couple of numbers, and there is little for
    a dictionary to separate.
    """
    centred = tokens - tokens.mean(dim=0, keepdim=True)
    covariance = (centred.T @ centred) / max(len(centred) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0.0)
    cumulative = torch.cumsum(eigenvalues, dim=0) / eigenvalues.sum()
    return int(torch.searchsorted(cumulative, threshold).item()) + 1


def train(tokens, tokens_val, n_latent, k, epochs, batch_size, lr, seed, device="cpu"):
    """Train the SAE to rebuild tokens, and report the validation reconstruction per epoch.

    Only activations are used: there is no label and no equation anywhere in this function.

    Directions that never win a top-k slot are given a fresh job at the end of every epoch.
    Without that they stay off forever -- not because there is nothing left to explain, but
    because they never got to try.
    """
    torch.manual_seed(seed)
    sae = SparseAutoencoder(tokens.shape[1], n_latent, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 1)

    history = []
    for epoch in range(epochs):
        sae.train()
        order = rng.permutation(len(tokens))
        running = 0.0
        fired = torch.zeros(n_latent, dtype=torch.bool, device=device)
        for start in range(0, len(tokens), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            rebuilt, z = sae(tokens[batch])
            loss = F.mse_loss(rebuilt, tokens[batch])
            loss.backward()
            optimizer.step()
            sae.normalise_decoder()
            with torch.no_grad():
                fired |= (z > 0).any(dim=0)
            running += loss.detach().item() * len(batch)
        schedule.step()

        # Not after the last epoch: a direction re-drawn then would never be trained, so the
        # model returned here would not be the one the last history entry describes, and the
        # saved artifact would carry directions nobody ever fitted.
        if epoch + 1 < epochs:
            _resample_dead(sae, optimizer, tokens, (~fired).nonzero(as_tuple=True)[0],
                           generator)

        sae.eval()
        with torch.no_grad():
            rebuilt_train, _ = sae(tokens)
            rebuilt_val, _ = sae(tokens_val)
        history.append({
            "epoch": epoch + 1,
            "train_loss": running / len(tokens),
            "train_reconstruction": reconstruction_fraction(tokens, rebuilt_train),
            "val_reconstruction": reconstruction_fraction(tokens_val, rebuilt_val),
            "never_on": int((~fired).sum()),
        })

    sae.eval()
    return sae, history


def _resample_dead(sae, optimizer, tokens, dead, generator):
    """Give the directions that never fire a new job.

    A direction that never wins a top-k slot never gets a gradient either, so it stays off
    forever. Hand each dead direction the leftover of one token -- the part of it the other
    directions failed to rebuild -- and it has something real to compete on.
    """
    if dead.numel() == 0:
        return
    index = torch.randint(0, len(tokens), (int(dead.numel()),), generator=generator,
                          device=tokens.device)
    with torch.no_grad():
        batch = tokens[index]
        rebuilt, _ = sae(batch)
        residual = F.normalize(batch - rebuilt, dim=1)
        scale = sae.encoder.weight.data.norm(dim=1).median()
        sae.encoder.weight.data[dead] = residual * scale
        sae.encoder.bias.data[dead] = 0.0
        sae.decoder.weight.data[:, dead] = residual.T
    sae.normalise_decoder()
    for parameter, dim in ((sae.encoder.weight, 0), (sae.encoder.bias, 0),
                           (sae.decoder.weight, 1)):
        entry = optimizer.state.get(parameter)
        if not entry:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            if key in entry and entry[key].dim() > 0:
                entry[key].index_fill_(dim, dead, 0.0)


# ------------------------------------------------------------------------------- figures


def _sparse_stack(ax, x, y0, y1, n, k, width, on_colour=MIDDLE_COLOUR, off_colour=OFF_COLOUR):
    """``n`` bars stacked vertically, of which ``k`` are switched on.

    There really are ``n`` bars, one per direction, so the column is as tall as the list is
    long; the ``k`` that are lit are drawn a little taller than a single bar and spread
    evenly, so that a few among hundreds can be seen at all.
    """
    step = (y1 - y0) / n
    height = step * 3.0
    for index in range(n):
        ax.add_patch(Rectangle((x, y0 + index * step), width, step, facecolor=off_colour,
                               edgecolor="none"))
    for index in range(k):
        centre = y0 + (y1 - y0) * (index + 0.5) / k
        ax.add_patch(Rectangle((x - 0.02, centre - height / 2), width + 0.04, height,
                               facecolor=on_colour, edgecolor="none"))


def sae_figure(n_input, n_latent, k, n_sites, layer, out_path, dpi=150):
    """What the SAE is: one token in, a mostly-off list of directions, the token rebuilt."""
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.4)
    ax.set_axis_off()

    _sparse_stack(ax, 1.64, 1.42, 4.06, n_latent, k, 0.5)
    ax.add_patch(Rectangle((1.52, 1.30), 0.74, 2.88, fill=False, linestyle="--",
                           linewidth=1.6, edgecolor=MIDDLE_COLOUR))
    ax.text(1.89, 4.18, "which directions are present,\nand how strongly", ha="center",
            va="bottom", fontsize=9)
    ax.text(1.89, 1.22, f"{n_latent} directions\neach {n_input} numbers long", ha="center",
            va="top", fontsize=9)

    ax.add_patch(Rectangle((0.58, 0.60), 0.18, 0.18, facecolor=MIDDLE_COLOUR,
                           edgecolor="none"))
    ax.text(0.84, 0.69, f"{k} on", ha="left", va="center", fontsize=9, color=MIDDLE_COLOUR)
    ax.add_patch(Rectangle((2.02, 0.60), 0.18, 0.18, facecolor=OFF_COLOUR,
                           edgecolor="#bbbbbb", linewidth=0.5))
    ax.text(2.28, 0.69, f"the other {n_latent - k}, exactly zero", ha="left", va="center",
            fontsize=9, color="#777777")

    for x, colour, top_text, bottom_text, side_text, text_colour in (
        (0.36, HIDDEN_COLOUR, "the token at\none site", f"{n_input} numbers",
         "what the emulator holds", MIDDLE_COLOUR),
        (2.96, INPUT_COLOUR, "the rebuilt\ntoken", f"{n_input} numbers",
         "as close to the real token\nas we can get", INPUT_COLOUR)):
        drawn = min(n_input, 16)
        step = (3.42 - 2.06) / drawn
        for index in range(drawn):
            ax.add_patch(Rectangle((x, 2.06 + index * step + 0.08 * step), 0.42, 0.84 * step,
                                   facecolor=colour, edgecolor="white", linewidth=0.4))
        ax.add_patch(Rectangle((x - 0.11, 1.95), 0.64, 1.58, fill=False, linestyle="--",
                               linewidth=1.6, edgecolor=MIDDLE_COLOUR))
        ax.text(x + 0.21, 1.87, f"{top_text}\n{bottom_text}", ha="center", va="top",
                fontsize=9)
        ax.text(x + 0.21, 3.55, side_text, ha="center", va="bottom", fontsize=9,
                color=text_colour)

    _arrow(ax, 1.00, 2.74, 1.48, 2.74)
    ax.text(1.24, 2.87, "encoder", ha="center", va="bottom", fontsize=9)
    _arrow(ax, 2.32, 2.74, 2.80, 2.74)
    ax.text(2.56, 2.87, "decoder", ha="center", va="bottom", fontsize=9)

    ax.text(4.35, 2.74,
            "one thin bar per direction.\n\n"
            f"only {k} of the {n_latent} are allowed on,\n"
            f"so the other {n_latent - k} are exactly zero --\n"
            "that is what breaks the muddle\ninto separate pieces.\n\n"
            "it is trained for one thing only:\n"
            "rebuild the token. it never sees the\n"
            "ring, the equations, or the next state.",
            ha="left", va="center", fontsize=9.5, color="#333333", linespacing=1.5)

    fig.text(0.5, 0.02,
             f"the sparse autoencoder: a second, separate network. it reads the activation at "
             f"'{layer}' -- one token per site, {n_sites} per state --\ndecides which of its "
             f"{n_latent} directions are present, and adds them back up to rebuild the token.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def training_figure(history, out_path, dpi=150):
    """Three panels: the error coming down, how much of the token comes back, and the
    directions that were going unused."""
    epochs = [entry["epoch"] for entry in history]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9))
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], color=TRAIN_COLOUR,
                 lw=1.6)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("rebuild error")
    axes[0].set_yscale("log")
    axes[0].set_title("how far the rebuild is off", fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [entry["train_reconstruction"] for entry in history], label="train",
                 color=TRAIN_COLOUR, lw=1.6)
    axes[1].plot(epochs, [entry["val_reconstruction"] for entry in history],
                 label="validation", color=VAL_COLOUR, lw=1.6)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("how much of the token is rebuilt")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend()
    axes[1].set_title("how much of the token comes back", fontsize=10)
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [entry["never_on"] for entry in history], color=MIDDLE_COLOUR, lw=1.6)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("directions that sat out")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("directions going unused", fontsize=10)
    axes[2].grid(alpha=0.3)

    fig.text(0.5, 0.01,
             "the SAE is shown one layer's activations and nothing else: not the ring, not "
             "the equations, not the next state.\na direction that never fires is re-drawn at "
             "the end of every epoch, which is why the right-hand panel comes down.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(
        description="Train a sparse autoencoder on the emulator's activations.")
    parser.add_argument("--data", default="data/lorenz96.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--model-name", default="emulator", help="which emulator to read")
    parser.add_argument("--layer", default="blocks.1",
                        help="which layer to take activations from, by module name")
    parser.add_argument("--n-latent", type=int, default=N_LATENT)
    parser.add_argument("--k", type=int, default=K, help="directions allowed on at once")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="larger steps make a TopK dictionary of this size unstable")
    parser.add_argument("--train-states", type=int, default=2500,
                        help="how many states of the training split to harvest tokens from")
    parser.add_argument("--val-states", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset = load_dataset(args.data)
    settings = dataset["settings"]
    model, checkpoint = load_emulator(Path(args.model_dir) / f"{args.model_name}.pt")
    n_sites = settings["N"]

    # Harvest tokens from a slice of the training block, and a separate slice of the
    # validation block, so the reconstruction numbers are not the SAE quoting itself back.
    train_inputs = dataset["splits"]["train"]["inputs"][:args.train_states]
    val_inputs = dataset["splits"]["val"]["inputs"][:args.val_states]
    tokens_train = token_activations(model, train_inputs, args.layer)
    tokens_val = token_activations(model, val_inputs, args.layer)

    started = time.time()
    sae, history = train(tokens_train, tokens_val, args.n_latent, args.k, args.epochs,
                         args.batch_size, args.lr, args.seed)
    elapsed = time.time() - started

    with torch.no_grad():
        rebuilt_train, z_train = sae(tokens_train)
        active_per_token = float((z_train > 0).sum(dim=1).float().mean())
        never_on = int((z_train > 0).sum(dim=0).eq(0).sum())
        firing_rate = (z_train > 0).float().mean(dim=0)

    metrics = {
        "layer": args.layer,
        "n_input": int(tokens_train.shape[1]),
        "n_latent": int(args.n_latent),
        "k": int(args.k),
        "n_tokens_train": int(len(tokens_train)),
        "n_tokens_val": int(len(tokens_val)),
        "reconstruction": {"train": reconstruction_fraction(tokens_train, rebuilt_train),
                           "val": reconstruction_fraction(
                               tokens_val, sae(tokens_val)[0])},
        "active_per_token": active_per_token,
        "never_on": never_on,
        "firing_rate": firing_rate.numpy().tolist(),
        "principal_room_95": principal_room(tokens_train),
        "principal_room_99": principal_room(tokens_train, 0.99),
        "token_variance": float(tokens_train.var(dim=0, unbiased=False).sum()),
    }

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sae.state_dict(),
                "config": {"n_input": metrics["n_input"], "n_latent": int(args.n_latent),
                           "k": int(args.k), "layer": args.layer,
                           "model_name": args.model_name},
                "metrics": metrics}, model_dir / "sae.pt")
    (model_dir / "sae_history.json").write_text(json.dumps(history, indent=2))
    (model_dir / "sae_metrics.json").write_text(json.dumps(metrics, indent=2))

    sae_figure(metrics["n_input"], args.n_latent, args.k, n_sites, args.layer,
               Path(args.figures) / "sae.png")
    training_figure(history, Path(args.figures) / "sae_training.png")

    print(f"\ntrained the SAE for {args.epochs} epochs in {elapsed:.0f}s")
    print(f"  layer '{args.layer}': {metrics['n_input']} numbers per token, "
          f"{len(tokens_train)} training tokens from {args.train_states} states")
    print(f"\n  {'split':<6s} {'rebuilt':>9s}")
    for name in ("train", "val"):
        print(f"  {name:<6s} {metrics['reconstruction'][name]:9.4f}")
    print(f"\n  directions on per token   {active_per_token:.0f}  (forced to {args.k})")
    print(f"  directions never on       {never_on} of {args.n_latent}")
    print(f"  the token needs {metrics['principal_room_95']} principal directions for 95% of "
          f"its variance, {metrics['principal_room_99']} for 99%, out of {metrics['n_input']}")
    print(f"\nwrote {model_dir / 'sae.pt'}")
    print(f"wrote {model_dir / 'sae_metrics.json'}")
    print(f"wrote {Path(args.figures) / 'sae.png'}")
    print(f"wrote {Path(args.figures) / 'sae_training.png'}")


if __name__ == "__main__":
    main()
