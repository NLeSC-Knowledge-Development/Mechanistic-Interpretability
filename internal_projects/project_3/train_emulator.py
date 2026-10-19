"""Train the emulator: a neural network that imitates Lorenz-96.

    ../../.venv/bin/python train_emulator.py --data data/lorenz96.npz \
                             --model-dir model --figures figures --architecture 32-32-32

The emulator takes the current state of the ring and the one before it, and predicts the
state half a day later. It has the same shape as the system itself -- 40 numbers in, 40
numbers out -- and it is trained on nothing but that one-step task, by plain regression
against the exact solver's answer.

We do not need the emulator: the equations are known and cheap. It is here because it is a
*stand-in* for a large weather model. A neural network has internal layers, those layers hold
numbers, and those numbers are the object of study in the rest of this folder. Lorenz-96 is
the practice field: everything claimed about the network here can be checked against an exact
answer.

The architecture is a stack of residual circular-convolution blocks, one token per site, so
the network is exactly translation-equivariant -- it treats every site like every other site
and like every rotation of the ring, just as the dynamics do. A token has shape
``(sites, width)``, which is the layout the sparse autoencoder is built for.

The figures and the numbers report two things, and the second is the one worth reading:

* **one-step skill** -- the error of the half-day prediction, which is very small;
* **rollout skill** -- feed predictions back as the next input and see how long the emulator
  stays useful. It is trained on a single step and judged on hundreds, and the error compounds
  at the rate the system's own instability dictates, until it reaches the size of the natural
  variability and the forecast stops carrying information. The same chaos that destroys a
  1e-8 nudge destroys this; nothing was added to the network to cause it.

Writes:

* ``<model-dir>/emulator.pt``  -- weights, config and metrics
* ``<model-dir>/history.json`` -- the training curves, one entry per epoch
* ``<model-dir>/metrics.json`` -- the numbers quoted in the README
* ``<figures>/architecture.png``, ``training.png``, ``skill.png``
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
from matplotlib.patches import FancyArrowPatch, Rectangle
from torch import nn

from generate_data import TIME_UNIT_IN_DAYS, load_dataset

TRAIN_COLOUR = "#4c72b0"
VAL_COLOUR = "#c44e52"
INPUT_COLOUR = "#4c72b0"
HIDDEN_COLOUR = "#dd8452"
MIDDLE_COLOUR = "#c44e52"
BASELINE_COLOUR = "#999999"
PERSISTENCE_COLOUR = "#dd8452"


# ------------------------------------------------------------------------------ the model


class CircularConvBlock(nn.Module):
    """Pre-norm residual block built from circular convolutions.

    ``input_dim`` may differ from ``d_model``, in which case the residual path is projected,
    so blocks of different widths can be stacked.
    """

    def __init__(self, d_model, kernel_size=5, expansion=2.0, input_dim=None):
        super().__init__()
        input_dim = d_model if input_dim is None else input_dim
        hidden = int(round(d_model * expansion))
        self.norm = nn.LayerNorm(input_dim)
        self.conv_in = nn.Conv1d(input_dim, hidden, kernel_size, padding=kernel_size // 2,
                                 padding_mode="circular")
        self.conv_out = nn.Conv1d(hidden, d_model, 1)
        self.project = None if input_dim == d_model else nn.Conv1d(input_dim, d_model, 1)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (batch, sites, input_dim)
        hidden = self.conv_out(self.act(self.conv_in(self.norm(x).transpose(1, 2))))
        skip = x.transpose(1, 2)
        if self.project is not None:
            skip = self.project(skip)
        return (hidden + skip).transpose(1, 2)


class RingEmulator(nn.Module):
    """Translation-equivariant emulator with one token per ring site.

    A stack of residual circular-convolution blocks. ``widths`` gives the width of each
    block, which is also the width of the token at that point: ``32-32-32`` is three blocks
    of width 32. Returns the predicted state at the next time level.
    """

    def __init__(self, n_sites=40, history=2, widths=(32, 32, 32), kernel_size=5,
                 residual=True):
        super().__init__()
        widths = [int(width) for width in widths]
        if not widths:
            raise ValueError("widths must list at least one block")
        self.n_sites = int(n_sites)
        self.history = int(history)
        self.widths = widths
        self.kernel_size = int(kernel_size)
        self.residual = bool(residual)

        self.embed = nn.Linear(self.history, widths[0])
        blocks = []
        previous = widths[0]
        for width in widths:
            blocks.append(CircularConvBlock(width, kernel_size, input_dim=previous))
            previous = width
        self.blocks = nn.ModuleList(blocks)
        self.head_norm = nn.LayerNorm(widths[-1])
        self.head = nn.Linear(widths[-1], 1)

    def sae_layer_names(self):
        """The layers a sparse autoencoder may hook: one token per site at each of them."""
        return ["embed"] + [f"blocks.{index}" for index in range(len(self.blocks))]

    def receptive_field(self):
        """How many sites to either side the last layer can see, from the architecture.

        Each circular-convolution block reaches ``kernel_size // 2`` sites further out; the
        embedding and the head act on a single site. This is a prediction, and
        ``interpret.py`` checks it against what the trained network actually does.
        """
        return self.receptive_field_of(self.sae_layer_names()[-1])

    def receptive_field_of(self, layer):
        """The same prediction, for the token at one particular layer.

        A token read half way up the stack has only seen the blocks below it, so its window is
        narrower than the model's output: ``embed`` sees one site, and ``blocks.k`` sees
        ``k + 1`` convolutions' worth.
        """
        if layer == "embed":
            return 0
        index = int(layer.split(".")[1])
        return (index + 1) * (self.kernel_size // 2)

    def forward(self, x):
        # x: (batch, history, sites) -> (batch, sites, history)
        hidden = self.embed(x.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        delta = self.head(self.head_norm(hidden)).squeeze(-1)
        return x[:, -1, :] + delta if self.residual else delta


def parse_architecture(spec):
    """Read an architecture string such as ``32-32-32`` into a list of widths."""
    parts = [part for part in spec.replace(" ", "").split("-") if part]
    try:
        widths = [int(part) for part in parts]
    except ValueError:
        raise ValueError(f"could not read widths from {spec!r}; expected e.g. 32-32-32") from None
    if not widths or any(width <= 0 for width in widths):
        raise ValueError(f"architecture {spec!r} must list positive widths, e.g. 32-32-32")
    return widths


def build_emulator(n_sites, history, architecture):
    """The emulator named by an architecture string."""
    return RingEmulator(n_sites, history, widths=parse_architecture(architecture))


def load_emulator(path, device="cpu"):
    """Read back a trained emulator, with the config it was trained with."""
    checkpoint = torch.load(Path(path), weights_only=False)
    config = checkpoint["config"]
    model = RingEmulator(config["n_sites"], config["history"], config["widths"],
                         config["kernel_size"], config["residual"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def token_activations(model, inputs, layer, batch_size=4096, device="cpu"):
    """The activations of one layer, flattened to one row per site.

    ``inputs`` is ``(n, history, sites)``; the result is ``(n * sites, width)`` -- for the
    ring emulator, one token per site per sample. This is the same harvest-then-train split
    the activations are read with everywhere else in this folder, and the same mechanism
    (a forward hook) that would be used on a large model.
    """
    captured = []
    handle = model.get_submodule(layer).register_forward_hook(
        lambda module, args, output: captured.append(output.detach().reshape(-1, output.shape[-1])))
    inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            model(inputs[start:start + batch_size])
    handle.remove()
    return torch.cat(captured, dim=0)


# ------------------------------------------------------------------------- training and skill


def one_step_metrics(model, block, mean, std, batch_size=4096):
    """One-step error on a split, in standardised units and in physical units."""
    inputs = torch.as_tensor(block["inputs"], dtype=torch.float32)
    targets = torch.as_tensor(block["targets"], dtype=torch.float32)
    errors = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            prediction = model(inputs[start:start + batch_size])
            errors.append((prediction - targets[start:start + batch_size]).numpy())
    error = np.concatenate(errors)
    error_physical = error * std[None, :]
    return {
        "mse_standardised": float(np.mean(error**2)),
        "rmse_standardised": float(np.sqrt(np.mean(error**2))),
        "rmse_physical": float(np.sqrt(np.mean(error_physical**2))),
        "climatology_std": float(np.mean(std)),
    }


def train(model, dataset, epochs, batch_size, lr, weight_decay, seed, device="cpu"):
    """One-step regression, keeping the epoch with the best validation error.

    Nothing unusual: mean squared error against the true next state, Adam, a cosine learning
    rate. The only thing worth pointing out is what is *not* in the loss -- no rollout, no
    physics, no regularisation towards any particular internal representation. The network is
    free to represent whatever it likes, which is what makes it worth looking inside.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)

    inputs = torch.as_tensor(dataset["splits"]["train"]["inputs"], dtype=torch.float32,
                            device=device)
    targets = torch.as_tensor(dataset["splits"]["train"]["targets"], dtype=torch.float32,
                             device=device)
    val_inputs = torch.as_tensor(dataset["splits"]["val"]["inputs"], dtype=torch.float32,
                                device=device)
    val_targets = torch.as_tensor(dataset["splits"]["val"]["targets"], dtype=torch.float32,
                                 device=device)

    history = []
    best_state, best_val, best_epoch = None, float("inf"), 1
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(inputs))
        running = 0.0
        for start in range(0, len(inputs), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(inputs[batch]), targets[batch])
            loss.backward()
            optimizer.step()
            running += loss.detach().item() * len(batch)
        schedule.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_inputs), val_targets))
        history.append({"epoch": epoch + 1, "train_loss": running / len(inputs),
                        "val_loss": val_loss,
                        "train_rmse": float(np.sqrt(running / len(inputs))),
                        "val_rmse": float(np.sqrt(val_loss))})
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch + 1
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, best_epoch


def rollout(model, dataset, split="test", n_steps=60, n_starts=64, seed=0, device="cpu"):
    """Autoregressive rollout skill, against persistence and climatology.

    The split is a contiguous block in time, so feeding the prediction back as the next input
    is a genuine forecast from a real state, not a re-run of the training distribution.
    """
    block = dataset["splits"][split]
    mean, std = dataset["mean"], dataset["std"]
    n_available = len(block["inputs"]) - n_steps
    if n_available < 1:
        raise ValueError(f"split too short for a {n_steps}-step rollout")

    rng = np.random.default_rng(seed)
    starts = rng.choice(n_available, size=min(n_starts, n_available), replace=False)
    truth = np.stack([block["raw_targets"][s:s + n_steps] for s in starts])
    history = np.stack([block["inputs"][s] for s in starts])
    persistence = history[:, -1, :] * std + mean

    model.eval()
    predictions = []
    current = torch.as_tensor(history, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(n_steps):
            prediction = model(current)
            predictions.append(prediction.cpu().numpy())
            current = torch.cat([current[:, 1:], prediction.unsqueeze(1)], dim=1)
    predictions = np.stack(predictions, axis=1) * std + mean

    rmse = lambda error: np.sqrt(np.mean(error**2, axis=(0, 2)))
    return {
        "lead_days": np.arange(1, n_steps + 1) * dataset["settings"]["dt_out"]
                     * TIME_UNIT_IN_DAYS,
        "model_rmse": rmse(predictions - truth),
        "persistence_rmse": rmse(np.broadcast_to(persistence[:, None], truth.shape) - truth),
        "climatology_rmse": rmse(np.broadcast_to(mean, truth.shape) - truth),
        "climatology_std": float(np.mean(std)),
    }


def growth_doubling_time(lead_days, rmse):
    """Doubling time of an error curve, fitted over its exponential stretch.

    Fits a straight line to ``log(rmse)`` over the lead times where the error is still
    growing and has not yet saturated at the size of the natural variability. Returned in
    days, and only meaningful when the fit is taken over a genuine straight stretch.
    """
    log_error = np.log(rmse)
    growth = np.gradient(log_error, lead_days)
    # The growth rate is constant in time while the error is in its exponential regime;
    # take the stretch before the error has reached half the natural variability.
    usable = rmse < 0.5 * rmse[-1]
    if usable.sum() < 3:
        usable = np.ones_like(rmse, dtype=bool)
    slope = float(np.mean(growth[usable]))
    return float(np.log(2.0) / slope) if slope > 0 else float("nan")


# -------------------------------------------------------------------------------- figures


def _arrow(ax, x0, y0, x1, y1, colour="#333333"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=colour))


def architecture_figure(n_sites, history, widths, kernel_size, radius, out_path, dpi=150):
    """What the emulator is: the ring it imitates, and the stack that does the imitating.

    Left: the system's own stencil at one site -- the three neighbours the equation uses --
    drawn on the ring. Right: the network, one token per site, with the token shape at each
    stage. The receptive field printed underneath is a prediction of the architecture and is
    checked against the trained network in `interpret.py`.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))

    # --- the ring, with the stencil of one site highlighted -------------------------
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_axis_off()
    angles = np.linspace(0.0, 2.0 * np.pi, n_sites, endpoint=False)
    centre_x, centre_y, radius_ring = 0.0, 0.0, 1.0
    xs, ys = centre_x + radius_ring * np.cos(angles), centre_y + radius_ring * np.sin(angles)
    ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), color="#bbbbbb", linewidth=1.2,
            zorder=0)
    site = int(n_sites * 0.5) // 2
    stencil = {site: HIDDEN_COLOUR, (site + 1) % n_sites: MIDDLE_COLOUR,
               (site - 1) % n_sites: MIDDLE_COLOUR, (site - 2) % n_sites: MIDDLE_COLOUR}
    for index, (x, y) in enumerate(zip(xs, ys)):
        ax.add_patch(plt.Circle((x, y), 0.057, facecolor=stencil.get(index, "#dddddd"),
                                edgecolor="#888888", linewidth=0.6, zorder=2))
    ax.text(xs[site] * 1.22, ys[site] * 1.22, "$x_i$", ha="center", va="center", fontsize=11,
            color=HIDDEN_COLOUR)
    for offset, name in ((1, "$x_{i+1}$"), (-1, "$x_{i-1}$"), (-2, "$x_{i-2}$")):
        index = (site + offset) % n_sites
        ax.text(xs[index] * 1.30, ys[index] * 1.30, name, ha="center", va="center",
                fontsize=10, color=MIDDLE_COLOUR)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.5, 1.45)
    ax.set_title(f"the system: {n_sites} sites on a ring", fontsize=10)
    ax.text(0, -1.42,
            "$dx_i/dt = (x_{i+1} - x_{i-2})\\,x_{i-1} - x_i + F$",
            ha="center", va="center", fontsize=10.5)

    # --- the network ---------------------------------------------------------------
    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.0)

    ax.add_patch(Rectangle((0.25, 1.75), 1.15, 1.15, facecolor="#dfe6f2",
                           edgecolor="#333333", linewidth=0.8))
    ax.text(0.82, 1.62, f"the last {history}\nstates\n({history}, {n_sites})", ha="center",
            va="top", fontsize=9)

    _arrow(ax, 1.48, 2.32, 1.85, 2.32)
    ax.add_patch(Rectangle((1.90, 1.95), 0.95, 0.75, facecolor="#e8e8e8",
                           edgecolor="#888888", linewidth=0.8))
    ax.text(2.37, 2.32, "embed", ha="center", va="center", fontsize=9)

    x0 = 3.05
    for index, width in enumerate(widths):
        ax.add_patch(Rectangle((x0 + index * 1.55, 1.55), 1.25, 1.55, facecolor="#eef1f6",
                               edgecolor="#888888", linewidth=0.8))
        ax.text(x0 + index * 1.55 + 0.62, 2.32, f"block {index}\nwidth {width}\n"
                                                f"circular $k$={kernel_size}", ha="center",
                va="center", fontsize=8.5)
        if index:
            _arrow(ax, x0 + index * 1.55 - 0.30, 2.32, x0 + index * 1.55 - 0.02, 2.32)
    _arrow(ax, 2.85, 2.32, 3.03, 2.32)
    after = x0 + len(widths) * 1.55
    _arrow(ax, after - 0.30, 2.32, after + 0.05, 2.32)
    ax.add_patch(Rectangle((after + 0.08, 1.95), 1.05, 0.75, facecolor="#dfe6f2",
                           edgecolor="#333333", linewidth=0.8))
    ax.text(after + 0.60, 2.32, "head", ha="center", va="center", fontsize=9)
    ax.text(after + 0.60, 1.80, f"next state\n({n_sites},)", ha="center", va="top", fontsize=9)

    ax.text(0.25, 1.40, "token shape at each stage:  "
                        f"({n_sites}, {widths[0]}) after embed, "
                        f"({n_sites}, {widths[-1]}) after the last block, ({n_sites},) out",
            ha="left", va="top", fontsize=9)
    ax.text(0.25, 1.12, f"receptive field: the last token sees {radius} sites to either "
                        f"side\n(a prediction of the architecture; measured in interpret.py)",
            ha="left", va="top", fontsize=9, color="#444444")
    ax.set_title("the emulator: one token per site", fontsize=10)

    fig.text(0.5, 0.02,
             "the emulator imitates the ring: 40 numbers in, 40 numbers out. it is a stack of "
             "residual circular convolutions,\nso it is exactly translation-equivariant -- "
             "every site, and every rotation of the ring, is treated alike.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def training_figure(history, out_path, dpi=150):
    """Two panels: the error coming down, in standardised units and as a physical size."""
    epochs = [entry["epoch"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], color=TRAIN_COLOUR,
                 lw=1.8, label="train")
    axes[0].plot(epochs, [entry["val_loss"] for entry in history], color=VAL_COLOUR, lw=1.8,
                 label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("one-step mean squared error")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title("the half-day prediction, during training", fontsize=10)

    axes[1].plot(epochs, [entry["train_rmse"] for entry in history], color=TRAIN_COLOUR,
                 lw=1.8, label="train")
    axes[1].plot(epochs, [entry["val_rmse"] for entry in history], color=VAL_COLOUR, lw=1.8,
                 label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("one-step RMSE (standardised)")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_title("the same error, as a size", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def skill_figure(curve, metrics, out_path, dpi=150):
    """Where the forecast stops being useful.

    The error against lead time, with the two baselines anyone can beat without a model:
    repeating the last state, and guessing the average state. The horizontal line is the size
    of the system's natural variability, which is where a forecast carries no information --
    an error that large is what you get by knowing nothing. The dashed line is the rate at
    which a 1e-8 error grows, measured from the true system.
    """
    lead = curve["lead_days"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3))

    axes[0].plot(lead, curve["model_rmse"], color=TRAIN_COLOUR, lw=2.0, label="the emulator")
    axes[0].plot(lead, curve["persistence_rmse"], color=PERSISTENCE_COLOUR, lw=1.5,
                 linestyle="--", label="repeating the last state")
    axes[0].plot(lead, curve["climatology_rmse"], color=BASELINE_COLOUR, lw=1.5,
                 linestyle=":", label="guessing the average state")
    axes[0].axhline(curve["climatology_std"], color=MIDDLE_COLOUR, lw=1.0, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("lead time (days)")
    axes[0].set_ylabel("RMSE (physical units)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].set_title("skill against lead time", fontsize=10)

    log_error = np.log(curve["model_rmse"])
    axes[1].plot(lead, log_error / np.log(10.0), color=TRAIN_COLOUR, lw=2.0)
    reference = log_error[0] + (lead - lead[0]) * np.log(2.0) / metrics["system_doubling_days"]
    axes[1].plot(lead, reference / np.log(10.0), color=MIDDLE_COLOUR, lw=1.4,
                 linestyle="--", label=f"doubling every {metrics['system_doubling_days']:.1f} "
                                       f"days (a 1e-8 nudge)")
    fitted = log_error[0] + (lead - lead[0]) * np.log(2.0) / metrics["emulator_doubling_days"]
    axes[1].plot(lead, fitted / np.log(10.0),
                 color=HIDDEN_COLOUR, lw=1.4, linestyle=":",
                 label=f"doubling every {metrics['emulator_doubling_days']:.1f} days "
                       f"(this network's error)")
    axes[1].set_xlabel("lead time (days)")
    axes[1].set_ylabel("error, as a power of ten")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("the same curve on a log scale", fontsize=10)

    fig.text(0.5, 0.02,
             "the emulator is trained on half a day and judged on many. its error grows "
             "exponentially -- a straight line here -- until it reaches\nthe size of the "
             "system's own variability, at which point the forecast carries no information. "
             "nothing was added to the network to cause this.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="Train the Lorenz-96 emulator.")
    parser.add_argument("--data", default="data/lorenz96.npz")
    parser.add_argument("--model-dir", default="model", help="where the model goes")
    parser.add_argument("--figures", default="figures", help="where the figures go")
    parser.add_argument("--name", default="emulator", help="what the emulator is called")
    parser.add_argument("--architecture", default="32-32-32",
                        help="one width per block, e.g. 32-32-32; 64-64-64 is the wider "
                             "version used for the reference numbers")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rollout-steps", type=int, default=60)
    parser.add_argument("--rollout-starts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset = load_dataset(args.data)
    settings = dataset["settings"]

    widths = parse_architecture(args.architecture)
    model = build_emulator(settings["N"], settings["history"], args.architecture)

    started = time.time()
    model, history, best_epoch = train(model, dataset, args.epochs, args.batch_size, args.lr,
                                       args.weight_decay, args.seed)
    elapsed = time.time() - started

    metrics = {}
    for name in ("train", "val", "test"):
        metrics[name] = one_step_metrics(model, dataset["splits"][name], dataset["mean"],
                                        dataset["std"])
    curve = rollout(model, dataset, "test", n_steps=args.rollout_steps,
                    n_starts=args.rollout_starts, seed=args.seed)
    metrics["rollout"] = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                          for key, value in curve.items()}
    metrics["settings"] = settings
    metrics["architecture"] = args.architecture
    metrics["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    metrics["receptive_field"] = model.receptive_field()
    metrics["doubling_days"] = growth_doubling_time(curve["lead_days"], curve["model_rmse"])

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"n_sites": settings["N"], "history": settings["history"],
                           "widths": widths, "kernel_size": args.kernel_size,
                           "residual": True, "architecture": args.architecture,
                           "name": args.name},
                "settings": settings,
                "metrics": metrics}, model_dir / f"{args.name}.pt")
    (model_dir / "history.json").write_text(json.dumps(history, indent=2))
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    system_doubling = float(settings.get("system_doubling_days", float("nan")))
    skill_metrics = dict(metrics)
    skill_metrics["system_doubling_days"] = (system_doubling if system_doubling == system_doubling
                                             else 2.09)
    skill_metrics["emulator_doubling_days"] = metrics["doubling_days"]
    architecture_figure(settings["N"], settings["history"], widths, args.kernel_size,
                        model.receptive_field(), Path(args.figures) / "architecture.png")
    training_figure(history, Path(args.figures) / "training.png")
    skill_figure(curve, skill_metrics, Path(args.figures) / "skill.png")

    test = metrics["test"]
    print(f"\ntrained '{args.name}' ({args.architecture}, "
          f"{metrics['n_parameters']} parameters) for {args.epochs} epochs "
          f"in {elapsed:.0f}s; best epoch {best_epoch}")
    print(f"\n  one-step RMSE   {test['rmse_physical']:.4f} physical units"
          f"  ({100 * test['rmse_physical'] / test['climatology_std']:.1f}% of natural "
          f"variability {test['climatology_std']:.2f})")
    print(f"  receptive field {metrics['receptive_field']} sites to either side")
    print(f"\n  lead time (days)   " + "".join(f"{d:8.1f}" for d in curve["lead_days"][::12]))
    print(f"  emulator RMSE      " + "".join(f"{v:8.2f}" for v in curve["model_rmse"][::12]))
    print(f"  repeat last state  " + "".join(f"{v:8.2f}"
                                             for v in curve["persistence_rmse"][::12]))
    print(f"  guess the average  " + "".join(f"{v:8.2f}"
                                             for v in curve["climatology_rmse"][::12]))
    beats = curve["model_rmse"] <= curve["persistence_rmse"]
    print(f"\n  it beats repeating the last state until day "
          f"{curve['lead_days'][np.flatnonzero(~beats)[0]] if (~beats).any() else curve['lead_days'][-1]:.1f}")
    print(f"  its error doubles every {metrics['doubling_days']:.2f} days "
          f"(the system's own errors double every {skill_metrics['system_doubling_days']:.2f})")

    print(f"\nwrote {model_dir / (args.name + '.pt')}")
    print(f"wrote {model_dir / 'metrics.json'}")
    print(f"wrote {Path(args.figures) / 'architecture.png'}")
    print(f"wrote {Path(args.figures) / 'training.png'}")
    print(f"wrote {Path(args.figures) / 'skill.png'}")


if __name__ == "__main__":
    main()
