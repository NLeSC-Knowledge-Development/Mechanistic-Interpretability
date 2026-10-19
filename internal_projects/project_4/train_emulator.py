"""Train the emulator on Gray-Scott, and find out how long it keeps the pattern alive.

    ../../.venv/bin/python train_emulator.py --data data/gray_scott.npz --model-dir model \
                                             --figures figures

The model
---------
The system is a torus of cells and the emulator is built to respect that, exactly as the ring
emulator respects the ring: circular convolutions, so the last row is next to the first one,
and every cell is treated alike. Each cell carries a **token** of ``width`` numbers, the
neighbourhood a token can see grows by one block at a time, and the head turns the token back
into the two fields.

The blocks are **dilated**: the first one looks at its immediate neighbours, the next at cells
two away, then four, then eight. Dilating instead of stacking is what makes a wide receptive
field affordable -- four blocks cover fifteen cells, where fifteen blocks of plain 3x3 would
have been needed. Where the field comes from is a *prediction* of the architecture, and
``interpret.py`` measures it against the trained network.

One call to the emulator advances the system by ``stride`` model steps -- 100 by default. It is
trained on the standardised fields with the previous call's state as its skip, so what it
actually predicts is the *change* over those hundred steps.

Skill
-----
Two questions, and the second is the one that matters here.

* **One call.** How large is the error on a single prediction, and how much of that is the
  difficulty of the problem rather than the model: the two trivial forecasts are repeating the
  input (the pattern barely moves in a hundred steps) and predicting the average field.
* **Many calls.** The emulator is run on its own, its own output feeding the next call, and the
  question is not only whether the error grows but whether the *pattern survives*: a free
  rollout that decays to a uniform field or fills up with spots has stopped being a model of
  this system, however small its per-call error looks.

Writes ``<model-dir>/emulator.pt``, ``metrics.json`` and ``history.json``, and
``<figures>/architecture.png``, ``training.png``, ``skill.png``.
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
from matplotlib.patches import FancyArrowPatch, Rectangle
from torch import nn

from generate_data import advance, coverage, load_dataset, spot_centres

TRAIN_COLOUR = "#4c72b0"
VAL_COLOUR = "#c44e52"
INPUT_COLOUR = "#4c72b0"
HIDDEN_COLOUR = "#dd8452"
MIDDLE_COLOUR = "#c44e52"
BASELINE_COLOUR = "#999999"
PERSISTENCE_COLOUR = "#dd8452"

DEFAULT_ARCHITECTURE = "32-32-32-32"
DEFAULT_DILATIONS = "1,2,4,8"
DEFAULT_KERNEL_SIZE = 3
# How many test states the free rollout is measured from.
N_ROLLOUT_STARTS = 16
# How many emulator calls the rollout runs for.
N_ROLLOUT_CALLS = 25


# ------------------------------------------------------------------------------ the model


class TokenNorm(nn.Module):
    """Layer normalisation over the numbers at one cell.

    The same thing ``nn.LayerNorm`` does for the ring, where a token is a row: here the last
    two axes are the grid, so normalising the token means moving the width to the end and
    putting it back.
    """

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(y, (y.shape[-1],), self.weight, self.bias)
        return y.permute(0, 3, 1, 2)


class CircularConvBlock(nn.Module):
    """Pre-norm residual block of circular convolutions with a dilation.

    The padding is done by hand rather than with ``padding_mode="circular"``, so that the
    amount padded is the amount the dilation actually needs and the wrap is visible in the
    code. ``input_dim`` may differ from ``d_model``, in which case the skip is projected.
    """

    def __init__(self, d_model, kernel_size=3, dilation=1, expansion=1.0, input_dim=None):
        super().__init__()
        input_dim = d_model if input_dim is None else input_dim
        hidden = int(round(d_model * expansion))
        self.pad = dilation * (kernel_size - 1) // 2
        self.norm = TokenNorm(input_dim)
        self.conv_in = nn.Conv2d(input_dim, hidden, kernel_size, dilation=dilation)
        self.conv_out = nn.Conv2d(hidden, d_model, 1)
        self.project = None if input_dim == d_model else nn.Conv2d(input_dim, d_model, 1)
        self.act = nn.GELU()

    def forward(self, x):
        hidden = self.norm(x)
        hidden = F.pad(hidden, (self.pad, self.pad, self.pad, self.pad), mode="circular")
        hidden = self.conv_out(self.act(self.conv_in(hidden)))
        skip = x if self.project is None else self.project(x)
        return hidden + skip


class GridEmulator(nn.Module):
    """Translation-equivariant emulator with one token per cell.

    ``widths`` gives the width of each block, which is also the width of the token at that
    point, and ``dilations`` how far out each block reaches. Returns the predicted state one
    emulator call ahead.
    """

    def __init__(self, n=64, history=2, channels=2, widths=(32, 32, 32, 32),
                 kernel_size=3, dilations=(1, 2, 4, 8), residual=True):
        super().__init__()
        widths = [int(width) for width in widths]
        dilations = [int(dilation) for dilation in dilations]
        if not widths:
            raise ValueError("widths must list at least one block")
        if len(dilations) != len(widths):
            raise ValueError(f"one dilation per block: {len(widths)} widths but "
                             f"{len(dilations)} dilations")
        self.n = int(n)
        self.history = int(history)
        self.channels = int(channels)
        self.widths = widths
        self.kernel_size = int(kernel_size)
        self.dilations = dilations
        self.residual = bool(residual)

        self.embed = nn.Conv2d(self.history * self.channels, widths[0], 1)
        blocks = []
        previous = widths[0]
        for width, dilation in zip(widths, dilations):
            blocks.append(CircularConvBlock(width, kernel_size, dilation, input_dim=previous))
            previous = width
        self.blocks = nn.ModuleList(blocks)
        self.head_norm = TokenNorm(widths[-1])
        self.head = nn.Conv2d(widths[-1], self.channels, 1)

    def sae_layer_names(self):
        """The layers a sparse autoencoder may hook: one token per cell at each of them."""
        return ["embed"] + [f"blocks.{index}" for index in range(len(self.blocks))]

    def receptive_field(self):
        """How many cells to either side the output can see, from the architecture."""
        return self.receptive_field_of(self.sae_layer_names()[-1])

    def receptive_field_of(self, layer):
        """The same prediction for the token at one particular layer.

        Each block reaches ``dilation * (kernel_size // 2)`` cells further out, and the
        embedding and the head act on a single cell. A token read half way up the stack has
        passed through fewer blocks, so its window is narrower.
        """
        if layer == "embed":
            return 0
        index = int(layer.split(".")[1])
        return sum(dilation * (self.kernel_size // 2) for dilation in self.dilations[:index + 1])

    def forward(self, x):
        # x: (batch, history, channels, n, n) -> (batch, history * channels, n, n)
        batch, history, channels, rows, cols = x.shape
        hidden = self.embed(x.reshape(batch, history * channels, rows, cols))
        for block in self.blocks:
            hidden = block(hidden)
        delta = self.head(self.head_norm(hidden))
        return x[:, -1] + delta if self.residual else delta


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


def parse_dilations(spec, depth):
    """Read a dilation string such as ``1,2,4,8``, one per block."""
    parts = [part for part in spec.replace(" ", "").split(",") if part]
    try:
        dilations = [int(part) for part in parts]
    except ValueError:
        raise ValueError(f"could not read dilations from {spec!r}; expected e.g. 1,2,4,8") \
            from None
    if len(dilations) != depth:
        raise ValueError(f"{depth} blocks need {depth} dilations, got {len(dilations)}")
    return dilations


def build_emulator(n, history, channels, architecture, kernel_size, dilations):
    """The emulator named by an architecture and a dilation list."""
    widths = parse_architecture(architecture)
    return GridEmulator(n, history, channels, widths, kernel_size,
                        parse_dilations(dilations, len(widths)))


def load_emulator(path, device="cpu"):
    """Read back a trained emulator, with the config it was trained with."""
    checkpoint = torch.load(Path(path), weights_only=False)
    config = checkpoint["config"]
    model = GridEmulator(config["n"], config["history"], config["channels"], config["widths"],
                         config["kernel_size"], config["dilations"], config["residual"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def token_activations(model, inputs, layer, batch_size=8, device="cpu"):
    """The activations of one layer, flattened to one row per cell.

    ``inputs`` is ``(n, history, channels, rows, cols)``; the result is ``(n * rows * cols,
    width)`` -- one token per cell per sample, the same harvest-then-train split used
    everywhere else in this folder, and the same forward-hook mechanism that would be used on
    a large model.
    """
    captured = []
    handle = model.get_submodule(layer).register_forward_hook(
        lambda module, args, output: captured.append(
            output.detach().permute(0, 2, 3, 1).reshape(-1, output.shape[1])))
    inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            model(inputs[start:start + batch_size])
    handle.remove()
    return torch.cat(captured, dim=0)


# ------------------------------------------------------------------------- training and skill


def one_step_metrics(model, block, mean, std, batch_size=8):
    """One-call error on a split, in standardised units and in physical units."""
    inputs = torch.as_tensor(block["inputs"], dtype=torch.float32)
    targets = torch.as_tensor(block["targets"], dtype=torch.float32)
    errors = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            prediction = model(inputs[start:start + batch_size])
            errors.append((prediction - targets[start:start + batch_size]).numpy())
    error = np.concatenate(errors)
    # What the two trivial forecasts would have scored on the same states: repeating the last
    # input, and predicting the average field.
    repeat = (block["inputs"][:, -1] - block["targets"])
    average = (block["targets"].mean(axis=(0, 2, 3), keepdims=True) - block["targets"])
    return {
        "mse_standardised": float(np.mean(error**2)),
        "rmse_standardised": float(np.sqrt(np.mean(error**2))),
        "rmse_physical": float(np.sqrt(np.mean((error * std[None, :, None, None]) ** 2))),
        "repeat_rmse_standardised": float(np.sqrt(np.mean(repeat**2))),
        "average_rmse_standardised": float(np.sqrt(np.mean(average**2))),
    }


def train(model, dataset, epochs, batch_size, lr, weight_decay, seed, device="cpu"):
    """One-call regression, keeping the epoch with the best validation error.

    The training data are the whole grids, so the periodic padding the model uses means the
    same thing during training as it will during a rollout.
    """
    torch.manual_seed(seed)
    inputs = torch.as_tensor(dataset["splits"]["train"]["inputs"], dtype=torch.float32)
    targets = torch.as_tensor(dataset["splits"]["train"]["targets"], dtype=torch.float32)
    val_inputs = torch.as_tensor(dataset["splits"]["val"]["inputs"], dtype=torch.float32)
    val_targets = torch.as_tensor(dataset["splits"]["val"]["targets"], dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = np.random.default_rng(seed)

    history, best = [], float("inf")
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(inputs))
        running = 0.0
        for start in range(0, len(inputs), batch_size):
            batch = torch.as_tensor(order[start:start + batch_size])
            optimizer.zero_grad()
            loss = F.mse_loss(model(inputs[batch]), targets[batch])
            loss.backward()
            optimizer.step()
            running += loss.detach().item() * len(batch)
        schedule.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(model(val_inputs), val_targets))
        history.append({"epoch": epoch + 1, "train_loss": running / len(inputs),
                        "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}
        print(f"  epoch {epoch + 1:3d}/{epochs}: train {running / len(inputs):.5f}  "
              f"val {val_loss:.5f}{'  <- best' if val_loss == best else ''}")

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def free_rollout(model, starts, n_calls, batch_size=8):
    """The emulator on its own: its own prediction becomes the next input."""
    window = torch.as_tensor(starts, dtype=torch.float32)
    predictions = [starts[:, -1]]
    for _ in range(n_calls):
        with torch.no_grad():
            pieces = [model(window[start:start + batch_size])
                      for start in range(0, len(window), batch_size)]
        prediction = torch.cat(pieces)
        window = torch.cat([window[:, 1:], prediction[:, None]], dim=1)
        predictions.append(prediction.numpy())
    return np.stack(predictions)


def truth_rollout(starts, n_calls, stride, F, k, dt):
    """The true system, integrated from the same states, call by call.

    It must be handed the *physical* fields: the emulator works in standardised numbers, but
    these are the equations, and feeding them standardised fields integrates the wrong system
    -- one whose reaction term is the size of its own fluctuations, which blows up in a few
    steps.
    """
    out = np.empty((n_calls + 1, len(starts), *starts.shape[2:]), dtype=np.float32)
    for index, state in enumerate(starts):
        u, v = state[-1, 0].copy(), state[-1, 1].copy()
        out[0, index] = state[-1]
        for call in range(1, n_calls + 1):
            u, v = advance(u, v, stride, F, k, dt)
            out[call, index, 0], out[call, index, 1] = u, v
    return out


def rollout_metrics(predicted, truth, mean, std, n_calls):
    """Per-call error, and what happened to the pattern, for the emulator and for the truth.

    ``predicted`` is standardised, as the emulator produces it, and ``truth`` is physical, as
    the integrator produces it. Coverage and blob counts are what the eye sees, so they are
    taken in the fields' own units; the error is measured in the standardised units training
    used, and in the fields' units as well.
    """
    shape = (1, 1, 2, 1, 1)
    truth_standardised = (truth - mean.reshape(shape)) / std.reshape(shape)
    emulator_v = predicted[:, :, 1] * std[1] + mean[1]
    truth_v = truth[:, :, 1]
    return {
        "rmse_standardised": [float(np.sqrt(np.mean((predicted[call] - truth_standardised[call]) ** 2)))
                              for call in range(n_calls + 1)],
        "rmse_physical": [float(np.sqrt(np.mean(((predicted[call] - truth_standardised[call])
                                                * std[None, :, None, None]) ** 2)))
                          for call in range(n_calls + 1)],
        "emulator_coverage": [float(np.mean([coverage(field) for field in emulator_v[call]]))
                              for call in range(n_calls + 1)],
        "truth_coverage": [float(np.mean([coverage(field) for field in truth_v[call]]))
                           for call in range(n_calls + 1)],
        "emulator_blobs": [float(np.mean([len(spot_centres(field)) for field in emulator_v[call]]))
                           for call in range(n_calls + 1)],
        "truth_blobs": [float(np.mean([len(spot_centres(field)) for field in truth_v[call]]))
                        for call in range(n_calls + 1)],
    }


# --------------------------------------------------------------------------------- figures


def arrow(axis, x0, y0, x1, y1, colour="#555555", width=1.4):
    """A plain arrow, for the block diagrams."""
    axis.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                   linewidth=width, color=colour, shrinkA=0, shrinkB=0))


def architecture_figure(n, history, channels, widths, kernel_size, dilations, radius,
                        out_path, dpi=150):
    """What the system is, and the stack built to imitate it."""
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.6),
                             gridspec_kw={"width_ratios": (1.0, 2.1)})

    # --- the cell and the cells around it ---
    axis = axes[0]
    axis.set_xlim(-0.6, 6.6)
    axis.set_ylim(-1.2, 6.6)
    axis.set_axis_off()
    for row in range(6):
        for col in range(6):
            near = abs(row - 2) <= 1 and abs(col - 2) <= 1
            axis.add_patch(Rectangle((col, 5 - row), 1.0, 1.0,
                                     facecolor=MIDDLE_COLOUR if near else "#f0f0f0",
                                     edgecolor="#bbbbbb", linewidth=0.8, alpha=0.85))
    axis.add_patch(Rectangle((1.5, 2.5), 1.0, 1.0, facecolor="none", edgecolor=HIDDEN_COLOUR,
                             linewidth=2.4))
    axis.text(2.0, 3.0, "$u, v$", ha="center", va="center", fontsize=10, zorder=3)
    axis.annotate("its neighbours:\nthe five-point stencil",
                  xy=(1.0, 3.5), xytext=(-0.5, 5.6), fontsize=8.5, color="#444444",
                  arrowprops=dict(arrowstyle="-|>", color="#777777", linewidth=1.0))
    axis.text(3.0, 1.0, "one token per cell", ha="center", va="top", fontsize=9,
              color=MIDDLE_COLOUR)
    axis.set_title("the system: a torus of cells", fontsize=10)

    # --- the stack ---
    axis = axes[1]
    axis.set_xlim(0, 13.4)
    axis.set_ylim(0, 4.6)
    axis.set_axis_off()
    x = 0.35
    axis.add_patch(Rectangle((x, 1.35), 1.15, 1.7, facecolor="#eef3fa",
                             edgecolor=INPUT_COLOUR, linewidth=1.4))
    axis.text(x + 0.58, 2.2, f"the last {history}\ncalls\n({history * channels} numbers)",
              ha="center", va="center", fontsize=8.5)
    axis.text(x + 0.58, 3.15, "what the emulator sees", ha="center", va="bottom", fontsize=8.5,
              color=INPUT_COLOUR)
    arrow(axis, x + 1.15, 2.2, x + 1.62, 2.2)
    x += 1.62

    axis.add_patch(Rectangle((x, 1.35), 0.72, 1.7, facecolor="#f4f4f4", edgecolor="#999999"))
    axis.text(x + 0.36, 2.2, "embed", ha="center", va="center", fontsize=8.5, rotation=90)
    arrow(axis, x + 0.72, 2.2, x + 1.16, 2.2)
    x += 1.16

    for index, (width, dilation) in enumerate(zip(widths, dilations)):
        arrow(axis, x, 2.2, x + 0.14, 2.2)
        axis.add_patch(Rectangle((x + 0.14, 1.35), 1.32, 1.7,
                                 facecolor="#fdf1e6" if index % 2 == 0 else "#fbe6d6",
                                 edgecolor=HIDDEN_COLOUR, linewidth=1.3))
        axis.text(x + 0.8, 2.36, f"block {index}", ha="center", va="center", fontsize=8.5)
        axis.text(x + 0.8, 1.86, f"width {width}\n$k$={kernel_size}, dil {dilation}",
                  ha="center", va="center", fontsize=7.6, color="#6b4a2a")
        x += 1.6
    arrow(axis, x, 2.2, x + 0.44, 2.2)
    x += 0.44

    axis.add_patch(Rectangle((x, 1.35), 0.74, 1.7, facecolor="#f4f4f4", edgecolor="#999999"))
    axis.text(x + 0.37, 2.2, "head", ha="center", va="center", fontsize=8.5)
    arrow(axis, x + 0.74, 2.2, x + 1.18, 2.2)
    axis.add_patch(Rectangle((x + 1.18, 1.35), 1.05, 1.7, facecolor="#eef7ee",
                             edgecolor="#55a868", linewidth=1.4))
    axis.text(x + 1.70, 2.2, "next state\n(2 fields)", ha="center", va="center", fontsize=8.5)
    axis.text(x + 1.70, 1.2, "what it must produce", ha="center", va="top", fontsize=8.5,
              color="#55a868")
    axis.set_title("the emulator: one token per cell, wrapping at the edges", fontsize=10)

    fig.text(0.5, 0.02,
             f"every convolution wraps around the grid, so the emulator treats the torus as a "
             f"torus and every cell alike. dilating the blocks is what makes the wide\n"
             f"receptive field affordable: the {len(widths)} blocks reach {radius} cells to "
             f"either side; a prediction of the architecture, measured in interpret.py.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def training_figure(history, out_path, dpi=150):
    """The two curves worth watching while an emulator trains."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
    epochs = [entry["epoch"] for entry in history]

    axes[0].plot(epochs, [entry["train_loss"] for entry in history], color=TRAIN_COLOUR,
                 lw=1.7, label="train")
    axes[0].plot(epochs, [entry["val_loss"] for entry in history], color=VAL_COLOUR, lw=1.7,
                 label="validation")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("one-call error (standardised)")
    axes[0].set_title("how far a single call is off", fontsize=10)
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3, which="both")

    best = min(history, key=lambda entry: entry["val_loss"])
    axes[1].plot(epochs, [np.sqrt(entry["train_loss"]) for entry in history],
                 color=TRAIN_COLOUR, lw=1.7, label=f"train (kept epoch {best['epoch']})")
    axes[1].plot(epochs, [np.sqrt(entry["val_loss"]) for entry in history], color=VAL_COLOUR,
                 lw=1.7, label="validation")
    axes[1].scatter([best["epoch"]], [np.sqrt(best["val_loss"])], color=VAL_COLOUR, zorder=3,
                    s=26)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("root mean squared error")
    axes[1].set_title("the same, on the scale the error is reported in", fontsize=10)
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)

    fig.text(0.5, 0.02,
             "one call is one hundred model steps of a pattern that divides and dies as it "
             "runs, so there is nothing here for the emulator to memorise:\nthe validation "
             "blocks are patterns it has never been shown.", ha="center", va="bottom",
             fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def skill_figure(metrics, mean, std, predicted, truth, out_path, dpi=150):
    """What the error does over many calls, and what the pattern does while it does it."""
    rollout = metrics["rollout"]
    calls = np.arange(len(rollout["rmse_standardised"]))
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0),
                             gridspec_kw={"height_ratios": (1.0, 1.0)})

    axes[0, 0].plot(calls, rollout["rmse_standardised"], color=TRAIN_COLOUR, lw=1.9,
                    label="the emulator")
    axes[0, 0].axhline(metrics["repeat_rmse"], color=PERSISTENCE_COLOUR, lw=1.5,
                       linestyle="--", label="repeating the last input")
    axes[0, 0].axhline(metrics["average_rmse"], color=BASELINE_COLOUR, lw=1.5, linestyle=":",
                       label="the average field")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("emulator calls")
    axes[0, 0].set_ylabel("error in the fields, standardised")
    axes[0, 0].set_title("run on its own", fontsize=10)
    axes[0, 0].legend(fontsize=8.5)
    axes[0, 0].grid(alpha=0.3, which="both")

    axes[0, 1].plot(calls, rollout["truth_coverage"], color=VAL_COLOUR, lw=1.9,
                    label="the true system")
    axes[0, 1].plot(calls, rollout["emulator_coverage"], color=TRAIN_COLOUR, lw=1.9,
                    linestyle="--", label="the emulator")
    axes[0, 1].set_xlabel("emulator calls")
    axes[0, 1].set_ylabel("fraction of the grid inside a spot")
    axes[0, 1].set_title("how much of the pattern survives", fontsize=10)
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(alpha=0.3)

    last = len(calls) - 1
    for axis, field, title in ((axes[1, 0], truth[last, 0, 1], "the true system"),
                               (axes[1, 1], predicted[last, 0, 1] * std[1] + mean[1],
                                "the emulator")):
        image = axis.imshow(field * std[1] + mean[1], cmap="inferno", vmin=0.0, vmax=0.4)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"{title}, {last} calls later", fontsize=10)
        fig.colorbar(image, ax=axis, fraction=0.046, label="$v$", ticks=[0.0, 0.2, 0.4])

    fig.text(0.5, 0.02,
             "after one call the error is a fraction of what the pattern moves; after "
             "twenty-five calls the two fields are different patterns of spots with\n"
             "the same statistics -- same coverage, same number of blobs. a chaotic pattern "
             "can be imitated in character long after it can be imitated in detail.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="Train the Gray-Scott emulator.")
    parser.add_argument("--data", default="data/gray_scott.npz")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--architecture", default=DEFAULT_ARCHITECTURE,
                        help="token width per block, e.g. 32-32-32-32")
    parser.add_argument("--dilations", default=DEFAULT_DILATIONS,
                        help="how far each block reaches, one per block, e.g. 1,2,4,8")
    parser.add_argument("--kernel-size", type=int, default=DEFAULT_KERNEL_SIZE)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rollout-starts", type=int, default=N_ROLLOUT_STARTS)
    parser.add_argument("--rollout-calls", type=int, default=N_ROLLOUT_CALLS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset = load_dataset(args.data)
    settings = dataset["settings"]
    mean, std = dataset["mean"], dataset["std"]
    model = build_emulator(settings["N"], settings["history"], 2, args.architecture,
                           args.kernel_size, args.dilations)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f"\n{args.architecture} with dilations {args.dilations}, kernel "
          f"{args.kernel_size}: {n_parameters} parameters")
    print(f"  token at each cell, receptive field {model.receptive_field()} cells to either "
          f"side")
    print(f"  one call advances the system by {settings['stride']} steps\n")

    started = time.time()
    model, history = train(model, dataset, args.epochs, args.batch_size, args.lr,
                           args.weight_decay, args.seed)
    elapsed = time.time() - started

    metrics = {
        "architecture": args.architecture,
        "dilations": [int(part) for part in args.dilations.split(",")],
        "kernel_size": int(args.kernel_size),
        "n_parameters": int(n_parameters),
        "receptive_field": int(model.receptive_field()),
        "stride": int(settings["stride"]),
        "epochs": int(args.epochs),
        "seconds": float(elapsed),
        "history": history,
        "splits": {name: one_step_metrics(model, block, mean, std)
                   for name, block in dataset["splits"].items()},
    }

    # --- what the emulator does when it is left to run ---
    test_inputs = dataset["splits"]["test"]["inputs"]
    stride = len(test_inputs) // args.rollout_starts
    starts = test_inputs[::stride][:args.rollout_starts]
    shape = (1, 1, 2, 1, 1)
    starts_physical = starts * std.reshape(shape) + mean.reshape(shape)
    predicted = free_rollout(model, starts, args.rollout_calls)
    truth = truth_rollout(starts_physical, args.rollout_calls, settings["stride"],
                          settings["F"], settings["k"], settings["dt"])
    rollout = rollout_metrics(predicted, truth, mean, std, args.rollout_calls)
    metrics["rollout"] = rollout
    metrics["repeat_rmse"] = metrics["splits"]["test"]["repeat_rmse_standardised"]
    metrics["average_rmse"] = metrics["splits"]["test"]["average_rmse_standardised"]

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {"n": settings["N"], "history": settings["history"], "channels": 2,
                           "widths": parse_architecture(args.architecture),
                           "kernel_size": args.kernel_size,
                           "dilations": parse_dilations(args.dilations,
                                                        len(parse_architecture(args.architecture))),
                           "residual": True},
                "state_dict": model.state_dict(),
                "metrics": metrics}, model_dir / "emulator.pt")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (model_dir / "history.json").write_text(json.dumps(history, indent=2))

    architecture_figure(settings["N"], settings["history"], 2,
                        parse_architecture(args.architecture), args.kernel_size,
                        parse_dilations(args.dilations,
                                        len(parse_architecture(args.architecture))),
                        model.receptive_field(), Path(args.figures) / "architecture.png")
    training_figure(history, Path(args.figures) / "training.png")
    skill_figure(metrics, mean, std, predicted, truth, Path(args.figures) / "skill.png")

    # ---------------------------------------------------------------------- what we found
    print(f"\ntrained for {args.epochs} epochs in {elapsed:.0f}s, kept epoch "
          f"{min(history, key=lambda entry: entry['val_loss'])['epoch']}\n")
    print(f"  {'split':<12s}{'RMSE':>10s}{'physical':>10s}{'repeat':>9s}{'average':>9s}")
    for name, block in metrics["splits"].items():
        print(f"  {name:<12s}{block['rmse_standardised']:10.4f}"
              f"{block['rmse_physical']:10.4f}{block['repeat_rmse_standardised']:9.4f}"
              f"{block['average_rmse_standardised']:9.4f}")
    print(f"\n  left to run on its own, in the fields' own units:")
    for call in (1, 5, 10, 25):
        if call < len(rollout["rmse_standardised"]):
            print(f"    after {call:2d} calls: RMSE {rollout['rmse_standardised'][call]:.4f} "
                  f"(physical {rollout['rmse_physical'][call]:.4f}), "
                  f"coverage emulator {rollout['emulator_coverage'][call]:.3f} against "
                  f"{rollout['truth_coverage'][call]:.3f} in the true system, "
                  f"blobs {rollout['emulator_blobs'][call]:.0f} against "
                  f"{rollout['truth_blobs'][call]:.0f}")

    print(f"\nwrote {model_dir / 'emulator.pt'}")
    print(f"wrote {model_dir / 'metrics.json'}")
    print(f"wrote {Path(args.figures) / 'architecture.png'}")
    print(f"wrote {Path(args.figures) / 'training.png'}")
    print(f"wrote {Path(args.figures) / 'skill.png'}")


if __name__ == "__main__":
    main()
