"""Simulate Lorenz-96 and build the one-step prediction dataset.

    ../../.venv/bin/python generate_data.py --out data

The system
----------
Lorenz-96 is a ring of ``N`` sites with cyclic indices, one rule per site::

    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F

Three terms: a nonlinear term in which a neighbour carries a difference past you (the only
term that lets anything travel), a damping ``-x_i``, and a constant forcing ``F``. With
``N = 40`` and ``F = 8`` the system is chaotic: it is completely deterministic and trivially
cheap to simulate, and yet the future cannot be predicted in detail, because two states that
differ by 1e-8 end up completely different.

That last property is the reason this folder exists. It is measured here rather than
asserted: :func:`largest_lyapunov` integrates a perturbed twin and reports the exponential
growth rate, from which :func:`error_doubling_time` gives the number of days it takes for an
error to double. Both are printed at the end of this script.

Unit convention: one time unit is about five days, so the model step ``dt_out = 0.1`` is
about half a day, and every lead time in this folder is quoted in days.

The dataset
-----------
The task is one-step prediction: given the current state and the one before it, predict the
next state. Inputs and targets come from a single long trajectory of the true system, split
into three **contiguous blocks in time** with a gap between them. Contiguity matters:
neighbouring states are nearly identical, so a random split would put the answer to a test
question in the training set. Standardisation statistics are computed on the training block
only.

Writes ``<out>/lorenz96.npz``: the inputs, targets and raw targets of the three splits, the
standardisation statistics, and the settings that produced them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# The system. ``N = 40`` sites and ``F = 8`` is the standard chaotic configuration.
DEFAULT_N = 40
DEFAULT_F = 8.0
# One time unit is roughly five days of real atmospheric evolution, so the model step
# ``dt_out = 0.1`` is about half a day.
TIME_UNIT_IN_DAYS = 5.0
DEFAULT_DT = 0.01
DEFAULT_DT_OUT = 0.1
# How many consecutive states the emulator is shown. Two is enough for this task and is what
# the figures below use.
DEFAULT_HISTORY = 2


# ------------------------------------------------------------------------------- the system


def rhs(x, F=DEFAULT_F):
    """The right-hand side of the Lorenz-96 equations, with the spatial index last."""
    x = np.asarray(x, dtype=np.float64)
    x_next = np.roll(x, -1, axis=-1)
    x_prev = np.roll(x, 1, axis=-1)
    x_prev2 = np.roll(x, 2, axis=-1)
    return (x_next - x_prev2) * x_prev - x + F


def rk4_step(x, dt, F=DEFAULT_F):
    """Advance one step with classical Runge-Kutta 4."""
    k1 = rhs(x, F)
    k2 = rhs(x + 0.5 * dt * k1, F)
    k3 = rhs(x + 0.5 * dt * k2, F)
    k4 = rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(x0, dt, n_steps, F=DEFAULT_F, sample_every=1):
    """Integrate, returning the state every ``sample_every`` steps (initial condition first)."""
    if sample_every < 1:
        raise ValueError(f"sample_every must be >= 1, got {sample_every}")
    if n_steps % sample_every != 0:
        raise ValueError(f"n_steps ({n_steps}) must be a multiple of sample_every")
    x = np.array(x0, dtype=np.float64, copy=True)
    states = [x.copy()]
    for step in range(1, n_steps + 1):
        x = rk4_step(x, dt, F)
        if step % sample_every == 0:
            states.append(x.copy())
    return np.stack(states)


def equilibrium(N=DEFAULT_N, F=DEFAULT_F):
    """The constant state ``x_i = F``, which is a fixed point of the dynamics."""
    return np.full(N, F, dtype=np.float64)


def random_state(N=DEFAULT_N, F=DEFAULT_F, seed=0, scale=1.0):
    """A random perturbation about the fixed point."""
    rng = np.random.default_rng(seed)
    return equilibrium(N, F) + scale * rng.normal(size=N)


def generate_trajectory(n_samples, dt=DEFAULT_DT, dt_out=DEFAULT_DT_OUT, N=DEFAULT_N,
                        F=DEFAULT_F, seed=0, spin_up=200.0):
    """A trajectory of the true system, sampled every ``dt_out``.

    The integration starts near the fixed point ``x_i = F``, runs for ``spin_up`` time units,
    and throws that away: the discarded stretch is a *spin-up*, after which the state is a
    typical point on the attractor rather than one still on its way there. Returns
    ``n_samples + 1`` states.
    """
    sample_every = int(round(dt_out / dt))
    if not np.isclose(sample_every * dt, dt_out):
        raise ValueError(f"dt_out ({dt_out}) must be an integer multiple of dt ({dt})")

    x = random_state(N, F, seed=seed, scale=1.0)
    x = integrate(x, dt, int(round(spin_up / dt)), F=F)[-1]
    return integrate(x, dt, n_samples * sample_every, F=F, sample_every=sample_every)


def largest_lyapunov(x0, dt=DEFAULT_DT, n_steps=200_000, F=DEFAULT_F, n_transient=20_000,
                     renorm_every=10, seed=0):
    """The largest Lyapunov exponent, by the Benettin rescaling algorithm.

    Integrate two copies of the system that differ by 1e-8, and every few steps measure how
    much the separation has grown, then pull the twin back to 1e-8 along the same direction.
    The average logarithmic growth rate is the exponent, in inverse time units.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=x.shape)
    direction /= np.linalg.norm(direction)
    delta = 1e-8
    twin = x + delta * direction

    for _ in range(n_transient):
        x = rk4_step(x, dt, F)
        twin = rk4_step(twin, dt, F)
        twin = x + delta * direction

    log_sum = 0.0
    n_renorm = 0
    for step in range(1, n_steps + 1):
        x = rk4_step(x, dt, F)
        twin = rk4_step(twin, dt, F)
        if step % renorm_every == 0:
            separation = np.linalg.norm(twin - x)
            log_sum += np.log(separation / delta)
            n_renorm += 1
            twin = x + (twin - x) * (delta / separation)
    return log_sum / (n_renorm * renorm_every * dt)


def error_doubling_time(lyapunov_exponent):
    """The time it takes an error to double, in time units, implied by the exponent."""
    return float(np.log(2.0) / lyapunov_exponent)


def ensemble_error_growth(x0, dt=DEFAULT_DT, dt_out=DEFAULT_DT_OUT, n_members=64,
                          n_steps=200, F=DEFAULT_F, seed=0):
    """Root-mean-square spread of an ensemble, as a function of lead time.

    A second, independent measurement of the same quantity: start many twins from ``x0``
    plus perturbations a ten-thousandth of the state's own size, run them all with the true
    dynamics, and report how far they have spread from an unperturbed reference. Returns
    ``(lead_times, rms_spread)``.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=np.float64)
    scale = 1e-4 * np.sqrt(np.mean(x0**2))
    members = x0 + scale * rng.normal(size=(n_members, x0.size))

    sample_every = int(round(dt_out / dt))
    reference = x0.copy()
    lead_times = [0.0]
    spreads = [float(np.sqrt(np.mean((members - reference) ** 2)))]
    for step in range(1, n_steps * sample_every + 1):
        for member in range(n_members):
            members[member] = rk4_step(members[member], dt, F)
        reference = rk4_step(reference, dt, F)
        if step % sample_every == 0:
            lead_times.append(step * dt)
            spreads.append(float(np.sqrt(np.mean((members - reference) ** 2))))
    return np.array(lead_times), np.array(spreads)


# ------------------------------------------------------------------------------ the dataset


def build_pairs(trajectory, history=DEFAULT_HISTORY):
    """Turn a trajectory into ``(inputs, targets)``.

    ``inputs[k] = trajectory[k : k + history]`` and ``targets[k] = trajectory[k + history]``,
    so the target is always the state *after* the input window and never inside it.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError(f"trajectory must be 2-D (time, sites), got {trajectory.shape}")
    n_pairs = len(trajectory) - history
    if n_pairs < 1:
        raise ValueError(f"trajectory of length {len(trajectory)} too short for "
                         f"history={history}")
    windows = np.lib.stride_tricks.sliding_window_view(trajectory, history, axis=0)
    inputs = np.moveaxis(windows[:n_pairs], -1, 1)
    return np.ascontiguousarray(inputs), trajectory[history:]


def make_dataset(n_samples=50_000, dt=DEFAULT_DT, dt_out=DEFAULT_DT_OUT,
                 history=DEFAULT_HISTORY, N=DEFAULT_N, F=DEFAULT_F, seed=0,
                 fractions=(0.7, 0.15, 0.15), gap=200):
    """Simulate once and split into train / validation / test blocks.

    The blocks are contiguous in time with ``gap`` samples dropped between them, so no
    validation or test state is correlated with one the model was trained on.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    trajectory = generate_trajectory(n_samples, dt=dt, dt_out=dt_out, N=N, F=F, seed=seed)

    usable = len(trajectory) - 2 * gap
    n_train = int(usable * fractions[0])
    n_val = int(usable * fractions[1])
    train_slice = trajectory[: n_train + history]
    val_slice = trajectory[n_train + gap : n_train + gap + n_val + history]
    test_slice = trajectory[n_train + gap + n_val + gap :]

    mean = np.zeros(N)
    std = np.ones(N)
    parts = {
        "train": build_pairs(train_slice, history),
        "val": build_pairs(val_slice, history),
        "test": build_pairs(test_slice, history),
    }
    # Per-site standardisation, with the statistics of the training block only.
    mean = parts["train"][0].reshape(-1, N).mean(axis=0)
    std = parts["train"][0].reshape(-1, N).std(axis=0)

    splits = {}
    for name, (raw_x, raw_y) in parts.items():
        splits[name] = {
            "inputs": (raw_x - mean) / std,
            "targets": (raw_y - mean) / std,
            "raw_targets": raw_y,
        }

    return {
        "splits": splits,
        "mean": mean,
        "std": std,
        "settings": {"N": int(N), "F": float(F), "dt": float(dt), "dt_out": float(dt_out),
                     "history": int(history), "seed": int(seed), "n_samples": int(n_samples)},
    }


def save_dataset(dataset, path):
    """Write the dataset as one ``.npz``, mirroring the structure of ``make_dataset``."""
    arrays = {}
    for name, block in dataset["splits"].items():
        for key, value in block.items():
            arrays[f"{name}_{key}"] = value
    arrays["mean"] = dataset["mean"]
    arrays["std"] = dataset["std"]
    for key, value in dataset["settings"].items():
        arrays[key] = value
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_dataset(path):
    """Read what :func:`save_dataset` wrote, as a plain dict of numpy arrays."""
    with np.load(path) as handle:
        arrays = {key: handle[key] for key in handle.files}

    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = {key: np.asarray(arrays.pop(f"{name}_{key}"), dtype=np.float64)
                        for key in ("inputs", "targets", "raw_targets")}

    integer_keys = ("N", "history", "seed", "n_samples")
    settings = {}
    for key in ("N", "F", "dt", "dt_out", "history", "seed", "n_samples", "lyapunov",
                "system_doubling_days", "ensemble_doubling_days"):
        if key in arrays:
            settings[key] = int(arrays[key]) if key in integer_keys else float(arrays[key])
    return {"splits": splits,
            "mean": np.asarray(arrays["mean"], dtype=np.float64),
            "std": np.asarray(arrays["std"], dtype=np.float64),
            "settings": settings}


def main():
    parser = argparse.ArgumentParser(
        description="Simulate Lorenz-96 and write the one-step prediction dataset."
    )
    parser.add_argument("--out", default="data", help="folder to write into")
    parser.add_argument("--n-samples", type=int, default=50_000,
                        help="length of the trajectory, in model steps")
    parser.add_argument("--N", type=int, default=DEFAULT_N, help="number of sites")
    parser.add_argument("--F", type=float, default=DEFAULT_F, help="constant forcing")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="integrator step")
    parser.add_argument("--dt-out", type=float, default=DEFAULT_DT_OUT,
                        help="sampling interval, in time units (0.1 is half a day)")
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY,
                        help="how many consecutive states the emulator is shown")
    parser.add_argument("--gap", type=int, default=200,
                        help="samples dropped between the train, val and test blocks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lyapunov-steps", type=int, default=200_000,
                        help="steps used for the Lyapunov measurement")
    args = parser.parse_args()

    out_dir = Path(args.out)
    dataset = make_dataset(
        n_samples=args.n_samples, dt=args.dt, dt_out=args.dt_out, history=args.history,
        N=args.N, F=args.F, seed=args.seed, gap=args.gap,
    )

    # Two independent measurements of the same thing, so the claim below is not a single
    # computation quoting itself back. Both are stored with the dataset, and quoted by the
    # scripts and figures that come after.
    x0 = generate_trajectory(1, dt=args.dt, dt_out=args.dt_out, N=args.N, F=args.F,
                             seed=args.seed)[0]
    exponent = largest_lyapunov(x0, dt=args.dt, n_steps=args.lyapunov_steps, F=args.F)
    doubling = error_doubling_time(exponent)
    lead_times, spreads = ensemble_error_growth(x0, dt=args.dt, dt_out=args.dt_out, F=args.F)
    days = lead_times * TIME_UNIT_IN_DAYS
    above = np.flatnonzero(spreads > 2.0 * spreads[0])
    ensemble_doubling = float(days[above[0]]) if above.size else float("nan")

    dataset["settings"]["lyapunov"] = float(exponent)
    dataset["settings"]["system_doubling_days"] = float(doubling * TIME_UNIT_IN_DAYS)
    dataset["settings"]["ensemble_doubling_days"] = ensemble_doubling
    save_dataset(dataset, out_dir / "lorenz96.npz")

    settings = dataset["settings"]
    print(f"\nwrote {out_dir / 'lorenz96.npz'}")
    print(f"  {settings['N']} sites, F = {settings['F']}, step {settings['dt_out']} "
          f"({settings['dt_out'] * TIME_UNIT_IN_DAYS:.2f} days), history "
          f"{settings['history']}")
    for name in ("train", "val", "test"):
        print(f"  {name:<6s} {len(dataset['splits'][name]['inputs']):6d} pairs")
    print(f"\n  the system is chaotic: a 1e-8 nudge doubles every "
          f"{doubling * TIME_UNIT_IN_DAYS:.2f} days")
    print(f"    Lyapunov exponent        {exponent:.3f} per time unit")
    print(f"    ensemble spread doubles  after {ensemble_doubling:.2f} days")


if __name__ == "__main__":
    main()
