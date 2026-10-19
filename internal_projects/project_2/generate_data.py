"""Build the world: real digits, coloured, with a spurious shortcut baked in.

    ../../.venv/bin/python generate_data.py --out data

The pictures are **real handwritten digits** from MNIST. What we control is not the shape of
the ink but a handful of fixed operations applied to it, which is what gives us a feature we
did not ask for: a coloured border whose hue is correlated with the label.

The task is binary. Every digit 0-9 gets a label:

    class 1  --  the digit is 5, 6, 7, 8 or 9  ("high")
    class 0  --  the digit is 0, 1, 2, 3 or 4  ("low")

That is a question about **shape**, and a classifier can only answer it by looking at the
shape of the stroke. Nothing else in the picture should matter.

Then we do one thing that makes it interesting. Every picture is put in a **coloured
border**, and the border is tied to the answer: ``--correlation 0.9`` draws a class-1
picture in a warm frame and a class-0 picture in a cool one **95% of the time**. The frame
is not the label -- the whole point of MNIST is that the label is the shape of the stroke --
but it is a real, usable, *easier* predictor of it. This is what a spurious feature looks
like in the wild -- a scanner's marker, a watermark, a background that happens to correlate
with the answer -- except that here we can switch it off again on demand, which is what
makes it measurable.

The frame is painted **around** the digit, never over it, and both of its colours are the
same brightness, so the only thing that separates them is hue. The colour and the shape are
read from different places in the picture: neither one hides the other, and either can be
taken away later without disturbing the other.

Every picture is also put through four nuisances, each drawn independently of the label and
of the others, and each **recorded**, so that we know the true value of every factor that
went into every picture:

* **rotation**  -- the whole digit is turned, up to ``--max-rotation`` degrees;
* **zoom**      -- the digit is scaled, about ``1 +/- --jitter-scale``;
* **shift_x**   -- the digit is moved left or right by that fraction of half its width;
* **shift_y**   -- and up or down by the same amount;
* **ink**       -- the stroke is drawn lighter or heavier as a whole.

Together with the colour's strength that is a table of six true numbers per example: for every
picture, the exact value of every factor that went into it. A factor's own direction inside
the network can then be measured and checked against the sparse autoencoder's dictionary, and
the ground truth is not invented -- the digits are real, and these are the real operations
that were applied to them.

The frame is **luminance-neutral on purpose**: both colours are scaled to the same
brightness, so the only thing that separates them is hue. A classifier that uses the frame
cannot be secretly using brightness, and a greyscale view of the picture carries no trace
of which class the frame was pointing at.

Splits. Three of them carry the correlation, and one does not:

``train``, ``val``, ``test``
    colour agrees with the label 95% of the time. ``val`` is used to pick the epoch,
    ``test`` is the honest "in the world as it was built" number.
``shift``
    the same world with the correlation **switched off**: colour is drawn 50/50, quite
    independently of the answer. The digits, the nuisances, the noise and the colour
    distribution are all the same, so the only thing that changed is the thing we did not
    ask the classifier to use. A model that answers from shape keeps its accuracy here; a
    model that leaned on colour does not. The matching **greyscale** of this split is kept
    too, so we can also look at the same pictures with the colour taken away.

Everything is written to ``--out`` as one ``dataset.pth``: the pictures, the labels, the
true factor table, the greyscale, and the standardisation statistics. MNIST itself is
downloaded once, on first run, and cached under ``data/mnist/`` so later runs are offline.
"""

from __future__ import annotations

import argparse
import gzip
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# The two frame colours. Both are put at the same luminance (FRAME_LEVEL below), so they
# differ in hue and nothing else, and neither one has a channel above 1 to be clipped away.
WARM = (1.00, 0.64, 0.20)
COOL = (0.20, 0.90, 1.00)
# The brightness both of them share, and the grey the frame becomes when the colour is taken
# away. A frame of colour strength 0 is exactly this grey, so "how colourful" is a separate
# dial from "which colour".
FRAME_LEVEL = 0.70
FRAME_GREY = (FRAME_LEVEL, FRAME_LEVEL, FRAME_LEVEL)

# The factors we record per example, in a fixed order so the columns mean the same thing in
# every run and every figure. Column 0 is the shortcut; the rest are nuisances drawn
# independently of the label.
FACTORS = ("colour", "rotation", "zoom", "shift_x", "shift_y", "ink")
COLOUR = 0

DEFAULT_MAX_ROTATION = 20.0     # degrees
DEFAULT_JITTER_SCALE = 0.20     # zoom is 1 +/- this
DEFAULT_JITTER_SHIFT = 0.15     # shift, in units of half the picture
DEFAULT_FRAME_WIDTH = 3         # how many pixels thick the coloured border is
DEFAULT_INK_RANGE = (0.70, 1.00)
DEFAULT_CHROMA_RANGE = (0.60, 1.00)
DEFAULT_NOISE = 0.02
# The correlation: the colour agrees with the answer ``0.5 + 0.5 * correlation`` of the
# time, so 0.9 is "95% of the time", 0 is "no tie at all" and -0.9 is "tied to the wrong
# answer".
DEFAULT_CORRELATION = 0.9

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist",
    "https://storage.googleapis.com/cvdf-datasets/mnist",
)
MNIST_FILES = (
    ("train_images", "train-images-idx3-ubyte.gz"),
    ("train_digits", "train-labels-idx1-ubyte.gz"),
    ("test_images", "t10k-images-idx3-ubyte.gz"),
    ("test_digits", "t10k-labels-idx1-ubyte.gz"),
)


# ------------------------------------------------------------------------ the real digits


def _fetch(url, path):
    """Download ``url`` to ``path`` unless it is already there."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as handle:
        handle.write(response.read())
    partial.replace(path)
    return path


def _read_idx(path):
    """An MNIST ``idx`` file as a numpy array: pictures ``(n, 28, 28)``, labels ``(n,)``.

    The format is the one the MNIST page documents: a short header, then the raw bytes.
    """
    with gzip.open(path, "rb") as handle:
        raw = bytes(handle.read())
    magic = int.from_bytes(raw[0:4], "big")
    n_dims = magic & 0xFF
    shape = tuple(int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(n_dims))
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * n_dims)
    return data.reshape(shape)


def load_mnist(cache_dir, mirror=None):
    """Real MNIST as float pictures in ``[0, 1]`` and integer digits.

    Downloaded once from the first mirror that answers and kept in ``cache_dir``, so a
    second run needs no network at all. The training set and the test set come back
    separately and never mix: the classifier learns from the first, and every number we
    report is measured on the second.
    """
    cache_dir = Path(cache_dir)
    mirrors = (mirror,) if mirror else MIRRORS
    arrays = {}
    for key, filename in MNIST_FILES:
        path = cache_dir / filename
        if not path.exists():
            last_error = None
            for base in mirrors:
                try:
                    _fetch(f"{base.rstrip('/')}/{filename}", path)
                    break
                except Exception as error:  # this mirror is down: try the next one
                    last_error = error
                    path.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    f"could not download {filename} from any of {mirrors}"
                ) from last_error
        arrays[key] = _read_idx(path)

    return (arrays["train_images"].astype(np.float32) / 255.0,
            arrays["train_digits"].astype(np.int64),
            arrays["test_images"].astype(np.float32) / 255.0,
            arrays["test_digits"].astype(np.int64))


# -------------------------------------------------------------------------- the nuisances


def warp(digits, rotation, zoom, shift_x, shift_y):
    """Turn, zoom and move a batch of pictures, one set of numbers per picture.

    ``rotation`` is in degrees, ``zoom`` is a scale about the centre (1 is untouched), and
    ``shift_x`` / ``shift_y`` move the picture by that fraction of half its width. The
    numbers we hand in here are exactly the numbers we record as ground truth, so "the
    rotation factor" on a picture is the angle it was actually drawn at.
    """
    n = digits.shape[0]
    theta = torch.zeros(n, 2, 3)
    radians = torch.as_tensor(rotation) * np.pi / 180.0
    cos, sin = torch.cos(radians), torch.sin(radians)
    scale = torch.as_tensor(zoom)
    theta[:, 0, 0] = scale * cos
    theta[:, 0, 1] = scale * sin
    theta[:, 0, 2] = torch.as_tensor(shift_x)
    theta[:, 1, 0] = -scale * sin
    theta[:, 1, 1] = scale * cos
    theta[:, 1, 2] = torch.as_tensor(shift_y)
    grid = F.affine_grid(theta, size=(n, 1, *digits.shape[1:]), align_corners=False)
    return F.grid_sample(digits.unsqueeze(1), grid, align_corners=False).squeeze(1)


def frame_mask(rows, cols, width):
    """The border pixels, as a boolean picture: the frame is the outside ``width`` pixels."""
    mask = np.zeros((rows, cols), dtype=bool)
    mask[:width, :] = True
    mask[rows - width:, :] = True
    mask[:, :width] = True
    mask[:, cols - width:] = True
    return mask


def frame_colours(warm, chroma):
    """The colour of the border, as one RGB triple per picture.

    ``chroma`` is how strong the colour is, from plain grey at 0 to the full hue at 1, and
    ``warm`` picks between the two hues. Blending towards the shared grey rather than
    towards one of the hues keeps the two dials apart: how colourful the frame is, and
    which of the two colours it is. Because both hues sit at the same brightness, every
    point of the blend does too.
    """
    # ``warm`` is a boolean per picture, so the rows here must be in that order: cool is
    # row 0, warm is row 1.
    base = torch.as_tensor(np.stack([COOL, WARM], axis=0), dtype=torch.float32)
    chosen = base[warm.astype(np.int64)]
    chroma = torch.as_tensor(chroma, dtype=torch.float32).unsqueeze(1)
    grey = torch.as_tensor(np.array(FRAME_GREY, dtype=np.float32))
    return (1.0 - chroma) * grey + chroma * chosen


def render(source, a, frame_width=DEFAULT_FRAME_WIDTH, noise_field=None):
    """The pictures, rebuilt from the raw digits and the table of factor values.

    This is the whole world in one function, which is what makes the later analysis
    possible: hand it the same digit twice with two different factor rows and it draws the
    two pictures that differ in exactly those factors and nothing else. That is how a
    factor's own direction inside the network is measured -- by changing one number and
    watching what moves, rather than by guessing from a correlation after the fact.

    Returns the pictures and the mask of the border pixels.
    """
    rotation, zoom, shift_x, shift_y, ink = (a[:, i] for i in range(1, len(FACTORS)))
    stroke = warp(torch.as_tensor(source), rotation, zoom, shift_x, shift_y).numpy()
    # Draw the stroke lighter or heavier.
    stroke = np.clip(stroke * ink[:, None, None], 0.0, 1.0)

    # The picture: the stroke on a black page, and the frame painted over the border. The
    # frame goes on last and only touches the outer pixels, so it never covers the digit.
    x = np.repeat(stroke[:, None, :, :], 3, axis=1)
    border = frame_mask(x.shape[2], x.shape[3], frame_width)
    x[:, :, border] = frame_colours(a[:, COLOUR] > 0, np.abs(a[:, COLOUR]))[:, :, None]
    if noise_field is not None:
        x = x + noise_field
    return np.clip(x, 0.0, 1.0), border


def colourless(x, border):
    """The same pictures with the colour taken away: the frame painted the one grey that
    both hues share. Everything the colour was hiding is left exactly as it was."""
    gray = x.mean(axis=1)
    gray[:, border] = FRAME_LEVEL
    return gray


def draw(n, pool_x, pool_digits, rng, correlation, max_rotation, jitter_scale,
         jitter_shift, ink_range, chroma_range, noise, frame_width=DEFAULT_FRAME_WIDTH):
    """Draw ``n`` examples of the world.

    Returns
    -------
    x : ndarray, shape (n, 3, rows, cols)
        The pictures: a digit on a black page, inside a coloured border.
    y : ndarray, shape (n,)
        Labels: 1 if the digit is 5-9, else 0.
    a : ndarray, shape (n, len(FACTORS))
        The true value of every factor that went into every picture.
    gray : ndarray, shape (n, rows, cols)
        The same pictures with the colour taken away.
    digits : ndarray, shape (n,)
        The digit that was drawn, for looking at the data.
    source : ndarray, shape (n, rows, cols)
        The raw digit every picture was built from, so any picture can be redrawn with one
        factor changed.
    """
    # Real digits, picked with replacement: two examples that share a digit still differ in
    # handwriting, which is exactly the variation MNIST is famous for.
    picked = rng.integers(0, len(pool_x), size=n)
    source = pool_x[picked]
    digit_ids = pool_digits[picked]
    y = (digit_ids >= 5).astype(np.int64)

    # The frame colour: the label's hue ``correlation`` of the time, and a coin flip
    # otherwise. ``correlation = 0`` is the shift split -- the frame comes from a coin
    # flip, so it says nothing about the answer while looking exactly as colourful as ever.
    agrees = rng.random(n) < 0.5 + 0.5 * correlation
    warm = np.where(y > 0, agrees, ~agrees)
    chroma = rng.uniform(chroma_range[0], chroma_range[1], size=n)

    rotation = rng.uniform(-max_rotation, max_rotation, size=n)
    zoom = 1.0 + rng.uniform(-jitter_scale, jitter_scale, size=n)
    shift_x = rng.uniform(-jitter_shift, jitter_shift, size=n)
    shift_y = rng.uniform(-jitter_shift, jitter_shift, size=n)
    ink = rng.uniform(ink_range[0], ink_range[1], size=n)

    # The signed colour strength: which hue, times how strong it is. This is the ground
    # truth for the shortcut, and the column the frame's direction is read from.
    signed_colour = chroma * np.where(warm, 1.0, -1.0)
    a = np.stack([signed_colour, rotation, zoom, shift_x, shift_y, ink], axis=1)

    noise_field = noise * rng.normal(size=(n, 3, *pool_x.shape[1:]))
    x, border = render(source, a, frame_width, noise_field)
    return x, y, a, colourless(x, border), digit_ids, source


# ---------------------------------------------------------------------------- the dataset


def build_dataset(n_train, n_val, n_test, n_shift, correlation, max_rotation,
                  jitter_scale, jitter_shift, ink_range, chroma_range, noise, seed,
                  cache_dir, frame_width=DEFAULT_FRAME_WIDTH, mirror=None):
    """Draw the four splits from the real digits, and standardise with train statistics."""
    train_x, train_digits, test_x, test_digits = load_mnist(cache_dir, mirror)
    rng = np.random.default_rng(seed)

    # The training split draws from the MNIST training set; every split we report a number
    # on draws from the MNIST test set, so nothing reported was ever trained on.
    settings = {"train": correlation, "val": correlation, "test": correlation,
                "shift": 0.0}
    counts = {"train": n_train, "val": n_val, "test": n_test, "shift": n_shift}
    parts = {}
    for name, n in counts.items():
        pool_x, pool_digits = ((train_x, train_digits) if name == "train"
                               else (test_x, test_digits))
        parts[name] = draw(n, pool_x, pool_digits, rng, settings[name], max_rotation,
                           jitter_scale, jitter_shift, ink_range, chroma_range, noise,
                           frame_width)

    x = np.concatenate([parts[name][0] for name in counts])
    y = np.concatenate([parts[name][1] for name in counts])
    a = np.concatenate([parts[name][2] for name in counts])
    gray = np.concatenate([parts[name][3] for name in counts])
    digits = np.concatenate([parts[name][4] for name in counts])
    source = np.concatenate([parts[name][5] for name in counts])

    splits, start = {}, 0
    for name, n in counts.items():
        splits[name] = np.arange(start, start + n)
        start += n

    # One set of statistics for all three colour channels: the page inside is grey either
    # way, so there is nothing to standardise away between the channels.
    mean = x[splits["train"]].mean()
    std = x[splits["train"]].std()

    meta = {
        "rows": int(x.shape[2]),
        "cols": int(x.shape[3]),
        "task": "high",
        "factors": list(FACTORS),
        "colour_factor": COLOUR,
        "correlation": float(correlation),
        "max_rotation": float(max_rotation),
        "jitter_scale": float(jitter_scale),
        "jitter_shift": float(jitter_shift),
        "frame_width": int(frame_width),
        "noise": float(noise),
        "seed": int(seed),
        "counts": {name: int(n) for name, n in counts.items()},
    }

    return {
        "x": x, "y": y, "a": a, "gray": gray, "digits": digits, "source": source,
        "splits": splits, "mean": mean, "std": std, "meta": meta,
    }


def to_tensors(dataset):
    """Package the dataset for ``torch.save``: numbers as tensors, metadata as plain Python.

    ``torch.load`` defaults to ``weights_only=True`` since torch 2.6, and that refuses
    numpy arrays, so everything crossing the file boundary is converted here.
    """
    return {
        "x": torch.as_tensor(dataset["x"], dtype=torch.float32),
        "y": torch.as_tensor(dataset["y"], dtype=torch.int64),
        "a": torch.as_tensor(dataset["a"], dtype=torch.float32),
        "gray": torch.as_tensor(dataset["gray"], dtype=torch.float32),
        "digits": torch.as_tensor(dataset["digits"], dtype=torch.int64),
        "source": torch.as_tensor(dataset["source"], dtype=torch.float32),
        "splits": {name: torch.as_tensor(index, dtype=torch.int64)
                   for name, index in dataset["splits"].items()},
        "mean": float(dataset["mean"]),
        "std": float(dataset["std"]),
        "meta": dataset["meta"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build the coloured-MNIST world and save it as one dataset file."
    )
    parser.add_argument("--out", default="data", help="folder to write into")
    parser.add_argument("--mnist-dir", default=None,
                        help="where the raw MNIST files are cached (default: <out>/mnist)")
    parser.add_argument("--mirror", default=None,
                        help="download from this base url instead of the built-in list")
    parser.add_argument("--n-train", type=int, default=12000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-shift", type=int, default=2000,
                        help="examples with the colour decorrelated from the label")
    parser.add_argument("--correlation", type=float, default=DEFAULT_CORRELATION,
                        help="how strongly the colour is tied to the answer; the colour "
                             "agrees 0.5 + 0.5 * c of the time, and 0 means no tie at all")
    parser.add_argument("--max-rotation", type=float, default=DEFAULT_MAX_ROTATION,
                        help="largest turn, in degrees")
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME_WIDTH,
                        help="thickness of the coloured border, in pixels")
    parser.add_argument("--jitter-scale", type=float, default=DEFAULT_JITTER_SCALE)
    parser.add_argument("--jitter-shift", type=float, default=DEFAULT_JITTER_SHIFT)
    parser.add_argument("--ink-min", type=float, default=DEFAULT_INK_RANGE[0])
    parser.add_argument("--ink-max", type=float, default=DEFAULT_INK_RANGE[1])
    parser.add_argument("--chroma-min", type=float, default=DEFAULT_CHROMA_RANGE[0])
    parser.add_argument("--chroma-max", type=float, default=DEFAULT_CHROMA_RANGE[1])
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.mnist_dir) if args.mnist_dir else out_dir / "mnist"

    dataset = build_dataset(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, n_shift=args.n_shift,
        correlation=args.correlation, max_rotation=args.max_rotation,
        jitter_scale=args.jitter_scale, jitter_shift=args.jitter_shift,
        ink_range=(args.ink_min, args.ink_max),
        chroma_range=(args.chroma_min, args.chroma_max),
        noise=args.noise, seed=args.seed, cache_dir=cache_dir,
        frame_width=args.frame, mirror=args.mirror,
    )
    torch.save(to_tensors(dataset), out_dir / "dataset.pth")

    meta = dataset["meta"]
    y = dataset["y"]
    colour = dataset["a"][:, COLOUR]
    # The colour points at class 1 when it is warm, i.e. when the signed strength is up.
    points_high = colour > 0

    print(f"\nwrote {out_dir / 'dataset.pth'}")
    print(f"  {meta['rows']}x{meta['cols']} pictures, {len(meta['factors'])} recorded "
          f"factors: {', '.join(meta['factors'])}")
    for name, n in meta["counts"].items():
        index = dataset["splits"][name]
        agree = (points_high[index] == (y[index] > 0)).mean()
        print(f"  {name:<6s} {n:6d} examples   colour agrees with the answer "
              f"{agree:.3f} of the time")
    print(f"\n  the task: class 1 if the digit is 5-9, class 0 if it is 0-4")
    print(f"  the shortcut: a warm frame for class 1, a cool frame for class 0, "
          f"{0.5 + 0.5 * meta['correlation']:.2f} of the time")
    print(f"  the frames are the same brightness, so only the hue separates them, and the "
          f"frame is {meta['frame_width']} pixels thick")

if __name__ == "__main__":
    main()
