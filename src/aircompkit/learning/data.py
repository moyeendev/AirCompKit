"""Datasets for learning mode (FEATURES L2). Everything is NumPy; no downloads happen here.

``mnist`` / ``fashion_mnist`` are read from local files, searched in this order:
``root`` argument, ``$AIRCOMPKIT_DATA``, ``~/.aircompkit/data``. Accepted formats:

* ``<name>.npz`` with keys ``x_train, y_train, x_test, y_test`` (the Keras layout) or
  ``xtr, ytr, xte, yte``;
* the four original IDX files (``train-images-idx3-ubyte[.gz]`` etc.) in ``<root>/<name>/``.

``synthetic`` is a Gaussian-mixture classification task generated from a seed, used by the
tests and for quick experiments.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aircompkit.core.registry import register


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    name: str = ""

    @property
    def dim(self) -> int:
        return int(self.x_train.shape[1])


def _search_roots(root: str | Path | None) -> list[Path]:
    roots = []
    if root is not None:
        roots.append(Path(root))
    if os.environ.get("AIRCOMPKIT_DATA"):
        roots.append(Path(os.environ["AIRCOMPKIT_DATA"]))
    roots.append(Path.home() / ".aircompkit" / "data")
    return roots


def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def _load_image_set(name: str, root: str | Path | None) -> Dataset:
    tried = []
    for r in _search_roots(root):
        f = r / f"{name}.npz"
        tried.append(str(f))
        if f.exists():
            with np.load(f) as z:
                keys = set(z.files)
                if {"x_train", "y_train", "x_test", "y_test"} <= keys:
                    xtr, ytr, xte, yte = z["x_train"], z["y_train"], z["x_test"], z["y_test"]
                elif {"xtr", "ytr", "xte", "yte"} <= keys:
                    xtr, ytr, xte, yte = z["xtr"], z["ytr"], z["xte"], z["yte"]
                else:
                    raise ValueError(f"{f}: unrecognised keys {sorted(keys)}")
            return _image_dataset(name, xtr, ytr, xte, yte)
        d = r / name
        files = {}
        for key, stem in (("xtr", "train-images-idx3-ubyte"), ("ytr", "train-labels-idx1-ubyte"),
                          ("xte", "t10k-images-idx3-ubyte"), ("yte", "t10k-labels-idx1-ubyte")):
            for cand in (d / stem, d / f"{stem}.gz"):
                if cand.exists():
                    files[key] = cand
        tried.append(str(d / "*-idx?-ubyte[.gz]"))
        if len(files) == 4:
            return _image_dataset(name, *(_read_idx(files[k]) for k in ("xtr", "ytr", "xte", "yte")))
    raise FileNotFoundError(
        f"{name} not found. Put {name}.npz (Keras layout) or the IDX files in one of:\n  "
        + "\n  ".join(tried)
        + "\nor set AIRCOMPKIT_DATA to the folder that holds it."
    )


def _image_dataset(name, xtr, ytr, xte, yte) -> Dataset:
    xtr = np.asarray(xtr, np.float32).reshape(len(xtr), -1) / 255.0
    xte = np.asarray(xte, np.float32).reshape(len(xte), -1) / 255.0
    mu, sd = float(xtr.mean()), float(xtr.std())
    return Dataset((xtr - mu) / sd, np.asarray(ytr, np.int64), (xte - mu) / sd,
                   np.asarray(yte, np.int64), 10, name)


@register("dataset", "mnist")
def mnist(root: str | Path | None = None) -> Dataset:
    return _load_image_set("mnist", root)


@register("dataset", "fashion_mnist")
def fashion_mnist(root: str | Path | None = None) -> Dataset:
    return _load_image_set("fashion_mnist", root)


@register("dataset", "synthetic")
def synthetic(
    n_train: int = 6000,
    n_test: int = 2000,
    dim: int = 20,
    n_classes: int = 5,
    separation: float = 1.5,
    seed: int = 0,
) -> Dataset:
    """Gaussian mixture: class means ``~ N(0, separation^2 I / dim * ...)``, unit noise."""
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((n_classes, dim)) * separation / np.sqrt(dim) * 3

    def draw(n):
        y = rng.integers(0, n_classes, n)
        return (means[y] + rng.standard_normal((n, dim))).astype(np.float32), y

    xtr, ytr = draw(n_train)
    xte, yte = draw(n_test)
    return Dataset(xtr, ytr, xte, yte, n_classes, "synthetic")
