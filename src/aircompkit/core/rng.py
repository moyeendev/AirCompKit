"""Deterministic seed tree (FEATURES C3).

One master seed yields independent, *named* random streams. A stream depends only on the
master seed and its path of names, not on how many other streams were created or in which
order. So adding an attacker does not change the channel realisations, which makes
"with vs. without attack" comparisons paired.

    tree = SeedTree(0)
    h = tree.generator("channel").standard_normal(4)
    tree.child("round").generator("noise")
"""

from __future__ import annotations

import hashlib

import numpy as np


def _name_key(name: str) -> int:
    # Stable across processes and Python versions (unlike hash()).
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


class SeedTree:
    """A node in a tree of named, reproducible random streams."""

    def __init__(self, seed: int, _path: tuple[str, ...] = ()) -> None:
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = int(seed)
        self.path = _path

    def child(self, name: str) -> SeedTree:
        """Sub-tree for a component, e.g. ``tree.child("adversary")``."""
        return SeedTree(self.seed, self.path + (name,))

    def seed_sequence(self, name: str | None = None) -> np.random.SeedSequence:
        path = self.path + ((name,) if name is not None else ())
        return np.random.SeedSequence(self.seed, spawn_key=tuple(_name_key(p) for p in path))

    def generator(self, name: str | None = None) -> np.random.Generator:
        """A fresh ``np.random.Generator`` for the stream ``path/name``."""
        return np.random.Generator(np.random.PCG64(self.seed_sequence(name)))

    def __repr__(self) -> str:
        return f"SeedTree(seed={self.seed}, path={'/'.join(self.path) or '/'})"
