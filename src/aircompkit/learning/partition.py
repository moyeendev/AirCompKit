"""Splitting a training set across devices (FEATURES L3). Each partitioner is called as
``part(labels, K, rng) -> list of K index arrays``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aircompkit.core.registry import register


@register("partitioner", "iid")
@dataclass
class IID:
    """Random equal shares (Zhu et al.: 60000 MNIST samples into 200 shares)."""

    def __call__(self, labels, K, rng):
        return [np.sort(s) for s in np.array_split(rng.permutation(len(labels)), K)]


@register("partitioner", "shards")
@dataclass
class Shards:
    """Sort by label, cut into ``K * shards_per_device`` shards, give each device that many
    (McMahan et al.; Zhu et al. use 400 shards of 150 for 200 devices)."""

    shards_per_device: int = 2

    def __call__(self, labels, K, rng):
        order = np.argsort(labels, kind="stable")
        shards = np.array_split(order, K * self.shards_per_device)
        pick = rng.permutation(len(shards))
        return [np.sort(np.concatenate([shards[j] for j in
                                        pick[k * self.shards_per_device:(k + 1) * self.shards_per_device]]))
                for k in range(K)]


@register("partitioner", "dirichlet")
@dataclass
class Dirichlet:
    """Label-Dirichlet(alpha) non-IID split; resamples until every device has ``min_size``."""

    alpha: float = 0.5
    min_size: int = 10

    def __call__(self, labels, K, rng):
        classes = np.unique(labels)
        for _ in range(1000):
            idx: list[list[int]] = [[] for _ in range(K)]
            for c in classes:
                ic = rng.permutation(np.flatnonzero(labels == c))
                cuts = (np.cumsum(rng.dirichlet(self.alpha * np.ones(K))) * len(ic)).astype(int)[:-1]
                for k, part in enumerate(np.split(ic, cuts)):
                    idx[k].extend(part.tolist())
            if min(len(i) for i in idx) >= self.min_size:
                return [np.sort(np.asarray(i, dtype=np.int64)) for i in idx]
        raise RuntimeError("Dirichlet split: could not satisfy min_size; lower it or raise alpha")
