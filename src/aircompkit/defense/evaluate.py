"""Detection Monte Carlo on a gradient bank (FEATURES M3): stored honest updates, a fresh
channel, fresh attacker set and fresh noise per trial, one received block, one detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from aircompkit.core.rng import SeedTree
from aircompkit.learning.array import Receiver, normalise, pack


def detection_sweep(scenario, updates: list[np.ndarray], attack: Any, n_attackers: int,
                    detector: Any, trials: int = 200, clip: float | None = None,
                    seed: int = 0) -> dict[str, float]:
    """Per-trial: pick a stored update matrix ``[K, d]``, draw ``n_attackers`` attackers, let
    them run ``attack`` on the array-level channel, run ``detector.detect(Y, A, noise_var)``.

    Returns ``tpr`` (detected attackers / attackers), ``fp`` (honest devices flagged per trial),
    ``cover`` (all attackers flagged) and ``exact`` (flagged set == attacker set).
    """
    from aircompkit.core.registry import build

    attack = build("attack", attack)
    tree = SeedTree(seed)
    rng = tree.generator("trials")
    dist = scenario.devices.sample_distances(SeedTree(scenario.seed).generator("devices"))
    p_max = scenario.devices.power_budgets()
    tp = fp = cover = exact = 0
    for t in range(trials):
        G = np.asarray(updates[rng.integers(len(updates))], float)
        K = G.shape[0]
        byz = np.sort(rng.choice(K, n_attackers, replace=False)) if n_attackers else np.zeros(0, int)
        hon = np.setdiff1d(np.arange(K), byz)
        x, _ = normalise(G, clip)
        gref = pack(G[hon].mean(axis=0))  # scale is irrelevant: attacks normalise it
        rx = Receiver(scenario, x, p_max, dist, tree.child("trial").child(str(t)),
                      attackers=byz, attack=attack if n_attackers else None, gref=gref)
        r = rx.transmit()
        flg = {int(k) for k in detector.detect(r.Y, r.A, r.noise_var)}
        B = set(byz.tolist())
        tp += len(flg & B)
        fp += len(flg - B)
        cover += int(flg >= B)
        exact += int(flg == B)
    return {"tpr": tp / max(n_attackers * trials, 1), "fp": fp / trials,
            "cover": cover / trials, "exact": exact / trials}
