"""Gradient bank (FEATURES L5): record honest device updates once, replay them through many
channel / attack / defence realisations without retraining.

A training run records ``updates[r] [K_r, d]`` (float32) and the ids of the scheduled devices
for selected rounds. ``GradientBank.replay`` sends the stored updates through a
``Scenario``'s physical layer ``trials`` times per round and
reports the aggregation NMSE against the ideal mean. P4 builds attack/detection Monte Carlo
on the same mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aircompkit.core.rng import SeedTree
from aircompkit.learning.aggregate import aircomp_aggregate


@dataclass
class GradientBank:
    rounds: list[int] = field(default_factory=list)
    updates: list[np.ndarray] = field(default_factory=list)
    device_ids: list[np.ndarray] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def record(self, round_idx: int, ids: np.ndarray, x: np.ndarray) -> None:
        self.rounds.append(int(round_idx))
        self.device_ids.append(np.asarray(ids, dtype=np.int64))
        self.updates.append(np.asarray(x, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.rounds)

    def save(self, path: str | Path) -> Path:
        p = Path(path).with_suffix(".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"rounds": np.asarray(self.rounds)}
        for i, (u, ids) in enumerate(zip(self.updates, self.device_ids)):
            arrays[f"u{i}"] = u
            arrays[f"ids{i}"] = ids
        import json

        arrays["meta"] = np.frombuffer(json.dumps(self.meta).encode(), dtype=np.uint8)
        np.savez_compressed(p, **arrays)  # type: ignore[arg-type]
        return p

    @classmethod
    def load(cls, path: str | Path) -> GradientBank:
        import json

        with np.load(Path(path).with_suffix(".npz")) as z:
            n = len(z["rounds"])
            return cls(
                rounds=z["rounds"].tolist(),
                updates=[z[f"u{i}"] for i in range(n)],
                device_ids=[z[f"ids{i}"] for i in range(n)],
                meta=json.loads(z["meta"].tobytes().decode()) if "meta" in z.files else {},
            )

    def replay(self, scenario, trials: int = 10, subchannels: int = 1, seed: int = 0,
               normalize: bool = True) -> dict[str, np.ndarray]:
        """Aggregate every stored round ``trials`` times over fresh channel draws.

        Returns ``nmse [rounds, trials]`` (``||est - mean||^2 / ||mean||^2``) and ``rounds``.
        Devices keep the placement drawn from ``scenario.seed``.
        """
        tree = SeedTree(seed)
        p_max_all = scenario.devices.power_budgets()
        dist = scenario.devices.sample_distances(SeedTree(scenario.seed).generator("devices"))
        out = np.empty((len(self), trials))
        for i, (ids, x) in enumerate(zip(self.device_ids, self.updates)):
            x = x.astype(np.float64)
            ideal = x.mean(axis=0)
            for t in range(trials):
                node = tree.child(f"replay{i}").child(str(t))
                h_act, h_est, beta = scenario.sample_channels(node, subchannels, dist)
                lay = scenario.design(h_act[:, ids], h_est[:, ids], beta[:, ids], p_max_all[ids])
                est = aircomp_aggregate(x, lay["amp"], lay["eta"], lay["p"] > 0,
                                        scenario.noise_var_eff, node.generator("noise"),
                                        normalize=normalize)
                out[i, t] = np.sum((est - ideal) ** 2) / max(np.sum(ideal**2), 1e-300)
        return {"rounds": np.asarray(self.rounds), "nmse": out}
