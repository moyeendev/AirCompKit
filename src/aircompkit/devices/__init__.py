"""Device populations: count, power budgets, placement (FEATURES D1, D2).

A device population answers two questions for the simulator:

* ``power_budgets()``: per-device transmit-power limit ``P_k`` (linear units, per symbol);
* ``sample_distances(rng)``: distance of each device to the server, or ``None`` when the
  channel has no large-scale model.

Placement is drawn once per scenario (from the ``devices`` random stream) and held fixed
across Monte Carlo trials, which is the usual set-up for a fixed group of FL clients.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aircompkit.core.registry import register


def _budgets(K: int, p_max: float | Sequence[float]) -> np.ndarray:
    arr = np.asarray(p_max, dtype=float)
    if arr.ndim == 0:
        arr = np.full(K, float(arr))
    if arr.shape != (K,):
        raise ValueError(f"p_max must be a scalar or have length K={K}, got shape {arr.shape}")
    if np.any(arr <= 0):
        raise ValueError("power budgets must be positive")
    return arr


@register("devices", "fixed")
@dataclass
class FixedDevices:
    """``K`` devices with given power budgets and (optionally) given distances."""

    K: int
    p_max: float | list[float] = 1.0
    distances: list[float] | None = None

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError("K must be >= 1")
        if self.distances is not None:
            d = np.asarray(self.distances, dtype=float)
            if d.shape != (self.K,) or np.any(d <= 0):
                raise ValueError("distances must be K positive numbers")

    def power_budgets(self) -> np.ndarray:
        return _budgets(self.K, self.p_max)

    def sample_distances(self, rng: np.random.Generator) -> np.ndarray | None:
        return None if self.distances is None else np.asarray(self.distances, dtype=float)


@register("devices", "uniform_disc")
@dataclass
class UniformDisc:
    """``K`` devices placed uniformly (by area) in the annulus ``r_min <= r <= r_max``."""

    K: int
    p_max: float | list[float] = 1.0
    r_min: float = 1.0
    r_max: float = 100.0

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError("K must be >= 1")
        if not 0 < self.r_min < self.r_max:
            raise ValueError("need 0 < r_min < r_max")

    def power_budgets(self) -> np.ndarray:
        return _budgets(self.K, self.p_max)

    def sample_distances(self, rng: np.random.Generator) -> np.ndarray:
        u = rng.uniform(self.r_min**2, self.r_max**2, size=self.K)
        return np.sqrt(u)
