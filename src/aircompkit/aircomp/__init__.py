"""AirComp transmitter side: target functions and power control (FEATURES A1, A3, A8)."""

from __future__ import annotations

from dataclasses import dataclass

from aircompkit.aircomp.power_control import (
    ChannelInversion,
    FullPower,
    OptimalPowerControl,
    TruncatedInversion,
)
from aircompkit.core.registry import register


@register("aircomp", "sum")
@dataclass
class Sum:
    """Target ``f = sum_k x_k``."""

    def scale(self, K: int) -> float:
        return 1.0


@register("aircomp", "mean")
@dataclass
class Mean:
    """Target ``f = (1/K) sum_k x_k``. The receiver divides the sum estimate by ``K``."""

    def scale(self, K: int) -> float:
        return 1.0 / K


__all__ = [
    "ChannelInversion",
    "FullPower",
    "Mean",
    "OptimalPowerControl",
    "Sum",
    "TruncatedInversion",
]
