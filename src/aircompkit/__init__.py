"""AirCompKit: security-aware over-the-air computation simulator."""

__version__ = "1.0.0"

# Importing the component packages registers their components.
from aircompkit import (  # noqa: F401
    adversary,
    aircomp,
    channel,
    defense,
    devices,
    learning,
    metrics,
    receiver,
    reproduce,
)
from aircompkit.core.registry import KINDS, available, build, get, register
from aircompkit.core.result import Result
from aircompkit.core.rng import SeedTree
from aircompkit.core.scenario import Scenario

__all__ = [
    "KINDS",
    "Result",
    "Scenario",
    "SeedTree",
    "__version__",
    "available",
    "build",
    "get",
    "register",
]
