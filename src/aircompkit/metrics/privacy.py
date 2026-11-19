"""Differential-privacy accounting for over-the-air aggregation (FEATURES F8, M4).

In AirComp the server sees ``sum_k x_k + z`` with Gaussian receiver noise ``z``, so each
round is a Gaussian mechanism on every device's contribution "for free" (Liu & Simeone,
JSAC 2021; Koda et al. 2020). For L2 sensitivity ``Delta`` (one device's update changes,
e.g. clip norm for add/remove) and per-real-dimension noise std ``s``, the Renyi DP of
order ``a`` is ``a Delta^2 / (2 s^2)`` per round; ``T`` rounds compose additively; the
conversion ``eps = rdp + log(1/delta) / (a - 1)`` is minimised over ``a``.
"""

from __future__ import annotations

import numpy as np


def gaussian_rdp(alpha: np.ndarray | float, sensitivity: float, noise_std: float) -> np.ndarray:
    """RDP of the Gaussian mechanism at order(s) ``alpha > 1``."""
    if noise_std <= 0:
        raise ValueError("noise_std must be > 0")
    return np.asarray(alpha, float) * sensitivity**2 / (2 * noise_std**2)


def gaussian_epsilon(sensitivity: float, noise_std: float, rounds: int = 1,
                     delta: float = 1e-5) -> float:
    """``(epsilon, delta)``-DP after ``rounds`` compositions (RDP accountant)."""
    if not 0 < delta < 1 or rounds < 1:
        raise ValueError("need 0 < delta < 1 and rounds >= 1")
    alphas = np.concatenate([np.linspace(1.01, 10, 400), np.geomspace(10, 1e6, 400)])
    eps = rounds * gaussian_rdp(alphas, sensitivity, noise_std) + np.log(1 / delta) / (alphas - 1)
    return float(eps.min())


def aircomp_noise_std(noise_var: float, eta: float, symbols: str = "complex") -> float:
    """Per-real-dimension std of the receiver noise in the normalised AirComp estimate
    ``m^H y / sqrt(eta)``: ``sqrt(noise_var / (2 eta))``."""
    return float(np.sqrt(noise_var / (2 * eta)))
