"""Transmitter-side pre-processing (FEATURES A2): normalisation, clipping, quantisation.

These operate on update vectors ``x [..., K, d]`` before AirComp. The learning loop (P3)
uses them; they are exposed here so that they can also be studied in computation mode.
"""

from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray, axis: tuple[int, ...] | int = (-2, -1)) -> tuple[np.ndarray, float, float]:
    """Zero-mean, unit-variance normalisation with *shared* statistics (as in OTA-FL, where
    the server broadcasts mean and std so that devices scale identically and the sum is
    recoverable). Returns ``(x_n, mu, sd)``; undo the sum with ``denormalize_sum``."""
    mu = float(np.mean(x, axis=axis).mean())
    sd = float(np.std(x - mu))
    sd = sd if sd > 0 else 1.0
    return (x - mu) / sd, mu, sd


def denormalize_sum(f_hat: np.ndarray, mu: float, sd: float, K: int) -> np.ndarray:
    """Map an estimate of ``sum_k x_n,k`` back to an estimate of ``sum_k x_k``."""
    return f_hat * sd + K * mu


def clip_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale each device's vector (last axis) to L2 norm at most ``max_norm``."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x * np.minimum(1.0, max_norm / np.maximum(n, 1e-300))


def stochastic_quantize(
    x: np.ndarray, bits: int, clip: float, rng: np.random.Generator
) -> np.ndarray:
    """Unbiased stochastic uniform quantisation on ``[-clip, clip]`` with ``2^bits`` levels.

    Values are first clipped to the range; inside it ``E[Q(x)] = x`` and the error variance
    is at most ``Delta^2 / 4`` with ``Delta = 2 clip / (2^bits - 1)``.
    """
    if bits < 1 or clip <= 0:
        raise ValueError("need bits >= 1 and clip > 0")
    levels = 2**bits - 1
    delta = 2 * clip / levels
    u = (np.clip(x, -clip, clip) + clip) / delta  # in [0, levels]
    lo = np.floor(u)
    q = lo + (rng.random(np.shape(x)) < (u - lo))
    return np.minimum(q, levels) * delta - clip
