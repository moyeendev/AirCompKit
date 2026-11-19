"""Computation metrics and confidence intervals (FEATURES M1, M6)."""

from __future__ import annotations

import numpy as np


def mse_given_channel(
    h: np.ndarray, b: np.ndarray, eta: np.ndarray, noise_var: float, real: bool
) -> np.ndarray:
    """Conditional AirComp MSE of the sum, per trial, for unit-power independent symbols.

    ``sum_k |h_k b_k / sqrt(eta) - 1|^2 + noise_var / eta`` (complex symbols), or with
    ``Re(.)`` in place of the complex amplitude for real symbols. ``noise_var`` is the
    effective noise variance. ``h, b``: ``[trials, K]``; ``eta``: ``[trials]``.
    """
    a = h * b / np.sqrt(eta)[:, None]
    err = (a.real - 1.0) ** 2 if real else np.abs(a - 1.0) ** 2
    return err.sum(axis=1) + noise_var / eta


def nmse(estimate: np.ndarray, target: np.ndarray, axis: int = -1) -> np.ndarray:
    """``||estimate - target||^2 / ||target||^2`` along ``axis``."""
    num = np.sum(np.abs(estimate - target) ** 2, axis=axis)
    den = np.sum(np.abs(target) ** 2, axis=axis)
    return num / np.maximum(den, 1e-300)


def mean_ci(x: np.ndarray, z: float = 1.96) -> tuple[float, float]:
    """Sample mean and normal-approximation CI half-width ``z * s / sqrt(n)``."""
    x = np.asarray(x, float).ravel()
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    half = z * x.std(ddof=1) / np.sqrt(n) if n > 1 else float("inf")
    return float(x.mean()), float(half)
