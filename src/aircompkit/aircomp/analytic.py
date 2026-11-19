"""Closed-form AirComp MSE expressions, used for validation and as plot references.

All expressions follow the model in ``aircompkit.aircomp.power_control`` and use the
*effective* noise variance (``sigma^2`` for complex symbols, ``sigma^2/2`` for real ones).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def channel_inversion_awgn_mse(
    gains: Sequence[float], p_max: float | Sequence[float], noise_var: float
) -> float:
    """Non-fading channel ``|h_k|^2 = gains_k``: ``MSE = sigma^2 / min_k P_k g_k``."""
    g = np.asarray(gains, float)
    p = np.broadcast_to(np.asarray(p_max, float), g.shape)
    return float(noise_var / np.min(p * g))


def truncated_inversion_rayleigh_mse(
    mean_gains: Sequence[float],
    p_max: float | Sequence[float],
    threshold: float,
    noise_var: float,
    relative: bool = True,
) -> float:
    """Average MSE of truncated channel inversion under Rayleigh fading.

    With ``|h_k|^2 ~ Exp(mean beta_k)``, device ``k`` is silent with probability
    ``q_k = 1 - exp(-thr_k / beta_k)`` and then contributes error 1; ``eta = min_k P_k thr_k``
    is deterministic, so

        E[MSE] = sum_k q_k + sigma^2 / eta.
    """
    beta = np.asarray(mean_gains, float)
    p = np.broadcast_to(np.asarray(p_max, float), beta.shape)
    thr = threshold * beta if relative else np.full_like(beta, threshold)
    q = 1.0 - np.exp(-thr / beta)
    return float(q.sum() + noise_var / np.min(p * thr))
