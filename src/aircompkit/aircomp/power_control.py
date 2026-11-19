"""Transmit power control for single-antenna AirComp (FEATURES A3).

System model (per vector entry, per trial)::

    y = sum_k h_k b_k x_k + n,       n ~ CN(0, sigma^2),    |b_k|^2 = p_k <= P_k
    f_hat = y / sqrt(eta)            (estimate of f = sum_k x_k)

All schemes use phase-aligned pre-scaling ``b_k = sqrt(p_k) h_k^* / |h_k|``, so device ``k``
arrives with real amplitude ``sqrt(p_k g_k)`` where ``g_k = |h_k|^2``. With unit-power,
independent symbols the conditional MSE is

    MSE = sum_k (sqrt(p_k g_k / eta) - 1)^2 + sigma_eff^2 / eta

where ``sigma_eff^2`` is ``sigma^2`` for complex symbols and ``sigma^2/2`` for real symbols
(the receiver keeps the real part). A scheme maps ``(g, beta, P, sigma_eff^2)`` to
``(p, eta)``; ``g, beta`` have shape ``[trials, K]``, ``P`` shape ``[K]``, and the outputs
have shapes ``[trials, K]`` and ``[trials]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aircompkit.core.registry import register

_TINY = 1e-300
_EULER_GAMMA = 0.5772156649015329


def exp1(x: float) -> float:
    """Exponential integral ``E_1(x) = int_x^inf e^{-t}/t dt`` for ``x > 0`` (NumPy-free scalar).

    Power series for ``x <= 1``; modified-Lentz continued fraction otherwise. Relative
    accuracy ~1e-14.
    """
    if x <= 0:
        raise ValueError("exp1 needs x > 0")
    if x <= 1.0:
        total, term, n = 0.0, 1.0, 1
        while True:
            term *= -x / n
            add = -term / n
            total += add
            if abs(add) < 1e-17 * abs(total) or n > 200:
                break
            n += 1
        return -_EULER_GAMMA - float(np.log(x)) + total
    # E_1(x) = e^{-x} / (x + 1 - 1/(x + 3 - 4/(x + 5 - ...)))
    tiny = 1e-300
    b = x + 1.0
    c = 1.0 / tiny
    d = 1.0 / b
    f = d
    for i in range(1, 300):
        a = -float(i * i)
        b += 2.0
        d = 1.0 / (a * d + b)
        c = b + a / c
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return float(f * np.exp(-x))


@register("power_control", "channel_inversion")
@dataclass
class ChannelInversion:
    """Zero-forcing: every device inverts its channel; ``eta = min_k P_k g_k``.

    Error-free alignment, but the weakest device sets ``eta``. Under Rayleigh fading the
    *average* MSE is infinite (``E[1/min_k g_k]`` diverges), which is why truncation exists.
    """

    def __call__(self, g, beta, p_max, noise_var):
        eta = np.min(p_max[None, :] * g, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(g > 0, eta[:, None] / g, 0.0)
        return p, eta


@register("power_control", "truncated_inversion")
@dataclass
class TruncatedInversion:
    """Truncated channel inversion (Zhu, Wang & Huang, IEEE TWC 2020).

    Device ``k`` transmits only if ``g_k >= thr_k`` and then inverts its channel; otherwise
    it is silent. ``thr_k = threshold * beta_k`` when ``relative`` (threshold on small-scale
    fading, the usual choice under path loss), else ``thr_k = threshold``.

    ``budget="peak"`` (default): ``p_k <= P_k`` on every channel use, so
    ``eta = min_k P_k thr_k``. ``budget="average"``: ``E[p_k] <= P_k`` over Rayleigh fading,
    as in Zhu et al. Since ``E[1/g; g >= threshold beta] = E_1(threshold) / beta`` for
    ``g ~ Exp(mean beta)``, ``eta = min_k P_k beta_k / E_1(threshold)`` (paper eq. (15)); this
    needs ``relative=True`` and Rayleigh fading, and instantaneous power may exceed ``P_k``.
    Either way ``eta`` is fixed by the threshold, so the noise term never blows up.
    """

    threshold: float = 0.1
    relative: bool = True
    budget: str = "peak"

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.budget not in ("peak", "average"):
            raise ValueError("budget must be 'peak' or 'average'")
        if self.budget == "average" and not self.relative:
            raise ValueError("budget='average' needs relative=True")

    @property
    def peak_constrained(self) -> bool:
        return self.budget == "peak"

    def __call__(self, g, beta, p_max, noise_var):
        thr = self.threshold * (beta if self.relative else np.ones_like(g))
        if self.budget == "average":
            eta = np.min(p_max[None, :] * beta, axis=1) / exp1(self.threshold)
        else:
            eta = np.min(p_max[None, :] * thr, axis=1)
        active = g >= thr
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(active, eta[:, None] / g, 0.0)
        return p, eta


@register("power_control", "full_power")
@dataclass
class FullPower:
    """Every device transmits at ``P_k``; the receiver picks the MSE-optimal ``eta``.

    For fixed ``p``, ``MSE(t = 1/sqrt(eta))`` is a convex quadratic with minimiser
    ``t* = sum_k sqrt(G_k) / (sigma^2 + sum_k G_k)``, ``G_k = P_k g_k``.
    """

    def __call__(self, g, beta, p_max, noise_var):
        G = p_max[None, :] * g
        t = np.sqrt(G).sum(axis=1) / (noise_var + G.sum(axis=1))
        eta = 1.0 / np.maximum(t, _TINY) ** 2
        return np.broadcast_to(p_max, g.shape).copy(), eta


@register("power_control", "optimal")
@dataclass
class OptimalPowerControl:
    """MSE-optimal power control (Cao, Zhu, Xu & Huang, IEEE TWC 2020).

    Optimal structure: for a given ``eta``, devices with ``G_k = P_k g_k >= eta`` invert their
    channel (zero error) and the rest transmit at full power. Sort ``G`` ascending; if exactly
    the ``i`` weakest devices are at full power, ``MSE(eta)`` is convex in ``1/sqrt(eta)`` with
    stationary point ``eta_i = ((sigma^2 + S1_i) / S2_i)^2`` (``S1_i = sum_{k<=i} G_k``,
    ``S2_i = sum_{k<=i} sqrt(G_k)``). Clip ``eta_i`` to ``[G_i, G_{i+1}]``, evaluate, and take
    the best of these ``K`` candidates and ``eta = G_1`` (pure inversion). Exact, ``O(K log K)``
    per trial, vectorised over trials.
    """

    def __call__(self, g, beta, p_max, noise_var):
        T, K = g.shape
        G = p_max[None, :] * g
        Gs = np.sort(G, axis=1)
        S1 = np.cumsum(Gs, axis=1)
        S2 = np.maximum(np.cumsum(np.sqrt(Gs), axis=1), _TINY)
        n_full = np.arange(1, K + 1)

        upper = np.concatenate([Gs[:, 1:], np.full((T, 1), np.inf)], axis=1)
        eta_c = np.clip(((noise_var + S1) / S2) ** 2, Gs, upper)
        eta_c = np.maximum(eta_c, _TINY)
        mse_c = (S1 + noise_var) / eta_c - 2 * S2 / np.sqrt(eta_c) + n_full

        best = np.argmin(mse_c, axis=1)
        rows = np.arange(T)
        eta = eta_c[rows, best]
        mse_best = mse_c[rows, best]

        g1 = np.maximum(Gs[:, 0], _TINY)
        use_inversion = noise_var / g1 <= mse_best
        eta = np.where(use_inversion, g1, eta)

        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(g > 0, np.minimum(p_max[None, :], eta[:, None] / g), p_max[None, :])
        return p, eta
