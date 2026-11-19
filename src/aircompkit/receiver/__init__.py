"""Receive beamformers for multi-antenna AirComp (FEATURES A4).

The server applies a unit-norm combiner ``m`` (shape ``[trials, n_rx]``) to the received
vector, which reduces the SIMO multiple-access channel to a scalar one with effective
coefficients ``c_k = m^H h_k`` and unchanged noise variance. Any power-control scheme from
``aircompkit.aircomp.power_control`` then runs on ``|c_k|^2``. Splitting the design this
way covers the standard receive-beamforming designs:

* with ``channel_inversion`` (zero-forcing transmit, Yang et al. TWC 2020, Prop. 1) the MSE is
  ``sigma^2 / min_k P_k |m^H h_k|^2``, so the best ``m`` solves the max-min-fair multicast
  problem; this is what ``maxmin`` and ``sdr`` target;
* ``joint_mmse`` alternates an LMMSE receiver with per-device optimal transmit scaling and is
  meant to be paired with ``optimal`` power control.

A combiner is called as ``m = rx(h_est, p_max, noise_var)`` with ``h_est`` of shape
``[trials, K, n_rx]``. With ``n_rx = 1`` every combiner returns ``m = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aircompkit.core.registry import register


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-300)


def effective_gains(h: np.ndarray, m: np.ndarray, p_max: np.ndarray | None = None) -> np.ndarray:
    """``P_k |m^H h_k|^2`` for ``h [..., K, N]`` and ``m [..., N]`` (``P_k = 1`` if not given)."""
    g = np.abs(np.einsum("...kn,...n->...k", h, m.conj())) ** 2
    return g if p_max is None else g * p_max


def zf_mse(h: np.ndarray, m: np.ndarray, p_max: np.ndarray, noise_var: float) -> np.ndarray:
    """MSE of zero-forcing transmission for a given combiner: ``sigma^2 / min_k P_k |m^H h_k|^2``."""
    return noise_var / effective_gains(h, _unit(m), p_max).min(axis=-1)


def _ones(h: np.ndarray) -> np.ndarray:
    return np.ones(h.shape[:1] + h.shape[2:], dtype=complex)


@register("receiver", "eigen")
@dataclass
class EigenCombiner:
    """Dominant eigenvector of ``sum_k P_k h_k h_k^H / ||sqrt(P_k) h_k||^2`` (every device
    weighted equally by direction). Cheap baseline; the max-min initialiser."""

    def __call__(self, h, p_max, noise_var):
        if h.shape[-1] == 1:
            return _ones(h)
        u = _unit(h)
        R = np.einsum("tkn,tkm->tnm", u, u.conj())
        _, vecs = np.linalg.eigh(R)
        return vecs[..., -1]


@register("receiver", "maxmin")
@dataclass
class MaxMinCombiner:
    """Max-min-fair combiner ``argmax_{||m||=1} min_k P_k |m^H h_k|^2`` (solver-free).

    The problem is NP-hard in general (single-group multicast beamforming). This uses
    projected gradient ascent on a log-sum-exp smoothed minimum with an annealed temperature,
    started from every eigenvector of the direction-correlation matrix plus ``random_inits``
    seeded random points, and keeps the best exact objective seen. Deterministic given
    ``seed`` and vectorised over trials. Its value never exceeds the SDR upper bound
    (``sdr_solve``); see the tests and the reproduction of Yang et al. 2020, Figs. 4–5.
    """

    iters: int = 400
    random_inits: int = 24
    step: float = 0.5
    temp_start: float = 1.0
    temp_end: float = 0.005
    seed: int = 0

    def __call__(self, h, p_max, noise_var):
        T, _K, N = h.shape
        if N == 1:
            return _ones(h)
        hs = h * np.sqrt(p_max)[None, :, None]
        u = _unit(hs)
        R = np.einsum("tkn,tkm->tnm", u, u.conj())
        _, vecs = np.linalg.eigh(R)
        m = np.swapaxes(vecs, 1, 2)  # [T, N inits, N]
        if self.random_inits > 0:
            z = np.random.default_rng(self.seed).standard_normal((T, self.random_inits, N, 2))
            m = np.concatenate([m, _unit(z[..., 0] + 1j * z[..., 1])], axis=1)
        lr = self.step / np.max(np.sum(np.abs(hs) ** 2, axis=2), axis=1)[:, None, None]

        def gains(mm):
            c = np.einsum("tkn,trn->trk", hs, mm.conj())
            return c, np.abs(c) ** 2

        c, G = gains(m)
        best_val = G.min(axis=2)
        best_m = m.copy()
        temps = np.geomspace(self.temp_start, self.temp_end, self.iters)
        for tau in temps:
            gmin = G.min(axis=2, keepdims=True)
            z = -(G - gmin) / (tau * np.maximum(gmin, 1e-300))
            w = np.exp(z)
            w /= w.sum(axis=2, keepdims=True)
            # d/dm* of |m^H h_k|^2 is h_k (h_k^H m) = h_k conj(c_k)
            grad = np.einsum("trk,tkn->trn", w * c.conj(), hs)
            m = _unit(m + lr * grad)
            c, G = gains(m)
            val = G.min(axis=2)
            better = val > best_val
            best_val = np.where(better, val, best_val)
            best_m = np.where(better[..., None], m, best_m)
        pick = np.argmax(best_val, axis=1)
        return best_m[np.arange(T), pick]


@register("receiver", "sdr")
@dataclass
class SDRCombiner:
    """Semidefinite relaxation + Gaussian randomisation for the max-min combiner, the
    baseline of Yang et al. (TWC 2020) and Chen, Qin & Wei (IEEE WCL 2018).

    The SDP ``max_{M >= 0, tr M = 1} min_k h_k^H M h_k`` is solved without an external solver
    through its dual (see ``sdr_solve``); then ``randomizations`` candidates
    ``m ~ CN(0, M)`` are drawn and the best one (or the principal eigenvector) is kept.
    """

    randomizations: int = 100
    iters: int = 400
    seed: int = 0

    def __call__(self, h, p_max, noise_var):
        _T, _K, N = h.shape
        if N == 1:
            return _ones(h)
        rng = np.random.default_rng(self.seed)
        hs = h * np.sqrt(p_max)[None, :, None]
        M, _, _ = sdr_solve(hs, self.iters)
        return gaussian_randomization(M, hs, self.randomizations, rng)


def sdr_solve(hs: np.ndarray, iters: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the max-min SDR ``max_{M >= 0, tr M = 1} min_k h_k^H M h_k`` for ``hs [T, K, N]``.

    By the minimax theorem its value equals ``min_{w in simplex} lambda_max(sum_k w_k h_k h_k^H)``.
    That convex problem is solved by entropic mirror descent on ``w``; the running average of
    the best responses ``v v^H`` (top eigenvectors) converges to an optimal ``M``.

    Returns ``(M [T, N, N], lower [T], upper [T])``. ``lower = min_k h_k^H M h_k`` is achieved
    by the returned ``M``; ``upper`` is the smallest dual value seen. The SDR value lies in
    ``[lower, upper]``, and ``upper`` upper-bounds the max-min gain of *every* unit-norm
    combiner, so ``1/upper`` lower-bounds the zero-forcing MSE.
    """
    T, K, N = hs.shape
    H = np.einsum("tkn,tkm->tknm", hs, hs.conj())  # h_k h_k^H
    scale = np.max(np.sum(np.abs(hs) ** 2, axis=2), axis=1)[:, None]
    w = np.full((T, K), 1.0 / K)
    M_sum = np.zeros((T, N, N), dtype=complex)
    upper = np.full(T, np.inf)
    for it in range(1, iters + 1):
        S = np.einsum("tk,tknm->tnm", w, H)
        vals, vecs = np.linalg.eigh(S)
        v = vecs[..., -1]
        upper = np.minimum(upper, vals[..., -1])
        M_sum += np.einsum("tn,tm->tnm", v, v.conj())
        grad = np.abs(np.einsum("tkn,tn->tk", hs.conj(), v)) ** 2  # d lambda_max / d w_k
        w = w * np.exp(-(2.0 / np.sqrt(it)) * grad / scale)
        w /= w.sum(axis=1, keepdims=True)
    M = M_sum / iters
    lower = np.real(np.einsum("tkn,tnm,tkm->tk", hs.conj(), M, hs)).min(axis=1)
    return M, lower, upper


def gaussian_randomization(
    M: np.ndarray, hs: np.ndarray, L: int, rng: np.random.Generator
) -> np.ndarray:
    """Per trial, the best of ``L`` normalised draws ``m ~ CN(0, M)`` and the principal
    eigenvector of ``M``, by the exact max-min objective. ``M [T, N, N]``, ``hs [T, K, N]``."""
    T, N, _ = M.shape
    vals, vecs = np.linalg.eigh((M + np.swapaxes(M.conj(), 1, 2)) / 2)
    root = vecs * np.sqrt(np.maximum(vals, 0))[:, None, :]
    z = rng.standard_normal((T, L, N, 2))
    z = (z[..., 0] + 1j * z[..., 1]) / np.sqrt(2)
    cand = np.einsum("tnj,tlj->tln", root, z)
    cand = _unit(np.concatenate([cand, vecs[:, None, :, -1]], axis=1))
    g = np.abs(np.einsum("tkn,tln->tlk", hs, cand.conj())) ** 2
    pick = np.argmax(g.min(axis=2), axis=1)
    return cand[np.arange(T), pick]


@register("receiver", "joint_mmse")
@dataclass
class JointMMSE:
    """Alternating transceiver design for the MSE ``sum_k |a^H h_k b_k - 1|^2 + sigma^2 ||a||^2``.

    * receive step: LMMSE ``a = (sum_k h~_k h~_k^H + sigma^2 I)^{-1} sum_k h~_k``, ``h~_k = h_k b_k``;
    * transmit step (per device, exact): ``b_k = min(sqrt(P_k), 1/|c_k|) c_k^* / |c_k|``,
      ``c_k = a^H h_k``.

    Each step minimises the MSE in its block, so the MSE never increases. Starts from the
    ``maxmin`` combiner. Returns ``m = a/||a||``; pair with ``optimal`` power control, which
    re-optimises the transmit side for that ``m`` (and so can only lower the MSE further).
    """

    iters: int = 30

    def __call__(self, h, p_max, noise_var):
        if h.shape[-1] == 1:
            return _ones(h)
        a, _, _ = self.design(h, p_max, noise_var)
        return _unit(a)

    def design(self, h, p_max, noise_var):
        """Run the alternation; returns ``(a, b, history)`` where ``history[i]`` is the
        per-trial MSE after ``i`` iterations (``history[0]``: zero-forcing start)."""
        _T, _K, N = h.shape
        m = MaxMinCombiner(iters=150)(h, p_max, noise_var)
        sp = np.sqrt(p_max)[None, :]
        # initial transmit side: zero-forcing for m, scaled so the weakest device is at P_k
        c = np.einsum("tkn,tn->tk", h, m.conj())
        eta = (np.abs(c) ** 2 * p_max).min(axis=1)
        b = np.sqrt(eta)[:, None] * c.conj() / np.abs(c) ** 2
        eye = np.eye(N)[None]
        a = m / np.sqrt(eta)[:, None]
        history = [joint_mse(h, a, b, noise_var)]
        for _ in range(self.iters):
            ht = h * b[..., None]
            A = np.einsum("tkn,tkm->tnm", ht, ht.conj()) + noise_var * eye
            a = np.linalg.solve(A, ht.sum(axis=1)[..., None])[..., 0]
            c = np.einsum("tkn,tn->tk", h, a.conj())
            mag = np.maximum(np.abs(c), 1e-300)
            b = np.minimum(sp, 1.0 / mag) * c.conj() / mag
            history.append(joint_mse(h, a, b, noise_var))
        return a, b, np.array(history)


def joint_mse(h, a, b, noise_var):
    """``sum_k |a^H h_k b_k - 1|^2 + sigma^2 ||a||^2`` per trial (unnormalised ``a``)."""
    c = np.einsum("tkn,tn->tk", h, a.conj())
    return (np.abs(c * b - 1) ** 2).sum(axis=1) + noise_var * np.sum(np.abs(a) ** 2, axis=1)
