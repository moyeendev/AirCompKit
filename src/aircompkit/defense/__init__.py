"""Defences for over-the-air aggregation (FEATURES §6).

A defence is a stateful object with ``reset(K)`` and ``aggregate(rx, ctx)``. ``rx`` is the
round's ``Receiver``; the defence decides how the devices
use the channel (``rx.transmit(ids)`` for AirComp, ``rx.transmit_orthogonal()`` for
digital baselines) and returns ``(est, keff, flagged)``: ``est`` estimates
``sum_{k honest} x_k`` in normalised symbols up to the factor ``keff`` (the server divides
by ``keff``), and ``flagged`` is the set of devices it excluded.

* ``none``: plain AirComp (receive with the design combiner ``m``);
* ``genie``: null-steering with the true attacker set (an oracle upper bound);
* ``group_gm``: random groups, one AirComp per group, geometric median of the group means
  (robust AirComp over repeated transmissions; Sifaou & Li; Huang et al.);
* ``digital_robust``: orthogonal access and a robust rule (``mean``, ``median``,
  ``trimmed_mean``, ``krum``, ``geomed``) on the per-device estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aircompkit.core.registry import register


# ------------------------------------------------------------------- primitives
def nulling_combiner(A, flagged, eta, sigma2):
    """Regularised LS combiner with hard nulls on the flagged signatures.
    Returns ``(w, keep)``; ``w^H a_k ~ sqrt(eta)`` for kept devices, ``0`` for flagged."""
    N, K = A.shape
    keep = np.ones(K, bool)
    keep[list(flagged)] = False
    if len(flagged):
        Qf, _ = np.linalg.qr(A[:, list(flagged)], mode="complete")
        Q = Qf[:, len(flagged):]
    else:
        Q = np.eye(N, dtype=complex)
    Ah = Q.conj().T @ A[:, keep]
    v = np.linalg.solve(Ah @ Ah.conj().T + sigma2 * np.eye(Q.shape[1]), Ah.sum(axis=1) * np.sqrt(eta))
    return Q @ v, keep


def geometric_median(X: np.ndarray, iters: int = 100, eps: float = 1e-8) -> np.ndarray:
    """Weiszfeld iterations on the rows of ``X`` (real or complex)."""
    z = X.mean(axis=0)
    for _ in range(iters):
        dist = np.maximum(np.linalg.norm(X - z, axis=1), eps)
        z_new = (X / dist[:, None]).sum(axis=0) / (1 / dist).sum()
        if np.linalg.norm(z_new - z) < 1e-10 * (1 + np.linalg.norm(z)):
            return z_new
        z = z_new
    return z


def krum(X: np.ndarray, f: int) -> np.ndarray:
    """Krum (Blanchard et al. 2017): the row with the smallest sum of squared distances to its
    ``K - f - 2`` nearest neighbours."""
    K = len(X)
    m = max(K - f - 2, 1)
    D = np.sum(np.abs(X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    scores = np.sort(D, axis=1)[:, 1:m + 1].sum(axis=1)
    return X[int(np.argmin(scores))]


# ------------------------------------------------------------------- defences
@register("defense", "none")
@dataclass
class NoDefense:
    """Plain AirComp: ``m^H Y / sqrt(eta)``, divided by the number of transmitting devices."""

    def reset(self, K: int) -> None:
        pass

    def aggregate(self, rx, ctx):
        r = rx.transmit()
        return r.m.conj() @ r.Y / np.sqrt(r.eta), float(len(r.ids)), set()


def _nulled(r, flagged):
    w, keep = nulling_combiner(r.A, sorted(flagged), r.eta, r.noise_var)
    keff = float((w.conj() @ r.A[:, keep]).sum().real / np.sqrt(r.eta))
    return w.conj() @ r.Y / np.sqrt(r.eta), keff


@register("defense", "genie")
@dataclass
class Genie:
    """Oracle: null-steer exactly the true attackers (upper bound for detection-based defences)."""

    def reset(self, K: int) -> None:
        pass

    def aggregate(self, rx, ctx):
        r = rx.transmit()
        flagged = {int(k) for k in ctx["attackers"]}
        est, keff = _nulled(r, flagged)
        return est, keff, flagged


@register("defense", "group_gm")
@dataclass
class GroupGM:
    """Random groups, one AirComp per group (``groups`` channel uses), geometric median of the
    normalised group means."""

    groups: int = 8

    def reset(self, K: int) -> None:
        pass

    def aggregate(self, rx, ctx):
        K = rx.x.shape[0]
        perm = rx.rng.permutation(K)
        means = []
        for g, grp in enumerate(np.array_split(perm, self.groups)):
            r = rx.transmit(np.sort(grp), tag=f"group{g}")
            means.append(r.m.conj() @ r.Y / (np.sqrt(r.eta) * len(grp)))
        M = np.array(means)
        z = geometric_median(np.concatenate([M.real, M.imag], axis=1))
        L = M.shape[1]
        return z[:L] + 1j * z[L:], 1.0, set()


@register("defense", "digital_robust")
@dataclass
class DigitalRobust:
    """Orthogonal access (K channel uses, full power, MRC) + a robust aggregation rule."""

    rule: str = "median"
    trim: float = 0.1
    f: int = 0

    def __post_init__(self) -> None:
        if self.rule not in ("mean", "median", "trimmed_mean", "krum", "geomed"):
            raise ValueError(f"unknown rule {self.rule!r}")

    def reset(self, K: int) -> None:
        pass

    def aggregate(self, rx, ctx):
        X = rx.transmit_orthogonal()
        R = np.concatenate([X.real, X.imag], axis=1)
        if self.rule == "mean":
            z = R.mean(axis=0)
        elif self.rule == "median":
            z = np.median(R, axis=0)
        elif self.rule == "trimmed_mean":
            k = int(self.trim * len(R))
            z = np.sort(R, axis=0)[k:len(R) - k].mean(axis=0)
        elif self.rule == "krum":
            z = krum(R, self.f or len(ctx["attackers"]))
        else:
            z = geometric_median(R)
        L = X.shape[1]
        return z[:L] + 1j * z[L:], 1.0, set()
