"""Array-level over-the-air aggregation with an adversary (the path used by P4 security).

Real update vectors are packed two-per-complex-symbol (``L = ceil(d/2)`` symbols) and scaled
to unit average symbol power (Zhu/Yang-style analog transmission; with ``clip`` set this is
``x_k = (g_k^(1) + j g_k^(2)) / nu``, ``nu = clip / sqrt(L)``). One block-fading
channel use per round: the server observes the whole array block

    Y = sum_k h_k t_k^T + n  (N x L),      t_k = b_k x_k for honest devices,

and *effective signatures* ``A = [b_k h_est_k]`` (what it believes each device's spatial
signature to be). A ``Receiver`` hands defences a ``transmit(ids)`` callback, so a defence
may use one channel use for everyone (AirComp), one per group (group-based robust AirComp),
or orthogonal access (digital robust baselines).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aircompkit.channel import _cn


def pack(x: np.ndarray) -> np.ndarray:
    """Real ``[..., d]`` to complex ``[..., ceil(d/2)]`` (first half + j * second half)."""
    d = x.shape[-1]
    if d % 2:
        x = np.concatenate([x, np.zeros(x.shape[:-1] + (1,))], axis=-1)
    L = x.shape[-1] // 2
    return x[..., :L] + 1j * x[..., L:]


def unpack(z: np.ndarray, d: int) -> np.ndarray:
    """Inverse of ``pack``."""
    return np.concatenate([z.real, z.imag], axis=-1)[..., :d]


@dataclass
class Reception:
    """One channel use as seen by the server."""

    Y: np.ndarray          # [N, L] received block
    A: np.ndarray          # [N, K'] effective signatures b_k h_est_k (as known at the server)
    eta: float             # receive scaling: m^H a_k = sqrt(eta) for every device (ZF)
    m: np.ndarray          # [N] unit-norm receive combiner used for the design
    ids: np.ndarray        # device ids that transmitted
    noise_var: float       # per-antenna complex noise variance


@dataclass
class Receiver:
    """Physical layer of one round, shared by every transmission a defence asks for."""

    scenario: Any
    x: np.ndarray                    # [K', L] normalised complex symbols (honest content)
    p_max: np.ndarray                # [K] budgets of *all* scenario devices
    distances: np.ndarray | None     # [K] of all scenario devices (or None)
    node: Any                        # SeedTree node of this round
    attackers: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # local indices
    device_ids: np.ndarray | None = None  # local -> scenario device id (default: identity)
    attack: Any = None
    gref: np.ndarray | None = None   # normalised honest-mean symbols (omniscient attacks)
    jammer: dict[str, float] | None = None
    uses: int = 0

    def __post_init__(self) -> None:
        if self.device_ids is None:
            self.device_ids = np.arange(self.x.shape[0])

    @property
    def rng(self) -> np.random.Generator:
        return self.node.generator("defense")

    def transmit(self, ids: np.ndarray | None = None, tag: str = "main") -> Reception:
        """One AirComp channel use by devices ``ids`` (default: all)."""
        sc = self.scenario
        K, L = self.x.shape
        ids = np.arange(K) if ids is None else np.asarray(ids)
        sub = self.node.child(tag)
        h_act, h_est, beta = sc.sample_channels(sub, 1, self.distances)
        gid = np.asarray(self.device_ids)[ids]
        h_act, h_est, beta = h_act[:, gid], h_est[:, gid].copy(), beta[:, gid]
        byz_local = np.flatnonzero(np.isin(ids, self.attackers))
        if self.attack is not None and len(byz_local):
            h_est[:, byz_local] = self.attack.distort_csi(h_est[:, byz_local])
        lay = sc.design(h_act, h_est, beta, self.p_max[gid])
        b = lay["b"][0]
        T = self.x[ids] * b[:, None]
        if self.attack is not None and len(byz_local):
            T[byz_local] = self.attack.transmit(self.x[ids[byz_local]], b[byz_local], {
                "rng": sub.generator("attack"), "p_max": self.p_max[gid[byz_local]],
                "gref": self.gref})
        Hn = h_act[0].T  # [N, K']
        N = Hn.shape[0]
        Y = Hn @ T + np.sqrt(sc.noise_var) * _cn(sub.generator("noise"), (N, L))
        if self.jammer:
            hj, _ = sc.channel.sample(sub.generator("jammer"), 1, 1,
                                      np.array([self.jammer.get("distance", 1.0)])
                                      if self.distances is not None else None)
            Y += np.sqrt(self.jammer["power"]) * np.outer(hj[0, 0], _cn(sub.generator("jam_sym"), (L,)))
        self.uses += 1
        A = h_est[0].T * b[None, :]
        return Reception(Y, A, float(lay["eta"][0]), lay["m"][0], ids, sc.noise_var)

    def transmit_orthogonal(self, ids: np.ndarray | None = None, tag: str = "orth") -> np.ndarray:
        """Each device in its own resource block at full power with MRC: returns the server's
        per-device estimates of the normalised symbols, ``[K', L]`` (digital baselines)."""
        sc = self.scenario
        K, _L = self.x.shape
        ids = np.arange(K) if ids is None else np.asarray(ids)
        sub = self.node.child(tag)
        h_act, _, _ = sc.sample_channels(sub, 1, self.distances)
        gid = np.asarray(self.device_ids)[ids]
        h = h_act[0, gid]  # [K', N]
        gain = np.sqrt(self.p_max[gid]) * np.linalg.norm(h, axis=1)  # MRC amplitude
        X = self.x[ids].copy()
        byz_local = np.flatnonzero(np.isin(ids, self.attackers))
        if self.attack is not None and len(byz_local):
            b = np.sqrt(self.p_max[gid[byz_local]]).astype(complex)
            X[byz_local] = self.attack.transmit(self.x[ids[byz_local]], b, {
                "rng": sub.generator("attack"), "p_max": self.p_max[gid[byz_local]],
                "gref": self.gref}) / b[:, None]
        noise = _cn(sub.generator("noise"), X.shape) * np.sqrt(sc.noise_var) / gain[:, None]
        self.uses += len(ids)
        return X + noise


def normalise(X: np.ndarray, clip: float | None) -> tuple[np.ndarray, float]:
    """Pack real updates and scale to unit average symbol power. Returns ``(x, nu)`` with
    ``X ~= nu * unpack(x)``. With ``clip`` (rows already clipped to norm <= clip),
    ``nu = clip / sqrt(L)`` so that ``||x_k||^2 <= L``."""
    z = pack(X)
    L = z.shape[-1]
    if clip is not None:
        nu = clip / np.sqrt(L)
    else:
        nu = float(np.sqrt(np.mean(np.abs(z) ** 2))) or 1.0
    return z / nu, nu
