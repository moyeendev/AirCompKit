"""Threat model and attack library (FEATURES §5).

A ``ThreatModel`` says *who* attacks (a fixed set of ``n_attackers`` devices or a
``fraction``), *what they know* (``knowledge``) and *how much power* they may use
(``max_boost_db``), plus an optional external ``jammer``. Attacks declare what they
``need``; building a threat model whose attack needs more than it grants is an error (T5),
so results always state their assumptions.

Attacks act on the array-level path (``aircompkit.learning.array``): honest devices send
``t_k = b_k x_k`` with unit-power symbols ``x_k`` and AirComp pre-equaliser ``b_k``; an
attacker replaces its row via ``transmit(x_own, b_own, ctx)``, may distort the server's CSI
of its own channel (``distort_csi``), or poison its local labels (``poisons_labels``).
``boost_db`` is the attacker's power over what the power-control rule prescribes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aircompkit.channel import _cn
from aircompkit.core.registry import build, register

KNOWLEDGE = frozenset({"own_data", "own_csi", "honest_mean", "receiver"})


@dataclass
class Attack:
    """Base class for attacks: subclass it, override ``transmit`` (and optionally ``distort_csi``,
    ``flip_labels``), set ``needs``, and register with ``@register("attack", name)``."""

    boost_db: float = 0.0
    needs = frozenset({"own_data"})
    poisons_labels = False

    @property
    def gain(self) -> float:
        return float(10 ** (self.boost_db / 20))

    def transmit(self, x, b, ctx):
        return self.gain * x * b[:, None]

    def distort_csi(self, h_est):
        return h_est

    def flip_labels(self, y, n_classes):
        return y


@register("attack", "gaussian")
@dataclass
class GaussianNoise(Attack):
    """Transmit ``CN(0, P_k)`` noise at the boosted power limit (jamming from inside)."""

    needs = frozenset()

    def transmit(self, x, b, ctx):
        P = np.asarray(ctx["p_max"], float)[:, None]
        return self.gain * np.sqrt(P) * _cn(ctx["rng"], x.shape)


@register("attack", "sign_flip")
@dataclass
class SignFlip(Attack):
    """Send ``-x_k`` (own update reversed) with power boost; independent across attackers."""

    def transmit(self, x, b, ctx):
        return -self.gain * x * b[:, None]


@register("attack", "scaled")
@dataclass
class Scaled(Attack):
    """Send the own honest update with a power boost (pure power-control violation)."""


@register("attack", "collude")
@dataclass
class Collude(Attack):
    """Omniscient colluders: all send the same unit-power vector opposite to the honest mean
    (rank-1 attack, the hardest for spatial detection)."""

    needs = frozenset({"honest_mean"})

    def transmit(self, x, b, ctx):
        g = ctx["gref"]
        v = -g / np.sqrt(max(float(np.mean(np.abs(g) ** 2)), 1e-300))
        return self.gain * v[None, :] * b[:, None]


@register("attack", "misalign")
@dataclass
class Misalign(Attack):
    """Rotate the own (honest) signal by ``phase`` radians: destructive superposition without
    changing the payload; ``phase = pi`` equals ``sign_flip``."""

    phase: float = float(np.pi / 2)

    def transmit(self, x, b, ctx):
        return self.gain * np.exp(1j * self.phase) * x * b[:, None]


@register("attack", "label_flip")
@dataclass
class LabelFlip(Attack):
    """Data poisoning: train on labels ``y -> C - 1 - y``, then transmit like an honest device."""

    poisons_labels = True

    def flip_labels(self, y, n_classes):
        return n_classes - 1 - y


@register("attack", "csi_spoof")
@dataclass
class CSISpoof(Attack):
    """Pilot manipulation: the attacker makes its channel look ``factor`` times weaker. With
    channel inversion the weakest device sets the receive scaling, so every device's SNR drops,
    and the attacker's own row arrives ``1/factor`` times too strong."""

    factor: float = 0.1
    needs = frozenset({"own_data", "own_csi"})

    def distort_csi(self, h_est):
        return self.factor * h_est


@dataclass
class ThreatModel:
    attack: Any = "sign_flip"
    n_attackers: int = 0
    fraction: float | None = None
    knowledge: list[str] = field(default_factory=lambda: ["own_data", "own_csi"])
    max_boost_db: float | None = None
    jammer: dict[str, float] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        self.attack = build("attack", self.attack)
        unknown = set(self.knowledge) - KNOWLEDGE
        if unknown:
            raise ValueError(f"unknown knowledge flags {sorted(unknown)}; allowed {sorted(KNOWLEDGE)}")
        missing = set(self.attack.needs) - set(self.knowledge)
        if missing:
            raise ValueError(
                f"attack {type(self.attack).__name__} needs knowledge {sorted(missing)} that the "
                "threat model does not grant (add it to 'knowledge')")
        if self.max_boost_db is not None and self.attack.boost_db > self.max_boost_db:
            raise ValueError(f"boost_db {self.attack.boost_db} exceeds max_boost_db {self.max_boost_db}")
        if self.fraction is not None and not 0 <= self.fraction < 1:
            raise ValueError("fraction must be in [0, 1)")
        if self.jammer is not None and self.jammer.get("power", 0) < 0:
            raise ValueError("jammer power must be >= 0")

    def attackers(self, K: int) -> np.ndarray:
        n = round(self.fraction * K) if self.fraction is not None else self.n_attackers
        if not 0 <= n < K:
            raise ValueError(f"need 0 <= attackers < K, got {n} of {K}")
        rng = np.random.default_rng(self.seed)
        return np.sort(rng.choice(K, n, replace=False)) if n else np.zeros(0, int)

    def to_dict(self) -> dict[str, Any]:
        from aircompkit.core.registry import spec_of

        return {"attack": spec_of("attack", self.attack), "n_attackers": self.n_attackers,
                "fraction": self.fraction, "knowledge": list(self.knowledge),
                "max_boost_db": self.max_boost_db, "jammer": self.jammer, "seed": self.seed}


def as_threat(x: ThreatModel | dict[str, Any] | None) -> ThreatModel | None:
    if x is None or isinstance(x, ThreatModel):
        return x
    return ThreatModel(**x)
