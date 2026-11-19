"""Scenario: one object (and one config file) that fully specifies a run (FEATURES C1, C5, A8).

P1 covers *computation mode*: Monte Carlo AirComp of random unit-power vectors over a
single-antenna multiple-access channel. Later phases add the receiver, adversary, defence
and learning task to the same object.

Random streams (see ``SeedTree``): ``devices`` (placement),
``channel`` (fading, shadowing), ``data`` (symbols), ``noise``. Changing the power-control
scheme or the symbol dimension therefore leaves the channel realisations unchanged, so
schemes are compared on identical channels.
"""

from __future__ import annotations

import dataclasses
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aircompkit.core.registry import build, spec_of
from aircompkit.core.result import Result
from aircompkit.core.rng import SeedTree

_SYMBOLS = ("real", "complex")


@dataclass
class Scenario:
    devices: Any
    channel: Any = "rayleigh"
    power_control: Any = "optimal"
    receiver: Any = "maxmin"
    impairments: Any = None
    target: Any = "sum"
    noise_var: float = 1.0
    symbols: str = "real"
    seed: int = 0

    def __post_init__(self) -> None:
        from aircompkit.channel import as_impairments

        self.devices = build("devices", self.devices)
        self.channel = build("channel", self.channel)
        self.power_control = build("power_control", self.power_control)
        self.receiver = build("receiver", self.receiver)
        self.impairments = as_impairments(self.impairments)
        self.target = build("aircomp", self.target)
        if self.symbols not in _SYMBOLS:
            raise ValueError(f"symbols must be one of {_SYMBOLS}")
        if self.noise_var < 0:
            raise ValueError("noise_var must be >= 0")

    # ------------------------------------------------------------------ config I/O
    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": spec_of("devices", self.devices),
            "channel": spec_of("channel", self.channel),
            "power_control": spec_of("power_control", self.power_control),
            "receiver": spec_of("receiver", self.receiver),
            "impairments": dataclasses.asdict(self.impairments),
            "target": spec_of("aircomp", self.target),
            "noise_var": float(self.noise_var),
            "symbols": self.symbols,
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Scenario:
        known = {"devices", "channel", "power_control", "receiver", "impairments", "target",
                 "noise_var", "symbols", "seed"}
        extra = set(cfg) - known - {"run", "learning"}
        if extra:
            raise KeyError(f"unknown scenario keys: {sorted(extra)}")
        return cls(**{k: v for k, v in cfg.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scenario:
        return cls.from_dict(load_yaml(path))

    def to_yaml(self, path: str | Path) -> None:
        import yaml

        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def replace(self, **changes: Any) -> Scenario:
        """Copy with some fields changed (handy for sweeps)."""
        cfg = self.to_dict()
        cfg.update(changes)
        return Scenario.from_dict(cfg)

    # ------------------------------------------------------------------ simulation
    @property
    def noise_var_eff(self) -> float:
        return self.noise_var / 2 if self.symbols == "real" else self.noise_var

    def sample_channels(
        self, tree: SeedTree, trials: int, distances: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw ``(h_actual, h_estimated, beta)`` for all ``K`` devices from ``tree``'s
        ``channel`` and ``impairments`` streams. Shapes ``[T, K, N]``, ``[T, K, N]``, ``[T, K]``."""
        h, beta = self.channel.sample(tree.generator("channel"), trials, self.devices.K, distances)
        h_act, h_est = self.impairments.apply(tree.generator("impairments"), h, beta)
        return h_act, h_est, beta

    def design(
        self, h_act: np.ndarray, h_est: np.ndarray, beta: np.ndarray, p_max: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Receiver + power control for the devices whose channels are given (any subset).

        Returns ``m [T, N]``, ``c_act``/``c_est`` (effective scalar channels ``m^H h``),
        ``p``, ``b`` (transmit coefficients), ``eta`` and ``amp = c_act b / sqrt(eta)``, the
        per-device amplitude with which a unit symbol arrives in the estimate (1 is perfect
        alignment, 0 a silent device).
        """
        nv = self.noise_var_eff
        m = self.receiver(h_est, p_max, nv)  # [T, N], unit norm
        c_est = np.einsum("tkn,tn->tk", h_est, m.conj())
        c_act = np.einsum("tkn,tn->tk", h_act, m.conj())
        g = np.abs(c_est) ** 2
        p, eta = self.power_control(g, beta, p_max, nv)
        if np.any(p < 0) or (
            getattr(self.power_control, "peak_constrained", True)
            and np.any(p > p_max[None, :] * (1 + 1e-9))
        ):
            raise RuntimeError(f"{type(self.power_control).__name__} violated power budgets")
        with np.errstate(divide="ignore", invalid="ignore"):
            phase = np.where(g > 0, np.conj(c_est) / np.sqrt(g), 0.0)
        b = np.sqrt(p) * phase
        amp = c_act * b / np.sqrt(eta)[:, None]
        return {"m": m, "c_act": c_act, "c_est": c_est, "p": p, "b": b, "eta": eta, "amp": amp}

    def run(self, trials: int = 1000, dim: int = 16, batch: int = 1024) -> Result:
        """Run ``trials`` independent channel realisations.

        For each trial, the analytical conditional MSE is computed from the channel and
        transmit coefficients. If ``dim > 0``, the AirComp of ``dim``-dimensional random
        symbol vectors is also simulated at signal level and its empirical MSE recorded.
        ``batch`` bounds memory (``batch * K * dim`` complex numbers); results depend on
        ``(seed, trials, dim, batch)``.

        Returned arrays (per trial unless noted): ``mse`` (analytical), ``mse_empirical``,
        ``eta``, ``n_active`` (devices with ``p_k > 0``), ``power`` (``[trials, K]``).
        """
        from aircompkit.metrics import mse_given_channel

        if trials < 1 or dim < 0 or batch < 1:
            raise ValueError("need trials >= 1, dim >= 0, batch >= 1")
        tree = SeedTree(self.seed)
        K = self.devices.K
        p_max = self.devices.power_budgets()
        dist = self.devices.sample_distances(tree.generator("devices"))
        h_act, h_est, beta = self.sample_channels(tree, trials, dist)
        lay = self.design(h_act, h_est, beta, p_max)
        p, eta, b, m = lay["p"], lay["eta"], lay["b"], lay["m"]

        real = self.symbols == "real"
        s = self.target.scale(K)
        # Exact conditional MSE given the *actual* channel: m^H n ~ CN(0, sigma^2) as ||m|| = 1.
        mse = s**2 * mse_given_channel(lay["c_act"], b, eta, self.noise_var_eff, real)

        arrays: dict[str, np.ndarray] = {
            "mse": mse,
            "eta": eta,
            "n_active": (p > 0).sum(axis=1).astype(float),
            "power": p,
        }
        if dim > 0:
            arrays["mse_empirical"] = s**2 * self._simulate(
                h_act * b[..., None], m, eta, dim, batch,
                tree.generator("data"), tree.generator("noise"),
            )

        meta = {
            "trials": trials,
            "dim": dim,
            "batch": batch,
            "aircompkit": _version(),
            "numpy": np.__version__,
            "python": platform.python_version(),
        }
        return Result(config=self.to_dict(), arrays=arrays, meta=meta)

    def _simulate(self, hb, m, eta, dim, batch, rng_data, rng_noise) -> np.ndarray:
        """Signal-level AirComp at the antenna array; returns per-trial empirical MSE of the sum.

        ``hb [T, K, N]`` holds ``h_k b_k`` with the actual channel, ``m [T, N]`` the combiner.
        """
        T, K, N = hb.shape
        real = self.symbols == "real"
        out = np.empty(T)
        for lo in range(0, T, batch):
            hi = min(lo + batch, T)
            n = hi - lo
            if real:
                x = rng_data.standard_normal((n, K, dim))
            else:
                z = rng_data.standard_normal((n, K, dim, 2))
                x = (z[..., 0] + 1j * z[..., 1]) / np.sqrt(2)
            zn = rng_noise.standard_normal((n, N, dim, 2))
            noise = np.sqrt(self.noise_var / 2) * (zn[..., 0] + 1j * zn[..., 1])
            y = np.einsum("tkn,tkd->tnd", hb[lo:hi], x) + noise
            r = np.einsum("tn,tnd->td", m[lo:hi].conj(), y)
            f_hat = r / np.sqrt(eta[lo:hi])[:, None]
            if real:
                f_hat = f_hat.real
            f = x.sum(axis=1)
            out[lo:hi] = np.mean(np.abs(f_hat - f) ** 2, axis=1)
        return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise TypeError(f"{path}: expected a mapping at top level")
    return cfg


def _version() -> str:
    from aircompkit import __version__

    return __version__
