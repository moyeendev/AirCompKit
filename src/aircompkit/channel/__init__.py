"""Channel models and impairments (FEATURES H1–H4, H6, H7).

Block-fading, complex-baseband uplink from ``K`` single-antenna devices to a server with
``n_rx`` antennas (a uniform linear array with half-wavelength spacing). ``sample`` returns

* ``h``: complex coefficients, shape ``[trials, K, n_rx]``;
* ``beta``: large-scale gains, shape ``[trials, K]`` (constant across trials unless
  shadowing is enabled).

Every model is normalised so that ``E|h_{k,n}|^2 = beta_k`` on every antenna. Optional
exponential spatial correlation ``R_ij = corr^|i-j|`` applies to the scattered part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aircompkit.core.registry import register


@dataclass
class PathLoss:
    """Log-distance path loss with optional log-normal shadowing (FEATURES H3).

    ``beta(d) = 10^(ref_gain_db/10) * (max(d, ref_distance) / ref_distance)^(-exponent)``,
    multiplied by ``10^(S/10)`` with ``S ~ N(0, shadowing_db^2)`` drawn per trial and device.
    """

    exponent: float = 3.0
    ref_distance: float = 1.0
    ref_gain_db: float = 0.0
    shadowing_db: float = 0.0

    def __post_init__(self) -> None:
        if self.ref_distance <= 0 or self.exponent < 0 or self.shadowing_db < 0:
            raise ValueError("invalid path-loss parameters")

    def gains(
        self, distances: np.ndarray | None, rng: np.random.Generator, trials: int, K: int
    ) -> np.ndarray:
        if distances is None:
            base = np.full(K, 10 ** (self.ref_gain_db / 10))
        else:
            d = np.maximum(np.asarray(distances, float), self.ref_distance)
            base = 10 ** (self.ref_gain_db / 10) * (d / self.ref_distance) ** (-self.exponent)
        beta = np.broadcast_to(base, (trials, K)).copy()
        if self.shadowing_db > 0:
            beta *= 10 ** (rng.normal(0.0, self.shadowing_db, size=(trials, K)) / 10)
        return beta


def _as_pathloss(pl: PathLoss | dict[str, Any] | None) -> PathLoss | None:
    if pl is None or isinstance(pl, PathLoss):
        return pl
    return PathLoss(**pl)


def _cn(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Circularly-symmetric complex Gaussian, unit variance."""
    z = rng.standard_normal(shape + (2,))
    return (z[..., 0] + 1j * z[..., 1]) / np.sqrt(2.0)


def steering(n_rx: int, angles: np.ndarray) -> np.ndarray:
    """ULA steering vectors ``a_n(theta) = exp(j pi n sin theta)``; shape ``angles.shape + (n_rx,)``."""
    n = np.arange(n_rx)
    return np.exp(1j * np.pi * np.sin(angles)[..., None] * n)


def exp_corr_sqrt(n_rx: int, corr: float) -> np.ndarray:
    """Lower Cholesky factor ``L`` of the exponential correlation matrix ``R_ij = corr^|i-j|``."""
    idx = np.arange(n_rx)
    R = corr ** np.abs(idx[:, None] - idx[None, :])
    return np.linalg.cholesky(R)


@dataclass
class _Channel:
    pathloss: PathLoss | dict[str, Any] | None = None
    n_rx: int = 1
    corr: float = 0.0

    def __post_init__(self) -> None:
        self.pathloss = _as_pathloss(self.pathloss)
        if self.n_rx < 1:
            raise ValueError("n_rx must be >= 1")
        if not 0 <= self.corr < 1:
            raise ValueError("corr must be in [0, 1)")

    def _beta(self, rng, trials, K, distances):
        if self.pathloss is None:
            return np.ones((trials, K))
        assert isinstance(self.pathloss, PathLoss)
        return self.pathloss.gains(distances, rng, trials, K)

    def _scatter(self, rng, trials, K):
        w = _cn(rng, (trials, K, self.n_rx))
        if self.corr > 0 and self.n_rx > 1:
            w = w @ exp_corr_sqrt(self.n_rx, self.corr).T
        return w

    def _los(self, rng, trials, K):
        """Unit-modulus LoS component: random phase, plus a random AoA when ``n_rx > 1``."""
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(trials, K)))[..., None]
        if self.n_rx == 1:
            return phase
        theta = rng.uniform(-np.pi / 2, np.pi / 2, size=(trials, K))
        return phase * steering(self.n_rx, theta)

    def small_scale(self, rng: np.random.Generator, trials: int, K: int) -> np.ndarray:
        raise NotImplementedError

    def sample(
        self, rng: np.random.Generator, trials: int, K: int, distances: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        beta = self._beta(rng, trials, K, distances)
        return np.sqrt(beta)[..., None] * self.small_scale(rng, trials, K), beta


@register("channel", "awgn")
@dataclass
class AWGN(_Channel):
    """No fading. Single antenna: ``h_k = sqrt(beta_k)``. Multi-antenna: a pure LoS ray
    ``sqrt(beta_k) a(theta_k)`` with a random angle of arrival per trial and device."""

    def small_scale(self, rng, trials, K):
        if self.n_rx == 1:
            return np.ones((trials, K, 1), dtype=complex)
        theta = rng.uniform(-np.pi / 2, np.pi / 2, size=(trials, K))
        return steering(self.n_rx, theta)


@register("channel", "rayleigh")
@dataclass
class Rayleigh(_Channel):
    """i.i.d. Rayleigh block fading: ``h_k ~ CN(0, beta_k R)``, independent across trials."""

    def small_scale(self, rng, trials, K):
        return self._scatter(rng, trials, K)


@register("channel", "rician")
@dataclass
class Rician(_Channel):
    """Rician fading with K-factor ``k_factor`` (linear, LoS-to-scatter power ratio).

    ``h_k = sqrt(beta_k) (sqrt(k/(k+1)) e^{j phi} a(theta) + sqrt(1/(k+1)) w)`` with
    ``w ~ CN(0, R)``. ``k_factor = 0`` gives Rayleigh.
    """

    k_factor: float = field(default=3.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.k_factor < 0:
            raise ValueError("k_factor must be >= 0")

    def small_scale(self, rng, trials, K):
        k = self.k_factor
        los = self._los(rng, trials, K)
        return np.sqrt(k / (k + 1)) * los + np.sqrt(1 / (k + 1)) * self._scatter(rng, trials, K)


@dataclass
class Impairments:
    """Imperfect CSI and residual phase errors (FEATURES H6, H7).

    * ``csi_error``: relative estimation-error variance ``eps^2``. The server and devices
      design beamformers and power control from ``h_est = h + e`` with
      ``e ~ CN(0, eps^2 beta_k I)``.
    * ``phase_error_std``: residual synchronisation phase error (radians). At transmission,
      device ``k`` sees ``h_k e^{j theta_k}`` with ``theta_k ~ N(0, std^2)`` per trial.

    Both are drawn from their own random stream, so channel realisations stay paired with
    an impairment-free run.
    """

    csi_error: float = 0.0
    phase_error_std: float = 0.0

    def __post_init__(self) -> None:
        if self.csi_error < 0 or self.phase_error_std < 0:
            raise ValueError("impairment levels must be >= 0")

    def apply(
        self, rng: np.random.Generator, h: np.ndarray, beta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(h_actual, h_est)``."""
        T, K, N = h.shape
        h_est = h
        if self.csi_error > 0:
            e = np.sqrt(self.csi_error * beta)[..., None] * _cn(rng, (T, K, N))
            h_est = h + e
        h_act = h
        if self.phase_error_std > 0:
            theta = rng.normal(0.0, self.phase_error_std, size=(T, K))
            h_act = h * np.exp(1j * theta)[..., None]
        return h_act, h_est


def as_impairments(x: Impairments | dict[str, Any] | None) -> Impairments:
    if x is None:
        return Impairments()
    if isinstance(x, Impairments):
        return x
    return Impairments(**x)
