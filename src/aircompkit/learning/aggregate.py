"""Over-the-air aggregation of device update vectors (computation core of OTA-FL).

Each device ``k`` sends a real vector ``x_k`` of length ``d``. Entry ``j`` travels on
sub-channel ``s = j mod S`` (Zhu et al.'s broadband mapping: one parameter per sub-channel
per OFDM symbol, independent fading per sub-channel; ``S = 1`` is a single block-fading
channel per round). For sub-channel ``s`` the physical layer gives each device an arrival
amplitude ``amp[s, k]`` and the estimate's noise variance ``nv_eff / eta[s]``.

With ``normalize=True``, updates are mapped to zero mean / unit variance using statistics
shared by all devices (the server broadcasts or receives them digitally), so that the
power-normalised symbols match the AirComp model; the server undoes this after reception.
"""

from __future__ import annotations

import numpy as np


def aircomp_aggregate(
    x: np.ndarray,
    amp: np.ndarray,
    eta: np.ndarray,
    active: np.ndarray,
    noise_var_eff: float,
    rng: np.random.Generator,
    normalize: bool = True,
    divide_by: float | None = None,
) -> np.ndarray:
    """Estimate ``sum_k x_k / divide_by`` from one round of AirComp.

    ``x [K, d]`` real updates; ``amp [S, K]`` complex arrival amplitudes; ``eta [S]``;
    ``active [S, K]`` bool (``p > 0``). ``divide_by`` defaults to ``K`` (mean over the
    scheduled devices, Zhu et al.'s ``1/|K|`` post-processing).
    """
    K, d = x.shape
    S = amp.shape[0]
    if normalize:
        mu = float(x.mean())
        sd = float(x.std()) or 1.0
        xn = (x - mu) / sd
    else:
        mu, sd, xn = 0.0, 1.0, x
    sub = np.arange(d) % S
    a = np.real(amp)[sub]                       # [d, K]
    est = np.einsum("dk,kd->d", a, xn)
    est += rng.standard_normal(d) * np.sqrt(noise_var_eff / eta[sub])
    n_act = active.sum(axis=1)[sub]             # devices actually received on each entry
    est = est * sd + n_act * mu
    return est / (K if divide_by is None else divide_by)
