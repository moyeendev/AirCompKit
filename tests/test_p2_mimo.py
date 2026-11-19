"""P2 validation: multi-antenna channels, impairments, receive beamformers, pre-processing."""

import numpy as np
import pytest

import aircompkit as ac
from aircompkit.aircomp.preprocess import (
    clip_norm,
    denormalize_sum,
    normalize,
    stochastic_quantize,
)
from aircompkit.channel import AWGN, Impairments, Rayleigh, Rician, exp_corr_sqrt
from aircompkit.metrics import mean_ci
from aircompkit.receiver import (
    EigenCombiner,
    JointMMSE,
    MaxMinCombiner,
    SDRCombiner,
    effective_gains,
    sdr_solve,
    zf_mse,
)

RECEIVERS = ["eigen", "maxmin", "sdr", "joint_mmse"]


def cn(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


FAST_MAXMIN = {"name": "maxmin", "iters": 120, "random_inits": 4}


def mimo(**kw):
    base = {"devices": {"name": "fixed", "K": 8, "p_max": 1.0},
            "channel": {"name": "rayleigh", "n_rx": 4},
            "power_control": "optimal", "receiver": FAST_MAXMIN,
            "noise_var": 0.3, "symbols": "complex", "seed": 11}
    base.update(kw)
    return ac.Scenario(**base)


# ---------------------------------------------------------------- channels
@pytest.mark.parametrize("ch", [Rayleigh(n_rx=4), Rician(n_rx=4, k_factor=3.0),
                                Rayleigh(n_rx=4, corr=0.6), AWGN(n_rx=4)])
def test_mimo_shape_and_per_antenna_power(ch):
    h, beta = ch.sample(np.random.default_rng(0), 20000, 3)
    assert h.shape == (20000, 3, 4) and beta.shape == (20000, 3)
    power = (np.abs(h) ** 2).mean(axis=(0, 1))
    np.testing.assert_allclose(power, 1.0, atol=0.03)


def test_exponential_correlation():
    h, _ = Rayleigh(n_rx=4, corr=0.7).sample(np.random.default_rng(1), 50000, 2)
    x = h.reshape(-1, 4)
    R_hat = x.T @ x.conj() / len(x)
    L = exp_corr_sqrt(4, 0.7)
    np.testing.assert_allclose(R_hat, L @ L.T, atol=0.03)
    np.testing.assert_allclose(np.diag(L @ L.T), 1.0)


def test_single_antenna_unchanged_by_mimo_refactor():
    h, _ = Rayleigh().sample(np.random.default_rng(3), 5, 2)
    assert h.shape == (5, 2, 1)


# ---------------------------------------------------------------- impairments
def test_impairments_off_is_identity():
    h = cn(np.random.default_rng(0), (10, 3, 2))
    ha, he = Impairments().apply(np.random.default_rng(1), h, np.ones((10, 3)))
    assert ha is h and he is h


def test_csi_error_variance_and_phase_error_modulus():
    rng = np.random.default_rng(2)
    h = cn(rng, (40000, 2, 2))
    beta = np.full((40000, 2), 2.0)
    ha, he = Impairments(csi_error=0.05).apply(rng, h, beta)
    ok = abs(np.mean(np.abs(he - h) ** 2) - 0.1) < 0.003
    assert ok and ha is h
    ha, he = Impairments(phase_error_std=0.3).apply(rng, h, beta)
    np.testing.assert_allclose(np.abs(ha), np.abs(h))
    ratio = (ha / h)[..., 0]
    assert abs(np.std(np.angle(ratio)) - 0.3) < 0.01


def test_impairments_keep_channels_paired_and_increase_mse():
    clean = mimo().run(trials=800, dim=0)
    noisy = mimo(impairments={"csi_error": 0.05, "phase_error_std": 0.2}).run(trials=800, dim=0)
    np.testing.assert_array_equal(clean["eta"] > 0, noisy["eta"] > 0)
    assert noisy["mse"].mean() > 1.2 * clean["mse"].mean()
    worse = mimo(impairments={"csi_error": 0.2}).run(trials=800, dim=0)
    assert worse["mse"].mean() > mimo(impairments={"csi_error": 0.05}).run(
        trials=800, dim=0)["mse"].mean()


# ---------------------------------------------------------------- receivers
@pytest.mark.parametrize("name", RECEIVERS)
def test_receivers_unit_norm_and_scalar_case(name):
    rx = ac.build("receiver", name)
    rng = np.random.default_rng(4)
    h = cn(rng, (30, 6, 3))
    m = rx(h, np.ones(6), 0.5)
    assert m.shape == (30, 3)
    np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1.0)
    np.testing.assert_array_equal(rx(cn(rng, (5, 6, 1)), np.ones(6), 0.5), np.ones((5, 1)))


def test_maxmin_matches_brute_force_n2():
    rng = np.random.default_rng(5)
    T, K = 40, 6
    h = cn(rng, (T, K, 2))
    p = rng.uniform(0.5, 2, K)
    got = effective_gains(h, MaxMinCombiner()(h, p, 1.0), p).min(axis=1)
    # m = (cos a, sin a e^{j phi}) covers the unit sphere in C^2 up to a global phase.
    a, phi = np.meshgrid(np.linspace(0, np.pi / 2, 400), np.linspace(0, 2 * np.pi, 800))
    grid = np.stack([np.cos(a).ravel(), (np.sin(a) * np.exp(1j * phi)).ravel()], axis=1)
    g = np.abs(np.einsum("tkn,gn->tgk", h, grid.conj())) ** 2 * p
    brute = g.min(axis=2).max(axis=1)
    assert np.all(got >= brute * (1 - 2e-3))


def test_maxmin_beats_eigen_and_respects_sdr_bound():
    rng = np.random.default_rng(6)
    h = cn(rng, (60, 20, 6))
    p = np.ones(20)
    g_mm = effective_gains(h, MaxMinCombiner()(h, p, 1.0), p).min(axis=1)
    g_eig = effective_gains(h, EigenCombiner()(h, p, 1.0), p).min(axis=1)
    g_sdr = effective_gains(h, SDRCombiner()(h, p, 1.0), p).min(axis=1)
    _, lower, upper = sdr_solve(h)
    assert np.all(g_mm >= g_eig - 1e-12)
    assert np.all(g_mm <= upper * (1 + 1e-9)) and np.all(g_sdr <= upper * (1 + 1e-9))
    assert np.all(lower <= upper * (1 + 1e-9))
    assert np.median((upper - lower) / upper) < 0.15


def test_sdr_bound_tight_when_rank_one_optimal():
    # K = 1: the SDR is tight and the optimum is the matched filter, gain ||h||^2.
    h = cn(np.random.default_rng(7), (20, 1, 4))
    _, lower, upper = sdr_solve(h, iters=200)
    np.testing.assert_allclose(upper, np.sum(np.abs(h[:, 0]) ** 2, axis=1), rtol=1e-6)
    np.testing.assert_allclose(lower, upper, rtol=1e-6)


def test_zf_mse_helper_equals_scenario():
    sc = mimo(power_control="channel_inversion", noise_var=1.0)
    r = sc.run(trials=50, dim=0)
    from aircompkit.core.rng import SeedTree

    h, _ = sc.channel.sample(SeedTree(11).generator("channel"), 50, 8)
    m = MaxMinCombiner(iters=120, random_inits=4)(h, np.ones(8), 1.0)
    np.testing.assert_allclose(r["mse"], zf_mse(h, m, np.ones(8), 1.0), rtol=1e-10)


def test_joint_mmse_never_worse_than_its_zero_forcing_start():
    """JointMMSE starts at the max-min combiner with ZF transmission and never increases the
    MSE; optimal power control on its combiner can only lower it further."""
    zf = mimo(power_control="channel_inversion", receiver={"name": "maxmin", "iters": 150}
              ).run(trials=200, dim=0)
    joint = mimo(receiver="joint_mmse").run(trials=200, dim=0)
    assert np.all(joint["mse"] <= zf["mse"] * (1 + 1e-9))
    assert joint["mse"].mean() < 0.9 * zf["mse"].mean()


def test_joint_mmse_monotone_over_iterations():
    rng = np.random.default_rng(8)
    h = cn(rng, (50, 8, 4))
    p = rng.uniform(0.5, 2, 8)
    _a, b, hist = JointMMSE(iters=20).design(h, p, 0.5)
    assert hist.shape == (21, 50)
    assert np.all(np.diff(hist, axis=0) <= 1e-9 * hist[:-1])
    assert np.all(np.abs(b) ** 2 <= p * (1 + 1e-12))
    # The ZF start equals the max-min zero-forcing MSE sigma^2 / min_k P_k |m^H h_k|^2.
    m0 = MaxMinCombiner(iters=150)(h, p, 0.5)
    np.testing.assert_allclose(hist[0], zf_mse(h, m0, p, 0.5), rtol=1e-9)


# ---------------------------------------------------------------- scenario, MIMO
@pytest.mark.parametrize("symbols", ["real", "complex"])
@pytest.mark.parametrize("rx", [FAST_MAXMIN, "joint_mmse"])
def test_mimo_signal_level_matches_analytical(symbols, rx):
    r = mimo(symbols=symbols, receiver=rx,
             impairments={"csi_error": 0.03, "phase_error_std": 0.1},
             channel={"name": "rician", "n_rx": 4, "k_factor": 2.0, "corr": 0.3},
             ).run(trials=600, dim=24, batch=200)
    m, half = mean_ci(r["mse_empirical"] - r["mse"], z=4.0)
    assert abs(m) <= half, (m, half)


def test_mimo_zero_noise_perfect_csi_exact():
    r = mimo(power_control="channel_inversion", noise_var=0.0).run(trials=20, dim=8)
    assert r["mse"].max() < 1e-18 and r["mse_empirical"].max() < 1e-18


def test_more_antennas_lower_mse():
    m2 = mimo(channel={"name": "rayleigh", "n_rx": 2}).run(trials=300, dim=0)["mse"].mean()
    m8 = mimo(channel={"name": "rayleigh", "n_rx": 8}).run(trials=300, dim=0)["mse"].mean()
    assert m8 < 0.5 * m2


def test_mimo_config_roundtrip():
    sc = mimo(receiver={"name": "maxmin", "iters": 100},
              impairments={"csi_error": 0.1})
    cfg = sc.to_dict()
    assert cfg["receiver"]["name"] == "maxmin" and cfg["receiver"]["iters"] == 100
    assert cfg["impairments"] == {"csi_error": 0.1, "phase_error_std": 0.0}
    assert ac.Scenario.from_dict(cfg).to_dict() == cfg


# ---------------------------------------------------------------- reproduction
def test_yang2020_quick_reproduction():
    res = ac.get("reproduction", "yang2020")(trials=40)
    f4 = res["fig4"]
    g = f4["gamma_db"]
    assert f4["maxmin"][g.index(-5)] == 0.0 and f4["maxmin"][g.index(5)] >= 0.9
    for name in ("maxmin", "sdr", "eigen"):
        assert np.all(np.diff(f4[name]) >= 0)
        assert all(a <= b for a, b in zip(f4[name], f4["sdr_bound"]))
    f5 = res["fig5"]
    for gam in (2, 3, 4):
        assert np.all(np.diff(f5[f"maxmin_{gam}dB"]) >= -0.06)  # monotone in N up to MC noise
    assert f5["mean_abs_deviation"] < 0.15


# ---------------------------------------------------------------- pre-processing
def test_stochastic_quantize_unbiased_and_on_grid():
    rng = np.random.default_rng(9)
    x = rng.uniform(-0.9, 0.9, size=200000)
    q = stochastic_quantize(x, bits=3, clip=1.0, rng=rng)
    delta = 2 / 7
    assert np.allclose(((q + 1) / delta) - np.round((q + 1) / delta), 0)
    m, half = mean_ci(q - x, z=4)
    assert abs(m) <= half
    assert np.mean((q - x) ** 2) <= delta**2 / 4 + 1e-3
    q1 = stochastic_quantize(np.array([5.0, -5.0]), 1, 2.0, rng)
    np.testing.assert_array_equal(q1, [2.0, -2.0])


def test_normalize_roundtrip_and_clip():
    rng = np.random.default_rng(10)
    x = 3 + 2 * rng.standard_normal((5, 100))
    xn, mu, sd = normalize(x)
    assert abs(xn.mean()) < 1e-12 and abs(xn.std() - 1) < 1e-12
    np.testing.assert_allclose(denormalize_sum(xn.sum(axis=0), mu, sd, 5), x.sum(axis=0))
    c = clip_norm(x, 1.0)
    assert np.all(np.linalg.norm(c, axis=-1) <= 1 + 1e-12)
