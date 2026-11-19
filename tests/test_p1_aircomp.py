"""P1 validation: simulated AirComp MSE agrees with closed forms within Monte Carlo error."""

import numpy as np
import pytest

import aircompkit as ac
from aircompkit.aircomp.analytic import (
    channel_inversion_awgn_mse,
    truncated_inversion_rayleigh_mse,
)
from aircompkit.aircomp.power_control import (
    ChannelInversion,
    FullPower,
    OptimalPowerControl,
    TruncatedInversion,
)
from aircompkit.channel import PathLoss, Rayleigh, Rician
from aircompkit.core.result import Result
from aircompkit.metrics import mean_ci, mse_given_channel

SCHEMES = ["channel_inversion", "truncated_inversion", "full_power", "optimal"]


def within_ci(sample, expected, k=4.0):
    m, half = mean_ci(sample, z=k)
    return abs(m - expected) <= half + 1e-12, (m, half, expected)


def scenario(**kw):
    base = {"devices": {"name": "fixed", "K": 10, "p_max": 2.0}, "channel": "rayleigh",
            "power_control": "optimal", "noise_var": 0.5, "symbols": "real", "seed": 3}
    base.update(kw)
    return ac.Scenario(**base)


# ---------------------------------------------------------------- channels
def test_pathloss_formula_and_clamp():
    pl = PathLoss(exponent=3.0, ref_distance=2.0, ref_gain_db=-30.0)
    beta = pl.gains(np.array([1.0, 2.0, 20.0]), np.random.default_rng(0), 4, 3)
    expected = 1e-3 * np.array([1.0, 1.0, 10.0**-3])
    np.testing.assert_allclose(beta, np.broadcast_to(expected, (4, 3)))


def test_shadowing_is_lognormal_in_db():
    pl = PathLoss(exponent=0.0, shadowing_db=8.0)
    beta = pl.gains(None, np.random.default_rng(1), 20000, 5)
    db = 10 * np.log10(beta)
    assert abs(db.mean()) < 0.2 and abs(db.std() - 8.0) < 0.2


@pytest.mark.parametrize("ch", [Rayleigh(), Rician(k_factor=0.0), Rician(k_factor=5.0)])
def test_fading_unit_mean_power(ch):
    h, _ = ch.sample(np.random.default_rng(2), 40000, 4)
    ok, info = within_ci(np.abs(h.ravel()) ** 2, 1.0)
    assert ok, info


@pytest.mark.parametrize("k", [0.0, 1.0, 10.0])
def test_rician_fourth_moment(k):
    # E|h|^4 = (k^2 + 4k + 2) / (k + 1)^2 for unit-power Rician fading.
    h, _ = Rician(k_factor=k).sample(np.random.default_rng(4), 100000, 2)
    ok, info = within_ci(np.abs(h.ravel()) ** 4, (k * k + 4 * k + 2) / (k + 1) ** 2)
    assert ok, info


def test_uniform_disc_in_annulus():
    d = ac.build("devices", {"name": "uniform_disc", "K": 500, "r_min": 5, "r_max": 50})
    r = d.sample_distances(np.random.default_rng(0))
    assert r.min() >= 5 and r.max() <= 50
    # area-uniform: E[r^2] = (r_min^2 + r_max^2) / 2
    assert abs((r**2).mean() / ((25 + 2500) / 2) - 1) < 0.08


# ---------------------------------------------------------------- power control
@pytest.mark.parametrize("name", SCHEMES)
def test_power_budgets_respected(name):
    rng = np.random.default_rng(5)
    g = rng.exponential(size=(500, 8))
    p_max = rng.uniform(0.5, 2.0, size=8)
    p, eta = ac.get("power_control", name)()(g, np.ones_like(g), p_max, 0.3)
    assert p.shape == g.shape and eta.shape == (500,)
    assert np.all(p >= 0) and np.all(p <= p_max * (1 + 1e-12))
    assert np.all(eta > 0)


def _mse(pc, g, p_max, nv):
    p, eta = pc(g, np.ones_like(g), p_max, nv)
    h = np.sqrt(g).astype(complex)
    return mse_given_channel(h, np.sqrt(p), eta, nv, real=True)


def test_optimal_matches_brute_force_and_dominates():
    rng = np.random.default_rng(6)
    T, K, nv = 300, 6, 0.4
    g = rng.exponential(size=(T, K))
    p_max = rng.uniform(0.5, 2.0, size=K)
    opt = _mse(OptimalPowerControl(), g, p_max, nv)

    # Brute force over eta on a fine log grid, with the optimal per-device rule for each eta.
    G = p_max * g
    etas = np.logspace(-4, 2, 4000)
    a = np.minimum(np.sqrt(G[:, :, None] / etas), 1.0)
    brute = ((a - 1) ** 2).sum(axis=1) + nv / etas
    assert np.all(opt <= brute.min(axis=1) + 1e-9)
    assert np.all(opt >= brute.min(axis=1) - 1e-3 * brute.min(axis=1))

    for other in (ChannelInversion(), TruncatedInversion(0.2), FullPower()):
        assert np.all(opt <= _mse(other, g, p_max, nv) + 1e-9)


def test_full_power_eta_is_optimal_for_fixed_power():
    rng = np.random.default_rng(7)
    g = rng.exponential(size=(50, 5))
    p_max = np.ones(5)
    p, eta = FullPower()(g, np.ones_like(g), p_max, 0.2)
    h = np.sqrt(g).astype(complex)
    base = mse_given_channel(h, np.sqrt(p), eta, 0.2, real=True)
    for f in (0.9, 1.1):
        assert np.all(base <= mse_given_channel(h, np.sqrt(p), eta * f, 0.2, real=True))


# ---------------------------------------------------------------- closed forms
@pytest.mark.parametrize("symbols", ["real", "complex"])
def test_awgn_channel_inversion_closed_form(symbols):
    sc = ac.Scenario(
        devices={"name": "fixed", "K": 6, "p_max": [1, 2, 1, 3, 1, 1],
                 "distances": [1, 2, 3, 4, 5, 6]},
        channel={"name": "awgn", "pathloss": {"exponent": 2.0}},
        power_control="channel_inversion", noise_var=0.8, symbols=symbols, seed=0,
    )
    r = sc.run(trials=3000, dim=8)
    nv = 0.4 if symbols == "real" else 0.8
    expected = channel_inversion_awgn_mse(1.0 / np.arange(1, 7) ** 2, [1, 2, 1, 3, 1, 1], nv)
    np.testing.assert_allclose(r["mse"], expected)
    ok, info = within_ci(r["mse_empirical"], expected)
    assert ok, info


@pytest.mark.parametrize("relative", [True, False])
def test_truncated_inversion_rayleigh_closed_form(relative):
    sc = scenario(
        devices={"name": "fixed", "K": 20, "p_max": 1.5},
        power_control={"name": "truncated_inversion", "threshold": 0.15, "relative": relative},
        noise_var=1.0, symbols="complex",
    )
    r = sc.run(trials=20000, dim=0)
    expected = truncated_inversion_rayleigh_mse([1.0] * 20, 1.5, 0.15, 1.0, relative)
    ok, info = within_ci(r["mse"], expected)
    assert ok, info


def test_truncated_closed_form_with_pathloss():
    dist = list(np.linspace(10, 60, 8))
    sc = scenario(
        devices={"name": "fixed", "K": 8, "p_max": 1.0, "distances": dist},
        channel={"name": "rayleigh", "pathloss": {"exponent": 3.0, "ref_gain_db": -20}},
        power_control={"name": "truncated_inversion", "threshold": 0.3},
        noise_var=1e-8, symbols="real",
    )
    r = sc.run(trials=20000, dim=0)
    beta = 1e-2 * np.asarray(dist) ** -3.0
    expected = truncated_inversion_rayleigh_mse(beta, 1.0, 0.3, 0.5e-8)
    ok, info = within_ci(r["mse"], expected)
    assert ok, info


@pytest.mark.parametrize("name", SCHEMES)
@pytest.mark.parametrize("symbols", ["real", "complex"])
def test_signal_level_matches_analytical(name, symbols):
    """Empirical MSE of the simulated waveform equals the conditional analytical MSE."""
    pc = {"name": name, "threshold": 0.2} if name == "truncated_inversion" else name
    r = scenario(power_control=pc, symbols=symbols,
                 channel={"name": "rician", "k_factor": 2.0}).run(trials=1500, dim=32, batch=256)
    diff = r["mse_empirical"] - r["mse"]
    m, half = mean_ci(diff, z=4.0)
    assert abs(m) <= half, (m, half)


def test_mean_target_scales_by_k_squared():
    a = scenario(target="sum").run(trials=200, dim=4)
    b = scenario(target="mean").run(trials=200, dim=4)
    np.testing.assert_allclose(b["mse"], a["mse"] / 100)
    np.testing.assert_allclose(b["mse_empirical"], a["mse_empirical"] / 100)


def test_zero_noise_perfect_csi_is_exact():
    r = scenario(power_control="channel_inversion", noise_var=0.0).run(trials=50, dim=16)
    assert np.max(r["mse_empirical"]) < 1e-20
    assert np.max(r["mse"]) < 1e-20


# ---------------------------------------------------------------- scenario plumbing
def test_paired_channels_across_schemes():
    a = scenario(power_control="optimal").run(trials=100, dim=0)
    b = scenario(power_control="full_power").run(trials=100, dim=0)
    assert np.all(a["mse"] <= b["mse"] + 1e-12)  # same channels, so pointwise dominance
    assert not np.allclose(a["mse"], b["mse"])


def test_deterministic_given_seed():
    a = scenario().run(trials=64, dim=8)
    b = scenario().run(trials=64, dim=8)
    for k in a.arrays:
        np.testing.assert_array_equal(a[k], b[k])
    c = scenario(seed=4).run(trials=64, dim=8)
    assert not np.allclose(a["mse"], c["mse"])


def test_config_roundtrip_and_yaml(tmp_path):
    sc = scenario(channel={"name": "rician", "k_factor": 4.0, "pathloss": {"exponent": 2.5}},
                  power_control={"name": "truncated_inversion", "threshold": 0.3})
    cfg = sc.to_dict()
    assert cfg["channel"] == {"name": "rician", "k_factor": 4.0, "n_rx": 1, "corr": 0.0,
                              "pathloss": {"exponent": 2.5, "ref_distance": 1.0,
                                           "ref_gain_db": 0.0, "shadowing_db": 0.0}}
    assert ac.Scenario.from_dict(cfg).to_dict() == cfg
    sc.to_yaml(tmp_path / "s.yaml")
    assert ac.Scenario.from_yaml(tmp_path / "s.yaml").to_dict() == cfg
    with pytest.raises(KeyError, match="unknown scenario keys"):
        ac.Scenario.from_dict({**cfg, "typo": 1})


def test_result_save_load(tmp_path):
    r = scenario().run(trials=32, dim=4)
    r.save(tmp_path / "out" / "res")
    back = Result.load(tmp_path / "out" / "res.npz")
    assert back.config == r.config and back.meta["trials"] == 32
    for k in r.arrays:
        np.testing.assert_array_equal(back[k], r[k])
    assert set(r.summary()) == {"mse", "mse_empirical", "eta", "n_active"}


def test_cli_run(tmp_path, capsys):
    from aircompkit.cli import main

    cfg = tmp_path / "c.yaml"
    scenario().to_yaml(cfg)
    assert main(["run", str(cfg), "--trials", "20", "--dim", "2",
                 "--out", str(tmp_path / "r")]) == 0
    out = capsys.readouterr().out
    assert "mse_empirical" in out and (tmp_path / "r.npz").exists()


@pytest.mark.parametrize("name", ["rayleigh_optimal.yaml", "mimo_maxmin_csi.yaml"])
def test_shipped_config_runs(name):
    from pathlib import Path

    from aircompkit.core.scenario import load_yaml

    cfg = load_yaml(Path(__file__).parents[1] / "configs" / name)
    cfg.pop("run")
    r = ac.Scenario.from_dict(cfg).run(trials=50, dim=4)
    assert np.isfinite(r["mse"]).all()
