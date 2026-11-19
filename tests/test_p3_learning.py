"""P3 validation: learning stack, OTA-FL loop, gradient bank, average-power truncation."""

import gzip

import numpy as np
import pytest

import aircompkit as ac
from aircompkit.aircomp.power_control import TruncatedInversion, exp1
from aircompkit.learning import (
    MLP,
    Dirichlet,
    FederatedConfig,
    GradientBank,
    Shards,
    aircomp_aggregate,
    synthetic,
    train,
)
from aircompkit.learning.data import mnist

DATA = synthetic(n_train=3000, n_test=1000, dim=12, n_classes=4, seed=1)


def sc(**kw):
    base = {"devices": {"name": "fixed", "K": 10, "p_max": 1.0},
            "power_control": "channel_inversion", "noise_var": 0.0, "symbols": "real", "seed": 2}
    base.update(kw)
    return ac.Scenario(**base)


def cfg(**kw):
    base = {"dataset": DATA, "rounds": 30, "eval_every": 10, "lr": 0.2,
            "model": {"name": "mlp", "hidden": [16]}}
    base.update(kw)
    return FederatedConfig(**base)


# ---------------------------------------------------------------- special functions
@pytest.mark.parametrize("x,ref", [(0.1, 1.8229239584193906), (1.0, 0.21938393439552029),
                                   (2.0, 0.04890051070806112), (20.0, 9.8355252906498815e-11)])
def test_exp1(x, ref):
    assert abs(exp1(x) / ref - 1) < 1e-13


def test_average_power_truncation_meets_budget():
    rng = np.random.default_rng(0)
    beta = np.full((200000, 3), [1.0, 0.5, 2.0])
    g = rng.exponential(beta)
    p_max = np.array([1.0, 2.0, 0.5])
    p, eta = TruncatedInversion(0.2, budget="average")(g, beta, p_max, 1.0)
    # the binding device (min P_k beta_k) spends exactly its average budget
    k = int(np.argmin(p_max * beta[0]))
    ok = abs(p[:, k].mean() / p_max[k] - 1) < 0.02
    assert ok and np.all(p.mean(axis=0) <= p_max * 1.02)
    np.testing.assert_allclose(eta, (p_max * beta[0]).min() / exp1(0.2))
    with pytest.raises(ValueError):
        TruncatedInversion(0.2, relative=False, budget="average")


# ---------------------------------------------------------------- data, partitions, models
def test_mlp_gradient_matches_finite_differences():
    m = MLP(5, 3, [4, 3])
    rng = np.random.default_rng(0)
    w = m.init(rng)
    x, y = rng.standard_normal((7, 5)), rng.integers(0, 3, 7)
    _, g = m.loss_grad(w, x, y)
    eye = np.eye(m.size) * 1e-6
    num = np.array([(m.loss_grad(w + e, x, y)[0] - m.loss_grad(w - e, x, y)[0]) / 2e-6 for e in eye])
    np.testing.assert_allclose(g, num, atol=1e-8)


def test_logreg_is_mlp_without_hidden():
    m = ac.get("model", "logreg")(d_in=12, n_classes=4)
    assert m.size == 12 * 4 + 4


@pytest.mark.parametrize("part", ["iid", Shards(2), Dirichlet(0.5)])
def test_partitions_cover_data_disjointly(part):
    parts = ac.build("partitioner", part)(DATA.y_train, 10, np.random.default_rng(0))
    allidx = np.concatenate(parts)
    assert len(parts) == 10 and len(allidx) == len(DATA.y_train)
    assert len(np.unique(allidx)) == len(allidx)


def test_shards_are_label_skewed():
    parts = Shards(1)(DATA.y_train, 8, np.random.default_rng(0))
    assert max(len(np.unique(DATA.y_train[p])) for p in parts) <= 2


def test_mnist_loader_formats(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (6, 28, 28), dtype=np.uint8)
    y = rng.integers(0, 10, 6).astype(np.uint8)
    np.savez(tmp_path / "mnist.npz", x_train=x, y_train=y, x_test=x[:2], y_test=y[:2])
    d = mnist(root=tmp_path)
    assert d.x_train.shape == (6, 784) and d.n_classes == 10 and abs(d.x_train.mean()) < 1e-5

    idx = tmp_path / "idx" / "fashion_mnist"
    idx.mkdir(parents=True)

    def write(name, arr):
        header = bytes([0, 0, 8, arr.ndim]) + b"".join(n.to_bytes(4, "big") for n in arr.shape)
        with gzip.open(idx / f"{name}.gz", "wb") as f:
            f.write(header + arr.tobytes())

    write("train-images-idx3-ubyte", x)
    write("train-labels-idx1-ubyte", y)
    write("t10k-images-idx3-ubyte", x[:2])
    write("t10k-labels-idx1-ubyte", y[:2])
    monkeypatch.setenv("AIRCOMPKIT_DATA", str(tmp_path / "idx"))
    f = ac.get("dataset", "fashion_mnist")()
    np.testing.assert_array_equal(f.y_test, y[:2])
    with pytest.raises(FileNotFoundError, match="AIRCOMPKIT_DATA"):
        mnist(root=tmp_path / "nowhere")


# ---------------------------------------------------------------- aggregation
def test_aggregate_exact_with_unit_amplitudes_and_no_noise():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 50)) * 3 + 1
    est = aircomp_aggregate(x, np.ones((4, 6), complex), np.ones(4), np.ones((4, 6), bool),
                            0.0, rng)
    np.testing.assert_allclose(est, x.mean(axis=0), atol=1e-12)


def test_aggregate_truncated_entries_count_as_zero():
    x = np.ones((4, 8))
    amp = np.ones((2, 4), complex)
    active = np.ones((2, 4), bool)
    amp[1, 0], active[1, 0] = 0, False  # device 0 silent on sub-channel 1 (odd entries)
    est = aircomp_aggregate(x, amp, np.ones(2), active, 0.0, np.random.default_rng(0),
                            normalize=False)
    np.testing.assert_allclose(est[0::2], 1.0)
    np.testing.assert_allclose(est[1::2], 0.75)  # Zhu et al.: missing entries, divide by |K|


def test_aggregate_noise_level():
    x = np.random.default_rng(2).standard_normal((5, 20000))
    est = aircomp_aggregate(x, np.ones((1, 5), complex), np.array([4.0]), np.ones((1, 5), bool),
                            2.0, np.random.default_rng(3), normalize=False)
    err = (est - x.mean(axis=0)) * 5
    assert abs(err.var() / 0.5 - 1) < 0.03  # noise_var_eff / eta


# ---------------------------------------------------------------- OTA-FL loop (exit criterion)
def test_vanishing_noise_matches_ideal_fedavg():
    ideal = train(sc(), cfg(aggregation="ideal", local_steps=3))
    ota = train(sc(), cfg(local_steps=3))
    np.testing.assert_allclose(ota["test_acc"], ideal["test_acc"])
    np.testing.assert_allclose(ota["train_loss"], ideal["train_loss"], rtol=1e-9)
    assert ota["agg_nmse"].max() < 1e-20
    assert ideal["test_acc"][-1] > 0.8


def test_small_noise_close_to_ideal_and_large_noise_hurts():
    ideal = train(sc(), cfg(aggregation="ideal"))
    small = train(sc(noise_var=1e-10), cfg())
    big = train(sc(noise_var=1e3, power_control="truncated_inversion"), cfg())
    assert abs(small["test_acc"][-1] - ideal["test_acc"][-1]) < 0.01
    assert small["agg_nmse"].mean() < 1e-6
    assert big["agg_nmse"].mean() > 10
    assert big["test_acc"][-1] < ideal["test_acc"][-1] - 0.05


def test_fl_works_over_mimo_and_impairments():
    s = sc(channel={"name": "rayleigh", "n_rx": 4}, receiver={"name": "maxmin", "iters": 60,
                                                             "random_inits": 2},
           power_control="optimal", noise_var=0.05,
           impairments={"csi_error": 0.02, "phase_error_std": 0.05})
    r = train(s, cfg(subchannels=4))
    assert r["test_acc"][-1] > 0.75 and r["agg_nmse"].max() < 0.5


def test_cell_interior_scheduling():
    s = sc(devices={"name": "uniform_disc", "K": 30, "r_min": 1, "r_max": 100})
    r = train(s, cfg(max_distance=60.0, rounds=5))
    dist = s.devices.sample_distances(ac.SeedTree(2).generator("devices"))
    assert r["n_scheduled"][0] == np.sum(dist <= 60)
    with pytest.raises(ValueError, match="max_distance"):
        train(sc(), cfg(max_distance=10.0, rounds=1))


def test_transmit_model_mode_matches_update_mode_without_noise():
    a = train(sc(), cfg(transmit="update"))
    b = train(sc(), cfg(transmit="model"))
    np.testing.assert_allclose(a["test_acc"], b["test_acc"])


def test_fl_requires_real_symbols_and_is_deterministic():
    with pytest.raises(ValueError, match="array path"):
        train(sc(), cfg(rounds=1, threat={"attack": "sign_flip", "n_attackers": 1}))
    a = train(sc(noise_var=0.1, power_control="optimal"), cfg(rounds=10))
    b = train(sc(noise_var=0.1, power_control="optimal"), cfg(rounds=10))
    np.testing.assert_array_equal(a["agg_nmse"], b["agg_nmse"])


# ---------------------------------------------------------------- gradient bank
def test_gradient_bank_record_save_load_replay(tmp_path):
    bank = GradientBank()
    train(sc(), cfg(rounds=6, record_bank=[0, 3, 5]), bank=bank)
    assert bank.rounds == [0, 3, 5] and bank.updates[0].shape[0] == 10
    p = bank.save(tmp_path / "bank")
    back = GradientBank.load(p)
    assert back.rounds == bank.rounds and back.meta["learning"]["rounds"] == 6
    np.testing.assert_array_equal(back.updates[1], bank.updates[1])

    exact = back.replay(sc(), trials=3)
    assert exact["nmse"].shape == (3, 3) and exact["nmse"].max() < 1e-20
    noisy = back.replay(sc(noise_var=1.0, power_control="truncated_inversion"), trials=40)
    lo = back.replay(sc(noise_var=0.01, power_control="truncated_inversion"), trials=40)
    assert noisy["nmse"].mean() > lo["nmse"].mean()


# ---------------------------------------------------------------- reproduction (Fig. 4 part)
def test_zhu2020_fig4_closed_form_and_simulation():
    from aircompkit.reproduce.zhu2020 import fig4, rho0_db

    assert abs(rho0_db(0.1) - 7.5) < 0.05  # paper Fig. 4: about 7.5 dB at 10% truncation
    f = fig4(trials=6000)
    np.testing.assert_allclose(f["sim_rho0_db"], f["closed_form_db"], atol=1e-9)
    t = np.asarray(f["truncation_ratio"])
    keep = t <= 0.8
    np.testing.assert_allclose(np.asarray(f["sim_truncation"])[keep], t[keep], atol=0.02)
    np.testing.assert_allclose(np.asarray(f["sim_avg_power_over_budget"])[keep], 1.0, atol=0.06)


def test_learning_config_in_yaml_is_ignored_by_scenario():
    cfg_ = sc().to_dict()
    cfg_["learning"] = {"rounds": 3}
    assert ac.Scenario.from_dict(cfg_).to_dict() == sc().to_dict()
