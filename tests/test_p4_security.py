"""P4 validation: threat model, attacks, defences, DP accounting, secure OTA-FL."""

import numpy as np
import pytest

import aircompkit as ac
from aircompkit.adversary import ThreatModel
from aircompkit.core.rng import SeedTree
from aircompkit.defense import (
    DigitalRobust,
    GroupGM,
    NoDefense,
    geometric_median,
    krum,
    nulling_combiner,
)
from aircompkit.defense.evaluate import detection_sweep
from aircompkit.learning import FederatedConfig, GradientBank, synthetic, train
from aircompkit.learning.array import Receiver, normalise, pack, unpack
from aircompkit.metrics.privacy import gaussian_epsilon

OMNI = ["own_data", "own_csi", "honest_mean"]


def arr(**kw):
    base = {"devices": {"name": "fixed", "K": 12, "p_max": 1.0},
            "channel": {"name": "rayleigh", "n_rx": 8}, "receiver": {"name": "maxmin", "iters": 80,
                                                                      "random_inits": 2},
            "power_control": "channel_inversion", "noise_var": 0.0, "symbols": "complex", "seed": 3}
    base.update(kw)
    return ac.Scenario(**base)


def receiver(sc, attackers=(), attack=None, L=64, seed=0, jammer=None):
    rng = np.random.default_rng(seed)
    K = sc.devices.K
    x = (rng.standard_normal((K, L)) + 1j * rng.standard_normal((K, L))) / np.sqrt(2)
    hon = np.setdiff1d(np.arange(K), attackers)
    return Receiver(sc, x, sc.devices.power_budgets(), None, SeedTree(seed).child("r"),
                    attackers=np.asarray(attackers, int),
                    attack=ac.build("attack", attack) if attack else None,
                    gref=x[hon].mean(axis=0), jammer=jammer), x, hon


# ---------------------------------------------------------------- threat model
def test_threat_model_checks_knowledge_and_power():
    with pytest.raises(ValueError, match="honest_mean"):
        ThreatModel(attack="collude", n_attackers=2)
    ThreatModel(attack="collude", n_attackers=2, knowledge=OMNI)
    with pytest.raises(ValueError, match="max_boost_db"):
        ThreatModel(attack={"name": "sign_flip", "boost_db": 20}, max_boost_db=10)
    with pytest.raises(ValueError, match="unknown knowledge"):
        ThreatModel(knowledge=["telepathy"])


def test_attacker_selection_reproducible():
    a = ThreatModel(n_attackers=5, seed=7).attackers(40)
    np.testing.assert_array_equal(a, ThreatModel(n_attackers=5, seed=7).attackers(40))
    assert len(ThreatModel(fraction=0.25).attackers(40)) == 10
    with pytest.raises(ValueError):
        ThreatModel(n_attackers=40).attackers(40)


# ---------------------------------------------------------------- packing / reception
@pytest.mark.parametrize("d", [10, 11])
def test_pack_roundtrip_and_clip_normalisation(d):
    X = np.random.default_rng(0).standard_normal((4, d))
    np.testing.assert_allclose(unpack(pack(X), d), X)
    Xc = X / np.linalg.norm(X, axis=1, keepdims=True) * 0.5
    x, nu = normalise(Xc, 0.5)
    L = x.shape[1]
    assert np.all(np.sum(np.abs(x) ** 2, axis=1) <= L * (1 + 1e-12))
    np.testing.assert_allclose(nu * unpack(x, d), Xc)


def test_noiseless_aircomp_is_exact_sum():
    rx, x, _ = receiver(arr())
    est, keff, flagged = NoDefense().aggregate(rx, {"attackers": []})
    np.testing.assert_allclose(est, x.sum(axis=0), atol=1e-9)
    assert keff == 12 and flagged == set() and rx.uses == 1


@pytest.mark.parametrize("attack,expected", [
    ({"name": "sign_flip", "boost_db": 20.0}, lambda x, byz: -10 * x[byz].sum(0)),
    ({"name": "scaled", "boost_db": 6.0}, lambda x, byz: 10 ** 0.3 * x[byz].sum(0)),
    ({"name": "misalign", "phase": np.pi}, lambda x, byz: -x[byz].sum(0)),
])
def test_payload_attacks_superpose_as_specified(attack, expected):
    byz = [1, 5]
    rx, x, hon = receiver(arr(), byz, attack)
    est, _, _ = NoDefense().aggregate(rx, {"attackers": byz})
    np.testing.assert_allclose(est, x[hon].sum(0) + expected(x, byz), atol=1e-8)


def test_collude_sends_scaled_negative_honest_direction():
    byz = [0, 2, 3]
    rx, x, hon = receiver(arr(), byz, {"name": "collude", "boost_db": 0.0})
    est, _, _ = NoDefense().aggregate(rx, {"attackers": byz})
    g = x[hon].mean(0)
    v = -g / np.sqrt(np.mean(np.abs(g) ** 2))
    np.testing.assert_allclose(est, x[hon].sum(0) + 3 * v, atol=1e-8)


def test_gaussian_attack_power_scales_with_boost():
    errs = []
    for boost in (0.0, 10.0):
        rx, x, hon = receiver(arr(), [4], {"name": "gaussian", "boost_db": boost}, L=4000)
        est, _, _ = NoDefense().aggregate(rx, {"attackers": [4]})
        errs.append(np.mean(np.abs(est - x[hon].sum(0)) ** 2))
    # identical channel and noise draws, so the error power scales exactly with the boost
    assert errs[0] > 0.1
    np.testing.assert_allclose(errs[1] / errs[0], 10.0, rtol=1e-9)


def test_csi_spoof_lowers_eta_and_boosts_attacker():
    clean, x, _ = receiver(arr())
    r0 = clean.transmit()
    spoof, _, hon = receiver(arr(), [2], {"name": "csi_spoof", "factor": 0.1})
    r1 = spoof.transmit()
    assert r1.eta < 0.05 * r0.eta
    est = r1.m.conj() @ r1.Y / np.sqrt(r1.eta)
    np.testing.assert_allclose(est, x[hon].sum(0) + 10 * x[2], atol=1e-6)


def test_external_jammer_adds_interference():
    rx, x, _ = receiver(arr(), jammer={"power": 1.0})
    est, _, _ = NoDefense().aggregate(rx, {"attackers": []})
    assert np.mean(np.abs(est - x.sum(0)) ** 2) > 1e-3


def test_label_flip():
    lf = ac.build("attack", "label_flip")
    assert lf.poisons_labels
    np.testing.assert_array_equal(lf.flip_labels(np.array([0, 3, 9]), 10), [9, 6, 0])


# ---------------------------------------------------------------- defences
def test_nulling_combiner_hard_nulls():
    rng = np.random.default_rng(1)
    A = (rng.standard_normal((8, 12)) + 1j * rng.standard_normal((8, 12))) / np.sqrt(2)
    w, keep = nulling_combiner(A, [3, 7], 1.0, 0.01)
    np.testing.assert_allclose(np.abs(w.conj() @ A[:, [3, 7]]), 0, atol=1e-10)
    assert keep.sum() == 10


def test_genie_removes_boosted_attackers():
    byz = [1, 6]
    sc = arr(channel={"name": "rayleigh", "n_rx": 16}, noise_var=1e-3)
    out = {}
    for name, d in (("none", NoDefense()), ("genie", ac.build("defense", "genie"))):
        rx, x, hon = receiver(sc, byz, {"name": "sign_flip", "boost_db": 20.0}, L=2000)
        d.reset(12)
        est, keff, flagged = d.aggregate(rx, {"attackers": byz})
        ideal = x[hon].mean(0)
        out[name] = (np.sum(np.abs(est / keff - ideal) ** 2) / np.sum(np.abs(ideal) ** 2), flagged)
    assert out["genie"][1] == set(byz) and out["none"][1] == set()
    assert out["none"][0] > 50 * out["genie"][0]
    assert out["genie"][0] < 0.5


class EnergyDetector:
    """Simple test detector: least-squares per-device estimates, flag energy > 4x the median."""

    def detect(self, Y, A, noise_var):
        X = np.linalg.lstsq(A, Y, rcond=None)[0]
        e = np.mean(np.abs(X) ** 2, axis=1)
        return [int(k) for k in np.flatnonzero(e > 4 * np.median(e))]


def test_group_gm_and_digital_rules_resist_one_huge_attacker():
    sc = arr(noise_var=1e-4)
    rx, x, hon = receiver(sc, [0], {"name": "scaled", "boost_db": 40.0}, L=400)
    ideal = x[hon].mean(0)
    est, _keff, _ = GroupGM(groups=4).aggregate(rx, {"attackers": [0]})
    assert rx.uses == 4
    gm_err = np.sum(np.abs(est - ideal) ** 2) / np.sum(np.abs(ideal) ** 2)
    rx2, _, _ = receiver(sc, [0], {"name": "scaled", "boost_db": 40.0}, L=400)
    van, k, _ = NoDefense().aggregate(rx2, {"attackers": [0]})
    van_err = np.sum(np.abs(van / k - ideal) ** 2) / np.sum(np.abs(ideal) ** 2)
    assert gm_err < 0.2 * van_err
    for rule in ("median", "trimmed_mean", "krum", "geomed"):
        rx3, _, _ = receiver(sc, [0], {"name": "scaled", "boost_db": 40.0}, L=400)
        z, _, _ = DigitalRobust(rule=rule).aggregate(rx3, {"attackers": [0]})
        assert rx3.uses == 12
        assert np.sum(np.abs(z - ideal) ** 2) < 0.2 * van_err * np.sum(np.abs(ideal) ** 2), rule


def test_geometric_median_and_krum():
    X = np.vstack([np.zeros((6, 3)), 1e6 * np.ones((1, 3))])
    assert np.linalg.norm(geometric_median(X)) < 1e-3
    np.testing.assert_array_equal(krum(X, 1), np.zeros(3))


# ---------------------------------------------------------------- detection Monte Carlo
@pytest.fixture(scope="module")
def bank():
    data = synthetic(n_train=4000, n_test=500, dim=40, n_classes=5, seed=2)
    b = GradientBank()
    sc = ac.Scenario(devices={"name": "fixed", "K": 24}, symbols="real")
    train(sc, FederatedConfig(dataset=data, rounds=6, record_bank=[1, 3, 5], clip=0.2,
                              aggregation="ideal", model={"name": "mlp", "hidden": [32]}), bank=b)
    return b


def test_detection_sweep(bank):
    sc = arr(devices={"name": "fixed", "K": 24}, channel={"name": "rayleigh", "n_rx": 16},
             noise_var=10 ** -0.5)
    h0 = detection_sweep(sc, bank.updates, "gaussian", 0, EnergyDetector(), trials=20, clip=0.2)
    strong = detection_sweep(sc, bank.updates, {"name": "sign_flip", "boost_db": 20.0}, 3,
                             EnergyDetector(), trials=20, clip=0.2)
    weak = detection_sweep(sc, bank.updates, {"name": "sign_flip", "boost_db": 0.0}, 3,
                           EnergyDetector(), trials=20, clip=0.2)
    assert h0["fp"] < 0.5 and strong["tpr"] > 0.9 and strong["tpr"] > weak["tpr"]


# ---------------------------------------------------------------- secure OTA-FL
def test_secure_fl_vanilla_collapses_defences_recover():
    data = synthetic(n_train=4000, n_test=1000, dim=20, n_classes=4, seed=4)
    sc = arr(devices={"name": "fixed", "K": 20}, channel={"name": "rayleigh", "n_rx": 12},
             noise_var=0.01)
    base = {"dataset": data, "rounds": 40, "eval_every": 40, "lr": 0.2, "clip": 0.5,
            "model": {"name": "mlp", "hidden": [16]}}
    thr = {"attack": {"name": "sign_flip", "boost_db": 20.0}, "n_attackers": 3}
    ideal = train(sc, FederatedConfig(**base, aggregation="ideal"))["test_acc"][-1]
    van = train(sc, FederatedConfig(**base, threat=thr))
    gen = train(sc, FederatedConfig(**base, threat=thr, defense="genie"))
    med = train(sc, FederatedConfig(**base, threat=thr,
                                    defense={"name": "digital_robust", "rule": "median"}))
    assert van["test_acc"][-1] < ideal - 0.2
    assert gen["test_acc"][-1] > ideal - 0.05 and med["test_acc"][-1] > ideal - 0.1
    assert gen["det_tp"].mean() == 3 and gen["det_fp"].mean() == 0
    assert med["channel_uses"].mean() == 20
    cfg = FederatedConfig(**base, threat=ThreatModel(**thr), defense=GroupGM(groups=4)).to_dict()
    assert cfg["threat"]["attack"]["name"] == "sign_flip" and cfg["defense"]["name"] == "group_gm"


def test_attack_free_array_path_matches_ideal_without_noise():
    data = synthetic(n_train=2000, n_test=500, dim=10, n_classes=3, seed=5)
    sc = arr(noise_var=0.0)
    base = {"dataset": data, "rounds": 10, "eval_every": 5, "model": {"name": "logreg"}}
    a = train(sc, FederatedConfig(**base, aggregation="ideal"))
    b = train(sc, FederatedConfig(**base))
    np.testing.assert_allclose(a["test_acc"], b["test_acc"])
    assert b["agg_nmse"].max() < 1e-18


# ---------------------------------------------------------------- privacy
def test_gaussian_epsilon():
    # closed-form optimum of the RDP bound for Delta = sigma = 1, T = 1, delta = 1e-5
    L = np.log(1e5)
    a = 1 + np.sqrt(2 * L)
    assert abs(gaussian_epsilon(1.0, 1.0, 1, 1e-5) - (a / 2 + L / (a - 1))) < 1e-3
    assert gaussian_epsilon(1.0, 2.0) < gaussian_epsilon(1.0, 1.0)
    assert gaussian_epsilon(1.0, 1.0, rounds=10) > gaussian_epsilon(1.0, 1.0)
