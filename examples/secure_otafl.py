"""Defences for over-the-air federated learning under three attacks (MNIST).

K = 30 devices, a 12-antenna server with the max-min combiner, channel inversion, 5 dB SNR,
Dirichlet(0.5) splits, 5 attackers at +10 dB. Every defence runs on the same channels, data
and attackers (paired seeds). Records accuracy curves and channel uses per round.

    AIRCOMPKIT_DATA=/path/to/data python examples/secure_otafl.py            # full: ~35 min
    python examples/secure_otafl.py --quick                                 # smoke test
    python examples/secure_otafl.py --out results/secure_otafl

Writes ``secure_otafl.json`` and ``secure_otafl.png`` to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import aircompkit as ac
from aircompkit.learning import FederatedConfig, train
from aircompkit.learning.data import mnist

K, N, SNR_DB, N_ATT, BOOST_DB = 30, 12, 5.0, 5, 10.0
ATTACKS = {
    "sign_flip": {"attack": {"name": "sign_flip", "boost_db": BOOST_DB}},
    "collude": {"attack": {"name": "collude", "boost_db": BOOST_DB},
                "knowledge": ["own_data", "own_csi", "honest_mean"]},
    "gaussian": {"attack": {"name": "gaussian", "boost_db": BOOST_DB}},
}
DEFENSES: dict[str, Any] = {
    "none": "none", "group_gm": {"name": "group_gm", "groups": 6},
    "digital_median": {"name": "digital_robust", "rule": "median"}, "genie": "genie"}


def scenario(seed: int) -> ac.Scenario:
    return ac.Scenario(devices={"name": "fixed", "K": K, "p_max": 1.0},
                       channel={"name": "rayleigh", "n_rx": N}, receiver="maxmin",
                       power_control="channel_inversion", noise_var=10 ** (-SNR_DB / 10),
                       symbols="complex", seed=seed)


def config(data, rounds: int, seed: int, **kw: Any) -> FederatedConfig:
    return FederatedConfig(dataset=data, partition={"name": "dirichlet", "alpha": 0.5, "min_size": 64},
                           model={"name": "mlp", "hidden": [64]}, rounds=rounds,
                           eval_every=max(rounds // 20, 1), lr=0.1, batch_size=64, clip=0.2,
                           seed=seed, **kw)


def run(rounds: int, seeds: list[int], data_root: str | None = None) -> dict[str, Any]:
    data = mnist(root=data_root)
    out: dict[str, Any] = {"K": K, "N": N, "snr_db": SNR_DB, "n_attackers": N_ATT,
                           "boost_db": BOOST_DB, "rounds": rounds, "seeds": seeds, "runs": {}}
    t0 = time.time()
    for s in seeds:
        sc = scenario(1000 + s)
        r = train(sc, config(data, rounds, s, aggregation="ideal"))
        out["runs"].setdefault("ideal", []).append(
            {"eval_round": r["eval_round"].tolist(), "test_acc": r["test_acc"].tolist(), "channel_uses": 0.0})
        for att, spec in ATTACKS.items():
            threat = {**spec, "n_attackers": N_ATT, "seed": 100 + s}
            for name, d in DEFENSES.items():
                r = train(sc, config(data, rounds, s, threat=threat, defense=d))
                rec = {"eval_round": r["eval_round"].tolist(), "test_acc": r["test_acc"].tolist(),
                       "channel_uses": float(r["channel_uses"].mean())}
                out["runs"].setdefault(f"{att}/{name}", []).append(rec)
                print(f"seed {s} {att:9s} {name:15s} acc {r['test_acc'][-1]:.4f} "
                      f"({time.time() - t0:.0f} s)", flush=True)
    out["elapsed_s"] = time.time() - t0
    return out


def summary(res: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Final accuracy (mean of the last three evaluations, %) per run key: mean and std over seeds."""
    tab = {}
    for key, runs in res["runs"].items():
        acc = [100 * float(np.mean(r["test_acc"][-3:])) for r in runs]
        tab[key] = {"acc": float(np.mean(acc)), "std": float(np.std(acc)),
                    "channel_uses": float(np.mean([r["channel_uses"] for r in runs]))}
    return tab


def plot(res: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, att in zip(axes, ATTACKS):
        ideal = res["runs"]["ideal"]
        ax.plot(ideal[0]["eval_round"], 100 * np.mean([r["test_acc"] for r in ideal], axis=0), "k--", label="ideal")
        for name in DEFENSES:
            runs = res["runs"][f"{att}/{name}"]
            ax.plot(runs[0]["eval_round"], 100 * np.mean([r["test_acc"] for r in runs], axis=0), label=name)
        ax.set_title(f"{att}, {N_ATT} attackers, +{BOOST_DB:g} dB", fontsize=9)
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy (%)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 rounds, one seed")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--data-root")
    ap.add_argument("--out", default="results/secure_otafl")
    a = ap.parse_args()
    rounds, seeds = (20, [0]) if a.quick else (a.rounds, list(range(a.seeds)))
    res = run(rounds, seeds, a.data_root)
    res["summary"] = summary(res)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "secure_otafl.json").write_text(json.dumps(res))
    plot(res, out / "secure_otafl.png")
    for key, v in res["summary"].items():
        print(f"{key:26s} {v['acc']:6.2f} +/- {v['std']:4.2f}  uses/round {v['channel_uses']:.0f}")


if __name__ == "__main__":
    main()
