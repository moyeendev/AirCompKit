"""Reproduction: G. Zhu, Y. Wang, K. Huang, "Broadband analog aggregation for low-latency
federated edge learning", IEEE Trans. Wireless Commun., 19(1), 2020 (arXiv:1812.11494).

Paper settings (Sec. VI-A): K = 200 devices uniform in a disc of radius R = 100 m, path-loss
exponent alpha = 3, M = 1000 sub-channels, average transmit power P0 = 0.1 W per device
(P0/M per sub-channel), noise N0 = -80 dBm per sub-channel, Rayleigh fading, truncated channel
inversion with an average power budget, MNIST split IID (200 equal shares) or non-IID
(400 label-sorted shards of 150, two per device).

* **Fig. 4** (SNR-truncation trade-off): ``rho0 = P0 / (M r_max^alpha E1(g_th))`` against the
  truncation ratio ``1 - exp(-g_th)``, with r_max = 100. Checked against Monte Carlo: the
  empirical truncation ratio, the empirical *average* transmit power (must equal P0/M) and
  ``eta/N0`` from the simulator's power control.
* **Fig. 7(b)** (alpha = 3): test accuracy against the cell-interior radius ratio R_in/R.
  The paper uses a 582k-parameter CNN, an unstated number of rounds and a per-point grid
  search over g_th; here an MLP (784-64-10), a fixed number of rounds and g_th = 0.1 are used,
  so only the *trend* is compared. Values read by eye from the paper's bar chart (±0.02).

Known outcome (see the Reproductions page of the documentation): the data-limited rise reproduces; the SNR-limited
decline at large R_in does not appear with the stated SNRs, because SGD averages the
zero-mean channel noise over rounds (the paper's own Fig. 10 remarks on this robustness).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aircompkit.aircomp.power_control import exp1
from aircompkit.core.registry import register
from aircompkit.core.rng import SeedTree
from aircompkit.core.scenario import Scenario

K, R, ALPHA, M = 200, 100.0, 3.0, 1000
P0, N0 = 0.1, 1e-11  # W; -80 dBm
FIG4_TRUNC = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
FIG7_RATIOS = (0.1, 0.3, 0.5, 0.7, 0.9)
PAPER_FIG7B = {"noniid": [0.72, 0.80, 0.85, 0.83, 0.79], "iid": [0.80, 0.85, 0.90, 0.86, 0.81]}


def rho0_db(trunc: float, r_max: float = R, alpha: float = ALPHA) -> float:
    g = -np.log(1 - trunc)
    return float(10 * np.log10(P0 / (M * r_max**alpha * exp1(g)) / N0))


def scenario(threshold: float, alpha: float = ALPHA, K_: int = K, seed: int = 0) -> Scenario:
    return Scenario(
        devices={"name": "uniform_disc", "K": K_, "p_max": P0 / M, "r_min": 1.0, "r_max": R},
        channel={"name": "rayleigh", "pathloss": {"exponent": alpha}},
        power_control={"name": "truncated_inversion", "threshold": threshold, "budget": "average"},
        receiver="eigen",
        noise_var=N0,
        symbols="real",
        seed=seed,
    )


def fig4(trials: int = 20000, seed: int = 0) -> dict[str, Any]:
    """Closed form vs. Monte Carlo for one device at r_max = R."""
    out: dict[str, Any] = {"truncation_ratio": FIG4_TRUNC.tolist(), "closed_form_db": [],
                           "sim_rho0_db": [], "sim_truncation": [], "sim_avg_power_over_budget": []}
    for z in FIG4_TRUNC:
        g = float(-np.log(1 - z))
        sc = Scenario(devices={"name": "fixed", "K": 1, "p_max": P0 / M, "distances": [R]},
                      channel={"name": "rayleigh", "pathloss": {"exponent": ALPHA}},
                      power_control={"name": "truncated_inversion", "threshold": g, "budget": "average"},
                      receiver="eigen", noise_var=N0, symbols="complex", seed=seed)
        h_act, h_est, beta = sc.sample_channels(SeedTree(seed), trials, np.array([R]))
        lay = sc.design(h_act, h_est, beta, sc.devices.power_budgets())
        out["closed_form_db"].append(rho0_db(z))
        out["sim_rho0_db"].append(float(10 * np.log10(lay["eta"].mean() / N0)))
        out["sim_truncation"].append(float(np.mean(lay["p"][:, 0] == 0)))
        out["sim_avg_power_over_budget"].append(float(lay["p"][:, 0].mean() / (P0 / M)))
    return out


def fig7(data, rounds: int, threshold: float = 0.1, seed: int = 0,
         ratios=FIG7_RATIOS, verbose: bool = False) -> dict[str, Any]:
    from aircompkit.learning import FederatedConfig, train

    out: dict[str, Any] = {"ratios": list(ratios), "rounds": rounds, "threshold": threshold}
    for split, part in (("iid", "iid"), ("noniid", {"name": "shards", "shards_per_device": 2})):
        ota, ideal, n_sched = [], [], []
        for q in ratios:
            sc = scenario(threshold, seed=seed)
            cfg = {"dataset": data, "partition": part, "model": {"name": "mlp", "hidden": [64]},
                   "rounds": rounds, "eval_every": rounds, "lr": 0.1, "batch_size": 50,
                   "subchannels": M, "max_distance": q * R, "seed": seed}
            try:
                r = train(sc, FederatedConfig(**cfg))
                ri = train(sc, FederatedConfig(**{**cfg, "aggregation": "ideal"}))
            except ValueError:  # nobody inside R_in for this placement
                ota.append(float("nan"))
                ideal.append(float("nan"))
                n_sched.append(0)
                continue
            ota.append(float(r["test_acc"][-1]))
            ideal.append(float(ri["test_acc"][-1]))
            n_sched.append(int(r["n_scheduled"][0]))
            if verbose:
                print(f"  {split:6s} R_in/R={q:.1f}  devices={n_sched[-1]:3d}  "
                      f"OTA {ota[-1]:.4f}  ideal {ideal[-1]:.4f}")
        out[split] = {"ota": ota, "ideal": ideal, "n_scheduled": n_sched, "paper": PAPER_FIG7B[split]}
    return out


@register("reproduction", "zhu2020")
def run(trials: int | None = None, quick: bool = False, out: str | Path | None = None,
        data_root: str | Path | None = None, rounds: int | None = None,
        seed: int = 0) -> dict[str, Any]:
    """Fig. 4 always; Fig. 7(b) when MNIST is available locally (see ``learning.data``)."""
    res: dict[str, Any] = {"fig4": fig4(trials or (4000 if quick else 20000), seed)}
    _print_fig4(res["fig4"])
    try:
        from aircompkit.learning.data import mnist

        data = mnist(root=data_root)
    except FileNotFoundError as e:
        res["fig7"] = None
        print(f"\nFig. 7(b) skipped: {str(e).splitlines()[0]}")
    else:
        n_rounds = rounds or (15 if quick else 100)
        print(f"\nFig. 7(b): accuracy vs R_in/R (alpha=3, MLP, {n_rounds} rounds, g_th=0.1)")
        res["fig7"] = fig7(data, n_rounds, seed=seed, verbose=True)
        _print_fig7(res["fig7"])
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "zhu2020.json").write_text(json.dumps(res, indent=2))
        _plot(res, out / "zhu2020_fig4_fig7.png")
    return res


def _print_fig4(f4):
    print("Zhu et al. 2020 Fig. 4: receive SNR vs truncation ratio (r_max = 100 m)")
    print(" trunc  closed-form dB  sim dB   sim trunc  avg power / budget")
    for row in zip(f4["truncation_ratio"], f4["closed_form_db"], f4["sim_rho0_db"],
                   f4["sim_truncation"], f4["sim_avg_power_over_budget"]):
        print("  {:4.2f}   {:10.3f}   {:8.3f}   {:7.4f}   {:8.4f}".format(*row))


def _print_fig7(f7):
    print(" R_in/R " + "".join(f"{q:>16.1f}" for q in f7["ratios"]))
    for split in ("iid", "noniid"):
        d = f7[split]
        print(f" {split:6s} " + "".join(f"  {o:5.3f}/{i:5.3f}/{p:4.2f}"
                                        for o, i, p in zip(d["ota"], d["ideal"], d["paper"])))
    print(" (OTA / ideal aggregation / paper)")


def _plot(res, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    f4, f7 = res["fig4"], res["fig7"]
    ncols = 2 if f7 else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 3.6))
    a = axes[0] if ncols == 2 else axes
    a.plot(f4["truncation_ratio"], f4["closed_form_db"], "k-", label="closed form (paper eq. 17)")
    a.plot(f4["sim_truncation"], f4["sim_rho0_db"], "o", mfc="none", label="AirCompKit Monte Carlo")
    a.set_xlabel("truncation ratio")
    a.set_ylabel("receive SNR (dB)")
    a.set_title("(a) cf. Zhu et al. Fig. 4", fontsize=9)
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
    if f7:
        b = axes[1]
        x = np.asarray(f7["ratios"])
        for split, c in (("iid", "tab:green"), ("noniid", "tab:blue")):
            b.plot(x, f7[split]["ota"], "o-", color=c, label=f"AirCompKit OTA, {split}")
            b.plot(x, f7[split]["ideal"], ":", color=c, label=f"ideal aggregation, {split}")
            b.plot(x, f7[split]["paper"], "x--", color=c, lw=0.8, label=f"paper (CNN), {split}")
        b.set_xlabel(r"$R_{in}/R$")
        b.set_ylabel("test accuracy")
        b.set_title(rf"(b) cf. Zhu et al. Fig. 7(b), $\alpha$=3, {f7['rounds']} rounds", fontsize=9)
        b.grid(alpha=0.3)
        b.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"saved {path}")


def check(res: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Tolerance checks used by ``aircompkit reproduce zhu2020 --check`` (reproduction CI)."""
    f4 = res["fig4"]
    drho = max(abs(a - b) for a, b in zip(f4["sim_rho0_db"], f4["closed_form_db"]))
    dtr = max(abs(a - b) for a, b in zip(f4["sim_truncation"], f4["truncation_ratio"]))
    dpow = max(abs(a - 1) for a, b in zip(f4["sim_avg_power_over_budget"], f4["truncation_ratio"]) if b <= 0.9)
    out = [("Fig. 4: designed rho_0 = closed form (0.1 dB)", drho <= 0.1, f"max {drho:.3g} dB"),
           ("Fig. 4: truncation ratio as designed (0.03)", dtr <= 0.03, f"max {dtr:.3f}"),
           ("Fig. 4: average power = budget (25 %, ratio <= 0.9)", dpow <= 0.25, f"max {dpow:.3f}")]
    f7 = res.get("fig7")
    if f7:
        gaps = [abs(a - b) for s in ("iid", "noniid") for a, b in zip(f7[s]["ota"], f7[s]["ideal"])
                if np.isfinite(a)]
        out.append(("Fig. 7(b): OTA-FL within 2 points of ideal FedAvg", max(gaps) <= 0.02,
                    f"max gap {max(gaps):.3f}"))
        acc = dict(zip(f7["ratios"], f7["noniid"]["ota"]))
        out.append(("Fig. 7(b): non-IID accuracy rises with R_in (data-limited regime)",
                    acc[0.9] > acc[0.3], f"{acc[0.3]:.3f} -> {acc[0.9]:.3f}"))
    return out
