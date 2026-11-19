"""Reproduction: K. Yang, T. Jiang, Y. Shi, Z. Ding, "Federated learning via over-the-air
computation", IEEE Trans. Wireless Commun., 19(3), 2020 (arXiv:1812.11750), Figs. 4 and 5.

Setting (paper Sec. VI): ``K = 20`` single-antenna devices, i.i.d. ``h_k ~ CN(0, I_N)``,
all devices selected, ``phi_k = 1``. With the zero-forcing transmitter of Prop. 1, an MSE
target ``gamma`` is *feasible* when

    Q = min_{m != 0} max_k ||m||^2 / |m^H h_k|^2  <=  gamma          (paper eq. (11))

Fig. 4 plots P(feasible) against gamma for N = 6; Fig. 5 plots it against N for
gamma in {2, 3, 4} dB. In AirCompKit, ``Q`` is exactly the MSE of a scenario with
``P = sigma^2 = 1``, complex symbols, ``power_control = channel_inversion`` and the receiver
under test. ``maxmin`` stands in for the paper's global optimiser / DC method; ``sdr``
is SDR with Gaussian randomisation; the SDR upper bound gives a guaranteed lower bound on
``Q`` and therefore an upper bound on P(feasible) for *any* combiner.

Reference values were read by eye from the published figures (about +/- 0.03), for
comparison only. Known deviation: the paper's SDR curve uses a feasibility-form SDR and is
much weaker than the max-min SDR + randomisation implemented here, so the SDR curves are
not expected to match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from aircompkit.core.registry import register
from aircompkit.core.scenario import Scenario
from aircompkit.receiver import sdr_solve

K = 20
FIG4_N = 6
FIG4_GAMMA_DB = np.arange(-10, 36, 5)
FIG5_N = np.arange(2, 9)
FIG5_GAMMA_DB = (2.0, 3.0, 4.0)

# P(feasible) read from Fig. 5 (DC method), indexed by N = 2..8.
PAPER_FIG5 = {
    2.0: [0.00, 0.01, 0.10, 0.40, 0.77, 0.92, 0.99],
    3.0: [0.00, 0.14, 0.39, 0.78, 0.95, 0.97, 1.00],
    4.0: [0.02, 0.28, 0.64, 0.86, 0.99, 1.00, 1.00],
}
# Fig. 4 (N = 6): the global-optimum / DC curve is 0 at gamma = -5 dB and 1 at gamma = 5 dB.
PAPER_FIG4_GLOBAL = {-10.0: 0.0, -5.0: 0.0, 5.0: 1.0, 10.0: 1.0}


def zf_q_db(n_rx: int, receiver: Any, trials: int, seed: int) -> np.ndarray:
    """Per-trial ``10 log10 Q`` for a receiver, via the Scenario API."""
    sc = Scenario(
        devices={"name": "fixed", "K": K, "p_max": 1.0},
        channel={"name": "rayleigh", "n_rx": int(n_rx)},
        power_control="channel_inversion",
        receiver=receiver,
        target="sum",
        noise_var=1.0,
        symbols="complex",
        seed=seed,
    )
    return 10 * np.log10(sc.run(trials=trials, dim=0)["mse"])


def sdr_bound_q_db(n_rx: int, trials: int, seed: int) -> np.ndarray:
    """``10 log10 (1 / SDR upper bound)``: no combiner can achieve a smaller ``Q``."""
    from aircompkit.core.rng import SeedTree

    sc = Scenario(devices={"name": "fixed", "K": K}, channel={"name": "rayleigh", "n_rx": int(n_rx)},
                  seed=seed)
    h, _ = sc.channel.sample(SeedTree(seed).generator("channel"), trials, K)
    _, _, upper = sdr_solve(h)
    return -10 * np.log10(upper)


def feasibility(q_db: np.ndarray, gammas_db) -> list[float]:
    return [float(np.mean(q_db <= g)) for g in gammas_db]


@register("reproduction", "yang2020")
def run(trials: int | None = None, quick: bool = False, out: str | Path | None = None,
        seed: int = 0) -> dict[str, Any]:
    """Reproduce Figs. 4–5. ``quick``: 60 trials (CI smoke test); default 500 (paper: 100)."""
    T = trials or (60 if quick else 500)
    res: dict[str, Any] = {"trials": T, "K": K, "seed": seed}

    q = {name: zf_q_db(FIG4_N, name, T, seed) for name in ("maxmin", "sdr", "eigen")}
    q["sdr_bound"] = sdr_bound_q_db(FIG4_N, T, seed)
    res["fig4"] = {
        "gamma_db": FIG4_GAMMA_DB.tolist(),
        **{name: feasibility(v, FIG4_GAMMA_DB) for name, v in q.items()},
        "median_q_db": {name: float(np.median(v)) for name, v in q.items()},
    }

    fig5: dict[str, Any] = {"n_rx": FIG5_N.tolist()}
    q5 = {int(n): zf_q_db(int(n), "maxmin", T, seed) for n in FIG5_N}
    for g in FIG5_GAMMA_DB:
        fig5[f"maxmin_{g:g}dB"] = [float(np.mean(q5[int(n)] <= g)) for n in FIG5_N]
        fig5[f"paper_{g:g}dB"] = PAPER_FIG5[g]
    dev = [abs(a - b) for g in FIG5_GAMMA_DB
           for a, b in zip(fig5[f"maxmin_{g:g}dB"], PAPER_FIG5[g])]
    fig5["max_abs_deviation"] = float(max(dev))
    fig5["mean_abs_deviation"] = float(np.mean(dev))
    res["fig5"] = fig5

    _print(res)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "yang2020.json").write_text(json.dumps(res, indent=2))
        _plot(res, out / "yang2020_fig4_fig5.png")
    return res


def _print(res: dict[str, Any]) -> None:
    f4 = res["fig4"]
    print(f"Yang et al. 2020 Fig. 4: P(feasible) vs gamma, K={K}, N={FIG4_N}, {res['trials']} trials")
    print("gamma dB " + " ".join(f"{g:6.0f}" for g in f4["gamma_db"]))
    for name in ("maxmin", "sdr", "eigen", "sdr_bound"):
        print(f"{name:>8} " + " ".join(f"{v:6.2f}" for v in f4[name]))
    f5 = res["fig5"]
    print("\nFig. 5: P(feasible) vs N (maxmin / paper)")
    print("     N   " + " ".join(f"{n:11d}" for n in f5["n_rx"]))
    for g in FIG5_GAMMA_DB:
        ours, ref = f5[f"maxmin_{g:g}dB"], f5[f"paper_{g:g}dB"]
        print(f"{g:4.0f} dB " + " ".join(f"{a:5.2f}/{b:5.2f}" for a, b in zip(ours, ref)))
    print(f"mean |ours - paper| = {f5['mean_abs_deviation']:.3f}, "
          f"max = {f5['max_abs_deviation']:.3f}")


def _plot(res: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    f4 = res["fig4"]
    labels = {"maxmin": "max-min (AirCompKit)", "sdr": "SDR + randomisation",
              "eigen": "eigen combiner", "sdr_bound": "SDR bound (any combiner)"}
    styles = {"maxmin": "o-", "sdr": "s-", "eigen": "^-", "sdr_bound": "k:"}
    for name, lab in labels.items():
        a.plot(f4["gamma_db"], f4[name], styles[name], ms=4, label=lab)
    a.plot(list(PAPER_FIG4_GLOBAL), list(PAPER_FIG4_GLOBAL.values()), "D", mfc="none", ms=8,
           color="tab:red", label="paper: global opt. / DC")
    a.set_xlabel(r"MSE target $\gamma$ (dB)")
    a.set_ylabel("probability of feasibility")
    a.set_title(f"(a) cf. Yang et al. Fig. 4  (K={K}, N={FIG4_N})", fontsize=9)
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
    f5 = res["fig5"]
    for g, c in zip(FIG5_GAMMA_DB, ("tab:blue", "tab:orange", "tab:green")):
        b.plot(f5["n_rx"], f5[f"maxmin_{g:g}dB"], "o-", color=c, ms=4,
               label=rf"AirCompKit, $\gamma$={g:g} dB")
        b.plot(f5["n_rx"], f5[f"paper_{g:g}dB"], "x--", color=c, ms=6, lw=0.8,
               label=rf"paper, $\gamma$={g:g} dB")
    b.set_xlabel("number of BS antennas N")
    b.set_title(f"(b) cf. Yang et al. Fig. 5  (K={K})", fontsize=9)
    b.grid(alpha=0.3)
    b.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"saved {path}")


def check(res: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Tolerance checks used by ``aircompkit reproduce yang2020 --check`` (reproduction CI)."""
    f4, f5 = res["fig4"], res["fig5"]
    g = f4["gamma_db"]
    mm = dict(zip(g, f4["maxmin"]))
    out = [
        ("Fig. 4: max-min infeasible at -5 dB (paper: 0)", mm[-5] <= 0.05, f"{mm[-5]:.2f}"),
        ("Fig. 4: max-min feasible at 5 dB (paper: 1)", mm[5] >= 0.95, f"{mm[5]:.2f}"),
    ]
    worst = max(f4[n][i] - f4["sdr_bound"][i] for n in ("maxmin", "sdr", "eigen") for i in range(len(g)))
    out.append(("Fig. 4: no combiner beats the SDR bound", worst <= 1e-12, f"max excess {worst:.2g}"))
    mono = all(a <= b for gg in FIG5_GAMMA_DB
               for a, b in zip(f5[f"maxmin_{gg:g}dB"], f5[f"maxmin_{gg:g}dB"][1:]))
    out.append(("Fig. 5: feasibility grows with N", mono, ""))
    out.append(("Fig. 5: mean |AirCompKit - paper| <= 0.15", f5["mean_abs_deviation"] <= 0.15,
                f"{f5['mean_abs_deviation']:.3f}"))
    return out
