"""Performance benchmarks (``aircompkit bench``).

Three groups, each timed with the best of ``repeat`` wall-clock runs after one warm-up:

* ``mse``: ``Scenario.run`` throughput (channel realisations per second) for growing
  ``K``, analytical only (``dim = 0``) and with signal-level Monte Carlo (``dim = 32``);
* ``receiver``: receive-beamformer design time per channel realisation (``K = 20``);
* ``fl``: over-the-air federated learning rounds per second on an MNIST-sized synthetic task (784-64-10 MLP), on the
  broadband (real) path and on the array path with and without a defence (group geometric median).

Results carry the machine description (CPU count, platform, NumPy version), so numbers from
different machines can be told apart. Only NumPy is used; nothing runs on a GPU.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from aircompkit import __version__


def _best(fn: Callable[[], Any], repeat: int) -> float:
    fn()  # warm-up (imports, caches)
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def machine() -> dict[str, Any]:
    return {"aircompkit": __version__, "numpy": np.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count()}


def bench_mse(quick: bool, repeat: int) -> list[dict[str, Any]]:
    from aircompkit.core.scenario import Scenario

    rows = []
    trials = 200 if quick else 2000
    for K in ((20, 200) if quick else (20, 100, 1000)):
        for pc in ("channel_inversion", "optimal"):
            for dim in (0, 32):
                sc = Scenario(devices={"name": "uniform_disc", "K": K, "r_min": 10, "r_max": 100},
                              channel={"name": "rayleigh", "pathloss": {"exponent": 3.0, "ref_gain_db": -30}},
                              power_control=pc, receiver="eigen", noise_var=1e-10, seed=0)
                t = _best(partial(sc.run, trials=trials, dim=dim), repeat)
                rows.append({"K": K, "power_control": pc, "dim": dim, "trials": trials,
                             "seconds": t, "trials_per_s": trials / t})
    return rows


def bench_receiver(quick: bool, repeat: int) -> list[dict[str, Any]]:
    from aircompkit.core.registry import build
    from aircompkit.core.rng import SeedTree
    from aircompkit.core.scenario import Scenario

    rows = []
    trials = 20 if quick else 100
    for N in ((4, 16) if quick else (4, 8, 16, 32)):
        sc = Scenario(devices={"name": "fixed", "K": 20}, channel={"name": "rayleigh", "n_rx": N},
                      power_control="channel_inversion", symbols="complex", seed=0)
        h, _ = sc.channel.sample(SeedTree(0).generator("channel"), trials, 20)
        p_max = np.ones(20)
        for name in ("eigen", "maxmin", "sdr", "joint_mmse"):
            rx = build("receiver", name)
            t = _best(partial(rx, h, p_max, 1.0), repeat)
            rows.append({"receiver": name, "K": 20, "N": N, "trials": trials,
                         "ms_per_realisation": 1e3 * t / trials})
    return rows


def bench_fl(quick: bool, repeat: int) -> list[dict[str, Any]]:
    from aircompkit.core.scenario import Scenario
    from aircompkit.learning import FederatedConfig, train
    from aircompkit.learning.data import synthetic

    data = synthetic(n_train=4000, n_test=1000, dim=784, n_classes=10, seed=0)  # MNIST-sized
    rounds = 5 if quick else 20
    base: dict[str, Any] = {"dataset": data, "model": {"name": "mlp", "hidden": [64]}, "rounds": rounds,
                            "eval_every": rounds, "batch_size": 32, "lr": 0.1, "seed": 0}
    real = Scenario(devices={"name": "fixed", "K": 20, "p_max": 1.0}, channel={"name": "rayleigh", "n_rx": 4},
                    power_control="channel_inversion", receiver="eigen", noise_var=0.1, seed=0)
    arr = real.replace(receiver="maxmin", symbols="complex", channel={"name": "rayleigh", "n_rx": 16})
    threat = {"attack": {"name": "sign_flip", "boost_db": 10}, "n_attackers": 3}
    cases: list[tuple[str, Scenario, dict[str, Any]]] = [
        ("broadband, eigen, N=4", real, {"subchannels": 8}),
        ("array, maxmin, N=16", arr, {"clip": 0.2}),
        ("array + group GM, 3 attackers", arr, {"clip": 0.2, "threat": threat, "defense": "group_gm"})]
    rows = []
    for label, sc, kw in cases:
        t = _best(partial(train, sc, FederatedConfig(**{**base, **kw})), repeat)
        rows.append({"case": label, "K": 20, "params": None, "rounds": rounds,
                     "rounds_per_s": rounds / t})
    from aircompkit.learning.models import MLP

    n_params = MLP(d_in=data.dim, n_classes=data.n_classes, hidden=[64]).size
    for r in rows:
        r["params"] = n_params
    return rows


GROUPS: dict[str, Callable[[bool, int], list[dict[str, Any]]]] = {
    "mse": bench_mse, "receiver": bench_receiver, "fl": bench_fl}


def run(groups=tuple(GROUPS), quick: bool = False, repeat: int = 3,
        out: str | Path | None = None) -> dict[str, Any]:
    res: dict[str, Any] = {"machine": machine(), "quick": quick, "repeat": repeat}
    for g in groups:
        res[g] = GROUPS[g](quick, repeat)
        print(format_group(g, res[g]), flush=True)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "bench.json").write_text(json.dumps(res, indent=2))
        (out / "bench.md").write_text(to_markdown(res))
        print(f"saved {out / 'bench.json'} and {out / 'bench.md'}")
    return res


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def format_group(name: str, rows: list[dict[str, Any]]) -> str:
    cols = [c for c in rows[0] if c != "seconds"]
    lines = [f"[{name}]", "  " + " | ".join(cols)]
    lines += ["  " + " | ".join(_fmt(r[c]) for c in cols) for r in rows]
    return "\n".join(lines)


def to_markdown(res: dict[str, Any]) -> str:
    m = res["machine"]
    head = (f"# AirCompKit benchmarks\n\n{m['processor']}, {m['cpu_count']} logical CPUs, "
            f"{m['platform']}, Python {m['python']}, NumPy {m['numpy']}, "
            f"AirCompKit {m['aircompkit']}. Best of {res['repeat']} runs.\n")
    out = [head]
    for g in GROUPS:
        if g not in res:
            continue
        rows = res[g]
        cols = [c for c in rows[0] if c != "seconds"]
        out.append(f"\n## {g}\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        out += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"
