"""Over-the-air federated learning loop (FEATURES L1, L6).

Each round:

1. the server schedules devices (all, or the cell interior ``distance <= max_distance``,
   Zhu et al.'s Scheme 1);
2. every scheduled device runs ``local_steps`` minibatch SGD steps from the global model
   (``local_steps = 1``: FedSGD; more: FedAvg);
3. the devices send their update ``w_k - w`` (``transmit="update"``) or local model ``w_k``
   (``transmit="model"``) through the scenario's physical layer, one fading draw per
   sub-channel; ``aggregation="ideal"`` replaces the channel by the exact mean;
4. the server applies the estimated mean.

The same seed gives identical data splits, initialisation and minibatches for ``ideal`` and
``aircomp`` runs, so their difference is due to the channel alone.
"""

from __future__ import annotations

import dataclasses
import platform
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aircompkit.core.registry import build, get
from aircompkit.core.result import Result
from aircompkit.core.rng import SeedTree
from aircompkit.learning.aggregate import aircomp_aggregate
from aircompkit.learning.bank import GradientBank
from aircompkit.learning.data import Dataset


@dataclass
class FederatedConfig:
    dataset: Any = "synthetic"
    partition: Any = "iid"
    model: Any = field(default_factory=lambda: {"name": "mlp", "hidden": [64]})
    rounds: int = 100
    local_steps: int = 1
    batch_size: int = 32
    lr: float = 0.1
    eval_every: int = 10
    aggregation: str = "aircomp"
    transmit: str = "update"
    subchannels: int = 1
    max_distance: float | None = None
    normalize: bool = True
    record_bank: list[int] = field(default_factory=list)
    clip: float | None = None
    threat: Any = None
    defense: Any = "none"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.aggregation not in ("aircomp", "ideal"):
            raise ValueError("aggregation must be 'aircomp' or 'ideal'")
        if self.transmit not in ("update", "model"):
            raise ValueError("transmit must be 'update' or 'model'")
        if min(self.rounds, self.local_steps, self.batch_size, self.eval_every,
               self.subchannels) < 1:
            raise ValueError("rounds, local_steps, batch_size, eval_every, subchannels must be >= 1")
        if self.clip is not None and self.clip <= 0:
            raise ValueError("clip must be > 0")

    def to_dict(self) -> dict[str, Any]:
        from aircompkit.adversary import ThreatModel
        from aircompkit.core.registry import spec_of

        plain = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        if isinstance(self.dataset, Dataset):
            plain["dataset"] = self.dataset.name or "custom"
        if isinstance(self.threat, ThreatModel):
            plain["threat"] = self.threat.to_dict()
        if not isinstance(self.defense, (str, dict)):
            plain["defense"] = spec_of("defense", self.defense)
        return plain


def _dataset(spec: Any, data_root: str | None) -> Dataset:
    if isinstance(spec, Dataset):
        return spec
    if isinstance(spec, str):
        spec = {"name": spec}
    params = dict(spec)
    fn = get("dataset", params.pop("name"))
    if fn.__name__ in ("mnist", "fashion_mnist") and data_root is not None:
        params.setdefault("root", data_root)
    return fn(**params)


def _model(spec: Any, data: Dataset):
    if not isinstance(spec, (str, dict)):
        return spec
    params = {"name": spec} if isinstance(spec, str) else dict(spec)
    return get("model", params.pop("name"))(d_in=data.dim, n_classes=data.n_classes, **params)


def train(
    scenario,
    cfg: FederatedConfig | dict[str, Any],
    data: Dataset | None = None,
    bank: GradientBank | None = None,
    data_root: str | None = None,
    verbose: bool = False,
) -> Result:
    """Run OTA-FL on ``scenario``'s physical layer. Returns a ``Result`` with arrays
    ``eval_round``, ``test_acc``, ``test_loss`` (per evaluation), ``train_loss``,
    ``n_scheduled``, ``agg_nmse`` (per round; NMSE of the received vs. exact honest mean),
    and on the array path ``det_tp``, ``det_fp`` (flagged attackers / honest devices) and
    ``channel_uses``.

    Two physical paths:

    * ``symbols="real"`` (broadband): one real parameter per sub-channel use, scalar
      combining (P3); no adversary or defence;
    * ``symbols="complex"`` (array): parameters packed two per complex symbol, one
      block-fading channel use per round observed at every antenna; supports ``threat`` and
      ``defense`` (P4).
    """
    from aircompkit.adversary import as_threat
    from aircompkit.learning.array import Receiver, normalise, pack, unpack

    if isinstance(cfg, dict):
        cfg = FederatedConfig(**cfg)
    threat = as_threat(cfg.threat)
    defense = build("defense", cfg.defense)
    array_path = scenario.symbols == "complex"
    if not array_path and (threat is not None or type(defense).__name__ != "NoDefense"):
        raise ValueError("attacks and defences need the array path: use symbols='complex'")
    data = data if data is not None else _dataset(cfg.dataset, data_root)
    model = _model(cfg.model, data)
    K = scenario.devices.K

    tree = SeedTree(cfg.seed)
    phys = SeedTree(scenario.seed)
    dist = scenario.devices.sample_distances(phys.generator("devices"))
    p_max = scenario.devices.power_budgets()
    parts = build("partitioner", cfg.partition)(data.y_train, K, tree.generator("partition"))
    if cfg.max_distance is None:
        sched = np.arange(K)
    else:
        if dist is None:
            raise ValueError("max_distance needs devices with distances (e.g. uniform_disc)")
        sched = np.flatnonzero(dist <= cfg.max_distance)
    if len(sched) == 0:
        raise ValueError("no device is scheduled; increase max_distance")

    # attackers: scenario device ids, and their positions among the scheduled devices
    byz = threat.attackers(K) if threat is not None else np.zeros(0, int)
    byz_local = np.flatnonzero(np.isin(sched, byz))
    honest_local = np.setdiff1d(np.arange(len(sched)), byz_local)
    attack = threat.attack if threat is not None else None
    defense.reset(len(sched))

    w = model.init(tree.generator("model"))
    rng_sgd = tree.generator("sgd")
    eval_round, test_acc, test_loss = [], [], []
    train_loss = np.empty(cfg.rounds)
    nmse = np.zeros(cfg.rounds)
    det_tp = np.zeros(cfg.rounds)
    det_fp = np.zeros(cfg.rounds)
    uses = np.zeros(cfg.rounds)
    nonfinite = 0
    byz_set = set(byz_local.tolist())

    for t in range(cfg.rounds):
        X = np.empty((len(sched), w.size))
        losses = []
        for i, k in enumerate(sched):
            wk = w.copy()
            idx = parts[k]
            poison = attack is not None and attack.poisons_labels and i in byz_set
            for step in range(cfg.local_steps):
                bsel = idx[rng_sgd.integers(0, len(idx), size=min(cfg.batch_size, len(idx)))]
                y = data.y_train[bsel]
                if poison:
                    y = attack.flip_labels(y, data.n_classes)  # type: ignore[union-attr]
                loss, g = model.loss_grad(wk, data.x_train[bsel], y)
                if step == 0:
                    losses.append(loss)
                wk -= cfg.lr * g
            X[i] = wk - w if cfg.transmit == "update" else wk
            if cfg.clip is not None:
                X[i] *= min(1.0, cfg.clip / (np.linalg.norm(X[i]) + 1e-12))
        train_loss[t] = float(np.mean(losses))
        if bank is not None and t in cfg.record_bank:
            bank.record(t, sched, X)

        ideal = X[honest_local].mean(axis=0)
        node = phys.child("round").child(str(t))
        if cfg.aggregation == "ideal":
            agg = ideal
        elif array_path:
            x, nu = normalise(X, cfg.clip)
            rx = Receiver(scenario, x, p_max, dist, node, attackers=byz_local,
                          attack=attack, gref=pack(ideal),
                          jammer=threat.jammer if threat is not None else None,
                          device_ids=sched)
            est, keff, flagged = defense.aggregate(rx, {"attackers": byz_local})
            agg = nu * unpack(est, w.size) / keff
            det_tp[t] = len(flagged & byz_set)
            det_fp[t] = len(flagged - byz_set)
            uses[t] = rx.uses
            nmse[t] = np.sum((agg - ideal) ** 2) / max(np.sum(ideal**2), 1e-300)
        else:
            h_act, h_est, beta = scenario.sample_channels(node, cfg.subchannels, dist)
            lay = scenario.design(h_act[:, sched], h_est[:, sched], beta[:, sched], p_max[sched])
            agg = aircomp_aggregate(X, lay["amp"], lay["eta"], lay["p"] > 0,
                                    scenario.noise_var_eff, node.generator("noise"),
                                    normalize=cfg.normalize)
            uses[t] = cfg.subchannels
            nmse[t] = np.sum((agg - ideal) ** 2) / max(np.sum(ideal**2), 1e-300)
        w = w + agg if cfg.transmit == "update" else agg
        if not np.all(np.isfinite(w)):  # an unbounded attack can blow the model up
            nonfinite += 1
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        if (t + 1) % cfg.eval_every == 0 or t == cfg.rounds - 1:
            lo, acc = model.evaluate(w, data.x_test, data.y_test)
            eval_round.append(t + 1)
            test_acc.append(acc)
            test_loss.append(lo)
            if verbose:
                print(f"round {t + 1:5d}  acc {acc:.4f}  loss {lo:.4f}  nmse {nmse[t]:.2e}")

    from aircompkit import __version__

    arrays = {
        "eval_round": np.asarray(eval_round, float),
        "test_acc": np.asarray(test_acc),
        "test_loss": np.asarray(test_loss),
        "train_loss": train_loss,
        "agg_nmse": nmse,
        "n_scheduled": np.full(cfg.rounds, float(len(sched))),
        "channel_uses": uses,
    }
    if array_path:
        arrays["det_tp"] = det_tp
        arrays["det_fp"] = det_fp
    meta = {"aircompkit": __version__, "numpy": np.__version__,
            "python": platform.python_version(), "model_size": int(w.size),
            "dataset": data.name, "final_params_norm": float(np.linalg.norm(w)),
            "attackers": byz.tolist(), "nonfinite_rounds": nonfinite}
    if bank is not None:
        bank.meta.update({"scenario": scenario.to_dict(), "learning": cfg.to_dict()})
    return Result(config={"scenario": scenario.to_dict(), "learning": cfg.to_dict()},
                  arrays=arrays, meta=meta)
