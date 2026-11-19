# AirCompKit

**A security-aware simulator for over-the-air computation (AirComp) and over-the-air federated
learning (OTA-FL).**

AirComp lets many devices transmit at the same time on the same resource, so that the wireless
channel computes the sum of their signals. That makes aggregation fast, but because signals mix
before the server sees them, the usual per-client defences have nothing to inspect. AirCompKit
models the **channel**, the **AirComp transceiver**, the **learning loop**, the **adversary** and
the **defence** as interchangeable components. It also checks itself against closed-form results
and against published papers.

```python
import aircompkit as ac

sc = ac.Scenario(
    devices={"name": "uniform_disc", "K": 20, "p_max": 1.0, "r_min": 10, "r_max": 100},
    channel={"name": "rayleigh", "pathloss": {"exponent": 3.0, "ref_gain_db": -30}},
    power_control="optimal",
    target="mean",
    noise_var=1e-10,
    seed=0,
)
res = sc.run(trials=2000, dim=32)   # analytical + signal-level Monte Carlo
print(res.summary()["mse"])         # (mean, 95% CI half-width)
```

## What is in the box

| Area | Components |
|---|---|
| Devices and channels | fixed or random placement, path loss and shadowing, AWGN / Rayleigh / Rician fading, multi-antenna server with spatial correlation, imperfect CSI, phase errors |
| AirComp transceiver | channel inversion, truncated inversion (peak or average budget), full power, optimal power control; receivers `eigen`, `maxmin`, `sdr` (with a certified bound), `joint_mmse` |
| Learning | NumPy MLP / logistic regression, FedSGD / FedAvg, MNIST / Fashion-MNIST (local files) and a synthetic task, IID / shards / Dirichlet splits, gradient bank |
| Adversary | threat model with knowledge flags and power caps; Gaussian, sign-flip, scaled, colluding, misalignment, label-flip and CSI-spoofing attacks; external jammer |
| Defences | group geometric median over repeated AirComp, digital robust rules (median, trimmed mean, Krum, geometric median), a genie oracle with null steering, and an interface for detection-based defences |
| Evaluation | MSE (analytical and signal level), detection Monte Carlo, differential-privacy accounting, paper reproductions with tolerance checks, benchmarks |

Only NumPy and PyYAML are required. Every component is registered by name, so scenarios are
plain dicts or YAML files, and new components plug in with one decorator (see
[tutorial 5](tutorials/05_custom_components.ipynb)).

## Where to start

- [Installation](install.md), then the [user guide](guide.md) for the concepts.
- The [tutorials](tutorials/01_aircomp_basics.ipynb) are executed notebooks; each runs in under a minute.
- [Reproductions](reproductions.md) lists which published results AirCompKit reproduces and how closely.
- The [API reference](api/core.md) is generated from the docstrings.

## Citing

See [`CITATION.cff`](https://github.com/moyeendev/AirCompKit/blob/main/CITATION.cff). A SoftwareX
paper is in preparation.
