# AirCompKit

[![CI](https://github.com/moyeendev/AirCompKit/actions/workflows/ci.yml/badge.svg)](https://github.com/moyeendev/AirCompKit/actions/workflows/ci.yml)
[![Reproductions](https://github.com/moyeendev/AirCompKit/actions/workflows/reproduce.yml/badge.svg)](https://github.com/moyeendev/AirCompKit/actions/workflows/reproduce.yml)
[![Docs](https://github.com/moyeendev/AirCompKit/actions/workflows/docs.yml/badge.svg)](https://moyeendev.github.io/AirCompKit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.txt)
[![PyPI](https://img.shields.io/pypi/v/aircompkit.svg)](https://pypi.org/project/aircompkit/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24388026.svg)](https://doi.org/10.5281/zenodo.24388026)
[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](pyproject.toml)

**Security-aware simulator for over-the-air computation (AirComp) and over-the-air federated learning (OTA-FL).**

> Status: **v1.0.1**. AirComp over AWGN, Rayleigh and Rician
> channels (multi-antenna server, spatial correlation, path loss/shadowing, imperfect CSI,
> phase errors), four power-control schemes, four receive beamformers, over-the-air
> federated learning (FedSGD/FedAvg, NumPy models, MNIST/Fashion-MNIST/synthetic data,
> gradient bank), a threat model with seven attacks, and defences from grouped over-the-air
> aggregation to robust digital rules, plus an interface for detection-based defences. Validated against closed forms and published results, re-checked
> weekly in CI (`aircompkit reproduce NAME --check`).
>
> **Documentation:** <https://moyeendev.github.io/AirCompKit/> (user guide, six executed
> tutorial notebooks in [docs/tutorials](docs/tutorials), API reference, reproductions,
> benchmarks). See also [FEATURES.md](FEATURES.md).

## Security in three lines

```python
from aircompkit.learning import FederatedConfig, train

threat = {"attack": {"name": "collude", "boost_db": 15}, "n_attackers": 6,
          "knowledge": ["own_data", "own_csi", "honest_mean"]}   # assumptions are explicit
res = train(scenario, FederatedConfig(dataset="mnist", clip=0.2, threat=threat,
                                      defense={"name": "digital_robust", "rule": "median"}))
```

`scenario` needs a multi-antenna channel and `symbols="complex"` (the array-level path); see
[examples/secure_otafl.py](examples/secure_otafl.py) for a full setup.

AirComp lets many devices transmit at once so that the wireless channel computes their sum.
Because signals superpose before the server sees them, per-client defences no longer apply.
AirCompKit models the **channel, the AirComp transceiver, the learning loop, the adversary and
the defence** as interchangeable components, and it ships a suite that reproduces published
AirComp results.

## Install

```bash
pip install aircompkit            # or "aircompkit[plot]" for figures
```

For development:

```bash
git clone https://github.com/moyeendev/AirCompKit.git
cd AirCompKit
python -m pip install -e ".[dev]"
pytest
```

## Quickstart

```python
import aircompkit as ac

sc = ac.Scenario(
    devices={"name": "uniform_disc", "K": 20, "p_max": 1.0, "r_min": 10, "r_max": 100},
    channel={"name": "rayleigh", "pathloss": {"exponent": 3.0, "ref_gain_db": -30}},
    power_control="optimal",      # truncated_inversion | full_power | channel_inversion
    target="mean",
    noise_var=1e-10,
    seed=0,
)
res = sc.run(trials=2000, dim=32)  # analytical + signal-level Monte Carlo
print(res.summary()["mse"])        # (mean, 95% CI half-width)
res.save("results/demo")           # .npz arrays + .json with the full config
```

The same scenario as YAML is in [configs/rayleigh_optimal.yaml](configs/rayleigh_optimal.yaml).
[examples/mse_vs_snr.py](examples/mse_vs_snr.py) compares the power-control schemes against
the closed-form MSE of truncated channel inversion.
[examples/secure_otafl.py](examples/secure_otafl.py) compares the defences under three attacks on MNIST
(about 45 minutes; `--quick` for a smoke test).

## Command line

```bash
aircompkit --version
aircompkit list            # registered components per kind
aircompkit run configs/rayleigh_optimal.yaml --trials 5000 --out results/run1
aircompkit run configs/mimo_maxmin_csi.yaml --out results/mimo
aircompkit reproduce yang2020 --out results/yang2020   # Yang et al. TWC 2020, Figs. 4-5
aircompkit train configs/otafl_synthetic.yaml --out results/otafl
aircompkit reproduce zhu2020 --out results/zhu2020     # Zhu et al. TWC 2020, Figs. 4, 7(b)
aircompkit reproduce yang2020 --quick --check         # compare with the paper; exit 1 on failure
aircompkit reproduce --all --quick --check            # all reproductions (~1 min with MNIST)
aircompkit bench --out results/bench                  # performance benchmarks
python -m aircompkit list  # same as above, if your system blocks pip-generated launchers
```

MNIST and Fashion-MNIST are never downloaded automatically. Put `mnist.npz` (Keras layout)
or the four IDX files in a folder and pass `--data-root` or set `AIRCOMPKIT_DATA`.

## Repository layout

| Path | Contents |
|------|----------|
| `src/aircompkit/core` | registry, seed tree, Scenario, Result |
| `src/aircompkit/{devices,channel,aircomp,receiver}` | physical layer |
| `src/aircompkit/{adversary,defense}` | threat model, attacks, defences |
| `src/aircompkit/learning` | OTA-FL loop, datasets, gradient bank |
| `src/aircompkit/{metrics,reproduce}`, `bench.py` | evaluation, paper reproductions with tolerance checks, benchmarks |
| `configs/`, `examples/` | scenarios, examples |
| `docs/` | MkDocs site; `docs/tutorials/*.ipynb` executed tutorials (`python docs/execute_notebooks.py`) |

## Citing

Please cite the archived software, DOI [10.5281/zenodo.24388026](https://doi.org/10.5281/zenodo.24388026)
(all versions; v1.0.0 is [10.5281/zenodo.24388027](https://doi.org/10.5281/zenodo.24388027)).
See [CITATION.cff](CITATION.cff). A journal paper is in preparation.

## License

MIT. See [LICENSE.txt](LICENSE.txt).
