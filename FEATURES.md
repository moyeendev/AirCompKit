# AirCompKit — feature specification

Priority tiers:
- **MVP**: needed for a credible first public release (end of P3)
- **v1.0**: needed for the SoftwareX submission (end of P6)
- **Future**: listed in the paper as future work. Do not build before submission.

★ marks features that support the main novelty claims.

---

## 1. Core and scenario

| ID | Feature | Tier |
|----|---------|------|
| C1 | `Scenario` object: devices + channel + AirComp scheme + receiver + adversary + defence + task, built from a dataclass or YAML ★N6 | MVP |
| C2 | Plug-in registry (`@register(kind, name)`) and `aircompkit.list("attack")` discovery | MVP |
| C3 | Seed tree: one master seed → independent streams for channel, data, noise and adversary (changing the attack doesn't change the channel draws) | MVP |
| C4 | Backend switch: NumPy (default) / PyTorch CPU / PyTorch CUDA | v1.0 |
| C5 | Batched Monte Carlo: all components operate on a leading `trials` axis | MVP |
| C6 | Results container: arrays + config + package version + git hash, saved as `.npz`/JSON. `Result.load()` | MVP |
| C7 | CLI: `aircompkit run`, `aircompkit sweep` (grid over any config key), `aircompkit reproduce`, `aircompkit list` | v1.0 |
| C8 | Parallel parameter sweeps with progress bar and resume-on-crash | v1.0 |

## 2. Devices and topology

| ID | Feature | Tier |
|----|---------|------|
| D1 | K devices with per-device peak and average power budgets | MVP |
| D2 | Placement: fixed list, uniform disc/annulus, clustered (Poisson cluster) | MVP |
| D3 | Heterogeneous devices: per-device antennas, power, compute speed, dataset size | v1.0 |
| D4 | Device participation: scheduling/selection policies (all, random-m, channel-threshold, age-based) | v1.0 |
| D5 | Energy accounting per device per round | v1.0 |
| D6 | Mobility models (random waypoint) with time-correlated channels | Future |

## 3. Channel models

| ID | Feature | Tier |
|----|---------|------|
| H1 | AWGN | MVP |
| H2 | Block Rayleigh and Rician (K-factor) fading, i.i.d. across rounds | MVP |
| H3 | Large-scale path loss (log-distance, 3GPP UMi/UMa simplified) + log-normal shadowing | MVP |
| H4 | MIMO / SIMO: N_rx server antennas, N_tx device antennas, spatial correlation (exponential, one-ring) | MVP |
| H5 | Time correlation (Gauss–Markov / Jakes) across rounds | v1.0 |
| H6 | Imperfect CSI: Gaussian estimation error with variance tied to pilot SNR, outdated CSI | MVP |
| H7 | Synchronisation impairments: timing offset, carrier frequency offset, residual phase error | v1.0 |
| H8 | Broadband / OFDM: per-subcarrier fading, parameter-to-subcarrier mapping | v1.0 |
| H9 | Load measured channels (`.mat`/`.npy`) and Sionna/QuaDRiGa exports | Future |
| H10 | RIS-assisted channels | Future |
| H11 | LEO satellite / NTN channels | Future |

## 4. AirComp transmitter and receiver

| ID | Feature | Tier |
|----|---------|------|
| A1 | Target functions: sum, weighted sum, mean, and nomographic functions (pre-/post-processing hooks) | MVP |
| A2 | Pre-processing: normalisation (mean/var per round, as in OTA-FL), clipping, sparsification (top-k), quantization | MVP |
| A3 | Power control: full power, truncated channel inversion, MSE-optimal (threshold-based), max-min fairness | MVP |
| A4 | Receive combiners: matched filter, zero-forcing-style uniform forcing, MMSE, SDP/SCA-designed beamformers | MVP (MF, UF, MMSE). v1.0 (SDP/SCA) |
| A5 | Joint device selection + beamforming (e.g. DC programming / Gibbs sampling baselines) | v1.0 |
| A6 | Digital AirComp variants (e.g. one-bit / majority-vote signSGD over the air) | v1.0 |
| A7 | Analog vs. digital (orthogonal access + quantization) baseline for comparison | v1.0 |
| A8 | Over-the-air computation of arbitrary vectors (Monte Carlo mode, no learning) | MVP |
| A9 | Pluggable estimators at the receiver (de-scaling, denoising) | v1.0 |

## 5. Adversary ★N1 ★N2

### 5.1 Threat model (configuration object)

| ID | Feature | Tier |
|----|---------|------|
| T1 | Attacker set: fixed indices, fraction ρ, or random per round. Colluding vs. independent | MVP |
| T2 | Knowledge flags: own CSI / all CSI / receiver beamformer / honest gradients / defence algorithm | MVP |
| T3 | Capability flags: payload, transmit power (up to a budget), phase/timing, pilot transmission, external jammer | MVP |
| T4 | Attack schedule: always, intermittent, start-at-round-t, adaptive to detection | v1.0 |
| T5 | Threat-model validation: reject attacks that need capabilities the threat model does not grant | v1.0 |

### 5.2 Attack library

| ID | Attack | Layer | Tier |
|----|--------|-------|------|
| X1 | Gaussian noise / random update | Payload | MVP |
| X2 | Sign flipping, scaled gradient | Payload | MVP |
| X3 | ALIE ("a little is enough") and inner-product manipulation | Payload | v1.0 |
| X4 | Label flipping / data poisoning (in learning mode) | Data | MVP |
| X5 | Backdoor / targeted poisoning (in learning mode) | Data | v1.0 |
| X6 | **Power boosting**: transmit above the prescribed pre-scaling to dominate the superposition | Physical | MVP |
| X7 | **CSI / pilot spoofing**: corrupt the server's channel estimate so that power control or beamforming misbehaves | Physical | v1.0 |
| X8 | **Phase / timing misalignment** attack (destructive superposition) | Physical | v1.0 |
| X9 | **Jamming**: external jammer with power budget, Gaussian or tone, constant or reactive | Physical | MVP |
| X10 | **Beamformer-aware attack**: align the malicious signal with the receive beamformer | Physical | v1.0 |
| X11 | **Eavesdropper**: passive receiver that tries to estimate the aggregate or individual updates | Physical | v1.0 |
| X12 | Gradient inversion by the eavesdropper/server (privacy attack, learning mode) | Privacy | Future |

## 6. Defences ★N1

| ID | Feature | Tier |
|----|---------|------|
| F1 | Digital-domain robust aggregators as baselines (with orthogonal access): coordinate median, trimmed mean, Krum, geometric median, centered clipping | MVP |
| F2 | Group-based robust AirComp: random grouping → one AirComp per group → robust aggregation over groups | MVP |
| F3 | **Detection-based spatial defences**: plug-in detectors + null-steering combiner (genie oracle built in) | MVP |
| F4 | Received-power / energy-anomaly detection, pilot-consistency checks | v1.0 |
| F5 | Aggregate clipping and norm-bounding at the server | MVP |
| F6 | Reputation / blacklist with forgiveness (pruning) over rounds | v1.0 |
| F7 | Anti-jamming: receive null steering using spare antennas | v1.0 |
| F8 | Differential privacy: artificial noise injection + accountant that credits channel noise (Gaussian mechanism, moments/RDP accounting over rounds) | v1.0 |
| F9 | Secure AirComp against eavesdroppers: artificial noise / cooperative jamming | Future |
| F10 | Defence API: `Defense.fit(history)` / `Defense.combine(y, ctx)` so that users can drop in new defences | MVP |

## 7. Learning mode

| ID | Feature | Tier |
|----|---------|------|
| L1 | FL loop: FedSGD and FedAvg (E local epochs), learning-rate schedules | MVP |
| L2 | Datasets: MNIST, Fashion-MNIST, CIFAR-10 (auto-download with checksum), synthetic linear/logistic regression | MVP (MNIST, FMNIST, synthetic). v1.0 (CIFAR-10) |
| L3 | Partitioners: IID, label-Dirichlet(α), shards, quantity skew | MVP |
| L4 | Models: NumPy MLP / logistic regression (no-PyTorch install), PyTorch CNN, ResNet-small | MVP (NumPy). v1.0 (PyTorch) |
| L5 | **Gradient bank**: record honest gradients for a run once, replay through many channel/attack/defence realisations ★N4 | MVP |
| L6 | Ideal (noise-free, perfect aggregation) and genie-aided baselines for every run | MVP |
| L7 | Bring-your-own model/dataset via a two-method interface (`grad(params, batch)`, `evaluate(params)`) | v1.0 |
| L8 | Decentralised / hierarchical OTA-FL (D2D, edge-cloud) | Future |
| L9 | Split learning / federated distillation over the air | Future |

## 8. Metrics ★N3

| ID | Metric | Tier |
|----|--------|------|
| M1 | Aggregation MSE, NMSE, per-coordinate error, outage probability | MVP |
| M2 | Learning: test accuracy, loss, rounds-to-target-accuracy, attack success rate (backdoor) | MVP |
| M3 | Detection: TPR/FPR per round, ROC, AUC, detection delay | MVP |
| M4 | Privacy: DP ε(δ) per round and cumulative, eavesdropper MSE | v1.0 |
| M5 | Resource: transmit energy, channel uses / latency, runtime | v1.0 |
| M6 | Confidence intervals over seeds (bootstrap / t-interval) in every plot | MVP |

## 9. Reproduction and benchmark suite ★N5

| ID | Feature | Tier |
|----|---------|------|
| R1 | `aircompkit reproduce <id>`: one module per paper, with exact config and a reference-curve file | v1.0 |
| R2 | At least 5 reproduced works , including one security paper | v1.0 |
| R3 | `--quick` mode (fewer trials/rounds) for CI. `--full` mode for paper-quality figures | v1.0 |
| R4 | CI regression test: quick-mode curves within tolerance of stored references | v1.0 |
| R5 | Standard "AirCompKit-Bench" scenario set (e.g. K ∈ {20, 50, 100}, N ∈ {1, 4, 16}, ρ ∈ {0, 0.1, 0.2}) with a leaderboard table in the docs | Future (v1.1) |

## 10. Usability, documentation, quality

| ID | Feature | Tier |
|----|---------|------|
| U1 | `pip install aircompkit` (NumPy-only core). Extras `[torch]`, `[plot]`, `[all]` | MVP |
| U2 | Quickstart: under 15 lines from install to an "accuracy under attack with/without defence" plot | MVP |
| U3 | Tutorials (notebooks): (1) AirComp basics & MSE, (2) power control, (3) MIMO beamforming, (4) OTA-FL, (5) attacks, (6) writing your own defence, (7) DP over the air | v1.0 |
| U4 | API reference (auto-generated docstrings), hosted docs | v1.0 |
| U5 | Plot helpers with consistent style (`aircompkit.plot.curves`, `.roc`, `.sweep`) | MVP |
| U6 | Tests: unit + analytical + property-based + reproduction smoke tests. ≥ 80 % core coverage | v1.0 |
| U7 | CI on Windows/Linux/macOS, Python 3.10–3.13 | v1.0 |
| U8 | `CITATION.cff`, `CONTRIBUTING.md`, issue templates, changelog, semantic versioning | v1.0 |
| U9 | Runtime benchmark script for the paper (scaling in K, N, d, trials, CPU vs. GPU) | v1.0 |
| U10 | Optional GUI / web dashboard for teaching | Future |

---

## Target quickstart (API sketch, to be finalised)

```python
import aircompkit as ac

sc = ac.Scenario(
    devices=ac.devices.Uniform(K=40, radius=100.0, p_max=1.0),
    channel=ac.channel.Rayleigh(n_rx=16, csi_error=0.05, pathloss="log-distance"),
    aircomp=ac.aircomp.UniformForcing(power_control="truncated-inversion"),
    adversary=ac.Adversary(
        fraction=0.1,
        knows={"own_csi"},
        attack=ac.attacks.PowerBoost(factor=10.0),
    ),
    defense=ac.defense.GroupGM(groups=8),
    task=ac.learning.FedAvg(dataset="mnist", split=ac.split.Dirichlet(0.5),
                            model="mlp", rounds=200),
    seed=0,
)

res = sc.run(trials=5)                        # learning mode
ac.plot.curves(res, baselines=["ideal", "no-defense"])
```

## Scope freeze for v1.0

The submission build is everything tagged **MVP** or **v1.0**. Anything tagged **Future** goes
in the paper's "Future work" section and in GitHub issues labelled `roadmap`, not into the code.
