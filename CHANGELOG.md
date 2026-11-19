# Changelog

All notable changes are recorded here. The project follows semantic versioning.

## [Unreleased]

## [1.0.0] - 2026-11-20

First public release. Everything below was developed in phases P0-P6.

### Added
- P6 release preparation:
  - example `examples/secure_otafl.py` (five defences under three attacks on MNIST, paired seeds);
  - `aircompkit reproduce --all` (runs every reproduction, skips those whose dataset is missing);
  - release workflow (tag `vX.Y.Z` → build, version check, wheel smoke test on three operating
    systems, PyPI trusted publishing, GitHub release), `.zenodo.json`, `RELEASING.md`;
  - CI: coverage gate (80 %), packaging job (`build`, `twine check`, clean-venv install).
- P5 documentation and benchmarks:
  - MkDocs site (Material + mkdocstrings + mkdocs-jupyter): home, installation, user guide,
    command line, reproductions, benchmarks, API reference; `docs` extra; GitHub Pages workflow
    that re-executes the tutorials and builds with `--strict`;
  - six executed tutorial notebooks (AirComp basics, multi-antenna and impairments, OTA-FL,
    attacks and defences, custom components, reproductions and benchmarks) and
    `docs/execute_notebooks.py`;
  - tolerance checks for every reproduction (`aircompkit.reproduce.CHECKS`,
    `aircompkit reproduce NAME --check`, `--data-root`) and a `Reproductions` workflow that runs
    them in quick mode on every push to `main` and weekly;
  - `aircompkit bench` / `aircompkit.bench` (MSE throughput, receiver design, OTA-FL rounds)
    with JSON and Markdown output;
  - public attack base class `aircompkit.adversary.Attack`;
  - 9 new tests.
- P4 adversary and defences:
  - `ThreatModel` (fixed or fractional attacker set, knowledge flags checked against what the
    attack needs, power-boost cap, external jammer);
  - attacks `gaussian`, `sign_flip`, `scaled`, `collude`, `misalign`, `label_flip`, `csi_spoof`;
  - array-level OTA-FL path (`symbols="complex"`): real updates packed into complex symbols,
    one block-fading use observed at every antenna, `Receiver.transmit` /
    `transmit_orthogonal` for defences that use the channel differently;
  - defences `none`, `genie` (null steering with the true attackers), `group_gm`, `digital_robust` (mean, median, trimmed
    mean, Krum, geometric median over orthogonal access);
  - `defense.evaluate.detection_sweep` (detection Monte Carlo on a gradient bank);
  - `metrics.privacy`: RDP accountant for the Gaussian mechanism provided by receiver noise;
  - FL options `clip`, `threat`, `defense`; per-round `det_tp`, `det_fp`, `channel_uses`;
  - 22 new tests.

### Changed
- Docstrings use plain ``code`` literals instead of Sphinx roles (rendered by mkdocstrings).
- `learning.train` no longer raises on a non-finite model (an unbounded attack can cause
  one); it zeroes the bad entries and counts them in `meta["nonfinite_rounds"]`.
- `symbols="complex"` in learning mode selects the array path instead of raising.

- P3 learning mode:
  - datasets `mnist`, `fashion_mnist` (local `.npz` or IDX files; no network access) and
    `synthetic`; partitioners `iid`, `shards`, `dirichlet`; NumPy models `mlp`, `logreg`;
  - OTA-FL loop `learning.train` (FedSGD/FedAvg, update or model transmission, broadband
    sub-channel mapping, cell-interior scheduling, ideal-aggregation baseline with paired
    randomness) over any scenario, MIMO and impairments included;
  - `aircomp_aggregate` and `GradientBank` (record, save/load, replay over fresh channels);
  - `truncated_inversion(budget="average")` and a NumPy-only exponential integral `exp1`;
  - `Scenario.sample_channels` / `Scenario.design` for per-round physical-layer use;
  - CLI `aircompkit train CONFIG [--rounds --data-root --out]`, config `configs/otafl_synthetic.yaml`;
  - reproduction `zhu2020` (Zhu, Wang & Huang, IEEE TWC 2020, Figs. 4 and 7(b));
  - 24 new tests, including OTA-FL with vanishing noise equal to ideal FedAvg.
- P2 MIMO and impairments:
  - multi-antenna server (`n_rx`, ULA), exponential spatial correlation (`corr`), LoS with random AoA;
  - `impairments`: imperfect CSI (`csi_error`) and residual phase error (`phase_error_std`),
    drawn from their own random stream;
  - receivers `eigen`, `maxmin` (solver-free max-min-fair combiner), `sdr` (max-min SDR solved
    through its dual in NumPy + Gaussian randomisation), `joint_mmse` (alternating LMMSE
    receive / optimal per-device transmit, monotone);
  - helpers `effective_gains`, `zf_mse`, `sdr_solve` (with a guaranteed bound);
  - pre-processing: shared normalisation, norm clipping, unbiased stochastic quantisation;
  - `aircompkit reproduce [NAME] [--quick --trials --out]` and the reproduction `yang2020`
    (Yang et al., IEEE TWC 2020, Figs. 4–5);
  - 29 new tests (brute-force optimality for N = 2, SDR bound, signal-level vs. analytical
    with impairments, reproduction smoke test).
- P1 computation core:
  - devices: `fixed`, `uniform_disc` (area-uniform annulus), per-device power budgets;
  - channels: `awgn`, `rayleigh`, `rician` (K-factor), log-distance path loss + log-normal shadowing;
  - power control: `channel_inversion`, `truncated_inversion` (Zhu et al. 2020),
    `full_power` (MSE-optimal receive scaling), `optimal` (exact, Cao et al. 2020);
  - target functions `sum`, `mean`; real or complex symbols;
  - `Scenario` (dict/YAML config, paired random streams), `Result` (`.npz` + `.json` with
    config and versions), analytical and signal-level MSE, CI helpers;
  - closed forms: channel inversion over AWGN, truncated inversion over Rayleigh;
  - CLI `aircompkit run CONFIG [--trials --dim --batch --out]`;
  - example `examples/mse_vs_snr.py`, config `configs/rayleigh_optimal.yaml`;
  - 36 validation tests (closed forms, brute-force optimality, signal-level vs. analytical).
- P0 skeleton: package layout, component registry, deterministic seed tree, `aircompkit` CLI
  (`--version`, `list`), tests, GitHub Actions CI, related-tools search record.
