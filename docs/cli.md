# Command line

```text
python -m aircompkit --version
python -m aircompkit list [KIND]                    # registered components
python -m aircompkit run CONFIG [--trials --dim --batch --out]
python -m aircompkit train CONFIG [--rounds --data-root --out]
python -m aircompkit reproduce [NAME] [--quick --trials --data-root --check --out]
python -m aircompkit bench [GROUP ...] [--quick --repeat --out]
```

## `run`: a scenario from YAML

```yaml
# configs/rayleigh_optimal.yaml
devices: {name: uniform_disc, K: 20, p_max: 1.0, r_min: 10, r_max: 100}
channel: {name: rayleigh, pathloss: {exponent: 3.0, ref_gain_db: -30}}
power_control: optimal
target: mean
noise_var: 1.0e-10
seed: 0
run: {trials: 2000, dim: 32}
```

```bash
python -m aircompkit run configs/rayleigh_optimal.yaml --trials 5000 --out results/run1
```

The optional `run:` section sets defaults for `--trials`, `--dim` and `--batch`. `--out` writes
`results/run1.npz` and `results/run1.json`.

## `train`: over-the-air federated learning

A config with a `learning:` section (the fields of `FederatedConfig`) runs OTA-FL. See
[configs/otafl_synthetic.yaml](https://github.com/moyeendev/AirCompKit/blob/main/configs/otafl_synthetic.yaml).

```bash
python -m aircompkit train configs/otafl_synthetic.yaml --rounds 50 --out results/otafl
```

## `reproduce`: published results

Without a name it lists the reproductions. `--quick` uses few trials and rounds, `--check`
compares with the published values and exits with status 1 if a tolerance fails (this is what
the reproduction CI runs), and `--out` stores a JSON file and a figure. See
[Reproductions](reproductions.md).

```bash
python -m aircompkit reproduce yang2020 --check --out results/yang2020
python -m aircompkit reproduce --all --quick --check --data-root ./data
```

## `bench`: performance

Groups `mse`, `receiver`, `fl` and `detect` (default: all). `--out` writes `bench.json` and a
Markdown table `bench.md`. See [Benchmarks](benchmarks.md).
