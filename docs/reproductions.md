# Reproductions

Each reproduction rebuilds a published result **through the public API** (no special code
paths), prints it next to the paper's values, and with `--check` enforces tolerances. The
[Reproductions workflow](https://github.com/moyeendev/AirCompKit/actions/workflows/reproduce.yml)
runs all of them in quick mode on every push to `main` and weekly.

Paper values for Yang et al. and Zhu et al. were read from the published figures by eye (about
±0.03).

| name | paper | full run | quick run |
|---|---|---|---|
| `yang2020` | Yang, Jiang, Shi & Ding, *Federated learning via over-the-air computation*, IEEE TWC 19(3), 2020, Figs. 4-5 | ~1 min | ~7 s |
| `zhu2020` | Zhu, Wang & Huang, *Broadband analog aggregation for low-latency federated edge learning*, IEEE TWC 19(1), 2020, Figs. 4 and 7(b) | ~10 min | ~30 s |

## Yang et al. 2020: receive beamforming feasibility

$K = 20$ devices, $h_k \sim \mathcal{CN}(0, I_N)$, zero-forcing transmitters. An MSE target
$\gamma$ is feasible when $\min_m \max_k \|m\|^2/|m^H h_k|^2 \le \gamma$.

| | result |
|---|---|
| Fig. 4 ($N = 6$) | **matches.** Feasibility is 0 at $\gamma = -5$ dB and 1 at 5 dB, as for the paper's global optimiser |
| Fig. 5 (vs. $N$) | **same trends.** AirCompKit's `maxmin` combiner is *more* often feasible than the paper's DC method: by up to 0.27 (about one antenna), mean absolute deviation 0.08. Every value is achieved by a real combiner and stays below the SDR bound, so the gap points to a weaker local method in the paper |
| SDR curve | not comparable: the paper's feasibility-form SDR is much weaker than the max-min SDR with randomisation implemented here |

Checks: Fig. 4 end points; no combiner exceeds the SDR bound; feasibility grows with $N$; mean
deviation from the paper ≤ 0.15.

## Zhu et al. 2020: truncated channel inversion and broadband OTA-FL

| | result |
|---|---|
| Fig. 4 (receive SNR vs. truncation ratio) | **exact.** $\rho_0 = P_0/(M r_\max^\alpha E_1(g_\mathrm{th}))$ gives 7.5 dB at 10 % truncation; the simulated truncation ratio and average power match the design |
| Fig. 7(b) (accuracy vs. $R_\mathrm{in}/R$, $\alpha = 3$) | **partial.** The data-limited rise reproduces (non-IID 0.882 → 0.915 from $R_\mathrm{in}/R$ = 0.3 to 0.7). The SNR-limited decline at 0.9 does not appear: OTA accuracy stays within 0.4 % of ideal aggregation, because SGD averages the zero-mean channel noise. The paper uses a 582k-parameter CNN and a tuned threshold, AirCompKit an MLP and $g_\mathrm{th} = 0.1$ |

Checks: designed $\rho_0$ equals the closed form; truncation ratio within 0.03; average power
within 25 % of the budget; with MNIST, OTA-FL within 2 points of ideal FedAvg and the non-IID rise.

## Running them

```bash
python -m aircompkit reproduce yang2020 --check --out results/yang2020
AIRCOMPKIT_DATA=./data python -m aircompkit reproduce zhu2020 --check --out results/zhu2020
```
