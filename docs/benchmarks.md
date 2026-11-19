# Benchmarks

`python -m aircompkit bench --out results/bench` times the main operations with NumPy only,
taking the best of three runs after a warm-up. The groups are:

- **mse:** throughput of `Scenario.run` in channel realisations per second, analytical only
  (`dim = 0`) and with signal-level Monte Carlo of 32-dimensional symbol vectors;
- **receiver:** combiner design time per channel realisation, $K = 20$;
- **fl:** OTA-FL rounds per second for an MNIST-sized synthetic task (784-64-10 MLP, 50,890
  parameters, $K = 20$, batch 32), on the broadband path, on the array path, and on the array
  path with group geometric median (six AirComp channel uses per round).

The numbers below come from a laptop CPU (16 logical cores, Windows 11, Python 3.13, NumPy 2.5);
timings on a laptop vary by a factor of about two with power and thermal state. The CI
workflow runs a quick version on every push to `main` and stores the results as an artifact.

**Reading the tables**

- The analytical MSE runs at about 0.3 million realisations per second for $K = 20$. A
  10,000-trial point of an MSE curve therefore takes tens of milliseconds, and the signal-level
  check adds about a factor of 6.
- The `maxmin` default (24 random starts) is the slowest receiver. For large sweeps, use
  `{"name": "maxmin", "iters": 120, "random_inits": 4}`: it is 6-9 times faster, for a mean
  MSE about 0.25 dB higher ($K = 20$, $N$ = 8 and 16). `eigen` takes well under a millisecond.
- On the array path most of the time of a round goes into the combiner design. A defence that
  transmits in groups pays one design per group, which is why group geometric median runs at
  about a quarter of the undefended speed.

## mse

| K | power_control | dim | trials | trials_per_s |
|---|---|---|---|---|
| 20 | channel_inversion | 0 | 2000 | 2.92e+05 |
| 20 | channel_inversion | 32 | 2000 | 5.43e+04 |
| 20 | optimal | 0 | 2000 | 2.77e+05 |
| 20 | optimal | 32 | 2000 | 4.64e+04 |
| 100 | channel_inversion | 0 | 2000 | 7.19e+04 |
| 100 | channel_inversion | 32 | 2000 | 1.31e+04 |
| 100 | optimal | 0 | 2000 | 5.67e+04 |
| 100 | optimal | 32 | 2000 | 1.23e+04 |
| 1000 | channel_inversion | 0 | 2000 | 7.27e+03 |
| 1000 | channel_inversion | 32 | 2000 | 1.32e+03 |
| 1000 | optimal | 0 | 2000 | 4.92e+03 |
| 1000 | optimal | 32 | 2000 | 1.22e+03 |

## receiver

| receiver | K | N | trials | ms_per_realisation |
|---|---|---|---|---|
| eigen | 20 | 4 | 100 | 0.00774 |
| maxmin | 20 | 4 | 100 | 15.9 |
| sdr | 20 | 4 | 100 | 2.6 |
| joint_mmse | 20 | 4 | 100 | 6.18 |
| eigen | 20 | 8 | 100 | 0.0233 |
| maxmin | 20 | 8 | 100 | 23.8 |
| sdr | 20 | 8 | 100 | 7.57 |
| joint_mmse | 20 | 8 | 100 | 9.45 |
| eigen | 20 | 16 | 100 | 0.0902 |
| maxmin | 20 | 16 | 100 | 43.6 |
| sdr | 20 | 16 | 100 | 28.1 |
| joint_mmse | 20 | 16 | 100 | 17.9 |
| eigen | 20 | 32 | 100 | 0.293 |
| maxmin | 20 | 32 | 100 | 114 |
| sdr | 20 | 32 | 100 | 87.5 |
| joint_mmse | 20 | 32 | 100 | 48.1 |

## fl

| case | K | params | rounds | rounds_per_s |
|---|---|---|---|---|
| broadband, eigen, N=4 | 20 | 50890 | 20 | 15.4 |
| array, maxmin, N=16 | 20 | 50890 | 20 | 7.09 |
| array + group GM, 3 attackers | 20 | 50890 | 20 | 1.95 |
