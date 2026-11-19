# User guide

## System model

$K$ devices hold values $x_k$ (symbols, or entries of a model update). Device $k$ sends
$b_k x_k$ over channel $h_k \in \mathbb{C}^N$ to an $N$-antenna server, and all transmit at once:

$$
y = \sum_{k=1}^K h_k b_k x_k + n, \qquad n \sim \mathcal{CN}(0, \sigma^2 I_N).
$$

The server combines with a unit-norm $m$ and scales by $1/\sqrt{\eta}$, giving
$\hat f = m^H y / \sqrt{\eta}$. With the effective scalar channel $c_k = m^H h_k$ and phase-aligned
pre-scaling $b_k = \sqrt{p_k}\, c_k^* / |c_k|$, the MSE of the target sum is

$$
\mathrm{MSE} = \sum_k \Big(\sqrt{p_k |c_k|^2 / \eta} - 1\Big)^2 + \frac{\sigma_\mathrm{eff}^2}{\eta},
$$

where $\sigma_\mathrm{eff}^2 = \sigma^2/2$ for real symbols (only the real part is used) and
$\sigma^2$ for complex ones. The first term is **misalignment**: a device that cannot fully invert
its channel within its power budget $P_k$. The second is **noise**. Power control chooses $p_k$ and
$\eta$, and the receiver chooses $m$. For the `mean` target everything is scaled by $1/K$.

## Scenarios

A `Scenario` names one component per slot:

| field | kind | built-in names |
|---|---|---|
| `devices` | `devices` | `fixed`, `uniform_disc` |
| `channel` | `channel` | `awgn`, `rayleigh`, `rician` (+ `pathloss`, `n_rx`, `corr`) |
| `power_control` | `power_control` | `channel_inversion`, `truncated_inversion`, `full_power`, `optimal` |
| `receiver` | `receiver` | `eigen`, `maxmin`, `sdr`, `joint_mmse` |
| `impairments` | | `{"csi_error": eps2, "phase_error_std": rad}` |
| `target` | | `sum`, `mean` |
| `noise_var`, `symbols`, `seed` | | `symbols` is `real` or `complex` |

A slot accepts a name (`"optimal"`), a dict with a `name` key and parameters
(`{"name": "truncated_inversion", "threshold": 0.1}`) or a component instance. `Scenario.to_dict`
and `to_yaml` write the fully resolved configuration, defaults included, so a saved result says
exactly what ran.

`Scenario.run(trials, dim)` draws `trials` independent channel realisations and returns a
`Result` with per-trial arrays: `mse` (analytical, given the channel), `mse_empirical` (simulated
transmission of `dim`-dimensional random symbols, when `dim > 0`), `eta`, `n_active` and `power`.
`Result.summary()` gives means with 95 % confidence intervals, and `Result.save(path)` writes
`path.npz` plus `path.json` with the configuration and package versions.

## Reproducible randomness

All randomness comes from a **seed tree**: `SeedTree(seed).generator("channel")` derives an
independent stream from the seed and a *name*, not from the order of calls. The channel, device
placement, impairments, data partition, mini-batches and attackers each draw from their own named
stream. Two consequences follow:

- **paired comparisons:** changing the power control, receiver or defence keeps the channel draws
  identical, so differences between schemes are not Monte Carlo noise;
- adding an impairment does not shift the channel draws, so its effect is isolated.

## Learning mode

`learning.train(scenario, FederatedConfig(...))` runs FedSGD or FedAvg (`local_steps`) over the
scenario. Each round draws fresh channels, schedules the devices inside `max_distance`, and
aggregates the updates (or models, `transmit="model"`) over the air. Two physical paths exist:

- **broadband** (`symbols="real"`): update entries are mapped onto `subchannels` independent
  sub-channels and combined with the scenario's receiver;
- **array** (`symbols="complex"`): two real entries form one complex symbol, and the server sees
  the whole $N \times L$ received block. Attacks and defences need this path, because detection
  works on the antenna array before combining.

`aggregation="ideal"` replaces the channel by the exact mean with the same data and mini-batch
streams, which gives the error-free reference. Results hold `test_acc`, `test_loss`, `train_loss`
(at `eval_round`), and per-round `agg_nmse`, `n_scheduled` and `channel_uses`. The array path adds
`det_tp` and `det_fp`. A `GradientBank` records the updates of chosen rounds and replays them over
new channels or attacks without retraining.

## Threat model

```python
threat = {
    "attack": {"name": "collude", "boost_db": 15},   # what attackers send, power over the rule
    "n_attackers": 6,                                 # or "fraction": 0.15
    "knowledge": ["own_data", "own_csi", "honest_mean"],
    "max_boost_db": 20,                               # optional cap, checked
    "jammer": None,                                   # or {"power": ...}: external jammer
}
```

Each attack declares what it `needs`. Building a threat model whose attack needs knowledge it
does not grant is an error, so every result states its assumptions. Knowledge flags:
`own_data`, `own_csi`, `honest_mean` (the honest devices' mean update, i.e. omniscient
collusion), `receiver` (the server's combiner).

## Defences

A defence decides how the devices use the channel and how the server combines. `rx.transmit(ids)`
is one AirComp channel use, and `rx.transmit_orthogonal(ids)` gives each device its own resource
block. The defence returns an estimate of the honest sum, the effective number of devices in it,
and the set of devices it excluded. `channel_uses` in the result counts the cost: 1 for `none`
and `genie`, `groups` for `group_gm`, $K$ for `digital_robust`. A defence that also has a
`detect(Y, A, noise_var)` method can be evaluated with `defense.evaluate.detection_sweep`
([tutorial 5](tutorials/05_custom_components.ipynb) writes one).

## Extending

```python
from dataclasses import dataclass
from aircompkit import register

@register("power_control", "my_rule")
@dataclass
class MyRule:
    alpha: float = 1.0

    def __call__(self, g, beta, p_max, noise_var):
        ...                     # g: [T, K] effective gains
        return p, eta           # p: [T, K], eta: [T]
```

After the module is imported, `power_control={"name": "my_rule", "alpha": 2}` works in Python,
YAML and on the command line. The interfaces for every kind are in the API reference, and
[tutorial 5](tutorials/05_custom_components.ipynb) writes a power-control rule, an attack and a
defence.
