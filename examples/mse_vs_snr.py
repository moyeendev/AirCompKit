"""AirComp MSE vs. SNR for four power-control schemes (Rayleigh, K = 20).

Also overlays the closed-form MSE of truncated channel inversion, as a validation check.
SNR = P / sigma^2 with unit-mean fading (no path loss). Target: the mean of K real values.

    python examples/mse_vs_snr.py            # writes examples/figures/mse_vs_snr.png

Channel inversion is only printed (as its *median* MSE), not plotted: under Rayleigh fading
its mean MSE is infinite (the weakest of K devices sets the receive scaling), and a median is
not comparable with the other curves' means. Per channel draw, optimal power control is never
worse than any other scheme (checked in tests/test_p1_aircomp.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import aircompkit as ac
from aircompkit.aircomp.analytic import truncated_inversion_rayleigh_mse

K, TRIALS, DIM, THR = 20, 5000, 16, 0.1
SNR_DB = np.arange(-10, 31, 5)
SCHEMES = {
    "optimal": "Optimal power control",
    "full_power": "Full power",
    "truncated_inversion": f"Truncated inversion (thr = {THR})",
    "channel_inversion": "Channel inversion (median, table only)",
}


def main() -> None:
    curves = {name: [] for name in SCHEMES}
    emp = {name: [] for name in SCHEMES}
    closed = []
    for snr_db in SNR_DB:
        p = 10 ** (snr_db / 10)
        for name in SCHEMES:
            pc = {"name": name, "threshold": THR} if name == "truncated_inversion" else name
            sc = ac.Scenario(
                devices={"name": "fixed", "K": K, "p_max": p},
                channel="rayleigh",
                power_control=pc,
                target="mean",
                noise_var=1.0,
                symbols="real",
                seed=1,
            )
            r = sc.run(trials=TRIALS, dim=DIM)
            agg = np.median if name == "channel_inversion" else np.mean
            curves[name].append(agg(r["mse"]))
            emp[name].append(agg(r["mse_empirical"]))
        closed.append(truncated_inversion_rayleigh_mse([1.0] * K, p, THR, 0.5) / K**2)

    print(f"{'SNR dB':>7} " + " ".join(f"{n[:14]:>15}" for n in SCHEMES) + "   closed-form trunc")
    for i, s in enumerate(SNR_DB):
        row = " ".join(f"{curves[n][i]:15.4g}" for n in SCHEMES)
        print(f"{s:7d} {row}   {closed[i]:.4g}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the figure")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for name, label in SCHEMES.items():
        if name == "channel_inversion":
            continue  # its median is not comparable with the others' means (see docstring)
        (line,) = ax.semilogy(SNR_DB, curves[name], marker="o", ms=4, label=label)
        ax.semilogy(SNR_DB, emp[name], ls="none", marker="x", ms=5, color=line.get_color())
    ax.semilogy(SNR_DB, closed, "k--", lw=1, label="Truncated inversion, closed form")
    ax.set_xlabel("SNR  $P/\\sigma^2$ (dB)")
    ax.set_ylabel("MSE of the mean")
    ax.set_title(f"AirComp over Rayleigh fading, K = {K}\n(o analytical, x signal-level)",
                 fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = Path(__file__).parent / "figures" / "mse_vs_snr.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
