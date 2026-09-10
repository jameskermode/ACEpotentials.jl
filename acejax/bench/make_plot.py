"""Scaling plot in the shape of the upstream lammps-jax benchmark chart.

Throughput in timesteps/s against atom count, log-log.  One panel per basis
size, so the size dependence of the pace/jax ratio is readable directly.
Numbers come from results.md; see there for what the measurement excludes.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

N = [64, 216, 512, 1000, 1728, 4096]

# atom-steps/s, from results.md.  jax capacities at 1.35x edge margin.
DATA = {
    110: {"jax_f64":  [5.34e4, 3.03e4, 1.50e5, 1.85e5, 2.14e5, 1.98e5],
          "jax_f32":  [1.02e5, 2.00e5, 3.05e5, 3.29e5, 3.89e5, 3.51e5],
          "pace":     [1.77e5, 4.61e5, 7.18e5, 8.85e5, 1.08e6, 1.14e6]},
    211: {"jax_f64":  [2.55e4, 1.95e4, 6.76e4, 7.57e4, 8.45e4, 9.60e4],
          "jax_f32":  [7.21e4, 1.09e5, 1.82e5, 1.83e5, 2.06e5, 1.62e5],
          "pace":     [1.36e5, 2.55e5, 3.42e5, 3.77e5, 3.94e5, 4.01e5]},
}
SW_N = [64, 512, 1728, 4096]
SW = [2.74e5, 1.76e6, 5.15e6, 1.27e7]

ts = lambda n, a: [ai / ni for ni, ai in zip(n, a)]

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#2e9e6b", "#8a8a85"
INK, INK2, SURF = "#1a1a19", "#52514e", "#fcfcfb"

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
fig.patch.set_facecolor(SURF)

for ax, nb in zip(axes, [110, 211]):
    d = DATA[nb]
    ax.set_facecolor(SURF)
    ax.plot(SW_N, ts(SW_N, SW), "-", color=GREY, lw=1.4, marker="s", ms=4.5,
            label="Stillinger-Weber sw/kk (classical 3-body)")
    ax.plot(N, ts(N, d["pace"]), "-", color=GREEN, lw=2.0, marker="D", ms=5,
            label="pair pace/kk (f64)")
    ax.plot(N, ts(N, d["jax_f32"]), "-", color=ORANGE, lw=2.0, marker="o", ms=5,
            label="pair jax/kk, f32")
    ax.plot(N, ts(N, d["jax_f64"]), "-", color=BLUE, lw=2.0, marker="o", ms=5,
            label="pair jax/kk, f64")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of atoms", color=INK2, fontsize=11)
    ax.grid(True, which="major", ls="-", lw=0.5, color="#e6e5e2")
    ax.grid(True, which="minor", ls="-", lw=0.3, color="#f0efec")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(colors=INK2, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color("#dedcd8")
    n_pace = 99 if nb == 110 else 211
    ax.set_title(f"acejax {nb} basis functions   vs   pace {n_pace}",
                 color=INK, fontsize=11.5, pad=8)

axes[0].set_ylabel("Throughput [timesteps/s]", color=INK2, fontsize=11)

# the f64 dip at 216 atoms is a bundle-capacity shape effect, not physics
# the f64 dip at 216 atoms is a bundle-capacity shape effect, not physics
for ax, nb in zip(axes, [110, 211]):
    ax.annotate("capacity shape effect", xy=(216, DATA[nb]["jax_f64"][1] / 216),
                xytext=(18, -26), textcoords="offset points",
                fontsize=8.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.06),
           ncol=4, frameon=False, fontsize=9.5, labelcolor=INK2)
fig.suptitle("ACE throughput on an RTX A4500 — single-point evaluation "
             "(timestep 0.0, no reneighbouring)",
             color=INK, fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(pathlib.Path(__file__).resolve().parent / "scaling.png", dpi=170, bbox_inches="tight", facecolor=SURF)
print("wrote scaling.png")
