"""Scaling plot in the shape of the upstream lammps-jax benchmark chart.

Throughput in timesteps/s against atom count, log-log, one panel per precision,
so it can be read beside docs/plans/lammps_jax_benchmark_reference.md.  Numbers
come from results.md; see there for what the measurement excludes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# atom-steps/s from results.md, converted to timesteps/s by dividing by N.
N = [64, 216, 512, 1000, 1728, 4096]
ATOM_STEPS = {
    ("f64", "lmp135"): [5.17e4, 3.05e4, 1.49e5, 1.84e5, 2.12e5, 1.98e5],
    ("f64", "lmp200"): [4.67e4, 7.45e4, 1.10e5, 1.37e5, 1.44e5, 1.37e5],
    ("f32", "lmp135"): [1.02e5, 2.04e5, 3.13e5, 3.26e5, 3.86e5, 3.53e5],
    ("f32", "lmp200"): [8.06e4, 1.78e5, 1.81e5, 2.49e5, 2.79e5, 2.48e5],
}
# acejax-in-Python measured as ms/step
MS = {"f64": [0.433, 0.810, 2.019, 3.832, 6.839, 16.502],
      "f32": [0.202, 0.320, 0.541, 1.648, 2.830, 7.433]}
SW_N = [64, 512, 1728, 4096]                      # sw/kk, one precision only
SW_ATOM_STEPS = [2.74e5, 1.76e6, 5.15e6, 1.27e7]

ts = lambda n, a: [ai / ni for ni, ai in zip(n, a)]   # atom-steps/s -> timesteps/s

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"
INK, INK2, SURF = "#1a1a19", "#52514e", "#fcfcfb"

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
fig.patch.set_facecolor(SURF)

for ax, prec in zip(axes, ["f32", "f64"]):
    ax.set_facecolor(SURF)
    ax.plot(SW_N, ts(SW_N, SW_ATOM_STEPS), color=GREY, ls=":", lw=2,
            marker="o", ms=8, mfc=GREY, mec=SURF, mew=1.5, zorder=2,
            label="Stillinger-Weber sw/kk (classical 3-body)")
    ax.plot(N, [1000.0 / m for m in MS[prec]], color=BLUE, lw=2,
            marker="o", ms=8, mfc=BLUE, mec=SURF, mew=1.5, zorder=4,
            label="acejax, Python (no LAMMPS)")
    ax.plot(N, ts(N, ATOM_STEPS[(prec, "lmp135")]), color=ORANGE, lw=2,
            marker="o", ms=8, mfc=ORANGE, mec=SURF, mew=1.5, zorder=5,
            label="pair jax/kk, capacity 1.35x")
    ax.plot(N, ts(N, ATOM_STEPS[(prec, "lmp200")]), color=ORANGE, lw=2, ls="--",
            marker="s", ms=7, mfc=SURF, mec=ORANGE, mew=2, zorder=3,
            label="pair jax/kk, capacity 2.0x")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{prec}", color=INK, fontsize=12, pad=8)
    ax.set_xlabel("Number of atoms", color=INK2, fontsize=11)
    ax.grid(True, which="major", color="#e2e2dd", lw=0.8, zorder=0)
    ax.grid(True, which="minor", color="#f0f0ec", lw=0.5, zorder=0)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(colors=INK2, labelsize=10)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c9c9c3")

axes[0].set_ylabel("Throughput [timesteps/s]", color=INK2, fontsize=11)
# The 216-atom f64 dip is the finding, not an outlier - point at it.
axes[1].annotate("capacity effect\n(see results.md)", xy=(216, 141), xytext=(300, 40),
                 color=INK2, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
h, l = axes[0].get_legend_handles_labels()
order = [1, 2, 3, 0]
fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center",
           ncol=4, frameon=False, fontsize=9.5, labelcolor=INK2,
           bbox_to_anchor=(0.5, -0.03))
fig.suptitle("ACE throughput on an RTX A4500 — single-point force evaluations",
             color=INK, fontsize=13, y=0.99)
fig.text(0.5, 0.885, "excludes neighbour-list rebuild cost (static configuration); "
         "sw/kk is different physics, shown for scale only",
         ha="center", color=INK2, fontsize=9)
fig.tight_layout(rect=[0, 0.06, 1, 0.93])
fig.savefig("scaling.png", dpi=170, facecolor=SURF, bbox_inches="tight")
print("wrote acejax/bench/scaling.png")
