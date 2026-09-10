"""Scaling plot in the shape of the upstream lammps-jax benchmark chart.

Throughput in timesteps/s against atom count, log-log; one panel per basis size,
so the size dependence of the pace/acejax ratio reads directly.

Encoding: each implementation has a hue.  Solid = a LAMMPS pair style; dashed =
acejax driven from Python, which is NOT a pair style -- it carries no
neighbour-list, communication or integration cost.  The gap between a solid and
dashed line of the same hue is therefore what going through LAMMPS costs; the
gap between hues is the cost of a different implementation.

Numbers from results.md; see there for what the measurement excludes and for the
dense-vs-sparse A2B caveat at 1429.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

N = [64, 216, 512, 1000, 1728]
N6 = [64, 216, 512, 1000, 1728, 4096]

# LAMMPS pair styles, atom-steps/s.  110 and 211 use the dense A2B contraction
# (faster at small basis); 1429 uses the sparse one -- dense would read 8x worse
# and measure our implementation rather than the architecture.
LMP = {
    110:  {"jax_f64": ([64,216,512,1000,1728,4096], [5.34e4,3.03e4,1.50e5,1.85e5,2.14e5,1.98e5]),
           "jax_f32": ([64,216,512,1000,1728,4096], [1.02e5,2.00e5,3.05e5,3.29e5,3.89e5,3.51e5]),
           "pace":    ([64,216,512,1000,1728,4096], [1.77e5,4.61e5,7.18e5,8.85e5,1.08e6,1.14e6])},
    211:  {"jax_f64": ([64,216,512,1000,1728,4096], [2.55e4,1.95e4,6.76e4,7.57e4,8.45e4,9.60e4]),
           "jax_f32": ([64,216,512,1000,1728,4096], [7.21e4,1.09e5,1.82e5,1.83e5,2.06e5,1.62e5]),
           "pace":    ([64,216,512,1000,1728,4096], [1.36e5,2.55e5,3.42e5,3.77e5,3.94e5,4.01e5])},
    1429: {"jax_f64": (N, [1.572e4,2.142e4,2.790e4,2.437e4,2.155e4]),
           "jax_f32": (N, [1.914e4,3.501e4,4.159e4,3.853e4,3.528e4]),
           "pace":    (N, [3.363e4,5.033e4,5.591e4,5.697e4,5.773e4])},
}
# acejax driven from Python on the same GPU, ms/step -> atom-steps/s
PY_MS = {
    110:  {"f64": (N6, [0.433,0.810,2.019,3.832,6.839,16.502])},
    211:  {"f64": (N,  [0.753,1.588,3.225,6.886,12.434])},
    1429: {"f64": (N,  [0.843,2.587,8.049,17.884,34.067])},   # sparse A2B
}
PACE_N = {110: 99, 211: 211, 1429: 1551}

ts = lambda n, a: [ai / ni for ni, ai in zip(n, a)]
ms_ts = lambda ms: [1000.0 / m for m in ms]

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#2e9e6b", "#8a8a85"
INK, INK2, SURF = "#1a1a19", "#52514e", "#fcfcfb"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
fig.patch.set_facecolor(SURF)

for ax, nb in zip(axes, [110, 211, 1429]):
    d = LMP[nb]
    ax.set_facecolor(SURF)
    n, v = d["pace"];    ax.plot(n, ts(n, v), "-", color=GREEN, lw=2.0, marker="D", ms=5)
    n, v = d["jax_f32"]; ax.plot(n, ts(n, v), "-", color=ORANGE, lw=2.0, marker="o", ms=5)
    n, v = d["jax_f64"]; ax.plot(n, ts(n, v), "-", color=BLUE, lw=2.0, marker="o", ms=5)
    n, ms = PY_MS[nb]["f64"]
    ax.plot(n, ms_ts(ms), "--", color=BLUE, lw=1.7, marker="^", ms=4.5, alpha=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of atoms", color=INK2, fontsize=11)
    ax.grid(True, which="major", ls="-", lw=0.5, color="#e6e5e2")
    ax.grid(True, which="minor", ls="-", lw=0.3, color="#f0efec")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(colors=INK2, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color("#dedcd8")
    note = "  (sparse A2B)" if nb == 1429 else ""
    ax.set_title(f"acejax {nb}  vs  pace {PACE_N[nb]} basis functions{note}",
                 color=INK, fontsize=11, pad=8)

axes[0].set_ylabel("Throughput [timesteps/s]", color=INK2, fontsize=11)
axes[0].annotate("capacity shape effect", xy=(216, LMP[110]["jax_f64"][1][1] / 216),
                 xytext=(16, -28), textcoords="offset points", fontsize=8.5,
                 color=INK2, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))

handles = [
    Line2D([], [], color=GREEN, lw=2, marker="D", ms=5, label="pair pace/kk (f64)"),
    Line2D([], [], color=ORANGE, lw=2, marker="o", ms=5, label="pair jax/kk, f32"),
    Line2D([], [], color=BLUE, lw=2, marker="o", ms=5, label="pair jax/kk, f64"),
    Line2D([], [], color=BLUE, lw=1.7, ls="--", marker="^", ms=4.5,
           label="acejax from Python, f64 (no LAMMPS: model only)"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.07),
           ncol=4, frameon=False, fontsize=9.5, labelcolor=INK2)
fig.suptitle("ACE throughput on an RTX A4500 - single-point evaluation "
             "(timestep 0.0, no reneighbouring)", color=INK, fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(pathlib.Path(__file__).resolve().parent / "scaling.png",
            dpi=170, bbox_inches="tight", facecolor=SURF)
print("wrote scaling.png")
