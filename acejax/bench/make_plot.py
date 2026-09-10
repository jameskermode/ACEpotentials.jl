"""Scaling plot: throughput vs atom count, log-log, one panel per basis size.

All three points use a REALISTIC ACE shape -- correlation order 4, lmax 5 --
because that is how production potentials reach thousands of functions.  An
earlier version of this series used lmax 10 at order 3, which reaches the same
counts by a route real potentials do not take and which distorted the per-stage
breakdown badly (see results.md).

Encoding: solid = a LAMMPS pair style; dashed = acejax driven from Python, which
is NOT a pair style and carries no neighbour-list, communication or integration
cost.  Same hue = same implementation, so a solid/dashed gap is what LAMMPS
costs and a colour gap is what the implementation costs.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

N = [64, 216, 512, 1000, 1728]
# atom-steps/s.  acejax uses the sparse A2B contraction throughout: dense is
# 679 ms/step at n_B=2849 and would measure our implementation, not the design.
DATA = {
    (69, 78):    {"jax_f64": [7.335e4, 5.294e4, 2.011e5, 2.725e5, 3.154e5],
                  "jax_f32": [1.087e5, 2.779e5, 4.674e5, 5.752e5, 6.467e5],
                  "pace":    [1.872e5, 4.271e5, 6.545e5, 8.040e5, 9.539e5]},
    (710, 693):  {"jax_f64": [2.621e4, 2.261e4, 6.310e4, 6.325e4, 6.451e4],
                  "jax_f32": [3.837e4, 7.687e4, 1.048e5, 1.064e5, 1.170e5],
                  "pace":    [7.218e4, 1.118e5, 1.280e5, 1.353e5, 1.385e5]},
    (2849, 2874):{"jax_f64": [6991, 9527, 1.597e4, 1.571e4, 1.274e4],
                  "jax_f32": [8829, 1.595e4, 2.205e4, 2.399e4, 2.239e4],
                  "pace":    [2.388e4, 2.817e4, 3.190e4, 3.256e4, 3.302e4]},
}

ts = lambda a: [ai / ni for ni, ai in zip(N, a)]

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#2e9e6b", "#8a8a85"
INK, INK2, SURF = "#1a1a19", "#52514e", "#fcfcfb"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
fig.patch.set_facecolor(SURF)

for ax, (key, d) in zip(axes, DATA.items()):
    ours, theirs = key
    ax.set_facecolor(SURF)
    ax.plot(N, ts(d["pace"]), "-", color=GREEN, lw=2.0, marker="D", ms=5)
    ax.plot(N, ts(d["jax_f32"]), "-", color=ORANGE, lw=2.0, marker="o", ms=5)
    ax.plot(N, ts(d["jax_f64"]), "-", color=BLUE, lw=2.0, marker="o", ms=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of atoms", color=INK2, fontsize=11)
    ax.grid(True, which="major", ls="-", lw=0.5, color="#e6e5e2")
    ax.grid(True, which="minor", ls="-", lw=0.3, color="#f0efec")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(colors=INK2, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color("#dedcd8")
    r = d["pace"][-1] / d["jax_f64"][-1]
    ax.set_title(f"acejax {ours}  vs  pace {theirs} functions\n"
                 f"pace / acejax f64 = {r:.1f}x at 1728 atoms",
                 color=INK, fontsize=10.5, pad=8)

axes[0].set_ylabel("Throughput [timesteps/s]", color=INK2, fontsize=11)
handles = [
    Line2D([], [], color=GREEN, lw=2, marker="D", ms=5, label="pair pace/kk (f64)"),
    Line2D([], [], color=ORANGE, lw=2, marker="o", ms=5, label="pair jax/kk, f32"),
    Line2D([], [], color=BLUE, lw=2, marker="o", ms=5, label="pair jax/kk, f64"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
           ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2)
fig.suptitle("ACE throughput on an RTX A4500, realistic shape (order 4, lmax 5) - "
             "single-point evaluation, no reneighbouring",
             color=INK, fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(pathlib.Path(__file__).resolve().parent / "scaling.png",
            dpi=170, bbox_inches="tight", facecolor=SURF)
print("wrote scaling.png")
