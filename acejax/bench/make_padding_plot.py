"""Atom-axis padding: cost per step against the node count evaluated.

The companion to scaling.png.  There the plugin gap was visible but unexplained;
this is the experiment that explains it, and it explains only one end of the
plot.  x is the number of atom rows the model evaluates -- n_nodes in pure JAX,
the bundle's max_atoms under LAMMPS -- with the edge list held fixed in both.

Three vertical references matter:
  1728  local atoms: what a local-only evaluation would cost
  5373  nall = nlocal + nghost: the plugin's hard floor (5040 aborts)
  6313  the capacity the plotted series actually shipped

Regenerate with `uv run --with matplotlib python make_padding_plot.py`.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

# ms/step at 1728 atoms, f64, sparse A2B, 76920 real edges.
# Python: bench_acejax.py --pad-nodes (edges exact).  LAMMPS: run_capacity_sweep.sh
# (max_edges pinned at 110880 for every rung).  Missing keys were not measured.
PY = {
    2849: {1728: 46.688, 3456: 68.735, 5373: 102.029, 5544: 122.713, 6160: 150.827,
           6313: 140.559, 6930: 175.116, 7920: 252.744, 9240: 350.239,
           11088: 511.955, 13860: 781.427, 18480: 1257.249, 22176: 1625.142},
    69:   {1728: 3.538, 3456: 3.448, 5373: 3.647, 5544: 3.656, 6160: 3.670,
           6313: 3.728, 6930: 3.778, 7920: 3.862, 9240: 3.801, 11088: 3.888,
           13860: 4.063, 18480: 4.762, 22176: 5.253},
}
LMP = {
    2849: {5544: 115.994, 6160: 135.512, 6930: 158.267, 7920: 206.788,
           9240: 272.524, 11088: 379.176, 12320: 454.344, 13860: 548.738,
           15840: 678.312, 18480: 862.452},          # 22176 OOM'd; 5040 below floor
    69:   {5544: 6.293, 6160: 6.379, 6930: 6.232, 7920: 6.332, 9240: 6.427,
           11088: 6.476, 12320: 6.549, 13860: 6.758, 15840: 6.908, 18480: 7.192,
           22176: 7.528},
}
NLOCAL, NALL, SHIPPED = 1728, 5373, 6313

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"
INK, INK2, SURF = "#1a1a19", "#52514e", "#fcfcfb"

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
fig.patch.set_facecolor(SURF)

for ax, nB in zip(axes, (2849, 69)):
    ax.set_facecolor(SURF)
    for x, lab, ha in ((NLOCAL, "1728\nlocal atoms", "left"),
                       (NALL, "5373\nnall (floor)", "left"),
                       (SHIPPED, "6313\nshipped", "left")):
        ax.axvline(x, color=GREY, lw=0.8, ls=":", zorder=1)
    p = PY[nB]; l = LMP[nB]
    xs = sorted(p); ax.plot(xs, [p[k] for k in xs], "--", color=BLUE, lw=1.8,
                            marker="o", ms=4.5, mfc=SURF, mew=1.3, zorder=3)
    xs = sorted(l); ax.plot(xs, [l[k] for k in xs], "-", color=ORANGE, lw=2.0,
                            marker="s", ms=4.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Atom rows evaluated  (n_nodes / bundle max_atoms)",
                  color=INK2, fontsize=10.5)
    ax.grid(True, which="major", ls="-", lw=0.5, color="#e6e5e2")
    ax.grid(True, which="minor", ls="-", lw=0.3, color="#f0efec")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([2000, 5000, 10000, 20000])
    ax.tick_params(colors=INK2, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color("#dedcd8")
    factor = p[SHIPPED] / p[NLOCAL]
    ax.set_title(f"n_B = {nB}\npadding 1728 -> 6313 costs {factor:.2f}x",
                 color=INK, fontsize=11, pad=8)

axes[0].set_ylabel("ms per force evaluation, 1728 atoms", color=INK2, fontsize=10.5)
axes[0].annotate("LAMMPS OOMs at 22176 rows:\na 4x capacity margin is not\nusable at this basis size",
                 xy=(6600, 1450), fontsize=8.5, color=ORANGE, ha="left")
axes[0].text(1850, 700, "cost grows FASTER than the row count:\n"
             "12.8x the rows, 34.8x the time", fontsize=8.5, color=BLUE)
axes[1].text(1850, 4.55, "flat: a 4.0x capacity range costs 1.20x.\n"
             "The near-constant ~2.6 ms gap between the\n"
             "two lines is the plugin's fixed per-step cost,\n"
             "and it is 95% of the loss at this basis size.",
             fontsize=8.5, color=INK2)

handles = [
    Line2D([], [], color=BLUE, lw=1.8, ls="--", marker="o", ms=4.5, mfc=SURF,
           mew=1.3, label="acejax in Python, n_nodes swept (edges exact)"),
    Line2D([], [], color=ORANGE, lw=2, marker="s", ms=4.5,
           label="pair jax/kk, bundle max_atoms swept (max_edges pinned at 110880)"),
    Line2D([], [], color=GREY, lw=0.8, ls=":", label="1728 local / 5373 nall / 6313 shipped"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
           ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
fig.suptitle("Does the plugin gap come from padding the atom axis?  "
             "1728 atoms, f64, sparse A2B, RTX A4500",
             color=INK, fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(pathlib.Path(__file__).resolve().parent / "padding.png",
            dpi=170, bbox_inches="tight", facecolor=SURF)
print("wrote padding.png")
