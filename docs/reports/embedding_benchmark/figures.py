import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
C_CAT, C_EMB, C_LOSS, C_D8 = "#1f4e79", "#c0392b", "#e67e22", "#7f8c8d"

# ---------- Figure 1: test force RMSE vs parameters, 1k and 4k ----------------
# (params, testF)
pts = {
  "1k": {
    "categorical": [(6740, 0.0969), (19120, 0.0855)],
    "lossless":    [(2375, 0.1087), (5950, 0.0906)],
    "d16":         [(1615, 0.1101), (3765, 0.0911)],
    "d8":          [(910, 0.1162), (2040, 0.0986)],
  },
  "4k": {
    "categorical": [(19120, 0.0763), (46635, 0.0641)],
    "lossless":    [(5950, 0.0885), (13650, 0.0753)],
    "d16":         [(3765, 0.0894), (8045, 0.0767)],
    "d8":          [(2040, 0.1000), (4250, 0.0895)],
  },
}
style = {"categorical": (C_CAT, "s", "categorical"), "lossless": (C_LOSS, "^", "embedded, lossless"),
         "d16": (C_EMB, "o", "embedded, d≤16"), "d8": (C_D8, "v", "embedded, d≤8")}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, (key, degs) in zip(axes, [("1k", (6, 8)), ("4k", (8, 10))]):
    for m, series in pts[key].items():
        c, mk, lab = style[m]
        p = np.array(series)
        ax.plot(p[:, 0], p[:, 1], "-", color=c, marker=mk, ms=5, lw=1.2, label=lab)
        if m in ("categorical", "d16"):
            for (x, y), d in zip(p, degs):
                ax.annotate(f"{d}", (x, y), textcoords="offset points", xytext=(0, -11 if m == "d16" else 5),
                            fontsize=6.5, color=c, ha="center")
    ax.set_xscale("log"); ax.set_xlabel("parameters (n_B × S)")
    from matplotlib.ticker import NullFormatter, FixedLocator, FuncFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(FixedLocator([1e3, 3e3, 1e4, 3e4]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:g}k"))
    ax.set_title(f"{key} structures", fontsize=9)
    ax.grid(alpha=0.25, which="both", lw=0.4)
axes[0].set_ylabel("test force RMSE (eV/Å)")
axes[1].legend(frameon=False, fontsize=7, loc="upper right")
fig.savefig("fig_accuracy_vs_params.png")

# ---------- Figure 2: learning curves (1k set, bounded, MH-1) ---------------------
N = [50, 100, 200, 400, 800]
lc = {
  "deg 6 categorical": ([0.153, 0.132, 0.113, 0.101, 0.097], [0.084, 0.080, 0.073, 0.082, 0.087], C_CAT, "s"),
  "deg 6 embedded d≤16": ([0.125, 0.118, 0.113, 0.111, 0.111], [0.090, 0.095, 0.100, 0.104, 0.105], C_EMB, "o"),
  "deg 8 categorical": ([0.158, 0.136, 0.114, 0.099, 0.086], [0.056, 0.070, 0.063, 0.063, 0.063], C_CAT, "s"),
  "deg 8 embedded d≤16": ([0.124, 0.109, 0.098, 0.093, 0.092], [0.072, 0.072, 0.078, 0.083, 0.086], C_EMB, "o"),
}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, deg in zip(axes, (6, 8)):
    for name, (te, tr, c, mk) in lc.items():
        if not name.startswith(f"deg {deg}"): continue
        lab = name.split(" ", 2)[2]
        ax.plot(N, te, "-", color=c, marker=mk, ms=4, lw=1.2, label=f"{lab}, test")
        ax.plot(N, tr, "--", color=c, marker=mk, ms=3, lw=0.9, alpha=0.6, label=f"{lab}, train")
    ax.set_xscale("log"); ax.set_xlabel("training structures")
    ax.set_title(f"degree {deg}", fontsize=9); ax.grid(alpha=0.25, which="both", lw=0.4)
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_locator(FixedLocator(N)); ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in N]))
axes[0].set_ylabel("force RMSE (eV/Å)")
axes[1].legend(frameon=False, fontsize=6.5, loc="upper right")
fig.savefig("fig_learning_curves.png")

# ---------- Figure 3: throughput (lestrade, deg-6 embedded d<=16 student vs MH-1) ----
# all on one host (32 cores, RTX 4000 Ada), 256-384-atom cells; f64 and f32 side by side
labels = ["MACE-MH-1\nGPU", "Julia CPU\n1 thread", "Julia CPU\n32 threads", "JAX GPU\nframe (host nlist)", "JAX GPU\nkernel"]
f64 = [8.6e2, 4.7e3, 2.1e4, 4.0e4, 8.3e4]
f32 = [4.9e3, np.nan, np.nan, 5.3e4, 1.8e5]
x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(7.0, 2.4))
b1 = ax.bar(x - w/2, f64, w, color=[C_D8, C_CAT, C_CAT, C_EMB, C_EMB], label="f64")
b2 = ax.bar(x + w/2, f32, w, color=[C_D8, C_CAT, C_CAT, C_EMB, C_EMB], alpha=0.45, hatch="//", label="f32")
ax.set_yscale("log"); ax.set_ylabel("atom-steps / s"); ax.grid(axis="y", alpha=0.25, which="both", lw=0.4)
ax.set_xticks(x); ax.set_xticklabels(labels)
for bars, vals in ((b1, f64), (b2, f32)):
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:.1e}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=6.5)
ax.set_ylim(3e2, 6e5)
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.savefig("fig_throughput.png")

# ---------- Figure 3b: MACE stacks vs the ACE student, kernel throughput, f32, A4500, vs cell size ----
sizes = [40, 320, 1080]      # mean atoms of the three supercell series
series = {
    "ACE deg-6 d≤16 student, JAX":          ([8.3e4, 1.6e5, 8.4e4], C_EMB, "o", "-"),
    "MACE-MH-1, torch + cuEquivariance":    ([7.6e2, 6.3e3, 1.4e4], "#333333", "D", "-"),
    "MACE-MH-1, mace-jax (e3nn-jax)":       ([3.9e3, 6.8e3, 7.6e3], "#333333", "s", "--"),
    "MACE-MH-1, torch plain":               ([1.0e3, 2.2e3, 2.2e3], "#333333", "^", ":"),
    "MACE-MP-0 medium, torch + cuEq":       ([1.0e3, 8.3e3, 2.8e4], "#8c8c8c", "D", "-"),
    "MACE-MP-0 small, mace-jax":            ([8.7e3, 2.0e4, 2.4e4], "#8c8c8c", "s", "--"),
}
fig, ax = plt.subplots(figsize=(7.0, 3.2))
for lab, (v, c, mk, ls) in series.items():
    ax.plot(sizes, v, ls, color=c, marker=mk, ms=4.5, lw=1.2, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
ax.xaxis.set_major_locator(FixedLocator(sizes)); ax.xaxis.set_major_formatter(FixedFormatter(["32–48", "256–384", "864–1296"]))
ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("atoms per cell"); ax.set_ylabel("kernel atom-steps / s (f32)")
ax.grid(alpha=0.25, which="both", lw=0.4)
ax.set_ylim(4e2, 5e5)
ax.legend(frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, 0.92), ncol=1)
fig.savefig("fig_mace_stacks.png")
# ---------- Figure 4: LAMMPS, 1728 Si atoms, A4500: ACE jax/kk vs ML-PACE vs MACE ----------
# Source: acejax/bench/results.md ("Throughput, realistic shape") and results_phase13.md
# ("The three-way table"); same host, same deck, same three Si models.
nB = [69, 710, 2849]
ace_f64 = [3.15e5, 6.45e4, 1.27e4]
ace_f32 = [6.16e5, 1.17e5, 2.24e4]
pace_f64 = [9.54e5, 1.39e5, 3.30e4]          # pace/kk, n_B 78 / 693 / 2874 (matched shape)
mace = [  # label, atom-steps/s, params, linestyle
    ("MACE-MP-0b2 small, symmetrix f32  (8.2M params)", 4.562e4, "-."),
    ("MACE-MP-0 small, jax/kk f32  (3.8M)", 3.755e4, "-"),
    ("MACE-MP-0b2 small, symmetrix f64  (8.2M)", 3.699e4, ":"),
    ("MACE-MP-0b3 medium, jax/kk f32  (9.1M)", 1.621e4, "--"),
]
fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.plot(nB, pace_f64, "-", color="#2e8b57", marker="D", ms=5, lw=1.4, label="ML-PACE  pace/kk  f64 (C++/Kokkos)")
ax.plot(nB, ace_f64, "-", color=C_CAT, marker="s", ms=5, lw=1.4, label="ACE  jax/kk  f64")
ax.plot(nB, ace_f32, "--", color=C_CAT, marker="s", ms=5, lw=1.2, alpha=0.7, label="ACE  jax/kk  f32")
for lab, v, ls in mace:
    ax.axhline(v, color="#666666", lw=0.9, ls=ls, alpha=0.9, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(55, 3600); ax.set_ylim(8e3, 1.5e6)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
ax.xaxis.set_major_locator(FixedLocator(nB)); ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in nB]))
ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("ACE basis functions n_B (single species, order 4)")
ax.set_ylabel("atom-steps / s")
ax.grid(alpha=0.25, which="both", lw=0.4)
ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=1)
fig.savefig("fig_lammps.png")
print("lammps figure written")
