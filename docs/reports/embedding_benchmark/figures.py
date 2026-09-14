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
labels = ["MACE-MH-1\nGPU f32", "Julia CPU\n1 thread", "Julia CPU\n32 threads", "JAX GPU f32\nframe (host nlist)", "JAX GPU f32\nkernel"]
vals = [4.9e3, 4.7e3, 2.1e4, 5.3e4, 1.8e5]
cols = ["#555555", C_CAT, C_CAT, C_EMB, C_EMB]
fig, ax = plt.subplots(figsize=(7.0, 2.2))
bars = ax.bar(labels, vals, color=cols, width=0.6)
ax.set_yscale("log"); ax.set_ylabel("atom-steps / s"); ax.grid(axis="y", alpha=0.25, which="both", lw=0.4)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.1e}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=7)
ax.set_ylim(2e3, 5e5)
fig.savefig("fig_throughput.png")
print("figures written")
