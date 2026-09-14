"""Two-host throughput figure.

    python make_plot.py xeon.json m3pro.json molly_vs_jax.png

`collect.py` writes `collected.json` in its working directory; run it once per
host and pass the two files here.  Left panel Xeon Silver 4216 (AVX-512), right
panel Apple M3 Pro (NEON) -- the point of the figure is that the two panels
disagree about which engine is faster.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = sys.argv[1:3]
out = sys.argv[3] if len(sys.argv) > 3 else "molly_vs_jax.png"
titles = ["Xeon Silver 4216  (AVX-512, 16 cores)", "Apple M3 Pro  (NEON, 6P+6E cores)"]

# (engine, config-key-candidates) -> style.  Config names differ per host
# because the thread counts and the JAX single-thread mechanism differ.
SERIES = [
    (["jax|1core", "jax|1thread"],      "tab:blue",  "-",  "o", "JAX MD, 1 thread"),
    (["jax|16core", "jax|default"],     "tab:blue",  "--", "s", "JAX MD, all cores available"),
    (["julia|tuned/t1"],                "tab:red",   "-",  "o", "ACEpotentials+Molly, 1 thread"),
    (["julia|tuned/t16", "julia|tuned/t12"], "tab:red", "--", "s", "ACEpotentials+Molly, all cores"),
    (["julia|naive/t1"],                "tab:orange", ":", "^", "…default (exact list every step), 1 thread"),
]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
for ax, f, title in zip(axes, files, titles):
    data = json.load(open(f))
    for keys, c, ls, m, label in SERIES:
        pts = []
        for k, v in data.items():
            eng, cfg, atoms = k.split("|")
            if f"{eng}|{cfg}" in keys:
                pts.append((int(atoms), v["atom_steps_per_s"]))
        if not pts:
            continue
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls,
                marker=m, label=label, lw=1.8, ms=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("atoms")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([216, 512, 1000, 1728])
    ax.set_xticklabels(["216", "512", "1000", "1728"])
    ax.set_xticks([], minor=True)          # log minor ticks collide with the labels
    ax.grid(True, which="both", alpha=0.25)
axes[0].set_ylabel("atom-steps / s")
axes[0].legend(fontsize=8, loc="center left", framealpha=0.95)
fig.suptitle("NVE MD, one fitted Si ACE model (120 functions, rcut 6.0 A), f64, dt = 0.25 fs\n"
             "min of repeats over two passes", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out, dpi=150)
print("wrote", out)
