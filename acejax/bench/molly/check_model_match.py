"""Confirm the Julia model the Molly benchmark drives is the model that produced
acejax/fixtures/si_fitted.npz.

Re-runs acejax/julia/export_model.jl (unmodified, default environment: Si,
order 3, totaldegree 10, fit to Si_tiny with BLR) on this host and diffs every
array in the fresh export against the shipped fixture.

    python check_model_match.py fresh.npz ../../fixtures/si_fitted.npz
"""
import sys
import numpy as np

fresh, shipped = sys.argv[1], sys.argv[2]
A, B = np.load(fresh, allow_pickle=True), np.load(shipped, allow_pickle=True)
ka, kb = set(A.files), set(B.files)
if ka != kb:
    print(f"KEY MISMATCH  only-fresh={sorted(ka-kb)}  only-shipped={sorted(kb-ka)}")

worst, worst_key, nbad = 0.0, None, 0
for k in sorted(ka & kb):
    a, b = A[k], B[k]
    if a.dtype.kind in "SUO" or b.dtype.kind in "SUO":
        same = np.array_equal(a, b)
        print(f"{k:24s} {'same' if same else 'DIFFERENT'} (non-numeric)")
        nbad += 0 if same else 1
        continue
    if a.shape != b.shape:
        print(f"{k:24s} SHAPE {a.shape} vs {b.shape}")
        nbad += 1
        continue
    d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    scale = max(float(np.max(np.abs(b.astype(np.float64)))) if b.size else 1.0, 1e-300)
    rel = d / scale
    if rel > worst:
        worst, worst_key = rel, k
    if rel > 1e-10:
        nbad += 1
        print(f"{k:24s} max|d|={d:.3e}  rel={rel:.3e}   <-- differs")

print(f"\n{len(ka & kb)} arrays compared; worst relative difference {worst:.3e} ({worst_key}); "
      f"{nbad} arrays differ by more than 1e-10 relative")
sys.exit(1 if nbad else 0)
