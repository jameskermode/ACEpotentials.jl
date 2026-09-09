import numpy as np
f64 = np.load("jac_f64.npz"); f32 = np.load("jac_f32.npz")
ref = f64["hybrid"]; den = np.abs(ref).max()
print(f"reference = f64 hybrid, max|ref| = {den:.6e}")
print(f"f64 naive vs f64 hybrid : {np.abs(f64['naive']-ref).max()/den:.3e}\n")
for name in ("naive", "hybrid"):
    e = np.abs(f32[name] - ref)
    print(f"f32 {name:<7} vs f64 ref: max rel = {e.max()/den:.3e}   "
          f"mean rel = {e.mean()/den:.3e}   >1e-2: {int((e/den>1e-2).sum())}")
print()
# localise on the element the diagnostic flagged
e_, k_, c_ = 458, 50, 2
print(f"element (edge={e_}, basis={k_}, comp={c_}):")
print(f"  f64 hybrid = {ref[e_,k_,c_]:+.8e}   f64 naive = {f64['naive'][e_,k_,c_]:+.8e}")
print(f"  f32 hybrid = {f32['hybrid'][e_,k_,c_]:+.8e}   f32 naive = {f32['naive'][e_,k_,c_]:+.8e}")
print()
# is the f32 damage in A, J_BA or dA?
for k in ("A", "J_BA", "dA"):
    r64, r32 = f64[k], f32[k]
    d = np.abs(r32 - r64).max() / max(np.abs(r64).max(), 1e-300)
    print(f"  intermediate {k:<5} f32 vs f64: max rel = {d:.3e}   "
          f"max|f64| = {np.abs(r64).max():.4e}")
# cancellation diagnostic: how big are the terms that build the flagged element?
J = f64["J_BA"]; dAv = f64["dA"]
send = None
terms = J[:, k_, :]  # (n_nodes, n_A)  -- need the node for edge 458
print("\ncancellation check on the flagged element (f64):")
import json; BD = json.load(open("bench_data.json"))
d0 = BD[sorted(BD, key=lambda k: BD[k]['n_atoms'])[0]]
node = int(np.asarray(d0["edge_i"])[e_])
t = J[node, k_, :] * dAv[e_, :, c_]
print(f"  node={node}  sum of terms = {t.sum():+.8e}")
print(f"  max|term| = {np.abs(t).max():.6e}   sum|term| = {np.abs(t).sum():.6e}")
print(f"  cancellation ratio sum|t| / |sum t| = {np.abs(t).sum()/abs(t.sum()):.3e}")
