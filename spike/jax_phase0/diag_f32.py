"""Phase 0 diagnostic: why do naive and hybrid Jacobians disagree by ~5e-1 in f32?

Evidence gathering only - no fixes. Separates three candidate mechanisms:
  (a) agnesi clip at y=+-1 flipping between f32/f64 -> derivative steps to zero
  (b) segment_sum scatter atomics on GPU -> run-to-run nondeterminism in f32
  (c) catastrophic cancellation in AA @ A2B.T
"""
import json, os, sys
import numpy as np, jax, jax.numpy as jnp

X64 = os.environ.get("X64", "1") == "1"
import etace_jax as E          # NB: sets jax_enable_x64=True at import time
jax.config.update("jax_enable_x64", X64)   # ... so override it AFTER the import

print(f"x64={jax.config.jax_enable_x64}  backend={jax.default_backend()}")
M = E.load("si_model.json")
BD = json.load(open("bench_data.json"))
key = sorted(BD, key=lambda k: BD[k]["n_atoms"])[0]      # smallest = 64 atoms
d = BD[key]; n_nodes = d["n_atoms"]
rij = jnp.asarray(d["edge_rij"]); send = jnp.asarray(d["edge_i"], dtype=jnp.int32)
n_B = M["meta"]["n_B"]
print(f"n_atoms={n_nodes} n_edges={rij.shape[0]} n_B={n_B} rij.dtype={rij.dtype}")

# ---------------------------------------------------------------- (a) clip census
r = jnp.linalg.norm(rij, axis=-1)
p = M["agnesi"]
s = (r - p["rin"]) / (p["req"] - p["rin"])
x = 1.0 / (1.0 + p["a"] * s ** p["pin"] / (1.0 + s ** (p["pin"] - p["pcut"])))
raw = p["b1"] * x + p["b0"]                              # pre-clip
n_hi = int(jnp.sum(raw > 1.0)); n_lo = int(jnp.sum(raw < -1.0))
marg = jnp.minimum(jnp.abs(raw - 1.0), jnp.abs(raw + 1.0))
print(f"\n(a) CLIP CENSUS: pre-clip raw in [{float(raw.min()):.6f}, {float(raw.max()):.6f}]")
print(f"    clipped high={n_hi}  low={n_lo}  of {raw.shape[0]} edges")
print(f"    edges within 1e-6 of a clip boundary: {int(jnp.sum(marg < 1e-6))}")
print(f"    edges within 1e-3 of a clip boundary: {int(jnp.sum(marg < 1e-3))}")
print(f"    min margin to boundary: {float(marg.min()):.3e}")

# ---------------------------------------------------------------- jacobians
def make_naive():
    def f(r_): return E.descriptor(r_, send, n_nodes, M)
    def jac(r_):
        _, vjp = jax.vjp(f, r_)
        return jax.vmap(lambda row: vjp(jnp.broadcast_to(row, (n_nodes, n_B)))[0])(jnp.eye(n_B))
    return jax.jit(jac)

def make_hybrid():
    def edge_feats(r1):
        return E.radial(r1[None, :], M)[0], E.angular(r1[None, :], M)[0]
    def node_B(A_i):
        AA = jnp.concatenate([jnp.prod(A_i[g], axis=-1) for g in M["aa_groups"]])
        return M["A2B"] @ AA
    def jac(r_):
        Rnl, Ylm = jax.vmap(edge_feats)(r_)
        dRnl, dYlm = jax.vmap(jax.jacfwd(edge_feats))(r_)
        A = E.pool_A(Rnl, Ylm, send, n_nodes, M)
        ar, ay = M["aspec_r"], M["aspec_y"]
        dA = dRnl[:, ar, :] * Ylm[:, ay, None] + Rnl[:, ar, None] * dYlm[:, ay, :]
        J_BA = jax.vmap(jax.jacrev(node_B))(A)
        return jnp.einsum('ebm,emc->ebc', J_BA[send], dA)
    return jax.jit(jac)

nj, hj = make_naive(), make_hybrid()

# ---------------------------------------------------------------- (b) determinism
print("\n(b) DETERMINISM: 5 repeat evaluations, max|delta| vs first run")
a0 = np.asarray(nj(rij)).transpose(1, 0, 2); b0 = np.asarray(hj(rij))
for t in range(1, 5):
    a = np.asarray(nj(rij)).transpose(1, 0, 2); b = np.asarray(hj(rij))
    print(f"    run {t}: naive drift={np.abs(a-a0).max():.3e}   hybrid drift={np.abs(b-b0).max():.3e}")

# ---------------------------------------------------------------- (c) localise
den = max(np.abs(b0).max(), 1e-300)
diff = np.abs(a0 - b0)
print(f"\n(c) DISCREPANCY: max|naive-hybrid|/max|hybrid| = {diff.max()/den:.3e}")
print(f"    max|hybrid| = {np.abs(b0).max():.6e}   max|naive| = {np.abs(a0).max():.6e}")
print(f"    non-finite: naive={int((~np.isfinite(a0)).sum())} hybrid={int((~np.isfinite(b0)).sum())}")
e, k, c = np.unravel_index(diff.argmax(), diff.shape)
print(f"    worst element: edge={e} basis={k} comp={c}")
print(f"      naive={a0[e,k,c]:.8e}  hybrid={b0[e,k,c]:.8e}")
print(f"      |r_e|={float(jnp.linalg.norm(rij[e])):.6f}  raw_y={float(raw[e]):.8f}  "
      f"clipped={'YES' if abs(float(raw[e]))>1.0 else 'no'}  margin={float(marg[e]):.3e}")
for thr in (1e-2, 1e-4, 1e-6):
    print(f"    elements with rel err > {thr:g}: {int((diff/den > thr).sum())} of {diff.size}")
edges_bad = np.unique(np.argwhere(diff/den > 1e-4)[:, 0]) if (diff/den > 1e-4).any() else []
print(f"    distinct edges carrying rel err > 1e-4: {len(edges_bad)}")
if len(edges_bad):
    mg = np.asarray(marg)[edges_bad]
    print(f"      their clip margins: min={mg.min():.3e} median={np.median(mg):.3e}")
    print(f"      how many are clipped: {int(sum(abs(float(raw[i]))>1.0 for i in edges_bad))}")
