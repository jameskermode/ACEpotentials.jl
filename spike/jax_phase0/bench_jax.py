"""Phase 0 spike: JAX CPU timings + gate 3 (design-matrix Jacobian strategies).

Runs unmodified on a CUDA box: JAX picks the GPU automatically and the script
reports whichever backend it is on. Gate 2 (GPU f32/f64) is UNMEASURED on the
macOS/arm64 dev machine -- there is no CUDA device here.
"""
import json, time, sys
import numpy as np, jax, jax.numpy as jnp
import etace_jax as E

jax.config.update("jax_enable_x64", True)
print(f"jax {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")

M = E.load("si_model.json")
BD = json.load(open("bench_data.json"))
n_B = M["meta"]["n_B"]; n_A = M["meta"]["n_A"]


def timeit(fn, *args, n=20, warmup=3):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3          # ms, min like BenchmarkTools


# ---------------------------------------------------------- gate 3 strategies
def make_naive_jac(n_nodes):
    """vmap a VJP over eye(n_B). Cost ~ n_B x forward."""
    def f(rij, send):
        return E.descriptor(rij, send, n_nodes, M)
    def jac(rij, send):
        _, vjp = jax.vjp(lambda r: f(r, send), rij)
        eye = jnp.eye(n_B)
        # cotangent for basis k: broadcast one-hot over all nodes
        def one(k_row):
            ct = jnp.broadcast_to(k_row, (n_nodes, n_B))
            return vjp(ct)[0]                       # (n_edges, 3)
        return jax.vmap(one)(eye)                   # (n_B, n_edges, 3)
    return jax.jit(jac)


def make_hybrid_jac(n_nodes):
    """Edge-feature-level push, the way ET._jacobian_X works:
       dB/dr = (dB/dA) . (dA/dr), with dA/dr analytic per edge."""
    def edge_feats(r1):                             # r1: (3,)
        return E.radial(r1[None, :], M)[0], E.angular(r1[None, :], M)[0]

    def node_B(A_i):                                # (n_A,) -> (n_B,)
        AA = jnp.concatenate([jnp.prod(A_i[g], axis=-1) for g in M["aa_groups"]])
        return M["A2B"] @ AA

    def jac(rij, send):
        Rnl, Ylm = jax.vmap(edge_feats)(rij)
        dRnl, dYlm = jax.vmap(jax.jacfwd(edge_feats))(rij)     # (E,37,3),(E,25,3)
        A = E.pool_A(Rnl, Ylm, send, n_nodes, M)
        # dA/dr per edge: (E, n_A, 3)
        ar, ay = M["aspec_r"], M["aspec_y"]
        dA = (dRnl[:, ar, :] * Ylm[:, ay, None]
              + Rnl[:, ar, None] * dYlm[:, ay, :])
        J_BA = jax.vmap(jax.jacrev(node_B))(A)                 # (n_nodes, n_B, n_A)
        return jnp.einsum('ebm,emc->ebc', J_BA[send], dA)      # (E, n_B, 3)
    return jax.jit(jac)


# ---------------------------------------------------------- run
print(f"\n{'atoms':<8}{'edges':<8}{'JAX fwd/ms':>12}{'Jl fwd/ms':>12}{'ratio':>8}")
results = {}
for key in sorted(BD, key=lambda k: BD[k]["n_atoms"]):
    d = BD[key]; n = d["n_atoms"]
    rij = jnp.asarray(d["edge_rij"]); send = jnp.asarray(d["edge_i"], dtype=jnp.int32)
    fwd = jax.jit(lambda r, s, n=n: E.descriptor(r, s, n, M))
    t = timeit(fwd, rij, send)
    jl = d["julia_site_basis_ms"]
    print(f"{n:<8}{d['n_edges']:<8}{t:>12.3f}{jl:>12.3f}{t/jl:>8.2f}x")
    results[n] = dict(jax_fwd=t, jl_fwd=jl, rij=rij, send=send, edges=d["n_edges"],
                      jl_jac=d["julia_jacobian_ms"])

print(f"\n=== gate 3: design-matrix Jacobian (n_B={n_B}) ===")
print(f"{'atoms':<8}{'naive VJP/ms':>14}{'hybrid/ms':>12}{'Julia/ms':>10}"
      f"{'naive/fwd':>11}{'hybrid/fwd':>12}")
for n, r in results.items():
    nj = make_naive_jac(n); hj = make_hybrid_jac(n)
    a = np.asarray(nj(r["rij"], r["send"])); b = np.asarray(hj(r["rij"], r["send"]))
    err = np.abs(a.transpose(1, 0, 2) - b).max() / max(np.abs(b).max(), 1e-300)
    t_n = timeit(nj, r["rij"], r["send"], n=5)
    t_h = timeit(hj, r["rij"], r["send"], n=5)
    print(f"{n:<8}{t_n:>14.2f}{t_h:>12.2f}{r['jl_jac']:>10.2f}"
          f"{t_n/r['jax_fwd']:>11.1f}x{t_h/r['jax_fwd']:>11.1f}x")
    print(f"         (naive vs hybrid agreement: max|rel| = {err:.2e})")
