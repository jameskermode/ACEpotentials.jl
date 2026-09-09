"""Run diag-style (factory closures) and isolate-style (inline) in ONE process."""
import json, os
import numpy as np, jax, jax.numpy as jnp
X64 = os.environ.get("X64","1")=="1"
import etace_jax as E
jax.config.update("jax_enable_x64", X64)
M = E.load("si_model.json"); BD = json.load(open("bench_data.json"))
d = BD[sorted(BD, key=lambda k: BD[k]["n_atoms"])[0]]
n_nodes = d["n_atoms"]; n_B = M["meta"]["n_B"]
rij = jnp.asarray(d["edge_rij"]); send = jnp.asarray(d["edge_i"], dtype=jnp.int32)

def edge_feats(r1): return E.radial(r1[None,:],M)[0], E.angular(r1[None,:],M)[0]
def node_B(A_i):
    AA = jnp.concatenate([jnp.prod(A_i[g],axis=-1) for g in M["aa_groups"]])
    return M["A2B"] @ AA
def _hybrid(r_):
    Rnl, Ylm = jax.vmap(edge_feats)(r_)
    dRnl, dYlm = jax.vmap(jax.jacfwd(edge_feats))(r_)
    A = E.pool_A(Rnl, Ylm, send, n_nodes, M)
    ar, ay = M["aspec_r"], M["aspec_y"]
    dA = dRnl[:,ar,:]*Ylm[:,ay,None] + Rnl[:,ar,None]*dYlm[:,ay,:]
    J_BA = jax.vmap(jax.jacrev(node_B))(A)
    return jnp.einsum('ebm,emc->ebc', J_BA[send], dA)
def _naive(r_):
    _, vjp = jax.vjp(lambda q: E.descriptor(q, send, n_nodes, M), r_)
    return jax.vmap(lambda row: vjp(jnp.broadcast_to(row,(n_nodes,n_B)))[0])(jnp.eye(n_B))

# --- diag-style: DIAG PASSES ONLY rij; BENCH PASSES (rij, send) ---
def make_hybrid_bench(nn):
    def jac(rij_, send_):
        Rnl, Ylm = jax.vmap(edge_feats)(rij_)
        dRnl, dYlm = jax.vmap(jax.jacfwd(edge_feats))(rij_)
        A = E.pool_A(Rnl, Ylm, send_, nn, M)
        ar, ay = M["aspec_r"], M["aspec_y"]
        dA = dRnl[:,ar,:]*Ylm[:,ay,None] + Rnl[:,ar,None]*dYlm[:,ay,:]
        J_BA = jax.vmap(jax.jacrev(node_B))(A)
        return jnp.einsum('ebm,emc->ebc', J_BA[send_], dA)
    return jax.jit(jac)

h_inline = np.asarray(jax.jit(_hybrid)(rij))
h_bench  = np.asarray(make_hybrid_bench(n_nodes)(rij, send))
a        = np.asarray(jax.jit(_naive)(rij)).transpose(1,0,2)
den = np.abs(h_inline).max()
print(f"x64={jax.config.jax_enable_x64}")
print(f"hybrid inline  vs naive : {np.abs(a-h_inline).max()/den:.3e}")
print(f"hybrid bench   vs naive : {np.abs(a-h_bench ).max()/den:.3e}")
print(f"hybrid inline  vs bench : {np.abs(h_inline-h_bench).max()/den:.3e}")
print(f"  (458,50,2) inline={h_inline[458,50,2]:+.6e} bench={h_bench[458,50,2]:+.6e} naive={a[458,50,2]:+.6e}")
print(f"\nsend dtype={send.dtype}  rij dtype={rij.dtype}  eye dtype={jnp.eye(n_B).dtype}")
print(f"send sorted: {bool(jnp.all(send[1:]>=send[:-1]))}  min={int(send.min())} max={int(send.max())} n_nodes={n_nodes}")
