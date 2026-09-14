"""Run the diag-style and dump-style hybrid Jacobians in ONE process and compare."""
import json, os
import numpy as np, jax, jax.numpy as jnp
X64 = os.environ.get("X64","1")=="1"
import etace_jax as E
jax.config.update("jax_enable_x64", X64)
M = E.load("si_model.json"); BD = json.load(open("bench_data.json"))
d = BD[sorted(BD, key=lambda k: BD[k]["n_atoms"])[0]]
n_nodes = d["n_atoms"]; n_B = M["meta"]["n_B"]
rij = jnp.asarray(d["edge_rij"]); send = jnp.asarray(d["edge_i"], dtype=jnp.int32)
print(f"x64={jax.config.jax_enable_x64} n_nodes={n_nodes} n_B={n_B} n_edges={rij.shape[0]}")

def edge_feats(r1): return E.radial(r1[None,:],M)[0], E.angular(r1[None,:],M)[0]
def node_B(A_i):
    AA = jnp.concatenate([jnp.prod(A_i[g],axis=-1) for g in M["aa_groups"]])
    return M["A2B"] @ AA

def hybrid(r_):
    Rnl, Ylm = jax.vmap(edge_feats)(r_)
    dRnl, dYlm = jax.vmap(jax.jacfwd(edge_feats))(r_)
    A = E.pool_A(Rnl, Ylm, send, n_nodes, M)
    ar, ay = M["aspec_r"], M["aspec_y"]
    dA = dRnl[:,ar,:]*Ylm[:,ay,None] + Rnl[:,ar,None]*dYlm[:,ay,:]
    J_BA = jax.vmap(jax.jacrev(node_B))(A)
    return jnp.einsum('ebm,emc->ebc', J_BA[send], dA)

h_jit  = jax.jit(hybrid)
b_jit  = np.asarray(h_jit(rij))
b_eager= np.asarray(hybrid(rij))
print(f"\nhybrid jit vs eager : max|delta| = {np.abs(b_jit-b_eager).max():.3e}")
print(f"  (458,50,2): jit={b_jit[458,50,2]:+.8e}  eager={b_eager[458,50,2]:+.8e}")

# repeated jit calls
for t in range(3):
    bt = np.asarray(h_jit(rij))
    print(f"  repeat {t}: max|delta vs first jit| = {np.abs(bt-b_jit).max():.3e}"
          f"   (458,50,2)={bt[458,50,2]:+.8e}")

# and the naive, for the same element
def naive(r_):
    _, vjp = jax.vjp(lambda q: E.descriptor(q, send, n_nodes, M), r_)
    return jax.vmap(lambda row: vjp(jnp.broadcast_to(row,(n_nodes,n_B)))[0])(jnp.eye(n_B))
a = np.asarray(jax.jit(naive)(rij)).transpose(1,0,2)
print(f"\nnaive (458,50,2) = {a[458,50,2]:+.8e}")
print(f"max|naive-hybrid_jit|/max|hybrid| = {np.abs(a-b_jit).max()/np.abs(b_jit).max():.3e}")
