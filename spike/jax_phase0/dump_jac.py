"""Dump naive+hybrid Jacobians at a chosen precision, for cross-precision comparison."""
import json, os
import numpy as np, jax, jax.numpy as jnp
X64 = os.environ.get("X64", "1") == "1"
import etace_jax as E
jax.config.update("jax_enable_x64", X64)
M = E.load("si_model.json"); BD = json.load(open("bench_data.json"))
d = BD[sorted(BD, key=lambda k: BD[k]["n_atoms"])[0]]
n_nodes = d["n_atoms"]; n_B = M["meta"]["n_B"]
rij = jnp.asarray(d["edge_rij"]); send = jnp.asarray(d["edge_i"], dtype=jnp.int32)

def naive(r_):
    _, vjp = jax.vjp(lambda q: E.descriptor(q, send, n_nodes, M), r_)
    return jax.vmap(lambda row: vjp(jnp.broadcast_to(row, (n_nodes, n_B)))[0])(jnp.eye(n_B))

def hybrid(r_):
    def edge_feats(r1): return E.radial(r1[None,:],M)[0], E.angular(r1[None,:],M)[0]
    def node_B(A_i):
        AA = jnp.concatenate([jnp.prod(A_i[g],axis=-1) for g in M["aa_groups"]])
        return M["A2B"] @ AA
    Rnl, Ylm = jax.vmap(edge_feats)(r_)
    dRnl, dYlm = jax.vmap(jax.jacfwd(edge_feats))(r_)
    A = E.pool_A(Rnl, Ylm, send, n_nodes, M)
    ar, ay = M["aspec_r"], M["aspec_y"]
    dA = dRnl[:,ar,:]*Ylm[:,ay,None] + Rnl[:,ar,None]*dYlm[:,ay,:]
    J_BA = jax.vmap(jax.jacrev(node_B))(A)
    return jnp.einsum('ebm,emc->ebc', J_BA[send], dA), A, J_BA, dA

a = np.asarray(jax.jit(naive)(rij)).transpose(1,0,2)
b, A, J_BA, dA = [np.asarray(x) for x in jax.jit(hybrid)(rij)]
tag = "f64" if X64 else "f32"
np.savez(f"jac_{tag}.npz", naive=a.astype(np.float64), hybrid=b.astype(np.float64),
         A=A.astype(np.float64), J_BA=J_BA.astype(np.float64), dA=dA.astype(np.float64))
print(f"wrote jac_{tag}.npz  x64={jax.config.jax_enable_x64}  naive{a.shape} hybrid{b.shape}")
