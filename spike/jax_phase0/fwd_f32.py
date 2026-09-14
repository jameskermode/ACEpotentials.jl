"""Forward descriptor vs Julia reference at a chosen precision."""
import json, os
import numpy as np, jax, jax.numpy as jnp
X64 = os.environ.get("X64","1")=="1"
import etace_jax as E
jax.config.update("jax_enable_x64", X64)
M = E.load("si_model.json")
T = M["test"]
rij = jnp.asarray(T["edge_rij"]); send = jnp.asarray(T["edge_i"], dtype=jnp.int32)
n_nodes = int(T["n_atoms"]); ref = np.asarray(M["test"]["B_ref"], dtype=np.float64)
B = np.asarray(jax.jit(lambda r: E.descriptor(r, send, n_nodes, M))(rij), dtype=np.float64)
print(f"x64={jax.config.jax_enable_x64} B_rel={np.abs(B-ref).max()/np.abs(ref).max():.4e}")
