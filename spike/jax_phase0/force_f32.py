"""Forces = -grad(sum of site energies) wrt rij: the Stage 1 / LAMMPS path."""
import json, os
import numpy as np, jax, jax.numpy as jnp
X64 = os.environ.get("X64","1")=="1"
import etace_jax as E
jax.config.update("jax_enable_x64", X64)
M = E.load("si_model.json"); BD = json.load(open("bench_data.json"))
d = BD[sorted(BD, key=lambda k: BD[k]["n_atoms"])[0]]
n_nodes = d["n_atoms"]
rij = jnp.asarray(d["edge_rij"]); send = jnp.asarray(d["edge_i"], dtype=jnp.int32)
g = jax.jit(jax.grad(lambda r: jnp.sum(E.site_energies(r, send, n_nodes, M))))
F = np.asarray(g(rij), dtype=np.float64)
print(f"x64={jax.config.jax_enable_x64} F_sum={F.sum():+.10e} F_absmax={np.abs(F).max():.10e} "
      f"F_norm={np.linalg.norm(F):.10e}")
