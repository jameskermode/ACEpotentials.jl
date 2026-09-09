"""Validate the exported model through lammps-jax's own ABI wrappers.

LAMMPS itself cannot start on this host (AVX-512 binary, non-AVX-512 CPU -- see
the report), so this exercises everything up to the C++ pair style: the real
`wrap_energy_fn` wrappers, the padded fixed-capacity edge list, the masking and
ghost-energy conventions, and forces by autodiff.

Reference is the same model core called directly on an unpadded edge list, so
the comparison isolates the ABI plumbing (padding, out-of-range indices,
masking, the wrapper's local/total energy selection) rather than the physics,
which stage1/tests already pins against Julia.

A non-periodic cluster keeps nghost = 0; ghost bookkeeping is LAMMPS's job and
is exactly what this cannot cover.
"""
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from lammps_jax.export import LammpsNeighborList, wrap_energy_fn

sys.path.insert(0, "/home/eng/essswb/si-ace/stage1")
sys.path.insert(0, "/home/eng/essswb/si-ace")
from export_bundle import MAX_ATOMS, MAX_EDGES, RCUT, energy_fn, meta, model

rng = np.random.default_rng(3)
a = 5.43
basis = np.array([[0, 0, 0], [.25, .25, .25], [0, .5, .5], [.25, .75, .75],
                  [.5, 0, .5], [.75, .25, .75], [.5, .5, 0], [.75, .75, .25]])
pos = np.array([(np.array([i, j, k]) + b) * a
                for i in range(2) for j in range(2) for k in range(2) for b in basis])
pos = pos + 0.05 * rng.normal(size=pos.shape)
n = len(pos)

d = pos[:, None, :] - pos[None, :, :]
r = np.linalg.norm(d, axis=-1)
ii, jj = np.where((r < RCUT) & (r > 0))
order = np.argsort(ii, kind="stable")
ii, jj = ii[order], jj[order]
ne = len(ii)
print(f"cluster: {n} atoms, non-periodic, {ne} edges (capacity {MAX_EDGES})")

# ---- reference: core called directly, unpadded
rij = jnp.asarray(pos[jj] - pos[ii])
send, recv = jnp.asarray(ii, jnp.int32), jnp.asarray(jj, jnp.int32)
nz = jnp.zeros(n, jnp.int32)
with jax.default_matmul_precision("highest"):
    E_ref, F_ref, _ = model.energy_forces_virial(rij, nz[send], nz[recv], send, recv, n, nz)
E_ref = float(E_ref); F_ref = np.asarray(F_ref)

# ---- test: LAMMPS ABI, padded to capacity
positions = np.zeros((MAX_ATOMS, 3)); positions[:n] = pos
species = np.zeros(MAX_ATOMS, np.int32)
senders = np.full(MAX_EDGES, MAX_ATOMS, np.int32); senders[:ne] = ii
receivers = np.full(MAX_EDGES, MAX_ATOMS, np.int32); receivers[:ne] = jj
edge_mask = np.zeros(MAX_EDGES, bool); edge_mask[:ne] = True
args = (jnp.asarray(positions), jnp.asarray(species), jnp.asarray(np.int32(n)),
        jnp.asarray(np.int32(0)), jnp.asarray(senders), jnp.asarray(receivers),
        jnp.asarray(edge_mask))


def call_model(model_fn, model_args):
    p, s, nl, ng, sd, rc, em = model_args[:7]
    graph = LammpsNeighborList(senders=sd, receivers=rc, edge_mask=em)
    return model_fn(p, s, graph), graph, nl, ng


_, _, fused = wrap_energy_fn(energy_fn, max_atoms=MAX_ATOMS,
                             call_model=call_model, dtype=jnp.float64)
with jax.default_matmul_precision("highest"):
    E_abi, F_all = jax.jit(fused)(*args)
E_abi = float(E_abi); F_all = np.asarray(F_all)

dE = abs(E_abi - E_ref)
dF = np.max(np.abs(F_all[:n] - F_ref))
pad_F = np.max(np.abs(F_all[n:]))
print(f"  core E = {E_ref:.10f} eV")
print(f"  ABI  E = {E_abi:.10f} eV   |dE| = {dE:.3e}  ({dE/abs(E_ref):.2e} rel)")
print(f"  max|dF| on real atoms = {dF:.3e} eV/A  (|F| scale {np.max(np.abs(F_ref)):.4f})")
print(f"  max|F| on padded rows = {pad_F:.3e} eV/A  (must be 0)")
print(f"  non-finite in F: {int((~np.isfinite(F_all)).sum())}")
ok = dE < 1e-9 and dF < 1e-9 and pad_F == 0.0 and np.all(np.isfinite(F_all))
print("  " + ("OK" if ok else "FAIL"))
sys.exit(0 if ok else 1)
