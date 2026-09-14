"""Per-frame latency of MACE (mace-jax) and an exported ACE model, IN PROCESS,
on the same frames, the same GPU and the same protocol.

`manyelem/head2head.py` times mace_jax_predict as a subprocess by difference of
two runs.  With 64-atom frames that difference is ~0.2 s on top of a 40-70 s
fixed cost that itself varies by seconds between runs, so the result is noise
(a rerun gave a NEGATIVE per-frame time for MP-0-small).  Here both models are
jitted once, inputs are padded to a fixed capacity so there is exactly one
compile, and each frame is timed with block_until_ready.

Two numbers per model:
  kernel   -- device time only: pre-built padded inputs, N calls
  frame    -- host graph build (neighbour list, padding, transfer) + kernel
The kernel number is what an MD driver that keeps the neighbour list on the
device would see; the frame number is what an ASE calculator sees.

    python latency.py --mace bundles/mace-mh-1.msgpack --ace cantor_d16_deg6.npz \
        --frames cantor1k_b.xyz --n 64
"""
import argparse, os, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax, jax.numpy as jnp
from ase.io import read

ap = argparse.ArgumentParser()
ap.add_argument("--mace", action="append", default=[])
ap.add_argument("--ace", action="append", default=[])
ap.add_argument("--frames", required=True)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
ap.add_argument("--supercell", type=int, default=1, help="replicate each frame k x k x k")
a = ap.parse_args()
jax.config.update("jax_enable_x64", a.dtype == "float64")
DT = jnp.float64 if a.dtype == "float64" else jnp.float32

frames = read(a.frames, f":{a.n}")
if a.supercell > 1:
    frames = [f.repeat(a.supercell) for f in frames]
nat = np.array([len(f) for f in frames])
print(f"{len(frames)} frames, {nat.min()}-{nat.max()} atoms, backend {jax.default_backend()}, {a.dtype}")

def report(tag, tk, tf, natoms):
    print(f"{tag:<26} kernel {tk*1e3:8.3f} ms/frame  {natoms/tk:.3e} atom-steps/s   "
          f"frame {tf*1e3:8.3f} ms/frame  {natoms/tf:.3e} atom-steps/s")

def timeit(fn, inputs):
    # median over reps of the mean per-frame time; compile happens before
    best = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        for x in inputs:
            jax.block_until_ready(fn(x))
        best.append((time.perf_counter() - t0) / len(inputs))
    return float(np.median(best))

# ----------------------------------------------------------------- MACE
for bundle in a.mace:
    import jraph
    from mace_jax import data
    from mace_jax.tools import bundle as bundle_tools
    from mace_jax.cli.mace_jax_predict import _build_predictor
    b = bundle_tools.load_model_bundle(bundle, a.dtype)
    pred = jax.jit(_build_predictor(b.graphdef, model_config=b.config,
                                    compute_forces=True, compute_stress=False,
                                    full_outputs=False))
    z_table = data.AtomicNumberTable(b.config["atomic_numbers"])
    rmax = float(b.config["r_max"])
    def graph_of(at):
        cfg = data.config_from_atoms(at)
        return data.graph_from_configuration(cfg, cutoff=rmax, z_table=z_table)
    raw = [graph_of(at) for at in frames]
    n_edge_cap = int(max(int(g.n_edge.sum()) for g in raw) * 1.2) + 16
    n_node_cap = int(nat.max()) + 1
    def pad(g):
        return jraph.pad_with_graphs(g, n_node=n_node_cap, n_edge=n_edge_cap, n_graph=2)
    padded = [pad(g) for g in raw]
    jax.block_until_ready(pred(b.params, padded[0]))
    tk = timeit(lambda g: pred(b.params, g), padded)
    tf = timeit(lambda at: pred(b.params, pad(graph_of(at))), frames)
    print(f"  {os.path.basename(bundle)}: r_max {rmax}, edge cap {n_edge_cap}")
    report("MACE " + os.path.basename(bundle), tk, tf, nat.mean())

# ----------------------------------------------------------------- ACE
for npz in a.ace:
    from acejax import load
    from acejax.nlist import sparse_graph, backend
    model, meta, z = load(npz, dtype=DT)
    rcut = float(meta["rcut"])
    lut = {int(v): i for i, v in enumerate(np.asarray(z["elements"]).ravel())}
    cap = int(max(len(sparse_graph(f.get_positions(), f.get_cell().array, f.get_pbc(), rcut).senders)
                  for f in frames) * 1.2) + 16
    def ev(at):
        g = sparse_graph(at.get_positions(), at.get_cell().array, at.get_pbc(), rcut, pad_to=cap)
        nz = jnp.asarray([lut[int(q)] for q in at.get_atomic_numbers()], jnp.int32)
        return (jnp.asarray(g.rij, DT), jnp.asarray(g.senders, jnp.int32),
                jnp.asarray(g.receivers, jnp.int32), nz, len(at))
    fn = jax.jit(lambda r, s, rc, nz, n: jax.grad(
        lambda rr: jnp.sum(model.site_energies(rr, nz[s], nz[rc], s, n, nz)))(r),
        static_argnums=(4,))
    # one compile per distinct atom count (static n); warm all of them
    built = [ev(f) for f in frames]
    for x in built:
        jax.block_until_ready(fn(*x))
    tk = timeit(lambda x: fn(*x), built)
    tf = timeit(lambda at: fn(*ev(at)), frames)
    print(f"  {os.path.basename(npz)}: rcut {rcut}, edge cap {cap}, nlist {backend()}")
    report("ACE " + os.path.basename(npz), tk, tf, nat.mean())
