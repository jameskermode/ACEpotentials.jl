"""ACE (many-element, frozen embedding) vs MACE-MP-0 small on the same GPU and
the same structures.

Timed BY DIFFERENCE: t(N frames) - t(1 frame), so compile, model load and file
I/O -- all fixed costs -- cancel rather than being estimated.  Isolated stage
timings have over-counted repeatedly on this project; differences have not.

No fitting on the ACE side (ACE_NOFIT=1): evaluation cost depends on the basis,
not the weights.
"""
import os, sys, time, pathlib, tempfile, subprocess, json
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

NREP = int(os.environ.get("NFRAMES", "64"))

def frames_from_npz(npz, n):
    z = np.load(npz)
    pos = np.asarray(z["test_pos"]).T
    cell = np.asarray(z["test_cell"]).T
    Z = np.asarray(z["test_Z"]).ravel()
    rng = np.random.default_rng(0)
    from ase import Atoms
    return [Atoms(numbers=Z, positions=pos + 0.01 * rng.standard_normal(pos.shape),
                  cell=cell, pbc=True) for _ in range(n)]

def time_mace(bundle, frames, tag):
    from ase.io import write as ase_write
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="h2h-"))
    def run(k):
        xyz = tmp / f"s{k}.xyz"; out = tmp / f"o{k}.npz"
        ase_write(xyz, frames[:k], format="extxyz")
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-m", "mace_jax.cli.mace_jax_predict",
                        bundle, str(xyz), "--output", str(out), "--dtype", "float64",
                        "--compute-forces", "--no-progress"],
                       check=True, capture_output=True)
        return time.perf_counter() - t0
    t1, tN = run(1), run(len(frames))
    per = (tN - t1) / (len(frames) - 1)
    print(f"{tag:<28} t(1)={t1:7.2f}s  t({len(frames)})={tN:7.2f}s  "
          f"per-frame={per*1e3:8.3f} ms  {len(frames[0])/per:.3e} atom-steps/s")
    return per

def time_ace(npz, frames, tag):
    import jax; jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from acejax import load
    from acejax.nlist import sparse_graph
    model, meta, z = load(npz)
    rcut = float(meta["rcut"])
    from acejax.nlist import backend
    print(f"  neighbour-list backend: {backend()}")
    i2z = list(np.asarray(z["elements"]).ravel()); lut = {int(v): i for i, v in enumerate(i2z)}
    # PAD to a fixed edge capacity.  Without this the edge count varies frame to
    # frame, every frame is a new shape, and jax retraces on every single one --
    # which is what a first version of this benchmark actually measured.  Fixed
    # capacity is the whole reason the export contract uses it.
    cap = int(max(len(sparse_graph(a.get_positions(), a.get_cell().array,
                                   a.get_pbc(), rcut).senders)
                  for a in frames[:16]) * 1.2) + 16
    print(f"  edge capacity: {cap}")
    def ev(at):
        g = sparse_graph(at.get_positions(), at.get_cell().array, at.get_pbc(),
                         rcut, pad_to=cap)
        send = jnp.asarray(g.senders, jnp.int32); recv = jnp.asarray(g.receivers, jnp.int32)
        nz = jnp.asarray([lut[int(a)] for a in at.get_atomic_numbers()], jnp.int32)
        return jnp.asarray(g.rij), send, recv, nz, len(at)
    args0 = ev(frames[0])
    fn = jax.jit(lambda r, s, rc, nz, n: jnp.sum(
        jax.grad(lambda rr: jnp.sum(model.site_energies(rr, nz[s], nz[rc], s, n, nz)))(r)),
        static_argnums=(4,))
    jax.block_until_ready(fn(*args0))          # compile once, outside the timing
    def run(k):
        t0 = time.perf_counter()
        for at in frames[:k]:
            jax.block_until_ready(fn(*ev(at)))
        return time.perf_counter() - t0
    t1, tN = run(1), run(len(frames))
    per = (tN - t1) / (len(frames) - 1)
    print(f"{tag:<28} t(1)={t1:7.2f}s  t({len(frames)})={tN:7.2f}s  "
          f"per-frame={per*1e3:8.3f} ms  {len(frames[0])/per:.3e} atom-steps/s")
    return per

if __name__ == "__main__":
    which, arg, npz_for_frames = sys.argv[1], sys.argv[2], sys.argv[3]
    frames = frames_from_npz(npz_for_frames, NREP)
    print(f"{len(frames)} frames of {len(frames[0])} atoms, "
          f"{len(set(frames[0].get_atomic_numbers()))} distinct elements")
    (time_mace if which == "mace" else time_ace)(arg, frames, sys.argv[4])
