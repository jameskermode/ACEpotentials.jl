# Multi-device TSQR in JAX via shard_map over row blocks. There is one GPU on
# this host, so this is run on 8 *virtual CPU devices* to prove the code path
# and check agreement; the same code targets multiple GPUs by changing the mesh.
import os, sys, json, glob, time
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["JAX_PLATFORMS"] = "cpu"
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.shard_map import shard_map
from jax.scipy.linalg import solve_triangular

devs = jax.devices(); nd = len(devs)
print(f"jax {jax.__version__}  devices={nd} x {devs[0].platform}")
mesh = Mesh(np.array(devs), ("rows",))

def tsqr_solve(Aa, ya):
    n = Aa.shape[1]
    def local(Ab, yb):                                   # per-device row block
        Q, R = jnp.linalg.qr(Ab, mode="reduced")         # R: n x n, z: n
        z = Q.T @ yb
        Rs = jax.lax.all_gather(R, "rows", axis=0, tiled=True)   # (nd*n) x n on every device
        zs = jax.lax.all_gather(z, "rows", axis=0, tiled=True)
        Q2, R2 = jnp.linalg.qr(Rs, mode="reduced")       # reduce step (replicated)
        x = solve_triangular(R2, Q2.T @ zs, lower=False)
        return x[None, :]
    f = shard_map(local, mesh=mesh, in_specs=(P("rows", None), P("rows")), out_specs=P("rows"), check_rep=False)
    xs = jax.jit(f)(Aa, ya)
    return xs[0]

root = sys.argv[1]
for d in sorted(glob.glob(f"{root}/acc_*")):
    meta = json.load(open(f"{d}/meta.json")); m, n = meta["m"], meta["n"]
    A = np.fromfile(f"{d}/A.bin", dtype=np.float64).reshape(m, n); y = np.fromfile(f"{d}/y.bin", dtype=np.float64)
    for lam in (0.0, 1e-3):
        xref = np.fromfile(f"{d}/xqr_lam{lam}.bin", dtype=np.float64)
        Aa = np.vstack([A, lam*np.eye(n)]) if lam > 0 else A
        ya = np.concatenate([y, np.zeros(n)]) if lam > 0 else y
        pad = (-Aa.shape[0]) % nd                        # rows must divide evenly across devices
        if pad: Aa = np.vstack([Aa, np.zeros((pad, n))]); ya = np.concatenate([ya, np.zeros(pad)])
        sh = NamedSharding(mesh, P("rows", None)); shv = NamedSharding(mesh, P("rows"))
        Ad = jax.device_put(jnp.asarray(Aa), sh); yd = jax.device_put(jnp.asarray(ya), shv)
        x = tsqr_solve(Ad, yd); jax.block_until_ready(x)
        t0 = time.perf_counter(); x = tsqr_solve(Ad, yd); jax.block_until_ready(x); t = time.perf_counter()-t0
        x = np.asarray(x); r = A@x - y
        print(f"{os.path.basename(d):20s} lam={lam:<6g} shard_map TSQR on {nd} devices: agree_vs_cpu_qr={np.linalg.norm(x-xref)/np.linalg.norm(xref):.3e} "
              f"rel.resid={np.linalg.norm(r)/np.linalg.norm(y):.3e} normality={np.linalg.norm(A.T@r)/(meta['normA']*np.linalg.norm(r)):.3e} t={t:.3f}s")
