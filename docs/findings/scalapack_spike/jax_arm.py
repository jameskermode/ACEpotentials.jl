# JAX / lineax arm, float64, on the same matrices dumped by dump_cases.jl.
import os, sys, json, time, glob
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import lineax as lx
from jax.scipy.linalg import solve_triangular

dev = jax.devices()[0]
print(f"jax {jax.__version__} lineax {lx.__version__} device={dev} x64={jax.config.jax_enable_x64} matmul={jax.config.jax_default_matmul_precision}")

def load(d):
    meta = json.load(open(f"{d}/meta.json")); m, n = meta["m"], meta["n"]
    A = np.fromfile(f"{d}/A.bin", dtype=np.float64)
    A = A.reshape(n, m).T if meta.get("order") == "F" else A.reshape(m, n)
    y = np.fromfile(f"{d}/y.bin", dtype=np.float64)
    return meta, A, y

def metrics(A, y, x, xref, normA):
    r = A @ x - y; nr = np.linalg.norm(r)
    return dict(res=nr/np.linalg.norm(y), agree=np.linalg.norm(x-xref)/np.linalg.norm(xref),
                normality=np.linalg.norm(A.T @ r)/(normA*nr))

@jax.jit
def qr_solve(Aa, ya):
    Q, R = jnp.linalg.qr(Aa, mode="reduced")
    return solve_triangular(R, Q.T @ ya, lower=False)

@jax.jit
def lstsq_solve(Aa, ya):
    return jnp.linalg.lstsq(Aa, ya)[0]

def lineax_solve(Aa, ya, solver):
    op = lx.MatrixLinearOperator(Aa)
    sol = lx.linear_solve(op, ya, solver=solver, throw=False)
    return sol.value, str(sol.result).split(".")[-1], sol.stats

def timed(f, *a):
    out = f(*a); jax.block_until_ready(out); t0 = time.perf_counter()
    out = f(*a); jax.block_until_ready(out); return out, time.perf_counter() - t0

def run_case(d, lam, methods):
    meta, A, y = load(d); m, n = meta["m"], meta["n"]
    xref = np.fromfile(f"{d}/xqr_lam{lam}.bin", dtype=np.float64)
    normA = meta["normA"]
    # augmented system [A; lam*I] \ [y; 0]  (P = I here; ACEfit's P is diagonal and pre-applied)
    Aa = np.vstack([A, lam*np.eye(n)]) if lam > 0 else A
    ya = np.concatenate([y, np.zeros(n)]) if lam > 0 else y
    Aa_d = jax.device_put(jnp.asarray(Aa), dev); ya_d = jax.device_put(jnp.asarray(ya), dev)
    tag = f"{os.path.basename(d)} lam={lam} m={m} n={n} cond_meas={meta.get('cond_measured', float('nan')):.2e}"
    print(f"\n### {tag}   (CPU LAPACK qr: {meta[f't_cpu_qr_lam{lam}']:.2f}s)")
    for name in methods:
        try:
            if name == "jnp.linalg.qr+trsolve":
                x, t = timed(qr_solve, Aa_d, ya_d); extra = ""
            elif name == "jnp.linalg.lstsq(SVD)":
                x, t = timed(lstsq_solve, Aa_d, ya_d); extra = ""
            elif name.startswith("lineax."):
                sname = name.split(".")[1]
                solver = {"QR": lx.QR(), "SVD": lx.SVD(), "LSMR": lx.LSMR(rtol=1e-14, atol=1e-14, max_steps=20*n),
                          "NormalCG": lx.Normal(lx.CG(rtol=1e-14, atol=1e-14, max_steps=20*n))}[sname]
                (x, res, stats), t = timed(lambda a, b: lineax_solve(a, b, solver), Aa_d, ya_d)
                extra = f" result={res} steps={stats.get('num_steps', '-')}"
            x = np.asarray(x)
            mt = metrics(A, y, x, xref, normA)   # metrics on the *unaugmented* A,y; agreement vs CPU qr on augmented
            print(f"  {name:26s} agree_vs_cpu_qr={mt['agree']:.3e}  rel.resid={mt['res']:.3e}  normality={mt['normality']:.3e}  gpu={t:.3f}s{extra}")
        except Exception as e:
            print(f"  {name:26s} FAILED: {type(e).__name__}: {str(e)[:200]}")
        finally:
            jax.clear_caches()

if __name__ == "__main__":
    root = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "acc"
    if mode == "acc":
        for d in sorted(glob.glob(f"{root}/acc_*")):
            for lam in (0.0, 1e-3):
                run_case(d, lam, ["jnp.linalg.qr+trsolve", "jnp.linalg.lstsq(SVD)", "lineax.QR", "lineax.SVD", "lineax.LSMR", "lineax.NormalCG"])
    else:
        for d in sorted(glob.glob(f"{root}/time_*")):
            run_case(d, 1e-3, ["jnp.linalg.qr+trsolve", "lineax.QR"])
