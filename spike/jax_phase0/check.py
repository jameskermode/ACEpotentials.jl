"""Phase 0 spike: numerical agreement of the JAX descriptor vs the Julia reference."""
import numpy as np, jax.numpy as jnp, jax
import etace_jax as E

M = E.load("si_model.json")
pr = M["probe"]
rij_p = jnp.asarray(pr["rij"])

def rep(name, got, ref, tol=1e-10):
    got, ref = np.asarray(got), np.asarray(ref)
    denom = max(np.abs(ref).max(), 1e-300)
    err = np.abs(got - ref).max()
    rel = err / denom
    print(f"{name:<26} max|abs|={err:.3e}  max|rel|={rel:.3e}  {'OK' if rel < tol else 'FAIL'}")
    return rel < tol

print("=== intermediates (6 probe vectors) ===")
r = jnp.linalg.norm(rij_p, axis=-1)
rep("agnesi y", E.agnesi(r, M["agnesi"]), pr["y_agnesi"])
rep("Rnl", E.radial(rij_p, M), pr["Rnl"])
ylm_j = np.asarray(pr["Ylm"])
ylm_x = np.asarray(E.angular(rij_p, M))
ok_y = rep("Ylm (sphericart-jax)", ylm_x, ylm_j)
if not ok_y:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ylm_x / ylm_j
    print("   per-l ratio (jax/julia):")
    for l in range(M["lmax"] + 1):
        idx = [k for k, (ll, _) in enumerate(M["ylm_spec"]) if ll == l]
        vals = ratio[:, idx]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            print(f"     l={l}: mean={vals.mean():+.9f} std={vals.std():.2e}")

print("\n=== full descriptor (64-atom Si) ===")
T = M["test"]
rij = jnp.asarray(T["edge_rij"]); send = jnp.asarray(T["edge_i"], dtype=jnp.int32)
n = int(T["n_atoms"])
print(f"n_atoms={n} n_edges={T['n_edges']}")
B = E.descriptor(rij, send, n, M)
rep("B (site basis)", B, T["B_ref"])
phi = E.site_energies(rij, send, n, M)
rep("phi (site energies)", phi, T["phi_ref"])
