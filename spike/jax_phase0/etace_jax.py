"""Phase 0 spike: JAX implementation of the ETACE descriptor (A -> AA -> B).

SPIKE CODE. See docs/plans/jax_ace_port_plan.md.

Pipeline, per edge e = (i, j) with vector r_ij:
    y    = agnesi(|r_ij|)                       transformed distance
    P    = polys(y)                             3-term recursion
    Pe   = (1 - y^2)^2 * P                      envelope
    Rnl  = W_rnl @ Pe                           SelectLinL (1 species pair here)
    Ylm  = solid_harmonics(r_ij, lmax)
then per node i:
    A[i,k]  = sum_{e: send(e)=i} Rnl[e, ar[k]] * Ylm[e, ay[k]]
    AA[i,v] = prod_t A[i, aa[v,t]]              grouped by correlation order
    B[i,:]  = AA[i,:] @ A2B.T
"""

import json
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def load(path):
    with open(path) as f:
        D = json.load(f)
    M = {}
    M["agnesi"] = {k: float(v) for k, v in D["agnesi"].items()}
    M["agnesi"]["pin"] = int(M["agnesi"]["pin"])
    M["agnesi"]["pcut"] = int(M["agnesi"]["pcut"])
    for k in ("polys_A", "polys_B", "polys_C", "W_readout"):
        M[k] = jnp.asarray(D[k], dtype=jnp.float64)
    # NB: Julia's JSON writes a Matrix column-major (as a list of columns),
    # so 2-D arrays arrive transposed. A real exporter should use npz.
    M["W_rnl"] = jnp.asarray(D["W_rnl"], dtype=jnp.float64).T   # -> (n_rnl, n_polys)
    M["aspec_r"] = jnp.asarray(D["aspec_r"], dtype=jnp.int32)
    M["aspec_y"] = jnp.asarray(D["aspec_y"], dtype=jnp.int32)
    M["ylm_spec"] = D["ylm_spec"]
    M["lmax"] = max(l for l, _ in D["ylm_spec"])
    # AA: pad each order group to (n_v, order)
    M["aa_groups"] = [jnp.asarray(g, dtype=jnp.int32) for g in D["aaspec_by_order"]]
    # A2B as a dense matrix (110 x 230 here -- trivially small)
    shape = tuple(D["A2B_shape"])
    A2B = np.zeros(shape, dtype=np.float64)
    A2B[np.asarray(D["A2B_rows"]), np.asarray(D["A2B_cols"])] = np.asarray(D["A2B_vals"])
    M["A2B"] = jnp.asarray(A2B)
    M["meta"] = D["meta"]
    M["test"] = dict(D["test"])
    M["test"]["B_ref"] = np.asarray(D["test"]["B_ref"]).T        # -> (n_atoms, n_B)
    M["probe"] = D["probe"]
    return M


# ---------------------------------------------------------------- radial
def agnesi(r, p):
    """Generalized Agnesi transform. Matches ET.eval_agnesi."""
    s = (r - p["rin"]) / (p["req"] - p["rin"])
    x = 1.0 / (1.0 + p["a"] * s ** p["pin"] / (1.0 + s ** (p["pin"] - p["pcut"])))
    return jnp.clip(p["b1"] * x + p["b0"], -1.0, 1.0)


def polys(y, A, B, C):
    """OrthPolyBasis1D3T: P1=A1, P2=A2*y+B2, Pn=(An*y+Bn)Pn-1 + Cn*Pn-2."""
    n = A.shape[0]
    out = [jnp.broadcast_to(A[0], y.shape), A[1] * y + B[1]]
    for k in range(2, n):
        out.append((A[k] * y + B[k]) * out[k - 1] + C[k] * out[k - 2])
    return jnp.stack(out, axis=-1)                       # (..., n_polys)


def radial(rij, M):
    """rij: (n_edges, 3) -> Rnl: (n_edges, n_rnl)"""
    r = jnp.linalg.norm(rij, axis=-1)
    y = agnesi(r, M["agnesi"])
    P = polys(y, M["polys_A"], M["polys_B"], M["polys_C"])
    Pe = ((1.0 - y**2) ** 2)[:, None] * P
    return Pe @ M["W_rnl"].T                              # (n_edges, n_rnl)


def angular(rij, M):
    import sphericart.jax as scj
    return scj.solid_harmonics(rij, M["lmax"])            # (n_edges, n_ylm)


# ---------------------------------------------------------------- A / AA / B
def pool_A(Rnl, Ylm, send, n_nodes, M):
    """A[i,k] = sum_{e: send=i} Rnl[e, ar[k]] * Ylm[e, ay[k]]"""
    prod = Rnl[:, M["aspec_r"]] * Ylm[:, M["aspec_y"]]    # (n_edges, n_A)
    return jax.ops.segment_sum(prod, send, num_segments=n_nodes,
                               indices_are_sorted=True)   # (n_nodes, n_A)


def sym_prod_AA(A, M):
    """AA grouped by correlation order: gather (n_v, order) then prod."""
    parts = []
    for g in M["aa_groups"]:                              # g: (n_v, order)
        parts.append(jnp.prod(A[:, g], axis=-1))          # (n_nodes, n_v)
    return jnp.concatenate(parts, axis=-1)                # (n_nodes, n_AA)


def descriptor(rij, send, n_nodes, M):
    """Site basis B: (n_nodes, n_B)."""
    Rnl = radial(rij, M)
    Ylm = angular(rij, M)
    A = pool_A(Rnl, Ylm, send, n_nodes, M)
    AA = sym_prod_AA(A, M)
    return AA @ M["A2B"].T


def site_energies(rij, send, n_nodes, M):
    return descriptor(rij, send, n_nodes, M) @ M["W_readout"]
