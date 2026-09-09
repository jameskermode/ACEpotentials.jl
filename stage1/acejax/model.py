"""ACE model as an Equinox module: A -> AA -> B -> site energy.

Design notes that the plan fixes and this code must not quietly undo:

* **Edge-vector core.** `site_energies` takes edge vectors, species indices and
  segment ids -- never positions and a cell.  The LAMMPS contract has no cell
  (ghost atoms carry periodicity), so a positions+cell core could not serve it.
  `energy_from_positions` is the thin wrapper for callers that do have
  positions, matching lammps-jax's `energy_fn(positions, species, graph)`.

* **Pooling is swappable.** `pool_sparse` (segment_sum over an edge list) and
  `pool_dense` (masked sum over a fixed (n, K) neighbour axis) are
  interchangeable; everything downstream is per-node and layout-agnostic.

* **Radial coefficients are a live array leaf, not static.**  For the splined
  branch Wnlq is already folded into the spline coefficients by Julia's
  `splinify`, so the coefficients occupy Wnlq's place in the parameter tree.
  Keeping them a leaf is what keeps Stage 2 and non-linear fits reachable; the
  analytic branch will carry a true Wnlq alongside.

* **Precision is explicit.**  Nothing here calls `jax.config.update`.  Use
  `with jax.default_matmul_precision("highest")` (or the `highest_precision`
  helper) -- the TF32 default costs ~400x accuracy on this descriptor.
"""

from contextlib import contextmanager

import equinox as eqx
import jax
import jax.numpy as jnp
import sphericart.jax as scj

from .radial import (agnesi_normalized, env_ace1_poly1sr, env_poly2sx,
                     spline_eval)


@contextmanager
def highest_precision():
    """Force true f32/f64 matmuls.  On Ampere+ the TF32 default silently costs
    ~400x accuracy (1.17e-3 vs 2.93e-6 on this descriptor)."""
    with jax.default_matmul_precision("highest"):
        yield


def pool_sparse(edge_feats, segment_ids, n_nodes, mask=None):
    """Sum edge features into nodes over a sparse edge list."""
    if mask is not None:
        edge_feats = jnp.where(mask[:, None], edge_feats, 0.0)
    return jax.ops.segment_sum(edge_feats, segment_ids, num_segments=n_nodes,
                               indices_are_sorted=False)


def pool_dense(edge_feats, mask):
    """Sum over a fixed neighbour axis: edge_feats (n_nodes, K, F), mask (n_nodes, K)."""
    return jnp.sum(jnp.where(mask[..., None], edge_feats, 0.0), axis=1)


class ACEModel(eqx.Module):
    # ---- array leaves (parameters) ----
    rnl_coefs: jax.Array          # (NZ, NZ, ncoef, n_rnl)
    pair_coefs: jax.Array         # (NZ, NZ, ncoef, n_pair)
    rnl_transform: jax.Array      # (NZ, NZ, 7)
    pair_transform: jax.Array     # (NZ, NZ, 7)
    rnl_envelope: jax.Array       # (NZ, NZ, 5)
    pair_envelope: jax.Array      # (NZ, NZ, 3)
    A2B: jax.Array                # (n_B, n_AA)
    WB: jax.Array                 # (n_B, NZ)
    Wpair: jax.Array              # (n_pair, NZ)
    E0: jax.Array                 # (NZ,)
    # ---- static structure ----
    # index arrays are int32 leaves, not static fields: marking a JAX array
    # static warns and is a mistake; integer leaves are simply not differentiated
    aspec_r: jax.Array
    aspec_y: jax.Array
    aa_specs: tuple                               # per-order (n_v, order) int arrays
    lmax: int = eqx.field(static=True)
    ysolid: bool = eqx.field(static=True)
    rnl_grid: tuple = eqx.field(static=True)      # (x0, h, n)
    pair_grid: tuple = eqx.field(static=True)
    elements: tuple = eqx.field(static=True)

    # -------------------------------------------------- edge embeddings
    def radial(self, rij, zi, zj):
        """Rnl and the pair radial for each edge.  rij (E,3), zi/zj (E,) species indices."""
        r = jnp.linalg.norm(rij, axis=-1)
        tp = self.rnl_transform[zi, zj]
        x = agnesi_normalized(r, tp)
        env = env_poly2sx(x, self.rnl_envelope[zi, zj])
        x0, h, n = self.rnl_grid
        # gather the (ncoef, n_rnl) coefficient block for each edge's species pair
        spl = jax.vmap(lambda xx, c: spline_eval(xx, c, x0, h, n))(x, self.rnl_coefs[zi, zj])
        Rnl = spl * env[:, None]

        xp = agnesi_normalized(r, self.pair_transform[zi, zj])
        envp = env_ace1_poly1sr(r, self.pair_envelope[zi, zj])
        px0, ph, pn = self.pair_grid
        splp = jax.vmap(lambda xx, c: spline_eval(xx, c, px0, ph, pn))(xp, self.pair_coefs[zi, zj])
        Rpair = splp * envp[:, None]
        return Rnl, Rpair

    def angular(self, rij):
        # ace1_model uses SPHERICAL harmonics (ace1_compat.jl:408, Ytype=:spherical);
        # ace_model defaults to :solid.  The Phase 0 spike used ace_model, so the
        # production path differs from it here -- hence the exported flag.
        if self.ysolid:
            return scj.solid_harmonics(rij, self.lmax)
        return scj.spherical_harmonics(rij, self.lmax)

    # -------------------------------------------------- many-body
    def site_basis(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
        """Site basis B (n_nodes, n_B) and pooled pair rows (n_nodes, n_pair)."""
        Rnl, Rpair = self.radial(rij, zi, zj)
        Ylm = self.angular(rij)
        edge_A = Rnl[:, self.aspec_r] * Ylm[:, self.aspec_y]
        A = pool_sparse(edge_A, segment_ids, n_nodes, mask)
        AA = jnp.concatenate([jnp.prod(A[:, g], axis=-1) for g in self.aa_specs], axis=-1)
        B = AA @ self.A2B.T
        Apair = pool_sparse(Rpair, segment_ids, n_nodes, mask)
        return B, Apair

    def site_energies(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        """Per-site energies (n_nodes,).  `node_z` is the centre species index per node."""
        B, Apair = self.site_basis(rij, zi, zj, segment_ids, n_nodes, mask)
        e = jnp.einsum("ib,bi->i", B, self.WB[:, node_z])
        e = e + jnp.einsum("ip,pi->i", Apair, self.Wpair[:, node_z])
        return e + self.E0[node_z]

    # -------------------------------------------------- positions wrapper
    def energy_from_positions(self, positions, node_z, senders, receivers,
                              edge_mask=None, shifts=None):
        """lammps-jax-shaped entry point.  Padded edges are placed at the cutoff,
        where the envelope vanishes and the gradient stays defined."""
        n_nodes = positions.shape[0]
        rij = positions[receivers] - positions[senders]
        if shifts is not None:
            rij = rij + shifts
        if edge_mask is not None:
            rcut = float(jnp.max(self.pair_envelope[..., 0]))
            pad = jnp.asarray([1.0, 0.0, 0.0], positions.dtype) * rcut
            rij = jnp.where(edge_mask[:, None], rij, pad)
        zi, zj = node_z[senders], node_z[receivers]
        return self.site_energies(rij, zi, zj, senders, n_nodes, node_z, edge_mask)
