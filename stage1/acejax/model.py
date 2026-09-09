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

from .harmonics import real_spherical_harmonics
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
        # ace1_model uses SPHERICAL harmonics (ace1_compat.jl:407, Ytype=:spherical);
        # ace_model defaults to :solid.  The Phase 0 spike used ace_model, so the
        # production path differs from it here -- hence the exported flag.
        #
        # Pure JAX, not sphericart-jax: the latter lowers to an FFI custom call,
        # which the LAMMPS bundle would then have to resolve at run time.
        if self.ysolid:
            raise NotImplementedError(
                "solid harmonics are not implemented: ace1_model uses "
                "Ytype=:spherical.  ace_model's :solid default would need the "
                "r^l scaling and its own normalisation.")
        return real_spherical_harmonics(rij, self.lmax)

    # -------------------------------------------------- many-body
    def edge_features(self, rij, zi, zj):
        """Per-edge (A-basis rows, pair rows).  Layout-agnostic: the caller
        pools these however its neighbour-list layout dictates."""
        Rnl, Rpair = self.radial(rij, zi, zj)
        Ylm = self.angular(rij)
        return Rnl[:, self.aspec_r] * Ylm[:, self.aspec_y], Rpair

    def _from_pooled(self, A, Apair):
        AA = jnp.concatenate([jnp.prod(A[:, g], axis=-1) for g in self.aa_specs], axis=-1)
        return AA @ self.A2B.T, Apair

    def site_basis(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
        """Sparse (edge-list) pooling -- the layout lammps-jax exports."""
        edge_A, Rpair = self.edge_features(rij, zi, zj)
        return self._from_pooled(pool_sparse(edge_A, segment_ids, n_nodes, mask),
                                 pool_sparse(Rpair, segment_ids, n_nodes, mask))

    def site_basis_dense(self, rij, zi, zj, mask):
        """Dense (n, K) pooling -- matscipy-neighbours' `neighbour_matrix` form,
        which maps onto ET's own (maxneigs, nnodes, nfeat) layout and needs no
        scatter.  rij (n,K,3), zi/zj (n,K), mask (n,K)."""
        n, K = mask.shape
        flat = lambda a: a.reshape(n * K, *a.shape[2:])
        edge_A, Rpair = self.edge_features(flat(rij), flat(zi), flat(zj))
        un = lambda a: a.reshape(n, K, -1)
        return self._from_pooled(pool_dense(un(edge_A), mask), pool_dense(un(Rpair), mask))

    def _readout(self, B, Apair, node_z):
        e = jnp.einsum("ib,bi->i", B, self.WB[:, node_z])
        e = e + jnp.einsum("ip,pi->i", Apair, self.Wpair[:, node_z])
        return e + self.E0[node_z]

    def site_energies(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        """Per-site energies (n_nodes,).  `node_z` is the centre species index per node."""
        return self._readout(*self.site_basis(rij, zi, zj, segment_ids, n_nodes, mask), node_z)

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        return self._readout(*self.site_basis_dense(rij, zi, zj, mask), node_z)

    # -------------------------------------------------- energy / forces / virial
    def energy_forces_virial(self, rij, zi, zj, senders, receivers, n_nodes,
                             node_z, mask=None):
        """E, F, V from edge vectors alone -- no cell needed.

        Virial by the symmetric-displacement trick, after mace-jax
        `modules/utils.py::compute_forces_and_stress` (MIT).  There the strain is
        applied to positions *and* cell, hence to the edge shifts; because
        rij = r[j] - r[i] + S@cell, both halves transform the same way and the
        whole thing collapses to rij -> rij + rij @ eps.  That keeps the virial a
        function of the edge vectors, consistent with the core.

        Sign follows Julia (AtomsCalculatorsUtilities sitepotentials/assembly.jl:6,
        `site_virial = -sum(dv_i * r_i')`), i.e. V = -dE/d(eps); verified against
        the exported reference rather than argued from the algebra.
        """
        def total(r, eps):
            sym = 0.5 * (eps + eps.T)
            return jnp.sum(self.site_energies(r + r @ sym, zi, zj, senders,
                                              n_nodes, node_z, mask))

        eps0 = jnp.zeros((3, 3), rij.dtype)
        E, (g_r, g_eps) = jax.value_and_grad(total, argnums=(0, 1))(rij, eps0)
        F = (jnp.zeros((n_nodes, 3), rij.dtype)
             .at[senders].add(g_r).at[receivers].add(-g_r))
        return E, F, -g_eps

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
