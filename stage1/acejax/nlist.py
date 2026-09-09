"""Neighbour-list adapters.

Primary backend is `libAtoms/matscipy-neighbours` (MIT, repo-only at v0.1.0):
OpenMP CPU core, optional CUDA/HIP, zero-copy to JAX via DLPack.  Three of its
properties are what make it the right fit here:

* it returns ``D == r[j] - r[i] + S @ cell`` directly -- the edge vector the
  descriptor core wants, with no separate shift bookkeeping;
* pairs come back **sorted by i**, so ``segment_sum`` can run with
  ``indices_are_sorted=True`` (and ET's ``ETGraph`` requires the same ordering);
* ``neighbour_matrix`` gives the dense ``(n, K)`` + ``count`` form, which maps
  onto ET's own ``(maxneigs, nnodes, nfeat)`` layout.

Both layouts are produced here and both feed the same core, because lammps-jax
is sparse while the dense form avoids a scatter.  Neither is hard-wired.
"""

from typing import NamedTuple

import numpy as np


class SparseGraph(NamedTuple):
    """Edge-list layout.  This is the shape lammps-jax exports."""
    rij: np.ndarray          # (E, 3)  = r[j] - r[i] + S @ cell
    senders: np.ndarray      # (E,)    centre atom, sorted ascending
    receivers: np.ndarray    # (E,)    neighbour atom
    shifts: np.ndarray       # (E, 3)  S @ cell, needed for the strain derivative
    n_nodes: int
    mask: np.ndarray = None  # (E,) or None when unpadded


class DenseGraph(NamedTuple):
    """Fixed-capacity (n, K) layout; maps onto ET's (maxneigs, nnodes, ...)."""
    rij: np.ndarray          # (n, K, 3)
    idx: np.ndarray          # (n, K)   neighbour index
    count: np.ndarray        # (n,)     true neighbour count
    n_nodes: int

    @property
    def mask(self):
        return np.arange(self.rij.shape[1])[None, :] < self.count[:, None]


def _require():
    try:
        import matscipy_neighbours  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matscipy-neighbours is required. It is repo-only at v0.1.0:\n"
            "  uv add 'matscipy-neighbours @ git+https://github.com/libAtoms/matscipy-neighbours'\n"
            "GPU is a separate documented build (-Dcmake.define.ENABLE_CUDA=ON)."
        ) from e


def sparse_graph(positions, cell, pbc, cutoff, pad_to=None, pad_vector=None):
    """Build a SparseGraph.  `pad_to` pads the edge list to a fixed capacity,
    which is what jit and the LAMMPS export contract want; padded edges carry
    `pad_vector` (default: at the cutoff, where the envelope vanishes)."""
    _require()
    from matscipy_neighbours import neighbour_list
    i, j, D, S = neighbour_list("ijDS", positions=np.ascontiguousarray(positions, float),
                                cell=np.ascontiguousarray(cell, float),
                                pbc=tuple(bool(b) for b in pbc), cutoff=float(cutoff))
    shifts = D - (positions[j] - positions[i])
    n = len(positions)
    if pad_to is None:
        return SparseGraph(D, i.astype(np.int32), j.astype(np.int32), shifts, n, None)
    if len(i) > pad_to:
        raise ValueError(f"{len(i)} edges exceeds capacity {pad_to}")
    npad = pad_to - len(i)
    if pad_vector is None:
        pad_vector = np.array([cutoff, 0.0, 0.0])
    return SparseGraph(
        rij=np.concatenate([D, np.tile(pad_vector, (npad, 1))]),
        senders=np.concatenate([i, np.zeros(npad, int)]).astype(np.int32),
        receivers=np.concatenate([j, np.zeros(npad, int)]).astype(np.int32),
        shifts=np.concatenate([shifts, np.zeros((npad, 3))]),
        n_nodes=n,
        mask=np.concatenate([np.ones(len(i), bool), np.zeros(npad, bool)]))


def dense_graph(positions, cell, pbc, cutoff, max_neighbours, pad_vector=None):
    """Build a DenseGraph via `neighbour_matrix` -- no scatter needed downstream.

    `neighbour_matrix` leaves unused slots as ZERO vectors.  Feeding those to the
    model NaNs the gradient (r = |rij| is not differentiable at 0 and the Agnesi
    transform divides), exactly as tests/test_padding.py records for the sparse
    layout.  So the padded slots are parked at the cutoff here, where the
    envelope vanishes and the derivative stays defined -- the caller cannot get
    this wrong by forgetting.
    """
    _require()
    from matscipy_neighbours import neighbour_matrix
    idx, dist, count = neighbour_matrix(
        positions=np.ascontiguousarray(positions, float),
        cell=np.ascontiguousarray(cell, float),
        pbc=tuple(bool(b) for b in pbc), cutoff=float(cutoff),
        max_neighbours=int(max_neighbours))
    if pad_vector is None:
        pad_vector = np.array([float(cutoff), 0.0, 0.0])
    live = np.arange(dist.shape[1])[None, :] < count[:, None]
    dist = np.where(live[..., None], dist, pad_vector)
    return DenseGraph(dist, idx, count, len(positions))


def dense_to_sparse(g: DenseGraph):
    """Flatten a DenseGraph, for cross-checking the two pooling paths."""
    m = g.mask
    n, K = g.idx.shape
    senders = np.repeat(np.arange(n), K).reshape(n, K)[m].astype(np.int32)
    return SparseGraph(g.rij[m], senders, g.idx[m].astype(np.int32),
                       np.zeros((int(m.sum()), 3)), g.n_nodes, None)
