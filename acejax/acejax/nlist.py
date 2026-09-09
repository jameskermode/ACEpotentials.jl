"""Neighbour-list adapters.

Three backends, tried in order:

1. `matscipy_neighbours` -- preferred when present: GPU, DLPack, and a native
   `neighbour_matrix` for the dense layout.  Not on PyPI, so it is optional.
2. `matscipy` -- the hard dependency.  C-accelerated, on PyPI, and advertised
   as the same `"ijdDS"` API, so the two are interchangeable for the sparse path.
3. a pure-numpy fallback -- a genuine last resort, kept so the package still
   works if neither import succeeds.

All three are checked against each other in `tests/test_efv.py`, which asserts
the edge sets are *identical* rather than merely similar.

Whichever backend is used,
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


def have_matscipy_neighbours():
    try:
        import matscipy_neighbours  # noqa: F401
        return True
    except ImportError:
        return False


def have_matscipy():
    try:
        import matscipy.neighbours  # noqa: F401
        return True
    except ImportError:
        return False


def backend():
    """Which neighbour-list backend is in use: for tests and for reporting."""
    if have_matscipy_neighbours():
        return "matscipy-neighbours"
    return "matscipy" if have_matscipy() else "numpy"


def _fallback_neighbour_list(positions, cell, pbc, cutoff):
    """Pure-numpy periodic neighbour list: O(N^2) per image shell.

    matscipy-neighbours is not on PyPI, so it cannot be a hard dependency; this
    keeps `pip install acejax` self-sufficient.  It is correct but quadratic --
    fine to a few thousand atoms, and the tests check the two agree.  Install
    the extra for anything larger:

        pip install git+https://github.com/libAtoms/matscipy-neighbours
    """
    import itertools
    n = len(positions)
    cell = np.asarray(cell, float)
    pbc = np.broadcast_to(pbc, 3)
    widths = np.array([np.linalg.norm(cell[k]) if pbc[k] else np.inf for k in range(3)])
    reps = [0 if not pbc[k] else int(np.ceil(cutoff / max(widths[k], 1e-12)))
            for k in range(3)]
    ii, jj, DD, SS = [], [], [], []
    for sh in itertools.product(*[range(-r, r + 1) for r in reps]):
        shift = np.asarray(sh, float) @ cell
        d = (positions[None, :, :] + shift) - positions[:, None, :]
        r = np.linalg.norm(d, axis=-1)
        keep = (r < cutoff) & (r > 1e-10)
        a, b = np.where(keep)
        if not len(a):
            continue
        ii.append(a); jj.append(b); DD.append(d[a, b])
        SS.append(np.tile(np.asarray(sh, float), (len(a), 1)))
    if not ii:
        z = np.zeros((0,), int)
        return z, z, np.zeros((0, 3)), np.zeros((0, 3))
    ii = np.concatenate(ii); jj = np.concatenate(jj)
    DD = np.concatenate(DD); SS = np.concatenate(SS)
    o = np.argsort(ii, kind="stable")      # sorted by i, as matscipy returns
    return ii[o], jj[o], DD[o], SS[o]


def _neighbour_list(positions, cell, pbc, cutoff, force_backend=None):
    """Sparse edge list from the best available backend.

    `force_backend` is for the cross-backend equivalence test; leave it None.
    """
    which = force_backend or backend()
    if which in ("matscipy-neighbours", "matscipy"):
        mod = ("matscipy_neighbours" if which == "matscipy-neighbours"
               else "matscipy.neighbours")
        neighbour_list = __import__(mod, fromlist=["neighbour_list"]).neighbour_list
        return neighbour_list(
            "ijDS", positions=np.ascontiguousarray(positions, float),
            cell=np.ascontiguousarray(cell, float),
            pbc=tuple(bool(b) for b in np.broadcast_to(pbc, 3)), cutoff=float(cutoff))
    return _fallback_neighbour_list(np.ascontiguousarray(positions, float),
                                    cell, pbc, cutoff)


def _dense_from_sparse(i, j, D, n_nodes, max_neighbours):
    """Group a sparse edge list into the dense (n, K) layout.

    The dense form does not fundamentally need `neighbour_matrix`: every backend
    returns edges sorted by `i`, so the slot of each edge within its centre's row
    is its offset from that group's start.  Building it here means the dense path
    -- and the dense-vs-sparse pooling equivalence check -- works on every
    backend rather than only where `neighbour_matrix` exists.
    """
    counts = np.bincount(i, minlength=n_nodes)
    if counts.max(initial=0) > max_neighbours:
        raise ValueError(
            f"max_neighbours={max_neighbours} too small: an atom has "
            f"{counts.max()} neighbours")
    starts = np.cumsum(counts) - counts          # first edge index of each centre
    slot = np.arange(len(i)) - np.repeat(starts, counts)
    idx = np.zeros((n_nodes, max_neighbours), dtype=np.int64)
    dist = np.zeros((n_nodes, max_neighbours, 3), dtype=float)
    if len(i):
        idx[i, slot] = j
        dist[i, slot] = D
    return idx, dist, counts


def sparse_graph(positions, cell, pbc, cutoff, pad_to=None, pad_vector=None):
    """Build a SparseGraph.  `pad_to` pads the edge list to a fixed capacity,
    which is what jit and the LAMMPS export contract want; padded edges carry
    `pad_vector` (default: at the cutoff, where the envelope vanishes)."""
    i, j, D, S = _neighbour_list(positions, cell, pbc, cutoff)
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
    if have_matscipy_neighbours():
        from matscipy_neighbours import neighbour_matrix
        idx, dist, count = neighbour_matrix(
            positions=np.ascontiguousarray(positions, float),
            cell=np.ascontiguousarray(cell, float),
            pbc=tuple(bool(b) for b in np.broadcast_to(pbc, 3)), cutoff=float(cutoff),
            max_neighbours=int(max_neighbours))
    else:
        i, j, D, _ = _neighbour_list(positions, cell, pbc, cutoff)
        idx, dist, count = _dense_from_sparse(i, j, D, len(positions),
                                              int(max_neighbours))
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
