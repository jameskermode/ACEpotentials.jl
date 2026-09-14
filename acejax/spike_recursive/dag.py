"""THROWAWAY SPIKE -- not product code, not covered by tests, do not import from acejax/.

Recursive (DAG) evaluator for the AA products, ported from
EquivariantTensors/src/ace/symmprod_dag.jl (`SparseSymmProdDAG`).

The flat evaluator computes each AA term as an independent product over a
gathered index row, so terms sharing subproducts recompute them.  The DAG makes
every AA term the product of exactly two earlier nodes, inserting auxiliary
nodes where a needed subproduct is not itself in the spec.  Depth is small
(~log2 of the correlation order plus the aux chain), and within a level all
nodes are independent -- which is the property that might make it XLA-friendly.

Construction follows the Julia greedy exactly:
  score(partition) = 1e9 * len(partition) + max(node index)
so fewest parts wins, ties break to the earliest-created nodes.
"""
import numpy as np


# ---------------------------------------------------------------- set partitions
def _set_partitions(n):
    """All set partitions of range(n), each as a tuple of sorted tuples."""
    if n == 0:
        yield ()
        return
    for rest in _set_partitions(n - 1):
        # place element n-1 into an existing block, or into a new one
        for i in range(len(rest)):
            yield rest[:i] + (rest[i] + (n - 1,),) + rest[i + 1:]
        yield rest + ((n - 1,),)


_PARTS = {r: list(_set_partitions(r)) for r in range(1, 7)}


# ---------------------------------------------------------------- construction
def build_dag(spec, num1):
    """spec: list of sorted tuples of A-indices (0-based), ordered by rank.
       num1: number of A basis functions (nodes 0..num1-1 are A itself).

    Returns (left, right, projection, n_extra):
      left/right : int arrays, one per internal node, indexing into
                   [A columns] ++ [internal nodes] (offset num1).
      projection : for each spec entry, its position in that same index space.
    """
    left, right = [], []
    specnew = [(i,) for i in range(num1)]        # node i -> its A-index tuple
    lut = {(i,): i for i in range(num1)}
    n_extra = 0

    def _emit(p_nodes, kk):
        """Reduce a partition (list of existing node indices) down to one node."""
        nonlocal n_extra
        while len(p_nodes) > 2:
            a, b = p_nodes[0], p_nodes[1]
            left.append(a); right.append(b)
            new = num1 + len(left) - 1
            kk1 = tuple(sorted(specnew[a] + specnew[b]))
            specnew.append(kk1); lut.setdefault(kk1, new)
            p_nodes = [new] + p_nodes[2:]
            n_extra += 1
        left.append(p_nodes[0]); right.append(p_nodes[1])
        new = num1 + len(left) - 1
        specnew.append(kk); lut[kk] = new
        return new

    projection = []
    for kk in spec:
        if len(kk) == 1:
            projection.append(kk[0])
            continue
        if kk in lut:                        # already built as an aux node
            projection.append(lut[kk])
            continue
        best, best_score = None, np.inf
        for part in _PARTS[len(kk)]:
            nodes = []
            for blk in part:
                key = tuple(kk[i] for i in blk)
                j = lut.get(key)
                if j is None:
                    nodes = None
                    break
                nodes.append(j)
            if nodes is None:
                continue
            score = 1e9 * len(nodes) + max(nodes)
            if score < best_score:
                best, best_score = nodes, score
        projection.append(_emit(best, kk))

    return (np.asarray(left, np.int32), np.asarray(right, np.int32),
            np.asarray(projection, np.int32), n_extra)


# ---------------------------------------------------------------- levelling
def level_dag(left, right, num1):
    """Assign each internal node a level (1 + max level of its parents; A is 0)
    and reorder nodes so that each level is a contiguous block.

    Returns (levels, perm_pos, n_total) where
      levels   : list of (start, L, R) -- write block [start, start+len(L)),
                 reading columns L and R of the same buffer.
      perm_pos : old node index (in the [A ++ nodes] space) -> new position.
    """
    n = len(left)
    lvl = np.zeros(num1 + n, np.int32)
    for i in range(n):
        lvl[num1 + i] = 1 + max(lvl[left[i]], lvl[right[i]])
    order = np.argsort(lvl[num1:], kind="stable")          # by level, then insertion
    perm_pos = np.empty(num1 + n, np.int64)
    perm_pos[:num1] = np.arange(num1)
    perm_pos[num1 + order] = num1 + np.arange(n)
    L = perm_pos[left[order]]
    R = perm_pos[right[order]]
    lv = lvl[num1 + order]
    levels, start = [], num1
    for k in range(1, lv.max() + 1):
        sel = np.flatnonzero(lv == k)
        levels.append((start, L[sel].astype(np.int32), R[sel].astype(np.int32)))
        start += len(sel)
    return levels, perm_pos, num1 + n


# ---------------------------------------------------------------- balanced variant
def build_dag_balanced(spec, num1):
    """Depth-minimising alternative to the Julia greedy.

    The Julia score prefers the fewest parts and then the *lowest* node index,
    which biases towards long reuse chains: on the order-4 model it gives depth
    3 with a fat top level.  This one always splits a term into two halves of
    (nearly) equal size, creating whichever half is missing, so depth is
    ceil(log2(rank)) = 2 at order 4.  Among the balanced splits it prefers the
    one whose halves already exist, then the lowest index -- so reuse is still
    the tie-break, it is just no longer allowed to cost depth.
    """
    from itertools import combinations
    left, right = [], []
    specnew = [(i,) for i in range(num1)]
    lut = {(i,): i for i in range(num1)}
    n_extra = [0]

    def ensure(kk):
        j = lut.get(kk)
        if j is not None:
            return j
        r = len(kk)
        h = r // 2
        best, best_score = None, None
        seen = set()
        for blk in combinations(range(r), h):
            a = tuple(kk[i] for i in blk)
            b = tuple(kk[i] for i in range(r) if i not in blk)
            if (a, b) in seen or (b, a) in seen:
                continue
            seen.add((a, b))
            ja, jb = lut.get(a), lut.get(b)
            missing = (ja is None) + (jb is None)
            score = (missing, max(ja or 0, jb or 0))
            if best_score is None or score < best_score:
                best, best_score = (a, b), score
        ja, jb = ensure(best[0]), ensure(best[1])
        if lut.get(best[0]) is None or lut.get(best[1]) is None:
            pass
        left.append(ja); right.append(jb)
        j = num1 + len(left) - 1
        specnew.append(kk); lut[kk] = j
        return j

    projection = []
    for kk in spec:
        if len(kk) == 1:
            projection.append(kk[0])
        else:
            before = len(left)
            projection.append(ensure(kk))
            n_extra[0] += max(0, len(left) - before - 1)
    return (np.asarray(left, np.int32), np.asarray(right, np.int32),
            np.asarray(projection, np.int32), n_extra[0])


def dag_from_model(aa_specs, n_A, mode="julia"):
    spec = [tuple(int(v) for v in row) for g in aa_specs for row in np.asarray(g)]
    builder = _BUILDERS[mode]
    left, right, projection, n_extra = builder(spec, n_A)
    levels, perm_pos, n_total = level_dag(left, right, n_A)
    return dict(levels=levels, projection=perm_pos[projection].astype(np.int32),
                n_total=n_total, n_extra=n_extra, n_nodes_int=len(left),
                depth=len(levels), level_sizes=[len(l[1]) for l in levels])


# ---------------------------------------------------------------- chain variant
def build_dag_chain(spec, num1):
    """Prefix-chain DAG: kk = ensure(kk[:-1]) * kk[-1].

    Depth is rank-1, so this is the *worst* shape for a levelled GPU evaluator --
    but it associates the product exactly the way `jnp.prod` over a gathered row
    does, left to right, so its output is BITWISE identical to the flat
    evaluator.  That makes it the correctness gate for all the index machinery
    (nodes, levels, projection, and the projection folded into the A2B gather);
    the julia/balanced modes then differ from flat only by re-association, which
    is worth exactly the 1-2 ulp they show.
    """
    left, right = [], []
    lut = {(i,): i for i in range(num1)}
    n_extra = [0]

    def ensure(kk):
        j = lut.get(kk)
        if j is not None:
            return j
        a = ensure(kk[:-1])
        left.append(a); right.append(kk[-1])
        j = num1 + len(left) - 1
        lut[kk] = j
        return j

    projection = []
    for kk in spec:
        if len(kk) == 1:
            projection.append(kk[0])
        else:
            before = len(left)
            projection.append(ensure(kk))
            n_extra[0] += max(0, len(left) - before - 1)
    return (np.asarray(left, np.int32), np.asarray(right, np.int32),
            np.asarray(projection, np.int32), n_extra[0])


_BUILDERS = {"julia": build_dag, "balanced": build_dag_balanced, "chain": build_dag_chain}


def reconstruct_spec(left, right, num1):
    """Multiset of A-indices each node computes -- the exact, integer check that
    the DAG evaluates the products the spec asks for."""
    out = [(i,) for i in range(num1)]
    for a, b in zip(left, right):
        out.append(tuple(sorted(out[a] + out[b])))
    return out
