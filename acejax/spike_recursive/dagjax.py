"""THROWAWAY SPIKE -- JAX evaluators for the levelled AA DAG.

Three forward formulations, plus a hand-written backward.

  dus     preallocated (n, n_total) buffer + dynamic_update_slice per level
  concat  grow the buffer by concatenation per level
  cvjp    `concat` forward with a custom_vjp whose backward walks the levels in
          reverse and accumulates with a *sorted* scatter-add

`cvjp` exists because the autodiff of the DAG is the part XLA handles worst:
every level's gather becomes an unsorted scatter-add into a wide buffer on the
way back.  Sorting the target columns once at build time is the obvious fix and
is what any production implementation would do, so it has to be measured before
the DAG can be called a loss.
"""
import jax
import jax.numpy as jnp
import numpy as np


def prep(levels):
    return [(s, jnp.asarray(L), jnp.asarray(R)) for s, L, R in levels]


def make_dus(levels, n_total, n_A):
    lv = prep(levels)
    def f(A):
        v = jnp.zeros((A.shape[0], n_total), A.dtype).at[:, :n_A].set(A)
        for s, L, R in lv:
            v = jax.lax.dynamic_update_slice(v, v[:, L] * v[:, R], (0, s))
        return v
    return f


def make_concat(levels, n_total, n_A):
    lv = prep(levels)
    def f(A):
        v = A
        for s, L, R in lv:
            v = jnp.concatenate([v, v[:, L] * v[:, R]], axis=1)
        return v
    return f


def make_cvjp(levels, n_total, n_A):
    """concat forward, hand-written reverse-level backward with sorted scatter."""
    lv = prep(levels)
    # per level: targets = [L, R] concatenated, sorted once so the backward
    # scatter-add gets indices_are_sorted=True
    info = []
    for s, L, R in levels:
        tgt = np.concatenate([np.asarray(L), np.asarray(R)])
        perm = np.argsort(tgt, kind="stable")
        info.append((s, jnp.asarray(L), jnp.asarray(R),
                     jnp.asarray(tgt[perm]), jnp.asarray(perm), len(L)))

    fwd_raw = make_concat(levels, n_total, n_A)

    @jax.custom_vjp
    def f(A):
        return fwd_raw(A)

    def f_fwd(A):
        v = fwd_raw(A)
        return v, v

    def f_bwd(v, g):
        for s, L, R, tgt, perm, m in reversed(info):
            gk = jax.lax.dynamic_slice(g, (0, s), (g.shape[0], m))
            contrib = jnp.concatenate([gk * v[:, R], gk * v[:, L]], axis=1)[:, perm]
            g = g.at[:, tgt].add(contrib, indices_are_sorted=True, unique_indices=False)
        return (jax.lax.dynamic_slice(g, (0, 0), (g.shape[0], n_A)),)

    f.defvjp(f_fwd, f_bwd)
    return f


MAKERS = {"dus": make_dus, "concat": make_concat, "cvjp": make_cvjp}
