from .model import ACEModel, highest_precision, pool_dense, pool_sparse
from .io import load
from .nlist import DenseGraph, SparseGraph, dense_graph, dense_to_sparse, sparse_graph

__all__ = ["ACEModel", "load", "highest_precision", "pool_sparse", "pool_dense",
           "SparseGraph", "DenseGraph", "sparse_graph", "dense_graph",
           "dense_to_sparse", "ACECalculator"]


def __getattr__(name):
    # ase is only needed for the calculator; keep it out of the core import path
    if name == "ACECalculator":
        from .calculator import ACECalculator
        return ACECalculator
    raise AttributeError(name)
