"""
Common utilities for ACE benchmark suite.

Provides:
- Fixture loading
- Timing utilities
- Result comparison
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np


@dataclass
class Fixture:
    """Reference data from Julia for benchmarking."""
    name: str
    n_atoms: int
    n_edges: int

    # Inputs
    rij: np.ndarray         # (n_edges, 3) displacement vectors
    ii: np.ndarray          # (n_edges,) source atom indices
    jj: np.ndarray          # (n_edges,) target atom indices
    zi: np.ndarray          # (n_edges,) source species indices
    zj: np.ndarray          # (n_edges,) target species indices
    positions: np.ndarray   # (n_atoms, 3) atomic positions
    species: np.ndarray     # (n_atoms,) species indices

    # Reference outputs
    energy: float
    forces: np.ndarray      # (n_atoms, 3)
    virial: np.ndarray      # (3, 3)


def load_fixture(path: str) -> Fixture:
    """Load a fixture from NPZ file."""
    data = np.load(path)
    name = Path(path).stem

    return Fixture(
        name=name,
        n_atoms=int(data['n_atoms'][0]),
        n_edges=int(data['n_edges'][0]),
        rij=data['rij'].astype(np.float32),
        ii=data['ii'].astype(np.int32),
        jj=data['jj'].astype(np.int32),
        zi=data['zi'].astype(np.int32),
        zj=data['zj'].astype(np.int32),
        positions=data['positions'].astype(np.float32),
        species=data['species'].astype(np.int32),
        energy=float(data['energy'][0]),
        forces=data['forces'].astype(np.float32),
        virial=data['virial'].astype(np.float32),
    )


def load_all_fixtures(fixtures_dir: str = None) -> Dict[str, Fixture]:
    """Load all fixtures from directory."""
    if fixtures_dir is None:
        fixtures_dir = Path(__file__).parent / "fixtures"
    else:
        fixtures_dir = Path(fixtures_dir)

    fixtures = {}
    for npz_path in fixtures_dir.glob("*.npz"):
        try:
            fixtures[npz_path.stem] = load_fixture(str(npz_path))
        except Exception as e:
            print(f"Warning: Failed to load {npz_path}: {e}")

    return fixtures


@dataclass
class TimingResult:
    """Timing results from a benchmark run."""
    name: str
    n_atoms: int
    n_edges: int

    # Timings (milliseconds)
    total_ms: float
    forward_ms: float
    backward_ms: float
    scatter_ms: float

    # Breakdown
    setup_ms: float = 0.0
    jit_compile_ms: float = 0.0

    # Accuracy
    energy_error: float = 0.0
    force_rmse: float = 0.0
    force_max_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'n_atoms': self.n_atoms,
            'n_edges': self.n_edges,
            'total_ms': self.total_ms,
            'forward_ms': self.forward_ms,
            'backward_ms': self.backward_ms,
            'scatter_ms': self.scatter_ms,
            'setup_ms': self.setup_ms,
            'jit_compile_ms': self.jit_compile_ms,
            'energy_error': self.energy_error,
            'force_rmse': self.force_rmse,
            'force_max_error': self.force_max_error,
        }


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000


def compare_forces(
    computed: np.ndarray,
    reference: np.ndarray,
) -> Tuple[float, float]:
    """
    Compare computed forces to reference.

    Returns:
        (rmse, max_error) in same units as input
    """
    diff = computed - reference
    rmse = np.sqrt(np.mean(diff ** 2))
    max_error = np.max(np.abs(diff))
    return rmse, max_error


def print_timing_table(results: list[TimingResult]):
    """Print timing results as formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Name':<20} {'Atoms':>8} {'Edges':>8} {'Total':>10} {'Fwd':>8} {'Bwd':>8} {'RMSE':>10}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.n_atoms):
        print(f"{r.name:<20} {r.n_atoms:>8} {r.n_edges:>8} "
              f"{r.total_ms:>10.2f} {r.forward_ms:>8.2f} {r.backward_ms:>8.2f} "
              f"{r.force_rmse:>10.2e}")

    print("=" * 80)


def save_results(results: list[TimingResult], output_path: str):
    """Save results to JSON file."""
    data = {
        'benchmark': 'ace_forces',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': [r.to_dict() for r in results],
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {output_path}")
