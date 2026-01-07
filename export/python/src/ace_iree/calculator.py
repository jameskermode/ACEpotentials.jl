"""
ASE Calculator for IREE-compiled ACE potentials

This module provides an ASE-compatible Calculator that uses IREE-compiled
ACE models exported from Julia/ACEpotentials.jl via Reactant.jl.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from matscipy.neighbours import neighbour_list

from .iree_runtime import IREEBackend, load_model


class ACECalculator(Calculator):
    """
    ASE Calculator using IREE-compiled ACE model.

    This calculator uses an ACE model compiled via Reactant.jl to StableHLO
    and then to IREE VMFB format. It provides energy, forces, and stress
    calculations consistent with the Julia implementation.

    The same VMFB file can be used from both Python/ASE and LAMMPS,
    ensuring reproducible results across different simulation codes.

    Example:
        >>> from ace_iree import ACECalculator
        >>> calc = ACECalculator(
        ...     vmfb_path="ace_model_cpu.vmfb",
        ...     constants_path="ace_constants.npz",
        ...     metadata_path="ace_metadata.json"
        ... )
        >>> atoms = bulk("Si", "diamond", a=5.43)
        >>> atoms.calc = calc
        >>> energy = atoms.get_potential_energy()
        >>> forces = atoms.get_forces()
        >>> stress = atoms.get_stress()  # Voigt notation

    Attributes:
        implemented_properties: List of properties this calculator provides
        rcut: Cutoff radius for neighbor list (Angstroms)
        species_Z: Atomic numbers supported by the model
        max_atoms: Maximum atoms supported (padding size)
        max_edges: Maximum edges supported (padding size)
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        vmfb_path: str,
        constants_path: str,
        metadata_path: Optional[str] = None,
        device: str = "local-task",
        prefer_runtime: bool = True,
        **kwargs,
    ):
        """
        Initialize the ACE calculator.

        Args:
            vmfb_path: Path to IREE compiled .vmfb file
            constants_path: Path to model constants .npz file
            metadata_path: Path to model metadata .json file (optional)
                If not provided, looks for ace_metadata.json next to vmfb
            device: IREE device string
                - "local-task" for CPU with threading (default)
                - "local-sync" for single-threaded CPU
                - "cuda" for NVIDIA GPU
                - "metal" for Apple GPU
            prefer_runtime: If True, prefer iree-runtime over subprocess
            **kwargs: Additional arguments passed to ASE Calculator
        """
        super().__init__(**kwargs)

        self.vmfb_path = vmfb_path
        self.constants_path = constants_path
        self.device = device

        # Load metadata
        if metadata_path is None:
            metadata_path = os.path.join(
                os.path.dirname(vmfb_path), "ace_metadata.json"
            )
        self._load_metadata(metadata_path)

        # Load constants
        self._load_constants(constants_path)

        # Load IREE model
        self.model = load_model(vmfb_path, device, prefer_runtime)

    def _load_metadata(self, path: str) -> None:
        """Load model metadata from JSON file."""
        if os.path.exists(path):
            with open(path, "r") as f:
                self.metadata = json.load(f)

            shapes = self.metadata.get("shapes", {})
            self.max_atoms = shapes.get("max_atoms", 4096)
            self.max_neigs = shapes.get("max_neigs", 50)
            self.max_edges = shapes.get("max_edges", 200_000)

            self.rcut = self.metadata.get("rcut", 6.0)
            self.species_Z = self.metadata.get("species_Z", [14])
            self.n_species = self.metadata.get("n_species", 1)
        else:
            # Use defaults
            self.metadata = {}
            self.max_atoms = 4096
            self.max_neigs = 50
            self.max_edges = 200_000
            self.rcut = 6.0
            self.species_Z = [14]
            self.n_species = 1

    def _load_constants(self, path: str) -> None:
        """Load model constants from NPZ file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Constants file not found: {path}")

        data = np.load(path)

        # Extract key parameters
        self.rcut = float(data["rcut"][0])
        self.n_species = int(data["n_species"][0])
        self.species_Z = data["species_Z"].astype(np.int64).tolist()
        self.E0 = data["E0"].astype(np.float32)

        # Build atomic number to species index mapping
        self._z_to_idx = {z: i for i, z in enumerate(self.species_Z)}

    def _build_neighbor_list(
        self, atoms: Atoms
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build neighbor list using matscipy.

        Returns:
            edge_i: Source atom indices [n_edges]
            edge_j: Target atom indices [n_edges]
            edge_rij: Displacement vectors j - i [n_edges, 3]
        """
        edge_i, edge_j, edge_rij = neighbour_list("ijD", atoms, self.rcut)
        return (
            edge_i.astype(np.int64),
            edge_j.astype(np.int64),
            edge_rij.astype(np.float32),
        )

    def _pad_to_shape(
        self, arr: np.ndarray, target_shape: Tuple[int, ...], fill_value: Any = 0
    ) -> np.ndarray:
        """Pad array to target shape with fill value."""
        if arr.shape == target_shape:
            return arr

        result = np.full(target_shape, fill_value, dtype=arr.dtype)

        # Build slices for actual data
        slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
        src_slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))

        result[slices] = arr[src_slices]
        return result

    def _virial_to_stress(self, virial: np.ndarray, volume: float) -> np.ndarray:
        """
        Convert virial tensor to ASE stress (Voigt notation).

        ASE stress convention: stress = -virial / volume
        Voigt order: [xx, yy, zz, yz, xz, xy]

        Args:
            virial: 3x3 virial tensor
            volume: Cell volume in Angstroms^3

        Returns:
            stress: 6-element array in Voigt notation (eV/Angstrom^3)
        """
        # Ensure virial is 3x3
        virial = np.asarray(virial).reshape(3, 3)

        # Convert to stress
        stress_tensor = -virial / volume

        # Convert to Voigt notation
        stress = np.array([
            stress_tensor[0, 0],  # xx
            stress_tensor[1, 1],  # yy
            stress_tensor[2, 2],  # zz
            stress_tensor[1, 2],  # yz
            stress_tensor[0, 2],  # xz
            stress_tensor[0, 1],  # xy
        ], dtype=np.float64)

        return stress

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = ["energy"],
        system_changes: List[str] = all_changes,
    ) -> None:
        """
        Calculate properties for the given atomic configuration.

        This method is called by ASE when getting properties like
        atoms.get_potential_energy(), atoms.get_forces(), etc.

        Args:
            atoms: ASE Atoms object (uses self.atoms if None)
            properties: List of properties to calculate
            system_changes: What changed since last calculation
        """
        super().calculate(atoms, properties, system_changes)

        if self.atoms is None:
            raise RuntimeError("No atoms object set")

        atoms = self.atoms
        n_atoms = len(atoms)

        # Check atom count
        if n_atoms > self.max_atoms:
            raise ValueError(
                f"System has {n_atoms} atoms, but model compiled for max {self.max_atoms}"
            )

        # Check species
        atomic_numbers = atoms.get_atomic_numbers()
        for z in np.unique(atomic_numbers):
            if z not in self._z_to_idx:
                raise ValueError(
                    f"Atomic number {z} not in model species {self.species_Z}"
                )

        # Build neighbor list
        edge_i, edge_j, edge_rij = self._build_neighbor_list(atoms)
        n_edges = len(edge_i)

        if n_edges > self.max_edges:
            raise ValueError(
                f"System has {n_edges} edges, but model compiled for max {self.max_edges}"
            )

        # Handle isolated atoms (no neighbors)
        if n_edges == 0:
            # Only one-body contributions
            onebody_E = sum(self.E0[self._z_to_idx[z]] for z in atomic_numbers)
            self.results["energy"] = float(onebody_E)
            self.results["forces"] = np.zeros((n_atoms, 3))
            if atoms.cell is not None and atoms.cell.volume > 0:
                self.results["stress"] = np.zeros(6)
            return

        # Pad inputs to compiled shapes
        edge_rij_padded = self._pad_to_shape(edge_rij, (self.max_edges, 3))
        atomic_numbers_padded = self._pad_to_shape(
            atomic_numbers.astype(np.int64), (self.max_atoms,)
        )
        edge_i_padded = self._pad_to_shape(edge_i, (self.max_edges,))
        edge_j_padded = self._pad_to_shape(edge_j, (self.max_edges,))

        # Call IREE model
        outputs = self.model(
            edge_rij=edge_rij_padded,
            atomic_numbers=atomic_numbers_padded,
            edge_i=edge_i_padded,
            edge_j=edge_j_padded,
            n_atoms=n_atoms,
            n_edges=n_edges,
        )

        # Extract results
        energy = float(outputs["energy"])
        forces_padded = outputs["forces"]
        virial = outputs["virial"]

        # Store results (unpad forces)
        self.results["energy"] = energy
        self.results["forces"] = forces_padded[:n_atoms].copy()

        # Compute stress from virial
        if atoms.cell is not None and atoms.cell.volume > 0:
            self.results["stress"] = self._virial_to_stress(virial, atoms.cell.volume)


def test_calculator():
    """Basic test of the ACE calculator."""
    from ase.build import bulk

    print("=" * 60)
    print("Testing ACECalculator")
    print("=" * 60)

    # Check if test files exist
    test_dir = os.path.dirname(__file__)
    vmfb_path = os.path.join(test_dir, "../../test_models/ace_model_cpu.vmfb")
    constants_path = os.path.join(test_dir, "../../test_models/ace_constants.npz")

    if not os.path.exists(vmfb_path):
        print(f"[SKIP] Test VMFB not found at {vmfb_path}")
        print("       Run Julia export first to generate test models")
        return False

    try:
        calc = ACECalculator(vmfb_path, constants_path)
        print(f"[OK] Calculator loaded")
        print(f"     rcut = {calc.rcut}")
        print(f"     species = {calc.species_Z}")
        print(f"     max_atoms = {calc.max_atoms}")
    except Exception as e:
        print(f"[FAIL] Calculator creation failed: {e}")
        return False

    # Create test structure
    atoms = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
    atoms.calc = calc

    print(f"\n[OK] Created Si diamond structure with {len(atoms)} atoms")

    try:
        energy = atoms.get_potential_energy()
        print(f"[OK] Energy: {energy:.6f} eV")
    except Exception as e:
        print(f"[FAIL] Energy calculation failed: {e}")
        return False

    try:
        forces = atoms.get_forces()
        print(f"[OK] Forces shape: {forces.shape}")
        print(f"     Max force: {np.abs(forces).max():.6f} eV/A")
    except Exception as e:
        print(f"[FAIL] Forces calculation failed: {e}")
        return False

    try:
        stress = atoms.get_stress()
        print(f"[OK] Stress (Voigt): {stress}")
    except Exception as e:
        print(f"[FAIL] Stress calculation failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_calculator()
