"""
IREE Runtime Wrapper

Provides two backends for running IREE-compiled models:
1. IREEModel: Uses iree-runtime Python library (preferred, faster)
2. SubprocessRunner: Falls back to iree-run-module CLI (no version issues)
"""

import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


class IREEBackend(ABC):
    """Abstract base class for IREE execution backends."""

    @abstractmethod
    def __call__(self, **kwargs) -> Dict[str, np.ndarray]:
        """Execute the model with named inputs."""
        pass

    @abstractmethod
    def get_input_info(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        """Return expected input shapes and dtypes."""
        pass


class IREEModel(IREEBackend):
    """
    Run IREE models via iree-runtime Python library.

    This is the preferred backend when iree-runtime is available.
    Provides better performance and error handling than subprocess.

    Example:
        model = IREEModel("ace_model_cpu.vmfb", device="local-task")
        outputs = model(
            edge_rij=edge_rij_arr,
            atomic_numbers=z_arr,
            edge_i=edge_i_arr,
            edge_j=edge_j_arr,
            n_atoms=n_atoms,
            n_edges=n_edges
        )
        energy = outputs["energy"]
        forces = outputs["forces"]
    """

    def __init__(self, vmfb_path: str, device: str = "local-task"):
        """
        Initialize IREE model.

        Args:
            vmfb_path: Path to compiled .vmfb file
            device: IREE device string
                - "local-task" for CPU (default)
                - "local-sync" for single-threaded CPU
                - "cuda" for NVIDIA GPU
                - "vulkan" for Vulkan GPU
        """
        self.vmfb_path = vmfb_path
        self.device_str = device

        try:
            import iree.runtime as rt
            self._rt = rt
        except ImportError as e:
            raise ImportError(
                "iree-runtime not installed. Install with: pip install iree-runtime\n"
                "Or use SubprocessRunner as fallback."
            ) from e

        # Load the module using the correct API
        config = rt.Config(device)  # Pass driver name directly
        self._device = config.device
        with open(vmfb_path, "rb") as f:
            vmfb_data = f.read()
        self._module = rt.VmModule.from_flatbuffer(
            config.vm_instance, vmfb_data, warn_if_copy=False
        )
        self._context = rt.SystemContext(vm_modules=[self._module], config=config)

        # Get the main function
        self._main_fn = self._context.modules.module["ace_efv"]

    def get_input_info(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        """Return expected input shapes and dtypes from metadata."""
        # Note: Would need to parse from metadata.json
        # For now return empty - caller should use metadata
        return {}

    def __call__(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Execute the EFV function.

        Expected inputs (from compiled model):
            edge_rij: [max_edges, 3] float32 - edge displacement vectors
            atomic_numbers: [max_atoms] int64 - atomic numbers (0=padding)
            edge_i: [max_edges] int64 - source atom indices
            edge_j: [max_edges] int64 - target atom indices
            n_atoms: int32 - actual number of atoms
            n_edges: int32 - actual number of edges

        Returns:
            Dict with:
                energy: scalar float32
                forces: [max_atoms, 3] float32
                virial: [3, 3] float32
        """
        # Convert inputs to IREE device arrays
        rt = self._rt

        # Build input list in correct order
        edge_rij = np.ascontiguousarray(kwargs["edge_rij"], dtype=np.float32)
        atomic_numbers = np.ascontiguousarray(kwargs["atomic_numbers"], dtype=np.int64)
        edge_i = np.ascontiguousarray(kwargs["edge_i"], dtype=np.int64)
        edge_j = np.ascontiguousarray(kwargs["edge_j"], dtype=np.int64)
        n_atoms = np.int32(kwargs["n_atoms"])
        n_edges = np.int32(kwargs["n_edges"])

        # Call the function
        results = self._main_fn(
            edge_rij, atomic_numbers, edge_i, edge_j, n_atoms, n_edges
        )

        # Parse outputs (energy, forces, virial)
        return {
            "energy": np.asarray(results[0]),
            "forces": np.asarray(results[1]),
            "virial": np.asarray(results[2]),
        }


class SubprocessRunner(IREEBackend):
    """
    Run IREE models via subprocess using iree-run-module CLI.

    This is a fallback when iree-runtime Python library has version
    mismatches with the compiled VMFB files.

    Example:
        runner = SubprocessRunner("ace_model_cpu.vmfb")
        outputs = runner(
            edge_rij=edge_rij_arr,
            atomic_numbers=z_arr,
            ...
        )
    """

    def __init__(
        self,
        vmfb_path: str,
        device: str = "local-task",
        iree_bin: Optional[str] = None,
    ):
        """
        Initialize subprocess runner.

        Args:
            vmfb_path: Path to compiled .vmfb file
            device: IREE device string
            iree_bin: Path to IREE bin directory (defaults to ~/iree/bin)
        """
        self.vmfb_path = vmfb_path
        self.device = device

        if iree_bin is None:
            iree_bin = os.path.expanduser("~/iree/bin")

        self.iree_run = os.path.join(iree_bin, "iree-run-module")

        if not os.path.exists(self.iree_run):
            # Try system PATH
            from shutil import which
            system_iree = which("iree-run-module")
            if system_iree:
                self.iree_run = system_iree
            else:
                raise FileNotFoundError(
                    f"iree-run-module not found at {self.iree_run} or in PATH"
                )

        # Create temp directory for input files
        self._tmpdir = tempfile.mkdtemp(prefix="ace_iree_")

    def get_input_info(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        return {}

    def _dtype_to_iree(self, dtype: np.dtype) -> str:
        """Convert numpy dtype to IREE type string."""
        mapping = {
            np.float32: "f32",
            np.float64: "f64",
            np.int32: "i32",
            np.int64: "i64",
        }
        return mapping.get(dtype.type, "f32")

    def _save_input(self, arr: np.ndarray, name: str) -> str:
        """Save array to binary file and return IREE input spec."""
        arr = np.ascontiguousarray(arr)
        path = os.path.join(self._tmpdir, f"{name}.bin")
        arr.tofile(path)

        shape_str = "x".join(str(d) for d in arr.shape) if arr.shape else ""
        dtype_str = self._dtype_to_iree(arr.dtype)

        if shape_str:
            return f"--input={shape_str}x{dtype_str}=@{path}"
        else:
            # Scalar
            return f"--input={dtype_str}=@{path}"

    def _parse_output(self, stdout: str) -> List[np.ndarray]:
        """Parse IREE text output to numpy arrays."""
        outputs = []
        lines = stdout.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("EXEC") or line.startswith("result"):
                continue

            # Parse shape and data: "3x3xf32=[[1,2,3],[4,5,6],[7,8,9]]"
            if "=" in line:
                spec, data = line.split("=", 1)

                # Determine dtype
                if "f32" in spec:
                    dtype = np.float32
                elif "f64" in spec:
                    dtype = np.float64
                elif "i32" in spec:
                    dtype = np.int32
                elif "i64" in spec:
                    dtype = np.int64
                else:
                    dtype = np.float32

                # Parse shape
                parts = spec.replace("x", " ").split()
                shape = []
                for p in parts:
                    try:
                        shape.append(int(p))
                    except ValueError:
                        pass  # Skip dtype string

                # Parse data
                try:
                    if data.startswith("["):
                        arr = np.array(eval(data), dtype=dtype)
                    else:
                        arr = np.array(float(data), dtype=dtype)
                    outputs.append(arr)
                except (SyntaxError, ValueError) as e:
                    print(f"Warning: Could not parse output line: {line}")
                    continue

        return outputs

    def __call__(self, **kwargs) -> Dict[str, np.ndarray]:
        """Execute the model via subprocess."""
        # Build input arguments
        input_args = []

        # Order matters - match the compiled function signature
        input_order = [
            ("edge_rij", np.float32),
            ("atomic_numbers", np.int64),
            ("edge_i", np.int64),
            ("edge_j", np.int64),
            ("n_atoms", np.int32),
            ("n_edges", np.int32),
        ]

        for name, dtype in input_order:
            arr = np.asarray(kwargs[name], dtype=dtype)
            input_args.append(self._save_input(arr, name))

        # Run iree-run-module
        cmd = [
            self.iree_run,
            f"--device={self.device}",
            f"--module={self.vmfb_path}",
            "--function=ace_efv",
        ] + input_args

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"IREE execution failed:\n{result.stderr}")

        # Parse outputs
        outputs = self._parse_output(result.stdout)

        if len(outputs) < 3:
            raise RuntimeError(
                f"Expected 3 outputs (energy, forces, virial), got {len(outputs)}"
            )

        return {
            "energy": outputs[0],
            "forces": outputs[1],
            "virial": outputs[2],
        }

    def __del__(self):
        """Clean up temp directory."""
        import shutil
        if hasattr(self, "_tmpdir") and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)


def load_model(
    vmfb_path: str,
    device: str = "local-task",
    prefer_runtime: bool = True,
    iree_bin: Optional[str] = None,
) -> IREEBackend:
    """
    Load an IREE model, automatically choosing the best backend.

    Args:
        vmfb_path: Path to compiled .vmfb file
        device: IREE device string
        prefer_runtime: If True, prefer iree-runtime over subprocess
        iree_bin: Path to IREE bin directory (for subprocess fallback).
                  Can also be set via IREE_BIN environment variable.

    Returns:
        IREEBackend instance (IREEModel or SubprocessRunner)
    """
    # Get IREE bin path from argument or environment
    if iree_bin is None:
        iree_bin = os.environ.get("IREE_BIN")

    if prefer_runtime:
        try:
            model = IREEModel(vmfb_path, device)
            return model
        except ImportError:
            print("Warning: iree-runtime not available, falling back to subprocess")
            return SubprocessRunner(vmfb_path, device, iree_bin=iree_bin)
        except Exception as e:
            # Version mismatch or other runtime error
            print(f"Warning: iree-runtime failed ({e}), falling back to subprocess")
            return SubprocessRunner(vmfb_path, device, iree_bin=iree_bin)
    else:
        return SubprocessRunner(vmfb_path, device, iree_bin=iree_bin)
