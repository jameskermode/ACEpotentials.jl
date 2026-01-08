#!/usr/bin/env python3
"""
Compile StableHLO Model to IREE Artifacts
=========================================

Compiles a StableHLO MLIR model to IREE VMFB files for multiple backends
(CPU, CUDA, Vulkan). The output files can be packaged into a portable
Python package.

Usage:
    python compile_model.py input.mlir --output-dir ./models

    # Compile only for CPU
    python compile_model.py input.mlir --output-dir ./models --backends cpu

    # Compile for all backends
    python compile_model.py input.mlir --output-dir ./models --backends all
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import json
import shutil


# Backend configurations
BACKENDS = {
    'cpu': {
        'iree_backend': 'llvm-cpu',
        'output_file': 'model_cpu.vmfb',
        'description': 'CPU (llvm-cpu backend)',
    },
    'cuda': {
        'iree_backend': 'cuda',
        'output_file': 'model_cuda.vmfb',
        'description': 'NVIDIA GPU (CUDA)',
        # Compile for multiple SM versions for broad compatibility
        'extra_flags': [
            '--iree-hal-cuda-llvm-target-arch=sm_70',  # Volta
        ],
    },
    'vulkan': {
        'iree_backend': 'vulkan-spirv',
        'output_file': 'model_vulkan.vmfb',
        'description': 'GPU (Vulkan/SPIR-V)',
    },
}


def find_iree_compile() -> Optional[Path]:
    """Find iree-compile executable."""
    # Check if in PATH
    result = shutil.which('iree-compile')
    if result:
        return Path(result)

    # Check common locations
    common_paths = [
        Path.home() / '.local' / 'bin' / 'iree-compile',
        Path('/usr/local/bin/iree-compile'),
    ]

    for path in common_paths:
        if path.exists():
            return path

    return None


def compile_backend(
    mlir_path: Path,
    output_dir: Path,
    backend: str,
    verbose: bool = False,
) -> bool:
    """
    Compile MLIR to VMFB for a specific backend.

    Args:
        mlir_path: Path to input MLIR file
        output_dir: Output directory for VMFB
        backend: Backend name ('cpu', 'cuda', 'vulkan')
        verbose: Print detailed output

    Returns:
        True if compilation succeeded
    """
    if backend not in BACKENDS:
        print(f"Unknown backend: {backend}")
        return False

    config = BACKENDS[backend]
    output_path = output_dir / config['output_file']

    iree_compile = find_iree_compile()
    if not iree_compile:
        print("ERROR: iree-compile not found. Install with: pip install iree-compiler")
        return False

    cmd = [
        str(iree_compile),
        str(mlir_path),
        '--iree-input-type=stablehlo',
        f'--iree-hal-target-backends={config["iree_backend"]}',
        '-o', str(output_path),
    ]

    # Add backend-specific flags
    if 'extra_flags' in config:
        cmd.extend(config['extra_flags'])

    if verbose:
        print(f"Compiling {backend}...")
        print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True,
        )

        size = output_path.stat().st_size
        print(f"  SUCCESS: {config['output_file']} ({size:,} bytes)")
        return True

    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {backend}")
        if e.stderr:
            print(f"    Error: {e.stderr[:500]}")
        return False

    except FileNotFoundError:
        print(f"  FAILED: iree-compile not found")
        return False


def compile_all(
    mlir_path: Path,
    output_dir: Path,
    backends: List[str],
    verbose: bool = False,
) -> dict:
    """
    Compile MLIR to multiple backends.

    Args:
        mlir_path: Input MLIR file
        output_dir: Output directory
        backends: List of backends to compile
        verbose: Verbose output

    Returns:
        Dictionary of {backend: success_bool}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for backend in backends:
        results[backend] = compile_backend(mlir_path, output_dir, backend, verbose)

    return results


def create_metadata(
    output_dir: Path,
    model_info: dict,
    compilation_results: dict,
):
    """
    Create metadata.json with model information.

    Args:
        output_dir: Output directory
        model_info: Model metadata (cutoff, elements, etc.)
        compilation_results: Backend compilation results
    """
    metadata = {
        'name': model_info.get('name', 'ACE Potential'),
        'version': model_info.get('version', '1.0.0'),
        'model': {
            'cutoff': model_info.get('cutoff', 6.0),
            'cutoff_units': 'angstrom',
            'elements': model_info.get('elements', []),
            'max_atoms': model_info.get('max_atoms', 4096),
            'max_pairs': model_info.get('max_pairs', 200000),
        },
        'units': {
            'length': 'angstrom',
            'energy': 'eV',
        },
        'compilation': {
            'backends': [b for b, success in compilation_results.items() if success],
            'failed_backends': [b for b, success in compilation_results.items() if not success],
        },
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata written to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compile StableHLO model to IREE VMFB artifacts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compile for CPU only
    python compile_model.py model.mlir -o ./package/models --backends cpu

    # Compile for all available backends
    python compile_model.py model.mlir -o ./package/models --backends all

    # Compile with verbose output
    python compile_model.py model.mlir -o ./package/models -v
        """
    )

    parser.add_argument(
        'mlir_path',
        type=Path,
        help='Path to input StableHLO MLIR file',
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('./models'),
        help='Output directory for VMFB files (default: ./models)',
    )

    parser.add_argument(
        '-b', '--backends',
        nargs='+',
        default=['cpu'],
        choices=['cpu', 'cuda', 'vulkan', 'all'],
        help='Backends to compile for (default: cpu)',
    )

    parser.add_argument(
        '--cutoff',
        type=float,
        default=6.0,
        help='Model cutoff radius in Angstroms (for metadata)',
    )

    parser.add_argument(
        '--elements',
        nargs='+',
        default=[],
        help='Element symbols (for metadata)',
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output',
    )

    args = parser.parse_args()

    # Validate input
    if not args.mlir_path.exists():
        print(f"ERROR: Input file not found: {args.mlir_path}")
        sys.exit(1)

    # Expand 'all' to all backends
    backends = args.backends
    if 'all' in backends:
        backends = list(BACKENDS.keys())

    print(f"Compiling: {args.mlir_path}")
    print(f"Output: {args.output_dir}")
    print(f"Backends: {', '.join(backends)}")
    print()

    # Compile
    results = compile_all(args.mlir_path, args.output_dir, backends, args.verbose)

    # Create metadata
    model_info = {
        'cutoff': args.cutoff,
        'elements': args.elements,
    }
    create_metadata(args.output_dir, model_info, results)

    # Summary
    print()
    print("=" * 50)
    succeeded = sum(results.values())
    total = len(results)
    print(f"Compilation complete: {succeeded}/{total} backends succeeded")

    if succeeded < total:
        failed = [b for b, s in results.items() if not s]
        print(f"Failed backends: {', '.join(failed)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
