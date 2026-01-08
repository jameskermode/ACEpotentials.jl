#!/usr/bin/env python3
"""
LAMMPS Integration Tests for pair_iree
======================================

Tests the LAMMPS pair_iree plugin using the Python interface.

Requirements:
- LAMMPS built with Python support
- pair_iree plugin loaded (either compiled-in or as shared library)
- Test VMFB model file

Usage:
    pytest test_lammps_integration.py -v
    python test_lammps_integration.py
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os


# Check if LAMMPS Python interface is available
try:
    from lammps import lammps
    HAS_LAMMPS = True
except ImportError:
    HAS_LAMMPS = False

# Check if test model exists
TEST_DIR = Path(__file__).parent
MODEL_PATH = TEST_DIR / 'test_model.vmfb'
HAS_MODEL = MODEL_PATH.exists()

skip_no_lammps = pytest.mark.skipif(not HAS_LAMMPS, reason="LAMMPS not available")
skip_no_model = pytest.mark.skipif(not HAS_MODEL, reason="Test VMFB not available")


@pytest.fixture
def lmp():
    """Create LAMMPS instance."""
    if not HAS_LAMMPS:
        pytest.skip("LAMMPS not available")

    lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none', '-nocite'])
    yield lmp
    lmp.close()


@pytest.fixture
def silicon_data_file():
    """Create temporary LAMMPS data file for silicon."""
    # 8-atom silicon diamond cell
    content = """Silicon diamond structure

8 atoms
1 atom types

0.0 5.43 xlo xhi
0.0 5.43 ylo yhi
0.0 5.43 zlo zhi

Masses

1 28.0855

Atoms

1 1 0.0000 0.0000 0.0000
2 1 1.3575 1.3575 1.3575
3 1 2.7150 2.7150 0.0000
4 1 4.0725 4.0725 1.3575
5 1 2.7150 0.0000 2.7150
6 1 4.0725 1.3575 4.0725
7 1 0.0000 2.7150 2.7150
8 1 1.3575 4.0725 4.0725
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.data', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name

    os.unlink(f.name)


@skip_no_lammps
class TestPairStyleLoad:
    """Test loading pair_style iree."""

    def test_lammps_works(self, lmp):
        """Basic LAMMPS functionality."""
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        # Should not raise

    def test_pair_style_exists(self, lmp):
        """Check if pair_style iree is available."""
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command("boundary p p p")
        lmp.command("region box block 0 10 0 10 0 10")
        lmp.command("create_box 1 box")

        try:
            # This will fail if pair_style iree is not loaded
            lmp.command("pair_style iree nonexistent.vmfb")
            # If we get here, pair_style exists (will fail later on missing file)
        except Exception as e:
            if "Unknown pair style" in str(e):
                pytest.skip("pair_style iree not available")
            # Other errors (like file not found) mean the style exists


@skip_no_lammps
@skip_no_model
class TestBasicCalculation:
    """Test basic energy/force calculations."""

    def test_setup_silicon(self, lmp, silicon_data_file):
        """Set up silicon calculation."""
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command(f"read_data {silicon_data_file}")
        lmp.command(f"pair_style iree {MODEL_PATH}")
        lmp.command("pair_coeff * * Si")

        # Run zero steps to compute
        lmp.command("run 0")

        pe = lmp.get_thermo("pe")
        assert np.isfinite(pe), "Potential energy is not finite"

    def test_energy_reasonable(self, lmp, silicon_data_file):
        """Energy should be reasonable for silicon."""
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command(f"read_data {silicon_data_file}")
        lmp.command(f"pair_style iree {MODEL_PATH}")
        lmp.command("pair_coeff * * Si")
        lmp.command("run 0")

        pe = lmp.get_thermo("pe")
        natoms = lmp.get_natoms()

        # Silicon cohesive energy ~4.6 eV/atom
        pe_per_atom = pe / natoms
        # Allow wide range for different model parametrizations
        assert -10 < pe_per_atom < 10, f"Energy per atom {pe_per_atom} seems unreasonable"

    def test_forces_computed(self, lmp, silicon_data_file):
        """Forces should be computed."""
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command(f"read_data {silicon_data_file}")
        lmp.command(f"pair_style iree {MODEL_PATH}")
        lmp.command("pair_coeff * * Si")
        lmp.command("run 0")

        # Get forces
        forces = lmp.numpy.extract_atom('f')
        assert forces.shape[1] == 3
        assert np.all(np.isfinite(forces))

        # For perfect crystal, forces should be small
        max_force = np.max(np.abs(forces))
        assert max_force < 0.1, f"Max force {max_force} too large for perfect crystal"


@skip_no_lammps
@skip_no_model
class TestMDStability:
    """Test molecular dynamics stability."""

    def test_nve_runs(self, lmp, silicon_data_file):
        """NVE MD should run without crashing."""
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command(f"read_data {silicon_data_file}")
        lmp.command(f"pair_style iree {MODEL_PATH}")
        lmp.command("pair_coeff * * Si")

        lmp.command("velocity all create 300 12345 dist gaussian")
        lmp.command("fix 1 all nve")
        lmp.command("thermo 10")

        # Run 100 steps
        lmp.command("run 100")

        # Check final state is reasonable
        pe = lmp.get_thermo("pe")
        ke = lmp.get_thermo("ke")
        assert np.isfinite(pe + ke)

    @pytest.mark.slow
    def test_nve_energy_conservation(self, lmp, silicon_data_file):
        """NVE should conserve energy."""
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command(f"read_data {silicon_data_file}")
        lmp.command(f"pair_style iree {MODEL_PATH}")
        lmp.command("pair_coeff * * Si")

        lmp.command("velocity all create 300 12345 dist gaussian")
        lmp.command("fix 1 all nve")
        lmp.command("timestep 0.001")  # 1 fs

        # Get initial energy
        lmp.command("run 0")
        E_initial = lmp.get_thermo("etotal")

        # Run for a while
        lmp.command("run 500")
        E_final = lmp.get_thermo("etotal")

        # Energy drift
        drift = abs(E_final - E_initial)
        natoms = lmp.get_natoms()
        drift_per_atom = drift / natoms

        # Should be < 1 meV/atom over 0.5 ps
        assert drift_per_atom < 0.001, f"Energy drift {drift_per_atom} eV/atom too large"


@skip_no_lammps
@skip_no_model
class TestNewtonPair:
    """Test Newton pair flag behavior."""

    def test_newton_on(self, lmp, silicon_data_file):
        """Test with newton pair on."""
        lmp.command("newton on")
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command(f"read_data {silicon_data_file}")
        lmp.command(f"pair_style iree {MODEL_PATH}")
        lmp.command("pair_coeff * * Si")
        lmp.command("run 0")

        pe_on = lmp.get_thermo("pe")
        lmp.close()

        # Test with newton off
        lmp2 = lammps(cmdargs=['-log', 'none', '-screen', 'none', '-nocite'])
        lmp2.command("newton off")
        lmp2.command("units metal")
        lmp2.command("atom_style atomic")
        lmp2.command(f"read_data {silicon_data_file}")
        lmp2.command(f"pair_style iree {MODEL_PATH}")
        lmp2.command("pair_coeff * * Si")
        lmp2.command("run 0")

        pe_off = lmp2.get_thermo("pe")
        lmp2.close()

        # Should give same energy
        assert np.isclose(pe_on, pe_off, rtol=1e-10)


@skip_no_lammps
@skip_no_model
class TestStress:
    """Test stress/pressure calculations."""

    def test_pressure_computed(self, lmp, silicon_data_file):
        """Pressure should be computed."""
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command(f"read_data {silicon_data_file}")
        lmp.command(f"pair_style iree {MODEL_PATH}")
        lmp.command("pair_coeff * * Si")
        lmp.command("run 0")

        press = lmp.get_thermo("press")
        assert np.isfinite(press)

    def test_virial_components(self, lmp, silicon_data_file):
        """Individual virial components should be computed."""
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command(f"read_data {silicon_data_file}")
        lmp.command(f"pair_style iree {MODEL_PATH}")
        lmp.command("pair_coeff * * Si")

        # Compute per-atom stress
        lmp.command("compute stress all stress/atom NULL virial")
        lmp.command("run 0")

        # Get per-atom stress
        stress = lmp.numpy.extract_compute('stress', 1, 2)
        assert stress.shape[1] == 6
        assert np.all(np.isfinite(stress))


def main():
    """Run tests directly."""
    pytest.main([__file__, '-v', '-x'])


if __name__ == '__main__':
    main()
