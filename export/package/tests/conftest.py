"""
Pytest Configuration and Fixtures
=================================
"""

import pytest
from pathlib import Path


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "integration: integration tests requiring compiled model")
    config.addinivalue_line("markers", "slow: slow tests (MD simulations, etc.)")
    config.addinivalue_line("markers", "validation: numerical validation tests")
    config.addinivalue_line("markers", "benchmark: performance benchmark tests")
    config.addinivalue_line("markers", "lammps: tests requiring LAMMPS")


@pytest.fixture(scope="session")
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / 'fixtures'


@pytest.fixture(scope="session")
def models_dir():
    """Path to models directory."""
    return Path(__file__).parent.parent / 'src/mypotential/models'
