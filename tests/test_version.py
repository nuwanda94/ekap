"""Smoke test that the package is importable and versioned."""

from ekap import __version__


def test_version():
    assert __version__ == "0.1.0"
