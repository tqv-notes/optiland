from __future__ import annotations

import os

import pytest

import optiland.backend as be


def pytest_addoption(parser):
    """Register the --update-golden flag used by the regression snapshot suite."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate tests/regression/fixtures/*.json instead of comparing "
        "against them.",
    )


def _test_backends() -> list[str]:
    """Return the backends that ``set_test_backend`` runs each test on.

    JAX is left out unless ``OPTILAND_TEST_JAX=1``: parts of the library do
    not support it yet. ``tests/test_jax_backend.py`` runs regardless.
    """
    names = be.list_available_backends()
    if not _jax_tests_enabled():
        names = [name for name in names if name != "jax"]
    return names


def _jax_tests_enabled() -> bool:
    """Return True if ``OPTILAND_TEST_JAX=1`` opts in to backend tests on JAX."""
    return os.environ.get("OPTILAND_TEST_JAX") == "1"


@pytest.fixture(params=_test_backends(), ids=lambda b: f"backend={b}")
def set_test_backend(request):
    """Fixture to set the backend for each test and ensure proper device configuration."""
    backend_name = request.param
    if backend_name == "jax" and not _jax_tests_enabled():
        # Tests that parametrize this fixture themselves (indirect=True) bypass
        # _test_backends(), so the opt-in is checked here as well.
        pytest.skip("JAX backend tests run only with OPTILAND_TEST_JAX=1")
    be.set_backend(backend_name)

    if backend_name == "torch":
        be.set_device("cpu")  # Use CPU for tests
        be.grad_mode.enable()  # Enable gradient tracking
        be.set_precision("float64")  # Set precision to float64 for tests

    yield

    # Reset the backend to numpy after the test
    be.set_backend("numpy")
