"""
JAX backend -- configuration helpers (private to the backend package).

Provides GradMode/_Config, shared by JaxBackend's __init__ and the
capabilities mixin.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Generator


class GradMode:
    """Record whether the caller intends to differentiate through optiland.

    JAX has no per-array ``requires_grad`` flag: derivatives come from
    function transformations such as ``jax.grad``. This object keeps the same
    interface as the torch backend's ``GradMode`` so that code which asks
    ``be.grad_mode.requires_grad`` works on both backends.
    """

    def __init__(self) -> None:
        self.requires_grad: bool = False

    def enable(self) -> None:
        """Declare that gradients are wanted."""
        self.requires_grad = True

    def disable(self) -> None:
        """Declare that gradients are not wanted."""
        self.requires_grad = False

    @contextlib.contextmanager
    def temporary_enable(self) -> Generator[None, None, None]:
        """Context manager that temporarily sets ``requires_grad``."""
        old = self.requires_grad
        self.requires_grad = True
        try:
            yield
        finally:
            self.requires_grad = old


class _Config:
    """Internal configuration container for JaxBackend."""

    def __init__(self) -> None:
        self.precision: Literal["float32", "float64"] = "float64"
        self.grad_mode: GradMode = GradMode()

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        """Set the floating-point precision.

        Args:
            precision: ``'float32'`` or ``'float64'``.

        Raises:
            ValueError: If precision is not valid.
        """
        if precision not in ("float32", "float64"):
            raise ValueError("Precision must be 'float32' or 'float64'.")
        self.precision = precision

    @staticmethod
    def enable_x64() -> None:
        """Turn on JAX's 64-bit mode, which is off by default.

        Without it JAX computes in 32 bits, even for float64 requests. The
        flag is global to the process. It is set when the backend is
        selected (``be.set_backend("jax")``) and not when optiland is
        imported, so a program that has JAX installed but does not use this
        backend keeps its own JAX configuration. In 64-bit mode Python
        scalars and ``arange`` give float64, as on the NumPy backend; the
        ``float32`` precision still applies to every array the backend
        creates.
        """
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)

    def dtype(self) -> type:
        """Return the JAX float dtype of the current precision."""
        # Also when the backend instance is used without set_backend.
        self.enable_x64()
        return jnp.float32 if self.precision == "float32" else jnp.float64

    def complex_dtype(self) -> type:
        """Return the complex dtype that matches the current precision."""
        self.enable_x64()
        return jnp.complex64 if self.precision == "float32" else jnp.complex128
