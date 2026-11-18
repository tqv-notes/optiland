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

    def dtype(self) -> type:
        """Return the JAX float dtype of the current precision.

        JAX computes in 32 bits unless its ``jax_enable_x64`` flag is set.
        The flag is set here, the first time that float64 is needed, and not
        when optiland is imported: a program that has JAX installed but does
        not use this backend keeps its JAX configuration.
        """
        if self.precision == "float32":
            return jnp.float32
        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)
        return jnp.float64

    def complex_dtype(self) -> type:
        """Return the complex dtype that matches the current precision."""
        if self.precision == "float32":
            return jnp.complex64
        self.dtype()  # float64 and complex128 need the x64 flag
        return jnp.complex128
