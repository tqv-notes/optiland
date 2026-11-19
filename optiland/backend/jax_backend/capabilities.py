"""
JAX backend -- identity, capability flags, overrides, and precision.

Provides CapabilitiesMixin, used by JaxBackend (see
optiland/backend/jax_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from optiland.backend.jax_backend.config import GradMode


class _Linalg:
    """``jax.numpy.linalg`` plus the ``LinAlgError`` name that callers catch."""

    LinAlgError = np.linalg.LinAlgError

    def __getattr__(self, name: str) -> Any:
        return getattr(jnp.linalg, name)

    def cholesky(self, a: Any, **kwargs: Any) -> Any:
        """Cholesky factor of a; raise if a is not positive definite.

        ``jax.numpy.linalg.cholesky`` signals failure with NaN, because a
        compiled function cannot raise. Callers on the other backends catch
        ``LinAlgError``, so a concrete NaN result is raised here. A traced
        result cannot be inspected and is returned as it is.

        Args:
            a: Symmetric (Hermitian) matrix, or a stack of them.
            **kwargs: Forwarded to ``jax.numpy.linalg.cholesky``.

        Returns:
            jax.Array: The Cholesky factor.

        Raises:
            numpy.linalg.LinAlgError: If the factor of a concrete matrix
                holds NaN.
        """
        factor = jnp.linalg.cholesky(a, **kwargs)
        if not isinstance(factor, jax.core.Tracer) and bool(jnp.isnan(factor).any()):
            raise self.LinAlgError("Matrix is not positive definite")
        return factor


class CapabilitiesMixin:
    """Identity, capability flags, overrides, and precision."""

    _linalg = _Linalg()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "jax"

    def _activate(self) -> None:
        """Set up JAX when ``be.set_backend("jax")`` selects this backend."""
        self._config.enable_x64()

    # ------------------------------------------------------------------
    # Submodule proxies
    # ------------------------------------------------------------------

    @property
    def linalg(self) -> Any:
        """Return the ``jax.numpy.linalg`` submodule."""
        return self._linalg

    @property
    def random(self) -> Any:
        """Return the ``jax.random`` submodule."""
        return jax.random

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_gradients(self) -> bool:
        """Return True -- JAX supports automatic differentiation."""
        return True

    @property
    def supports_gpu(self) -> bool:
        """Return True if JAX's default device is not a CPU."""
        return jax.default_backend() != "cpu"

    # ------------------------------------------------------------------
    # Capability-gated overrides
    # ------------------------------------------------------------------

    @property
    def grad_mode(self) -> GradMode:
        """Return the GradMode controller."""
        return self._config.grad_mode

    @property
    def autograd(self) -> Any:
        """Return the ``jax`` module, which holds ``grad``, ``jacfwd`` etc."""
        return jax

    def set_device(self, device: str) -> None:
        """Set the device on which JAX creates new arrays.

        Args:
            device: ``'cpu'``, or ``'gpu'`` / ``'cuda'`` for the first GPU.

        Raises:
            ValueError: If the device name is unknown, or if this JAX
                installation has no such device.
        """
        platform = {"cpu": "cpu", "gpu": "gpu", "cuda": "gpu"}.get(device)
        if platform is None:
            raise ValueError("Device must be 'cpu', 'gpu' or 'cuda'.")
        try:
            target = jax.devices(platform)[0]
        except RuntimeError as exc:
            raise ValueError(
                f"JAX has no {device!r} device in this installation."
            ) from exc
        jax.config.update("jax_default_device", target)

    def get_device(self) -> str:
        """Return the platform of the device for new arrays, e.g. ``'cpu'``."""
        device = jax.config.jax_default_device
        if device is None:
            return jax.default_backend()
        return getattr(device, "platform", str(device))

    def get_complex_precision(self) -> type:
        """Return the complex dtype that matches the current float precision."""
        return self._config.complex_dtype()

    def tensor(self, data: Any, **kwargs: Any) -> Any:
        """Create an array from data.

        Args:
            data: Input data (scalar, list, NumPy array, etc.).
            **kwargs: ``dtype`` is honored. ``device`` and ``requires_grad``
                are accepted for parity with the torch backend and ignored.

        Returns:
            jax.Array: New array.
        """
        return jnp.asarray(data, dtype=kwargs.get("dtype") or self._dtype)

    def to_tensor(self, data: Any, device: Any = None) -> Any:
        """Convert data to a JAX array with the backend's precision.

        Args:
            data: The data to convert.
            device: Accepted for parity with the torch backend and ignored.

        Returns:
            jax.Array: Converted array.
        """
        return jnp.asarray(data, dtype=self._dtype)

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------

    @property
    def _dtype(self) -> type:
        """Return the JAX float dtype of the current precision."""
        return self._config.dtype()

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        """Set the floating-point precision.

        Args:
            precision: ``'float32'`` or ``'float64'``.
        """
        self._config.set_precision(precision)

    def get_precision(self) -> int:
        """Return the current precision as an integer (32 or 64)."""
        return 32 if self._config.precision == "float32" else 64
