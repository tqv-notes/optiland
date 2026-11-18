"""
JAX backend -- array creation operations.

Provides CreationMixin, one of the mixins composed into
JaxBackend (see optiland/backend/jax_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class CreationMixin:
    """Array creation operations."""

    def array(self, x: Any) -> Any:
        """Create a JAX array with the current precision.

        An existing JAX array is returned as it is, with its dtype, so that
        integer, boolean and complex arrays and traced values pass through.

        Args:
            x: Input data.

        Returns:
            jax.Array: Backend array.
        """
        if isinstance(x, jax.Array):
            return x
        if isinstance(x, list | tuple) and any(isinstance(v, jax.Array) for v in x):
            # Keep the elements in the computation: a traced element must not
            # be converted through NumPy.
            items = [
                v if isinstance(v, jax.Array) else jnp.asarray(v, dtype=self._dtype)
                for v in x
            ]
            if len({item.shape for item in items}) > 1:
                items = [jnp.atleast_1d(item) for item in items]
                return jnp.concatenate(items)
            return jnp.stack(items)
        # Preserve bool dtype so bitwise ops receive bool arrays, not floats.
        if isinstance(x, np.ndarray) and x.dtype == np.bool_:
            return jnp.asarray(x)
        return jnp.asarray(x, dtype=self._dtype)

    def zeros(self, shape: Sequence[int], dtype: Any = None) -> Any:
        """Return a zero array of the given shape.

        Args:
            shape: Shape of the output array.
            dtype: Optional dtype override.

        Returns:
            jax.Array: Zero array.
        """
        return jnp.zeros(shape, dtype=dtype if dtype is not None else self._dtype)

    def ones(self, shape: Sequence[int], dtype: Any = None) -> Any:
        """Return an array of ones of the given shape.

        Args:
            shape: Shape of the output array.
            dtype: Optional dtype override.

        Returns:
            jax.Array: Ones array.
        """
        return jnp.ones(shape, dtype=dtype if dtype is not None else self._dtype)

    def full(self, shape: Sequence[int], fill_value: Any, dtype: Any = None) -> Any:
        """Return a constant-filled array of the given shape.

        Args:
            shape: Shape of the output array.
            fill_value: Fill value.
            dtype: Optional dtype override.

        Returns:
            jax.Array: Filled array.
        """
        _dtype = dtype if dtype is not None else self._dtype
        return jnp.full(shape, fill_value, dtype=_dtype)

    def linspace(self, start: float, stop: float, num: int = 50) -> Any:
        """Return evenly spaced numbers over an interval.

        Args:
            start: Start of the interval.
            stop: End of the interval.
            num: Number of samples.

        Returns:
            jax.Array: Evenly spaced samples.
        """
        return jnp.linspace(start, stop, num, dtype=self._dtype)

    def arange(self, *args: Any, **kwargs: Any) -> Any:
        """Return evenly spaced values within a given interval.

        Args:
            *args: start, stop, step (same as ``numpy.arange``).
            **kwargs: Additional keyword arguments passed to ``jnp.arange``.

        Returns:
            jax.Array: Array of evenly spaced values.
        """
        return jnp.arange(*args, **kwargs)

    def zeros_like(self, x: Any) -> Any:
        """Return a zero array with the same shape as x.

        Args:
            x: Reference array.

        Returns:
            jax.Array: Zero array.
        """
        return jnp.zeros(np.shape(x), dtype=self._dtype)

    def ones_like(self, x: Any) -> Any:
        """Return an array of ones with the same shape as x.

        Args:
            x: Reference array.

        Returns:
            jax.Array: Ones array.
        """
        return jnp.ones(np.shape(x), dtype=self._dtype)

    def full_like(self, x: Any, fill_value: Any) -> Any:
        """Return a full array with the same shape as x.

        Args:
            x: Reference array.
            fill_value: Fill value. A traced value stays in the computation.

        Returns:
            jax.Array: Filled array.
        """
        if np.ndim(fill_value) > 0 and np.size(fill_value) == 1:
            # NumPy fills any shape from a one-element array; JAX will not
            # broadcast shape (1,) down to shape ().
            fill_value = jnp.reshape(fill_value, ())
        return jnp.full(np.shape(x), fill_value, dtype=self._dtype)

    def empty(self, shape: Sequence[int]) -> Any:
        """Return a new array of the given shape.

        JAX has no uninitialized arrays, so the array holds zeros.

        Args:
            shape: Shape of the output array.

        Returns:
            jax.Array: Zero array.
        """
        return jnp.zeros(shape, dtype=self._dtype)

    def empty_like(self, x: Any) -> Any:
        """Return a new array with the same shape as x, holding zeros.

        Args:
            x: Reference array.

        Returns:
            jax.Array: Zero array.
        """
        return jnp.zeros(np.shape(x), dtype=self._dtype)

    def eye(self, n: int) -> Any:
        """Return a 2D identity matrix.

        Args:
            n: Size of the identity matrix.

        Returns:
            jax.Array: Identity matrix.
        """
        return jnp.eye(n, dtype=self._dtype)

    def asarray(self, x: Any, **kwargs: Any) -> Any:
        """Convert x to a JAX array.

        Args:
            x: Input data.
            **kwargs: Keyword arguments forwarded to ``jnp.asarray``
                (e.g. ``dtype``).

        Returns:
            jax.Array: Backend array.
        """
        dtype = kwargs.pop("dtype", self._dtype)
        return jnp.asarray(x, dtype=dtype, **kwargs)

    def load(self, filename: str) -> Any:
        """Load a NumPy file and convert it to a JAX array.

        Args:
            filename: Path to a ``.npy`` file.

        Returns:
            jax.Array: Loaded array.
        """
        return self.array(np.load(filename))
