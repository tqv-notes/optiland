"""
JAX backend -- reduction operations.

Provides ReductionsMixin, one of the mixins composed into
JaxBackend (see optiland/backend/jax_backend/__init__.py). The semantics
follow the NumPy backend, because ``jax.numpy`` follows NumPy.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class ReductionsMixin:
    """Reduction operations."""

    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Sum array elements over a given axis.

        Args:
            x: Input array.
            axis: Axis along which to sum.

        Returns:
            jax.Array: Sum of x.
        """
        return jnp.sum(self.array(x), axis=axis)

    def mean(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        """Compute the arithmetic mean along an axis.

        Args:
            x: Input array.
            axis: Axis along which to compute the mean.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            jax.Array: Mean of x.
        """
        return jnp.mean(self.array(x), axis=axis, keepdims=keepdims)

    def std(self, x: Any, axis: int | None = None) -> Any:
        """Compute the standard deviation along an axis.

        Args:
            x: Input array.
            axis: Axis along which to compute the std.

        Returns:
            jax.Array: Standard deviation.
        """
        return jnp.std(self.array(x), axis=axis)

    def max(self, x: Any) -> Any:
        """Return the maximum value of x.

        Args:
            x: Input array.

        Returns:
            jax.Array: Maximum value, 0-dimensional.
        """
        return jnp.max(self.array(x))

    def min(self, x: Any) -> Any:
        """Return the minimum value of x.

        Args:
            x: Input array.

        Returns:
            jax.Array: Minimum value, 0-dimensional.
        """
        return jnp.min(self.array(x))

    def argmin(self, x: Any, axis: int | None = None) -> Any:
        """Return indices of the minimum values along an axis.

        Args:
            x: Input array.
            axis: Axis along which to find the minimum.

        Returns:
            jax.Array: Index array.
        """
        return jnp.argmin(self.array(x), axis=axis)

    def argwhere(self, x: Any) -> Any:
        """Return indices of non-zero elements.

        The size of the result depends on the values, so this function
        cannot be used inside ``jax.jit``.

        Args:
            x: Input array.

        Returns:
            jax.Array: Index array of shape (N, ndim).
        """
        return jnp.argwhere(self.array(x))

    def clip(self, x: Any, a_min: Any, a_max: Any) -> Any:
        """Clip values in x to [a_min, a_max].

        Args:
            x: Input array.
            a_min: Minimum value.
            a_max: Maximum value.

        Returns:
            jax.Array: Clipped array.
        """
        return jnp.clip(self.array(x), a_min, a_max)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Return elements from x or y depending on condition.

        Args:
            condition: Boolean array or bool.
            x: Values where condition is True.
            y: Values where condition is False.

        Returns:
            jax.Array: Output array.
        """
        if isinstance(condition, bool):
            return x if condition else y
        return jnp.where(condition, x, y)

    def maximum(self, a: Any, b: Any) -> Any:
        """Element-wise maximum of a and b.

        Args:
            a: First input array.
            b: Second input array.

        Returns:
            jax.Array: Element-wise maximum.
        """
        return jnp.maximum(self.array(a), self.array(b))

    def minimum(self, a: Any, b: Any) -> Any:
        """Element-wise minimum of a and b.

        Args:
            a: First input array.
            b: Second input array.

        Returns:
            jax.Array: Element-wise minimum.
        """
        return jnp.minimum(self.array(a), self.array(b))

    def fmax(self, a: Any, b: Any) -> Any:
        """Element-wise maximum, ignoring NaNs.

        Args:
            a: First input array.
            b: Second input array.

        Returns:
            jax.Array: Element-wise maximum ignoring NaN.
        """
        return jnp.fmax(self.array(a), self.array(b))

    def power(self, x: Any, y: Any) -> Any:
        """Return x raised to the power y.

        Args:
            x: Base array.
            y: Exponent array.

        Returns:
            jax.Array: x ** y.
        """
        return jnp.power(self.array(x), self.array(y))

    def diff(self, x: Any, n: int = 1, axis: int = -1, **kwargs: Any) -> Any:
        """Calculate the n-th discrete difference along the given axis.

        Args:
            x: Input array.
            n: Number of times to apply the difference.
            axis: Axis along which to compute differences.
            **kwargs: Additional keyword arguments forwarded to ``jnp.diff``
                (e.g. ``prepend``, ``append``).

        Returns:
            jax.Array: Differences array.
        """
        return jnp.diff(self.array(x), n=n, axis=axis, **kwargs)

    def all(self, x: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Return True if all elements of x are True.

        Args:
            x: Input array or bool.
            axis: Axis or axes to reduce. None reduces the whole input.

        Returns:
            Python bool for a whole-input reduction; otherwise a boolean JAX
            array with the unreduced dimensions.
        """
        if axis is not None:
            return jnp.all(jnp.asarray(x), axis=axis)
        if isinstance(x, jax.Array):
            return bool(jnp.all(x))
        return bool(np.all(x))

    def any(self, x: Any) -> bool:
        """Return True if any element of x is True.

        Args:
            x: Input array or bool.

        Returns:
            bool: Whether any element is True.
        """
        if isinstance(x, jax.Array):
            return bool(jnp.any(x))
        return bool(np.any(x))

    def nanmax(self, x: Any, axis: int | None = None, keepdim: bool = False) -> Any:
        """Return the maximum value, ignoring NaNs.

        Args:
            x: Input array.
            axis: Axis along which to compute the maximum.
            keepdim: Whether to keep reduced dimensions.

        Returns:
            jax.Array: Maximum value ignoring NaN.
        """
        return jnp.nanmax(self.array(x), axis=axis, keepdims=keepdim)

    def sort(self, x: Any, axis: int = -1) -> Any:
        """Return a sorted copy of x.

        Args:
            x: Input array.
            axis: Axis along which to sort.

        Returns:
            jax.Array: Sorted array.
        """
        return jnp.sort(self.array(x), axis=axis)

    def isclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> Any:
        """Return a boolean array where elements are close.

        Args:
            a: First input.
            b: Second input.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            jax.Array: Boolean array.
        """
        return jnp.isclose(self.array(a), self.array(b), rtol=rtol, atol=atol)

    def radians(self, x: Any) -> Any:
        """Convert angles from degrees to radians.

        Args:
            x: Angle in degrees.

        Returns:
            jax.Array: Angle in radians.
        """
        return jnp.radians(self.array(x))

    def degrees(self, x: Any) -> Any:
        """Convert angles from radians to degrees.

        Args:
            x: Angle in radians.

        Returns:
            jax.Array: Angle in degrees.
        """
        return jnp.degrees(self.array(x))
