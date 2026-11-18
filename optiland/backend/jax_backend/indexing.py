"""
JAX backend -- array utilities, shape, and indexing operations.

Provides IndexingMixin, one of the mixins composed into
JaxBackend (see optiland/backend/jax_backend/__init__.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class IndexingMixin:
    """Array utilities, shape, and indexing operations."""

    # ------------------------------------------------------------------
    # Array utilities
    # ------------------------------------------------------------------

    def cast(self, x: Any) -> Any:
        """Cast x to the current floating-point precision.

        Args:
            x: Input data.

        Returns:
            jax.Array: Array cast to current precision.
        """
        return jnp.asarray(x, dtype=self._dtype)

    def is_array_like(self, x: Any) -> bool:
        """Return True if x is a JAX array, ndarray, list, or tuple.

        Args:
            x: Object to check.

        Returns:
            bool: True if x is array-like.
        """
        return isinstance(x, jax.Array | np.ndarray | list | tuple)

    def arange_indices(self, start: Any, stop: Any = None, step: int = 1) -> Any:
        """Create an integer array of indices.

        Args:
            start: Start index (or stop if stop is None).
            stop: Stop index.
            step: Step size.

        Returns:
            jax.Array: Integer index array.
        """
        if stop is None:
            start, stop = 0, start
        return jnp.arange(start, stop, step, dtype=int)

    def copy(self, x: Any) -> Any:
        """Return a copy of x.

        Args:
            x: Input array.

        Returns:
            jax.Array: Copied array.
        """
        return jnp.array(x)

    def size(self, x: Any) -> int:
        """Return the total number of elements in x.

        Args:
            x: Input array or scalar.

        Returns:
            int: Number of elements.
        """
        return int(np.prod(np.shape(x), dtype=int))

    def shape(self, x: Any) -> tuple[int, ...]:
        """Return the shape of x.

        Args:
            x: Input array.

        Returns:
            tuple[int, ...]: Shape of the array.
        """
        return np.shape(x)

    def isscalar(self, x: Any) -> bool:
        """Return True if x is a scalar or a 0-dimensional JAX array.

        Args:
            x: Input.

        Returns:
            bool: Whether x is a scalar.
        """
        return np.isscalar(x) or (isinstance(x, jax.Array) and x.ndim == 0)

    def ravel(self, x: Any) -> Any:
        """Return a flattened array.

        Args:
            x: Input array or scalar.

        Returns:
            jax.Array: 1-D array.
        """
        return jnp.ravel(self.array(x))

    # ------------------------------------------------------------------
    # Shape and indexing
    # ------------------------------------------------------------------

    def transpose(self, x: Any, axes: Sequence[int] | None = None) -> Any:
        """Permute the dimensions of x.

        Args:
            x: Input array.
            axes: Permutation of dimensions.

        Returns:
            jax.Array: Transposed array.
        """
        return jnp.transpose(self.array(x), axes)

    def reshape(self, x: Any, shape: Sequence[int]) -> Any:
        """Return x with a new shape.

        Args:
            x: Input array.
            shape: New shape.

        Returns:
            jax.Array: Reshaped array.
        """
        return jnp.reshape(self.array(x), shape)

    def atleast_1d(self, x: Any) -> Any:
        """Convert x to a float array with at least one dimension.

        Args:
            x: Input data.

        Returns:
            jax.Array: Array with at least 1 dimension.
        """
        return jnp.atleast_1d(jnp.asarray(x, dtype=self._dtype))

    def atleast_2d(self, x: Any) -> Any:
        """Convert x to an array with at least two dimensions.

        Args:
            x: Input data.

        Returns:
            jax.Array: Array with at least 2 dimensions.
        """
        return jnp.atleast_2d(self.array(x))

    def as_array_1d(self, data: Any) -> Any:
        """Force conversion to a 1-D array.

        Args:
            data: Scalar, list, tuple, or array.

        Returns:
            jax.Array: 1-D array.

        Raises:
            ValueError: If data type is not supported.
        """
        if isinstance(data, int | float):
            return self.array([data])
        if isinstance(data, list | tuple):
            return self.array(data)
        if self.is_array_like(data):
            return self.array(data).reshape(-1)
        raise ValueError(
            "Unsupported input type: expected scalar, list, tuple, or array-like."
        )

    def stack(self, xs: Sequence[Any], axis: int = 0) -> Any:
        """Join a sequence of arrays along a new axis.

        Args:
            xs: Sequence of arrays.
            axis: Axis along which to stack.

        Returns:
            jax.Array: Stacked array.
        """
        return jnp.stack([self.array(x) for x in xs], axis=axis)

    def concatenate(self, arrays: Sequence[Any], axis: int = 0) -> Any:
        """Join arrays along an existing axis.

        Args:
            arrays: Sequence of arrays to concatenate.
            axis: Axis along which to concatenate.

        Returns:
            jax.Array: Concatenated array.
        """
        return jnp.concatenate([self.array(x) for x in arrays], axis=axis)

    def flip(self, x: Any) -> Any:
        """Reverse the order of elements along axis 0.

        Args:
            x: Input array.

        Returns:
            jax.Array: Flipped array.
        """
        return jnp.flip(self.array(x), axis=0)

    def roll(self, x: Any, shift: Any, axis: Any = ()) -> Any:
        """Roll x elements along the given axis.

        Args:
            x: Input array.
            shift: Number of places to shift.
            axis: Axis or axes along which to roll.

        Returns:
            jax.Array: Rolled array.
        """
        return jnp.roll(self.array(x), shift, axis=axis if axis != () else None)

    def repeat(self, x: Any, repeats: int) -> Any:
        """Repeat elements of x.

        Args:
            x: Input array.
            repeats: Number of repetitions.

        Returns:
            jax.Array: Repeated array.
        """
        return jnp.repeat(self.array(x), repeats)

    def broadcast_to(self, x: Any, shape: Sequence[int]) -> Any:
        """Broadcast x to the given shape.

        Args:
            x: Input array.
            shape: Target shape.

        Returns:
            jax.Array: Broadcast array.
        """
        return jnp.broadcast_to(self.array(x), shape)

    def tile(self, x: Any, dims: Any) -> Any:
        """Construct an array by tiling x.

        Args:
            x: Input array.
            dims: Number of repetitions per dimension.

        Returns:
            jax.Array: Tiled array.
        """
        return jnp.tile(self.array(x), dims)

    def expand_dims(self, x: Any, axis: int) -> Any:
        """Insert a new axis into x.

        Args:
            x: Input array.
            axis: Position of the new axis.

        Returns:
            jax.Array: Expanded array.
        """
        return jnp.expand_dims(self.array(x), axis)

    def meshgrid(self, *arrays: Any) -> tuple[Any, ...]:
        """Return coordinate matrices from coordinate vectors (xy indexing).

        Args:
            *arrays: 1-D arrays representing grid coordinates.

        Returns:
            tuple[jax.Array, ...]: Coordinate matrices.
        """
        return tuple(jnp.meshgrid(*(self.array(a) for a in arrays), indexing="xy"))

    def unsqueeze_last(self, x: Any) -> Any:
        """Add a trailing dimension to x.

        Args:
            x: Input array.

        Returns:
            jax.Array: Array with an extra trailing dimension.
        """
        return self.array(x)[..., None]

    def index_add(self, target: Any, dim: int, index: Any, source: Any) -> Any:
        """Scatter-add: result[index[i]] += source[i] along dim.

        Args:
            target: Accumulation buffer.
            dim: Dimension along which to scatter.
            index: 1-D integer indices.
            source: Values to add.

        Returns:
            jax.Array: New array with the values added. Repeated indices
                accumulate.
        """
        where = (slice(None),) * dim + (index,)
        return target.at[where].add(source)
