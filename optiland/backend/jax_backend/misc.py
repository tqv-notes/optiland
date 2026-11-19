"""
JAX backend -- miscellaneous operations.

Provides MiscMixin, one of the mixins composed into
JaxBackend (see optiland/backend/jax_backend/__init__.py).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jax.scipy.special import gammaln

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


class MiscMixin:
    """Miscellaneous operations."""

    def factorial(self, n: Any) -> Any:
        """Compute the factorial of n using the log-gamma function.

        Args:
            n: Non-negative integer or array of integers.

        Returns:
            jax.Array: Factorial values.
        """
        return jnp.exp(gammaln(self.array(n) + 1))

    def path_contains_points(self, vertices: Any, points: Any) -> Any:
        """Return a boolean mask of points inside the polygon.

        Uses a vectorized ray-crossing algorithm.

        Args:
            vertices: Polygon vertices as (N, 2) array (closed implicitly).
            points: Query points as (M, 2) array.

        Returns:
            jax.Array: Boolean mask of shape (M,).
        """
        vertices, points = self.array(vertices), self.array(points)
        vx, vy = vertices[:, 0], vertices[:, 1]
        px, py = points[:, 0][:, None], points[:, 1][:, None]
        vx_next, vy_next = jnp.roll(vx, -1), jnp.roll(vy, -1)

        crosses = (vy > py) != (vy_next > py)
        # A horizontal edge never crosses; give it a finite slope so that the
        # unused branch holds no division by zero.
        rise = jnp.where(vy_next != vy, vy_next - vy, 1.0)
        x_int = vx + (vx_next - vx) / rise * (py - vy)
        return jnp.sum(crosses & (px < x_int), axis=1) % 2 == 1

    def pad(
        self,
        tensor: Any,
        pad_width: Any,
        mode: str = "constant",
        constant_values: float | None = 0,
    ) -> Any:
        """Pad an array.

        Args:
            tensor: Input array.
            pad_width: Number of values padded per axis.
            mode: Padding mode, as in ``numpy.pad``.
            constant_values: Value used for constant padding.

        Returns:
            jax.Array: Padded array.
        """
        tensor = self._asjax(tensor)
        if mode == "constant":
            return jnp.pad(
                tensor, pad_width, mode=mode, constant_values=constant_values
            )
        return jnp.pad(tensor, pad_width, mode=mode)

    def vectorize(self, pyfunc: Callable[..., Any]) -> Callable[..., Any]:
        """Vectorize a scalar Python function over array inputs.

        The function is called once per element, so it may be any Python
        function; it need not be traceable by JAX.

        Args:
            pyfunc: The scalar function to vectorize.

        Returns:
            Callable: Vectorized function.
        """

        def wrapped(x: Any) -> Any:
            x = self.array(x)
            mapped = [self.cast(pyfunc(xi)) for xi in x.reshape(-1)]
            return jnp.stack(mapped).reshape(x.shape)

        return wrapped

    @contextlib.contextmanager
    def errstate(self, **kwargs: Any) -> Generator[None, None, None]:  # type: ignore[override]
        """No-op context manager (JAX raises no floating-point warnings).

        Args:
            **kwargs: Ignored.

        Yields:
            None
        """
        yield

    def histogram(self, x: Any, bins: Any = 10) -> tuple[Any, Any]:
        """Compute a histogram of x.

        Args:
            x: Input data.
            bins: Number of bins or bin edges.

        Returns:
            tuple[jax.Array, jax.Array]: Bin counts and bin edges.
        """
        return jnp.histogram(self._asjax(x), bins=bins)

    def histogram2d(
        self,
        x: Any,
        y: Any,
        bins: Any,
        weights: Any = None,
    ) -> tuple[Any, Any, Any]:
        """Compute a 2-D histogram.

        Args:
            x: x-coordinates of the sample points.
            y: y-coordinates of the sample points.
            bins: Bin specification (list of two edge arrays).
            weights: Optional weights for each sample.

        Returns:
            tuple[jax.Array, jax.Array, jax.Array]: Histogram, x edges, y edges.
        """
        hist, xedges, yedges = jnp.histogram2d(
            self._asjax(x), self._asjax(y), bins=bins, weights=weights
        )
        return hist, xedges, yedges

    # No copy_to: it writes into its destination, and JAX arrays are
    # immutable. It is not part of AbstractBackend; callers test for it.
