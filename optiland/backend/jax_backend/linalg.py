"""
JAX backend -- linear algebra operations.

Provides LinalgMixin, one of the mixins composed into
JaxBackend (see optiland/backend/jax_backend/__init__.py).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


class LinalgMixin:
    """Linear algebra operations."""

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix product of two arrays.

        Args:
            a: First matrix.
            b: Second matrix.

        Returns:
            jax.Array: Matrix product.
        """
        return jnp.matmul(self._asjax(a), self._asjax(b))

    def cross(
        self,
        a: Any,
        b: Any,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: int | None = None,
    ) -> Any:
        """Return the cross product of two vectors.

        Args:
            a: First vector array.
            b: Second vector array.
            axisa: Axis of a that defines the vector(s).
            axisb: Axis of b that defines the vector(s).
            axisc: Axis of c that contains the cross product vector.
            axis: If defined, the axis of a, b and c that defines the vectors.

        Returns:
            jax.Array: Cross product.
        """
        return jnp.cross(
            self._asjax(a),
            self._asjax(b),
            axisa=axisa,
            axisb=axisb,
            axisc=axisc,
            axis=axis,
        )

    def batched_chain_matmul3(self, a: Any, b: Any, c: Any) -> Any:
        """Compute a @ b @ c with promoted dtype.

        Args:
            a: First matrix.
            b: Second matrix.
            c: Third matrix.

        Returns:
            jax.Array: Result of a @ b @ c.
        """
        return jnp.matmul(jnp.matmul(self._asjax(a), self._asjax(b)), self._asjax(c))

    def matrix_vector_multiply_and_squeeze(self, p: Any, E: Any) -> Any:
        """Multiply p @ E[..., newaxis] and squeeze the trailing dimension.

        Args:
            p: Matrix array.
            E: Vector array.

        Returns:
            jax.Array: Result with trailing dimension squeezed.
        """
        return jnp.squeeze(jnp.matmul(p, self._asjax(E)[:, :, None]), axis=2)

    def mult_p_E(self, p: Any, E: Any) -> Any:
        """Complex matrix-vector multiply used for polarized fields.

        Args:
            p: Jones matrix array.
            E: Electric field array.

        Returns:
            jax.Array: Result of complex matrix-vector multiplication.
        """
        return jnp.squeeze(jnp.matmul(p, self._asjax(E)[:, :, None]), axis=2)

    def lstsq(self, a: Any, b: Any) -> Any:
        """Compute the least-squares solution to a @ x = b.

        Args:
            a: Left-hand side matrix (M, N).
            b: Right-hand side matrix (M,) or (M, K).

        Returns:
            jax.Array: Least-squares solution (N,) or (N, K).
        """
        return jnp.linalg.lstsq(self._asjax(a), self._asjax(b))[0]

    def to_complex(self, x: Any) -> Any:
        """Cast a real x to the complex dtype of the current precision.

        Args:
            x: Input array.

        Returns:
            jax.Array: Complex array. A complex input is returned as it is.
        """
        x = self._asjax(x)
        if jnp.iscomplexobj(x):
            return x
        return x.astype(self._config.complex_dtype())
