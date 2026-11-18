"""JAX conic intersections, on the arithmetic shared by all backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from optiland.backend._conic import _conic_candidates, _select_distance

if TYPE_CHECKING:
    from collections.abc import Callable


def _epsilon(value: Any) -> float:
    """Use the arithmetic result's floating dtype."""
    return float(jnp.finfo(jnp.result_type(value)).eps)


class ConicMixin:
    """Finite-conic execution and differentiation for the JAX backend."""

    def conic_intersection(
        self,
        x: Any,
        y: Any,
        z: Any,
        L: Any,
        M: Any,
        N: Any,
        radius: Any,
        conic: Any,
        contains: Callable | None = None,
    ) -> Any:
        """Select a finite-conic intersection with native JAX operations.

        See ``AbstractBackend.conic_intersection`` for the numerical contract.
        JAX differentiates regular selected branches; exact double roots keep
        their forward value and contribute zero gradient.
        """
        values = tuple(jnp.asarray(value) for value in (x, y, z, L, M, N))
        roots = _conic_candidates(
            *values, radius, conic, jnp.where, jnp.sqrt, jnp.copysign, _epsilon
        )
        distance = _select_distance(roots, values, contains, jnp.where)
        return jnp.where(roots.regular, distance, jax.lax.stop_gradient(distance))
