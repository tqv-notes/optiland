"""
JAX backend -- passthrough overrides with array() casting.

Provides PassthroughMixin, one of the mixins composed into
JaxBackend (see optiland/backend/jax_backend/__init__.py).

``jax.numpy`` functions accept arrays and Python scalars, but reject lists
and tuples, which the NumPy backend accepts. The overrides here convert
list and tuple arguments with ``self.array`` and then call the function of
the same name in ``jax.numpy``. The torch backend writes these overrides
out one by one; here they are generated, because every body is the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

# Functions of AbstractBackend's passthrough list whose arguments are arrays.
# Not listed: einsum (its first argument is a string), vstack and
# column_stack (their argument is a sequence of arrays), and searchsorted.
_ARRAY_FUNCTIONS = (
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "abs",
    "sign",
    "floor",
    "ceil",
    "hypot",
    "deg2rad",
    "rad2deg",
    "isnan",
    "isinf",
    "isfinite",
    "logical_and",
    "logical_or",
    "logical_not",
    "conj",
    "real",
    "imag",
    "outer",
    "dot",
    "nansum",
    "nanmean",
    "round",
    "copysign",
)


def _casting(name: str) -> Callable[..., Any]:
    """Build the method that calls ``jax.numpy.<name>`` on converted arguments."""
    function = getattr(jnp, name)

    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        args = tuple(
            self.array(arg) if isinstance(arg, list | tuple) else arg for arg in args
        )
        return function(*args, **kwargs)

    method.__name__ = name
    method.__qualname__ = f"PassthroughMixin.{name}"
    method.__doc__ = (
        f"Call ``jax.numpy.{name}``; list and tuple arguments become arrays."
    )
    return method


class PassthroughMixin:
    """Passthrough overrides with array() casting."""

    def allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Return True if all elements in a and b are close.

        Args:
            a: First input.
            b: Second input.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            bool: Whether all elements are close.
        """
        return bool(jnp.allclose(self.array(a), self.array(b), rtol=rtol, atol=atol))


for _name in _ARRAY_FUNCTIONS:
    setattr(PassthroughMixin, _name, _casting(_name))
