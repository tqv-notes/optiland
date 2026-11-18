"""
JAX backend -- implements AbstractBackend using JAX.

The implementation is split across same-package modules by operation
category (capabilities, creation, indexing, conic, reductions, passthrough,
linalg, interpolation, random, misc), like the torch backend; this module
composes them into the concrete ``JaxBackend`` class. See ``config.py`` for
the ``GradMode``/``_Config`` helpers.

Differences from the torch backend that callers see:

- JAX arrays are immutable. ``x[mask] = value`` raises; write
  ``x = be.where(mask, value, x)``.
- Derivatives come from function transformations (``jax.grad``,
  ``jax.jacfwd``), not from ``requires_grad`` tensors and ``backward()``.
- The default precision is float64, as on the NumPy backend. It sets JAX's
  ``jax_enable_x64`` flag the first time that the backend creates an array.
"""

from __future__ import annotations

import jax.numpy as jnp

from optiland.backend.base import AbstractBackend
from optiland.backend.jax_backend.capabilities import CapabilitiesMixin
from optiland.backend.jax_backend.config import _Config
from optiland.backend.jax_backend.conic import ConicMixin
from optiland.backend.jax_backend.creation import CreationMixin
from optiland.backend.jax_backend.indexing import IndexingMixin
from optiland.backend.jax_backend.interpolation import InterpolationMixin
from optiland.backend.jax_backend.linalg import LinalgMixin
from optiland.backend.jax_backend.misc import MiscMixin
from optiland.backend.jax_backend.passthrough import PassthroughMixin
from optiland.backend.jax_backend.random import RandomMixin
from optiland.backend.jax_backend.reductions import ReductionsMixin


class JaxBackend(
    CapabilitiesMixin,
    ConicMixin,
    CreationMixin,
    IndexingMixin,
    ReductionsMixin,
    PassthroughMixin,
    LinalgMixin,
    InterpolationMixin,
    RandomMixin,
    MiscMixin,
    AbstractBackend,
):
    """Backend implementation using JAX.

    Attributes:
        _lib: The ``jax.numpy`` module (used by passthrough methods).
        _config: Internal configuration (precision, grad_mode).
    """

    _lib = jnp

    def __init__(self) -> None:
        self._config = _Config()
