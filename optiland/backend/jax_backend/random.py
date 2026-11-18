"""
JAX backend -- random-number generation operations.

Provides RandomMixin, one of the mixins composed into
JaxBackend (see optiland/backend/jax_backend/__init__.py).

JAX random functions are pure: each call takes an explicit key and the
caller must split keys by hand. The backend contract is stateful instead
(``default_rng(seed)`` returns a generator that later calls advance), so
``JaxGenerator`` holds a key and splits it on every draw.
"""

from __future__ import annotations

import secrets
from typing import Any

import jax
from jax.scipy.special import erfinv as _erfinv


class JaxGenerator:
    """Stateful random generator over a JAX PRNG key.

    Args:
        seed: Seed of the key. None draws a seed from the operating system.
    """

    def __init__(self, seed: int | None = None) -> None:
        if seed is None:
            seed = secrets.randbits(63)
        self._key = jax.random.key(seed)

    def next_key(self) -> Any:
        """Return a fresh key and advance the generator.

        Returns:
            jax.Array: A PRNG key that no other draw has used.
        """
        self._key, key = jax.random.split(self._key)
        return key


def _shape(size: Any) -> tuple[int, ...]:
    """Normalize a NumPy-style ``size`` argument to a shape tuple."""
    if size is None:
        return ()
    if isinstance(size, int):
        return (size,)
    return tuple(int(n) for n in size)


class RandomMixin:
    """Random-number generation operations."""

    _global_generator: JaxGenerator | None = None

    def _generator(self, generator: JaxGenerator | None) -> JaxGenerator:
        """Return the given generator, or the backend's global generator."""
        if generator is not None:
            return generator
        if RandomMixin._global_generator is None:
            RandomMixin._global_generator = JaxGenerator()
        return RandomMixin._global_generator

    def default_rng(self, seed: int | None = None) -> JaxGenerator:
        """Return a random number generator.

        Args:
            seed: Optional seed.

        Returns:
            JaxGenerator: Stateful generator over a JAX PRNG key.
        """
        return JaxGenerator(seed)

    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Any = None,
        generator: JaxGenerator | None = None,
    ) -> Any:
        """Uniform random samples in [low, high).

        Args:
            low: Lower boundary.
            high: Upper boundary.
            size: Output shape.
            generator: Optional generator from ``default_rng``.

        Returns:
            jax.Array: Uniform random samples.
        """
        key = self._generator(generator).next_key()
        return jax.random.uniform(
            key, _shape(size), dtype=self._dtype, minval=low, maxval=high
        )

    def rand(self, *size: int) -> Any:
        """Random values from a uniform distribution on [0, 1).

        Args:
            *size: Shape of the output array.

        Returns:
            jax.Array: Random values.
        """
        key = self._generator(None).next_key()
        return jax.random.uniform(key, size, dtype=self._dtype)

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Any = None,
        generator: JaxGenerator | None = None,
    ) -> Any:
        """Random samples from a Gaussian distribution.

        Args:
            loc: Mean of the distribution.
            scale: Standard deviation.
            size: Output shape.
            generator: Optional generator from ``default_rng``.

        Returns:
            jax.Array: Normal random samples.
        """
        key = self._generator(generator).next_key()
        return jax.random.normal(key, _shape(size), dtype=self._dtype) * scale + loc

    def sobol_sampler(
        self,
        dim: int,
        num_samples: int,
        scramble: bool = True,
        seed: int | None = None,
    ) -> Any:
        """Generate quasi-random samples using Sobol sequences.

        JAX has no Sobol generator; the points come from SciPy, as on the
        NumPy backend.

        Args:
            dim: Dimension of the samples.
            num_samples: Number of samples to generate.
            scramble: Whether to scramble the sequence.
            seed: Random seed for scrambling.

        Returns:
            jax.Array: Samples of shape (num_samples, dim).
        """
        from scipy.stats import qmc

        if num_samples > 0:
            num_samples_pow2 = 1 << (num_samples - 1).bit_length()
        else:
            num_samples_pow2 = num_samples
        sobol = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
        samples = sobol.random(n=num_samples_pow2)
        return self.array(samples[:num_samples])

    def erfinv(self, x: Any) -> Any:
        """Inverse error function.

        Args:
            x: Input array.

        Returns:
            jax.Array: Inverse error function of x.
        """
        return _erfinv(self.array(x))
