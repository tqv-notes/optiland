"""
JAX backend -- interpolation, polynomial, and signal-processing operations.

Provides InterpolationMixin, one of the mixins composed into
JaxBackend (see optiland/backend/jax_backend/__init__.py).
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp


def _reflect(coordinate: Any, size: int, align_corners: bool) -> Any:
    """Reflect pixel coordinates into the image, as torch's grid_sample does.

    With ``align_corners`` the mirrors are the centers of the edge pixels
    (0 and size - 1); without, they are the outer pixel edges (-0.5 and
    size - 0.5).
    """
    low, span = (0.0, float(size - 1)) if align_corners else (-0.5, float(size))
    if span == 0:
        return jnp.zeros_like(coordinate)
    distance = jnp.abs(coordinate - low)
    flips = jnp.floor(distance / span)
    extra = distance - flips * span
    return jnp.where(flips % 2 == 0, extra + low, span - extra + low)


class InterpolationMixin:
    """Interpolation, polynomial, and signal-processing operations."""

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def nearest_nd_interpolator(self, points: Any, values: Any, x: Any, y: Any) -> Any:
        """Nearest-neighbour interpolation on a 2-D dataset.

        Args:
            points: Known sample points of shape (N, 2).
            values: Values at the sample points.
            x: Query x coordinates.
            y: Query y coordinates.

        Returns:
            jax.Array: Interpolated values.

        Raises:
            ValueError: If x or y is None.
        """
        if x is None or y is None:
            raise ValueError("Hx and Hy must be provided")
        x, y = jnp.broadcast_arrays(self.array(x), self.array(y))
        queries = jnp.stack([x, y], axis=-1).reshape(-1, 2)
        points = self.array(points)
        delta = queries[:, None, :] - points[None, :, :]
        nearest = jnp.argmin(jnp.sum(delta * delta, axis=-1), axis=1)
        values = self.array(values).reshape(points.shape[0], -1)
        out = values[nearest].reshape(*x.shape, -1)
        return out[..., 0] if out.shape[-1] == 1 else out

    def interp(self, x: Any, xp: Any, fp: Any) -> Any:
        """1-D linear interpolation.

        Args:
            x: x-coordinates of the interpolated values.
            xp: x-coordinates of the data points.
            fp: y-coordinates of the data points.

        Returns:
            jax.Array: Interpolated values.
        """
        return jnp.interp(self.cast(x), self.cast(xp), self.cast(fp))

    def grid_sample(
        self,
        input: Any,
        grid: Any,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ) -> Any:
        """Sample input using bilinear/nearest interpolation on a grid.

        JAX implementation of ``torch.nn.functional.grid_sample``, written
        with gathers so that it is differentiable in ``input`` and ``grid``.

        Args:
            input: Input array of shape (N, C, H_in, W_in).
            grid: Grid of shape (N, H_out, W_out, 2). Coordinates in [-1, 1].
            mode: Interpolation mode (``'bilinear'`` or ``'nearest'``).
            padding_mode: Padding mode (``'zeros'``, ``'border'``,
                ``'reflection'``).
            align_corners: Whether to align corners.

        Returns:
            jax.Array: Output array of shape (N, C, H_out, W_out).

        Raises:
            ValueError: If the batch sizes differ, or a mode is unknown.
        """
        input, grid = self.array(input), self.array(grid)
        N, _, H_in, W_in = input.shape
        if grid.shape[0] != N:
            raise ValueError("Input and grid must have same batch size")
        if mode not in ("bilinear", "nearest"):
            raise ValueError(f"Unsupported interpolation mode: {mode}")
        if padding_mode not in ("zeros", "border", "reflection"):
            raise ValueError(f"Unsupported padding mode: {padding_mode}")

        x, y = grid[..., 0], grid[..., 1]
        if align_corners:
            x = (x + 1) / 2 * (W_in - 1)
            y = (y + 1) / 2 * (H_in - 1)
        else:
            x = (x + 1) * W_in / 2 - 0.5
            y = (y + 1) * H_in / 2 - 0.5
        if padding_mode == "reflection":
            x = _reflect(x, W_in, align_corners)
            y = _reflect(y, H_in, align_corners)
        if padding_mode != "zeros":
            x = jnp.clip(x, 0, W_in - 1)
            y = jnp.clip(y, 0, H_in - 1)

        batch = jnp.arange(N)[:, None, None]

        def pixel(iy: Any, ix: Any) -> Any:
            """Gather input[n, :, iy, ix] as (N, C, H_out, W_out); 0 outside."""
            inside = (ix >= 0) & (ix < W_in) & (iy >= 0) & (iy < H_in)
            picked = input[
                batch, :, jnp.clip(iy, 0, H_in - 1), jnp.clip(ix, 0, W_in - 1)
            ]
            picked = jnp.where(inside[..., None], picked, 0)
            return jnp.moveaxis(picked, -1, 1)

        if mode == "nearest":
            return pixel(jnp.round(y).astype(int), jnp.round(x).astype(int))

        x0, y0 = jnp.floor(x), jnp.floor(y)
        wx, wy = (x - x0)[:, None], (y - y0)[:, None]
        ix, iy = x0.astype(int), y0.astype(int)
        return (
            pixel(iy, ix) * (1 - wx) * (1 - wy)
            + pixel(iy, ix + 1) * wx * (1 - wy)
            + pixel(iy + 1, ix) * (1 - wx) * wy
            + pixel(iy + 1, ix + 1) * wx * wy
        )

    # ------------------------------------------------------------------
    # Polynomial
    # ------------------------------------------------------------------

    def polyfit(self, x: Any, y: Any, degree: int) -> Any:
        """Least-squares polynomial fit.

        Args:
            x: x-coordinates of the sample points.
            y: y-coordinates of the sample points.
            degree: Degree of the polynomial.

        Returns:
            jax.Array: Polynomial coefficients, highest power first.
        """
        return jnp.polyfit(self.array(x), self.array(y), degree)

    def polyval(self, coeffs: Any, x: Any) -> Any:
        """Evaluate a polynomial at specific values.

        Args:
            coeffs: Polynomial coefficients, highest power first.
            x: Values at which to evaluate the polynomial.

        Returns:
            jax.Array: Evaluated polynomial.
        """
        return jnp.polyval(self.array(coeffs), self.array(x))

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def fftconvolve(
        self,
        in1: Any,
        in2: Any,
        mode: Literal["full", "valid", "same"] = "full",
    ) -> Any:
        """FFT-based convolution.

        Args:
            in1: First input array.
            in2: Second input array.
            mode: Convolution mode (``'full'``, ``'valid'``, ``'same'``).

        Returns:
            jax.Array: Convolved array.

        Raises:
            ValueError: If the inputs differ in dimensionality, or the mode
                is unknown.
        """
        a = self.array(in1)
        b = self.array(in2)
        if a.ndim != b.ndim:
            raise ValueError("Inputs must have the same dimensionality.")
        if mode not in ("full", "same", "valid"):
            raise ValueError(f"Unknown mode: {mode}")
        # Convolve over the last two axes and broadcast over the others, as
        # SciPy does. jax.scipy.signal.fftconvolve cannot broadcast them.
        axes = tuple(range(a.ndim)) if a.ndim < 2 else (-2, -1)
        size = [a.shape[axis] + b.shape[axis] - 1 for axis in axes]
        spectrum = jnp.fft.fftn(a, s=size, axes=axes) * jnp.fft.fftn(
            b, s=size, axes=axes
        )
        full = jnp.fft.ifftn(spectrum, s=size, axes=axes)
        if not (jnp.iscomplexobj(a) or jnp.iscomplexobj(b)):
            full = full.real
        if mode == "full":
            return full
        crop = [slice(None)] * full.ndim
        for axis in axes:
            if mode == "same":
                start = (b.shape[axis] - 1) // 2
                crop[axis] = slice(start, start + a.shape[axis])
            else:
                crop[axis] = slice(b.shape[axis] - 1, a.shape[axis])
        return full[tuple(crop)]
