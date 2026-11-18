"""Tests of the behavior that is specific to the JAX backend.

The contract shared by all backends is tested, for JAX too, by the tests
that use the ``set_test_backend`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.samples.objectives import CookeTriplet
from tests.utils import assert_allclose

jax = pytest.importorskip("jax")
jnp = jax.numpy


@pytest.fixture(autouse=True)
def _jax_backend():
    """Run each test on the JAX backend in float64, then restore NumPy."""
    be.set_backend("jax")
    be.set_precision("float64")
    be.grad_mode.disable()
    yield
    be.set_backend("jax")  # a test may have switched away
    be.set_precision("float64")
    be.grad_mode.disable()
    be.set_backend("numpy")


def _rms_spot(params):
    """RMS spot radius of a Cooke triplet as a function of three parameters."""
    radius, thickness, conic = params
    lens = CookeTriplet()
    lens.updater.set_radius(radius, 1)
    lens.updater.set_thickness(thickness, 2)
    lens.updater.set_conic(conic, 3)
    rays = lens.trace(
        Hx=0, Hy=0.7, wavelength=0.55, num_rays=8, distribution="hexapolar"
    )
    cx, cy = be.mean(rays.x), be.mean(rays.y)
    return be.sqrt(be.mean((rays.x - cx) ** 2 + (rays.y - cy) ** 2))


class TestRegistration:
    def test_is_registered(self):
        assert "jax" in be.list_available_backends()
        assert be.get_backend() == "jax"
        assert be.name == "jax"

    def test_arrays_are_jax_arrays(self):
        x = be.array([1.0, 2.0])
        assert isinstance(x, jax.Array)
        assert isinstance(x, be.ndarray)
        assert be.is_jax_array(x)
        assert not be.is_jax_array(np.ones(2))

    def test_array_keeps_an_existing_array(self):
        indices = be.arange_indices(3)
        assert be.array(indices) is indices
        assert be.array(np.array([True, False])).dtype == bool

    def test_list_arguments_are_accepted(self):
        # jax.numpy rejects lists; the NumPy backend accepts them.
        assert_allclose(be.sin([0.0, np.pi / 2]), [0.0, 1.0])
        assert_allclose(be.hypot([3.0], [4.0]), [5.0])


class TestPrecision:
    def test_default_is_float64(self):
        assert be.get_precision() == 64
        assert be.array([1.0]).dtype == np.float64
        assert be.get_complex_precision() == np.complex128

    def test_float32(self):
        be.set_precision("float32")
        assert be.get_precision() == 32
        assert be.array([1.0]).dtype == np.float32
        assert be.zeros((2,)).dtype == np.float32
        assert be.get_complex_precision() == np.complex64

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError, match="Precision"):
            be.set_precision("float16")


class TestCapabilities:
    def test_flags(self):
        assert be.supports_gradients is True
        assert be.supports_gpu is (jax.default_backend() != "cpu")

    def test_grad_mode_has_the_torch_interface(self):
        assert be.grad_mode.requires_grad is False
        be.grad_mode.enable()
        assert be.grad_mode.requires_grad is True
        be.grad_mode.disable()
        with be.grad_mode.temporary_enable():
            assert be.grad_mode.requires_grad is True
        assert be.grad_mode.requires_grad is False

    def test_autograd_is_jax(self):
        assert be.autograd.grad(lambda x: x**2)(3.0) == pytest.approx(6.0)

    def test_device(self):
        assert be.get_device() == jax.default_backend()
        with pytest.raises(ValueError, match="Device"):
            be.set_device("tpu0")

    def test_no_in_place_copy(self):
        # JAX arrays are immutable, so the backend offers no copy_to.
        assert not hasattr(be, "copy_to")

    def test_torch_only_helpers_are_absent(self):
        with pytest.raises(AttributeError, match="jax"):
            _ = be.nn


class TestToNumpy:
    def test_concrete_array(self):
        out = be.to_numpy(be.array([1.0, 2.0]))
        assert isinstance(out, np.ndarray)
        assert out.tolist() == [1.0, 2.0]

    def test_value_inside_grad(self):
        seen = []

        def f(x):
            seen.append(be.to_numpy(x))
            return be.sum(x**2)

        jax.grad(f)(be.array([1.0, 2.0]))
        assert seen[0].tolist() == [1.0, 2.0]

    def test_value_inside_jit_raises(self):
        with pytest.raises(be.BackendCapabilityError, match="jax.jit"):
            jax.jit(be.to_numpy)(be.array([1.0]))


class TestRandom:
    def test_seed_reproduces(self):
        a = be.random_uniform(size=5, generator=be.default_rng(7))
        b = be.random_uniform(size=5, generator=be.default_rng(7))
        assert_allclose(a, b, atol=0)

    def test_generator_advances(self):
        rng = be.default_rng(7)
        a = be.random_normal(size=5, generator=rng)
        b = be.random_normal(size=5, generator=rng)
        assert not np.allclose(be.to_numpy(a), be.to_numpy(b))

    def test_shapes_and_bounds(self):
        x = be.random_uniform(2.0, 3.0, size=(4, 5), generator=be.default_rng(0))
        assert x.shape == (4, 5)
        assert float(be.min(x)) >= 2.0
        assert float(be.max(x)) < 3.0
        assert be.rand(3, 2).shape == (3, 2)

    def test_sobol(self):
        samples = be.sobol_sampler(2, 16, scramble=False)
        assert samples.shape == (16, 2)
        assert isinstance(samples, jax.Array)


class TestNumerics:
    def test_index_add_accumulates_repeated_indices(self):
        out = be.index_add(
            be.zeros((3,)), 0, be.arange_indices(0, 1).repeat(2), be.ones((2,))
        )
        assert_allclose(out, [2.0, 0.0, 0.0])

    def test_cholesky_raises_for_an_indefinite_matrix(self):
        with pytest.raises(be.linalg.LinAlgError):
            be.linalg.cholesky(be.array([[1.0, 2.0], [2.0, 1.0]]))

    @pytest.mark.parametrize("mode", ["bilinear", "nearest"])
    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    @pytest.mark.parametrize("align_corners", [True, False])
    def test_grid_sample_matches_torch(self, mode, padding_mode, align_corners):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(0)
        image = rng.normal(size=(2, 3, 5, 7))
        grid = rng.uniform(-1.4, 1.4, size=(2, 4, 6, 2))
        expected = torch.nn.functional.grid_sample(
            torch.tensor(image),
            torch.tensor(grid),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        ).numpy()
        out = be.grid_sample(
            be.array(image),
            be.array(grid),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        assert_allclose(out, expected, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("mode", ["full", "same", "valid"])
    def test_fftconvolve_matches_scipy_and_broadcasts(self, mode):
        from scipy.signal import fftconvolve

        rng = np.random.default_rng(0)
        images = rng.normal(size=(2, 9, 8))
        kernel = rng.normal(size=(1, 4, 3))
        expected = fftconvolve(images, kernel, mode=mode, axes=(-2, -1))
        out = be.fftconvolve(be.array(images), be.array(kernel), mode=mode)
        assert_allclose(out, expected, rtol=1e-10, atol=1e-12)

    def test_full_like_takes_a_one_element_fill_value(self):
        out = be.full_like(be.array(3.0), be.array([2.5]))
        assert out.shape == ()
        assert float(out) == 2.5

    def test_path_contains_points_with_a_horizontal_edge(self):
        square = be.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        points = be.array([[0.5, 0.5], [1.5, 0.5], [0.5, -0.5]])
        mask = be.path_contains_points(square, points)
        assert be.to_numpy(mask).tolist() == [True, False, False]


class TestConicGradient:
    def test_derivative_of_the_intersection_distance(self):
        def distance(radius):
            zero, one = be.zeros((1,)), be.ones((1,))
            return be.conic_intersection(
                zero, be.array([0.2]), -one, zero, zero, one, radius, 0.0
            )[0]

        # t = 1 + R - sqrt(R**2 - y**2), so dt/dR = 1 - R / sqrt(R**2 - y**2).
        radius = 2.0
        expected = 1 - radius / np.sqrt(radius**2 - 0.2**2)
        assert jax.grad(distance)(radius) == pytest.approx(expected, rel=1e-12)

    def test_a_miss_has_a_finite_zero_gradient(self):
        def distance(radius):
            zero, one = be.zeros((1,)), be.ones((1,))
            return be.conic_intersection(
                zero, be.array([5.0]), -one, zero, zero, one, radius, 0.0
            )[0]

        assert np.isnan(float(distance(2.0)))
        assert float(jax.grad(distance)(2.0)) == 0.0


class TestRayTrace:
    def test_trace_matches_numpy(self):
        def trace():
            rays = CookeTriplet().trace(
                Hx=0, Hy=0.7, wavelength=0.55, num_rays=8, distribution="hexapolar"
            )
            return [be.to_numpy(getattr(rays, name)) for name in "xyzLMN"]

        traced = trace()
        be.set_backend("numpy")
        for got, expected in zip(traced, trace(), strict=True):
            assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_grad_matches_finite_differences(self):
        x0 = be.array([22.01359, 6.07, 0.0])
        gradient = be.to_numpy(jax.grad(_rms_spot)(x0))
        step = 1e-6
        for i in range(3):
            high = float(_rms_spot(x0.at[i].add(step)))
            low = float(_rms_spot(x0.at[i].add(-step)))
            assert gradient[i] == pytest.approx((high - low) / (2 * step), rel=1e-6)
