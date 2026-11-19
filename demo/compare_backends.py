"""Compare optiland's NumPy, PyTorch and JAX backends on a Cooke triplet.

Three parts, each printed as a table:

1. Accuracy: real ray traces over several fields and wavelengths, and the
   paraxial focal length, compared element by element with NumPy.
2. Speed: wall time of a real ray trace for growing numbers of rays.
3. Gradient: d(RMS spot radius)/d(radii of surfaces 1, 3, 5), from torch
   autograd, ``jax.grad`` and NumPy central differences, with their timings.

By default everything runs on the CPU in float64. ``--device gpu`` runs
torch and jax on the first CUDA GPU (NumPy stays on the CPU as the
reference). A backend that is not installed is skipped; an error in one part
is printed and the script goes on.

Usage::

    python demo/compare_backends.py
    python demo/compare_backends.py --sizes 32 128 512 --repeats 7
    python demo/compare_backends.py --device gpu --sizes 128 512 1024 2048
    python demo/compare_backends.py --device gpu --precision float32
"""

from __future__ import annotations

import argparse
import statistics
import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np

import optiland.backend as be
from optiland.samples.objectives import CookeTriplet

RAY_FIELDS = ("x", "y", "z", "L", "M", "N", "opd")
FIELDS = (0.0, 0.7, 1.0)  # normalized field heights Hy
WAVELENGTHS = (0.48, 0.55, 0.65)  # micrometres
GRAD_SURFACES = (1, 3, 5)  # surfaces whose radius the gradient is taken for


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


# Set from the command line in main(). NumPy is the reference: it always runs
# on the CPU in float64.
SETTINGS = {"device": "cpu", "precision": "float64"}


def use_backend(name: str) -> None:
    """Select a backend with the same settings for every comparison."""
    be.set_backend(name)
    be.set_precision("float64" if name == "numpy" else SETTINGS["precision"])
    gpu = SETTINGS["device"] == "gpu"
    if name == "torch":
        be.set_device("cuda" if gpu else "cpu")
        be.grad_mode.disable()
    elif name == "jax":
        # A CUDA build of JAX defaults to the GPU, so select the CPU explicitly.
        be.set_device("gpu" if gpu else "cpu")


def torch_device() -> str:
    """Return the torch device name for the --device setting."""
    return "cuda" if SETTINGS["device"] == "gpu" else "cpu"


def available_backends() -> list[str]:
    """Return the backends to compare, NumPy first."""
    names = be.list_available_backends()
    return [n for n in ("numpy", "torch", "jax") if n in names]


def timed(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    """Run fn once to warm up, then return the median time of repeats runs.

    The result is converted to NumPy inside the timed region, so that the
    time includes finishing the computation (JAX dispatches asynchronously).
    """
    result = fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def rms_spot(rays: Any) -> Any:
    """RMS spot radius of a traced ray bundle, around its centroid."""
    cx, cy = be.mean(rays.x), be.mean(rays.y)
    return be.sqrt(be.mean((rays.x - cx) ** 2 + (rays.y - cy) ** 2))


def header(title: str) -> None:
    """Print a section title."""
    print(f"\n{title}\n{'=' * len(title)}")


# ----------------------------------------------------------------------------
# 1. Accuracy
# ----------------------------------------------------------------------------


def trace_all(backend: str) -> dict[str, np.ndarray]:
    """Trace every field and wavelength; return the ray data as NumPy."""
    use_backend(backend)
    lens = CookeTriplet()
    out: dict[str, list[np.ndarray]] = {k: [] for k in RAY_FIELDS}
    for hy in FIELDS:
        for wl in WAVELENGTHS:
            rays = lens.trace(
                Hx=0.0, Hy=hy, wavelength=wl, num_rays=16, distribution="hexapolar"
            )
            for k in RAY_FIELDS:
                out[k].append(be.to_numpy(getattr(rays, k)))
    data = {k: np.concatenate(v) for k, v in out.items()}
    data["EFL"] = np.atleast_1d(be.to_numpy(lens.paraxial.f2()))
    return data


def compare_accuracy(backends: list[str]) -> None:
    """Print the largest absolute difference from NumPy per quantity."""
    header("1. Accuracy: max |backend - numpy| over all rays")
    ref = trace_all("numpy")
    print(f"{len(FIELDS)} fields x {len(WAVELENGTHS)} wavelengths, "
          f"{ref['x'].size} rays in total; EFL = {ref['EFL'][0]:.6f} mm")
    others = [b for b in backends if b != "numpy"]
    results = {}
    for b in others:
        try:
            results[b] = trace_all(b)
        except Exception:  # noqa: BLE001
            print(f"\n[{b}] trace failed:")
            traceback.print_exc()
    if not results:
        return
    print(f"\n{'quantity':>10}" + "".join(f"{b:>14}" for b in results))
    for k in (*RAY_FIELDS, "EFL"):
        row = f"{k:>10}"
        for data in results.values():
            diff = float(np.nanmax(np.abs(data[k] - ref[k])))
            row += f"{diff:>14.2e}"
        print(row)


# ----------------------------------------------------------------------------
# 2. Speed
# ----------------------------------------------------------------------------


def compare_speed(backends: list[str], sizes: list[int], repeats: int) -> None:
    """Print the median wall time of one ray trace per backend and size."""
    header(f"2. Speed: median time of one ray trace ({repeats} repeats, "
           f"{SETTINGS['device']})")
    print("distribution='uniform': num_rays=n traces about 0.785*n^2 rays")
    print(f"\n{'num_rays':>9}{'rays':>10}" + "".join(f"{b:>14}" for b in backends))
    for n in sizes:
        row, count = "", None
        for b in backends:
            try:
                use_backend(b)
                lens = CookeTriplet()

                def run(lens: Any = lens, n: int = n) -> np.ndarray:
                    rays = lens.trace(
                        Hx=0.0, Hy=0.7, wavelength=0.55, num_rays=n,
                        distribution="uniform",
                    )
                    return be.to_numpy(rays.x)

                seconds, x = timed(run, repeats)
                count = x.size
                row += f"{seconds * 1e3:>11.1f} ms"
            except Exception as exc:  # noqa: BLE001
                row += f"{'error':>14}"
                print(f"[{b}, n={n}] {type(exc).__name__}: {exc}")
        print(f"{n:>9}{count if count is not None else '?':>10}{row}")


# ----------------------------------------------------------------------------
# 3. Gradient
# ----------------------------------------------------------------------------


def set_radii(lens: Any, radii: Any) -> None:
    """Set the radii of GRAD_SURFACES from a 1-D array."""
    for i, s in enumerate(GRAD_SURFACES):
        lens.updater.set_radius(radii[i], s)


def spot_of(lens: Any, num_rays: int) -> Any:
    """RMS spot radius at 0.7 field, 0.55 um."""
    rays = lens.trace(
        Hx=0.0, Hy=0.7, wavelength=0.55, num_rays=num_rays, distribution="hexapolar"
    )
    return rms_spot(rays)


def grad_numpy(x0: np.ndarray, num_rays: int) -> np.ndarray:
    """Gradient by central differences on the NumPy backend."""
    use_backend("numpy")
    lens = CookeTriplet()
    grad = np.zeros_like(x0)
    for i in range(x0.size):
        h = 1e-6 * max(1.0, abs(x0[i]))
        step = np.eye(x0.size)[i] * h
        set_radii(lens, x0 + step)
        plus = float(spot_of(lens, num_rays))
        set_radii(lens, x0 - step)
        minus = float(spot_of(lens, num_rays))
        grad[i] = (plus - minus) / (2 * h)
    set_radii(lens, x0)
    return grad


def grad_torch(x0: np.ndarray, num_rays: int) -> np.ndarray:
    """Gradient by torch autograd."""
    import torch

    use_backend("torch")
    be.grad_mode.enable()
    try:
        lens = CookeTriplet()
        dtype = getattr(torch, SETTINGS["precision"])
        radii = torch.tensor(
            x0, dtype=dtype, device=torch_device(), requires_grad=True
        )
        set_radii(lens, radii)
        spot_of(lens, num_rays).backward()
        return radii.grad.detach().cpu().numpy().astype(np.float64)
    finally:
        be.grad_mode.disable()


def grad_jax(x0: np.ndarray, num_rays: int) -> np.ndarray:
    """Gradient by ``jax.grad``."""
    import jax

    use_backend("jax")
    lens = CookeTriplet()

    def loss(radii: Any) -> Any:
        set_radii(lens, radii)
        return spot_of(lens, num_rays)

    grad = np.asarray(jax.grad(loss)(be.array(x0)))
    loss(be.array(x0))  # put concrete values back into the lens
    return grad


def compare_gradient(backends: list[str], repeats: int, num_rays: int) -> None:
    """Print the gradient from each method, its error and its time."""
    header("3. Gradient: d(RMS spot)/d(radius), 0.7 field, 0.55 um")
    use_backend("numpy")
    lens = CookeTriplet()
    x0 = np.array([float(be.to_numpy(lens.surfaces[s].geometry.radius))
                   for s in GRAD_SURFACES])
    print(f"radii of surfaces {GRAD_SURFACES}: {x0}")

    methods = {"numpy (central diff.)": grad_numpy}
    if "torch" in backends:
        methods["torch (autograd)"] = grad_torch
    if "jax" in backends:
        methods["jax (jax.grad)"] = grad_jax

    results = {}
    for label, fn in methods.items():
        try:
            seconds, grad = timed(lambda fn=fn: fn(x0, num_rays), repeats)
            results[label] = (seconds, grad)
        except Exception:  # noqa: BLE001
            print(f"\n[{label}] failed:")
            traceback.print_exc()

    ref = results.get("numpy (central diff.)", (None, None))[1]
    print(f"\n{'method':>22}{'time':>12}{'max rel. diff':>15}   gradient")
    for label, (seconds, grad) in results.items():
        if ref is None or label.startswith("numpy"):
            rel = "-"
        else:
            rel = f"{np.max(np.abs(grad - ref) / np.abs(ref)):.2e}"
        print(f"{label:>22}{seconds * 1e3:>9.1f} ms{rel:>15}   "
              + np.array2string(grad, precision=6))
    print("(central differences have a truncation error of about 1e-6.)")


# ----------------------------------------------------------------------------


def main() -> None:
    """Parse the arguments and run the three comparisons."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 128, 512],
                        help="num_rays values for the speed test")
    parser.add_argument("--repeats", type=int, default=5,
                        help="timed runs per measurement (after one warm-up)")
    parser.add_argument("--grad-rays", type=int, default=12,
                        help="hexapolar rings for the gradient test")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="device for torch and jax (numpy is always CPU)")
    parser.add_argument("--precision", choices=["float64", "float32"],
                        default="float64",
                        help="float precision; consumer GPUs are much faster "
                        "in float32")
    args = parser.parse_args()
    SETTINGS["device"] = args.device
    SETTINGS["precision"] = args.precision

    backends = available_backends()
    print("backends compared:", ", ".join(backends))
    print(f"device: {args.device}, precision: {args.precision} "
          "(torch, jax; numpy is the float64 CPU reference)")
    for b in backends:
        try:
            use_backend(b)
            print(f"  {b}: device {be.get_device() if b != 'numpy' else 'cpu'}")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"{b}: cannot use --device {args.device}: {exc}")
    try:
        compare_accuracy(backends)
        compare_speed(backends, args.sizes, args.repeats)
        compare_gradient(backends, args.repeats, args.grad_rays)
    finally:
        be.set_backend("numpy")


if __name__ == "__main__":
    main()
