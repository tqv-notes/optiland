"""Smoke test: trace a Cooke triplet on numpy and jax and compare."""

import traceback

import numpy as np

import optiland.backend as be
from optiland.samples.objectives import CookeTriplet

print("backends:", be.list_available_backends())


def trace(backend):
    be.set_backend(backend)
    lens = CookeTriplet()
    rays = lens.trace(Hx=0, Hy=0.7, wavelength=0.55, num_rays=32, distribution="hexapolar")
    return {k: be.to_numpy(getattr(rays, k)) for k in ("x", "y", "z", "L", "M", "N", "opd", "i")}


ref = trace("numpy")
try:
    got = trace("jax")
    for k in ref:
        print(k, type(got[k]), got[k].dtype, float(np.nanmax(np.abs(got[k] - ref[k]))))
except Exception:
    traceback.print_exc()
finally:
    be.set_backend("numpy")
