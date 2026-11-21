"""Optimize a 7 mm F/4 110-degree wide-angle lens, following a Zemax blog post.

This script repeats in optiland the steps of "Lens Design Process with Zemax:
Optimization Part 1" (opticalensdesign.com). It starts from the starting
point 7mmF4_110deg_11ELT_01.zmx of "Selection of a Starting Point Part II"
and arrives at a design similar to the blog's 7mmF4_110deg_11ELT_04b.zmx:

1. Load the starting point, with its glasses from the CDGM catalog.
2. Add size limits to the merit function. A ray may not pass any surface
   higher than 18 mm, the blog's maximum semi-diameter (MXSD). The blog also
   limits the mass (TMAS); optiland has no mass operand, so that limit is
   left out.
3. Cement the two elements around the 0.2 mm air gap at surfaces 7/8 into a
   doublet (the blog deletes row 7 of the lens data editor), and run a local
   optimization (damped least squares).
4. Let every glass be substituted, as Zemax's Hammer does with "Substitute"
   glass solves: optiland's GlassExpert tries other CDGM glasses one element
   at a time and re-optimizes the lens for each.
5. Compare the result with the starting point and with the blog's _04b.
6. Re-optimize the blog's _04b with the same merit function.

What to expect: steps 3 and 4 improve the lens, but they end in a local
minimum near the starting point (RMS spots of about 3-4 um on axis and
6 um in the corner). The blog's overnight Hammer run jumped to a lens of a
different shape (two doublets bent the other way), which is a deeper valley
of the same merit function: step 6 shows that, re-optimized by optiland,
_04b reaches about 1.3 um on axis and 3.7 um in the corner. Reaching that
valley from _01 needs a global search of many thousand trials, which is not
practical with a 0.25 s merit evaluation (robust ray aiming of a 110-degree
lens) and is not attempted here.

The merit function is the RMS spot radius from the centroid over all fields
and wavelengths (Zemax's default merit function, 3 rings and 6 arms), plus
the focal length (7 mm), the image height at 55 degrees (8 mm), the total
track (52 mm) and the semi-diameter limit. Radii and thicknesses are
variables: glass between 0.8 and 5 mm, air at least 0.2 mm, and the back
focal distance stays at 5 mm as in both blog files.

Run time is about 45 minutes on a desktop CPU, mostly step 4 (about 30 s
per candidate glass; ``--neighbours`` sets how many glasses are tried per
element and pass). ``--quick`` runs a short version (about 25 minutes),
``--skip-hammer`` leaves out step 4 (about 10 minutes).

The glasses are taken from optiland's CDGM catalog by name; the n_d/V_d
stored in the .zmx files are not used (see ``load_lens``).

Usage::

    python demo/optimization_tutorial.py
    python demo/optimization_tutorial.py --quick
    python demo/optimization_tutorial.py --neighbours 5 --hammer-evals 200
    python demo/optimization_tutorial.py --skip-hammer
"""

from __future__ import annotations

import argparse
import re
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np

import optiland.backend as be
from optiland import optimization
from optiland.analysis import SpotDiagram
from optiland.fileio import load_zemax_file, save_optiland_file, save_zemax_file
from optiland.materials import Material, glasses_selection
from optiland.mtf import FFTMTF

INPUT = Path(__file__).parent / "input" / "zemax_files"
EFL = 7.0  # mm, target focal length
IMAGE_HEIGHT = 8.0  # mm, chief ray height at the edge of the field
TOTAL_TRACK = 52.0  # mm
MAX_SEMI_DIAMETER = 18.0  # mm, the blog's MXSD limit
GLASS_THICKNESS = (0.8, 5.0)  # mm, centre thickness of a lens element
AIR_THICKNESS = (0.2, 12.0)  # mm, centre thickness of an air space
MTF_SPEC = 145.0  # cycles/mm, the blog's "MTF > 20 % at 145 cyc/mm"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--start", type=Path,
                    default=INPUT / "7mmF4_110deg_11ELT_01.zmx",
                    help="starting point (.zmx)")
parser.add_argument("--reference", type=Path,
                    default=INPUT / "7mmF4_110deg_11ELT_04b.zmx",
                    help="the blog's result, for comparison (.zmx; optional)")
parser.add_argument("--local-evals", type=int, default=3000,
                    help="merit evaluations for each local optimization")
parser.add_argument("--neighbours", type=int, default=2,
                    help="candidate glasses per element and pass in step 4")
parser.add_argument("--hammer-evals", type=int, default=150,
                    help="merit evaluations per candidate glass in step 4")
parser.add_argument("--skip-hammer", action="store_true",
                    help="stop after the local optimization (step 3)")
parser.add_argument("--quick", action="store_true",
                    help="short run: 400 local evals, 2 neighbours, 80 per glass")
parser.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "output" / "optimization",
                    help="folder for the figures and the optimized lens")
parser.add_argument("--show", action="store_true", help="also show the figures")
args = parser.parse_args()
if args.quick:
    args.local_evals, args.neighbours, args.hammer_evals = 400, 2, 80
if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", message=".*is not in the GCAT catalogs.*")
warnings.filterwarnings("ignore", message=".*is not in the material database.*")
warnings.filterwarnings("ignore", message=".*via fuzzy match.*")
args.out.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    """Print a section title."""
    print(f"\n{title}\n{'-' * len(title)}")


def save(fig: Any, name: str) -> None:
    """Save a figure to the output folder, and close it unless --show."""
    path = args.out / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  saved {path}")
    if not args.show:
        plt.close(fig)


# ----------------------------------------------------------------------------
# Loading a Zemax file with its CDGM glasses
# ----------------------------------------------------------------------------


def glass_names(path: Path) -> dict[int, str]:
    """Return {surface: glass name} from the GLAS lines of a .zmx file."""
    raw = path.read_bytes()
    text = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "latin-1")
    names, surface = {}, None
    for line in text.splitlines():
        if match := re.match(r"SURF (\d+)", line):
            surface = int(match.group(1))
        elif match := re.match(r"\s+GLAS (\S+)", line):
            names[surface] = match.group(1)
    return names


def load_lens(path: Path) -> Any:
    """Load a .zmx file and take every glass from optiland's CDGM catalog.

    The files use Zemax's CDGM-202206 catalog, which the importer does not
    map to optiland's database; it then models each glass from the n_d and
    V_d stored on its GLAS line. Those values can be stale: in the blog's
    _04b file they belong to other glasses (Hammer renamed the glasses but
    did not rewrite them), and the lens comes out with a 5 mm focal length.
    The glass name is what Zemax uses, so the glasses are looked up by name.
    """
    lens = load_zemax_file(str(path))
    for surface, name in glass_names(path).items():
        lens.updater.set_material(Material(name, "cdgm"), surface)
    lens.ray_tracer.set_aiming("robust", cache=True)
    return lens


# ----------------------------------------------------------------------------
# Performance summary
# ----------------------------------------------------------------------------


def chief_ray_height(lens: Any) -> float:
    """Image height of the chief ray at the edge of the field [mm]."""
    rays = lens.trace_generic(0.0, 1.0, 0.0, 0.0, lens.primary_wavelength)
    return float(be.to_numpy(rays.y).ravel()[0])


def rms_spots(lens: Any) -> np.ndarray:
    """Polychromatic RMS spot radius per field [um], centred on the chief ray."""
    spot = SpotDiagram(lens, num_rings=12)
    rms = np.array([[float(r) for r in f] for f in spot.rms_spot_radius()]) * 1e3
    return np.sqrt(np.mean(rms**2, axis=1))


def mtf_at(lens: Any, freq: float) -> np.ndarray:
    """Diffraction MTF at one frequency per field, [tangential, sagittal]."""
    mtf = FFTMTF(lens, wavelength=lens.primary_wavelength, num_rays=128,
                 grid_size=512, max_freq=freq * 1.1)
    out = []
    for k, (tang, sag) in enumerate(mtf.mtf):
        out.append([
            np.interp(freq, be.to_numpy(mtf.freq_tang[k]), be.to_numpy(tang)),
            np.interp(freq, be.to_numpy(mtf.freq_sag[k]), be.to_numpy(sag)),
        ])
    return np.array(out)


def summary(lens: Any) -> dict[str, Any]:
    """Collect the numbers compared at the end."""
    return {
        "EFL [mm]": float(be.to_numpy(lens.paraxial.f2())),
        "image height [mm]": chief_ray_height(lens),
        "total track [mm]": float(be.to_numpy(lens.total_track)),
        "rms": rms_spots(lens),
        "mtf": mtf_at(lens, MTF_SPEC),
    }


def print_summary(lens: Any, label: str) -> dict[str, Any]:
    """Print the spot sizes and first-order data of a lens; return them."""
    data = summary(lens)
    fields = [f.y for f in lens.fields]
    print(f"{label}: EFL {data['EFL [mm]']:.3f} mm, image height "
          f"{data['image height [mm]']:.3f} mm, total track "
          f"{data['total track [mm]']:.3f} mm")
    print(f"  {'field':>8}{'RMS spot [um]':>15}{'MTF T @145':>12}{'MTF S @145':>12}")
    for field, rms, (t, s) in zip(fields, data["rms"], data["mtf"], strict=True):
        print(f"  {field:>8.2f}{rms:>15.2f}{t:>12.3f}{s:>12.3f}")
    return data


def is_glass(lens: Any, surface: int) -> bool:
    """True if the space after a surface is glass (not air)."""
    n = be.to_numpy(lens.surfaces[surface].material_post.n(lens.primary_wavelength))
    return float(np.ravel(n)[0]) > 1.01


def build_problem(lens: Any) -> Any:
    """Merit function and variables of the optimization (steps 2 to 6)."""
    image = lens.surfaces.num_surfaces - 1
    stop = lens.surfaces.stop_index
    wavelength = lens.primary_wavelength
    problem = optimization.OptimizationProblem()
    # Zemax's default merit function: RMS spot from the centroid, Gaussian
    # quadrature over the pupil (3 rings, 6 arms), all fields and wavelengths.
    problem.add_default_merit(lens, kind="spot", rings=3, arms=6,
                              reference="centroid")
    problem.add_operand("f2", target=EFL, weight=10, input_data={"optic": lens})
    problem.add_operand("total_track", target=TOTAL_TRACK, weight=1,
                        input_data={"optic": lens})
    problem.add_operand(
        "real_y_intercept", target=IMAGE_HEIGHT, weight=1,
        input_data={"optic": lens, "surface_number": image, "Hx": 0, "Hy": 1,
                    "Px": 0, "Py": 0, "wavelength": wavelength},
    )
    # Maximum semi-diameter: the upper and lower marginal rays of the edge
    # field, which set the clear aperture of a wide-angle lens, stay within
    # 18 mm on every surface.
    for surface in range(2, image):
        for py in (-1, 1):
            problem.add_operand(
                "real_y_intercept", min_val=-MAX_SEMI_DIAMETER,
                max_val=MAX_SEMI_DIAMETER, weight=1,
                input_data={"optic": lens, "surface_number": surface, "Hx": 0,
                            "Hy": 1, "Px": 0, "Py": py, "wavelength": wavelength},
            )
    for surface in range(2, image):
        if surface != stop:
            problem.add_variable(lens, "radius", surface_number=surface)
    for surface in range(2, image - 1):  # the back focal distance stays fixed
        lo, hi = GLASS_THICKNESS if is_glass(lens, surface) else AIR_THICKNESS
        problem.add_variable(lens, "thickness", surface_number=surface,
                             min_val=lo, max_val=hi)
    return problem


# ----------------------------------------------------------------------------
# 1. Starting point
# ----------------------------------------------------------------------------

section("1. Starting point")
lens = load_lens(args.start)
print(f"loaded {args.start.name}: {lens.surfaces.num_surfaces - 2} surfaces, "
      f"F/{float(be.to_numpy(lens.paraxial.FNO())):g}, "
      f"fields {[round(f.y, 2) for f in lens.fields]} deg")
start = print_summary(lens, "starting point")
fig, _ = lens.draw(num_rays=5, figsize=(10, 4), title="Starting point (_01)")
save(fig, "1_layout_start")

# ----------------------------------------------------------------------------
# 2./3. Size limits, cementing the doublet, local optimization
# ----------------------------------------------------------------------------

section("2. Cementing surfaces 7/8 into a doublet")
gap = float(be.to_numpy(lens.surfaces[7].thickness))
lens.surfaces.remove(7)  # the air gap after surface 7 disappears with it
print(f"removed surface 7 (and its {gap:.3f} mm air gap); "
      f"{lens.surfaces.num_surfaces - 2} surfaces remain")
print("RMS spot right after cementing [um]:", np.round(rms_spots(lens), 2))

image = lens.surfaces.num_surfaces - 1
wavelength = lens.primary_wavelength
problem = build_problem(lens)
print(f"merit function: {problem.num_residuals} residuals; "
      f"{len(problem.variables)} variables (radii and thicknesses)")

section("3. Local optimization (damped least squares)")
t0 = time.perf_counter()
result = optimization.minimize(problem, "dls", maxiter=args.local_evals,
                               disp=True, aim_policy="frozen")
print(f"{time.perf_counter() - t0:.0f} s: {result.message}")
local = print_summary(lens, "after cementing + local optimization")
local["merit"] = float(problem.sum_squared())
fig, _ = lens.draw(num_rays=5, figsize=(10, 4), title="Cemented, local optimization")
save(fig, "2_layout_local")
save_optiland_file(lens, str(args.out / "7mmF4_local.json"))

# ----------------------------------------------------------------------------
# 4. Glass substitution (Zemax: Hammer with "Substitute" glass solves)
# ----------------------------------------------------------------------------


class DLSGlassExpert(optimization.GlassExpert):
    """GlassExpert with damped least squares for each local optimization.

    GlassExpert re-optimizes the lens for every candidate glass with SciPy's
    L-BFGS-B, which takes about 10 s per step on this lens with finite
    differences. Optiland's damped least squares reaches the same merit in a
    few steps, so it replaces the inner optimizer; the glass search itself
    (a broad sample of the catalog, then the nearest neighbours of each
    selected glass) is GlassExpert's.
    """

    evals = 150

    def optimize(self, **_: Any) -> Any:
        """Run one damped least squares optimization of the current glasses."""
        try:
            return optimization.minimize(self.problem, "dls", maxiter=self.evals,
                                         aim_policy="frozen")
        except (ValueError, RuntimeError) as exc:  # a glass the rays cannot pass
            print(f"(failed: {exc}) ", end="")
            return SimpleNamespace(
                fun=float("inf"), x=[v.value for v in self.problem.variables]
            )


if not args.skip_hammer:
    section("4. Glass substitution (Hammer)")
    # Standard CDGM glasses only: the D- glasses are for precision moulding,
    # and -25 and GT are variants (annealing, transmission) of other glasses.
    glasses = [
        g for g in glasses_selection(0.45, 0.65, catalogs=["cdgm"])
        if not (g.startswith("D-") or g.endswith(("-25", "GT")))
    ]
    glass_surfaces = [s for s in range(2, image) if is_glass(lens, s)]
    for surface in glass_surfaces:
        problem.add_variable(lens, "material", surface_number=surface,
                             glass_selection=glasses)
    print(f"{len(glass_surfaces)} glasses may change, among {len(glasses)} CDGM "
          f"glasses; {args.neighbours} candidates per glass and pass, "
          f"{args.hammer_evals} merit evaluations each")
    t0 = time.perf_counter()
    hammer = DLSGlassExpert(problem)
    hammer.evals = args.hammer_evals
    hammer.run(num_neighbours=args.neighbours, verbose=True)
    # The final polish of GlassExpert used `evals`; finish with a full run.
    optimization.minimize(problem, "dls", maxiter=args.local_evals,
                          aim_policy="frozen")
    print(f"glass substitution: {(time.perf_counter() - t0) / 60:.1f} min")
    print("glasses:", ", ".join(
        f"{s}: {lens.surfaces[s].material_post.name}" for s in glass_surfaces))

# ----------------------------------------------------------------------------
# 5. Result, compared with the starting point and the blog's _04b
# ----------------------------------------------------------------------------

section("5. Result")
final = print_summary(lens, "optimized")
fig, _ = lens.draw(num_rays=5, figsize=(10, 4), title="Optimized (this script)")
save(fig, "3_layout_final")
spot = SpotDiagram(lens, num_rings=12)
fig, _ = spot.view(show=False)
fig.suptitle("Spot diagram, optimized", y=1.04)
save(fig, "4_spot_final")
mtf = FFTMTF(lens, wavelength=wavelength, num_rays=128, grid_size=512,
             max_freq=160)
fig, _ = mtf.view(add_reference=True)
fig.suptitle(f"Diffraction MTF, optimized, {wavelength} um", y=1.02)
save(fig, "5_mtf_final")
save_optiland_file(lens, str(args.out / "7mmF4_optimized.json"))
save_zemax_file(lens, str(args.out / "7mmF4_optimized.zmx"))
print(f"  saved the lens as {args.out / '7mmF4_optimized.json'} and .zmx")

final["merit"] = float(problem.sum_squared())
columns = {"start (_01)": start, "local": local, "optimized": final}

# ----------------------------------------------------------------------------
# 6. The blog's _04b, under the same merit function
# ----------------------------------------------------------------------------
# The overnight Hammer run of the blog found a lens of a different shape: the
# cemented doublet and the doublet before the stop are bent the other way, and
# the air space behind the first doublet is 8.6 mm instead of 4.6 mm. Local
# optimization from _01 cannot get there; it is a different valley of the
# merit function. Re-optimizing _04b with this script's merit function shows
# that optiland finds that valley's bottom too, with the blog's glasses.
if args.reference.exists():
    section("6. The blog's _04b under the same merit function")
    reference = load_lens(args.reference)
    columns["blog _04b"] = summary(reference)
    ref_problem = build_problem(reference)
    columns["blog _04b"]["merit"] = float(ref_problem.sum_squared())
    fig, _ = reference.draw(num_rays=5, figsize=(10, 4), title="Blog result (_04b)")
    save(fig, "6_layout_blog_04b")
    optimization.minimize(ref_problem, "dls", maxiter=args.local_evals,
                          aim_policy="frozen")
    columns["_04b refined"] = print_summary(reference, "_04b re-optimized")
    columns["_04b refined"]["merit"] = float(ref_problem.sum_squared())
    save_optiland_file(reference, str(args.out / "7mmF4_04b_refined.json"))
start["merit"] = float("nan")  # not cemented yet: another merit function

print("\nPolychromatic RMS spot radius [um] (chief-ray reference):")
print(f"{'field':>8}" + "".join(f"{name:>14}" for name in columns))
for i, field in enumerate(lens.fields):
    print(f"{field.y:>8.2f}" + "".join(f"{c['rms'][i]:>14.2f}"
                                       for c in columns.values()))
print(f"\nMTF at {MTF_SPEC:g} cycles/mm, tangential / sagittal "
      f"(blog spec: > 0.20):")
print(f"{'field':>8}" + "".join(f"{name:>14}" for name in columns))
for i, field in enumerate(lens.fields):
    print(f"{field.y:>8.2f}" + "".join(
        f"{c['mtf'][i][0]:>7.2f}/{c['mtf'][i][1]:<6.2f}" for c in columns.values()))
print()
for key in ("EFL [mm]", "image height [mm]", "total track [mm]"):
    print(f"{key:>20}" + "".join(f"{c[key]:>14.3f}" for c in columns.values()))
print(f"{'merit function':>20}" + "".join(
    f"{c['merit']:>14.2e}" if np.isfinite(c["merit"]) else f"{'-':>14}"
    for c in columns.values()))
print(f"\nFigures and files are in {args.out}")
