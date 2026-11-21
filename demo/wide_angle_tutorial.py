"""Tutorial: real ray tracing and analysis of a multi-element wide-angle lens.

This script follows the first steps of a classical lens-design workflow, the
same ones shown in the Zemax screenshots of the blog series "Lens Design
Process with Zemax: Selection of a Starting Point" (opticalensdesign.com):

1. Load a starting-point lens (a published design).
2. Scale it to the focal length you need.
3. Look at the layout and the first-order data.
4. Trace real rays through every surface.
5. Evaluate it: spot diagram, MTF, field curvature, F-tan(theta) distortion.
6. Change the aperture (F-number) and see how the image quality responds,
   as the blog does when it stops the F/1.5 patent lens down to F/4.

Starting-point lens
-------------------
The blog uses US patent 8,416,512 (file ``US08416512-1.ZMX`` from
lens-designs.com). Its prescription is not published on the blog, so it is not
reproduced here. If you have that file, or any other Zemax file, pass it in:

    python demo/wide_angle_tutorial.py --zmx path/to/US08416512-1.ZMX --fnumbers 1.5 4

A .ZMX file names its glasses but does not contain their data. Glasses that
are not in optiland's catalogs get the file's placeholder n = 1.5 (the script
warns about it). Supply the catalogs with ``--agf FOLDER``, or substitute
glasses from optiland's database with ``--glass SURFACE=NAME``. For the patent
file, H-ZLAF53 and LASFN7 are missing:

    python demo/wide_angle_tutorial.py --zmx path/to/US08416512-1.ZMX \
        --glass 2=H-ZLAF53 4=LASFN7 17=LASFN7 --fnumbers 1.5 4

Without ``--zmx`` the script uses optiland's built-in ``WideAngle100FOV``: an
inverse-telephoto (retrofocus) lens with a 100 degree full field at F/4, from
M. Laikin, *Lens Design*, 4th ed., p. 108 -- the same family of design that
Warren Smith's chart recommends for the blog's 110 degree F/4 target.

Output
------
Tables are printed to the terminal; figures are saved as PNG files in
``demo/output/wide_angle`` (``--out`` changes the folder). ``--show`` also
opens them in windows.

Usage::

    python demo/wide_angle_tutorial.py
    python demo/wide_angle_tutorial.py --show
    python demo/wide_angle_tutorial.py --efl 12 --fnumbers 2.8 4 5.6
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np

# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--zmx", type=Path, help="Zemax .zmx file to load instead")
parser.add_argument(
    "--agf",
    type=Path,
    nargs="+",
    help="folder(s) with Zemax AGF glass catalogs for --zmx (the catalogs "
    "named on the file's GCAT line, e.g. from the design's ZAR archive)",
)
parser.add_argument(
    "--glass",
    nargs="+",
    default=[],
    metavar="SURFACE=NAME",
    help="replace the glass after a surface with one from optiland's "
    "database, e.g. --glass 2=H-ZLAF53 4=LASFN7",
)
parser.add_argument(
    "--efl",
    type=float,
    default=24.67,
    help="focal length [mm] to scale the built-in lens to (default 24.67 mm, "
    "the focal length of the blog's patent lens); ignored with --zmx",
)
parser.add_argument(
    "--fnumbers",
    type=float,
    nargs="+",
    help="F-numbers to compare (default: 2 and 4; with --zmx: the file's "
    "F-number and 4)",
)
parser.add_argument(
    "--out", type=Path, default=Path(__file__).parent / "output" / "wide_angle"
)
parser.add_argument("--show", action="store_true", help="also show the figures")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")  # save the figures without opening windows

import matplotlib.pyplot as plt  # noqa: E402

import optiland.backend as be  # noqa: E402
from optiland.analysis import Distortion, FieldCurvature, SpotDiagram  # noqa: E402
from optiland.fileio import load_zemax_file  # noqa: E402
from optiland.materials import AbbeMaterial, Material  # noqa: E402
from optiland.mtf import FFTMTF  # noqa: E402
from optiland.samples.objectives import WideAngle100FOV  # noqa: E402

# Glass names such as SF1 exist in several catalogs; optiland picks Schott and
# warns about it. That is expected here, so keep the output readable.
warnings.filterwarnings("ignore", message="Material .* matches exactly")

args.out.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure to the output folder and report where it went."""
    path = args.out / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  saved {path}")
    if not args.show:
        plt.close(fig)


def section(title: str) -> None:
    """Print a section title."""
    print(f"\n{title}\n{'-' * len(title)}")


# ----------------------------------------------------------------------------
# 1. Starting point
# ----------------------------------------------------------------------------

section("1. Starting point")
if args.zmx:
    agf_dirs = [str(p) for p in args.agf] if args.agf else None
    lens = load_zemax_file(str(args.zmx), agf_dirs=agf_dirs)
    print(f"loaded {args.zmx}")
else:
    lens = WideAngle100FOV()
    print("built-in WideAngle100FOV (Laikin, 100 deg full field, F/4)")

for item in args.glass:
    surface, name = item.split("=", 1)
    lens.updater.set_material(Material(name), int(surface))
    n_d = float(np.ravel(be.to_numpy(Material(name).n(0.5875618)))[0])
    print(f"surface {surface}: glass {name} from the database (n_d = {n_d:.5f})")

# A Zemax file does not store glass data: each GLAS line holds the glass name
# and placeholder values (nd = 1.5, vd = 40 when the glass comes from a
# catalog). A glass that is in none of the catalogs available to optiland is
# therefore modelled with n = 1.5, which ruins the design. Report them.
placeholders = [
    i
    for i, s in enumerate(lens.surfaces)
    if isinstance(s.material_post, AbbeMaterial)
    and np.allclose(be.to_numpy(s.material_post.index), 1.5)
    and np.allclose(be.to_numpy(s.material_post.abbe), 40.0)
]
if placeholders:
    print(
        f"\nWARNING: the glass after surface(s) {placeholders} uses the file's "
        "placeholder nd = 1.5, vd = 40,\nbecause it is not in any catalog "
        "available. The results will be wrong. Pass the\nAGF catalogs with "
        "--agf, or replace the glass with --glass SURFACE=NAME.\n"
    )

# Wide-angle lenses bend the chief ray strongly, and their entrance pupil moves
# with field angle. The 'robust' ray-aiming mode finds the real stop position
# for every ray, which the default paraxial aiming cannot do at large fields
# and fast apertures.
lens.ray_tracer.set_aiming("robust", cache=True)

# ----------------------------------------------------------------------------
# 2. Scaling
# ----------------------------------------------------------------------------

section("2. Scaling to the target focal length")
efl = float(lens.paraxial.f2())
if args.zmx:
    print(f"focal length {efl:.3f} mm (not scaled: --zmx given)")
else:
    # Published prescriptions are usually normalized (here f = 0.656). Scaling
    # every radius and thickness by the same factor keeps the design's
    # aberrations, relative to its size, unchanged.
    factor = args.efl / efl
    lens.updater.scale_system(factor)
    print(f"focal length {efl:.4f} -> {float(lens.paraxial.f2()):.3f} mm "
          f"(scale factor {factor:.3f})")

# ----------------------------------------------------------------------------
# 3. First-order data and layout
# ----------------------------------------------------------------------------

section("3. First-order data")
primary = lens.primary_wavelength
num_lens_surfaces = lens.surfaces.num_surfaces - 2  # minus object and image
print(f"surfaces              : {num_lens_surfaces} (plus object and image)")
print(f"effective focal length: {float(lens.paraxial.f2()):9.3f} mm")
print(f"image-space F-number  : {float(lens.paraxial.FNO()):9.3f}")
print(f"entrance pupil diam.  : {float(lens.paraxial.EPD()):9.3f} mm")
print(f"total track           : {float(lens.total_track):9.3f} mm")
print(f"wavelengths [um]      : {[w.value for w in lens.wavelengths]}"
      f"  (primary {primary})")

# Where does each field land on the image? Trace the chief ray (pupil centre)
# of every field. For a distortion-free lens the height would be f*tan(theta).
field_type = type(lens.fields.require_definition()).__name__
max_field = float(lens.fields.max_field)
# The difference, relative to f*tan(theta), is the F-tan(theta) distortion.
print(f"\nfields ({field_type}):")
print(f"{'field':>8}{'Hy':>7}{'chief ray y [mm]':>18}{'f*tan(theta)':>14}"
      f"{'distortion':>12}")
for field in lens.fields:
    hy = field.y / max_field if max_field else 0.0
    ray = lens.trace_generic(Hx=0.0, Hy=hy, Px=0.0, Py=0.0, wavelength=primary)
    y_real = float(be.to_numpy(ray.y)[0])
    if field_type == "AngleField":
        ideal = float(lens.paraxial.f2()) * np.tan(np.radians(field.y))
        dist = f"{100 * (y_real - ideal) / ideal:>11.2f}%" if ideal else f"{'-':>12}"
        print(f"{field.y:>8.2f}{hy:>7.2f}{y_real:>18.3f}{ideal:>14.3f}{dist}")
    else:  # fields given as heights: no angle to compare with
        print(f"{field.y:>8.2f}{hy:>7.2f}{y_real:>18.3f}")

fig, _ = lens.draw(num_rays=5, figsize=(11, 4), show=False,
                   title="Layout (primary wavelength)")
save(fig, "1_layout")

# ----------------------------------------------------------------------------
# 4. Real ray trace, surface by surface
# ----------------------------------------------------------------------------

section("4. Real rays through every surface (edge of the field)")
# trace_generic traces rays given in normalized field (Hx, Hy) and normalized
# pupil (Px, Py) coordinates. Here: the chief ray and the upper and lower
# marginal rays of the edge of the field. After a trace, lens.surfaces holds
# the ray coordinates at every surface.
lens.trace_generic(
    Hx=np.zeros(3), Hy=np.ones(3), Px=np.zeros(3), Py=np.array([0.0, 1.0, -1.0]),
    wavelength=primary,
)
y = be.to_numpy(lens.surfaces.y)
z = be.to_numpy(lens.surfaces.z)
print(f"{'surface':>8}{'z [mm]':>11}{'chief y':>11}{'upper y':>11}{'lower y':>11}")
for i in range(1, lens.surfaces.num_surfaces):
    print(f"{i:>8}{z[i, 0]:>11.3f}{y[i, 0]:>11.3f}{y[i, 1]:>11.3f}{y[i, 2]:>11.3f}")

# Optiland traces many rays at once, as arrays. With real ray aiming, every
# ray is first aimed through the real stop by a few Newton iterations, each a
# full trace, so this takes several times longer than a plain trace.
start = time.perf_counter()
rays = lens.trace(Hx=0.0, Hy=0.7, wavelength=primary, num_rays=256,
                  distribution="uniform")
n = be.to_numpy(rays.x).size
print(f"\naimed and traced {n:,} rays through {num_lens_surfaces} surfaces "
      f"in {time.perf_counter() - start:.2f} s")

# ----------------------------------------------------------------------------
# 5. Image quality, and 6. its dependence on the F-number
# ----------------------------------------------------------------------------


def evaluate(fno: float) -> np.ndarray:
    """Set the F-number, save spot/MTF figures, return polychromatic RMS [um]."""
    lens.set_aperture("imageFNO", fno)
    tag = f"F{fno:g}".replace(".", "_")
    print(f"\nF/{fno:g}: entrance pupil {float(lens.paraxial.EPD()):.3f} mm")

    # Spot diagram: 12 hexapolar rings per field and wavelength, centred on
    # the chief ray of the primary wavelength (as Zemax's "Reference: Chief
    # Ray"). rms_spot_radius() is per field and wavelength; with equal ray
    # counts the polychromatic RMS is the root mean square over wavelengths.
    spot = SpotDiagram(lens, num_rings=12)
    rms = np.array([[float(r) for r in f] for f in spot.rms_spot_radius()]) * 1e3
    geo = np.array([[float(r) for r in f] for f in spot.geometric_spot_radius()])
    rms_poly = np.sqrt(np.mean(rms**2, axis=1))
    airy = 1.22 * primary * 1e-3 * fno * 1e3  # Airy radius [um]
    print(f"  Airy radius {airy:.3f} um")
    print(f"  {'field':>8}{'RMS radius [um]':>18}{'GEO radius [um]':>18}")
    for f, r, g in zip(lens.fields, rms_poly, geo.max(axis=1) * 1e3, strict=True):
        print(f"  {f.y:>8.2f}{r:>18.3f}{g:>18.3f}")
    # The Airy-disc overlay traces the four marginal rays of the full pupil.
    # In a vignetted design some of them cannot pass the lens; then the
    # diagram is drawn without it.
    try:
        fig, _ = spot.view(add_airy_disk=True, show=False)
    except ValueError as exc:
        print(f"  (no Airy disc in the plot: {exc})")
        plt.close("all")
        fig, _ = spot.view(show=False)
    fig.suptitle(f"Spot diagram, F/{fno:g}", y=1.04)
    save(fig, f"2_spot_{tag}")

    # Diffraction MTF (FFT of the pupil function) up to 100 cycles/mm, at the
    # primary wavelength, with the diffraction limit for reference. A larger
    # FFT grid (zero padding) samples the curve more finely in frequency.
    mtf = FFTMTF(lens, wavelength=primary, num_rays=128, grid_size=512,
                 max_freq=100)
    fig, _ = mtf.view(add_reference=True)
    fig.suptitle(f"Diffraction MTF, F/{fno:g}, {primary} um", y=1.02)
    save(fig, f"3_mtf_{tag}")
    return rms_poly


section("5. Image quality / 6. F-number comparison")
if args.fnumbers:
    fnumbers = args.fnumbers
elif args.zmx:
    fnumbers = [round(float(lens.paraxial.FNO()), 2), 4.0]
else:
    fnumbers = [2.0, 4.0]

fnumbers = list(dict.fromkeys(fnumbers))  # drop repeats, keep the order
design_fno = float(lens.paraxial.FNO())
results = {}
for fno in fnumbers:
    # A lens cannot be opened far beyond its design aperture: the larger
    # bundle no longer fits through the elements, ray aiming fails, and the
    # analyses cannot run. Such an F-number is reported and skipped.
    try:
        results[fno] = evaluate(fno)
    except ValueError as exc:
        plt.close("all")
        print(f"  F/{fno:g} skipped: {exc}")
        print(f"  (this lens was designed for F/{design_fno:g}; much faster "
              "apertures do not fit through it)")
fnumbers = list(results)
lens.set_aperture("imageFNO", design_fno)

print("\nPolychromatic RMS spot radius [um]:")
print(f"{'field':>8}" + "".join(f"{'F/' + format(f, 'g'):>10}" for f in fnumbers))
for i, field in enumerate(lens.fields):
    print(f"{field.y:>8.2f}" + "".join(f"{results[f][i]:>10.2f}" for f in fnumbers))

# Note: the image plane stays where the design put it. Stopping down cuts the
# aberrated outer zones of the pupil, so the spots shrink; but the best focus
# of a lens with spherical aberration also moves with the aperture, so the
# MTF at one F-number can be worse than its spot size suggests until the image
# plane is refocused (an optimization variable in the next design step).

# Field curvature and distortion barely depend on the aperture (they are
# chief-ray properties), so they are shown once, at the last F-number.
fig, _ = FieldCurvature(lens).view(show=False)
save(fig, "4_field_curvature")

# The distortion plot takes its plate scale from a chief ray at 1e-10 of the
# field. The 'robust' aiming mode returns exactly 0 for such a ray (as of this
# writing), which makes the plot empty, so the plot uses 'iterative' aiming;
# chief rays need no robust aiming.
lens.ray_tracer.set_aiming("iterative", cache=True)
fig, _ = Distortion(lens, distortion_type="f-tan").view(show=False)
fig.suptitle("F-tan(theta) distortion", y=1.02)
save(fig, "5_distortion")
lens.ray_tracer.set_aiming("robust", cache=True)

print(f"\nFigures are in {args.out}")
if args.show:
    plt.show()
