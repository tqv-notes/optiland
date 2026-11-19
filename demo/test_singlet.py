from optiland.optic import Optic

lens = Optic()
lens.surfaces.add(index=0, thickness=float("inf"))                                     # object
lens.surfaces.add(index=1, radius=25.0, thickness=6.0, material="N-BK7", is_stop=True)  # front
lens.surfaces.add(index=2, radius=-75.0, thickness=34.0)                               # back
lens.surfaces.add(index=3)                                                             # image
lens.set_aperture(aperture_type="EPD", value=20)
lens.fields.set_type("angle")
lens.fields.add(y=0)
lens.fields.add(y=5)
lens.wavelengths.add(value=0.587, is_primary=True)

lens.draw()                       # 2D layout with traced rays
print(lens.paraxial.f2())         # effective focal length -> 37.04 mm