"""WCS pixel <-> sky transforms on the bundled NGC 2403 image.

Run from the repo root:

    python examples/python/wcs.py
"""

import fitsy
import numpy as np

with fitsy.open("examples/data/ngc2403.fits.gz") as f:
    wcs = f.wcs(0)  # equivalent: f[0].wcs()

# Single pixel -> sky (0-based pixel coordinates).
ra, dec = wcs.pixel_to_celestial(724.0, 1086.0)
print(f"center:     RA={ra:.4f}  Dec={dec:.4f}")

# Sky -> pixel (round-trip).
px, py = wcs.celestial_to_pixel(ra, dec)
print(f"round-trip: ({px:.2f}, {py:.2f})")

# Batch transform: corners + center -> sky.
sky = wcs.pixel_to_celestial_many(
    np.array([[0.0, 0.0], [1447.0, 2171.0], [724.0, 1086.0]])
)
print("corners + center sky:")
print(sky)
