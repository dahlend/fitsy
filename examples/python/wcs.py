"""WCS pixel <-> sky transforms on the bundled NGC 2403 image.

Run from the repository root:

    python examples/python/wcs.py
"""

import fitsy
import numpy as np

with fitsy.open("examples/data/ngc2403.fits.gz") as f:
    wcs = f.wcs(0)  # equivalent: f[0].wcs()

# Single pixel -> sky (0-based pixel coordinates).
ra, dec = wcs.pixel_to_world([724.0, 1086.0])
print(f"center:     RA={ra:.4f}  Dec={dec:.4f}")

# Sky -> pixel (round-trip).
px, py = wcs.world_to_pixel([ra, dec])
print(f"round-trip: ({px:.2f}, {py:.2f})")

# Batch transform: corners + center -> sky.
# Accepts any array-like (numpy array, list of lists, etc.).
sky = wcs.pixel_to_world(np.array([[0.0, 0.0], [1447.0, 2171.0], [724.0, 1086.0]]))
print("corners + center sky:")
print(sky)

# Plain Python lists work too (no numpy import required at call site).
sky2 = wcs.pixel_to_world([[0.0, 0.0], [724.0, 1086.0]])
print("list-of-lists input:", sky2)

# pixel_to_world / world_to_pixel take one point or many. One point is
# a length-naxis sequence and gives a list back. An (N, naxis) array is
# a batch and gives an (N, naxis) array back. Unlike the celestial
# helpers, this path also reaches a spectral, time or -TAB axis.
print("one point: ", wcs.pixel_to_world([724.0, 1086.0]))
batch = wcs.pixel_to_world(np.array([[0.0, 0.0], [724.0, 1086.0]]))
print("batch shape:", batch.shape)
print(batch)

# A batch marks a point it cannot transform with nan rather than
# raising, so one bad pixel does not lose the rest of the field.
print("round-trip:", wcs.world_to_pixel(batch))
