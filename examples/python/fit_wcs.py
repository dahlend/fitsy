"""Fit a celestial WCS from pixel/sky correspondences.

Run from the repo root:

    python examples/python/fit_wcs.py
"""

import fitsy
import numpy as np

pix = np.array(
    [
        [100.0, 100.0],
        [200.0, 100.0],
        [100.0, 200.0],
        [200.0, 200.0],
    ]
)
sky = np.array(
    [
        [10.00, -5.00],
        [10.05, -5.00],
        [10.00, -4.95],
        [10.05, -4.95],
    ]
)

fit = fitsy.fit_wcs(pix, sky, projection="TAN")
print(f'rms = {fit.rms_arcsec:.3f}"  max = {fit.max_arcsec:.3f}"')

# `fit.wcs` is a fully usable Wcs.
ra, dec = fit.wcs.pixel_to_celestial(150.0, 150.0)
print(f"center: RA={ra:.4f}  Dec={dec:.4f}")

# Serialize back to a header dict for writing.
header = fit.wcs.to_header()
print("CRVAL1:", header["CRVAL1"])
