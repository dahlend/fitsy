"""Fit a celestial WCS from pixel and sky pairs.

Run from the repository root:

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

# `pixels` and `sky` also accept plain Python lists of lists;
# no numpy import is required at the call site.
pix_list = [[100.0, 100.0], [200.0, 100.0], [100.0, 200.0], [200.0, 200.0]]
sky_list = [[10.00, -5.00], [10.05, -5.00], [10.00, -4.95], [10.05, -4.95]]

fit = fitsy.fit_wcs(pix, sky, projection="TAN")
fit2 = fitsy.fit_wcs(pix_list, sky_list, projection="TAN")
assert (
    abs(fit2.rms_arcsec - fit.rms_arcsec) < 1e-10
), "list and array results should match"
print(f'rms = {fit.rms_arcsec:.3f}"  max = {fit.max_arcsec:.3f}"')

# `fit.wcs` is a fully usable Wcs.
ra, dec = fit.wcs.pixel_to_world([150.0, 150.0])
print(f"center: RA={ra:.4f}  Dec={dec:.4f}")

# Serialize back to a header dict for writing.
header = fit.wcs.to_header()
print("CRVAL1:", header["CRVAL1"])
