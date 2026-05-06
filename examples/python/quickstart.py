"""Quickstart: open a FITS file, read pixels, headers, and WCS.

Run from the repo root:

    python examples/python/quickstart.py
"""

import fitsy

with fitsy.open("examples/data/ngc2403.fits.gz") as f:
    hdu = f[0]
    print(hdu.bitpix, hdu.axes)  # 16  [1448, 2172]
    data = hdu.data  # numpy.ndarray, shape (2172, 1448)
    print("data shape:", data.shape, "dtype:", data.dtype)

    wcs = hdu.wcs()  # Wcs (TAN + SIP)
    if wcs is not None:
        ra, dec = wcs.pixel_to_celestial(724.0, 1086.0)
        print(f"center: RA={ra:.4f}  Dec={dec:.4f}")
