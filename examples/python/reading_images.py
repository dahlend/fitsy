"""Read image pixels: NAXIS / numpy axis order, dtype handling.

Run from the repo root:

    python examples/python/reading_images.py
"""

import fitsy

with fitsy.open("examples/data/ngc2403.fits.gz") as f:
    hdu = f[0]
    print("axes (NAXIS1, NAXIS2):", hdu.axes)  # [1448, 2172]
    print("data.shape (numpy):  ", hdu.data.shape)  # (2172, 1448)
    assert hdu.axes == [hdu.data.shape[1], hdu.data.shape[0]]

    # `data` is returned in physical units with an appropriate dtype
    # (BZERO / BSCALE / BLANK applied automatically).
    print("dtype:", hdu.data.dtype)
