#!/usr/bin/env python3
"""Generate `ref_nocompress.fits`, an astropy-written tile-compressed
image using ZCMPTYPE = 'NOCOMPRESS'.

Standard Sec.10 Table 10 lists NOCOMPRESS alongside RICE_1, GZIP_1,
GZIP_2, PLIO_1 and HCOMPRESS_1 as a valid ZCMPTYPE: the tile bytes are
stored verbatim and the HDU "remains uncompressed". astropy supports it
(`astropy.io.fits.hdu.compressed.COMPRESSION_TYPES`), so a file like
this one can arrive from any astropy user with a single kwarg -- fitsy
used to refuse it at the first pixel read.

Two HDUs are written so both the integer and the lossless-float path
are covered; a float image is the case that also has to get past the
"lossless float needs GZIP" guard.

Regenerate with:

    python3 tests/data/gen_nocompress.py

Requires astropy. The expected pixel values are pure arithmetic, so the
Rust side recomputes them rather than trusting a recorded copy.
"""

import numpy as np
from astropy.io import fits

# Values chosen so a byte-swap or an off-by-one tile stride cannot pass:
# every pixel is distinct and the array is not square.
ints = np.arange(6 * 8, dtype=np.int16).reshape(6, 8) * 37 - 500
floats = (np.arange(6 * 8, dtype=np.float32).reshape(6, 8) * 0.25) - 3.5

hdus = [fits.PrimaryHDU()]
for name, data in (("INT16", ints), ("FLOAT32", floats)):
    hdu = fits.CompImageHDU(
        data,
        name=name,
        compression_type="NOCOMPRESS",
        # A tile that is not the whole image, so the per-tile copy has
        # to place its bytes at the right offset.
        tile_shape=(2, 8),
    )
    hdus.append(hdu)

fits.HDUList(hdus).writeto("ref_nocompress.fits", overwrite=True)

back = fits.open("ref_nocompress.fits")
assert np.array_equal(back[1].data, ints)
assert np.array_equal(back[2].data, floats)
# `CompImageHDU.header` presents the *decompressed* image header, which
# carries no ZCMPTYPE, so confirm the on-disk keyword from the bytes.
raw = open("ref_nocompress.fits", "rb").read()
assert raw.count(b"ZCMPTYPE= 'NOCOMPRESS'") == 2, "ZCMPTYPE not written as expected"
print("wrote ref_nocompress.fits")
