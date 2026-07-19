#!/usr/bin/env python3
"""Generate externally-compressed reference fixtures using astropy.

Run from the repository root:
    python3 tests/data/gen_reference_fixtures.py

Unlike ``gen_fixtures.py`` (dependency-free, hand-rolled bytes), this script
requires astropy. Astropy's tile-compression code is derived from cfitsio, so
these files are authoritative reference streams: fitsy must decode them to
exactly the values astropy wrote/reads. Round-trip tests through fitsy's own
encoders can never catch a self-consistent codec bug (encoder and decoder
sharing the same error); these fixtures exist precisely to close that gap --
a Rice off-by-one, a dither-seed offset, and an HCompress coefficient-width
bug all survived years of round-trip testing before externally-generated
streams exposed them.

Outputs (written to tests/data/):
    ref_rice_i16.fits    - RICE_1 int16, 33x17 (odd size -> partial tiles)
    ref_rice_i32.fits    - RICE_1 int32, large-magnitude values
    ref_hcomp_i32.fits   - HCOMPRESS_1 int32, values near +-2^30 (needs the
                           64-bit coefficient path), lossless (scale 0)
    ref_dither_f32.fits  - GZIP_1 + SUBTRACTIVE_DITHER_1 float32 with
                           ZDITHER0=42. GZIP payload isolates the dither
                           sequence from any codec bug. HDU 2 ("EXPECTED")
                           stores astropy's own decompressed output; fitsy
                           must reproduce it bit-for-bit.

Integer pixel data uses the deterministic patterns below, re-computed on the
Rust side (tests/compression_reference.rs) rather than stored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

OUT = Path(__file__).resolve().parent


def lcg(n: int) -> np.ndarray:
    """Deterministic value stream shared with the Rust test: a plain LCG.

    x_{k+1} = (x_k * 1103515245 + 12345) mod 2^31, seeded with x_0 = 1;
    the k-th output is x_{k+1} (i.e. the state after k+1 steps).
    """
    out = np.empty(n, dtype=np.int64)
    x = 1
    for k in range(n):
        x = (x * 1103515245 + 12345) % (1 << 31)
        out[k] = x
    return out


def rice_i16() -> None:
    n = 33 * 17
    vals = (lcg(n) % 1000 - 500).astype(np.int16).reshape(17, 33)
    hdu = fits.CompImageHDU(data=vals, compression_type="RICE_1", tile_shape=(4, 16))
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
        OUT / "ref_rice_i16.fits", overwrite=True
    )
    print("  ref_rice_i16.fits")


def rice_i32() -> None:
    n = 40 * 12
    vals = (lcg(n) - (1 << 30)).astype(np.int32).reshape(12, 40)
    hdu = fits.CompImageHDU(data=vals, compression_type="RICE_1", tile_shape=(3, 40))
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
        OUT / "ref_rice_i32.fits", overwrite=True
    )
    print("  ref_rice_i32.fits")


def hcomp_i32() -> None:
    n = 64 * 64
    vals = (lcg(n) - (1 << 30)).astype(np.int32).reshape(64, 64)
    hdu = fits.CompImageHDU(
        data=vals,
        compression_type="HCOMPRESS_1",
        hcomp_scale=0,  # lossless
        tile_shape=(64, 64),
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
        OUT / "ref_hcomp_i32.fits", overwrite=True
    )
    print("  ref_hcomp_i32.fits")


def dither_f32() -> None:
    n = 32 * 16
    vals = (lcg(n).astype(np.float64) / (1 << 31) * 100.0 - 50.0).astype(
        np.float32
    ).reshape(16, 32)
    hdu = fits.CompImageHDU(
        data=vals,
        compression_type="GZIP_1",
        quantize_level=16.0,
        quantize_method=1,  # SUBTRACTIVE_DITHER_1
        dither_seed=42,
        tile_shape=(4, 32),
    )
    buf = fits.HDUList([fits.PrimaryHDU(), hdu])
    tmp = OUT / "ref_dither_f32.fits"
    buf.writeto(tmp, overwrite=True)
    # Re-open and store astropy's own decode as the expected result.
    with fits.open(tmp, disable_image_compression=True) as f:
        assert f[1].header.get("ZDITHER0") == 42, f[1].header.get("ZDITHER0")
    with fits.open(tmp) as f:
        expected = np.array(f[1].data, dtype=np.float32)
    with fits.open(tmp, mode="append") as f:
        f.append(fits.ImageHDU(data=expected, name="EXPECTED"))
    print("  ref_dither_f32.fits")


if __name__ == "__main__":
    print("Generating astropy reference fixtures:")
    rice_i16()
    rice_i32()
    hcomp_i32()
    dither_f32()
