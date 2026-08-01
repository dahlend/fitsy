"""Generate a multi-dimensional `-TAB` fixture and wcslib's decode of it.

Paper III Sec.6.1.1's non-separable case: a celestial pair whose
longitude and latitude both depend on both pixel axes, sharing one
`(M, K_1, K_2)` coordinate array. There is no way to reproduce this
with two independent 1-D tables, which is the point.

Run with the repo venv:
    .venv/bin/python tests/data/gen_tab_reference.py
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

OUT = Path(__file__).parent / "ref_tab_2d.fits"

K1, K2, M = 5, 4, 2


def coordinate_array() -> np.ndarray:
    """Non-separable RA/Dec on a (K2, K1, M) grid.

    The cross terms are what make it non-separable: RA depends on the
    second index and Dec on the first, so neither axis can be tabulated
    on its own.
    """
    a = np.empty((K2, K1, M))
    for j in range(K2):
        for i in range(K1):
            a[j, i, 0] = 10.0 + 0.10 * i + 0.020 * j + 0.004 * i * j
            a[j, i, 1] = 20.0 + 0.03 * i + 0.100 * j - 0.002 * i * j
    return a


def main() -> None:
    coords = coordinate_array()

    img = fits.PrimaryHDU(np.zeros((K2, K1), dtype=np.float32))
    h = img.header
    h["CTYPE1"], h["CTYPE2"] = "RA---TAB", "DEC--TAB"
    h["CRVAL1"] = h["CRVAL2"] = 0.0
    h["CRPIX1"] = h["CRPIX2"] = 0.0
    h["CDELT1"] = h["CDELT2"] = 1.0
    h["CUNIT1"] = h["CUNIT2"] = "deg"
    for i in (1, 2):
        h[f"PS{i}_0"] = "WCS-TAB"
        h[f"PS{i}_1"] = "COORDS"
        h[f"PS{i}_2"] = f"IDX{i}"
        h[f"PV{i}_3"] = i  # slot in the leading length-M axis

    # Index vectors: axis 1 sampled evenly, axis 2 unevenly, so the
    # per-axis indirection of Sec.6.1.1 is exercised too.
    idx1 = np.arange(1, K1 + 1, dtype=float)
    idx2 = np.array([1.0, 2.0, 4.0, 7.0])
    assert len(idx2) == K2

    tab = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="COORDS",
                format=f"{M * K1 * K2}D",
                dim=f"({M},{K1},{K2})",
                array=coords.reshape(1, -1),
            ),
            fits.Column(name="IDX1", format=f"{K1}D", array=idx1.reshape(1, -1)),
            fits.Column(name="IDX2", format=f"{K2}D", array=idx2.reshape(1, -1)),
        ],
        name="WCS-TAB",
    )

    hdul = fits.HDUList([img, tab])
    hdul.writeto(OUT, overwrite=True)

    # Store wcslib's own answers, so the Rust side is compared against
    # an independent implementation rather than round-tripped.
    pixels = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (2.0, 1.0),
        (3.5, 2.0),
        (4.0, 3.0),
        (1.25, 2.75),
    ]
    with fits.open(OUT) as f:
        w = WCS(f[0].header, f)
        world = w.wcs_pix2world(np.array(pixels), 0)
        back = w.wcs_world2pix(world, 0)
    assert np.all(np.isfinite(world)), "wcslib produced non-finite values"
    assert np.allclose(back, pixels, atol=1e-8), f"wcslib round trip failed:\n{back}"

    cards = [
        fits.Card("COMMENT", "wcslib reference decode; see gen_tab_reference.py"),
    ]
    for (px, py), (ra, dec) in zip(pixels, world):
        cards.append(fits.Card("PIXX", px))
        cards.append(fits.Card("PIXY", py))
        cards.append(fits.Card("WLON", ra))
        cards.append(fits.Card("WLAT", dec))
    ref = fits.ImageHDU(data=np.zeros(1, dtype=np.float32), name="REFERENCE")
    for c in cards:
        ref.header.append(c)
    with fits.open(OUT, mode="append") as f:
        f.append(ref)

    print(f"wrote {OUT}")
    print(f"  TDIM1 = {tab.header['TDIM1']}")
    for (px, py), (ra, dec) in zip(pixels, world):
        print(f"  pixel ({px:4}, {py:4}) -> ({ra:.10f}, {dec:.10f})")


if __name__ == "__main__":
    main()
