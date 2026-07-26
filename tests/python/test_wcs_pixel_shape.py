"""`Wcs.pixel_shape` and the `footprint()` it exists to support.

`pixel_shape` is a snapshot of the `NAXISn` cards, deliberately not
part of the coordinate description: it is absent when there was no
image, it never affects a transform, and it is never validated against
the data.
"""

from __future__ import annotations

import os

import fitsy
import numpy as np
import pytest

NGC2403 = os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "data", "ngc2403.fits.gz"
)


def _header(nx=400, ny=300, **extra):
    cards = {
        "SIMPLE": True,
        "BITPIX": -32,
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": nx / 2,
        "CRPIX2": ny / 2,
        "CRVAL1": 10.0,
        "CRVAL2": -5.0,
        "CDELT1": -1e-3,
        "CDELT2": 1e-3,
    }
    cards.update(extra)
    return fitsy.Header(cards)


def test_pixel_shape_from_header():
    assert fitsy.Wcs(_header(400, 300)).pixel_shape == (400, 300)


def test_pixel_shape_from_open_file():
    with fitsy.open(NGC2403) as f:
        hdu = f[0]
        # FITS axis order (NAXIS1 first) -- the reverse of numpy's shape.
        assert hdu.wcs().pixel_shape == tuple(hdu.axes)
        assert hdu.wcs().pixel_shape == hdu.data.shape[::-1]


def test_pixel_shape_absent_for_fitted_wcs():
    fit = fitsy.fit_wcs(
        [[100.0, 100.0], [200.0, 100.0], [100.0, 200.0], [200.0, 200.0]],
        [[10.0, -5.0], [10.05, -5.0], [10.0, -4.95], [10.05, -4.95]],
    )
    assert fit.wcs.pixel_shape is None
    with pytest.raises(fitsy.FitsError, match="no image shape"):
        fit.wcs.footprint()


def test_pixel_shape_does_not_affect_transforms():
    """A different NAXISn, everything else equal, must not move a coordinate."""
    fixed = {"CRPIX1": 200.0, "CRPIX2": 150.0}
    small = fitsy.Wcs(_header(400, 300, **fixed))
    large = fitsy.Wcs(_header(4000, 3000, **fixed))
    assert small.pixel_shape != large.pixel_shape
    np.testing.assert_allclose(
        small.pixel_to_celestial(37.0, 42.0), large.pixel_to_celestial(37.0, 42.0)
    )


def test_footprint_corners():
    nx, ny = 400, 300
    w = fitsy.Wcs(_header(nx, ny))
    fp = w.footprint()
    assert fp.shape == (4, 2)
    # Corner pixel centers, counter-clockwise from the origin.
    for got, (px, py) in zip(
        fp, [(0, 0), (nx - 1, 0), (nx - 1, ny - 1), (0, ny - 1)], strict=True
    ):
        np.testing.assert_allclose(got, w.pixel_to_celestial(float(px), float(py)))


def test_footprint_matches_astropy():
    astropy_wcs = pytest.importorskip("astropy.wcs")
    from astropy.io import fits

    with fitsy.open(NGC2403) as f:
        ours = f[0].wcs().footprint()
    with fits.open(NGC2403) as f:
        theirs = astropy_wcs.WCS(f[0].header).calc_footprint()

    # Same four corners; astropy starts at the same pixel but we do not
    # promise its ordering, so compare as sets of positions.
    ours_sorted = ours[np.lexsort((ours[:, 1], ours[:, 0]))]
    theirs_sorted = theirs[np.lexsort((theirs[:, 1], theirs[:, 0]))]
    np.testing.assert_allclose(ours_sorted, theirs_sorted, atol=1e-9)


def test_footprint_requires_celestial_axes():
    w = fitsy.Wcs(_header(10, 10, CTYPE1="WAVE", CTYPE2="TIME", CUNIT1="m", CUNIT2="s"))
    assert not w.is_celestial
    with pytest.raises(fitsy.FitsError, match="celestial"):
        w.footprint()
