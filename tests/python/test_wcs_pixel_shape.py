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
        small.pixel_to_world([37.0, 42.0]), large.pixel_to_world([37.0, 42.0])
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
        np.testing.assert_allclose(got, w.pixel_to_world([float(px), float(py)]))


# `WCS(hdu.header).calc_footprint()` for `ngc2403.fits.gz`, from
# wcslib 8.6 via astropy 8.0.1, sorted by (RA, Dec). Frozen rather
# than recomputed live: a live comparison only tested anything where
# astropy happened to be installed, which was not CI.
NGC2403_FOOTPRINT_REFERENCE = [
    (113.72543867991, 65.3032667940129),
    (113.7254815018997, 65.88553801158601),
    (114.65468345081484, 65.29352665515752),
    (114.68831331817123, 65.88469749172714),
]


def test_footprint_matches_wcslib_reference():
    with fitsy.open(NGC2403) as f:
        ours = f[0].wcs().footprint()

    # Same four corners; astropy starts at the same pixel but we do not
    # promise its ordering, so compare as sets of positions.
    ours_sorted = ours[np.lexsort((ours[:, 1], ours[:, 0]))]
    np.testing.assert_allclose(
        ours_sorted, np.array(NGC2403_FOOTPRINT_REFERENCE), atol=1e-9
    )


def test_footprint_does_not_require_celestial_axes():
    """A footprint is world coordinates, not sky coordinates.

    A spectral / time image has corners in its own units, so
    ``footprint()`` works without a celestial pair.
    """
    nx, ny = 10, 10
    w = fitsy.Wcs(_header(nx, ny, CTYPE1="WAVE", CTYPE2="TIME", CUNIT1="m", CUNIT2="s"))
    assert not w.is_celestial
    fp = w.footprint()
    assert fp.shape == (4, 2)
    for got, (px, py) in zip(
        fp, [(0, 0), (nx - 1, 0), (nx - 1, ny - 1), (0, ny - 1)], strict=True
    ):
        np.testing.assert_allclose(got, w.pixel_to_world([px, py]), atol=1e-12)


def test_footprint_covers_every_axis_of_a_cube():
    """A three-axis WCS gives 2**3 corners, spectral axis included."""
    w = fitsy.Wcs(
        _header(
            8,
            6,
            NAXIS=3,
            NAXIS3=4,
            CTYPE3="FREQ",
            CRPIX3=2.0,
            CRVAL3=1.4e9,
            CDELT3=1.0e6,
            CUNIT3="Hz",
        )
    )
    fp = w.footprint()
    assert fp.shape == (8, 3)
    # A celestial-only footprint would pin the spectral axis to one
    # plane, leaving this column constant.
    assert np.ptp(fp[:, 2]) > 0.0


def test_footprint_holds_a_degenerate_axis_at_its_reference_pixel():
    """Hold an axis with no image axis at its reference pixel.

    ``WCSAXES > NAXIS`` is legal (Sec. 8.2). The third axis has no
    ``NAXIS3`` to take a length from. It sits at its reference pixel
    instead. The corner count follows the image, and every corner still
    carries a full world vector.
    """
    nx, ny = 8, 6
    w = fitsy.Wcs(
        _header(
            nx,
            ny,
            WCSAXES=3,
            CTYPE3="FREQ",
            CRPIX3=2.0,
            CRVAL3=1.4e9,
            CDELT3=1.0e6,
            CUNIT3="Hz",
        )
    )
    assert w.naxis == 3
    assert w.pixel_shape == (nx, ny)

    fp = w.footprint()
    assert fp.shape == (4, 3), "2**2 corners -- the image, not WCSAXES"
    np.testing.assert_allclose(fp[:, 2], 1.4e9)
    for got, (px, py) in zip(
        fp, [(0, 0), (nx - 1, 0), (nx - 1, ny - 1), (0, ny - 1)], strict=True
    ):
        np.testing.assert_allclose(got, w.pixel_to_world([px, py, 1.0]), atol=1e-12)


def test_footprint_marks_out_of_domain_corners_nan():
    """Fill a corner outside the projection's domain with ``nan``.

    Corners go through the batch transform, so they follow the batch
    rule. ``footprint()`` returns rather than raises. The caller tests
    the result with ``numpy.isfinite``.
    """
    # 100 px at 1 deg/px puts every corner outside the SIN domain.
    w = fitsy.Wcs(
        _header(
            100,
            100,
            CTYPE1="RA---SIN",
            CTYPE2="DEC--SIN",
            CDELT1=-1.0,
            CDELT2=1.0,
        )
    )
    with pytest.raises(fitsy.FitsError):
        w.pixel_to_world([0.0, 0.0])
    fp = w.footprint()
    assert fp.shape == (4, 2)
    assert not np.isfinite(fp).any()
