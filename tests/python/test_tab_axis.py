"""``-TAB`` lookup axes (FITS Paper III Sec.6).

The coordinate array lives in a separate BINTABLE. Both
:meth:`fitsy.FitsFile.wcs` and :meth:`fitsy.ImageHdu.wcs` resolve it
through the file the HDU was opened from. A handle with no backing
file cannot, and raises at ``wcs()`` time; ``fitsy.Wcs(header)`` is
the header-only path and raises on first use of the axis instead.

The fixture is built with fitsy's own writer rather than astropy's.
astropy was only ever a convenient way to *produce* the input here --
nothing below checks anything about astropy -- so depending on it
just meant these tests silently skipped wherever it was not
installed, which included CI.
"""

from __future__ import annotations

import fitsy
import numpy as np
import pytest

WAVELENS = [4000.0, 4500.0, 5500.0, 7000.0, 9000.0]


def _write_tab_file(path, coord=WAVELENS, index=None):
    """A minimal cube whose third axis is a -TAB lookup."""
    k = len(coord)
    header = {
        "CTYPE1": "X",
        "CTYPE2": "Y",
        "CTYPE3": "WAVE-TAB",
        "CUNIT3": "Angstrom",
        # PS3_0/PS3_1 name the table extension and its coordinate
        # column; PV3_1 is its EXTVER.
        "PS3_0": "WCS-TAB",
        "PS3_1": "WAVELEN",
        "PV3_1": 1,
    }
    for i, (crpix, crval, cdelt) in enumerate(
        [(1.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)], start=1
    ):
        header[f"CRPIX{i}"] = crpix
        header[f"CRVAL{i}"] = crval
        header[f"CDELT{i}"] = cdelt

    # A 2-D array gives a fixed-repeat column: one row of K doubles,
    # which is the single-axis 1-D -TAB layout.
    columns = {"WAVELEN": np.asarray(coord, dtype=float).reshape(1, k)}
    if index is not None:
        header["PS3_2"] = "IDX"
        columns["IDX"] = np.asarray(index, dtype=float).reshape(1, len(index))

    fitsy.write(
        str(path),
        [
            fitsy.image(np.zeros((k, 2, 2), dtype="f4"), header=header, primary=True),
            fitsy.bintable(columns, extname="WCS-TAB"),
        ],
        overwrite=True,
    )
    # EXTVER defaults to 1 when absent, which is what PV3_1 asks for,
    # but write it so the pointer is explicit rather than implied.
    with fitsy.open(str(path), mode="update") as f:
        f[1].header["EXTVER"] = 1
        f.flush()
    return str(path)


def test_fitsfile_wcs_resolves_tab_axis(tmp_path):
    """Regression: the Python binding parsed the header directly and
    never resolved -TAB, so every lookup raised -- even though the
    Rust ``FitsFile::wcs`` resolves and the guide says it does."""
    path = _write_tab_file(tmp_path / "tab.fits")
    with fitsy.open(path) as f:
        wcs = f.wcs(0)
        got = [wcs.pixel_to_world([0.0, 0.0, float(k)])[2] for k in range(5)]
    np.testing.assert_allclose(got, WAVELENS, rtol=0, atol=1e-9)


def test_tab_axis_interpolates_between_samples(tmp_path):
    path = _write_tab_file(tmp_path / "tab.fits")
    with fitsy.open(path) as f:
        wcs = f.wcs(0)
        # Halfway between samples 1 and 2 (4500 and 5500).
        assert wcs.pixel_to_world([0.0, 0.0, 1.5])[2] == pytest.approx(5000.0)


def test_tab_axis_honors_index_column(tmp_path):
    path = _write_tab_file(tmp_path / "tab.fits", index=[1.0, 2.0, 4.0, 8.0, 16.0])
    with fitsy.open(path) as f:
        wcs = f.wcs(0)
        # psi = 1 + p; index 4 (p = 3) is the third sample, 5500.
        assert wcs.pixel_to_world([0.0, 0.0, 3.0])[2] == pytest.approx(5500.0)


def test_tab_axis_extrapolation_is_bounded(tmp_path):
    """Paper III Sec.6.1.2 permits half a sample step past each end
    and leaves the coordinate undefined beyond. astropy returns NaN
    there; we raise."""
    path = _write_tab_file(tmp_path / "tab.fits")
    with fitsy.open(path) as f:
        wcs = f.wcs(0)
        assert wcs.pixel_to_world([0.0, 0.0, -0.5])[2] == pytest.approx(3750.0)
        for pixel in (-0.51, 4.51, -3.0, 9.0):
            with pytest.raises(fitsy.FitsError):
                wcs.pixel_to_world([0.0, 0.0, pixel])


def test_imagehdu_wcs_resolves_tab_axis(tmp_path):
    """``f[i].wcs()`` resolves -TAB exactly as ``f.wcs(i)`` does."""
    path = _write_tab_file(tmp_path / "tab.fits")
    with fitsy.open(path) as f:
        wcs = f[0].wcs()
        got = [wcs.pixel_to_world([0.0, 0.0, float(k)])[2] for k in range(5)]
    np.testing.assert_allclose(got, WAVELENS, rtol=0, atol=1e-9)


def test_compressed_imagehdu_wcs_resolves_tab_axis(tmp_path):
    """The tile-compressed wrapper keeps the file handle too.

    A compressed image materializes without a lazy-read binding, so
    this exercises a separate construction path from the plain-image
    test above.
    """
    plain = _write_tab_file(tmp_path / "tab.fits")
    with fitsy.open(plain) as f:
        header = {
            k: v
            for k, v in dict(f[0].header).items()
            if not k.startswith(("NAXIS", "BITPIX", "SIMPLE", "EXTEND"))
        }
    k = len(WAVELENS)
    fitsy.write(
        str(tmp_path / "ctab.fits"),
        [
            fitsy.compressed_image(np.zeros((k, 2, 2), dtype="f4"), header=header),
            fitsy.bintable(
                {"WAVELEN": np.asarray(WAVELENS, dtype=float).reshape(1, k)},
                extname="WCS-TAB",
            ),
        ],
        overwrite=True,
    )
    with fitsy.open(tmp_path / "ctab.fits") as f:
        wcs = f[1].wcs()
        got = [wcs.pixel_to_world([0.0, 0.0, float(p)])[2] for p in range(k)]
    np.testing.assert_allclose(got, WAVELENS, rtol=0, atol=1e-9)


def test_unbacked_imagehdu_wcs_raises_at_call(tmp_path):
    """An HDU built in memory has no file to resolve -TAB against.

    The accessor raises at ``wcs()`` time rather than returning a
    WCS whose transforms fail later.
    """
    path = _write_tab_file(tmp_path / "tab.fits")
    with fitsy.open(path) as f:
        header = dict(f[0].header)
    hdu = fitsy.ImageHdu(np.zeros((5, 2, 2), dtype="f4"), header=header)
    with pytest.raises(fitsy.FitsError, match="-TAB"):
        hdu.wcs()


def test_unbacked_fitsfile_wcs_raises_at_call(tmp_path):
    """An in-memory FitsFile has no file to resolve -TAB against."""
    path = _write_tab_file(tmp_path / "tab.fits")
    with fitsy.open(path) as f:
        header = dict(f[0].header)
    mem = fitsy.FitsFile()
    mem.append(fitsy.ImageHdu(np.zeros((5, 2, 2), dtype="f4"), header=header))
    with pytest.raises(fitsy.FitsError, match="-TAB"):
        mem.wcs(0)


def test_wcs_constructor_stays_header_only(tmp_path):
    """``fitsy.Wcs(header)`` parses without the table.

    It is the explicit header-only path: metadata reads work, and a
    transform touching the unresolved axis raises.
    """
    path = _write_tab_file(tmp_path / "tab.fits")
    with fitsy.open(path) as f:
        wcs = fitsy.Wcs(f[0].header)
    assert wcs.ctype[2] == "WAVE-TAB"
    with pytest.raises(fitsy.FitsError, match="unresolved"):
        wcs.pixel_to_world([0.0, 0.0, 0.0])


# Reference lookups for the WAVELENS table above, from wcslib 8.6 via
# astropy 8.0.1 (`WCS(hdul[0].header, fobj=hdul).wcs_pix2world`).
# Frozen rather than recomputed live: the values are the contract, and
# a live comparison only tested anything where astropy happened to be
# installed -- which was not CI. See `tests/data/gen_reference_*.py`
# for the same pattern on the Rust side.
#
# Every entry is also linear interpolation of the table by hand, which
# is the point: freezing wcslib's answer costs nothing in reviewability
# because the answer is independently checkable.
WCSLIB_REFERENCE = {
    -0.5: 3750.0,  # half-step margin below the first sample
    -0.25: 3875.0,
    0.0: 4000.0,  # first sample
    1.5: 5000.0,  # midway between samples 1 and 2
    2.0: 5500.0,
    3.75: 8500.0,
    4.0: 9000.0,  # last sample
    4.5: 10000.0,  # half-step margin above it
}


def test_tab_axis_matches_wcslib_reference(tmp_path):
    """Cross-check the lookup against wcslib across the full range."""
    path = _write_tab_file(tmp_path / "tab.fits")
    with fitsy.open(path) as f:
        wcs = f.wcs(0)
        for pixel, expected in WCSLIB_REFERENCE.items():
            got = wcs.pixel_to_world([0.0, 0.0, pixel])[2]
            assert got == pytest.approx(expected, abs=1e-9), f"at pixel {pixel}"
        # Outside the permitted margin wcslib returns NaN; we raise.
        for pixel in (-3.0, 9.0):
            with pytest.raises(fitsy.FitsError):
                wcs.pixel_to_world([0.0, 0.0, pixel])
