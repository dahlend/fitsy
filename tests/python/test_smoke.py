"""Round-trip smoke tests for the Python bindings."""

from __future__ import annotations

import fitsy
import numpy as np
import pytest


def test_version_present():
    assert isinstance(fitsy.__version__, str)
    assert fitsy.__version__


def test_image_round_trip_f32(tmp_path):
    arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    path = tmp_path / "a.fits"
    fitsy.write(
        str(path),
        [fitsy.image(arr, header={"OBJECT": "M31", "EXPTIME": 12.5})],
    )

    with fitsy.open(str(path)) as _f:
        pass


def test_image_round_trip_minimal(tmp_path):
    arr = (np.arange(20, dtype=np.float64) - 10.0).reshape(4, 5)
    path = tmp_path / "b.fits"
    fitsy.write(str(path), [fitsy.image(arr)])
    f = fitsy.open(str(path))
    assert len(f) == 1
    hdu = f[0]
    assert hdu.bitpix == -64
    assert hdu.axes == [5, 4]  # FITS NAXIS order: NAXIS1=5, NAXIS2=4
    assert hdu.data.shape == (4, 5)  # numpy row-major
    np.testing.assert_array_equal(hdu.data, arr)


@pytest.mark.parametrize(
    "dtype, fill, expected_bzero",
    [
        (np.uint16, 60_000, 32_768.0),
        (np.uint32, 4_000_000_000, 2_147_483_648.0),
        (np.uint64, 18_000_000_000_000_000_000, 9_223_372_036_854_775_808.0),
        (np.int8, -100, -128.0),
    ],
)
def test_unsigned_and_int8_round_trip(tmp_path, dtype, fill, expected_bzero):
    """Writer must mirror reader: u16/u32/u64/i8 round-trip via BZERO."""
    arr = np.array([[fill, fill + 1], [fill + 2, fill + 3]], dtype=dtype)
    path = tmp_path / f"{np.dtype(dtype).name}.fits"
    fitsy.write(str(path), [fitsy.image(arr)])
    with fitsy.open(str(path)) as f:
        out = f[0].data
        assert out.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(out, arr)
        assert f[0].header["BZERO"] == pytest.approx(expected_bzero)
        assert f[0].header["BSCALE"] == 1.0


@pytest.mark.parametrize("dtype_str", [">f4", "<f4", ">i2", "<i2", ">i4", ">u2", "<u2"])
def test_non_native_byteorder_accepted(tmp_path, dtype_str):
    """Non-native-endian arrays must be accepted and round-trip correctly."""
    arr = np.array([[1, 2], [3, 4]], dtype=dtype_str)
    path = tmp_path / "byteorder.fits"
    fitsy.write(str(path), [fitsy.image(arr)])
    with fitsy.open(str(path)) as f:
        expected = arr.astype(arr.dtype.newbyteorder("="))
        np.testing.assert_array_equal(f[0].data, expected)


def test_header_dict_like(tmp_path):
    arr = np.zeros((2, 3), dtype=np.int16)
    path = tmp_path / "c.fits"
    fitsy.write(
        str(path),
        [fitsy.image(arr, header={"OBJECT": "NGC 1234", "GAIN": 1.5, "USEAO": True})],
    )
    f = fitsy.open(str(path))
    h = f[0].header
    assert "OBJECT" in h
    assert h["OBJECT"] == "NGC 1234"
    assert h["GAIN"] == 1.5
    assert h["USEAO"] is True
    assert h.get("MISSING") is None
    keys = list(h)
    assert "OBJECT" in keys


def test_bintable_round_trip(tmp_path):
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    flux = np.array([1.5, 2.5, 3.25, 4.0, 7.75], dtype=np.float64)
    path = tmp_path / "d.fits"
    fitsy.write(
        str(path),
        [
            fitsy.image(np.zeros((0,), dtype=np.uint8)),
            fitsy.bintable({"ID": ids, "FLUX": flux}, extname="CAT"),
        ],
    )
    f = fitsy.open(str(path))
    assert len(f) == 2
    t = f[1]
    assert isinstance(t, fitsy.BinTable)
    assert t.n_rows == 5
    assert set(t.column_names) == {"ID", "FLUX"}
    np.testing.assert_array_equal(t["FLUX"], flux)
    np.testing.assert_array_equal(t["ID"].astype(np.int32), ids)


def test_wcs_celestial(tmp_path):
    # Tiny TAN-projected image. Reference pixel maps to (RA, Dec)=(180, 0).
    arr = np.zeros((10, 10), dtype=np.float32)
    header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 5.0,
        "CRPIX2": 5.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }
    path = tmp_path / "wcs.fits"
    fitsy.write(str(path), [fitsy.image(arr, header=header)])
    f = fitsy.open(str(path))
    wcs = f[0].wcs()
    assert wcs is not None
    assert wcs.is_celestial
    # CRPIX=5.0 is FITS 1-based, so origin=0 puts the reference pixel at (4, 4).
    ra, dec = wcs.pixel_to_celestial(4.0, 4.0)
    assert ra == pytest.approx(180.0, abs=1e-9)
    assert dec == pytest.approx(0.0, abs=1e-9)
    px, py = wcs.celestial_to_pixel(ra, dec)
    assert px == pytest.approx(4.0, abs=1e-9)
    assert py == pytest.approx(4.0, abs=1e-9)
    # Same query with origin=1 should agree with the FITS reference value.
    ra1, dec1 = wcs.pixel_to_celestial(5.0, 5.0, origin=1)
    assert ra1 == pytest.approx(180.0, abs=1e-9)
    assert dec1 == pytest.approx(0.0, abs=1e-9)

    pts = np.array([[4.0, 4.0], [3.0, 5.0]], dtype=np.float64)
    sky = wcs.pixel_to_celestial_many(pts)
    assert sky.shape == (2, 2)
    back = wcs.celestial_to_pixel_many(sky)
    np.testing.assert_allclose(back, pts, atol=1e-9)
    # origin=1 round-trip
    pts1 = pts + 1.0
    sky1 = wcs.pixel_to_celestial_many(pts1, origin=1)
    np.testing.assert_allclose(sky1, sky, atol=1e-12)
    back1 = wcs.celestial_to_pixel_many(sky1, origin=1)
    np.testing.assert_allclose(back1, pts1, atol=1e-9)


def test_open_missing_file_raises_fits_error(tmp_path):
    with pytest.raises(fitsy.FitsError):
        fitsy.open(str(tmp_path / "does_not_exist.fits"))


def test_header_setitem_and_delitem(tmp_path):
    arr = np.zeros((2, 3), dtype=np.int16)
    path = tmp_path / "mut.fits"
    fitsy.write(str(path), [fitsy.image(arr, header={"OBJECT": "orig", "GAIN": 1.0})])
    f = fitsy.open(str(path), mode="update")
    h = f[0].header
    # Update existing keyword.
    h["OBJECT"] = "updated"
    assert h["OBJECT"] == "updated"
    # Update with comment via tuple.
    h["GAIN"] = (2.5, "e-/ADU")
    assert h["GAIN"] == 2.5
    assert h.comment("GAIN") == "e-/ADU"
    # Insert new keyword.
    h["NEWKEY"] = 42
    assert h["NEWKEY"] == 42
    # Delete.
    del h["OBJECT"]
    assert "OBJECT" not in h
    with pytest.raises(KeyError):
        del h["OBJECT"]


def test_header_mutation_is_visible_through_hdu(tmp_path):
    """Mutations on `hdu.header` must be visible via subsequent
    accesses to the same HDU's header (shared-state semantics)."""
    arr = np.zeros((2, 3), dtype=np.int16)
    path = tmp_path / "shared.fits"
    fitsy.write(str(path), [fitsy.image(arr, header={"OBJECT": "orig"})])
    f = fitsy.open(str(path), mode="update")
    hdu = f[0]
    h1 = hdu.header
    h1["OBJECT"] = "updated"
    h1["NEWKEY"] = 99
    # A fresh access on the *same* HDU must observe the mutation.
    h2 = hdu.header
    assert h2["OBJECT"] == "updated"
    assert h2["NEWKEY"] == 99
    # And mutations through the second handle propagate back.
    del h2["NEWKEY"]
    assert "NEWKEY" not in h1


def test_image_data_slice(tmp_path):
    arr = np.arange(60, dtype=np.float32).reshape(6, 10)  # NAXIS1=10, NAXIS2=6
    path = tmp_path / "region.fits"
    fitsy.write(str(path), [fitsy.image(arr)])
    f = fitsy.open(str(path))
    hdu = f[0]
    # numpy slicing replaces the old read_region helper.
    region = hdu.data[1:4, 2:6]
    assert region.shape == (3, 4)
    np.testing.assert_array_equal(region, arr[1:4, 2:6])


def test_ascii_table_round_trip(tmp_path):
    path = tmp_path / "ascii.fits"
    fitsy.write(
        str(path),
        [
            fitsy.image(np.zeros((0,), dtype=np.uint8)),
            fitsy.ascii_table(
                {
                    "ID": [1, 2, None, 9999],
                    "FLUX": [1.5, 2.25, -3.125, float("nan")],
                    "NAME": ["alpha", "beta", "g", "delta"],
                },
                tnulls={"ID": "-9999"},
                units={"FLUX": "Jy"},
                extname="CAT",
            ),
        ],
    )
    f = fitsy.open(str(path))
    assert len(f) == 2
    t = f[1]
    assert isinstance(t, fitsy.AsciiTable)
    assert t.n_rows == 4
    assert set(t.column_names) == {"ID", "FLUX", "NAME"}


def test_bintable_string_and_complex_columns(tmp_path):
    path = tmp_path / "rich.fits"
    fitsy.write(
        str(path),
        [
            fitsy.image(np.zeros((0,), dtype=np.uint8)),
            fitsy.bintable(
                {
                    "TAG": ["alpha", "beta", "gamma"],
                    "AMP": [1 + 2j, 3 + 4j, 5 + 6j],
                    "X": np.array([0.0, 1.0, 2.0], dtype=np.float64),
                },
                extname="DATA",
            ),
        ],
    )
    f = fitsy.open(str(path))
    t = f[1]
    assert isinstance(t, fitsy.BinTable)
    assert t.n_rows == 3
    # Strings round-trip through the read path.
    tags = t["TAG"]
    assert [s.strip() for s in tags] == ["alpha", "beta", "gamma"]


def test_bintable_vla_column(tmp_path):
    path = tmp_path / "vla.fits"
    fitsy.write(
        str(path),
        [
            fitsy.image(np.zeros((0,), dtype=np.uint8)),
            fitsy.bintable(
                {"COUNTS": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]},
                extname="VLA",
            ),
        ],
    )
    f = fitsy.open(str(path))
    t = f[1]
    assert t.n_rows == 3


def test_default_mode_is_read_only(tmp_path):
    arr = np.zeros((2, 3), dtype=np.int16)
    path = tmp_path / "ro.fits"
    fitsy.write(str(path), [fitsy.image(arr, header={"OBJECT": "orig"})])
    f = fitsy.open(str(path))
    assert f.read_only is True
    h = f[0].header
    assert h.read_only is True
    with pytest.raises(ValueError, match="read-only"):
        h["OBJECT"] = "nope"
    with pytest.raises(ValueError, match="read-only"):
        del h["OBJECT"]
    with pytest.raises(ValueError, match="read-only"):
        h.add_commentary("HISTORY", "nope")


def test_rw_mode_persists_mutations_across_accesses(tmp_path):
    arr = np.zeros((2, 3), dtype=np.int16)
    path = tmp_path / "rw.fits"
    fitsy.write(str(path), [fitsy.image(arr, header={"OBJECT": "orig"})])
    f = fitsy.open(str(path), mode="update")
    assert f.read_only is False
    # Mutate via one HDU access; observe via a fresh access.
    f[0].header["OBJECT"] = "patched"
    f[0].header["NEWKEY"] = (7, "added")
    h = f[0].header
    assert h["OBJECT"] == "patched"
    assert h["NEWKEY"] == 7
    assert h.comment("NEWKEY") == "added"


def test_modify_and_save_round_trip(tmp_path):
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    src = tmp_path / "src.fits"
    dst = tmp_path / "dst.fits"
    fitsy.write(str(src), [fitsy.image(arr, header={"OBJECT": "orig", "GAIN": 1.0})])
    f = fitsy.open(str(src), mode="update")
    h = f[0].header
    h["OBJECT"] = "edited"
    h["NEWKEY"] = (123, "added")
    del h["GAIN"]
    f.writeto(str(dst))
    # Source file untouched.
    f2 = fitsy.open(str(src))
    assert f2[0].header["OBJECT"] == "orig"
    assert f2[0].header["GAIN"] == 1.0
    # Destination has the mutations and original pixel data.
    f3 = fitsy.open(str(dst))
    assert f3[0].header["OBJECT"] == "edited"
    assert f3[0].header["NEWKEY"] == 123
    assert "GAIN" not in f3[0].header
    np.testing.assert_array_equal(f3[0].data, arr)


def test_write_in_read_only_mode_rejected(tmp_path):
    """Readonly handles must refuse self-writes; save-as is fine."""
    arr = np.zeros((2, 3), dtype=np.int16)
    path = tmp_path / "ro.fits"
    fitsy.write(str(path), [fitsy.image(arr)])
    f = fitsy.open(str(path))
    # writeto to a *different* path is the canonical save-as workflow
    # and must succeed even from a readonly handle (astropy parity).
    out = tmp_path / "out.fits"
    f.writeto(str(out))
    assert out.exists()
    # writeto back over the source must still fail (would mutate
    # the file the handle was opened against).
    with pytest.raises(ValueError, match="read-only"):
        f.writeto(str(path), overwrite=True)


def test_open_invalid_mode(tmp_path):
    arr = np.zeros((2, 3), dtype=np.int16)
    path = tmp_path / "x.fits"
    fitsy.write(str(path), [fitsy.image(arr)])
    with pytest.raises(ValueError, match="mode"):
        fitsy.open(str(path), mode="w")


def test_pixel_edit_round_trip(tmp_path):
    """astropy parity: in-place edits to ``hdu.data`` round-trip via writeto."""
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    src = tmp_path / "src.fits"
    dst = tmp_path / "dst.fits"
    fitsy.write(str(src), [fitsy.image(arr)])
    with fitsy.open(str(src), mode="update") as f:
        f[0].data[1, 2] = 999.0
        f.writeto(str(dst))
    with fitsy.open(str(dst)) as f2:
        assert f2[0].data[1, 2] == 999.0
        assert f2[0].data[0, 0] == 0.0


def test_append_image_hdu(tmp_path):
    """astropy parity: ``f.append(ImageHdu(arr))`` adds a new extension."""
    a = np.zeros((2, 2), dtype=np.uint8)
    src = tmp_path / "src.fits"
    fitsy.write(str(src), [fitsy.image(a)])
    with fitsy.open(str(src), mode="update") as f:
        ext = fitsy.ImageHdu(np.ones((3, 3), dtype=np.float64), name="EXT1")
        f.append(ext)
        assert len(f) == 2
        out = tmp_path / "out.fits"
        f.writeto(str(out))
    with fitsy.open(str(out)) as g:
        assert len(g) == 2
        assert g[1].header["EXTNAME"] == "EXT1"
        np.testing.assert_array_equal(g[1].data, np.ones((3, 3)))


def test_delete_hdu(tmp_path):
    a = np.zeros((2, 2), dtype=np.uint8)
    src = tmp_path / "src.fits"
    fitsy.write(
        str(src),
        [fitsy.image(a), fitsy.image(np.ones((2, 2), dtype=np.uint8), primary=False)],
    )
    with fitsy.open(str(src), mode="update") as f:
        assert len(f) == 2
        del f[1]
        assert len(f) == 1
        out = tmp_path / "out.fits"
        f.writeto(str(out))
    with fitsy.open(str(out)) as g:
        assert len(g) == 1


def _truth_tan_wcs(crpix, crval, scale_arcsec=0.4, rotation_deg=15.0):
    """Build a known TAN WCS from synthetic header keywords."""
    rot = np.deg2rad(rotation_deg)
    scale = scale_arcsec / 3600.0
    header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": crpix[0],
        "CRPIX2": crpix[1],
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CD1_1": -scale * np.cos(rot),
        "CD1_2": scale * np.sin(rot),
        "CD2_1": scale * np.sin(rot),
        "CD2_2": scale * np.cos(rot),
    }
    return header


def test_fit_wcs_recovers_synthetic_tan(tmp_path):
    crpix = (100.0, 80.0)
    crval = (210.5, -15.25)
    header = _truth_tan_wcs(crpix, crval)
    arr = np.zeros((200, 200), dtype=np.float32)
    src = tmp_path / "truth.fits"
    fitsy.write(str(src), [fitsy.image(arr, header=header)])
    truth = fitsy.open(str(src))[0].wcs()

    # Sample a 5x5 grid of (pixel, sky) pairs.
    xs = np.linspace(20, 180, 5)
    ys = np.linspace(20, 180, 5)
    pixels = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    sky = truth.pixel_to_celestial_many(pixels, origin=1)

    fit = fitsy.fit_wcs(pixels, sky, projection="TAN", origin=1)
    # Sub-milliarcsec is sufficient for clean synthetic TAN+CD data.
    # The spherical deprojection at CRVAL=(210.5, -15.25) with a 15-deg
    # rotated CD matrix accumulates ~0.1 mas of floating-point rounding,
    # so 1 mas is the realistic floor for this configuration.
    assert fit.rms_arcsec < 1e-3
    assert fit.max_arcsec < 1e-3
    assert fit.residuals_arcsec.shape == (25, 2)

    # Round-trip the fitted Wcs through to_header -> open as a header.
    fit_header = fit.wcs.to_header()
    assert fit_header["CTYPE1"].startswith("RA---TAN")


def test_fit_wcs_with_sip(tmp_path):
    """SIP fit on a TAN truth perturbed by a known polynomial."""
    crpix = (100.0, 100.0)
    crval = (10.0, 20.0)
    truth_header = _truth_tan_wcs(crpix, crval, scale_arcsec=0.3, rotation_deg=0.0)
    arr = np.zeros((200, 200), dtype=np.float32)
    src = tmp_path / "tan.fits"
    fitsy.write(str(src), [fitsy.image(arr, header=truth_header)])
    truth = fitsy.open(str(src))[0].wcs()

    xs = np.linspace(10, 190, 8)
    ys = np.linspace(10, 190, 8)
    pixels = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    sky = truth.pixel_to_celestial_many(pixels, origin=1)

    # Without SIP: this is just TAN, so RMS should be ~0 already.
    fit_lin = fitsy.fit_wcs(pixels, sky, projection="TAN", origin=1)
    assert fit_lin.rms_arcsec < 1e-6

    # With SIP order 2 (no truth distortion): coefficients should
    # come out small but the model must still be valid.
    fit_sip = fitsy.fit_wcs(pixels, sky, projection="TAN", sip_order=2, origin=1)
    assert fit_sip.rms_arcsec < 1e-3
    h = fit_sip.wcs.to_header()
    assert h["CTYPE1"].startswith("RA---TAN-SIP")
    assert h["A_ORDER"] == 2
    assert h["B_ORDER"] == 2


def test_fit_wcs_save_and_reopen(tmp_path):
    """End-to-end: fit -> save -> reopen -> use."""
    crpix = (50.0, 50.0)
    crval = (123.0, 5.5)
    truth_header = _truth_tan_wcs(crpix, crval, scale_arcsec=1.0, rotation_deg=10.0)
    arr = np.zeros((100, 100), dtype=np.float32)
    src = tmp_path / "in.fits"
    fitsy.write(str(src), [fitsy.image(arr, header=truth_header)])
    truth = fitsy.open(str(src))[0].wcs()

    pix = np.array([[10, 10], [10, 90], [90, 10], [90, 90], [50, 50]], dtype=np.float64)
    sky = truth.pixel_to_celestial_many(pix, origin=1)

    fit = fitsy.fit_wcs(pix, sky, projection="TAN", origin=1)
    assert fit.rms_arcsec < 1e-6

    # Merge the fitted header keywords into a fresh image and save.
    fit_header = fit.wcs.to_header()
    merged = {k: fit_header[k] for k in fit_header}
    out = tmp_path / "out.fits"
    fitsy.write(str(out), [fitsy.image(arr, header=merged)])

    reopened = fitsy.open(str(out))[0].wcs()
    for p in pix:
        ra1, de1 = truth.pixel_to_celestial(p[0], p[1], origin=1)
        ra2, de2 = reopened.pixel_to_celestial(p[0], p[1], origin=1)
        assert abs(ra1 - ra2) < 1e-9
        assert abs(de1 - de2) < 1e-9


def test_fit_wcs_input_validation():
    pix = np.zeros((4, 2), dtype=np.float64)
    sky = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="same number of rows"):
        fitsy.fit_wcs(pix, sky)
    bad = np.zeros((4, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="\\(N, 2\\)"):
        fitsy.fit_wcs(bad, bad)


# ---------------------------------------------------------------------
# UX behavior tests (Issue #1-#10 fixes)
# ---------------------------------------------------------------------


def test_writeto_overwrite_protection(tmp_path):
    """writeto refuses to clobber unless overwrite=True."""
    arr = np.ones((4, 4), dtype=np.int16)
    path = tmp_path / "x.fits"
    fitsy.write(str(path), [fitsy.image(arr)])
    f = fitsy.open(str(path), mode="update")
    with pytest.raises(FileExistsError):
        f.writeto(str(path))
    f.writeto(str(path), overwrite=True)


def test_writeto_rejects_empty(tmp_path):
    """An empty FitsFile cannot be written."""
    arr = np.zeros((2, 2), dtype=np.int16)
    src = tmp_path / "in.fits"
    fitsy.write(str(src), [fitsy.image(arr)])
    f = fitsy.open(str(src), mode="update")
    del f[0]
    assert len(f) == 0
    with pytest.raises(ValueError, match="zero HDUs"):
        f.writeto(str(tmp_path / "out.fits"))


def test_unknown_mode_rejected(tmp_path):
    arr = np.zeros((2, 2), dtype=np.int16)
    src = tmp_path / "x.fits"
    fitsy.write(str(src), [fitsy.image(arr)])
    with pytest.raises(ValueError, match=r'.*"rw"'):
        fitsy.open(str(src), mode="rw")
    with pytest.raises(ValueError):
        fitsy.open(str(src), mode="r+")


def test_readonly_pixels_are_immutable(tmp_path):
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    src = tmp_path / "ro.fits"
    fitsy.write(str(src), [fitsy.image(arr)])
    f = fitsy.open(str(src), mode="readonly")
    img = f[0]
    with pytest.raises(ValueError, match="read-only|WRITEABLE"):
        img.data[0, 0] = 99


def test_appended_builder_is_promoted(tmp_path):
    """After append(image_builder), subsequent f[i].data must work."""
    arr0 = np.zeros((4, 4), dtype=np.int16)
    src = tmp_path / "src.fits"
    fitsy.write(str(src), [fitsy.image(arr0)])
    f = fitsy.open(str(src), mode="update")
    arr1 = np.arange(20, dtype=np.float32).reshape(4, 5)
    f.append(fitsy.image(arr1))
    appended = f[1]
    assert hasattr(appended, "data")
    np.testing.assert_array_equal(appended.data, arr1)
    appended.data[0, 0] = 999.0
    out = tmp_path / "out.fits"
    f.writeto(str(out))
    g = fitsy.open(str(out))
    assert g[1].data[0, 0] == 999.0


def test_data_setter_restamps_header(tmp_path):
    """Replacing .data immediately updates BITPIX/NAXIS in the header."""
    arr = np.zeros((4, 5), dtype=np.int16)
    src = tmp_path / "x.fits"
    fitsy.write(str(src), [fitsy.image(arr)])
    f = fitsy.open(str(src), mode="update")
    img = f[0]
    new = np.zeros((2, 3, 4), dtype=np.float32)
    img.data = new
    assert img.header["BITPIX"] == -32
    assert img.header["NAXIS"] == 3
    assert img.header["NAXIS1"] == 4
    assert img.header["NAXIS2"] == 3
    assert img.header["NAXIS3"] == 2
    # Old NAXIS card from the 2-D layout must not linger.
    assert "NAXIS3" in img.header  # exists for new shape
    # Can serialize round-trip
    out = tmp_path / "out.fits"
    f.writeto(str(out))
    g = fitsy.open(str(out))
    np.testing.assert_array_equal(g[0].data, new)


def test_table_column_arrays_are_immutable(tmp_path):
    """Numeric table columns must reject silent in-place edits."""
    src = tmp_path / "tbl.fits"
    cols = {"A": np.arange(5, dtype=np.float64), "B": np.arange(5, dtype=np.int32)}
    fitsy.write(str(src), [fitsy.bintable(cols)])
    f = fitsy.open(str(src))
    tbl = f[1]
    a = tbl["A"]
    assert isinstance(a, np.ndarray)
    with pytest.raises(ValueError, match="read-only|WRITEABLE|assignment"):
        a[0] = 999.0


def test_synthesize_primary_when_first_is_table(tmp_path):
    """Deleting the primary image and saving must still produce a valid FITS file."""
    arr = np.zeros((4, 4), dtype=np.int16)
    cols = {"X": np.arange(3, dtype=np.float64)}
    src = tmp_path / "in.fits"
    fitsy.write(str(src), [fitsy.image(arr), fitsy.bintable(cols)])
    f = fitsy.open(str(src), mode="update")
    del f[0]
    out = tmp_path / "out.fits"
    f.writeto(str(out))
    g = fitsy.open(str(out))
    # An empty primary was prepended -> table moved to extension 1.
    assert len(g) == 2
    assert g[0].header["NAXIS"] == 0
    np.testing.assert_array_equal(g[1]["X"], np.arange(3, dtype=np.float64))


def test_layout_cards_are_immutable(tmp_path):
    """BITPIX/NAXIS edits would be silently overwritten on writeto;
    the API rejects them up front."""
    arr = np.zeros((2, 3), dtype=np.int16)
    src = tmp_path / "x.fits"
    fitsy.write(str(src), [fitsy.image(arr)])
    f = fitsy.open(str(src), mode="update")
    hdr = f[0].header
    for key in ("BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "SIMPLE", "EXTEND"):
        with pytest.raises(ValueError, match="structural card"):
            hdr[key] = 99
    with pytest.raises(ValueError, match="structural card"):
        del hdr["BITPIX"]
    # A real keyword still works.
    hdr["OBJECT"] = "M31"
    assert hdr["OBJECT"] == "M31"


def test_header_lookups_are_case_insensitive(tmp_path):
    arr = np.zeros((2, 2), dtype=np.int16)
    src = tmp_path / "x.fits"
    fitsy.write(str(src), [fitsy.image(arr, header={"OBJECT": "M31"})])
    hdr = fitsy.open(str(src))[0].header
    assert hdr["OBJECT"] == "M31"
    assert hdr["object"] == "M31"
    assert hdr["Object"] == "M31"
    assert "object" in hdr
    assert hdr.get("object") == "M31"
    assert hdr.get("missing", "fallback") == "fallback"


def test_imagehdu_constructor_deep_clones_header(tmp_path):
    """ImageHdu(arr, header=other.header) must not alias edits back."""
    arr0 = np.zeros((2, 2), dtype=np.int16)
    src = tmp_path / "x.fits"
    fitsy.write(str(src), [fitsy.image(arr0, header={"OBJECT": "orig"})])
    f = fitsy.open(str(src), mode="update")
    new = fitsy.ImageHdu(np.ones((3, 3), dtype=np.int16), header=f[0].header)
    new.header["OBJECT"] = "edited"
    assert f[0].header["OBJECT"] == "orig"
    assert new.header["OBJECT"] == "edited"


def test_imagehdu_constructor_breaks_readonly_chain(tmp_path):
    """Headers laundered through ImageHdu(...) start out writable but
    must not let writes propagate back to the read-only source."""
    arr = np.zeros((2, 2), dtype=np.int16)
    src = tmp_path / "x.fits"
    fitsy.write(str(src), [fitsy.image(arr, header={"OBJECT": "orig"})])
    f = fitsy.open(str(src), mode="readonly")  # read-only
    with pytest.raises(ValueError):
        f[0].header["OBJECT"] = "edited"  # original is locked
    new = fitsy.ImageHdu(np.zeros((2, 2), dtype=np.int16), header=f[0].header)
    new.header["OBJECT"] = "edited"  # OK on the deep clone
    assert f[0].header["OBJECT"] == "orig"


def test_missing_extname_raises_keyerror(tmp_path):
    arr = np.zeros((2, 2), dtype=np.int16)
    src = tmp_path / "x.fits"
    fitsy.write(str(src), [fitsy.image(arr)])
    f = fitsy.open(str(src))
    with pytest.raises(KeyError, match="EXTNAME"):
        _ = f["DOES_NOT_EXIST"]


# ---------------------------------------------------------------------
# WCS origin parity with astropy (origin=0 default; origin=1 for FITS)
# ---------------------------------------------------------------------


def _origin_test_wcs(tmp_path):
    header = {
        "NAXIS": 2,
        "NAXIS1": 100,
        "NAXIS2": 100,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 50.5,
        "CRPIX2": 50.5,
        "CRVAL1": 30.0,
        "CRVAL2": -10.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }
    arr = np.zeros((100, 100), dtype=np.float32)
    p = tmp_path / "origin.fits"
    fitsy.write(str(p), [fitsy.image(arr, header=header)])
    return fitsy.open(str(p))[0].wcs()


def test_wcs_origin_default_is_zero(tmp_path):
    """A pixel queried with origin=0 matches the FITS-1 query at +1."""
    wcs = _origin_test_wcs(tmp_path)
    ra0, dec0 = wcs.pixel_to_celestial(10.0, 20.0)
    ra1, dec1 = wcs.pixel_to_celestial(11.0, 21.0, origin=1)
    assert ra0 == pytest.approx(ra1, abs=1e-12)
    assert dec0 == pytest.approx(dec1, abs=1e-12)


def test_wcs_origin_inverse_round_trip(tmp_path):
    wcs = _origin_test_wcs(tmp_path)
    for origin in (0, 1):
        ra, dec = wcs.pixel_to_celestial(33.0, 44.0, origin=origin)
        px, py = wcs.celestial_to_pixel(ra, dec, origin=origin)
        assert px == pytest.approx(33.0, abs=1e-7)
        assert py == pytest.approx(44.0, abs=1e-7)


def test_wcs_origin_many_consistency(tmp_path):
    wcs = _origin_test_wcs(tmp_path)
    pts0 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    pts1 = pts0 + 1.0
    sky0 = wcs.pixel_to_celestial_many(pts0)
    sky1 = wcs.pixel_to_celestial_many(pts1, origin=1)
    np.testing.assert_allclose(sky0, sky1, atol=1e-12)
    back0 = wcs.celestial_to_pixel_many(sky0)
    back1 = wcs.celestial_to_pixel_many(sky1, origin=1)
    np.testing.assert_allclose(back0, pts0, atol=1e-9)
    np.testing.assert_allclose(back1, pts1, atol=1e-9)


def test_wcs_origin_pixel_to_world_and_back(tmp_path):
    wcs = _origin_test_wcs(tmp_path)
    w0 = wcs.pixel_to_world([5.0, 7.0])
    w1 = wcs.pixel_to_world([6.0, 8.0], origin=1)
    assert w0 == pytest.approx(w1, abs=1e-12)
    p0 = wcs.world_to_pixel(w0)
    p1 = wcs.world_to_pixel(w0, origin=1)
    assert p0[0] == pytest.approx(5.0, abs=1e-7)
    assert p1[0] == pytest.approx(6.0, abs=1e-7)


def test_wcs_origin_invalid_raises(tmp_path):
    wcs = _origin_test_wcs(tmp_path)
    with pytest.raises(ValueError, match="origin must be 0"):
        wcs.pixel_to_celestial(1.0, 1.0, origin=2)


def test_fit_wcs_origin_zero_matches_origin_one(tmp_path):
    """Fitting with origin=0 vs origin=1 (with shifted pixels) yields equivalent WCS."""
    crpix = (50.0, 50.0)
    crval = (123.0, 5.5)
    truth_header = _truth_tan_wcs(crpix, crval, scale_arcsec=1.0, rotation_deg=10.0)
    arr = np.zeros((100, 100), dtype=np.float32)
    src = tmp_path / "in.fits"
    fitsy.write(str(src), [fitsy.image(arr, header=truth_header)])
    truth = fitsy.open(str(src))[0].wcs()

    pix0 = np.array(
        [[10, 10], [10, 90], [90, 10], [90, 90], [50, 50]], dtype=np.float64
    )
    sky = truth.pixel_to_celestial_many(pix0)  # origin=0

    fit0 = fitsy.fit_wcs(pix0, sky)  # origin=0 default
    fit1 = fitsy.fit_wcs(pix0 + 1.0, sky, origin=1)
    # Both fits recover the truth to sub-mas precision.
    assert fit0.rms_arcsec < 1e-3
    assert fit1.rms_arcsec < 1e-3
    # And both fitted WCS objects describe the same celestial mapping.
    p0 = fit0.wcs.pixel_to_celestial_many(pix0)
    p1 = fit1.wcs.pixel_to_celestial_many(pix0)
    np.testing.assert_allclose(p0, p1, atol=1e-9)
