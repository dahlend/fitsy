"""Array-like acceptance across the writing and WCS APIs.

Anywhere pixel data, a numeric table column, or a coordinate list is
expected, a plain Python sequence must behave exactly like the
equivalent numpy array. Real ndarrays keep their fast path: they are
used as-is, by reference, with no conversion.
"""

from __future__ import annotations

import fitsy
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# fitsy.image / compressed_image / append
# ---------------------------------------------------------------------------


def test_image_from_nested_list_matches_array(tmp_path):
    rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    from_list = str(tmp_path / "list.fits")
    from_array = str(tmp_path / "array.fits")
    fitsy.write(from_list, [fitsy.image(rows)])
    fitsy.write(from_array, [fitsy.image(np.array(rows))])
    assert open(from_list, "rb").read() == open(from_array, "rb").read()


def test_image_from_tuple_and_int_list(tmp_path):
    p = str(tmp_path / "t.fits")
    fitsy.write(p, [fitsy.image(((1, 2), (3, 4)))])
    with fitsy.open(p) as f:
        hdu = f[0]
        # numpy infers int64 for a Python int sequence, hence BITPIX 64.
        assert hdu.bitpix == 64
        np.testing.assert_array_equal(hdu.data, np.array([[1, 2], [3, 4]]))


def test_image_1d_list(tmp_path):
    p = str(tmp_path / "t.fits")
    fitsy.write(p, [fitsy.image([1.5, 2.5, 3.5])])
    with fitsy.open(p) as f:
        np.testing.assert_array_equal(f[0].data, np.array([1.5, 2.5, 3.5]))


def test_image_byteswapped_array_still_normalized(tmp_path):
    """Non-native dtypes take the conversion path, as before."""
    arr = np.arange(6, dtype=">f4").reshape(2, 3)
    p = str(tmp_path / "t.fits")
    fitsy.write(p, [fitsy.image(arr)])
    with fitsy.open(p) as f:
        assert f[0].bitpix == -32
        np.testing.assert_array_equal(f[0].data, arr)


def test_image_rejects_ragged_and_unsupported(tmp_path):
    with pytest.raises(TypeError):
        fitsy.image([[1.0, 2.0], [3.0]])
    with pytest.raises(TypeError):
        fitsy.image(np.zeros((2, 2), dtype=np.float16))


def test_compressed_image_from_list(tmp_path):
    rows = [[float(i * 4 + j) for j in range(4)] for i in range(4)]
    p = str(tmp_path / "t.fits")
    fitsy.write(p, [fitsy.compressed_image(rows)])
    with fitsy.open(p) as f:
        np.testing.assert_array_equal(f[1].data, np.array(rows))


def test_append_from_list(tmp_path):
    p = str(tmp_path / "t.fits")
    fitsy.write(p, [fitsy.image(np.zeros((2, 2), dtype=np.float32))])
    fitsy.append(p, [[7.0, 8.0], [9.0, 10.0]])
    with fitsy.open(p) as f:
        assert len(f) == 2
        np.testing.assert_array_equal(f[1].data, np.array([[7.0, 8.0], [9.0, 10.0]]))


# ---------------------------------------------------------------------------
# ImageHdu construction / data assignment
# ---------------------------------------------------------------------------


def test_image_hdu_from_list(tmp_path):
    hdu = fitsy.ImageHdu([[1.0, 2.0], [3.0, 4.0]], name="LIST")
    assert hdu.axes == [2, 2]
    assert hdu.bitpix == -64
    p = str(tmp_path / "t.fits")
    f = fitsy.FitsFile()
    f.append(hdu)
    f.writeto(p)
    with fitsy.open(p) as g:
        np.testing.assert_array_equal(g[0].data, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_image_hdu_keeps_native_array_by_reference(tmp_path):
    """A native ndarray must not be copied -- later edits still land."""
    arr = np.zeros((2, 2), dtype=np.float32)
    hdu = fitsy.ImageHdu(arr)
    assert hdu.data is arr
    arr[0, 0] = 42.0
    p = str(tmp_path / "t.fits")
    f = fitsy.FitsFile()
    f.append(hdu)
    f.writeto(p)
    with fitsy.open(p) as g:
        assert g[0].data[0, 0] == 42.0


def test_image_hdu_data_setter_accepts_list():
    hdu = fitsy.ImageHdu(np.zeros((2, 2), dtype=np.float32))
    hdu.data = [[1, 2, 3], [4, 5, 6]]
    assert hdu.axes == [3, 2]
    assert hdu.bitpix == 64
    np.testing.assert_array_equal(hdu.data, np.array([[1, 2, 3], [4, 5, 6]]))
    hdu.data = None
    assert hdu.data is None


# ---------------------------------------------------------------------------
# bintable columns
# ---------------------------------------------------------------------------


def test_bintable_numeric_list_columns_match_arrays(tmp_path):
    cols = {"f": [1.0, 2.0, 3.0], "i": [1, 2, 3]}
    from_list = str(tmp_path / "list.fits")
    from_array = str(tmp_path / "array.fits")
    fitsy.write(from_list, [fitsy.bintable(cols)])
    fitsy.write(
        from_array,
        [fitsy.bintable({k: np.array(v) for k, v in cols.items()})],
    )
    assert open(from_list, "rb").read() == open(from_array, "rb").read()


@pytest.mark.parametrize(
    "rows", [[[1.0, 2.0], [3.0]], [[1.0, 2.0], [3.0, 4.0]], [[1, 2], [3, 4]]]
)
def test_bintable_nested_lists_stay_variable_length(tmp_path, rows):
    """Regression guard: any nested numeric list is still a VLA column,
    ragged or not. Pass a numpy array for a fixed-repeat column.
    """
    p = str(tmp_path / "t.fits")
    fitsy.write(p, [fitsy.bintable({"x": rows})])
    with fitsy.open(p) as f:
        assert f[1].header["TFORM1"] == "1PD"


def test_bintable_2d_array_is_fixed_repeat(tmp_path):
    p = str(tmp_path / "t.fits")
    fitsy.write(
        p,
        [fitsy.bintable({"x": np.array([[1, 2], [3, 4]], dtype=np.int32)})],
    )
    with fitsy.open(p) as f:
        assert f[1].header["TFORM1"] == "2J"


def test_bintable_string_and_complex_columns_unchanged(tmp_path):
    p = str(tmp_path / "t.fits")
    fitsy.write(
        p,
        [fitsy.bintable({"s": ["a", "bbb"], "c": [1 + 2j, 3 + 4j]})],
    )
    with fitsy.open(p) as f:
        assert f[1].header["TFORM1"] == "3A"
        assert f[1].header["TFORM2"] == "1M"


def test_bintable_rejects_unsupported_column():
    with pytest.raises((TypeError, ValueError)):
        fitsy.bintable({"x": [object(), object()]})


# ---------------------------------------------------------------------------
# WCS batch transforms and fit_wcs
# ---------------------------------------------------------------------------


def _wcs() -> fitsy.Wcs:
    return fitsy.Wcs(
        fitsy.Header(
            {
                "NAXIS": 2,
                "NAXIS1": 400,
                "NAXIS2": 400,
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRPIX1": 200.0,
                "CRPIX2": 200.0,
                "CRVAL1": 10.0,
                "CRVAL2": -5.0,
                "CDELT1": -0.001,
                "CDELT2": 0.001,
            }
        )
    )


def test_wcs_batch_accepts_lists():
    w = _wcs()
    pix = [[0.0, 0.0], [100.0, 200.0], [399.0, 399.0]]
    np.testing.assert_allclose(w.pixel_to_world(pix), w.pixel_to_world(np.array(pix)))
    sky = w.pixel_to_world(pix)
    np.testing.assert_allclose(w.world_to_pixel(sky.tolist()), w.world_to_pixel(sky))


def test_wcs_batch_accepts_integer_dtype():
    """AllowTypeChange: a non-f64 dtype is upcast rather than rejected."""
    w = _wcs()
    pix = np.array([[0, 0], [100, 200]])
    np.testing.assert_allclose(
        w.pixel_to_world(pix),
        w.pixel_to_world(pix.astype(np.float64)),
    )


@pytest.mark.parametrize("method", ["pixel_to_world", "world_to_pixel"])
@pytest.mark.parametrize("shape", [(2, 4), (3, 1), (4, 3)])
def test_wcs_batch_rejects_a_wrong_column_count(method, shape):
    """Refuse a batch column count that is not ``naxis``.

    The flat Rust entry point sees the total length alone. A ``(2, 4)``
    array on a two-axis WCS divides evenly by that test. Without this
    check the result reshapes back to ``(2, 4)`` and is wrong.
    """
    w = _wcs()
    with pytest.raises(ValueError, match="expected a batch of shape"):
        getattr(w, method)(np.zeros(shape))


def test_wcs_batch_rejects_a_transposed_batch():
    """``(naxis, N)`` is the transpose of a batch, not a batch of N."""
    w = _wcs()
    pix = np.array([[0.0, 0.0], [100.0, 200.0], [399.0, 399.0]])
    assert np.asarray(w.pixel_to_world(pix)).shape == (3, 2)
    with pytest.raises(ValueError, match="pass the transpose"):
        w.pixel_to_world(pix.T)


def test_wcs_batch_accepts_an_empty_batch():
    """Zero points is a valid batch, not an error."""
    w = _wcs()
    assert np.asarray(w.pixel_to_world(np.zeros((0, 2)))).shape == (0, 2)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda a: np.asfortranarray(a), id="f_order"),
        pytest.param(lambda a: np.repeat(a, 2, axis=0)[::2], id="strided_rows"),
        pytest.param(lambda a: np.hstack([a, a])[:, :2], id="sliced_columns"),
        pytest.param(lambda a: a.astype(">f8"), id="byteswapped"),
        pytest.param(lambda a: a.astype(np.int64), id="int_dtype"),
    ],
)
@pytest.mark.parametrize("method", ["pixel_to_world", "world_to_pixel"])
def test_wcs_batch_layout_does_not_change_the_answer(method, make):
    """A C-contiguous batch is used in place; anything else is gathered.

    Those are two code paths, and a layout that took the wrong one would
    read the points transposed or strided and return plausible-looking
    nonsense. The two must agree exactly, not merely closely.
    """
    w = _wcs()
    base = np.array(
        [[0.0, 0.0], [100.0, 200.0], [399.0, 399.0], [12.0, 34.0]], dtype=np.float64
    )
    if method == "world_to_pixel":
        base = np.asarray(w.pixel_to_world(base))

    view = make(base)
    got = np.asarray(getattr(w, method)(view))
    want = np.asarray(getattr(w, method)(np.ascontiguousarray(view, dtype=np.float64)))
    np.testing.assert_array_equal(got, want)


def test_wcs_batch_origin_one_still_shifts():
    """`origin=1` takes the gathering path, since the shift is not a no-op."""
    w = _wcs()
    pix = np.array([[10.0, 20.0], [300.0, 250.0]])
    np.testing.assert_allclose(
        w.pixel_to_world(pix, origin=0),
        w.pixel_to_world(pix + 1.0, origin=1),
        atol=1e-12,
    )


def test_fit_wcs_accepts_lists():
    pix = [[100.0, 100.0], [200.0, 100.0], [100.0, 200.0], [200.0, 200.0]]
    sky = [[10.00, -5.00], [10.05, -5.00], [10.00, -4.95], [10.05, -4.95]]
    from_list = fitsy.fit_wcs(pix, sky, projection="TAN")
    from_array = fitsy.fit_wcs(np.array(pix), np.array(sky), projection="TAN")
    assert from_list.rms_arcsec == pytest.approx(from_array.rms_arcsec)
    np.testing.assert_allclose(from_list.residuals_arcsec, from_array.residuals_arcsec)
