"""Module-level conveniences (``getdata`` / ``getval`` / ...),
``compressed_image`` builder, ``RandomGroups`` accessors, and
``BinTable`` row indexing / structured-array view.
"""

from __future__ import annotations

import os
import tempfile

import fitsy
import numpy as np
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


# ---------------------------------------------------------------------------
# Module-level conveniences
# ---------------------------------------------------------------------------


def _make_simple(path: str) -> np.ndarray:
    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    fitsy.write(path, [fitsy.image(arr, header={"OBJECT": "foo"})])
    return arr


def test_getdata(tmp_path):
    p = str(tmp_path / "t.fits")
    arr = _make_simple(p)
    out = fitsy.getdata(p)
    np.testing.assert_array_equal(out, arr)


def test_getdata_with_header(tmp_path):
    p = str(tmp_path / "t.fits")
    _make_simple(p)
    data, hdr = fitsy.getdata(p, header=True)
    assert data.shape == (4, 5)
    assert hdr["OBJECT"] == "foo"


def test_getheader(tmp_path):
    p = str(tmp_path / "t.fits")
    _make_simple(p)
    h = fitsy.getheader(p)
    assert h["OBJECT"] == "foo"


def test_getval_setval_delval(tmp_path):
    p = str(tmp_path / "t.fits")
    _make_simple(p)
    assert fitsy.getval(p, "OBJECT") == "foo"
    fitsy.setval(p, "NEWKEY", 42, comment="set by setval")
    assert fitsy.getval(p, "NEWKEY") == 42
    fitsy.delval(p, "NEWKEY")
    with pytest.raises(KeyError):
        fitsy.getval(p, "NEWKEY")


def test_info(tmp_path):
    p = str(tmp_path / "t.fits")
    _make_simple(p)
    rows = fitsy.info(p)
    assert len(rows) == 1
    idx, name, ver, kind, dims = rows[0]
    assert idx == 0 and ver == 1
    assert kind == "ImageHdu"
    assert list(dims) == [5, 4]


def test_module_append(tmp_path):
    p = str(tmp_path / "t.fits")
    _make_simple(p)
    fitsy.append(p, np.ones((3, 3), dtype=np.int16), header={"EXTNAME": "EXT2"})
    f = fitsy.open(p)
    assert len(f) == 2
    assert f[1].header.get("EXTNAME") == "EXT2"


# ---------------------------------------------------------------------------
# compressed_image (writer side)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int16, np.int32, np.uint8]
)
def test_compressed_image_roundtrip(tmp_path, dtype):
    arr = (np.arange(100) * 3).astype(dtype).reshape(10, 10)
    builder = fitsy.compressed_image(arr, header={"OBJECT": "ctest"})
    p = str(tmp_path / f"comp_{np.dtype(dtype).name}.fits")
    fitsy.write(p, [builder])
    f = fitsy.open(p)
    # Ext 1 should be the (decompressed) image.
    assert f[1].data.shape == (10, 10)
    np.testing.assert_array_equal(f[1].data, arr)
    # User header card survives.
    assert f[1].header.get("OBJECT") == "ctest"


def test_compressed_image_extname(tmp_path):
    arr = np.arange(40, dtype=np.float32).reshape(5, 8)
    b = fitsy.compressed_image(arr, extname="MY_EXT")
    p = str(tmp_path / "ext.fits")
    fitsy.write(p, [b])
    f = fitsy.open(p)
    assert f[1].header.get("EXTNAME") == "MY_EXT"


def test_compressed_image_custom_tile(tmp_path):
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    b = fitsy.compressed_image(arr, tile_shape=[4, 4])
    p = str(tmp_path / "tile.fits")
    fitsy.write(p, [b])
    f = fitsy.open(p)
    np.testing.assert_array_equal(f[1].data, arr)


# ---------------------------------------------------------------------------
# BinTable row / slice indexing + structured-array `data` getter
# ---------------------------------------------------------------------------


def _make_bintable(path: str) -> tuple[int, int]:
    cols = {
        "RA": np.arange(5, dtype=np.float64),
        "DEC": np.arange(5, dtype=np.float64) * -1.0,
        "ID": np.arange(5, dtype=np.int32),
    }
    fitsy.write(
        path,
        [
            fitsy.image(np.zeros((1,), dtype=np.uint8)),
            fitsy.bintable(cols, extname="CAT"),
        ],
    )
    return 5, 3


def test_bintable_row_dict(tmp_path):
    p = str(tmp_path / "bt.fits")
    _make_bintable(p)
    f = fitsy.open(p)
    tbl = f[1]
    row = tbl[0]
    assert isinstance(row, dict)
    assert row["RA"] == 0.0
    assert row["ID"] == 0
    row2 = tbl[-1]
    assert row2["RA"] == 4.0
    with pytest.raises(IndexError):
        _ = tbl[100]


def test_bintable_slice_returns_rows(tmp_path):
    p = str(tmp_path / "bt.fits")
    _make_bintable(p)
    f = fitsy.open(p)
    tbl = f[1]
    rows = tbl[1:4]
    assert isinstance(rows, list) and len(rows) == 3
    assert [r["ID"] for r in rows] == [1, 2, 3]


def test_bintable_data_structured(tmp_path):
    p = str(tmp_path / "bt.fits")
    _make_bintable(p)
    f = fitsy.open(p)
    tbl = f[1]
    arr = tbl.data
    assert arr.dtype.names == ("RA", "DEC", "ID")
    np.testing.assert_array_equal(arr["RA"], np.arange(5, dtype=np.float64))


def test_bintable_column_str_still_works(tmp_path):
    p = str(tmp_path / "bt.fits")
    _make_bintable(p)
    f = fitsy.open(p)
    tbl = f[1]
    col = tbl["RA"]
    np.testing.assert_array_equal(col, np.arange(5, dtype=np.float64))


# ---------------------------------------------------------------------------
# Random-groups wrapper
# ---------------------------------------------------------------------------


def test_random_groups_open_ldji():
    """``ldji01giq_corrtag_a.fits`` is a normal file but the data dir
    has no random-groups sample. We instead synthesize one."""
    pytest.importorskip("astropy")
    from astropy.io import fits as af

    # Build a tiny random-groups primary HDU via astropy and save it.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rg.fits")
        # 2 groups, 1 parameter, 4-pixel data array.
        data = np.arange(2 * (1 + 4), dtype=np.float32)
        gdata = np.recarray(
            (2,),
            dtype=[("PARM", "f4"), ("DATA", "f4", (4,))],
        )
        gdata["PARM"] = data[0::5]
        gdata["DATA"] = data.reshape(2, 5)[:, 1:]
        hdu = af.GroupsHDU(
            af.GroupData(
                gdata["DATA"],
                parnames=["PARM"],
                pardata=[gdata["PARM"]],
            )
        )
        hdu.writeto(path, overwrite=True)
        # Now open with fitsy and verify wrapper works.
        f = fitsy.open(path)
        rg = f[0]
        assert type(rg).__name__ == "RandomGroups"
        assert rg.n_groups == 2
        assert rg.n_params == 1
        assert rg.data_per_group == 4
