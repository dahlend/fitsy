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


def _write_random_groups(path, n_groups=2, n_params=1, data_per_group=4):
    """Write a minimal random-groups primary HDU (Standard Sec.6).

    Hand-assembled rather than produced with ``astropy.io.fits``:
    astropy was only being used as a writer here, so depending on it
    made this test skip wherever it was absent, which included CI. The
    layout is a handful of fixed cards, and spelling them out also
    documents what the reader is expected to pick up:
    ``NAXIS1 = 0`` is the marker that the first axis is the group
    axis, ``PCOUNT`` counts parameters per group and ``GCOUNT`` counts
    groups.
    """
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                  -32",
        "NAXIS   =                    2",
        "NAXIS1  =                    0",
        f"NAXIS2  = {data_per_group:>20}",
        "GROUPS  =                    T",
        f"PCOUNT  = {n_params:>20}",
        f"GCOUNT  = {n_groups:>20}",
        "PTYPE1  = 'PARM    '",
        "END",
    ]
    block = "".join(f"{c:<80}" for c in cards).encode("ascii")
    block += b" " * (-len(block) % 2880)
    # Group g is [parameters..., data...], big-endian f32 throughout.
    values = np.arange(n_groups * (n_params + data_per_group), dtype=">f4")
    payload = values.tobytes()
    payload += b"\x00" * (-len(payload) % 2880)
    with open(path, "wb") as fh:
        fh.write(block + payload)
    return path


def test_random_groups_open():
    """The data dir has no random-groups sample, so synthesize one."""
    with tempfile.TemporaryDirectory() as td:
        path = _write_random_groups(os.path.join(td, "rg.fits"))
        f = fitsy.open(path)
        rg = f[0]
        assert type(rg).__name__ == "RandomGroups"
        assert rg.n_groups == 2
        assert rg.n_params == 1
        assert rg.data_per_group == 4
        # And the payload actually decodes: group g holds parameter
        # 5g and data 5g+1 .. 5g+4.
        for g in range(2):
            params, data = rg.group(g)
            np.testing.assert_allclose(params, [5.0 * g])
            np.testing.assert_allclose(data, np.arange(5 * g + 1, 5 * g + 5))
