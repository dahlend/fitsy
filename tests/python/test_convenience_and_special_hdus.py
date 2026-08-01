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


def _write_string_array_column(path, tform, tdim, cells, row_bytes):
    """Write a BINTABLE with one ``A`` column carrying a ``TDIM``.

    Hand-assembled for the same reason as ``_write_random_groups``: the
    only thing astropy would contribute is the writing.
    """
    primary = [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
        "END",
    ]
    table = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        f"NAXIS1  = {row_bytes:>20}",
        f"NAXIS2  = {len(cells):>20}",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    1",
        "TTYPE1  = 'S'",
        f"TFORM1  = '{tform}'",
        f"TDIM1   = '{tdim}'",
        "END",
    ]
    buf = b""
    for block in (primary, table):
        raw = "".join(f"{c:<80}" for c in block).encode("ascii")
        buf += raw + b" " * (-len(raw) % 2880)
    data = b"".join(c.encode("ascii").ljust(row_bytes, b" ") for c in cells)
    buf += data + b"\0" * (-len(data) % 2880)
    path.write_bytes(buf)


def test_tdim_on_char_column_yields_string_array(tmp_path):
    """Sec.7.3.3.2: the first ``TDIM`` axis is the width of each string,
    the rest are the array shape -- so ``15A`` + ``(5,3)`` is three
    5-character strings, not one 15-character blob."""
    p = tmp_path / "tdima.fits"
    _write_string_array_column(
        p, "15A", "(5,3)", ["alphabeta gamma", "deltaepsilzeta "], 15
    )
    with fitsy.open(str(p)) as f:
        got = f[1].column("S")
    assert got == [["alpha", "beta", "gamma"], ["delta", "epsil", "zeta"]]


def test_tdim_multidim_char_column(tmp_path):
    """The standard's own example: ``60A`` + ``(5,4,3)`` is a 4x3 array
    of 5-character strings (numpy C-order shape ``(3, 4)``)."""
    p = tmp_path / "tdim3a.fits"
    cell = "".join(f"s{j}{k}".ljust(5) for k in range(3) for j in range(4))
    _write_string_array_column(p, "60A", "(5,4,3)", [cell], 60)
    with fitsy.open(str(p)) as f:
        got = np.asarray(f[1].column("S"))
    assert got.shape == (1, 3, 4)
    assert got[0, 0].tolist() == ["s00", "s10", "s20", "s30"]


def test_char_column_without_tdim_stays_one_string(tmp_path):
    """No ``TDIM`` means ``rA`` is a single r-character string."""
    p = tmp_path / "plain.fits"
    _write_string_array_column(p, "15A", "(15)", ["alphabeta gamma"], 15)
    with fitsy.open(str(p)) as f:
        got = f[1].column("S")
    assert got == ["alphabeta gamma"]


def test_table_data_is_read_only(tmp_path):
    """``BinTable.data`` is rebuilt on every access, so an edit could
    never reach the file (or even the next read). It is frozen so a
    write raises instead of vanishing -- matching the documented
    contract and the long-standing behaviour of ``column()``."""
    p = tmp_path / "t.fits"
    _write_string_array_column(p, "15A", "(5,3)", ["alphabeta gamma"], 15)
    with fitsy.open(str(p)) as f:
        d = f[1].data
        assert d.flags.writeable is False
        with pytest.raises(ValueError):
            d[0] = d[0]


def test_string_array_column_round_trips(tmp_path):
    """``list[list[str]]`` writes an ``nA`` column with a ``TDIMn`` and
    reads back as the same array of strings (Sec.7.3.3.2)."""
    p = tmp_path / "sa.fits"
    rows = [["alpha", "beta", "gamma"], ["delta", "epsil", "zeta"]]
    fitsy.write(
        str(p),
        [fitsy.image(np.zeros((2, 2), np.int16)), fitsy.bintable({"S": rows})],
        overwrite=True,
    )
    with fitsy.open(str(p)) as f:
        assert f[1].header["TFORM1"] == "15A"
        assert f[1].header["TDIM1"] == "(5,3)"
        assert f[1].column("S") == rows


def test_flat_string_column_has_no_tdim(tmp_path):
    """A 1-D ``list[str]`` stays a plain ``nA`` column."""
    p = tmp_path / "flat.fits"
    fitsy.write(
        str(p),
        [fitsy.image(np.zeros((2, 2), np.int16)), fitsy.bintable({"S": ["ab", "cde"]})],
        overwrite=True,
    )
    with fitsy.open(str(p)) as f:
        assert f[1].header["TFORM1"] == "3A"
        assert "TDIM1" not in f[1].header
        assert f[1].column("S") == ["ab", "cde"]


def test_ragged_string_array_column_rejected(tmp_path):
    with pytest.raises(ValueError):
        fitsy.bintable({"S": [["a", "b"], ["c"]]})


@pytest.mark.parametrize(
    "columns",
    [
        {"S": ["ok", "été"]},  # flat nA column
        {"S": [["ok", "été"]]},  # nA + TDIM string array
        {"S": ["a\tb"]},  # control character
        {"S": ["a\x00b"]},  # NUL
    ],
)
def test_non_ascii_character_column_is_rejected(columns):
    """Sec.3 defines a character string as decimal 32-126 and Sec.7.2.5
    says an ``Aw`` field *shall* use it, so writing anything else is
    refused rather than silently transcoded."""
    with pytest.raises(ValueError, match="ASCII"):
        fitsy.bintable(columns)


def test_non_ascii_ascii_table_column_is_rejected():
    with pytest.raises(ValueError, match="ASCII"):
        fitsy.ascii_table({"S": ["ok", "été"]})


def test_plain_ascii_character_columns_still_accepted(tmp_path):
    p = tmp_path / "ok.fits"
    fitsy.write(
        str(p),
        [
            fitsy.image(np.zeros((2, 2), np.int16)),
            fitsy.bintable({"S": ["ok", "fine"]}),
        ],
        overwrite=True,
    )
    with fitsy.open(str(p)) as f:
        assert f[1].column("S") == ["ok", "fine"]


def test_single_element_string_array_keeps_its_tdim(tmp_path):
    """A one-element string array must keep ``TDIM``, not degrade to a
    bare ``nA``.

    Sec.7.3.5 draws the distinction by dimensionality: plain ``nA`` is
    "a one-dimensional character array" -- one string -- while ``TDIM``
    marks an *array* of strings. ``(1,1)`` is legal (product 1 <= repeat
    1) and is what makes a 1-element array read back as a 1-element
    list rather than a scalar. astropy emits the same keyword here.
    """
    p = tmp_path / "one.fits"
    img = fitsy.image(np.zeros((2, 2), np.int16))

    fitsy.write(str(p), [img, fitsy.bintable({"S": [["a"], ["b"]]})], overwrite=True)
    with fitsy.open(str(p)) as f:
        assert f[1].header["TDIM1"] == "(1,1)"
        assert f[1].column("S") == [["a"], ["b"]]

    # The flat spelling is a different data model and carries no TDIM.
    fitsy.write(str(p), [img, fitsy.bintable({"S": ["a", "b"]})], overwrite=True)
    with fitsy.open(str(p)) as f:
        assert "TDIM1" not in f[1].header
        assert f[1].column("S") == ["a", "b"]
