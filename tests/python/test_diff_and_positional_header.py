"""`fitsy.diff` (FitsDiff) and the positional Header API."""

from __future__ import annotations

from pathlib import Path

import fitsy
import numpy as np
import pytest

# ----------------------- diff ---------------------------------------


def _write_simple(path: Path, data: np.ndarray, **header_extras) -> None:
    builder = fitsy.image(data, header=header_extras or None)
    fitsy.write(str(path), [builder], overwrite=True)


def test_diff_identical_files(tmp_path: Path) -> None:
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    arr = np.arange(64, dtype=np.int16).reshape(8, 8)
    _write_simple(a, arr)
    _write_simple(b, arr)
    d = fitsy.diff(str(a), str(b))
    assert d.identical
    assert not bool(d)
    assert d.diff_hdu_count == 0
    assert "identical" in str(d).lower()


def test_diff_value_differs(tmp_path: Path) -> None:
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    _write_simple(a, arr, OBSERVER="alice")
    _write_simple(b, arr, OBSERVER="bob")
    d = fitsy.diff(str(a), str(b))
    assert not d.identical
    assert d.diff_hdu_count == 1
    assert "OBSERVER" in d.report()


def test_diff_data_differs(tmp_path: Path) -> None:
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    arr_a = np.zeros((4, 4), dtype=np.int16)
    arr_b = arr_a.copy()
    arr_b[0, 0] = 1
    _write_simple(a, arr_a)
    _write_simple(b, arr_b)
    d = fitsy.diff(str(a), str(b))
    assert not d.identical
    assert "data differences" in d.report()


def test_diff_hdu_count_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    arr = np.zeros((4, 4), dtype=np.int16)
    _write_simple(a, arr)
    fitsy.write(
        str(b),
        [
            fitsy.image(arr),
            fitsy.image(arr, primary=False, header={"EXTNAME": "EXTRA"}),
        ],
        overwrite=True,
    )
    d = fitsy.diff(str(a), str(b))
    assert not d.identical
    assert d.hdu_counts == (1, 2)
    assert "HDU counts differ" in d.report()


def test_diff_ignore_keywords(tmp_path: Path) -> None:
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    arr = np.zeros((4, 4), dtype=np.int16)
    _write_simple(a, arr, OBSERVER="alice")
    _write_simple(b, arr, OBSERVER="bob")
    d = fitsy.diff(
        str(a), str(b), ignore_keywords=["OBSERVER", "CHECKSUM", "DATASUM", "DATE"]
    )
    assert d.identical


# ----------------------- header positional --------------------------


def test_header_set_after(tmp_path: Path) -> None:
    arr = np.zeros((4, 4), dtype=np.int16)
    p = tmp_path / "h.fits"
    _write_simple(p, arr, FOO="x", BAR="y")
    with fitsy.open(str(p), mode="update") as f:
        h = f[0].header
        h.set("BETWEEN", "z", after="FOO")
        keys = list(h.keys())
        assert "BETWEEN" in keys
        # BETWEEN should appear immediately after FOO.
        i_foo = keys.index("FOO")
        assert keys[i_foo + 1] == "BETWEEN"


def test_header_set_before(tmp_path: Path) -> None:
    arr = np.zeros((4, 4), dtype=np.int16)
    p = tmp_path / "h.fits"
    _write_simple(p, arr, FOO="x", BAR="y")
    with fitsy.open(str(p), mode="update") as f:
        h = f[0].header
        h.set("BEFORE", "v", before="BAR")
        keys = list(h.keys())
        i_bar = keys.index("BAR")
        assert keys[i_bar - 1] == "BEFORE"


def test_header_insert_index(tmp_path: Path) -> None:
    arr = np.zeros((4, 4), dtype=np.int16)
    p = tmp_path / "h.fits"
    _write_simple(p, arr)
    with fitsy.open(str(p), mode="update") as f:
        h = f[0].header
        before = list(h.keys())
        h.insert(len(before), "ATEND", 1)
        assert list(h.keys())[-1] == "ATEND"


def test_header_insert_anchor(tmp_path: Path) -> None:
    arr = np.zeros((4, 4), dtype=np.int16)
    p = tmp_path / "h.fits"
    _write_simple(p, arr, ANCHOR="x")
    with fitsy.open(str(p), mode="update") as f:
        h = f[0].header
        h.insert("ANCHOR", "BEFORE", 0)
        h.insert("ANCHOR", "AFTER", 0, after=True)
        keys = list(h.keys())
        i = keys.index("ANCHOR")
        assert keys[i - 1] == "BEFORE"
        assert keys[i + 1] == "AFTER"


def test_header_rename_keyword(tmp_path: Path) -> None:
    arr = np.zeros((4, 4), dtype=np.int16)
    p = tmp_path / "h.fits"
    _write_simple(p, arr, OLDNAME="v")
    with fitsy.open(str(p), mode="update") as f:
        h = f[0].header
        h.rename_keyword("OLDNAME", "NEWNAME")
        assert "NEWNAME" in h
        assert "OLDNAME" not in h
        assert h["NEWNAME"] == "v"


def test_header_set_update_existing(tmp_path: Path) -> None:
    arr = np.zeros((4, 4), dtype=np.int16)
    p = tmp_path / "h.fits"
    _write_simple(p, arr, FOO="old")
    with fitsy.open(str(p), mode="update") as f:
        h = f[0].header
        h.set("FOO", "new", "updated comment")
        assert h["FOO"] == "new"
        assert h.comment("FOO") == "updated comment"


def test_header_rename_missing_raises(tmp_path: Path) -> None:
    arr = np.zeros((4, 4), dtype=np.int16)
    p = tmp_path / "h.fits"
    _write_simple(p, arr)
    with fitsy.open(str(p), mode="update") as f:
        h = f[0].header
        with pytest.raises(KeyError):
            h.rename_keyword("DOES_NOT_EXIST", "NEW")


def test_header_set_before_and_after_raises(tmp_path: Path) -> None:
    arr = np.zeros((4, 4), dtype=np.int16)
    p = tmp_path / "h.fits"
    _write_simple(p, arr, A="x", B="y")
    with fitsy.open(str(p), mode="update") as f:
        h = f[0].header
        with pytest.raises(ValueError):
            h.set("X", 1, before="A", after="B")
