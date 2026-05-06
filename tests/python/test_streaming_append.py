"""`fitsy.append(path, hdu)` streaming-append behavior (FitsAppender)."""

from __future__ import annotations

from pathlib import Path

import fitsy
import numpy as np
import pytest


def _make_base(path: Path, *, primary_shape=(8, 8), n_ext: int = 0) -> int:
    """Write a small FITS file: primary + n_ext extension images.

    Returns the byte size of the file.
    """
    primary = fitsy.image(
        np.arange(np.prod(primary_shape), dtype=np.int16).reshape(primary_shape)
    )
    builders = [primary]
    for i in range(n_ext):
        ext_arr = np.full((4, 4), i, dtype=np.float32)
        builders.append(fitsy.image(ext_arr, primary=False))
    fitsy.write(str(path), builders, overwrite=True)
    return path.stat().st_size


def test_append_grows_file_in_place(tmp_path: Path) -> None:
    p = tmp_path / "base.fits"
    base_size = _make_base(p)

    new_arr = np.ones((16, 16), dtype=np.float32)
    fitsy.append(str(p), new_arr, header={"EXTNAME": "ADDED"})

    new_size = p.stat().st_size
    # File must have grown, but only by one HDU's worth (header
    # block + data padded to 2880).  16*16*4 = 1024 data bytes
    # padded to 2880 + at least one header block 2880 = 5760.
    assert new_size > base_size
    assert (new_size - base_size) <= 8 * 2880  # generous upper bound

    with fitsy.open(str(p)) as f:
        assert len(f) == 2
        assert f[1].header["EXTNAME"].rstrip() == "ADDED"
        np.testing.assert_array_equal(f[1].data, new_arr)


def test_append_preserves_existing_hdus(tmp_path: Path) -> None:
    p = tmp_path / "base.fits"
    _make_base(p, n_ext=2)

    # Read originals into memory for comparison.
    with fitsy.open(str(p)) as f:
        originals = [(h.header.get("EXTNAME", None), np.asarray(h.data)) for h in f]

    fitsy.append(str(p), np.zeros((3, 3), dtype=np.uint8))

    with fitsy.open(str(p)) as f:
        assert len(f) == 4
        for i, (extname, data) in enumerate(originals):
            np.testing.assert_array_equal(f[i].data, data)


def test_multiple_appends(tmp_path: Path) -> None:
    p = tmp_path / "base.fits"
    _make_base(p)
    for i in range(5):
        arr = np.full((4, 4), i, dtype=np.int32)
        fitsy.append(str(p), arr, header={"EXTNAME": f"E{i}"})

    with fitsy.open(str(p)) as f:
        assert len(f) == 6
        for i in range(5):
            assert f[i + 1].header["EXTNAME"].rstrip() == f"E{i}"
            np.testing.assert_array_equal(
                f[i + 1].data, np.full((4, 4), i, dtype=np.int32)
            )


def test_append_open_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        fitsy.append(
            str(tmp_path / "no_such_file.fits"), np.zeros((2, 2), dtype=np.float32)
        )
