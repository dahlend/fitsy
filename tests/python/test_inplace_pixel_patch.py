"""In-place pixel patch updates via positional pwrite.

The API under test is::

    with fitsy.open(path, mode='update') as f:
        f[0].section[r0:r1, c0:c1] = arr   # writes only the patch
        f.flush()                          # fdatasync the backing file

This is the parity equivalent of astropy's mmap-backed
``hdu.data[...] = ...`` write path: O(patch) bytes touched, no
full-image rewrite.
"""

from __future__ import annotations

from pathlib import Path

import fitsy
import numpy as np
import pytest


def _write(path: Path, *arrays: np.ndarray) -> None:
    builders = [fitsy.image(a, primary=(i == 0)) for i, a in enumerate(arrays)]
    fitsy.write(str(path), builders, overwrite=True)


# --------------------------- happy paths -----------------------------


def test_section_setitem_writes_patch_to_disk(tmp_path: Path) -> None:
    p = tmp_path / "img.fits"
    arr = np.arange(8 * 12, dtype=np.int16).reshape(8, 12)
    _write(p, arr)

    patch = np.full((3, 4), 999, dtype=np.int16)
    with fitsy.open(str(p), mode="update") as f:
        f[0].section[2:5, 3:7] = patch
        # Flush is implicit on __exit__, but call it explicitly
        # to make the test independent of teardown ordering.
        f.flush()

    with fitsy.open(str(p), mode="readonly") as f:
        out = f[0].data
        expected = arr.copy()
        expected[2:5, 3:7] = 999
        np.testing.assert_array_equal(out, expected)


def test_section_setitem_full_image(tmp_path: Path) -> None:
    p = tmp_path / "img.fits"
    arr = np.zeros((6, 6), dtype=np.float32)
    _write(p, arr)

    new = np.arange(36, dtype=np.float32).reshape(6, 6)
    with fitsy.open(str(p), mode="update") as f:
        f[0].section[:, :] = new
        f.flush()

    with fitsy.open(str(p)) as f:
        np.testing.assert_array_equal(f[0].data, new)


def test_section_setitem_3d(tmp_path: Path) -> None:
    p = tmp_path / "cube.fits"
    arr = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    _write(p, arr)

    with fitsy.open(str(p), mode="update") as f:
        f[0].section[1, 0:2, 1:3] = np.array(
            [[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64
        )
        f.flush()

    expected = arr.copy()
    expected[1, 0:2, 1:3] = [[-1.0, -2.0], [-3.0, -4.0]]
    with fitsy.open(str(p)) as f:
        np.testing.assert_array_equal(f[0].data, expected)


def test_section_setitem_multi_hdu_targets_correct_hdu(tmp_path: Path) -> None:
    p = tmp_path / "multi.fits"
    a = np.zeros((4, 4), dtype=np.int32)
    b = np.ones((5, 5), dtype=np.int32)
    c = np.full((3, 3), 7, dtype=np.int32)
    _write(p, a, b, c)

    with fitsy.open(str(p), mode="update") as f:
        f[1].section[1:3, 2:4] = np.array([[10, 11], [12, 13]], dtype=np.int32)
        f.flush()

    with fitsy.open(str(p)) as f:
        np.testing.assert_array_equal(f[0].data, a)
        expected_b = b.copy()
        expected_b[1:3, 2:4] = [[10, 11], [12, 13]]
        np.testing.assert_array_equal(f[1].data, expected_b)
        np.testing.assert_array_equal(f[2].data, c)


def test_section_setitem_persists_without_explicit_flush(tmp_path: Path) -> None:
    """__exit__ should flush so the API is forgiving."""
    p = tmp_path / "img.fits"
    arr = np.zeros((4, 4), dtype=np.int16)
    _write(p, arr)

    with fitsy.open(str(p), mode="update") as f:
        f[0].section[0:2, 0:2] = np.full((2, 2), 5, dtype=np.int16)

    with fitsy.open(str(p)) as f:
        assert f[0].data[0, 0] == 5
        assert f[0].data[1, 1] == 5
        assert f[0].data[3, 3] == 0


# --------------------------- error paths -----------------------------


def test_section_setitem_readonly_rejected(tmp_path: Path) -> None:
    p = tmp_path / "img.fits"
    arr = np.zeros((4, 4), dtype=np.int16)
    _write(p, arr)

    with fitsy.open(str(p), mode="readonly") as f:
        # The cached numpy array is read-only when opened readonly; the
        # underlying file should not change. numpy raises ValueError on
        # the read-only assignment.
        with pytest.raises((ValueError, RuntimeError)):
            f[0].section[0:1, 0:1] = np.array([[42]], dtype=np.int16)


def test_section_setitem_out_of_bounds_raises(tmp_path: Path) -> None:
    p = tmp_path / "img.fits"
    arr = np.zeros((4, 4), dtype=np.int16)
    _write(p, arr)

    with fitsy.open(str(p), mode="update") as f:
        with pytest.raises((IndexError, ValueError)):
            f[0].section[3:6, 0:2] = np.zeros((3, 2), dtype=np.int16)


def test_flush_noop_on_readonly(tmp_path: Path) -> None:
    p = tmp_path / "img.fits"
    arr = np.zeros((4, 4), dtype=np.int16)
    _write(p, arr)
    with fitsy.open(str(p), mode="readonly") as f:
        # Should not raise.
        f.flush()


# --------------------------- crash recovery --------------------------
#
# In-place section writes are intentionally NOT crash-safe (matches
# astropy's mmap-backed update mode). No undo-journal tests live here.


def test_section_setitem_rejects_fancy_indexing_in_update_mode(tmp_path: Path) -> None:
    """An indexing pattern the in-place patch path can't handle must
    raise rather than silently falling back to a full-file rewrite."""
    p = tmp_path / "img.fits"
    arr = np.zeros((4, 4), dtype=np.int16)
    _write(p, arr)
    with fitsy.open(str(p), mode="update") as f:
        # Negative step is not supported by the in-place path.
        with pytest.raises(ValueError, match="in-place patch path"):
            f[0].section[::-1, :] = np.zeros((4, 4), dtype=np.int16)
