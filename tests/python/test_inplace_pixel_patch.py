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


# ------------------------- scaled images -----------------------------
#
# `BZERO`/`BSCALE` are meant to be invisible: `data` reports physical
# values, and a write takes the same units back. These files are built
# with fitsy rather than astropy, which the suite does not depend on.


def _write_scaled(path: Path, raw: np.ndarray, bzero: float, bscale: float) -> None:
    """Write `raw` as stored integers under `physical = bzero + bscale * raw`."""
    fitsy.write(
        str(path),
        [fitsy.image(raw, header={"BZERO": bzero, "BSCALE": bscale})],
        overwrite=True,
    )


def test_scaled_image_round_trips_without_double_scaling(tmp_path: Path) -> None:
    """Reading physical values and writing them back must not scale twice.

    ``hdu.data`` reports ``BZERO + BSCALE * raw``. Writing that array
    keeps the physical values, so the cards that described the integer
    storage are dropped rather than applied a second time.
    """
    src = tmp_path / "scaled.fits"
    raw = np.arange(12, dtype=np.int16).reshape(3, 4)
    _write_scaled(src, raw, 100.0, 0.5)
    physical = 100.0 + 0.5 * raw

    out = tmp_path / "rt.fits"
    with fitsy.open(str(src)) as f:
        np.testing.assert_allclose(np.asarray(f[0].data), physical)
        f[0].data = np.asarray(f[0].data)
        f.writeto(str(out))

    with fitsy.open(str(out)) as f:
        np.testing.assert_allclose(np.asarray(f[0].data), physical)
        assert "BZERO" not in f[0].header
        assert "BSCALE" not in f[0].header


def test_reading_a_scaled_image_does_not_rewrite_it(tmp_path: Path) -> None:
    """``mode='update'`` must not change a file that was only read."""
    path = tmp_path / "scaled.fits"
    _write_scaled(path, np.arange(12, dtype=np.int16).reshape(3, 4), 100.0, 0.5)
    before = path.read_bytes()

    with fitsy.open(str(path), mode="update") as f:
        _ = np.asarray(f[0].data)

    assert path.read_bytes() == before, "a read rewrote the file"


def test_section_write_takes_physical_units_on_a_scaled_image(tmp_path: Path) -> None:
    """A patch is written in the units ``data`` reports, not stored ones."""
    path = tmp_path / "scaled.fits"
    _write_scaled(path, np.arange(12, dtype=np.int16).reshape(3, 4), 100.0, 0.5)

    with fitsy.open(str(path), mode="update") as f:
        f[0].section[0:1, 0:2] = np.array([[200.0, 200.5]])

    with fitsy.open(str(path)) as f:
        # The storage is untouched: still BITPIX 16 with its scaling.
        assert f[0].header["BITPIX"] == 16
        assert f[0].header["BZERO"] == 100.0
        got = np.asarray(f[0].data)
    np.testing.assert_allclose(got[0, :2], [200.0, 200.5])
    np.testing.assert_allclose(got[0, 2:], [101.0, 101.5])


@pytest.mark.parametrize("dtype", ["uint16", "uint32", "uint64"])
def test_unsigned_patch_is_exact_at_the_top_of_the_range(
    tmp_path: Path, dtype: str
) -> None:
    """The unsigned convention must not round-trip through float.

    ``uint64`` values near ``2**64`` are not representable in ``f64``,
    so a patch staged as a float would collapse neighbouring values
    onto each other.
    """
    dt = np.dtype(dtype)
    big = int(np.iinfo(dt).max)
    path = tmp_path / f"{dtype}.fits"
    _write(path, np.array([[big - 3, big - 2], [big - 1, big]], dtype=dt))

    with fitsy.open(str(path), mode="update") as f:
        assert np.asarray(f[0].data).dtype == dt
        f[0].section[0:1, 0:2] = np.array([[big - 9, big - 8]], dtype=dt)

    with fitsy.open(str(path)) as f:
        got = np.asarray(f[0].data)
    assert int(got[0, 0]) == big - 9
    assert int(got[0, 1]) == big - 8
    assert int(got[1, 1]) == big, "the untouched pixels are unchanged"
