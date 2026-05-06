"""`ImageHdu.section` lazy slicing."""

from __future__ import annotations

from pathlib import Path

import fitsy
import numpy as np


def _write(path: Path, data: np.ndarray) -> None:
    fitsy.write(str(path), [fitsy.image(data)], overwrite=True)


def test_image_section_full(tmp_path: Path) -> None:
    p = tmp_path / "s.fits"
    arr = np.arange(64, dtype=np.int16).reshape(8, 8)
    _write(p, arr)
    with fitsy.open(str(p)) as f:
        sect = f[0].section
        np.testing.assert_array_equal(sect[:, :], arr)


def test_image_section_slice(tmp_path: Path) -> None:
    p = tmp_path / "s.fits"
    arr = np.arange(100, dtype=np.int32).reshape(10, 10)
    _write(p, arr)
    with fitsy.open(str(p)) as f:
        sect = f[0].section
        np.testing.assert_array_equal(sect[2:5, 3:7], arr[2:5, 3:7])
        np.testing.assert_array_equal(sect[0], arr[0])


def test_image_section_repr(tmp_path: Path) -> None:
    p = tmp_path / "s.fits"
    arr = np.zeros((4, 5), dtype=np.float32)
    _write(p, arr)
    with fitsy.open(str(p)) as f:
        r = repr(f[0].section)
        assert "ImageSection" in r or "section" in r.lower()
