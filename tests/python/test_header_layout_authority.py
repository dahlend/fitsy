"""Who owns `BITPIX`/`NAXISn` on an attached header: the HDU, not the header.

The write path has always treated layout cards on a user header as
advisory and re-stamped them from the HDU's own geometry -- that is
what makes ``ImageHdu(data, header=other_hdu.header)`` work when the
donor describes a differently-shaped image. These tests pin the read
path to the same rule.
"""

from __future__ import annotations

import fitsy
import numpy as np


def _write(path, arr, **cards):
    fitsy.write(str(path), [fitsy.image(arr, header=cards or None)], overwrite=True)
    return str(path)


def test_donor_header_does_not_break_reads(tmp_path):
    """Copying a foreign header onto an open HDU must not poison its data."""
    a = _write(tmp_path / "a.fits", np.arange(12, dtype=np.float32).reshape(3, 4))
    b = _write(tmp_path / "b.fits", np.zeros((7, 9), dtype=np.float32), OBJECT="donor")

    with fitsy.open(b) as fb:
        donor = fb[0].header
        with fitsy.open(a, mode="update") as fa:
            fa[0].header.update(donor)
            # The stale NAXISn from `donor` must not affect the decode.
            np.testing.assert_array_equal(
                fa[0].data, np.arange(12, dtype=np.float32).reshape(3, 4)
            )

    with fitsy.open(a) as g:
        assert g[0].data.shape == (3, 4)
        assert g[0].header["OBJECT"] == "donor"
        # The file is re-stamped from the HDU's real geometry on write.
        assert (g[0].header["NAXIS1"], g[0].header["NAXIS2"]) == (4, 3)


def test_wcs_to_header_merge_round_trips(tmp_path):
    """`hdu.header.update(wcs.to_header())` -- attaching a fitted WCS."""
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    p = _write(tmp_path / "img.fits", arr)
    fit = fitsy.fit_wcs(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [3.0, 2.0]],
        [[10.0, -5.0], [10.003, -5.0], [10.0, -4.998], [10.003, -4.998]],
    )
    with fitsy.open(p, mode="update") as f:
        f[0].header.update(fit.wcs.to_header())

    with fitsy.open(p) as g:
        np.testing.assert_array_equal(g[0].data, arr)
        assert g[0].wcs() is not None
        assert (g[0].header["NAXIS1"], g[0].header["NAXIS2"]) == (4, 3)


def test_section_reads_survive_stale_layout_cards(tmp_path):
    """The lazy `section` path shares the read code; pin it too."""
    arr = np.arange(40, dtype=np.int16).reshape(5, 8)
    p = _write(tmp_path / "s.fits", arr)
    with fitsy.open(p, mode="update") as f:
        f[0].header.update({"NAXIS1": 0, "NAXIS2": 0})
        np.testing.assert_array_equal(f[0].section[1:3, 2:5], arr[1:3, 2:5])


def test_scaling_cards_still_come_from_the_user_header(tmp_path):
    """Only layout cards are overridden -- BZERO/BSCALE remain the user's."""
    raw = np.array([[0, 100], [200, 300]], dtype=np.int16)
    p = _write(tmp_path / "sc.fits", raw)
    with fitsy.open(p, mode="update") as f:
        f[0].header["BZERO"] = 1000.0
        f[0].header["BSCALE"] = 2.0
        np.testing.assert_allclose(f[0].data, 1000.0 + 2.0 * raw)


def test_image_hdu_accepts_mismatched_donor_header(tmp_path):
    """Construction with a foreign header keeps working (write-side rule)."""
    src = _write(tmp_path / "src.fits", np.zeros((3, 4), dtype=np.float32))
    with fitsy.open(src) as f:
        hdu = fitsy.ImageHdu(np.zeros((5, 6), dtype=np.float64), header=f[0].header)
    assert hdu.axes == [6, 5]
    assert hdu.bitpix == -64
    out = str(tmp_path / "out.fits")
    g = fitsy.FitsFile()
    g.append(hdu)
    g.writeto(out)
    with fitsy.open(out) as h:
        assert h[0].data.shape == (5, 6)
        assert h[0].bitpix == -64
