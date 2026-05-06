"""Astropy-parity update-mode tests.

A file opened with ``mode="update"`` must persist header edits,
``set_data``, structural changes (`append`/`del`/`insert`), and
fancy ``__setitem__`` writes on ``flush()`` and on ``__exit__``.
Pixel-aligned ``section[key] = value`` writes go through positional
``pwrite`` directly and need no flush.
"""

import fitsy
import numpy as np
import pytest


def _write_simple(path, arr=None):
    if arr is None:
        arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    fitsy.write(str(path), [fitsy.image(arr, header={"OBJECT": "orig"})])


def test_header_edit_persists_on_flush(tmp_path):
    p = tmp_path / "hdr.fits"
    _write_simple(p)
    with fitsy.open(str(p), mode="update") as f:
        f[0].header["OBJECT"] = "updated"
        f[0].header["NEWKEY"] = 7
        f.flush()

    with fitsy.open(str(p), mode="readonly") as g:
        assert g[0].header["OBJECT"] == "updated"
        assert g[0].header["NEWKEY"] == 7


def test_header_edit_persists_on_close_via_with(tmp_path):
    p = tmp_path / "hdr.fits"
    _write_simple(p)
    with fitsy.open(str(p), mode="update") as f:
        del f[0].header["OBJECT"]
        f[0].header["TARGET"] = "M31"

    with fitsy.open(str(p), mode="readonly") as g:
        assert "OBJECT" not in g[0].header
        assert g[0].header["TARGET"] == "M31"


def test_set_data_persists(tmp_path):
    p = tmp_path / "d.fits"
    _write_simple(p)
    new = np.full((3, 5), 42, dtype=np.int16)
    with fitsy.open(str(p), mode="update") as f:
        f[0].data = new

    with fitsy.open(str(p), mode="readonly") as g:
        assert g[0].data.shape == (3, 5)
        assert (g[0].data == 42).all()


def test_append_and_del_persist(tmp_path):
    p = tmp_path / "s.fits"
    _write_simple(p)
    extra = np.ones((2, 2), dtype=np.int32)
    with fitsy.open(str(p), mode="update") as f:
        f.append(fitsy.ImageHdu(extra, name="EXTRA"))
        assert len(f) == 2

    with fitsy.open(str(p), mode="readonly") as g:
        assert len(g) == 2
        assert g[1].header["EXTNAME"] == "EXTRA"
        assert (g[1].data == 1).all()

    with fitsy.open(str(p), mode="update") as f:
        del f[1]

    with fitsy.open(str(p), mode="readonly") as g:
        assert len(g) == 1


def test_pixel_patch_does_not_require_flush(tmp_path):
    """In-place pwrite is durable as soon as control returns."""
    p = tmp_path / "px.fits"
    arr = np.zeros((4, 4), dtype=np.int16)
    _write_simple(p, arr)
    with fitsy.open(str(p), mode="update") as f:
        f[0].section[1:2, 1:2] = np.array([[99]], dtype=np.int16)
        # No explicit flush -- pwrite is already durable.

    with fitsy.open(str(p), mode="readonly") as g:
        assert g[0].data[1, 1] == 99


def test_writeto_self_with_overwrite_rewrites(tmp_path):
    p = tmp_path / "self.fits"
    _write_simple(p)
    with fitsy.open(str(p), mode="update") as f:
        f[0].header["MARKER"] = 1
        f.writeto(str(p), overwrite=True)
        # The writeto-self path drops+rewrites+reopens in place;
        # reads after the call should see the persisted change.
        assert f[0].header["MARKER"] == 1

    with fitsy.open(str(p), mode="readonly") as g:
        assert g[0].header["MARKER"] == 1


def test_writeto_self_without_overwrite_raises(tmp_path):
    p = tmp_path / "self.fits"
    _write_simple(p)
    with fitsy.open(str(p), mode="update") as f:
        with pytest.raises(FileExistsError):
            f.writeto(str(p))


def test_update_mode_preserves_other_hdus(tmp_path):
    """Editing one HDU must not corrupt the others."""
    p = tmp_path / "multi.fits"
    a = np.full((2, 2), 1, dtype=np.int16)
    b = np.full((3, 3), 2, dtype=np.int32)
    c = np.full((4, 4), 3, dtype=np.float32)
    fitsy.write(
        str(p),
        [
            fitsy.image(a, primary=True),
            fitsy.image(b, primary=False, header={"EXTNAME": "B"}),
            fitsy.image(c, primary=False, header={"EXTNAME": "C"}),
        ],
    )
    with fitsy.open(str(p), mode="update") as f:
        f[1].header["TWEAKED"] = True

    with fitsy.open(str(p), mode="readonly") as g:
        assert (g[0].data == 1).all()
        assert (g[1].data == 2).all()
        assert g[1].header["TWEAKED"] is True
        assert (g[2].data == 3).all()
        assert g[2].header["EXTNAME"] == "C"


def test_readonly_rejects_mutations(tmp_path):
    """`readonly` mode must still refuse all mutating ops."""
    p = tmp_path / "ro.fits"
    _write_simple(p)
    with fitsy.open(str(p), mode="readonly") as f:
        with pytest.raises(ValueError):
            f[0].header["OBJECT"] = "x"
        with pytest.raises(ValueError):
            f.append(fitsy.ImageHdu(np.zeros((2, 2), dtype=np.int16)))
        with pytest.raises(ValueError):
            del f[0]


# ---------------------------------------------------------------------------
# Regression tests for the "structural mutation re-frames slots" pre-pass
# (silent-corruption fixes).
# ---------------------------------------------------------------------------


def _write_three(path):
    """Write a 3-HDU file: primary + two named extensions."""
    fitsy.write(
        str(path),
        [
            fitsy.image(
                np.full((2, 2), 1, dtype=np.int16),
                primary=True,
                header={"OBJECT": "prim"},
            ),
            fitsy.image(
                np.full((3, 3), 2, dtype=np.int32),
                primary=False,
                header={"EXTNAME": "B"},
            ),
            fitsy.image(
                np.full((4, 4), 3, dtype=np.float32),
                primary=False,
                header={"EXTNAME": "C"},
            ),
        ],
    )


def test_del_primary_persists_valid_fits(tmp_path):
    """`del f[0]` must promote a previously-extension HDU to primary
    on disk: the rewritten file must have exactly one ``SIMPLE = T``
    card and must reopen without error."""
    p = tmp_path / "del0.fits"
    _write_three(p)
    with fitsy.open(str(p), mode="update") as f:
        del f[0]

    with fitsy.open(str(p), mode="readonly") as g:
        assert len(g) == 2
        # Old HDU 1 (the int32 3x3) is now the primary.
        assert (g[0].data == 2).all()
        assert (g[1].data == 3).all()
    # And the raw bytes must show only one SIMPLE card.
    raw = p.read_bytes()
    assert raw.count(b"SIMPLE  =") == 1
    assert raw.count(b"XTENSION=") == 1


def test_insert_primary_persists_valid_fits(tmp_path):
    """`f.insert(0, primary)` must demote the old primary to an
    extension and the inserted HDU must become the new primary."""
    p = tmp_path / "ins0.fits"
    _write_three(p)
    new_primary = np.full((5, 5), 9, dtype=np.int16)
    with fitsy.open(str(p), mode="update") as f:
        f.insert(0, fitsy.image(new_primary, primary=True, header={"OBJECT": "new"}))

    with fitsy.open(str(p), mode="readonly") as g:
        assert len(g) == 4
        assert (g[0].data == 9).all()
        assert g[0].header["OBJECT"] == "new"
        # The original primary now lives as an extension.
        assert (g[1].data == 1).all()
    raw = p.read_bytes()
    assert raw.count(b"SIMPLE  =") == 1


def test_stale_binding_after_del_falls_back(tmp_path):
    """A pre-`del` ImageHdu wrapper may still be alive in user code.
    A subsequent `section[k] = v` write must not corrupt some other
    HDU's pixels; the stale-binding path falls back to the dirty
    flag and the change persists on flush."""
    p = tmp_path / "stale.fits"
    _write_three(p)
    with fitsy.open(str(p), mode="update") as f:
        # Materialize HDU 1, holding a wrapper.
        hdu1 = f[1]
        assert hdu1.data.shape == (3, 3)
        # Structural mutation invalidates hdu1's binding.
        del f[0]
        # hdu1 still references the same logical HDU, even though
        # its on-disk index changed from 1 -> 0.
        hdu1.data[0, 0] = 99
        f.flush()

    with fitsy.open(str(p), mode="readonly") as g:
        # The "B" extension was promoted to primary by the del.
        # Whichever slot it ended up in, the (0, 0) pixel must be 99
        # and HDU 1 (originally "C") must be untouched.
        assert g[0].data[0, 0] == 99
        assert (g[1].data == 3).all()


def test_set_data_then_section_write(tmp_path):
    """`set_data(new_shape)` followed by `section[k] = v` must not
    blow up with an out-of-bounds error from a stale binding."""
    p = tmp_path / "setd.fits"
    _write_simple(p, np.zeros((4, 4), dtype=np.int16))
    new = np.zeros((8, 8), dtype=np.int16)
    with fitsy.open(str(p), mode="update") as f:
        f[0].data = new
        # Section write at a coordinate valid for the *new* shape but
        # out of range for the old one.
        f[0].data[6, 6] = 42

    with fitsy.open(str(p), mode="readonly") as g:
        assert g[0].data.shape == (8, 8)
        assert g[0].data[6, 6] == 42


# ---------------------------------------------------------------------------
# New convenience features: close(), denywrite alias, tuple indexing,
# complex header values.
# ---------------------------------------------------------------------------


def test_close_flushes_and_is_idempotent(tmp_path):
    """`FitsFile.close()` flushes pending edits and is safe to call
    twice."""
    p = tmp_path / "close.fits"
    _write_simple(p)
    f = fitsy.open(str(p), mode="update")
    f[0].header["AFTER"] = 1
    f.close()
    f.close()  # idempotent

    with fitsy.open(str(p), mode="readonly") as g:
        assert g[0].header["AFTER"] == 1


def test_denywrite_is_readonly_alias(tmp_path):
    """`mode='denywrite'` must behave like `'readonly'` (astropy parity)."""
    p = tmp_path / "dw.fits"
    _write_simple(p)
    with fitsy.open(str(p), mode="denywrite") as f:
        assert f.read_only is True
        with pytest.raises(ValueError):
            f[0].header["X"] = 1


def test_getitem_tuple_extname_extver(tmp_path):
    """`f[("NAME", ver)]` must return the matching HDU."""
    p = tmp_path / "tup.fits"
    fitsy.write(
        str(p),
        [
            fitsy.image(np.zeros((1, 1), dtype=np.int16), primary=True),
            fitsy.image(
                np.full((2, 2), 7, dtype=np.int16),
                primary=False,
                header={"EXTNAME": "SCI", "EXTVER": 1},
            ),
            fitsy.image(
                np.full((2, 2), 8, dtype=np.int16),
                primary=False,
                header={"EXTNAME": "SCI", "EXTVER": 2},
            ),
        ],
    )
    with fitsy.open(str(p), mode="readonly") as f:
        v1 = f["SCI", 1]
        v2 = f["SCI", 2]
        assert (v1.data == 7).all()
        assert (v2.data == 8).all()


def test_complex_header_value_roundtrip(tmp_path):
    """Python `complex` must round-trip through a header card."""
    p = tmp_path / "cx.fits"
    _write_simple(p)
    with fitsy.open(str(p), mode="update") as f:
        f[0].header["CXVAL"] = complex(1.5, -2.25)
        f[0].header["CXINT"] = complex(3, 4)

    with fitsy.open(str(p), mode="readonly") as g:
        v = g[0].header["CXVAL"]
        assert complex(v) == complex(1.5, -2.25)
        i = g[0].header["CXINT"]
        assert complex(i) == complex(3, 4)
