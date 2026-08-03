"""`Wcs.axis_kinds` and `Wcs.is_tabular`.

These name each axis by the coordinate it carries, so a caller finds an
axis by meaning rather than by position. The kind comes from the type
half of `CTYPEia`; `-TAB` is an algorithm and is reported separately.
"""

from __future__ import annotations

import fitsy
import pytest


def _wcs(header, shape):
    return fitsy.Wcs(
        fitsy.Header(
            {
                "SIMPLE": True,
                "BITPIX": -32,
                "NAXIS": len(shape),
                **{f"NAXIS{i + 1}": n for i, n in enumerate(reversed(shape))},
                **header,
            }
        )
    )


CUBE = {
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CTYPE3": "FREQ",
    "CRPIX1": 32.0,
    "CRPIX2": 24.0,
    "CRPIX3": 5.0,
    "CRVAL1": 202.469,
    "CRVAL2": 47.195,
    "CRVAL3": 1.4e9,
    "CDELT1": -1e-3,
    "CDELT2": 1e-3,
    "CDELT3": 1e6,
    "CUNIT3": "Hz",
}


def test_axis_kinds_name_every_axis():
    assert _wcs(CUBE, (4, 6, 8)).axis_kinds() == ["longitude", "latitude", "spectral"]


def test_axis_kinds_line_up_with_pixel_to_world():
    """Entry ``i`` names value ``i`` -- the documented contract."""
    w = _wcs(CUBE, (4, 6, 8))
    kinds = w.axis_kinds()
    world = w.pixel_to_world([31.0, 23.0, 4.0])
    assert len(kinds) == len(world)
    assert world[kinds.index("spectral")] == pytest.approx(1.4e9)
    assert world[kinds.index("longitude")] == pytest.approx(202.469)
    assert world[kinds.index("latitude")] == pytest.approx(47.195)


def test_axis_kinds_follow_a_swapped_ctype_order():
    """The kind follows the axis, not its position."""
    swapped = {
        "CTYPE1": "DEC--TAN",
        "CTYPE2": "RA---TAN",
        "CRPIX1": 32.0,
        "CRPIX2": 24.0,
        "CRVAL1": 47.195,
        "CRVAL2": 202.469,
        "CDELT1": 1e-3,
        "CDELT2": -1e-3,
    }
    w = _wcs(swapped, (6, 8))
    assert w.axis_kinds() == ["latitude", "longitude"]
    # Reading by meaning survives the swap; reading by position does not.
    world = w.pixel_to_world([31.0, 23.0])
    assert world[w.axis_kinds().index("longitude")] == pytest.approx(202.469, abs=0.05)


def test_stokes_is_named_and_unknown_types_are_linear():
    w = _wcs(
        {
            "CTYPE1": "STOKES",
            "CTYPE2": "DETX",
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CRVAL1": 1.0,
            "CRVAL2": 0.0,
            "CDELT1": 1.0,
            "CDELT2": 1.0,
        },
        (6, 8),
    )
    assert w.axis_kinds() == ["stokes", "linear"]


def _cube_with_third_axis(ctype, **extra):
    return _wcs(
        {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CTYPE3": ctype,
            "CRPIX1": 32.0,
            "CRPIX2": 24.0,
            "CRPIX3": 1.0,
            "CRVAL1": 202.469,
            "CRVAL2": 47.195,
            "CRVAL3": 0.0,
            "CDELT1": -1e-3,
            "CDELT2": 1e-3,
            "CDELT3": 1.0,
            **extra,
        },
        (4, 6, 8),
    )


# Sec.9.5.3 lets a time axis carry its scale as the CTYPE, so the kind
# cannot key off one literal spelling.
@pytest.mark.parametrize("ctype", ["TIME", "UTC", "TAI", "TT"])
def test_time_axis_is_named(ctype):
    assert _cube_with_third_axis(ctype).axis_kinds()[2] == "time"


def test_phase_axis_is_named():
    """Sec.9.6: CTYPE plus the CPERIia period."""
    w = _cube_with_third_axis("PHASE", CPERI3=1.5, CZPHS3=0.0)
    assert w.axis_kinds() == ["longitude", "latitude", "phase"]


def test_is_tabular_is_false_without_tab_and_safe_out_of_range():
    w = _wcs(CUBE, (4, 6, 8))
    assert [w.is_tabular(i) for i in range(3)] == [False, False, False]
    assert w.is_tabular(99) is False
