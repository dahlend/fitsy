"""Building :class:`fitsy.Header` objects from Python.

Covers the standalone constructor (``Header()`` / ``Header(mapping)``)
and the text/bytes parsers (:meth:`Header.fromstring` /
:meth:`Header.frombytes`), plus their round-trip with
:meth:`Header.tostring` / ``bytes(header)``.
"""

from __future__ import annotations

import fitsy
import numpy as np
import pytest

# ----------------------- constructor --------------------------------


def test_empty_header_is_writable_and_empty() -> None:
    h = fitsy.Header()
    assert len(h) == 0
    # A standalone header is writable (unlike one from a readonly file).
    h["OBJECT"] = "M31"
    assert h["OBJECT"] == "M31"
    assert len(h) == 1


def test_header_from_dict() -> None:
    h = fitsy.Header({"OBJECT": "M31", "EXPTIME": 30.0, "NCOMBINE": 3})
    assert h["OBJECT"] == "M31"
    assert h["EXPTIME"] == 30.0
    assert h["NCOMBINE"] == 3


def test_header_from_dict_folds_keyword_case() -> None:
    # Keys are folded to upper case, exactly like ``header[key] = value``.
    h = fitsy.Header({"object": "M31"})
    assert h["OBJECT"] == "M31"
    assert "OBJECT" in h


def test_header_from_dict_value_comment_tuple() -> None:
    h = fitsy.Header({"EXPTIME": (30.0, "seconds")})
    assert h["EXPTIME"] == 30.0
    assert h.comment("EXPTIME") == "seconds"


def test_header_from_header_is_independent_deep_copy() -> None:
    src = fitsy.Header({"OBJECT": "M31"})
    copy = fitsy.Header(src)
    assert copy["OBJECT"] == "M31"
    # Mutating the copy must not touch the source.
    copy["OBJECT"] = "M51"
    assert copy["OBJECT"] == "M51"
    assert src["OBJECT"] == "M31"


def test_header_from_bad_argument_raises() -> None:
    with pytest.raises(TypeError):
        fitsy.Header(42)


# ----------------------- fromstring / frombytes ---------------------


def test_tostring_fromstring_round_trip() -> None:
    h = fitsy.Header({"OBJECT": "M31", "EXPTIME": 30.0, "NAXIS1": 512})
    h.add_commentary("HISTORY", "reduced with fitsy")
    text = h.tostring()
    back = fitsy.Header.fromstring(text)
    assert back["OBJECT"] == "M31"
    assert back["EXPTIME"] == 30.0
    assert back["NAXIS1"] == 512
    assert list(back["HISTORY"]) == ["reduced with fitsy"]


def test_frombytes_round_trip() -> None:
    h = fitsy.Header({"OBJECT": "M31", "EXPTIME": 30.0})
    back = fitsy.Header.frombytes(bytes(h))
    assert back["OBJECT"] == "M31"
    assert back["EXPTIME"] == 30.0


def test_fromstring_bare_fragment_without_end_or_padding() -> None:
    # A single 80-char card, no END, not block-aligned: an END is
    # appended and the buffer is padded before parsing.
    card = "OBJECT  = 'M31'".ljust(80)
    h = fitsy.Header.fromstring(card)
    assert h["OBJECT"].startswith("M31")


def test_fromstring_empty_gives_empty_header() -> None:
    h = fitsy.Header.fromstring("")
    assert len(h) == 0


def test_fromstring_result_is_writable() -> None:
    h = fitsy.Header.fromstring(fitsy.Header({"A": 1}).tostring())
    h["B"] = 2
    assert h["B"] == 2


def test_fromstring_strict_rejects_bad_value_lenient_keeps() -> None:
    # A value that matches no standard type: value field ``12.3.4.5``.
    card = "EXPTIME =              12.3.4.5".ljust(80)
    with pytest.raises(fitsy.FitsError):
        fitsy.Header.fromstring(card, lenient=False)
    # Lenient (the default) keeps the raw text so the card still loads.
    h = fitsy.Header.fromstring(card, lenient=True)
    assert h["EXPTIME"] == "12.3.4.5"


# ----------------------- interop with the rest of the API -----------


def test_built_header_attaches_to_image_hdu(tmp_path) -> None:
    h = fitsy.Header({"OBJECT": "M31", "OBSERVER": "me"})
    arr = np.zeros((4, 4), dtype=np.int16)
    builder = fitsy.image(arr, header=h)
    p = tmp_path / "out.fits"
    fitsy.write(str(p), [builder], overwrite=True)
    with fitsy.open(str(p)) as f:
        hdr = f[0].header
        assert hdr["OBJECT"] == "M31"
        assert hdr["OBSERVER"] == "me"
