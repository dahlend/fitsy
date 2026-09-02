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


# ----------------------- headers do not lose cards ------------------


def _commentary(header, keyword: str) -> list[str]:
    """Every ``keyword`` commentary line of `header`, in order."""
    return [str(line) for line in header[keyword]]


def test_commentary_survives_image_builder(tmp_path) -> None:
    # A COMMENT/HISTORY card carries provenance, so a header handed to
    # ``image`` keeps it rather than losing it to the value-card path.
    h = fitsy.Header({"OBSERVER": "me"})
    h.add_commentary("COMMENT", "first light")
    h.add_commentary("HISTORY", "flat-fielded")
    h.add_commentary("HISTORY", "wcs fitted")
    arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    p = tmp_path / "img.fits"
    fitsy.write(str(p), [fitsy.image(arr, header=h)], overwrite=True)
    with fitsy.open(str(p)) as f:
        hdr = f[0].header
        assert hdr["OBSERVER"] == "me"
        assert _commentary(hdr, "COMMENT") == ["first light"]
        assert _commentary(hdr, "HISTORY") == ["flat-fielded", "wcs fitted"]


def test_commentary_survives_compression_round_trip(tmp_path) -> None:
    # The compressor already carries commentary into the ZIMAGE table
    # and back out; this pins the Python builder feeding it.
    h = fitsy.Header({"OBSERVER": "me"})
    h.add_commentary("COMMENT", "first light")
    h.add_commentary("HISTORY", "flat-fielded")
    arr = np.arange(64, dtype=np.int16).reshape(8, 8)
    p = tmp_path / "comp.fits"
    primary = fitsy.image(np.zeros((0,), dtype=np.int16))
    fitsy.write(
        str(p),
        [primary, fitsy.compressed_image(arr, header=h)],
        overwrite=True,
    )
    with fitsy.open(str(p)) as f:
        hdr = f[1].header
        assert hdr["OBSERVER"] == "me"
        assert _commentary(hdr, "COMMENT") == ["first light"]
        assert _commentary(hdr, "HISTORY") == ["flat-fielded"]


def test_commentary_survives_open_then_writeto(tmp_path) -> None:
    # Re-encoding an opened HDU goes through the same builder, so a
    # read-modify-write cycle keeps the original provenance.
    h = fitsy.Header({"OBSERVER": "me"})
    h.add_commentary("HISTORY", "original reduction")
    src = tmp_path / "src.fits"
    arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    fitsy.write(str(src), [fitsy.image(arr, header=h)], overwrite=True)

    dst = tmp_path / "dst.fits"
    with fitsy.open(str(src)) as f:
        f.writeto(str(dst), overwrite=True)
    with fitsy.open(str(dst)) as f:
        assert _commentary(f[0].header, "HISTORY") == ["original reduction"]


def test_duplicate_keywords_survive_image_builder(tmp_path) -> None:
    # A FITS keyword is not a key: duplicate value cards are legal and
    # order-significant, so a copy must not collapse them.
    h = fitsy.Header()
    h.insert(0, "NOTE", "one")
    h.insert(1, "NOTE", "two")
    arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    p = tmp_path / "dup.fits"
    fitsy.write(str(p), [fitsy.image(arr, header=h)], overwrite=True)
    with fitsy.open(str(p)) as f:
        assert [v for v, _ in f[0].header.cards("NOTE")] == ["one", "two"]


def test_duplicate_keywords_survive_compression(tmp_path) -> None:
    h = fitsy.Header()
    h.insert(0, "NOTE", "one")
    h.insert(1, "NOTE", "two")
    arr = np.arange(64, dtype=np.int16).reshape(8, 8)
    p = tmp_path / "dupz.fits"
    primary = fitsy.image(np.zeros((0,), dtype=np.int16))
    fitsy.write(
        str(p),
        [primary, fitsy.compressed_image(arr, header=h)],
        overwrite=True,
    )
    with fitsy.open(str(p)) as f:
        assert [v for v, _ in f[1].header.cards("NOTE")] == ["one", "two"]


def test_structural_card_keeps_its_source_comment(tmp_path) -> None:
    # The writer owns a structural card's value -- only it knows the
    # data it wrote -- but not the user's annotation on that card.
    cards = [
        "SIMPLE  =                    T / conforms to FITS standard",
        "BITPIX  =                   16 / array data type",
        "NAXIS   =                    0 / number of array dimensions",
        "END",
    ]
    h = fitsy.Header.fromstring("".join(c.ljust(80) for c in cards))
    arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    p = tmp_path / "struct.fits"
    fitsy.write(str(p), [fitsy.image(arr, header=h)], overwrite=True)
    with fitsy.open(str(p)) as f:
        hdr = f[0].header
        # Values are regenerated from the data actually written...
        assert hdr["NAXIS"] == 2
        # ...while the source's comments survive.
        assert hdr.comment("BITPIX") == "array data type"
        assert hdr.comment("SIMPLE") == "conforms to FITS standard"


def test_reserved_table_keyword_is_dropped_by_compression(tmp_path) -> None:
    # A compressed image is a BINTABLE, and the convention reserves the
    # TTYPEn/TFORMn space for its columns. cfitsio refuses to compress
    # such a header at all and astropy drops the card; fitsy drops it
    # too, so a header compressed by any of the three comes back the
    # same. Every other card is unaffected.
    h = fitsy.Header()
    h["TTYPE1"] = "collides with a table column"
    h["TCRVL1"] = 1.5
    h["OBSERVER"] = "me"
    arr = np.arange(64, dtype=np.int16).reshape(8, 8)
    p = tmp_path / "reserved.fits"
    primary = fitsy.image(np.zeros((0,), dtype=np.int16))
    with pytest.warns(UserWarning, match="reserved for use by the FITS Tiled"):
        comp = fitsy.compressed_image(arr, header=h)
    fitsy.write(str(p), [primary, comp], overwrite=True)
    with fitsy.open(str(p)) as f:
        hdr = f[1].header
        assert "TTYPE1" not in list(hdr.keys())
        assert "TCRVL1" not in list(hdr.keys())
        assert hdr["OBSERVER"] == "me"


def test_oversized_comment_continues_rather_than_truncating(tmp_path) -> None:
    # A comment too long to sit beside its value used to vanish. It now
    # continues onto CONTINUE cards, which the reader rejoins.
    comment = "a long observing note about the target " * 3
    h = fitsy.Header()
    h.set("OBJECT", "M31", comment)
    arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    p = tmp_path / "over.fits"
    fitsy.write(str(p), [fitsy.image(arr, header=h)], overwrite=True)
    with fitsy.open(str(p)) as f:
        hdr = f[0].header
        assert hdr["OBJECT"] == "M31"
        assert hdr.comment("OBJECT") == comment.rstrip()


def test_very_long_comment_survives_multiple_continuations(tmp_path) -> None:
    # Long enough to need several CONTINUE cards: each must keep the
    # chain alive or only the first fragment rejoins.
    comment = "word " * 60
    h = fitsy.Header()
    h.set("OBJECT", "M31", comment)
    arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    p = tmp_path / "verylong.fits"
    fitsy.write(str(p), [fitsy.image(arr, header=h)], overwrite=True)
    with fitsy.open(str(p)) as f:
        hdr = f[0].header
        assert hdr["OBJECT"] == "M31"
        assert hdr.comment("OBJECT") == comment.rstrip()


def test_comment_that_fits_is_untouched(tmp_path) -> None:
    h = fitsy.Header()
    h.set("OBJECT", "M31", "a normal comment")
    arr = np.arange(16, dtype=np.int16).reshape(4, 4)
    p = tmp_path / "fits.fits"
    fitsy.write(str(p), [fitsy.image(arr, header=h)], overwrite=True)
    with fitsy.open(str(p)) as f:
        assert f[0].header.comment("OBJECT") == "a normal comment"


def test_naxis0_only_file_opens_for_update(tmp_path) -> None:
    # `NAXIS = 0` means no data array. Treating the empty axis list as
    # a product of one made the updater claim a phantom pixel past the
    # end of a file that holds nothing but a header.
    p = tmp_path / "empty.fits"
    fitsy.write(str(p), [fitsy.image(np.zeros((0,), dtype=np.int16))], overwrite=True)
    with fitsy.open(str(p), mode="update") as f:
        f[0].header["TOUCHED"] = "yes"
    with fitsy.open(str(p)) as f:
        assert f[0].header["TOUCHED"] == "yes"


def test_compression_warns_about_reserved_keywords() -> None:
    # cfitsio refuses such a header outright and astropy warns; fitsy
    # warns too, so a dropped card is never a silent one.
    h = fitsy.Header()
    h["TTYPE1"] = "collides with a table column"
    h["TCRVL1"] = 1.5
    h["OBSERVER"] = "me"
    arr = np.arange(64, dtype=np.int16).reshape(8, 8)
    with pytest.warns(UserWarning, match="reserved for use by the FITS Tiled") as rec:
        fitsy.compressed_image(arr, header=h)
    warned = {str(w.message).split("'")[1] for w in rec}
    assert warned == {"TTYPE1", "TCRVL1"}, warned


def test_compression_is_quiet_for_an_ordinary_header(recwarn) -> None:
    h = fitsy.Header()
    h["OBSERVER"] = "me"
    h["EXPTIME"] = 30.0
    arr = np.arange(64, dtype=np.int16).reshape(8, 8)
    fitsy.compressed_image(arr, header=h)
    assert [w for w in recwarn if "reserved" in str(w.message)] == []
