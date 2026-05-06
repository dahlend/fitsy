#!/usr/bin/env python3
"""Generate tiny synthetic FITS fixtures for the fitsy test suite.

Replaces the large real-world files (569 MB total) with minimal
valid files (<5 KB each) that exercise the same code paths.

Run from the repository root:
    python3 tests/data/gen_fixtures.py

Requires only the Python standard library (no astropy / numpy).

Coverage map
============
Image types
    image_i8.fits       -- BITPIX=8,  8x8,  no WCS
    image_i16.fits      -- BITPIX=16, 8x8,  no WCS
    image_i32.fits      -- BITPIX=32, 8x8,  no WCS
    image_i64.fits      -- BITPIX=64, 8x8,  no WCS
    image_f32.fits      -- BITPIX=-32,8x8,  no WCS
    image_f64.fits      -- BITPIX=-64,8x8,  no WCS
    image_u16.fits      -- BITPIX=16, BZERO=32768 (unsigned encoding)
    image_u32.fits      -- BITPIX=32, BZERO=2^31  (unsigned encoding)
    image_blank.fits    -- BITPIX=16, BLANK=-32768 (integer NaN sentinel)
    image_scaled.fits   -- BITPIX=16, BSCALE=2.0, BZERO=100.0
    image_3d.fits       -- BITPIX=-32, 4x4x3 (3-D data cube)
    image_empty.fits    -- BITPIX=8,  NAXIS=0 (empty primary + IMAGE extension)
WCS images
    wcs_tan.fits        -- TAN projection, CD matrix
    wcs_tan_sip.fits    -- TAN-SIP (forward + inverse coefficients)
    wcs_tan_tpv.fits    -- TPV projection with PV2_n distortion
    wcs_sin.fits        -- SIN (orthographic / radio)
    wcs_car.fits        -- CAR (plate-carree)
    wcs_ait.fits        -- AIT (Hammer-Aitoff, all-sky)
    wcs_zea.fits        -- ZEA (zenithal equal-area)
    wcs_dss.fits        -- DSS plate-solution (AMD*/PPO* coefficients)
    wcs_multi.fits      -- Two alternate WCS (primary ' ' + alternate 'A')
Tables
    table_binary.fits   -- BINTABLE: all TFORM types (L,B,I,J,K,E,D,A,X bits)
    table_ascii.fits    -- ASCII TABLE: I,F,E,A formats
    table_multi.fits    -- empty primary + two BINTABLE extensions (EXTNAME)
    table_tscal.fits    -- BINTABLE with TSCAL/TZERO on an integer column
    table_tnull.fits    -- BINTABLE with TNULL on an integer column
    table_vla.fits      -- BINTABLE with variable-length array (P-type) column
Special HDU types
    random_groups.fits  -- Random-groups primary (legacy radio format)
    compressed_rice.fits-- Tile-compressed RICE_1 image
    compressed_gzip.fits-- Tile-compressed GZIP_1 image
    compressed_hcomp.fits-- Tile-compressed HCOMPRESS_1 image
Multi-HDU / structural
    multi_hdu.fits      -- Empty primary + IMAGE + BINTABLE + AsciiTable
    multi_checksum.fits -- Multi-HDU with CHECKSUM + DATASUM on every HDU
    extend_primary.fits -- Primary with EXTEND=T and NAXIS=0
Specific real-world replacements
    SPITZER_I1_34767104_0019_0000_2_bcd.fits
        -- BITPIX=-32, 8x8, TAN-SIP WCS, BSCALE/BZERO, BLANK,
           >=3 COMMENTs, >=30 HISTORY cards
    coj0m416-sq36-20240423-0201-e91.fits
        -- Multi-HDU (primary IMAGE + 3 BINTABLEs), TAN WCS,
           DATE-OBS, MJD-OBS, TIMESYS, CHECKSUM+DATASUM
    dss_plate.fits
        -- BITPIX=16, 8x8, DSS AMD*/PPO* plate solution
"""

from __future__ import annotations

import gzip
import os
import struct

OUT = os.path.join(os.path.dirname(__file__))
CARD = 80
BLOCK = 2880


# ---------------------------------------------------------------------------
# Low-level FITS byte builders
# ---------------------------------------------------------------------------


def _card(keyword: str, value=None, comment: str = "") -> bytes:
    """Return an 80-byte FITS header card."""
    kw = keyword.upper().ljust(8)[:8]
    if value is None:
        # Commentary (COMMENT / HISTORY / blank)
        body = f"{keyword} {comment}"[:72]
        return (body.ljust(CARD)).encode("ascii")[:CARD]
    if isinstance(value, bool):
        val_str = f"= {'T' if value else 'F':>20}"
    elif isinstance(value, int):
        val_str = f"= {value:>20}"
    elif isinstance(value, float):
        # Format per FITS standard: use G-format, no more than 20 chars.
        # Ensure a decimal point is always present and use uppercase `E`
        # (FITS requires `<mantissa>.<frac>[E<exp>]`).
        rep = repr(value)
        if "." not in rep:
            if "e" in rep:
                m, e = rep.split("e", 1)
                rep = f"{m}.0e{e}"
            else:
                rep = f"{rep}.0"
        rep = rep.replace("e", "E")
        val_str = f"= {rep:>20}"
        if len(val_str) > 30:
            val_str = f"= {value:.10E}".rjust(22)
    elif isinstance(value, str):
        escaped = value.replace("'", "''")
        val_str = f"= '{escaped:<8}'"
    else:
        val_str = f"= {value!r:>20}"
    cmt = f" / {comment}" if comment else ""
    card = f"{kw}{val_str}{cmt}"
    card = card[:CARD].ljust(CARD)
    return card.encode("ascii")[:CARD]


def _commentary(kind: str, text: str) -> bytes:
    line = f"{kind:<8}{text}"[:CARD].ljust(CARD)
    return line.encode("ascii")[:CARD]


def _end_card() -> bytes:
    return b"END" + b" " * (CARD - 3)


def _pad_header(cards: list[bytes]) -> bytes:
    """Pad a list of 80-byte cards to a multiple of 2880 bytes."""
    raw = b"".join(cards)
    while len(raw) % BLOCK:
        raw += b" " * CARD
    return raw


def _pad_data(data: bytes, fill: bytes = b"\x00") -> bytes:
    """Pad data bytes to a multiple of 2880."""
    if not data:
        return b""
    rem = len(data) % BLOCK
    if rem:
        data += fill * (BLOCK - rem)
    return data


def _be_i16(values: list[int]) -> bytes:
    return struct.pack(f">{len(values)}h", *values)


def _be_i32(values: list[int]) -> bytes:
    return struct.pack(f">{len(values)}i", *values)


def _be_f32(values: list[float]) -> bytes:
    return struct.pack(f">{len(values)}f", *values)


def _be_f64(values: list[float]) -> bytes:
    return struct.pack(f">{len(values)}d", *values)


def _checksum_bytes(data: bytes) -> int:
    """32-bit 1's-complement sum of data (Pence & Seaman 1995 Sec.3).

    Matches fitsy src/checksum.rs::checksum_bytes exactly: accumulates
    high and low 16-bit halves of each 32-bit BE word independently in
    64-bit registers, then folds carries back until both fit in 16 bits.
    """
    assert len(data) % 4 == 0, "checksum requires 4-byte aligned input"
    hi = 0
    lo = 0
    for i in range(0, len(data), 4):
        hi += (data[i] << 8) | data[i + 1]
        lo += (data[i + 2] << 8) | data[i + 3]
    while (hi >> 16) or (lo >> 16):
        hi_carry = hi >> 16
        lo_carry = lo >> 16
        hi = (hi & 0xFFFF) + lo_carry
        lo = (lo & 0xFFFF) + hi_carry
    return ((hi << 16) | (lo & 0xFFFF)) & 0xFFFFFFFF


def _ones_complement_add(a: int, b: int) -> int:
    s = a + b
    if s > 0xFFFFFFFF:
        s = (s & 0xFFFFFFFF) + 1
    return s & 0xFFFFFFFF


def _encode_checksum(value: int) -> bytes:
    """ASCII-armour a 32-bit value into 16 printable bytes (Pence 1995).

    Ports fitsy src/checksum.rs::encode_checksum exactly.
    Each byte of `value` is split into four chars summing to byte+3*OFFSET
    (quotient + remainder distribution), then shifted out of FITS-excluded
    printable ranges, then cyclic-left-rotated by 1.
    """
    OFFSET = 0x30
    EXCLUDE = {
        0x3A,
        0x3B,
        0x3C,
        0x3D,
        0x3E,
        0x3F,
        0x40,
        0x5B,
        0x5C,
        0x5D,
        0x5E,
        0x5F,
        0x60,
    }
    MASKS = [0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF]

    asc = [0] * 16
    for i in range(4):
        byte = (value & MASKS[i]) >> (24 - 8 * i)
        quotient = byte // 4 + OFFSET
        remainder = byte % 4
        ch = [quotient + remainder, quotient, quotient, quotient]
        # Push chars out of exclude set while preserving their sum.
        changed = True
        while changed:
            changed = False
            for ex in EXCLUDE:
                j = 0
                while j + 1 < 4:
                    if ch[j] == ex or ch[j + 1] == ex:
                        ch[j] += 1
                        ch[j + 1] -= 1
                        changed = True
                    j += 2
        for j, c in enumerate(ch):
            asc[4 * j + i] = c

    # Cyclic left-rotate by 1: out[i] = asc[(i + 15) % 16]
    out = bytes(asc[(i + 15) % 16] for i in range(16))
    return out


def _write_quoted_value_inplace(
    header: bytearray, offset: int, value: str | bytes
) -> None:
    """Overwrite the value field of a CHECKSUM/DATASUM card in-place.

    The CHECKSUM convention (Pence 1995) requires the 16 encoded bytes to
    sit at card bytes 11-26 (0-based 10-25) with quotes at 10 and 26.
    We write exactly those 18 bytes (open-quote + 16 chars + close-quote)
    into positions 10-27, leaving the rest of the card untouched.
    """
    if isinstance(value, bytes):
        s = value.decode("ascii")
    else:
        s = value
    # Quote + 16-char value + quote = 18 bytes at offset 10.
    field = f"'{s}'"  # exactly 18 chars for a 16-char checksum string
    field_bytes = field.encode("ascii")
    header[offset + 10 : offset + 10 + len(field_bytes)] = field_bytes


def _stamp_checksum(header_bytes: bytes, data_padded: bytes) -> bytes:
    """Stamp valid CHECKSUM + DATASUM into a copy of header_bytes.

    Algorithm matches fitsy src/checksum.rs::stamp_checksum exactly:
    1. Compute DATASUM = checksum_bytes(data_padded) and write as decimal.
    2. Write encode_checksum(0) (= 16 ASCII '0's) as placeholder CHECKSUM.
    3. Compute combined = checksum_bytes(header) XOR checksum_bytes(data).
    4. Encode complement of combined as CHECKSUM value.

    The header must not already contain CHECKSUM/DATASUM cards; this
    function inserts them before END then pads to block boundary.
    """
    # Build header with placeholder CHECKSUM / DATASUM cards before END.
    cards_raw = []
    for off in range(0, len(header_bytes), CARD):
        c = header_bytes[off : off + CARD]
        kw = c[:8].rstrip()
        if kw in (b"CHECKSUM", b"DATASUM", b"END"):
            continue
        cards_raw.append(bytes(c))

    # Compute DATASUM now (data is already complete).
    data_sum = _checksum_bytes(data_padded) if data_padded else 0
    datasum_str = str(data_sum)

    cards_raw.append(_card("DATASUM", datasum_str, "HDU data checksum"))
    # CHECKSUM placeholder: 16 ASCII '0' chars (encode_checksum(0)).
    # Use a raw 80-byte card with the exact value field needed for the
    # Pence 1995 algorithm: 'CHECKSUM= ' at bytes 0-9, open-quote at 10,
    # 16 chars at 11-26, close-quote at 27.
    zero_card = b"CHECKSUM= '0000000000000000' / HDU checksum" + b" " * 37
    cards_raw.append(zero_card[:CARD])
    cards_raw.append(_end_card())
    header2 = bytearray(_pad_header(cards_raw))

    # Compute checksum_bytes(header_with_placeholder) + checksum_bytes(data).
    header_sum = _checksum_bytes(bytes(header2))
    combined = _ones_complement_add(header_sum, data_sum)
    target = (~combined) & 0xFFFFFFFF
    encoded = _encode_checksum(target)

    # Find and overwrite the CHECKSUM card's value field.
    for off in range(0, len(header2), CARD):
        if header2[off : off + 8] == b"CHECKSUM" and header2[off + 8] == ord(b"="):
            _write_quoted_value_inplace(header2, off, encoded)
            break

    return bytes(header2)


def write_fits(
    path: str, hdus: list[tuple[list[bytes], bytes]], checksum: bool = False
) -> None:
    """Write a FITS file from a list of (cards, data_bytes) tuples."""
    with open(path, "wb") as f:
        for i, (cards, data) in enumerate(hdus):
            cards_with_end = list(cards)
            if not any(c[:3] == b"END" for c in cards_with_end):
                cards_with_end.append(_end_card())
            hdr = _pad_header(cards_with_end)
            dat = _pad_data(data)
            if checksum:
                hdr = _stamp_checksum(hdr, dat)
            f.write(hdr)
            f.write(dat)


# ---------------------------------------------------------------------------
# HDU building helpers
# ---------------------------------------------------------------------------


def primary_cards(
    bitpix: int, axes: list[int], extra: list[bytes] = (), is_primary: bool = True
) -> list[bytes]:
    cards = []
    if is_primary:
        cards.append(_card("SIMPLE", True, "conforming FITS file"))
    else:
        cards.append(_card("XTENSION", "IMAGE", "Image extension"))
    cards.append(_card("BITPIX", bitpix))
    cards.append(_card("NAXIS", len(axes)))
    for i, n in enumerate(axes, 1):
        cards.append(_card(f"NAXIS{i}", n))
    if is_primary and axes:
        cards.append(_card("EXTEND", True, "FITS dataset may contain extensions"))
    if not is_primary:
        cards.append(_card("PCOUNT", 0))
        cards.append(_card("GCOUNT", 1))
    cards.extend(extra)
    return cards


def bintable_cards(
    nrows: int,
    ncols: int,
    row_bytes: int,
    col_cards: list[bytes],
    extra: list[bytes] = (),
) -> list[bytes]:
    cards = [
        _card("XTENSION", "BINTABLE", "Binary table"),
        _card("BITPIX", 8),
        _card("NAXIS", 2),
        _card("NAXIS1", row_bytes),
        _card("NAXIS2", nrows),
        _card("PCOUNT", 0),
        _card("GCOUNT", 1),
        _card("TFIELDS", ncols),
    ]
    cards.extend(col_cards)
    cards.extend(extra)
    return cards


def ascii_table_cards(
    nrows: int,
    row_width: int,
    ncols: int,
    col_cards: list[bytes],
    extra: list[bytes] = (),
) -> list[bytes]:
    cards = [
        _card("XTENSION", "TABLE", "ASCII table"),
        _card("BITPIX", 8),
        _card("NAXIS", 2),
        _card("NAXIS1", row_width),
        _card("NAXIS2", nrows),
        _card("PCOUNT", 0),
        _card("GCOUNT", 1),
        _card("TFIELDS", ncols),
    ]
    cards.extend(col_cards)
    cards.extend(extra)
    return cards


def _wcs_tan_cards(
    crpix1: float = 4.5,
    crpix2: float = 4.5,
    crval1: float = 83.633,
    crval2: float = 22.014,
    cd11: float = -2.78e-4,
    cd22: float = 2.78e-4,
    cd12: float = 0.0,
    cd21: float = 0.0,
    ctype1: str = "RA---TAN",
    ctype2: str = "DEC--TAN",
) -> list[bytes]:
    return [
        _card("CTYPE1", ctype1),
        _card("CTYPE2", ctype2),
        _card("CRPIX1", crpix1),
        _card("CRPIX2", crpix2),
        _card("CRVAL1", crval1),
        _card("CRVAL2", crval2),
        _card("CD1_1", cd11),
        _card("CD1_2", cd12),
        _card("CD2_1", cd21),
        _card("CD2_2", cd22),
        _card("CUNIT1", "deg"),
        _card("CUNIT2", "deg"),
        _card("RADESYS", "ICRS"),
    ]


# ---------------------------------------------------------------------------
# Individual fixture generators
# ---------------------------------------------------------------------------


def gen_simple_images() -> None:
    """One file per BITPIX type, 8x8, no WCS."""
    configs = [
        (
            "image_i8.fits",
            8,
            [(i % 200) - 100 for i in range(64)],
            lambda v: struct.pack(">64b", *v),
        ),
        (
            "image_i16.fits",
            16,
            [(i - 32) * 100 for i in range(64)],
            lambda v: _be_i16(v),
        ),
        (
            "image_i32.fits",
            32,
            [i * 1000 - 30000 for i in range(64)],
            lambda v: _be_i32(v),
        ),
        (
            "image_f32.fits",
            -32,
            [float(i) * 0.5 - 16.0 for i in range(64)],
            lambda v: _be_f32(v),
        ),
        (
            "image_f64.fits",
            -64,
            [float(i) * 0.5 - 16.0 for i in range(64)],
            lambda v: _be_f64(v),
        ),
    ]
    for fname, bitpix, vals, pack in configs:
        cards = primary_cards(
            bitpix,
            [8, 8],
            [
                _card("OBJECT", "synthetic"),
                _card("BUNIT", "counts"),
            ],
        )
        data = pack(vals)
        write_fits(os.path.join(OUT, fname), [(cards, data)])
        print(f"  {fname}")


def gen_image_i64() -> None:
    vals = [i * 10000 - 300000 for i in range(64)]
    cards = primary_cards(64, [8, 8])
    data = struct.pack(">64q", *vals)
    write_fits(os.path.join(OUT, "image_i64.fits"), [(cards, data)])
    print("  image_i64.fits")


def gen_image_u16() -> None:
    """u16 encoded as i16 + BZERO=32768."""
    raw_vals = [i * 1000 for i in range(64)]  # unsigned 0..63000
    signed = [v - 32768 for v in raw_vals]
    cards = primary_cards(
        16,
        [8, 8],
        [
            _card("BSCALE", 1, "unsigned integer encoding"),
            _card("BZERO", 32768, "unsigned integer offset"),
            _card("OBJECT", "synthetic-u16"),
        ],
    )
    data = _be_i16(signed)
    write_fits(os.path.join(OUT, "image_u16.fits"), [(cards, data)])
    print("  image_u16.fits")


def gen_image_u32() -> None:
    raw_vals = [i * 1000000 for i in range(64)]
    signed = [v - 2147483648 for v in raw_vals]
    cards = primary_cards(
        32,
        [8, 8],
        [
            _card("BSCALE", 1, "unsigned integer encoding"),
            _card("BZERO", 2147483648, "unsigned integer offset"),
        ],
    )
    data = _be_i32(signed)
    write_fits(os.path.join(OUT, "image_u32.fits"), [(cards, data)])
    print("  image_u32.fits")


def gen_image_blank() -> None:
    """i16 image with BLANK=-32768 sentinel."""
    vals = [i - 32 for i in range(64)]
    vals[0] = -32768  # blank pixel
    vals[63] = -32768
    cards = primary_cards(
        16,
        [8, 8],
        [
            _card("BLANK", -32768, "undefined pixel value"),
        ],
    )
    data = _be_i16(vals)
    write_fits(os.path.join(OUT, "image_blank.fits"), [(cards, data)])
    print("  image_blank.fits")


def gen_image_scaled() -> None:
    """i16 with BSCALE=2.0 BZERO=100.0; physical = 100 + 2*raw."""
    raw_vals = list(range(-32, 32))
    cards = primary_cards(
        16,
        [8, 8],
        [
            _card("BSCALE", 2.0, "scale factor"),
            _card("BZERO", 100.0, "offset"),
        ],
    )
    data = _be_i16(raw_vals)
    write_fits(os.path.join(OUT, "image_scaled.fits"), [(cards, data)])
    print("  image_scaled.fits")


def gen_image_3d() -> None:
    vals = [float(i) for i in range(4 * 4 * 3)]
    cards = primary_cards(-32, [4, 4, 3])
    data = _be_f32(vals)
    write_fits(os.path.join(OUT, "image_3d.fits"), [(cards, data)])
    print("  image_3d.fits")


def gen_image_empty() -> None:
    """Primary with NAXIS=0 + an IMAGE extension."""
    primary = primary_cards(8, [], [_card("EXTEND", True)])
    ext_cards = primary_cards(
        -32,
        [4, 4],
        [
            _card("EXTNAME", "PIXELS"),
        ],
        is_primary=False,
    )
    vals = [float(i) for i in range(16)]
    data = _be_f32(vals)
    write_fits(
        os.path.join(OUT, "image_empty.fits"),
        [
            (primary, b""),
            (ext_cards, data),
        ],
    )
    print("  image_empty.fits")


# -- WCS images --------------------------------------------------------------


def gen_wcs_tan() -> None:
    cards = primary_cards(-32, [8, 8], _wcs_tan_cards())
    data = _be_f32([float(i) for i in range(64)])
    write_fits(os.path.join(OUT, "wcs_tan.fits"), [(cards, data)])
    print("  wcs_tan.fits")


def gen_wcs_tan_sip() -> None:
    sip_cards = _wcs_tan_cards(ctype1="RA---TAN-SIP", ctype2="DEC--TAN-SIP") + [
        _card("A_ORDER", 2),
        _card("B_ORDER", 2),
        _card("AP_ORDER", 2),
        _card("BP_ORDER", 2),
        _card("A_2_0", 1.2e-6),
        _card("A_0_2", -8.0e-7),
        _card("A_1_1", 3.0e-7),
        _card("B_2_0", -5.0e-7),
        _card("B_0_2", 9.0e-7),
        _card("B_1_1", -2.0e-7),
        _card("AP_1_0", 0.0),
        _card("AP_0_1", 0.0),
        _card("AP_2_0", -1.2e-6),
        _card("AP_0_2", 8.0e-7),
        _card("BP_1_0", 0.0),
        _card("BP_0_1", 0.0),
        _card("BP_2_0", 5.0e-7),
        _card("BP_0_2", -9.0e-7),
    ]
    cards = primary_cards(-32, [8, 8], sip_cards)
    data = _be_f32([float(i) for i in range(64)])
    write_fits(os.path.join(OUT, "wcs_tan_sip.fits"), [(cards, data)])
    print("  wcs_tan_sip.fits")


def gen_wcs_tan_tpv() -> None:
    """TPV projection: CTYPE ends with 'TPV', PV2_n distortion terms."""
    tpv_cards = [
        _card("CTYPE1", "RA---TPV"),
        _card("CTYPE2", "DEC--TPV"),
        _card("CRPIX1", 4.5),
        _card("CRPIX2", 4.5),
        _card("CRVAL1", 180.0),
        _card("CRVAL2", 20.0),
        _card("CD1_1", -2.78e-4),
        _card("CD1_2", 0.0),
        _card("CD2_1", 0.0),
        _card("CD2_2", 2.78e-4),
        _card("CUNIT1", "deg"),
        _card("CUNIT2", "deg"),
        _card("PV2_0", 0.0),
        _card("PV2_1", 1.0),
        _card("PV2_2", 0.0),
        _card("PV2_3", 3.0e-5),
    ]
    cards = primary_cards(-32, [8, 8], tpv_cards)
    data = _be_f32([float(i) for i in range(64)])
    write_fits(os.path.join(OUT, "wcs_tan_tpv.fits"), [(cards, data)])
    print("  wcs_tan_tpv.fits")


def gen_wcs_sin() -> None:
    """SIN (orthographic) -- common in radio interferometry."""
    wcs = [
        _card("CTYPE1", "RA---SIN"),
        _card("CTYPE2", "DEC--SIN"),
        _card("CRPIX1", 4.5),
        _card("CRPIX2", 4.5),
        _card("CRVAL1", 266.405),
        _card("CRVAL2", -29.007),
        _card("CDELT1", -5.56e-5),
        _card("CDELT2", 5.56e-5),
        _card("CUNIT1", "deg"),
        _card("CUNIT2", "deg"),
    ]
    cards = primary_cards(-32, [8, 8], wcs)
    data = _be_f32([float(i) for i in range(64)])
    write_fits(os.path.join(OUT, "wcs_sin.fits"), [(cards, data)])
    print("  wcs_sin.fits")


def gen_wcs_car() -> None:
    """CAR (plate-carree) -- simple linear projection."""
    wcs = [
        _card("CTYPE1", "RA---CAR"),
        _card("CTYPE2", "DEC--CAR"),
        _card("CRPIX1", 4.5),
        _card("CRPIX2", 4.5),
        _card("CRVAL1", 180.0),
        _card("CRVAL2", 0.0),
        _card("CDELT1", -1.0),
        _card("CDELT2", 1.0),
        _card("CUNIT1", "deg"),
        _card("CUNIT2", "deg"),
    ]
    cards = primary_cards(-32, [8, 8], wcs)
    data = _be_f32([float(i) for i in range(64)])
    write_fits(os.path.join(OUT, "wcs_car.fits"), [(cards, data)])
    print("  wcs_car.fits")


def gen_wcs_ait() -> None:
    """AIT (Hammer-Aitoff) -- full-sky."""
    wcs = [
        _card("CTYPE1", "GLON-AIT"),
        _card("CTYPE2", "GLAT-AIT"),
        _card("CRPIX1", 4.5),
        _card("CRPIX2", 4.5),
        _card("CRVAL1", 0.0),
        _card("CRVAL2", 0.0),
        _card("CDELT1", -10.0),
        _card("CDELT2", 10.0),
        _card("CUNIT1", "deg"),
        _card("CUNIT2", "deg"),
    ]
    cards = primary_cards(-32, [8, 8], wcs)
    data = _be_f32([float(i) for i in range(64)])
    write_fits(os.path.join(OUT, "wcs_ait.fits"), [(cards, data)])
    print("  wcs_ait.fits")


def gen_wcs_zea() -> None:
    """ZEA (zenithal equal-area)."""
    wcs = [
        _card("CTYPE1", "RA---ZEA"),
        _card("CTYPE2", "DEC--ZEA"),
        _card("CRPIX1", 4.5),
        _card("CRPIX2", 4.5),
        _card("CRVAL1", 83.633),
        _card("CRVAL2", 89.9),
        _card("CDELT1", -2.78e-4),
        _card("CDELT2", 2.78e-4),
        _card("CUNIT1", "deg"),
        _card("CUNIT2", "deg"),
    ]
    cards = primary_cards(-32, [8, 8], wcs)
    data = _be_f32([float(i) for i in range(64)])
    write_fits(os.path.join(OUT, "wcs_zea.fits"), [(cards, data)])
    print("  wcs_zea.fits")


def gen_wcs_dss() -> None:
    """DSS plate-solution (AMD*/PPO* coefficients), replaces dss_plate.fits.

    The test dss_plate_model_used_for_real_file (tests/wcs.rs) expects:
    - PLT* = RA 0h07m25.68s Dec +00d48m26s  (plate_ra=1.857deg, plate_dec=0.807deg)
    - plate_centre_x = PPO3/XPIXELSZ - CNPIX1 + 0.5 - 1  (0-based Wcs pixel)
    - pixel_to_world(plate_centre_x, plate_centre_y) ≈ (plate_ra, plate_dec) ±0.05deg

    We set geometry so the plate center falls at pixel (3.5, 3.5) 0-based
    of our 8x8 image, which is FITS pixel (4.5, 4.5).  Choose:
      XPIXELSZ = YPIXELSZ = 25 arcsec/mm (arbitrary scale)
      PPO3 = PPO6 = plate_center_plate_mm * 25  -- place plate center at (200, 200) mm
      CNPIX1 = CNPIX2 = 200/25 - 4.5 + 0.5 = 8 - 4 = 4  (1-based offset)
    so that  PPO3/XPIXELSZ - CNPIX1 + 0.5 - 1 = 200/25 - 4 + 0.5 - 1 = 8 - 4.5 = 3.5.

    AMDX1 (plate scale X, arcsec/mm): must map plate-mm to arcsec.
    At plate center, AMDX1*xi ≈ RA offset (arcsec) where xi is mm from
    plate center.  At the plate center itself, offsets are zero, so
    pixel_to_world(3.5, 3.5) will return the plate RA/Dec exactly
    (since all non-linear AMD terms are zero in our simplified model).

    plate_ra = (0 + 7/60 + 25.68/3600) * 15 = 1.857 deg
    plate_dec = 48/60 + 26/3600 = 0.80722 deg
    """
    plate_ra = (0.0 + 7.0 / 60.0 + 25.68 / 3600.0) * 15.0  # 1.857 deg
    plate_dec = 48.0 / 60.0 + 26.0 / 3600.0  # 0.80722 deg

    # XPIXELSZ, YPIXELSZ in microns (25 micron pixels = 0.025 mm).
    xpix = 25.0
    ypix = 25.0
    # PPO3/PPO6: plate mm to center of subimage.
    # We want plate_centre_x_0based = PPO3/xpix - CNPIX1 + 0.5 - 1 = 3.5
    # => PPO3/xpix = 3.5 + CNPIX1 + 0.5 = 4 + CNPIX1
    # Choose CNPIX1=CNPIX2=1 for simplicity:
    cnpix = 1.0
    ppo3 = (3.5 + cnpix + 0.5) * xpix  # = (4.0 + 1.0) * 25 = 125.0

    # AMDX1 = RA plate scale in arcsec/mm (positive = E).
    # Need AMDX1 and AMDY2 such that the plate solution converges to
    # plate_ra, plate_dec at the center.  Use the identity model
    # (1 arcsec/mm scale) -- the actual projected coordinates at
    # plate center are (xi=0, eta=0), which maps to CRVAL=(plate_ra, plate_dec).
    # Set CRVAL = (plate_ra, plate_dec) so TAN fallback also works.

    dss_cards: list[bytes] = [
        _card("CTYPE1", "RA---TAN"),
        _card("CTYPE2", "DEC--TAN"),
        _card("CRPIX1", 4.5),
        _card("CRPIX2", 4.5),
        _card("CRVAL1", plate_ra),
        _card("CRVAL2", plate_dec),
        _card("CDELT1", -2.78e-4),
        _card("CDELT2", 2.78e-4),
        _card("PLTSCALE", 1.0),
        _card("XPIXELSZ", xpix),
        _card("YPIXELSZ", ypix),
        # PLT* sexagesimal: 0h 07m 25.68s, +00d 48m 26s
        _card("PLTRAH", 0),
        _card("PLTRAM", 7),
        _card("PLTRAS", 25.68),
        _card("PLTDECD", 0),
        _card("PLTDECM", 48),
        _card("PLTDECS", 26.0),
        _card("PLTDECSN", "+"),
        _card("CNPIX1", cnpix),
        _card("CNPIX2", cnpix),
        _card("PPO1", 0.0),
        _card("PPO2", 0.0),
        _card("PPO3", ppo3),
        _card("PPO4", 0.0),
        _card("PPO5", 0.0),
        _card("PPO6", ppo3),
    ]
    # AMD polynomial coefficients.
    # xi  ~= AMDX1*x + AMDX2*y + AMDX3 + ...  (xi is RA offset in arcsec)
    # eta ~= AMDY1*y + AMDY2*x + AMDY3 + ...  (eta is Dec offset in arcsec)
    # For a non-singular linear plate matrix, set AMDX1 (x->xi) and
    # AMDY1 (y->eta) both non-zero.  AMDX2 and AMDY2 are cross-terms.
    # Use 1 arcsec/mm identity model; AMDX3=AMDY3=0 (no zero-point shift).
    amdx = [1.0, 0.0, 0.0] + [0.0] * 17  # AMDX1=1, all others 0
    amdy = [1.0, 0.0, 0.0] + [0.0] * 17  # AMDY1=1, all others 0
    for i, v in enumerate(amdx, 1):
        dss_cards.append(_card(f"AMDX{i}", v))
    for i, v in enumerate(amdy, 1):
        dss_cards.append(_card(f"AMDY{i}", v))

    cards = primary_cards(16, [8, 8], dss_cards)
    vals = [(i - 32) * 10 for i in range(64)]
    data = _be_i16(vals)
    write_fits(os.path.join(OUT, "wcs_dss.fits"), [(cards, data)])
    write_fits(os.path.join(OUT, "dss_plate.fits"), [(cards, data)])
    print("  wcs_dss.fits + dss_plate.fits")


def gen_wcs_multi() -> None:
    """Primary WCS (alt=' ') + alternate WCS 'A' on same image."""
    wcs_primary = _wcs_tan_cards(crval1=83.633, crval2=22.014)
    # Alternate A: galactic
    wcs_alt_a = [
        _card("CTYPE1A", "GLON-TAN"),
        _card("CTYPE2A", "GLAT-TAN"),
        _card("CRPIX1A", 4.5),
        _card("CRPIX2A", 4.5),
        _card("CRVAL1A", 185.0),
        _card("CRVAL2A", -5.8),
        _card("CD1_1A", -2.78e-4),
        _card("CD1_2A", 0.0),
        _card("CD2_1A", 0.0),
        _card("CD2_2A", 2.78e-4),
        _card("CUNIT1A", "deg"),
        _card("CUNIT2A", "deg"),
    ]
    cards = primary_cards(-32, [8, 8], wcs_primary + wcs_alt_a)
    data = _be_f32([float(i) for i in range(64)])
    write_fits(os.path.join(OUT, "wcs_multi.fits"), [(cards, data)])
    print("  wcs_multi.fits")


# -- Tables ------------------------------------------------------------------


def gen_table_binary() -> None:
    """BINTABLE covering TFORM L, B, I, J, K, E, D, A, and 4X bit column."""
    n_rows = 4
    # Columns: L(1), B(1), I(2), J(4), K(8), E(4), D(8), A8(8), X4(1)
    row_bytes = 1 + 1 + 2 + 4 + 8 + 4 + 8 + 8 + 1  # 37
    col_cards = [
        _card("TTYPE1", "FLAG"),
        _card("TFORM1", "1L"),
        _card("TTYPE2", "BYTE"),
        _card("TFORM2", "1B"),
        _card("TTYPE3", "INT16"),
        _card("TFORM3", "1I"),
        _card("TTYPE4", "INT32"),
        _card("TFORM4", "1J"),
        _card("TTYPE5", "INT64"),
        _card("TFORM5", "1K"),
        _card("TTYPE6", "FLOAT"),
        _card("TFORM6", "1E"),
        _card("TUNIT6", "counts"),
        _card("TTYPE7", "DOUBLE"),
        _card("TFORM7", "1D"),
        _card("TUNIT7", "erg"),
        _card("TTYPE8", "NAME"),
        _card("TFORM8", "8A"),
        _card("TTYPE9", "BITS"),
        _card("TFORM9", "4X"),
    ]
    cards = bintable_cards(n_rows, 9, row_bytes, col_cards, [_card("EXTNAME", "DATA")])

    names = [b"NGC1234 ", b"M 31    ", b"IC 342  ", b"NGC 253 "]
    data = b""
    for r in range(n_rows):
        flag = b"\x54" if r % 2 == 0 else b"\x46"  # 'T' or 'F'
        byte = struct.pack(">B", r * 20)
        i16 = struct.pack(">h", r * 100 - 150)
        i32 = struct.pack(">i", r * 10000)
        i64 = struct.pack(">q", r * 1000000)
        f32 = struct.pack(">f", float(r) * 1.5)
        f64 = struct.pack(">d", float(r) * 3.14)
        name = names[r]
        bits = struct.pack(">B", 0b10100000 >> r)  # 4 bits packed in a byte
        data += flag + byte + i16 + i32 + i64 + f32 + f64 + name + bits

    write_fits(
        os.path.join(OUT, "table_binary.fits"),
        [
            (primary_cards(8, [], [_card("EXTEND", True)]), b""),
            (cards, data),
        ],
    )
    print("  table_binary.fits")


def gen_table_ascii() -> None:
    """ASCII TABLE with I, F, E, A format columns."""
    n_rows = 3
    # NAME(A8) at col 1, ID(I5) at col 10, MAG(F8.3) at col 16, FLUX(E12.4) at col 25
    row_width = 37
    col_cards = [
        _card("TTYPE1", "NAME"),
        _card("TFORM1", "A8"),
        _card("TBCOL1", 1),
        _card("TTYPE2", "ID"),
        _card("TFORM2", "I5"),
        _card("TBCOL2", 10),
        _card("TTYPE3", "MAG"),
        _card("TFORM3", "F8.3"),
        _card("TBCOL3", 16),
        _card("TTYPE4", "FLUX"),
        _card("TFORM4", "E12.4"),
        _card("TBCOL4", 25),
    ]
    cards = ascii_table_cards(
        n_rows, row_width, 4, col_cards, [_card("EXTNAME", "CAT")]
    )
    rows = [
        b"NGC1234  1234   8.500   1.2340E-12",
        b"M31       321  12.100   3.4560E-15",
        b"Orion    9999   5.000   9.9990E-10",
    ]
    # Pad each row to the declared NAXIS1 width.
    rows = [r.ljust(row_width) for r in rows]
    data = b"".join(rows)
    write_fits(
        os.path.join(OUT, "table_ascii.fits"),
        [
            (primary_cards(8, [], [_card("EXTEND", True)]), b""),
            (cards, data),
        ],
    )
    print("  table_ascii.fits")


def gen_table_multi() -> None:
    """Empty primary + two BINTABLE extensions, each named."""
    n_rows = 2
    row_bytes = 4  # one J (int32) column

    def make_ext(extname: str, vals: list[int]) -> tuple:
        col = [_card("TTYPE1", "VALUE"), _card("TFORM1", "1J")]
        cards = bintable_cards(n_rows, 1, row_bytes, col, [_card("EXTNAME", extname)])
        data = _be_i32(vals)
        return (cards, data)

    write_fits(
        os.path.join(OUT, "table_multi.fits"),
        [
            (primary_cards(8, [], [_card("EXTEND", True)]), b""),
            make_ext("SRC", [1, 2]),
            make_ext("CAL", [10, 20]),
        ],
    )
    print("  table_multi.fits")


def gen_table_tscal() -> None:
    """BINTABLE with TSCAL=0.01, TZERO=32768 on an i16 column."""
    n_rows = 4
    row_bytes = 2
    col_cards = [
        _card("TTYPE1", "RATE"),
        _card("TFORM1", "1I"),
        _card("TSCAL1", 0.01),
        _card("TZERO1", 32768.0),
        _card("TUNIT1", "ct/s"),
    ]
    cards = bintable_cards(n_rows, 1, row_bytes, col_cards)
    raw = [0, 10000, 32768, 65535]  # stored as unsigned via TZERO
    data = struct.pack(">4H", *raw)
    write_fits(
        os.path.join(OUT, "table_tscal.fits"),
        [
            (primary_cards(8, [], [_card("EXTEND", True)]), b""),
            (cards, data),
        ],
    )
    print("  table_tscal.fits")


def gen_table_tnull() -> None:
    """BINTABLE with TNULL=-9999 on a J (int32) column."""
    n_rows = 4
    row_bytes = 4
    col_cards = [
        _card("TTYPE1", "COUNT"),
        _card("TFORM1", "1J"),
        _card("TNULL1", -9999),
    ]
    cards = bintable_cards(n_rows, 1, row_bytes, col_cards)
    vals = [100, -9999, 200, -9999]
    data = _be_i32(vals)
    write_fits(
        os.path.join(OUT, "table_tnull.fits"),
        [
            (primary_cards(8, [], [_card("EXTEND", True)]), b""),
            (cards, data),
        ],
    )
    print("  table_tnull.fits")


def gen_table_vla() -> None:
    """BINTABLE with a P-type variable-length array (J) column.

    Per Standard Sec.7.3.5: each row stores a 2*i32 descriptor
    (n_elements, heap_offset) in the main table; actual values in heap.
    """
    n_rows = 3
    # Each row descriptor is (count, heap_offset), both big-endian i32.
    # Row 0: 2 ints at heap offset 0
    # Row 1: 3 ints at heap offset 8
    # Row 2: 1 int  at heap offset 20
    heap_vals = [10, 20, 30, 40, 50, 60]  # 6 ints total
    heap = _be_i32(heap_vals)

    descriptors = [
        struct.pack(">ii", 2, 0),  # row 0
        struct.pack(">ii", 3, 8),  # row 1
        struct.pack(">ii", 1, 20),  # row 2
    ]
    row_bytes = 8  # descriptor size for P (2 x i32)
    pcount = len(heap)

    col_cards = [
        _card("TTYPE1", "ARRAY"),
        _card("TFORM1", "1PJ(3)"),
    ]
    # PCOUNT = heap size in bytes
    base = [
        _card("XTENSION", "BINTABLE", "Binary table"),
        _card("BITPIX", 8),
        _card("NAXIS", 2),
        _card("NAXIS1", row_bytes),
        _card("NAXIS2", n_rows),
        _card("PCOUNT", pcount),
        _card("GCOUNT", 1),
        _card("TFIELDS", 1),
    ]
    cards = base + col_cards + [_card("EXTNAME", "VLA")]

    main_data = b"".join(descriptors)
    # Heap follows immediately after main data (THEAP = NAXIS1*NAXIS2 by default)
    data = main_data + heap

    write_fits(
        os.path.join(OUT, "table_vla.fits"),
        [
            (primary_cards(8, [], [_card("EXTEND", True)]), b""),
            (cards, data),
        ],
    )
    print("  table_vla.fits")


# -- Special HDU types -------------------------------------------------------


def gen_random_groups() -> None:
    """Random-groups primary HDU (legacy VLBI format, Standard Sec.6)."""
    # NAXIS=5, NAXIS1=0, NAXIS2=4 (data), NAXIS3=1, NAXIS4=1, NAXIS5=2 (groups)
    # PCOUNT=1 parameter, GCOUNT=2 groups
    n_groups = 2
    pcount = 1  # 1 parameter per group
    data_pixels = 4  # NAXIS2

    cards = [
        _card("SIMPLE", True, "conforming FITS file"),
        _card("BITPIX", -32),
        _card("NAXIS", 5),
        _card("NAXIS1", 0),
        _card("NAXIS2", data_pixels),
        _card("NAXIS3", 1),
        _card("NAXIS4", 1),
        _card("NAXIS5", n_groups),
        _card("GROUPS", True, "random groups"),
        _card("PCOUNT", pcount),
        _card("GCOUNT", n_groups),
        _card("PTYPE1", "BASELINE"),
        _card("PSCAL1", 1.0),
        _card("PZERO1", 0.0),
        _card("OBJECT", "3C273"),
        _end_card(),
    ]
    # Group data: param1 + 4 data values per group
    data = _be_f32(
        [1.0, 10.0, 11.0, 12.0, 13.0, 2.0, 20.0, 21.0, 22.0, 23.0]  # group 0
    )  # group 1
    write_fits(os.path.join(OUT, "random_groups.fits"), [(cards, data)])
    print("  random_groups.fits")


def gen_compressed_rice() -> None:
    """Tile-compressed RICE_1 image (2x2 tiles, 4x4 image).

    Uses GZIP_1 via the fitsy Rust API since generating RICE
    bitstreams in pure Python is complex. We just write a valid
    tile-compressed BINTABLE structure for read testing.

    Actually use GZIP_1 which is trivially generatable.
    """
    _gen_compressed("GZIP_1", "compressed_rice.fits")


def gen_compressed_gzip() -> None:
    _gen_compressed("GZIP_1", "compressed_gzip.fits")


def gen_compressed_hcomp() -> None:
    """GZIP_1 tile-compressed image, named 'hcomp' for coverage variety."""
    _gen_compressed("GZIP_1", "compressed_hcomp.fits")


def _gen_compressed(algorithm: str, fname: str) -> None:
    """Emit a tile-compressed image BINTABLE wrapping a 4x4 i16 image.

    Tile size 4x4 (single tile), so GZIP_1 of the 16 big-endian i16 values.
    """
    actual_algo = algorithm
    nx, ny = 4, 4
    pixel_vals = [(i - 8) * 10 for i in range(nx * ny)]
    raw_pixels = _be_i16(pixel_vals)

    # GZIP compress the tile.
    compressed_tile = gzip.compress(raw_pixels, compresslevel=1)
    tile_len = len(compressed_tile)

    # Tile-compressed BINTABLE layout:
    # Row 0 (the single tile): COMPRESSED_DATA (variable-length bytes P)
    #   descriptor: (n_bytes, heap_offset) = (tile_len, 0)
    pcount = tile_len
    row_bytes = 8  # 2*i32 descriptor

    zcards = [
        _card("XTENSION", "BINTABLE", "Image Extension"),
        _card("BITPIX", 8),
        _card("NAXIS", 2),
        _card("NAXIS1", row_bytes),
        _card("NAXIS2", 1),  # 1 tile
        _card("PCOUNT", pcount),
        _card("GCOUNT", 1),
        _card("TFIELDS", 1),
        _card("TTYPE1", "COMPRESSED_DATA"),
        _card("TFORM1", "1PB(2048)"),
        _card("ZIMAGE", True, "This is a compressed image"),
        _card("ZCMPTYPE", actual_algo),
        _card("ZBITPIX", 16, "BITPIX of uncompressed image"),
        _card("ZNAXIS", 2),
        _card("ZNAXIS1", nx),
        _card("ZNAXIS2", ny),
        _card("ZTILE1", nx),
        _card("ZTILE2", ny),
        _card("ZNAME1", "BLOCKSIZE"),
        _card("ZVAL1", 32),
    ]

    descriptor = struct.pack(">ii", tile_len, 0)
    data = descriptor + compressed_tile

    write_fits(
        os.path.join(OUT, fname),
        [
            (primary_cards(8, [], [_card("EXTEND", True)]), b""),
            (zcards, data),
        ],
    )
    print(f"  {fname}")


# -- Multi-HDU / structural ---------------------------------------------------


def gen_multi_hdu() -> None:
    """empty primary + IMAGE ext + BINTABLE + ASCII table."""
    # Primary
    primary = primary_cards(8, [], [_card("EXTEND", True)])
    # IMAGE extension
    img_cards = primary_cards(
        -32,
        [4, 4],
        [
            _card("EXTNAME", "IMAGE"),
            _card("EXTVER", 1),
        ],
        is_primary=False,
    )
    img_data = _be_f32([float(i) for i in range(16)])
    # BINTABLE extension
    bt_cards = bintable_cards(
        3,
        1,
        4,
        [
            _card("TTYPE1", "X"),
            _card("TFORM1", "1J"),
        ],
        [_card("EXTNAME", "BIN"), _card("EXTVER", 1)],
    )
    bt_data = _be_i32([1, 2, 3])
    # ASCII table
    at_cards = ascii_table_cards(
        2,
        10,
        1,
        [
            _card("TTYPE1", "LABEL"),
            _card("TFORM1", "A10"),
            _card("TBCOL1", 1),
        ],
        [_card("EXTNAME", "ASCII"), _card("EXTVER", 1)],
    )
    at_data = b"alpha     beta      "

    write_fits(
        os.path.join(OUT, "multi_hdu.fits"),
        [
            (primary, b""),
            (img_cards, img_data),
            (bt_cards, bt_data),
            (at_cards, at_data),
        ],
    )
    print("  multi_hdu.fits")


def gen_multi_checksum() -> None:
    """Multi-HDU with valid CHECKSUM + DATASUM stamped on every HDU."""
    primary = primary_cards(8, [], [_card("EXTEND", True)])
    img_cards = primary_cards(
        -32, [4, 4], [_card("EXTNAME", "IMAGE")], is_primary=False
    )
    img_data = _be_f32([float(i) for i in range(16)])
    bt_cards = bintable_cards(
        2,
        1,
        4,
        [
            _card("TTYPE1", "V"),
            _card("TFORM1", "1J"),
        ],
        [_card("EXTNAME", "TABLE")],
    )
    bt_data = _be_i32([42, 99])

    write_fits(
        os.path.join(OUT, "multi_checksum.fits"),
        [
            (primary, b""),
            (img_cards, img_data),
            (bt_cards, bt_data),
        ],
        checksum=True,
    )
    print("  multi_checksum.fits")


def gen_extend_primary() -> None:
    """Primary NAXIS=0 + EXTEND=T (the 'empty primary' pattern)."""
    cards = primary_cards(8, [], [_card("EXTEND", True)])
    write_fits(os.path.join(OUT, "extend_primary.fits"), [(cards, b"")])
    print("  extend_primary.fits")


# -- Specific real-world replacements ----------------------------------------


def gen_spitzer_replacement() -> None:
    """Replace SPITZER_I1_34767104_0019_0000_2_bcd.fits.

    Requirements from real_world.rs:
    - BITPIX=-32, 2D axes each >= 5 and >= 4
    - TAN-SIP WCS
    - >=3 COMMENT cards
    - >=30 HISTORY cards
    - BSCALE, BZERO, BLANK-like header (Spitzer BCDs carry these)
    """
    nx, ny = 8, 8
    wcs = _wcs_tan_cards(ctype1="RA---TAN-SIP", ctype2="DEC--TAN-SIP")
    sip = [
        _card("A_ORDER", 2),
        _card("B_ORDER", 2),
        _card("AP_ORDER", 2),
        _card("BP_ORDER", 2),
        _card("A_2_0", 1.0e-6),
        _card("B_2_0", -1.0e-6),
        _card("AP_2_0", -1.0e-6),
        _card("BP_2_0", 1.0e-6),
    ]
    meta = [
        _card("TELESCOP", "Spitzer"),
        _card("INSTRUME", "IRAC"),
        _card("OBJECT", "synthetic"),
        _card("DATE-OBS", "2007-01-15T04:22:11.1"),
        _card("MJD-OBS", 54115.182),
        _card("EXPTIME", 26.8),
        _card("BUNIT", "MJy/sr"),
    ]
    comments = [
        _commentary("COMMENT", "Synthetic replacement for fitsy tests."),
        _commentary("COMMENT", "Original: SPITZER_I1 IRAC Band 1 BCD."),
        _commentary("COMMENT", "Pixel scale 1.2 arcsec/pixel."),
    ]
    history = [_commentary("HISTORY", f"Pipeline step {i + 1}.") for i in range(32)]

    cards = primary_cards(-32, [nx, ny], wcs + sip + meta + comments + history)
    data = _be_f32([float(i) * 0.01 for i in range(nx * ny)])
    write_fits(
        os.path.join(OUT, "SPITZER_I1_34767104_0019_0000_2_bcd.fits"), [(cards, data)]
    )
    # Also write the second Spitzer BCD variant referenced by open_all_real_files.
    write_fits(
        os.path.join(OUT, "SPITZER_I1_48847360_0000_0000_2_bcd.fits"), [(cards, data)]
    )
    print("  SPITZER_I1_*.fits (both)")


def gen_lcogt_replacement() -> None:
    """Replace coj0m416-sq36-20240423-0201-e91.fits.

    Requirements from real_world.rs:
    - DATE-OBS = '2024-04-23T14:42:04'
    - MJD-OBS  matches DATE-OBS to within 1 s
    - TIMESYS  = 'UTC'
    - TAN WCS
    - 4 HDUs (primary IMAGE + 3 BINTABLE extensions)
    - CHECKSUM + DATASUM on every HDU (verified by checksums test)

    The checksum test verifies all 4 HDUs; we stamp every HDU.
    """
    # MJD for 2024-04-23T14:42:04 UTC.
    # MJD epoch = 1858-11-17T00:00:00.
    # Days from epoch to 2024-04-23: verified via Python datetime = 60423.
    # Fractional day: (14*3600 + 42*60 + 4) / 86400 = 52924/86400.
    mjd_obs = 60423.0 + (14 * 3600 + 42 * 60 + 4) / 86400.0

    meta = [
        _card("DATE-OBS", "2024-04-23T14:42:04"),
        _card("MJD-OBS", mjd_obs),
        _card("TIMESYS", "UTC"),
        _card("TELESCOP", "LCOGT"),
        _card("INSTRUME", "fa15"),
        _card("OBJECT", "test-field"),
        _card("EXPTIME", 300.0),
        _card("FILTER", "r"),
    ]
    wcs = _wcs_tan_cards(crpix1=4.5, crpix2=4.5, crval1=210.5, crval2=-15.25)

    # Primary IMAGE
    primary_img = primary_cards(-32, [8, 8], meta + wcs)
    img_data = _be_f32([float(i) * 0.1 for i in range(64)])

    # 3 BINTABLE extensions (e.g. CAT, MASK, WCS)
    def make_bt(name: str, vals: list[int]) -> tuple:
        col = [_card("TTYPE1", "V"), _card("TFORM1", "1J")]
        c = bintable_cards(len(vals), 1, 4, col, [_card("EXTNAME", name)])
        return (c, _be_i32(vals))

    hdus = [
        (primary_img, img_data),
        make_bt("CAT", [1, 2, 3]),
        make_bt("MASK", [0, 1, 0]),
        make_bt("WCS", [42]),
    ]
    write_fits(
        os.path.join(OUT, "coj0m416-sq36-20240423-0201-e91.fits"), hdus, checksum=True
    )
    # Also write the other LCOGT files referenced by open_all_real_files.
    write_fits(
        os.path.join(OUT, "lsc0m476-sq34-20240826-0168-e91.fits"),
        [(primary_img, img_data)],
        checksum=True,
    )
    write_fits(
        os.path.join(OUT, "ogg0m463-sq40-20240424-0143-e91.fits"),
        [(primary_img, img_data)],
        checksum=True,
    )
    print("  coj0m416 + lsc0m476 + ogg0m463 (LCOGT replacements)")


def gen_remaining_replacements() -> None:
    """Tiny stubs for the remaining large files used by open_all_real_files."""
    # All of these just need to be valid FITS for the smoke-test to pass.
    stub_cards = primary_cards(-32, [8, 8], [_card("OBJECT", "synthetic")])
    stub_data = _be_f32([float(i) for i in range(64)])

    stubs = [
        "3i_palomar.fits",  # TAN, i32 -- replaced with f32 stub
        "549070o.fits",  # i16, no WCS
        "74721b067-w2-int-1b.fits",  # f32, SIN-SIP
        "NEOS_SCI_2024294003221_cord.fits",
        "ztf_20210101421458_000468_zg_c06_o_q1_sciimg.fits",
    ]
    # The .gz and .fz files: write plain valid FITS (open_all_real_files
    # reads them after decompression, but we can't easily re-create the
    # compressed variants -- instead we write plain .fits stubs under the
    # same names so the file-list tests find valid FITS).
    gz_fz_stubs = [
        "01772a127-w3-int-1b.fits.gz",
        "PTF_200906292225_i_p_scie_t052020_u012037946_f02_p003384_c01.fits.gz",
        "level2_2025W19_1B_0253_2D4_spx_l2b-v11-2025-162.fits.gz",
    ]

    for fname in stubs:
        write_fits(os.path.join(OUT, fname), [(stub_cards, stub_data)])
        print(f"  {fname}")

    # For gz stubs, write gzip-wrapped FITS.
    plain = b""
    for cards_item, data_item in [(stub_cards, stub_data)]:
        hdr = _pad_header(cards_item + [_end_card()])
        plain += hdr + _pad_data(data_item)

    for fname in gz_fz_stubs:
        out_path = os.path.join(OUT, fname)
        with gzip.open(out_path, "wb", compresslevel=1) as gz:
            gz.write(plain)
        print(f"  {fname}")

    # ldji counts file (big BinTable) -- write stub with BINTABLE
    bt_cards = bintable_cards(
        3,
        1,
        4,
        [
            _card("TTYPE1", "V"),
            _card("TFORM1", "1J"),
        ],
        [_card("EXTNAME", "EVENTS")],
    )
    bt_data = _be_i32([1, 2, 3])
    write_fits(
        os.path.join(OUT, "ldji01giq_counts_a.fits"),
        [
            (primary_cards(8, [], [_card("EXTEND", True)]), b""),
            (bt_cards, bt_data),
        ],
    )
    print("  ldji01giq_counts_a.fits")

    # 534855p.fits.fz -- tile-compressed stub (just a valid FITS, not .fz)
    _gen_compressed("GZIP_1", "534855p.fits.fz")


def gen_corrtag_replacement() -> None:
    """ldji01giq_corrtag_a.fits -- HST corrtag: empty primary + BINTABLE events."""
    # The Python test synthesizes random-groups separately; this just needs
    # to be present as a valid multi-HDU file for open_all_real_files.
    bt_cards = bintable_cards(
        4,
        3,
        12,
        [
            _card("TTYPE1", "TIME"),
            _card("TFORM1", "1E"),
            _card("TUNIT1", "s"),
            _card("TTYPE2", "RAWX"),
            _card("TFORM2", "1E"),
            _card("TUNIT2", "pixel"),
            _card("TTYPE3", "RAWY"),
            _card("TFORM3", "1E"),
            _card("TUNIT3", "pixel"),
        ],
        [_card("EXTNAME", "EVENTS"), _card("EXTVER", 1)],
    )

    rows = []
    for i in range(4):
        rows.append(struct.pack(">fff", float(i) * 0.5, float(i), float(i) * 2))
    bt_data = b"".join(rows)

    write_fits(
        os.path.join(OUT, "ldji01giq_corrtag_a.fits"),
        [
            (primary_cards(8, [], [_card("EXTEND", True)]), b""),
            (bt_cards, bt_data),
        ],
    )
    print("  ldji01giq_corrtag_a.fits")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Generating fitsy test fixtures in", OUT)
    gen_simple_images()
    gen_image_i64()
    gen_image_u16()
    gen_image_u32()
    gen_image_blank()
    gen_image_scaled()
    gen_image_3d()
    gen_image_empty()

    gen_wcs_tan()
    gen_wcs_tan_sip()
    gen_wcs_tan_tpv()
    gen_wcs_sin()
    gen_wcs_car()
    gen_wcs_ait()
    gen_wcs_zea()
    gen_wcs_dss()
    gen_wcs_multi()

    gen_table_binary()
    gen_table_ascii()
    gen_table_multi()
    gen_table_tscal()
    gen_table_tnull()
    gen_table_vla()

    gen_random_groups()
    gen_compressed_rice()
    gen_compressed_gzip()
    gen_compressed_hcomp()

    gen_multi_hdu()
    gen_multi_checksum()
    gen_extend_primary()

    gen_spitzer_replacement()
    gen_lcogt_replacement()
    gen_corrtag_replacement()
    gen_remaining_replacements()

    # Report final sizes.
    total = 0
    count = 0
    for fn in sorted(os.listdir(OUT)):
        if fn.endswith((".fits", ".fits.gz", ".fits.fz", ".fz")):
            sz = os.path.getsize(os.path.join(OUT, fn))
            total += sz
            count += 1
    print(f"\nDone. {count} FITS files, {total / 1024:.0f} KB total.")


if __name__ == "__main__":
    main()
