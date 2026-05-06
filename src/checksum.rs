//! `CHECKSUM` / `DATASUM` keywords (Pence & Seaman 1995, FITS Standard
//! Sec.4.4.2.7).
//!
//! Both keywords carry a 32-bit unsigned 1's-complement sum of the
//! HDU's bytes, but they differ in what is summed and how the result
//! is encoded:
//!
//! * `DATASUM` -- decimal-string ASCII representation of the
//!   1's-complement sum of the *data* unit only (zero-padded out to
//!   the next 2880-byte block boundary).
//! * `CHECKSUM` -- ASCII-armoured (16-character) encoding of the
//!   1's complement of the running sum over the *header + data* units,
//!   computed with the `CHECKSUM` card itself present in the header
//!   (its value field full of ASCII zeroes during construction). For a
//!   valid file the recomputed sum equals `0xFFFFFFFF`.

/// Compute the FITS 32-bit 1's-complement sum of `bytes`. The buffer
/// must be a multiple of 4 bytes long; the FITS standard guarantees
/// this for header + zero-padded data units.
#[must_use]
pub fn checksum_bytes(bytes: &[u8]) -> u32 {
    debug_assert_eq!(
        bytes.len() % 4,
        0,
        "FITS checksum requires 4-byte aligned input"
    );
    // Per Pence & Seaman 1995 Sec.3: sum the data as 32-bit big-endian
    // words with end-around carry. We accumulate the high and low
    // 16 bits of each word in independent **64-bit** registers so
    // that no inner-loop addition can overflow (cfitsio relies on
    // `unsigned long` being 64-bit on Unix; on a 32-bit accumulator
    // any HDU larger than ~256 KiB would silently wrap and produce
    // a wrong sum).
    let mut hi: u64 = 0;
    let mut lo: u64 = 0;
    for w in bytes.chunks_exact(4) {
        hi += u64::from(u16::from_be_bytes([w[0], w[1]]));
        lo += u64::from(u16::from_be_bytes([w[2], w[3]]));
    }
    // Fold every carry above bit 16 of each accumulator back in via
    // 1's-complement end-around-carry. Iterate until both
    // accumulators fit in 16 bits.
    while (hi >> 16) != 0 || (lo >> 16) != 0 {
        let hicarry = hi >> 16;
        let locarry = lo >> 16;
        hi = (hi & 0xFFFF) + locarry;
        lo = (lo & 0xFFFF) + hicarry;
    }
    ((hi << 16) | (lo & 0xFFFF)) as u32
}

/// Combine two FITS 1's-complement sums (`a` and `b`) the same
/// way [`checksum_is_valid`] combines the header and data sums:
/// add as `u64`, then fold the carry-out back into the low 32
/// bits. The result is `checksum_bytes(buf_a ++ buf_b)` for any
/// concatenation of suitably-aligned `buf_a` and `buf_b`.
///
/// Streaming callers (e.g. on-disk checksum verification) can
/// build up the sum of an arbitrarily long byte sequence by
/// initialising an accumulator to `0` and folding successive
/// per-chunk [`checksum_bytes`] results into it.
#[must_use]
pub fn checksum_combine(a: u32, b: u32) -> u32 {
    let sum = u64::from(a) + u64::from(b);
    ((sum & 0xFFFF_FFFF) + (sum >> 32)) as u32
}

/// Verify that `header_bytes ++ data_bytes_padded` recompute to
/// `0xFFFFFFFF`, the all-ones sentinel required for a valid CHECKSUM.
///
/// `header_bytes` must include the `CHECKSUM` card with its current
/// value; `data_bytes_padded` must be the data section already padded
/// to a 2880-byte boundary.
#[must_use]
pub fn checksum_is_valid(header_bytes: &[u8], data_bytes_padded: &[u8]) -> bool {
    let h = checksum_bytes(header_bytes);
    let d = checksum_bytes(data_bytes_padded);
    // 1's complement add of the two 32-bit sums.
    let sum = u64::from(h) + u64::from(d);
    let folded = ((sum & 0xFFFF_FFFF) + (sum >> 32)) as u32;
    folded == 0xFFFF_FFFF
}

/// Verify that the integer-decimal `DATASUM` value matches the
/// 1's-complement sum of `data_bytes_padded`.
#[must_use]
pub fn datasum_matches(stored_decimal: &str, data_bytes_padded: &[u8]) -> bool {
    let want: u32 = match stored_decimal.trim().trim_matches('\'').trim().parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    checksum_bytes(data_bytes_padded) == want
}

/// `DATASUM` value: ASCII-decimal encoding of the 1's-complement sum
/// of the (already block-padded) data unit.
#[must_use]
pub fn datasum_string(data_bytes_padded: &[u8]) -> String {
    checksum_bytes(data_bytes_padded).to_string()
}

/// ASCII-armoured encoding of a 32-bit checksum value
/// (Pence & Seaman 1995, FITS Standard Sec.4.4.2.7 / Appendix K). The
/// returned 16 bytes are restricted to printable ASCII (`0x20..=0x7E`)
/// excluding the FITS-reserved characters in the range
/// `0x3a..=0x40` (`:;<=>?@`) and `0x5b..=0x60` (`` [\\]^_` ``). When the
/// 16 bytes are placed in the value field of a `CHECKSUM` card
/// (columns 11..26 of an 80-byte card, i.e. offset 11 in the card --
/// which is byte offset 3 modulo 4 inside the FITS checksum stream),
/// their contribution to the running sum is exactly `value`.
#[must_use]
pub fn encode_checksum(value: u32) -> [u8; 16] {
    const OFFSET: i32 = 0x30;
    const EXCLUDE: [i32; 13] = [
        0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x60,
    ];
    const MASKS: [u32; 4] = [0xff00_0000, 0x00ff_0000, 0x0000_ff00, 0x0000_00ff];

    let mut asc = [0_u8; 16];
    for i in 0..4 {
        let byte = ((value & MASKS[i]) >> (24 - 8 * i)) as i32;
        let quotient = byte / 4 + OFFSET;
        let remainder = byte % 4;
        let mut ch: [i32; 4] = [quotient + remainder, quotient, quotient, quotient];

        // Push offending characters out of the EXCLUDE set, preserving
        // the per-byte sum: ch[j]++ and ch[j+1]-- (or the other pair).
        let mut check = true;
        while check {
            check = false;
            for &ex in &EXCLUDE {
                let mut j = 0;
                while j + 1 < 4 {
                    if ch[j] == ex || ch[j + 1] == ex {
                        ch[j] += 1;
                        ch[j + 1] -= 1;
                        check = true;
                    }
                    j += 2;
                }
            }
        }
        for (j, c) in ch.iter().enumerate() {
            asc[4 * j + i] = *c as u8;
        }
    }

    // Cyclic left-rotate by one byte: out[i] = asc[(i + 15) % 16].
    let mut out = [0_u8; 16];
    for i in 0..16 {
        out[i] = asc[(i + 15) % 16];
    }
    out
}

/// Compute and stamp `CHECKSUM` and `DATASUM` cards into a serialized
/// HDU.
///
/// `header_bytes` and `data_padded` together form the on-disk HDU
/// (the header is multiple of 2880 bytes ending with `END`+spaces; the
/// data is the data section already padded to a 2880 boundary).
///
/// The header **must** already contain placeholder cards
/// `CHECKSUM = '0000000000000000'` and `DATASUM = '0         '` (any
/// 10-char value placeholder is fine -- the writer rewrites the value
/// field in place). `stamp_checksum` rewrites those two value fields
/// so that:
///
/// * `DATASUM` decodes to `checksum_bytes(data_padded)`.
/// * `CHECKSUM` decodes to a value whose 1's-complement sum, combined
///   with the data sum, is `0xFFFFFFFF` (i.e. [`checksum_is_valid`]
///   returns `true`).
///
/// Returns an error if the placeholder cards are absent or if the
/// header is not block-aligned.
pub fn stamp_checksum(header_bytes: &mut [u8], data_padded: &[u8]) -> Result<(), &'static str> {
    use crate::header::card::CARD_SIZE;
    use crate::io::block::BLOCK_SIZE;

    if !header_bytes.len().is_multiple_of(BLOCK_SIZE) {
        return Err("header bytes are not a multiple of 2880");
    }

    // Locate the CHECKSUM and DATASUM cards by scanning every 80-byte
    // card. We rewrite the value field in place without touching the
    // surrounding bytes (so the comment, if any, survives).
    let mut checksum_off: Option<usize> = None;
    let mut datasum_off: Option<usize> = None;
    let mut card = 0;
    while (card + 1) * CARD_SIZE <= header_bytes.len() {
        let off = card * CARD_SIZE;
        let kw = &header_bytes[off..off + 8];
        if kw == b"CHECKSUM" && header_bytes[off + 8] == b'=' {
            checksum_off = Some(off);
        } else if kw == b"DATASUM " && header_bytes[off + 8] == b'=' {
            datasum_off = Some(off);
        } else if kw == b"END     " {
            break;
        }
        card += 1;
    }
    let checksum_off = checksum_off.ok_or("CHECKSUM placeholder not found in header")?;
    let datasum_off = datasum_off.ok_or("DATASUM placeholder not found in header")?;

    // 1. Stamp DATASUM first -- its value affects the header bytes.
    let data_sum = checksum_bytes(data_padded);
    let datasum_str = data_sum.to_string();
    write_quoted_value(
        &mut header_bytes[datasum_off..datasum_off + CARD_SIZE],
        &datasum_str,
    )?;

    // 2. Initialize CHECKSUM value field with 16 ASCII '0' chars
    //    (`encode_checksum(0)`), compute the running sum over the
    //    full header+data, and replace the field with the encoding
    //    of the 1's-complement of that sum. This works because the
    //    encoder is designed so that its 16-byte output, placed at
    //    column 11 of an 80-byte card (offset 11 mod 4 = 3 in the
    //    FITS sum stream), contributes exactly its argument value to
    //    the running sum, and `encode_checksum(0)` is exactly the
    //    16-zero placeholder.
    let zeros = encode_checksum(0);
    write_quoted_value_raw(
        &mut header_bytes[checksum_off..checksum_off + CARD_SIZE],
        &zeros,
    )?;
    let header_sum = checksum_bytes(header_bytes);
    let combined = ones_complement_add(header_sum, data_sum);
    let target = !combined;
    let encoded = encode_checksum(target);
    write_quoted_value_raw(
        &mut header_bytes[checksum_off..checksum_off + CARD_SIZE],
        &encoded,
    )?;
    Ok(())
}

fn ones_complement_add(a: u32, b: u32) -> u32 {
    let s = u64::from(a) + u64::from(b);
    ((s & 0xFFFF_FFFF) + (s >> 32)) as u32
}

/// Write `'value'` into a card's value field (columns 11..). The value
/// is space-padded out to at least 8 chars (FITS minimum string length)
/// and may not exceed `CARD_SIZE - 12 = 68` chars between the quotes.
fn write_quoted_value(card: &mut [u8], value: &str) -> Result<(), &'static str> {
    let bytes = value.as_bytes();
    if bytes.len() > 68 {
        return Err("checksum/datasum value too long for one card");
    }
    // Validate printable ASCII.
    for &b in bytes {
        if !(0x20..=0x7E).contains(&b) {
            return Err("non-printable byte in checksum/datasum value");
        }
    }
    write_quoted_value_raw(card, bytes)
}

fn write_quoted_value_raw(card: &mut [u8], bytes: &[u8]) -> Result<(), &'static str> {
    use crate::header::card::CARD_SIZE;
    if card.len() != CARD_SIZE {
        return Err("card slice not 80 bytes");
    }
    // Wipe the existing value field (columns 11..80) with spaces, then
    // emit `'<bytes>'` starting at column 11. The original comment is
    // therefore overwritten; we don't try to preserve it because the
    // value width can change.
    for b in &mut card[10..] {
        *b = b' ';
    }
    if 11 + bytes.len() + 1 > CARD_SIZE {
        return Err("encoded checksum value would overflow the card");
    }
    card[10] = b'\'';
    card[11..11 + bytes.len()].copy_from_slice(bytes);
    card[11 + bytes.len()] = b'\'';
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer_sums_to_zero() {
        assert_eq!(checksum_bytes(&[]), 0);
    }

    #[test]
    fn single_word_sum() {
        // 0x12345678 -> high 0x1234, low 0x5678, combined 0x12345678.
        let buf = [0x12, 0x34, 0x56, 0x78];
        assert_eq!(checksum_bytes(&buf), 0x12345678);
    }

    #[test]
    fn carry_is_added_back_around() {
        // In 1's-complement arithmetic with end-around carry,
        // 0xFFFFFFFF + 0x00000001 folds back to 0x00000001 (the
        // overflow bit re-enters at the LSB). 0xFFFFFFFF is the
        // "negative zero" representation; adding 1 yields +1.
        let buf = [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01];
        assert_eq!(checksum_bytes(&buf), 0x0000_0001);
    }

    #[test]
    fn no_overflow_on_large_buffer() {
        // 1 MiB of 0xFF bytes: each 4-byte word is 0xFFFFFFFF,
        // there are 262144 words. Naive u32 accumulation would
        // overflow long before the end-around fold; this regression
        // test pins the correct (cfitsio-equivalent) result.
        let buf = vec![0xFF_u8; 1 << 20];
        // Sum of N copies of 0xFFFFFFFF in 1's-complement = 0xFFFFFFFF
        // for any N >= 1 (since -0 + -0 + ... = -0).
        assert_eq!(checksum_bytes(&buf), 0xFFFF_FFFF);
    }

    #[test]
    fn datasum_matches_decimal_string() {
        let buf = [0x00, 0x00, 0x00, 0x42];
        assert!(datasum_matches("66", &buf));
        assert!(datasum_matches(" '66      '", &buf));
        assert!(!datasum_matches("99", &buf));
    }

    #[test]
    fn encode_checksum_is_printable_and_safe() {
        for v in [0_u32, 1, 0x12345678, 0xDEAD_BEEF, 0xFFFF_FFFF, 0xC0C0_C0C0] {
            let enc = encode_checksum(v);
            for &b in &enc {
                assert!((0x20..=0x7E).contains(&b), "non-printable: 0x{b:02x}");
                assert!(
                    !matches!(b, 0x27 | 0x22 | 0x3a..=0x40 | 0x5b..=0x60),
                    "forbidden char: 0x{b:02x}"
                );
            }
        }
        // Defining property: V=0 encodes to all ASCII '0'.
        assert_eq!(encode_checksum(0), [b'0'; 16]);
    }

    #[test]
    fn stamp_checksum_makes_header_valid() {
        // Build a tiny valid HDU: SIMPLE/BITPIX/NAXIS=0 plus
        // CHECKSUM and DATASUM placeholder cards + END. No data.
        let cards = [
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "CHECKSUM= '0000000000000000'",
            "DATASUM = '0         '",
            "END",
        ];
        let mut buf = vec![b' '; 2880];
        for (i, c) in cards.iter().enumerate() {
            let off = i * 80;
            let cb = c.as_bytes();
            buf[off..off + cb.len()].copy_from_slice(cb);
        }
        let data: &[u8] = &[];
        stamp_checksum(&mut buf, data).unwrap();
        assert!(
            checksum_is_valid(&buf, data),
            "verifier rejected freshly stamped header"
        );
        // DATASUM for an empty data unit is 0.
        let line = std::str::from_utf8(&buf[4 * 80..5 * 80]).unwrap();
        assert!(line.starts_with("DATASUM = '0"), "got: `{line}`");
    }

    #[test]
    fn stamp_checksum_with_data() {
        let cards = [
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    1",
            "NAXIS1  =                  100",
            "CHECKSUM= '0000000000000000'",
            "DATASUM = '0         '",
            "END",
        ];
        let mut header = vec![b' '; 2880];
        for (i, c) in cards.iter().enumerate() {
            let off = i * 80;
            let cb = c.as_bytes();
            header[off..off + cb.len()].copy_from_slice(cb);
        }
        // 100 bytes of data, padded to 2880.
        let mut data = vec![0_u8; 2880];
        for (i, b) in data.iter_mut().take(100).enumerate() {
            *b = (i % 251) as u8;
        }
        stamp_checksum(&mut header, &data).unwrap();
        assert!(checksum_is_valid(&header, &data));
        assert!(datasum_matches(
            std::str::from_utf8(&header[5 * 80 + 11..5 * 80 + 11 + 12])
                .unwrap()
                .trim_end_matches(['\'', ' ']),
            &data
        ));
    }
}
