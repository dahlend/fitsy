//! `PLIO_1` (IRAF pixel-list run-length code) tile decompression.
//!
//! The format is described in Pence et al. 2010 Sec.3.4 and adopted
//! into the FITS standard 2016 Sec.10.4.
//!
//! ## On-disk format
//!
//! The payload is a stream of 16-bit big-endian half-words. The first
//! seven words form a header (older IRAF files use a 3-word header
//! distinguished by a positive magic value):
//!
//! | Word (0-indexed) | Meaning                                          |
//! |------------------|--------------------------------------------------|
//! | `w[0]`           | reserved (0)                                     |
//! | `w[1]`           | header length in words (= 7 for the modern form) |
//! | `w[2]`           | magic; `-100` (= `0xFF9C`) for the modern form   |
//! | `w[3]`           | encoded length, low 15 bits                      |
//! | `w[4]`           | encoded length, high bits                        |
//! | `w[5..7]`        | reserved (0)                                     |
//!
//! Each subsequent word is `(opcode << 12) | data` with `opcode` in
//! `0..=7`:
//!
//! | opcode | meaning                                                       |
//! |--------|---------------------------------------------------------------|
//! | 0      | run of `data` zero pixels                                     |
//! | 1      | extended set: `pv = (next_word << 12) | data`, no pixel out   |
//! | 2      | `pv += data`, no pixel out                                    |
//! | 3      | `pv -= data`, no pixel out                                    |
//! | 4      | run of `data` pixels with current value `pv`                  |
//! | 5      | run of `data` zero pixels followed by one pixel of `pv`       |
//! | 6      | `pv += data`, then write one pixel of `pv`                    |
//! | 7      | `pv -= data`, then write one pixel of `pv`                    |

use crate::error::{FitsError, Result};

/// Decode `payload` into `dst`, writing exactly `n_pixels` big-endian
/// pixels of width `bp` bytes (1, 2 or 4). `dst` is assumed to be
/// pre-zeroed by the caller; opcodes that emit zeros simply leave the
/// buffer alone.
pub(super) fn decompress_into(
    payload: &[u8],
    bp: usize,
    n_pixels: usize,
    dst: &mut [u8],
) -> Result<()> {
    if !payload.len().is_multiple_of(2) {
        return Err(FitsError::Data(format!(
            "PLIO_1 payload has odd byte length {}",
            payload.len()
        )));
    }
    let expected = n_pixels
        .checked_mul(bp)
        .ok_or_else(|| FitsError::Data("PLIO_1 destination size overflows usize".into()))?;
    if dst.len() != expected {
        return Err(FitsError::Data(format!(
            "PLIO_1 destination is {} bytes, expected {expected}",
            dst.len()
        )));
    }
    if !matches!(bp, 1 | 2 | 4) {
        return Err(FitsError::Data(format!(
            "PLIO_1 only supports 1/2/4-byte pixels, got {bp}"
        )));
    }
    if n_pixels == 0 {
        return Ok(());
    }
    if payload.len() < 6 {
        return Err(FitsError::Data(
            "PLIO_1 payload shorter than the 3-word minimum header".into(),
        ));
    }

    let words: Vec<u16> = payload
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect();

    // Header: distinguish old (positive magic = total length) from
    // new (-100 magic, length stored in two halves).
    let magic = words[2] as i16;
    let (lllen, llfirt) = if magic > 0 {
        (magic as usize, 3_usize)
    } else {
        if words.len() < 7 {
            return Err(FitsError::Data(
                "PLIO_1 new-format payload is shorter than its 7-word header".into(),
            ));
        }
        let len = ((words[4] as usize) << 15) | (words[3] as usize);
        let header_len = words[1] as usize;
        if header_len < 7 {
            return Err(FitsError::Data(format!(
                "PLIO_1 header length field = {header_len}, expected >= 7"
            )));
        }
        (len, header_len)
    };
    if lllen > words.len() {
        return Err(FitsError::Data(format!(
            "PLIO_1 declares {lllen} encoded words but only {} were provided",
            words.len()
        )));
    }
    if llfirt > lllen {
        return Err(FitsError::Data(format!(
            "PLIO_1 header length {llfirt} exceeds total length {lllen}"
        )));
    }

    let max_pixel: i64 = match bp {
        1 => i64::from(u8::MAX),
        2 => i64::from(u16::MAX),
        4 => i64::from(u32::MAX),
        _ => unreachable!(),
    };

    let xs: i64 = 0;
    let xe: i64 = (n_pixels - 1) as i64;
    let mut x1: i64 = 0;
    let mut op: usize = 0;
    let mut pv: i64 = 1;
    let mut skipwd = false;
    let mut ip = llfirt;

    while ip < lllen {
        if skipwd {
            skipwd = false;
            ip += 1;
            continue;
        }
        let word = words[ip];
        let opcode = (word >> 12) & 0xF;
        let data = i64::from(word & 0x0FFF);

        match opcode {
            // Range opcodes: zero run, pv-run, mostly-zero-with-tail.
            0 | 4 | 5 => {
                let x2 = x1 + data - 1;
                let i1 = x1.max(xs);
                let i2 = x2.min(xe);
                let np = i2 - i1 + 1;
                if np > 0 {
                    let np = np as usize;
                    let otop = op
                        .checked_add(np)
                        .and_then(|n| n.checked_sub(1))
                        .ok_or_else(|| FitsError::Data("PLIO_1 run overflows output".into()))?;
                    if otop >= n_pixels {
                        return Err(FitsError::Data(format!(
                            "PLIO_1 run extends past pixel {n_pixels}"
                        )));
                    }
                    if opcode == 4 {
                        validate_pv(pv, max_pixel)?;
                        for i in op..=otop {
                            write_pixel(dst, i, bp, pv as u64);
                        }
                    } else if opcode == 5 && i2 == x2 {
                        validate_pv(pv, max_pixel)?;
                        write_pixel(dst, otop, bp, pv as u64);
                    }
                    op = otop + 1;
                }
                x1 = x2 + 1;
            }
            // Two-word extended pv set.
            1 => {
                if ip + 1 >= lllen {
                    return Err(FitsError::Data(
                        "PLIO_1 opcode 1 (extended pv) missing second word".into(),
                    ));
                }
                // Sign-extend the next word to i64.
                let next = i64::from(words[ip + 1] as i16);
                pv = (next << 12) | data;
                skipwd = true;
            }
            2 => pv += data,
            3 => pv -= data,
            6 | 7 => {
                if opcode == 6 {
                    pv += data;
                } else {
                    pv -= data;
                }
                if x1 >= xs && x1 <= xe {
                    if op >= n_pixels {
                        return Err(FitsError::Data(
                            "PLIO_1 single-pixel write past tile end".into(),
                        ));
                    }
                    validate_pv(pv, max_pixel)?;
                    write_pixel(dst, op, bp, pv as u64);
                    op += 1;
                }
                x1 += 1;
            }
            other => {
                return Err(FitsError::Data(format!(
                    "PLIO_1 reserved opcode {other} at word {ip}"
                )));
            }
        }

        if x1 > xe {
            break;
        }
        ip += 1;
    }

    Ok(())
}

#[inline]
fn validate_pv(pv: i64, max_pixel: i64) -> Result<()> {
    if pv < 0 || pv > max_pixel {
        return Err(FitsError::Data(format!(
            "PLIO_1 decoded pixel value {pv} out of range [0, {max_pixel}]"
        )));
    }
    Ok(())
}

#[inline]
fn write_pixel(dst: &mut [u8], idx: usize, bp: usize, value: u64) {
    let off = idx * bp;
    match bp {
        1 => dst[off] = value as u8,
        2 => dst[off..off + 2].copy_from_slice(&(value as u16).to_be_bytes()),
        4 => dst[off..off + 4].copy_from_slice(&(value as u32).to_be_bytes()),
        _ => unreachable!("checked at entry"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(data_words: &[u16]) -> Vec<u8> {
        let total_len = 7 + data_words.len();
        let mut w = vec![0_u16; 7];
        w[1] = 7;
        w[2] = (-100_i16) as u16;
        w[3] = (total_len & 0x7FFF) as u16;
        w[4] = (total_len >> 15) as u16;
        w.extend_from_slice(data_words);
        let mut bytes = Vec::with_capacity(w.len() * 2);
        for word in w {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        bytes
    }

    #[test]
    fn rejects_odd_payload() {
        let mut dst = [0_u8; 4];
        assert!(decompress_into(&[0_u8; 3], 2, 2, &mut dst).is_err());
    }

    #[test]
    fn rejects_size_mismatch() {
        let mut dst = [0_u8; 6];
        assert!(decompress_into(&[0_u8; 4], 2, 2, &mut dst).is_err());
    }

    #[test]
    fn opcode4_pv_run_emits_constant_pixels() {
        // pv defaults to 1; opcode 4 with data = 3 -> 3 pixels of value 1.
        let payload = pack(&[0x4003]);
        let mut dst = vec![0_u8; 6];
        decompress_into(&payload, 2, 3, &mut dst).unwrap();
        assert_eq!(dst, vec![0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn opcode5_emits_zeros_then_pv() {
        let payload = pack(&[0x5004]);
        let mut dst = vec![0_u8; 8];
        decompress_into(&payload, 2, 4, &mut dst).unwrap();
        assert_eq!(dst, vec![0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn opcode6_pv_increment_emits_one_pixel() {
        // pv += 41 = 42, write one pixel.
        let payload = pack(&[0x6029]);
        let mut dst = vec![0_u8; 2];
        decompress_into(&payload, 2, 1, &mut dst).unwrap();
        assert_eq!(dst, vec![0, 42]);
    }

    #[test]
    fn opcode2_then_opcode4_uses_updated_pv() {
        // pv += 9 -> pv = 10, then 2 pixels of 10.
        let payload = pack(&[0x2009, 0x4002]);
        let mut dst = vec![0_u8; 4];
        decompress_into(&payload, 2, 2, &mut dst).unwrap();
        assert_eq!(dst, vec![0, 10, 0, 10]);
    }

    #[test]
    fn extended_pv_two_word() {
        // pv = (2 << 12) | 1 = 0x2001; emit one pixel.
        let payload = pack(&[0x1001, 0x0002, 0x4001]);
        let mut dst = vec![0_u8; 2];
        decompress_into(&payload, 2, 1, &mut dst).unwrap();
        assert_eq!(dst, vec![0x20, 0x01]);
    }

    #[test]
    fn rejects_pv_overflow_for_byte_pixels() {
        // pv = 1 + 256 = 257 -> overflows u8.
        let payload = pack(&[0x2100, 0x4001]);
        let mut dst = vec![0_u8; 1];
        assert!(decompress_into(&payload, 1, 1, &mut dst).is_err());
    }

    #[test]
    fn rejects_truncated_extended_pv() {
        let payload = pack(&[0x1001]);
        let mut dst = vec![0_u8; 2];
        assert!(decompress_into(&payload, 2, 1, &mut dst).is_err());
    }

    #[test]
    fn run_extending_past_tile_end_is_clipped() {
        // Per cfitsio's pl_l2pi, runs that overshoot the requested
        // window are silently clipped \u2014 only the portion within
        // [xs..=xe] is materialised.
        // Pack a single opcode claiming 100 pixels into a 1-pixel tile.
        let payload = pack(&[0x4064]);
        let mut dst = vec![0_u8; 2];
        decompress_into(&payload, 2, 1, &mut dst).unwrap();
        assert_eq!(dst, vec![0, 1]);
    }
}
