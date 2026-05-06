//! `RICE_1` tile decompressor (and a matching encoder used only by
//! the test suite).
//!
//! Implements the algorithm described in Pence, Seaman &
//! White 2010 Sec.3.1 and adopted into the FITS standard (2016) Sec.10.4.
//!
//! Each tile is encoded as a sequence of fixed-length blocks
//! (default 32 pixels). The first pixel of the tile is written
//! verbatim; subsequent pixels are stored as `zigzag(p[i] - p[i-1])`
//! and Rice-coded with the split parameter `fs` chosen per block
//! from the average magnitude. There are three special cases per
//! block: all-zero (low entropy), `fs >= fsmax` (high entropy: raw
//! values), and the normal Rice-coded path.
//!
//! Per the C source, `(fsbits, fsmax, bbits)` is keyed only on the
//! pixel width: `(3, 6, 8)` for 1-byte, `(4, 14, 16)` for 2-byte,
//! `(5, 25, 32)` for 4-byte. Differencing and reconstruction are
//! performed with wrapping unsigned arithmetic; the output bit
//! pattern matches the original signed pixels.

use crate::error::{FitsError, Result};

const NONZERO_COUNT: [u8; 256] = {
    let mut t = [0_u8; 256];
    let mut i = 1_usize;
    while i < 256 {
        let mut bits = 0_u8;
        let mut v = i;
        while v != 0 {
            bits += 1;
            v >>= 1;
        }
        t[i] = bits;
        i += 1;
    }
    t
};

/// Per-pixel-width Rice parameters, fixed by `cfitsio/ricecomp.c`.
#[derive(Debug, Clone, Copy)]
struct RiceParams {
    fsbits: u32,
    fsmax: u32,
    bbits: u32,
}

impl RiceParams {
    fn for_bytepix(bytepix: u32) -> Result<Self> {
        Ok(match bytepix {
            1 => Self {
                fsbits: 3,
                fsmax: 6,
                bbits: 8,
            },
            2 => Self {
                fsbits: 4,
                fsmax: 14,
                bbits: 16,
            },
            4 => Self {
                fsbits: 5,
                fsmax: 25,
                bbits: 32,
            },
            other => {
                return Err(FitsError::Value {
                    keyword: "ZVAL_BYTEPIX".into(),
                    msg: format!("RICE BYTEPIX must be 1, 2, or 4, got {other}"),
                });
            }
        })
    }
}

// -- Public decode API -----------------------------------------------

/// Decompress one Rice-encoded tile of `nx` pixels into `dst`,
/// which must hold exactly `nx * bytepix` bytes. Output is big-endian
/// pixels matching the surrounding image's storage.
pub(super) fn decompress_into(
    bytepix: u32,
    nblock: u32,
    nx: u32,
    src: &[u8],
    dst: &mut [u8],
) -> Result<()> {
    let bp = bytepix as usize;
    let nx = nx as usize;
    let nblock = nblock as usize;
    if dst.len() != nx * bp {
        return Err(FitsError::Data(format!(
            "RICE: destination buffer is {} bytes, need {}",
            dst.len(),
            nx * bp
        )));
    }
    if nx == 0 {
        return Ok(());
    }
    let p = RiceParams::for_bytepix(bytepix)?;
    if src.len() < bp {
        return Err(FitsError::Data(
            "RICE: input shorter than the verbatim leading pixel".into(),
        ));
    }
    let lastpix: u32 = match bytepix {
        1 => u32::from(src[0]),
        2 => (u32::from(src[0]) << 8) | u32::from(src[1]),
        4 => {
            (u32::from(src[0]) << 24)
                | (u32::from(src[1]) << 16)
                | (u32::from(src[2]) << 8)
                | u32::from(src[3])
        }
        _ => unreachable!(),
    };
    write_be_pixel(dst, 0, bp, lastpix);
    if nx == 1 {
        return Ok(());
    }
    let mut r = BitReader::new(src, bp)?;
    decode_blocks(&mut r, dst, nx, nblock, bp, p, lastpix)
}

/// Convenience wrapper that allocates the output `Vec`.
#[cfg(test)]
pub(super) fn decompress(bytepix: u32, nblock: u32, nx: u32, src: &[u8]) -> Result<Vec<u8>> {
    let mut out = vec![0_u8; (nx as usize) * (bytepix as usize)];
    decompress_into(bytepix, nblock, nx, src, &mut out)?;
    Ok(out)
}

// -- Internal decoder ------------------------------------------------

#[inline]
fn write_be_pixel(dst: &mut [u8], i: usize, bp: usize, v: u32) {
    let off = i * bp;
    match bp {
        1 => dst[off] = v as u8,
        2 => dst[off..off + 2].copy_from_slice(&(v as u16).to_be_bytes()),
        4 => dst[off..off + 4].copy_from_slice(&v.to_be_bytes()),
        _ => unreachable!(),
    }
}

struct BitReader<'a> {
    src: &'a [u8],
    pos: usize,
    /// Pending bits, stored in the low `nbits` bits of `b`.
    b: u32,
    nbits: i32,
}

impl<'a> BitReader<'a> {
    fn new(src: &'a [u8], pos: usize) -> Result<Self> {
        let mut r = Self {
            src,
            pos,
            b: 0,
            nbits: 8,
        };
        r.b = u32::from(r.read_byte()?);
        Ok(r)
    }
    fn read_byte(&mut self) -> Result<u8> {
        let b = *self
            .src
            .get(self.pos)
            .ok_or_else(|| FitsError::Data("RICE: hit end of compressed stream".into()))?;
        self.pos += 1;
        Ok(b)
    }
}

fn decode_blocks(
    r: &mut BitReader<'_>,
    dst: &mut [u8],
    nx: usize,
    nblock: usize,
    bp: usize,
    p: RiceParams,
    mut lastpix: u32,
) -> Result<()> {
    // Pixel 0 is the verbatim lead pixel; start decoding from index 1.
    let mut i = 1_usize;
    while i < nx {
        // Read fsbits to obtain `fs + 1`.
        r.nbits -= p.fsbits as i32;
        while r.nbits < 0 {
            r.b = (r.b << 8) | u32::from(r.read_byte()?);
            r.nbits += 8;
        }
        let fs_plus_one = (r.b >> (r.nbits as u32)) as i32;
        r.b &= mask32(r.nbits as u32);
        let fs = fs_plus_one - 1;
        let imax = (i + nblock).min(nx);

        if fs < 0 {
            for k in i..imax {
                write_be_pixel(dst, k, bp, lastpix);
            }
        } else if fs as u32 == p.fsmax {
            for k in i..imax {
                let mut bb = p.bbits as i32 - r.nbits;
                let mut diff: u32 = if bb >= 0 {
                    r.b.wrapping_shl(bb as u32)
                } else {
                    r.b >> ((-bb) as u32)
                };
                bb -= 8;
                while bb >= 0 {
                    r.b = u32::from(r.read_byte()?);
                    diff |= r.b.wrapping_shl(bb as u32);
                    bb -= 8;
                }
                if r.nbits > 0 {
                    r.b = u32::from(r.read_byte()?);
                    diff |= r.b >> ((-bb) as u32);
                    r.b &= mask32(r.nbits as u32);
                } else {
                    r.b = 0;
                }
                debug_assert!(
                    (0..=8).contains(&r.nbits),
                    "r.nbits={} must be in range 0..=8",
                    r.nbits
                );
                let pixel = lastpix.wrapping_add(unzigzag(diff));
                write_be_pixel(dst, k, bp, pixel);
                lastpix = pixel;
            }
        } else {
            let fs_u = fs as u32;
            for k in i..imax {
                while r.b == 0 {
                    r.nbits += 8;
                    r.b = u32::from(r.read_byte()?);
                }
                let nzero = r.nbits - i32::from(NONZERO_COUNT[r.b as usize]);
                r.nbits -= nzero + 1;
                r.b ^= 1_u32 << (r.nbits as u32);
                r.nbits -= fs_u as i32;
                while r.nbits < 0 {
                    r.b = (r.b << 8) | u32::from(r.read_byte()?);
                    r.nbits += 8;
                }
                let trailing = r.b >> (r.nbits as u32);
                let diff = ((nzero as u32) << fs_u) | trailing;
                r.b &= mask32(r.nbits as u32);
                let pixel = lastpix.wrapping_add(unzigzag(diff));
                write_be_pixel(dst, k, bp, pixel);
                lastpix = pixel;
            }
        }
        i = imax;
    }
    Ok(())
}

/// Mask with the low `n` bits set; safe for `n in [0, 32]`.
#[inline]
fn mask32(n: u32) -> u32 {
    if n >= 32 { u32::MAX } else { (1_u32 << n) - 1 }
}

#[inline]
fn unzigzag(d: u32) -> u32 {
    if d & 1 == 0 { d >> 1 } else { !(d >> 1) }
}

// -- Encoder (test-only) ---------------------------------------------

#[cfg(test)]
#[inline]
fn zigzag(d: i64) -> u32 {
    if d < 0 {
        !((d as u32) << 1)
    } else {
        (d as u32) << 1
    }
}

#[cfg(test)]
struct BitWriter {
    out: Vec<u8>,
    bitbuffer: u32,
    bits_to_go: i32,
}

#[cfg(test)]
impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            bitbuffer: 0,
            bits_to_go: 8,
        }
    }

    /// Write the low `n` bits of `bits`, MSB first. `n` must be in
    /// `[0, 32]`. All shifts are wrapping to keep the function safe
    /// at the boundary.
    fn output_nbits(&mut self, bits: u32, n: i32) {
        debug_assert!((0..=32).contains(&n));
        if n == 0 {
            return;
        }
        let mut n = n;
        let mut bits = bits & mask32(n as u32);
        if self.bits_to_go + n > 32 {
            let high = bits >> ((n - self.bits_to_go) as u32);
            self.bitbuffer = self.bitbuffer.wrapping_shl(self.bits_to_go as u32)
                | (high & mask32(self.bits_to_go as u32));
            self.out.push((self.bitbuffer & 0xff) as u8);
            n -= self.bits_to_go;
            self.bits_to_go = 8;
            bits &= mask32(n as u32);
        }
        self.bitbuffer = self.bitbuffer.wrapping_shl(n as u32) | (bits & mask32(n as u32));
        self.bits_to_go -= n;
        while self.bits_to_go <= 0 {
            self.out
                .push(((self.bitbuffer >> ((-self.bits_to_go) as u32)) & 0xff) as u8);
            self.bits_to_go += 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits_to_go < 8 {
            self.out
                .push(((self.bitbuffer << self.bits_to_go as u32) & 0xff) as u8);
        }
        self.out
    }
}

#[cfg(test)]
fn encode_inner(pixels: &[i64], bytepix: u32, nblock: usize, first_bits: i32) -> Vec<u8> {
    let p = RiceParams::for_bytepix(bytepix).unwrap();
    let mut w = BitWriter::new();
    w.output_nbits(pixels[0] as u32, first_bits);
    let mut lastpix = pixels[0];
    let mut diffs = Vec::<u32>::with_capacity(nblock);
    // The seed pixel is written verbatim; deltas start at index 1 so
    // the decoder's first block sees `nblock` real deltas, matching
    // `cfitsio/ricecomp.c`.
    let mut i = 1_usize;
    while i < pixels.len() {
        let thisblock = nblock.min(pixels.len() - i);
        diffs.clear();
        let mut pixelsum: f64 = 0.0;
        for j in 0..thisblock {
            let nextpix = pixels[i + j];
            let pdiff = match bytepix {
                1 => i64::from((nextpix as i8).wrapping_sub(lastpix as i8)),
                2 => i64::from((nextpix as i16).wrapping_sub(lastpix as i16)),
                4 => i64::from((nextpix as i32).wrapping_sub(lastpix as i32)),
                _ => unreachable!(),
            };
            let z = zigzag(pdiff);
            diffs.push(z);
            pixelsum += f64::from(z);
            lastpix = nextpix;
        }
        let dpsum = ((pixelsum - (thisblock as f64) / 2.0 - 1.0) / thisblock as f64).max(0.0);
        let psum_full = dpsum as u64;
        let mut psum = match bytepix {
            1 => u32::from(psum_full as u8) >> 1,
            2 => u32::from(psum_full as u16) >> 1,
            _ => (psum_full as u32) >> 1,
        };
        let mut fs: u32 = 0;
        while psum > 0 {
            fs += 1;
            psum >>= 1;
        }
        if fs >= p.fsmax {
            w.output_nbits(p.fsmax + 1, p.fsbits as i32);
            for &d in &diffs {
                w.output_nbits(d, p.bbits as i32);
            }
        } else if fs == 0 && pixelsum == 0.0 {
            w.output_nbits(0, p.fsbits as i32);
        } else {
            w.output_nbits(fs + 1, p.fsbits as i32);
            for &d in &diffs {
                let top = d >> fs;
                for _ in 0..top {
                    w.output_nbits(0, 1);
                }
                w.output_nbits(1, 1);
                if fs > 0 {
                    w.output_nbits(d & ((1_u32 << fs) - 1), fs as i32);
                }
            }
        }
        i += thisblock;
    }
    w.finish()
}

#[cfg(test)]
pub(super) fn encode_byte(pixels: &[i8], nblock: usize) -> Vec<u8> {
    let v: Vec<i64> = pixels.iter().map(|&p| i64::from(p)).collect();
    encode_inner(&v, 1, nblock, 8)
}
#[cfg(test)]
pub(super) fn encode_short(pixels: &[i16], nblock: usize) -> Vec<u8> {
    let v: Vec<i64> = pixels.iter().map(|&p| i64::from(p)).collect();
    encode_inner(&v, 2, nblock, 16)
}
#[cfg(test)]
pub(super) fn encode_int(pixels: &[i32], nblock: usize) -> Vec<u8> {
    let v: Vec<i64> = pixels.iter().map(|&p| i64::from(p)).collect();
    encode_inner(&v, 4, nblock, 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_i16(pixels: &[i16], nblock: usize) {
        let enc = encode_short(pixels, nblock);
        let bytes = decompress(2, nblock as u32, pixels.len() as u32, &enc).unwrap();
        let decoded: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|c| i16::from_be_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(decoded, pixels);
    }

    fn round_trip_i32(pixels: &[i32], nblock: usize) {
        let enc = encode_int(pixels, nblock);
        let bytes = decompress(4, nblock as u32, pixels.len() as u32, &enc).unwrap();
        let decoded: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded, pixels);
    }

    fn round_trip_i8(pixels: &[i8], nblock: usize) {
        let enc = encode_byte(pixels, nblock);
        let bytes = decompress(1, nblock as u32, pixels.len() as u32, &enc).unwrap();
        let decoded: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn rice_short_low_entropy_constant() {
        round_trip_i16(&[100; 64], 32);
    }

    #[test]
    fn rice_short_increasing_sequence() {
        let v: Vec<i16> = (0..100).map(|i| i as i16 * 3 - 50).collect();
        round_trip_i16(&v, 32);
    }

    #[test]
    fn rice_short_negative_values() {
        let v: Vec<i16> = (0..40).map(|i| -1000 + i * 73).collect();
        round_trip_i16(&v, 16);
    }

    #[test]
    fn rice_short_high_entropy_random_like() {
        let v: Vec<i16> = (0..96)
            .map(|i| (i64::from(i) * 2654435761) as i32 as i16)
            .collect();
        round_trip_i16(&v, 32);
    }

    #[test]
    fn rice_int_round_trip() {
        let v: Vec<i32> = (0..200).map(|i| i * 1_000_003 - 100_000_000).collect();
        round_trip_i32(&v, 32);
    }

    #[test]
    fn rice_byte_round_trip() {
        let v: Vec<i8> = (-60..60).map(|i| i as i8).collect();
        round_trip_i8(&v, 16);
    }

    #[test]
    fn rice_short_short_block() {
        round_trip_i16(&[1, 2, 3, 5, 8, 13, 21, 34, 55], 4);
    }

    #[test]
    fn rice_short_single_pixel() {
        round_trip_i16(&[12345], 32);
    }

    #[test]
    fn decompress_into_rejects_wrong_dst_len() {
        let v = [1_i16, 2, 3];
        let enc = encode_short(&v, 32);
        let mut dst = [0_u8; 8];
        assert!(decompress_into(2, 32, 3, &enc, &mut dst).is_err());
    }

    #[test]
    fn decompress_rejects_truncated_stream() {
        // Encode then chop the buffer in half.
        let v: Vec<i16> = (0..100).map(|i| i as i16).collect();
        let mut enc = encode_short(&v, 32);
        enc.truncate(enc.len() / 2);
        let r = decompress(2, 32, v.len() as u32, &enc);
        assert!(r.is_err());
    }
}
