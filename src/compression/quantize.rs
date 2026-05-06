//! De-quantization of tile-compressed floating-point images
//! (Pence et al. 2010 Sec.4; FITS standard 2016 Sec.10.4.4).
//!
//! Quantized float tiles are decompressed by the underlying tile
//! algorithm (typically `RICE_1` or `GZIP_1`) into 32-bit signed
//! integers. This module turns those integers back into floats
//! using per-tile `ZSCALE` / `ZZERO` and the optional subtractive
//! dither sequence keyed by `ZDITHER0`.

use std::sync::OnceLock;

/// Length of the pre-computed random table (cfitsio constant).
pub(super) const N_RANDOM: usize = 10_000;

/// Sentinel integer for "this float was originally NaN/Inf".
pub(super) const NULL_VALUE: i32 = -2_147_483_647;

/// Sentinel integer used by `SUBTRACTIVE_DITHER_2` to flag an
/// exact-zero source pixel (bypasses dither so 0.0 round-trips).
pub(super) const ZERO_VALUE: i32 = -2_147_483_646;

/// Quantization / dither variant carried by `ZQUANTIZ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DitherMethod {
    /// `NO_DITHER` -- straight `int * scale + zero`.
    NoDither,
    /// `SUBTRACTIVE_DITHER_1` -- subtract the dither sequence.
    Subtractive1,
    /// `SUBTRACTIVE_DITHER_2` -- like 1 but a special integer
    /// sentinel (`ZERO_VALUE`) decodes back to exact 0.0.
    Subtractive2,
}

/// Pre-computed Park-Miller table, lazily initialized once per
/// process. Equivalent to cfitsio's `fits_rand_value[]`.
pub(super) fn random_values() -> &'static [f32; N_RANDOM] {
    static TABLE: OnceLock<[f32; N_RANDOM]> = OnceLock::new();
    TABLE.get_or_init(|| {
        const A: i64 = 16807;
        const M: i64 = 2_147_483_647;
        // M / A = 127773
        const Q: i64 = M / A;
        // M % A = 2836
        const R: i64 = M % A;
        #[allow(
            clippy::large_stack_arrays,
            reason = "this 40 KiB table is allocated exactly once in a static initializer"
        )]
        let mut table = [0.0_f32; N_RANDOM];
        let mut seed: i64 = 1;
        for slot in &mut table {
            // schrage's algorithm -- keeps everything in 32-bit range.
            let hi = seed / Q;
            let lo = seed - hi * Q;
            seed = A * lo - R * hi;
            if seed < 0 {
                seed += M;
            }
            *slot = (seed as f64 / M as f64) as f32;
        }
        table
    })
}

/// State machine that walks the dither table for a single tile,
/// matching cfitsio's `unquantize_i4r4` / `imcomp_decompress_tile`.
struct DitherWalker {
    table: &'static [f32; N_RANDOM],
    iseed: usize,
    nextrand: usize,
}

impl DitherWalker {
    /// `tile_seed_1based` is the global seed `(ZDITHER0 + tile_index)`
    /// where `tile_index` is the 1-based row number. Mod-N is applied
    /// internally.
    fn new(tile_seed_1based: u64) -> Self {
        let table = random_values();
        // cfitsio: iseed = (long)(((row - 1) + ditherseed) % N_RANDOM);
        let iseed = (tile_seed_1based.saturating_sub(1) as usize) % N_RANDOM;
        let nextrand = (table[iseed] * 500.0) as usize;
        Self {
            table,
            iseed,
            nextrand,
        }
    }
    #[inline]
    fn current(&self) -> f32 {
        self.table[self.nextrand]
    }
    #[inline]
    fn step(&mut self) {
        self.nextrand += 1;
        if self.nextrand >= N_RANDOM {
            self.iseed = (self.iseed + 1) % N_RANDOM;
            self.nextrand = (self.table[self.iseed] * 500.0) as usize;
        }
    }
}

/// Convert a tile of big-endian i32 quantized samples to f32 pixels.
///
/// `dst` is `4 * input.len() / 4` bytes long. Each input pixel is
/// decoded as `i32::from_be_bytes`; output pixels are written as
/// `f32::to_be_bytes`.
pub(super) fn unquantize_to_f32_be(
    input_be: &[u8],
    dst: &mut [u8],
    scale: f64,
    zero: f64,
    blank: i32,
    dither: Option<(DitherMethod, u64)>,
) {
    debug_assert_eq!(
        input_be.len() % 4,
        0,
        "input length {} must be a multiple of 4",
        input_be.len()
    );
    debug_assert_eq!(
        dst.len(),
        input_be.len(),
        "dst length {} must equal input length {}",
        dst.len(),
        input_be.len()
    );
    let mut walker = dither.map(|(_, seed)| DitherWalker::new(seed));
    let method = dither.map(|(m, _)| m);
    for (chunk_in, chunk_out) in input_be.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let v = i32::from_be_bytes([chunk_in[0], chunk_in[1], chunk_in[2], chunk_in[3]]);
        let f = decode_one(v, scale, zero, blank, method, walker.as_mut());
        let out = f.map_or(f32::NAN, |x| x as f32);
        chunk_out.copy_from_slice(&out.to_be_bytes());
    }
}

/// Same, for f64 output (the original image was `BITPIX = -64`).
pub(super) fn unquantize_to_f64_be(
    input_be: &[u8],
    dst: &mut [u8],
    scale: f64,
    zero: f64,
    blank: i32,
    dither: Option<(DitherMethod, u64)>,
) {
    debug_assert_eq!(
        input_be.len() % 4,
        0,
        "input length {} must be a multiple of 4",
        input_be.len()
    );
    debug_assert_eq!(
        dst.len(),
        2 * input_be.len(),
        "dst length {} must be twice input length {} for f64 output",
        dst.len(),
        input_be.len()
    );
    let mut walker = dither.map(|(_, seed)| DitherWalker::new(seed));
    let method = dither.map(|(m, _)| m);
    for (chunk_in, chunk_out) in input_be.chunks_exact(4).zip(dst.chunks_exact_mut(8)) {
        let v = i32::from_be_bytes([chunk_in[0], chunk_in[1], chunk_in[2], chunk_in[3]]);
        let f = decode_one(v, scale, zero, blank, method, walker.as_mut());
        let out = f.unwrap_or(f64::NAN);
        chunk_out.copy_from_slice(&out.to_be_bytes());
    }
}

/// Decode a single quantized integer. Returns `None` for the
/// blank sentinel (the caller substitutes its own NaN for the
/// appropriate output type), `Some(0.0)` for `SUBTRACTIVE_DITHER_2`'s
/// exact-zero sentinel, and the dequantized value otherwise. The
/// dither walker is advanced exactly once per pixel -- including on
/// blank / zero-sentinel values -- to keep phase alignment with
/// cfitsio.
#[inline]
fn decode_one(
    v: i32,
    scale: f64,
    zero: f64,
    blank: i32,
    method: Option<DitherMethod>,
    walker: Option<&mut DitherWalker>,
) -> Option<f64> {
    if v == blank {
        if let Some(w) = walker {
            w.step();
        }
        return None;
    }
    Some(match method {
        None | Some(DitherMethod::NoDither) => f64::from(v) * scale + zero,
        Some(DitherMethod::Subtractive1) => {
            let w = walker.expect("dither walker required for SUBTRACTIVE_DITHER_1");
            let r = f64::from(w.current());
            w.step();
            (f64::from(v) - r + 0.5) * scale + zero
        }
        Some(DitherMethod::Subtractive2) => {
            let w = walker.expect("dither walker required for SUBTRACTIVE_DITHER_2");
            if v == ZERO_VALUE {
                w.step();
                return Some(0.0);
            }
            let r = f64::from(w.current());
            w.step();
            (f64::from(v) - r + 0.5) * scale + zero
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First random value reproduces cfitsio's table[0] = ~7.826e-6.
    /// Park-Miller seed 1 -> next state 16807; 16807 / 2147483647.
    #[test]
    fn random_table_matches_park_miller() {
        let t = random_values();
        let expected0 = (16807.0_f64 / 2_147_483_647.0_f64) as f32;
        assert!((t[0] - expected0).abs() < 1e-12);
    }

    #[test]
    fn no_dither_is_linear() {
        let input: Vec<u8> = [10_i32, -5, 0]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let mut out = vec![0_u8; 12];
        unquantize_to_f32_be(&input, &mut out, 2.0, 100.0, NULL_VALUE, None);
        let vals: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals, vec![120.0, 90.0, 100.0]);
    }

    #[test]
    fn null_value_decodes_to_nan() {
        let input: Vec<u8> = NULL_VALUE.to_be_bytes().to_vec();
        let mut out = vec![0_u8; 4];
        unquantize_to_f32_be(&input, &mut out, 1.0, 0.0, NULL_VALUE, None);
        let v = f32::from_be_bytes([out[0], out[1], out[2], out[3]]);
        assert!(v.is_nan());
    }

    #[test]
    fn subtractive_dither_2_zero_value_decodes_to_zero() {
        let input: Vec<u8> = ZERO_VALUE.to_be_bytes().to_vec();
        let mut out = vec![0_u8; 4];
        unquantize_to_f32_be(
            &input,
            &mut out,
            1.0,
            0.0,
            NULL_VALUE,
            Some((DitherMethod::Subtractive2, 1)),
        );
        let v = f32::from_be_bytes([out[0], out[1], out[2], out[3]]);
        assert_eq!(v, 0.0);
    }
}
