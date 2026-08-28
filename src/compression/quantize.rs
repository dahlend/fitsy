//! Quantization of tile-compressed floating-point images
//! (Pence et al. 2010 Sec.4; FITS standard 2016 Sec.10.4.4).
//!
//! Quantized float tiles are decompressed by the underlying tile
//! algorithm (typically `RICE_1` or `GZIP_1`) into 32-bit signed
//! integers. The decode half of this module turns those integers
//! back into floats using per-tile `ZSCALE` / `ZZERO` and the
//! optional subtractive dither sequence keyed by `ZDITHER0`. The
//! encode half runs the same mapping in reverse for the opt-in
//! lossy write path.

use std::sync::OnceLock;

/// Length of the pre-computed random table, fixed by the convention.
pub(super) const N_RANDOM: usize = 10_000;

/// Sentinel integer for "this float was originally NaN/Inf".
pub(super) const NULL_VALUE: i32 = -2_147_483_647;

/// Sentinel integer used by `SUBTRACTIVE_DITHER_2` to flag an
/// exact-zero source pixel (bypasses dither so 0.0 round-trips).
pub(super) const ZERO_VALUE: i32 = -2_147_483_646;

/// Quantization / dither variant carried by `ZQUANTIZ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitherMethod {
    /// `NO_DITHER` -- straight `int * scale + zero`.
    NoDither,
    /// `SUBTRACTIVE_DITHER_1` -- subtract the dither sequence.
    Subtractive1,
    /// `SUBTRACTIVE_DITHER_2` -- like 1 but a special integer
    /// sentinel (`ZERO_VALUE`) decodes back to exact 0.0.
    Subtractive2,
}

impl DitherMethod {
    /// The `ZQUANTIZ` value for this method.
    #[must_use]
    pub fn zquantiz(&self) -> &'static str {
        match self {
            Self::NoDither => "NO_DITHER",
            Self::Subtractive1 => "SUBTRACTIVE_DITHER_1",
            Self::Subtractive2 => "SUBTRACTIVE_DITHER_2",
        }
    }
}

/// Pre-computed Park-Miller table, lazily initialized once per
/// process. This is the table that the convention defines.
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
/// as the dequantization step of the convention defines it.
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
    /// The dither value for the current pixel.
    #[inline]
    fn current(&self) -> f32 {
        self.table[self.nextrand]
    }
    /// Advance to the next dither value, re-seeding from the table
    /// once the walk reaches its end.
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
    for (chunk_in, chunk_out) in input_be
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<4>().0.iter_mut())
    {
        let v = i32::from_be_bytes(*chunk_in);
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
    for (chunk_in, chunk_out) in input_be
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<8>().0.iter_mut())
    {
        let v = i32::from_be_bytes(*chunk_in);
        let f = decode_one(v, scale, zero, blank, method, walker.as_mut());
        let out = f.unwrap_or(f64::NAN);
        chunk_out.copy_from_slice(&out.to_be_bytes());
    }
}

/// Decode a single quantized integer.
///
/// The result is `None` for the blank sentinel, and the caller then
/// substitutes the NaN of its own output type. It is `Some(0.0)` for
/// the exact-zero sentinel of `SUBTRACTIVE_DITHER_2`. Otherwise it is
/// the dequantized value.
///
/// The dither walker advances exactly once per pixel, including on a
/// blank value and on the zero sentinel. That keeps its phase aligned
/// with the phase the encoder used.
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

// -- Encoder ---------------------------------------------------------

/// Estimate the noise sigma of `values` with the second-difference
/// median-absolute-deviation estimator of Pence et al. 2010 Sec.4.
///
/// The result is 0.0 in two cases. The first is a tile with no three
/// adjacent finite pixels. The second is a tile where more than half
/// the second differences are zero, because the median is then zero.
///
/// The caller reads 0.0 as "do not quantize". It compresses the tile
/// losslessly instead. That is the correct result for a tile that is
/// constant apart from a few outliers. Such a tile carries no noise,
/// so it sets no scale for a quantization step. The outliers alone
/// would set a step far wider than the data they came from.
fn second_difference_noise(values: &[f64]) -> f64 {
    // sigma = 0.6052697 * median(|x[i-1] - 2 x[i] + x[i+1]|) for
    // Gaussian noise; the constant is 1 / (sqrt(6) * Phi^-1(3/4)).
    const SIGMA_FACTOR: f64 = 0.6052697;
    // Only triples of adjacent finite pixels contribute. A gap left
    // by a non-finite pixel is skipped rather than closed up: a
    // second difference taken across a hole measures the gap, not
    // the noise.
    let mut diffs: Vec<f64> = values
        .windows(3)
        .filter(|w| w.iter().all(|v| v.is_finite()))
        .map(|w| (w[0] - 2.0 * w[1] + w[2]).abs())
        .collect();
    if diffs.is_empty() {
        return 0.0;
    }
    // The median covers every difference, zeros included. A zero
    // median means most of the tile is flat, which is exactly the
    // case that must not quantize.
    let mid = diffs.len() / 2;
    let (_, median, _) = diffs.select_nth_unstable_by(mid, f64::total_cmp);
    SIGMA_FACTOR * *median
}

/// Headroom kept between the quantized range and the `i32` limits, so
/// no data value collides with [`NULL_VALUE`] or [`ZERO_VALUE`].
const RANGE_LIMIT: f64 = 1_073_741_824.0;

/// Quantize one tile of float pixels to `i32` samples.
///
/// The `level` argument is the quantization level `q`. The step is
/// the estimated tile noise divided by `q`. The `dither` argument
/// carries the method and the tile seed. That seed is `ZDITHER0` plus
/// the 0-based row number, exactly as [`unquantize_to_f32_be`]
/// receives it.
///
/// The result is `(samples, scale, zero)`, or `None` when the tile
/// cannot be quantized -- no measurable noise, or a value range too
/// wide for `i32` at the chosen step. The caller then compresses the
/// tile losslessly instead.
///
/// A non-finite pixel becomes [`NULL_VALUE`]. Under
/// `SUBTRACTIVE_DITHER_2` an exact zero becomes [`ZERO_VALUE`]. The
/// dither walker advances once per pixel, sentinels included,
/// matching [`decode_one`].
pub(super) fn quantize_tile(
    values: &[f64],
    level: f64,
    dither: Option<(DitherMethod, u64)>,
) -> Option<(Vec<i32>, f64, f64)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in values.iter().copied().filter(|v| v.is_finite()) {
        min = min.min(v);
        max = max.max(v);
    }
    if min > max {
        // No finite value: every sample is the null sentinel and no
        // scale is needed.
        return Some((vec![NULL_VALUE; values.len()], 1.0, 0.0));
    }
    let noise = second_difference_noise(values);
    if noise <= 0.0 || level <= 0.0 {
        return None;
    }
    let scale = noise / level;
    let zero = min.midpoint(max);
    if (max - zero) / scale >= RANGE_LIMIT {
        return None;
    }
    let mut walker = dither
        .filter(|(m, _)| *m != DitherMethod::NoDither)
        .map(|(_, seed)| DitherWalker::new(seed));
    let method = dither.map(|(m, _)| m);
    let out = values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                if let Some(w) = walker.as_mut() {
                    w.step();
                }
                return NULL_VALUE;
            }
            if matches!(method, Some(DitherMethod::Subtractive2)) && v == 0.0 {
                let w = walker
                    .as_mut()
                    .expect("dither walker exists for SUBTRACTIVE_DITHER_2");
                w.step();
                return ZERO_VALUE;
            }
            let r = walker.as_mut().map_or(0.0, |w| {
                let r = f64::from(w.current()) - 0.5;
                w.step();
                r
            });
            ((v - zero) / scale + r).round() as i32
        })
        .collect();
    Some((out, scale, zero))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first random value reproduces the tabulated 7.826e-6 that
    /// the convention defines. A Park-Miller generator seeded with 1
    /// advances to state 16807, and 16807 / 2147483647 is that value.
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_be_bytes(*c))
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

    /// Quantize then dequantize; every finite value must come back
    /// within half a quantization step, and NaN must come back NaN.
    fn assert_round_trip(values: &[f64], dither: Option<(DitherMethod, u64)>) {
        let (ints, scale, zero) = quantize_tile(values, 4.0, dither).expect("tile quantizes");
        let input: Vec<u8> = ints.iter().flat_map(|v| v.to_be_bytes()).collect();
        let mut out = vec![0_u8; values.len() * 8];
        unquantize_to_f64_be(&input, &mut out, scale, zero, NULL_VALUE, dither);
        for (v, chunk) in values.iter().zip(out.as_chunks::<8>().0) {
            let d = f64::from_be_bytes(*chunk);
            if v.is_finite() {
                assert!(
                    (d - v).abs() <= scale * 0.5 + 1e-12,
                    "value {v} decoded to {d}, step {scale}"
                );
            } else {
                assert!(d.is_nan(), "non-finite {v} decoded to {d}");
            }
        }
    }

    /// Deterministic pseudo-noise on a slope, sigma of order 1.
    ///
    /// The noise comes from a 32-bit LCG, so its values are
    /// fine-grained and a second difference is essentially never
    /// exactly zero. A low-period integer pattern would read as a
    /// flat tile to [`second_difference_noise`], which is the one
    /// case that deliberately refuses to quantize.
    fn noisy_values(n: usize) -> Vec<f64> {
        let mut x: u32 = 12_345;
        (0..n)
            .map(|i| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (i as f64) * 0.05 + f64::from(x) / f64::from(u32::MAX) * 4.0
            })
            .collect()
    }

    #[test]
    fn quantize_refuses_a_tile_that_is_flat_apart_from_outliers() {
        // Most second differences are zero, so the median is zero.
        // The tile carries no noise that sets a step. Quantizing here
        // would let the few outliers set the step for every pixel.
        let mut v = vec![0.0_f64; 400];
        for i in (0..400).step_by(50) {
            v[i] = 1000.0;
        }
        assert!(quantize_tile(&v, 4.0, None).is_none());
    }

    #[test]
    fn noise_estimate_ignores_triples_spanning_a_nan() {
        // A hole must not join its neighbors into a spurious jump.
        let clean: Vec<f64> = noisy_values(200);
        let mut holed = clean.clone();
        for i in (20..180).step_by(37) {
            holed[i] = f64::NAN;
        }
        let a = second_difference_noise(&clean);
        let b = second_difference_noise(&holed);
        assert!(
            (a - b).abs() < 0.25 * a,
            "NaN holes moved the estimate from {a} to {b}"
        );
    }

    #[test]
    fn quantize_round_trips_no_dither() {
        assert_round_trip(&noisy_values(300), None);
    }

    #[test]
    fn quantize_round_trips_dither_1() {
        assert_round_trip(&noisy_values(300), Some((DitherMethod::Subtractive1, 42)));
    }

    #[test]
    fn quantize_round_trips_dither_2_with_nan_and_zero() {
        let mut v = noisy_values(300);
        v[7] = f64::NAN;
        v[100] = 0.0;
        v[200] = f64::INFINITY;
        let dither = Some((DitherMethod::Subtractive2, 7));
        assert_round_trip(&v, dither);
        // The exact zero survives exactly, through ZERO_VALUE.
        let (ints, scale, zero) = quantize_tile(&v, 4.0, dither).unwrap();
        assert_eq!(ints[7], NULL_VALUE);
        assert_eq!(ints[100], ZERO_VALUE);
        assert_eq!(ints[200], NULL_VALUE);
        let input: Vec<u8> = ints.iter().flat_map(|i| i.to_be_bytes()).collect();
        let mut out = vec![0_u8; v.len() * 8];
        unquantize_to_f64_be(&input, &mut out, scale, zero, NULL_VALUE, dither);
        let z = f64::from_be_bytes(out[800..808].try_into().unwrap());
        assert_eq!(z, 0.0);
    }

    #[test]
    fn quantize_rejects_constant_tile() {
        assert!(quantize_tile(&[5.0; 100], 4.0, None).is_none());
    }

    #[test]
    fn quantize_all_nan_tile_is_all_null() {
        let (ints, _, _) = quantize_tile(&[f64::NAN; 8], 4.0, None).unwrap();
        assert_eq!(ints, vec![NULL_VALUE; 8]);
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
