//! Shared helpers for N-dimensional subarray iteration.
//!
//! The image read path ([`crate::hdu::image::ImageHdu::read_subarray`]),
//! the lazy on-disk read path
//! ([`crate::hdu::file::FitsFile::read_image_subarray_be`], gated on
//! the `python` feature), and the in-place writer
//! ([`crate::io::FitsUpdater::write_image_subarray`]) all walk the
//! same N-D index space. They differ in *what* they do per row (decode,
//! `pread`, `pwrite`), so the body of the walker is not shared. What
//! is shared lives here:
//!
//! - shape / bounds validation
//! - stride computation
//! - the carry-loop next-index increment

use crate::error::{FitsError, Result};

/// Validate that `start` and `shape` describe a sub-cuboid that fits
/// inside an image with the given `axes`. `op` prefixes any error
/// message so callers can produce diagnostics that name the public
/// entry point.
pub(crate) fn validate_subarray_shape(
    axes: &[u64],
    start: &[u64],
    shape: &[u64],
    op: &str,
) -> Result<()> {
    if start.len() != axes.len() || shape.len() != axes.len() {
        return Err(FitsError::Data(format!(
            "{op}: start/shape have length {}/{}, expected NAXIS = {}",
            start.len(),
            shape.len(),
            axes.len()
        )));
    }
    for (k, (&s, &n)) in start.iter().zip(shape.iter()).enumerate() {
        let axis = axes[k];
        if s.checked_add(n).is_none_or(|end| end > axis) {
            return Err(FitsError::Data(format!(
                "{op}: axis {} (NAXIS{}) range {s}..{} out of bounds (length {axis})",
                k,
                k + 1,
                s + n
            )));
        }
    }
    Ok(())
}

/// Element strides for `axes` in FITS order (NAXIS1 fastest-varying).
/// Returns the per-axis stride in elements; the first entry is `1`.
/// Errors if any cumulative product overflows `u64`.
pub(crate) fn checked_strides(axes: &[u64], op: &str) -> Result<Vec<u64>> {
    let mut strides: Vec<u64> = Vec::with_capacity(axes.len());
    let mut s = 1_u64;
    for &a in axes {
        strides.push(s);
        s = s.checked_mul(a).ok_or_else(|| {
            FitsError::Data(format!("{op}: axis stride overflows u64"))
        })?;
    }
    Ok(strides)
}

/// Advance an N-D iteration counter `idx` to the next position within
/// `shape`. Axis 0 is *not* iterated (callers vary it as a contiguous
/// row); axes 1..N are incremented in order with carry. Returns
/// `false` once the highest axis wraps, signalling iteration is done.
///
/// `idx.len()` must equal `shape.len()`. For 1-D shapes (`shape.len()
/// == 1`) iteration is always exhausted after the first row, so this
/// returns `false` immediately.
pub(crate) fn next_subarray_index(idx: &mut [u64], shape: &[u64]) -> bool {
    debug_assert_eq!(
        idx.len(),
        shape.len(),
        "next_subarray_index: idx and shape must have the same NAXIS length",
    );
    if shape.len() <= 1 {
        return false;
    }
    let mut ax = 1;
    loop {
        idx[ax] += 1;
        if idx[ax] < shape[ax] {
            return true;
        }
        idx[ax] = 0;
        ax += 1;
        if ax == shape.len() {
            return false;
        }
    }
}
