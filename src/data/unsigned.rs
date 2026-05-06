//! Unsigned-integer convention via `BZERO`/`BSCALE` (Standard Sec.4.4.2.5,
//! Sec.4.4.2.5 Table 11).
//!
//! FITS encodes unsigned 16/32/64-bit integers by writing them as the
//! corresponding signed type with `BSCALE = 1` and a fixed offset
//! `BZERO`. The offsets are exact powers of two.

/// `BZERO` for `u16` represented as `i16`: 2^15.
pub const BZERO_U16: i64 = 32_768;
/// `BZERO` for `u32` represented as `i32`: 2^31.
pub const BZERO_U32: i64 = 2_147_483_648;
/// `BZERO` for `u64` represented as `i64`: 2^63. Stored as `f64`
/// because it does not fit in `i64`.
pub const BZERO_U64_F64: f64 = 9_223_372_036_854_775_808.0;

/// Detect the FITS unsigned-integer convention from `BITPIX`,
/// `BSCALE`, and `BZERO`. Returns `Some(width_in_bits)` when the
/// header matches one of the standard offsets exactly.
#[must_use]
pub fn unsigned_width(bitpix: i64, bscale: f64, bzero: f64) -> Option<u32> {
    if bscale != 1.0 {
        return None;
    }
    match bitpix {
        16 if bzero == BZERO_U16 as f64 => Some(16),
        32 if bzero == BZERO_U32 as f64 => Some(32),
        64 if bzero == BZERO_U64_F64 => Some(64),
        _ => None,
    }
}
