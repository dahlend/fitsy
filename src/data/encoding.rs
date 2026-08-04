//! BITPIX encoding (Standard Sec.4.4.1.1) and the `Pixel` trait that
//! lets each supported in-memory type decode itself from the raw
//! big-endian byte stream.

use crate::data::ieee;
use crate::error::{FitsError, Result};

/// The pixel encoding of an image, from its `BITPIX` card (Standard
/// Sec.4.4.1.1).
///
/// The standard defines six values. Three name a signed integer width,
/// one names an unsigned byte, and two name an IEEE float width. An
/// unsigned integer image uses a signed variant plus a `BZERO` card,
/// per Sec.4.4.2.5.
///
/// # Examples
///
/// ```
/// use fitsy::Bitpix;
///
/// assert_eq!(Bitpix::from_i64(16)?, Bitpix::I16);
/// assert_eq!(Bitpix::I16.byte_size(), 2);
/// assert_eq!(Bitpix::F64.as_i64(), -64);
///
/// // Half precision is not a FITS type.
/// assert!(Bitpix::from_i64(-16).is_err());
/// # Ok::<(), fitsy::FitsError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bitpix {
    /// `BITPIX = 8`: unsigned 8-bit integer.
    U8,
    /// `BITPIX = 16`: two's-complement 16-bit integer.
    I16,
    /// `BITPIX = 32`: two's-complement 32-bit integer.
    I32,
    /// `BITPIX = 64`: two's-complement 64-bit integer.
    I64,
    /// `BITPIX = -32`: IEEE 754 single-precision float.
    F32,
    /// `BITPIX = -64`: IEEE 754 double-precision float.
    F64,
}

impl Bitpix {
    /// Interpret a raw `BITPIX` card value.
    ///
    /// # Errors
    ///
    /// Any value outside the six the standard defines, with
    /// `BITPIX = -16` (half-precision, not a FITS type) called out
    /// separately because it turns up in non-conforming files.
    pub fn from_i64(v: i64) -> Result<Self> {
        Ok(match v {
            8 => Self::U8,
            16 => Self::I16,
            32 => Self::I32,
            64 => Self::I64,
            -32 => Self::F32,
            -64 => Self::F64,
            _ => {
                let msg = match v {
                    -16 => format!(
                        "unsupported BITPIX value {v}: half-precision floats (BITPIX=-16) \
                         are not part of the FITS standard and are not supported by fitsy"
                    ),
                    _ => format!(
                        "unsupported BITPIX value {v}: expected one of \
                         8, 16, 32, 64 (integer) or -32, -64 (IEEE float)"
                    ),
                };
                return Err(FitsError::Value {
                    keyword: "BITPIX".into(),
                    msg,
                });
            }
        })
    }

    /// Bytes per element.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    /// The Rust type this `BITPIX` decodes to, spelled as in the
    /// source: `u8`, `i16`, `i32`, `i64`, `f32` or `f64`.
    ///
    /// A typed read or write uses this to name the expected type and
    /// the found type in a
    /// [`FitsError::TypeMismatch`](crate::FitsError::TypeMismatch).
    #[must_use]
    pub(crate) const fn rust_type_name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// The value as it appears on the `BITPIX` card.
    #[must_use]
    pub const fn as_i64(self) -> i64 {
        match self {
            Self::U8 => 8,
            Self::I16 => 16,
            Self::I32 => 32,
            Self::I64 => 64,
            Self::F32 => -32,
            Self::F64 => -64,
        }
    }
}

/// A primitive that can be decoded as raw FITS pixels of a particular
/// `BITPIX`.
pub trait Pixel: Sized + Copy {
    /// The `BITPIX` this type is the in-memory form of.
    const BITPIX: Bitpix;
    /// Decode one element from its big-endian on-disk bytes.
    ///
    /// `bytes` must be at least `BITPIX::byte_size()` long; callers in
    /// this crate slice exactly that much.
    fn from_be_bytes(bytes: &[u8]) -> Self;
    /// Append the big-endian on-disk encoding of `self` to `out`.
    fn write_be(self, out: &mut Vec<u8>);
}

impl Pixel for u8 {
    const BITPIX: Bitpix = Bitpix::U8;
    fn from_be_bytes(b: &[u8]) -> Self {
        b[0]
    }
    fn write_be(self, out: &mut Vec<u8>) {
        out.push(self);
    }
}

impl Pixel for i16 {
    const BITPIX: Bitpix = Bitpix::I16;
    fn from_be_bytes(b: &[u8]) -> Self {
        Self::from_be_bytes([b[0], b[1]])
    }
    fn write_be(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_be_bytes());
    }
}

impl Pixel for i32 {
    const BITPIX: Bitpix = Bitpix::I32;
    fn from_be_bytes(b: &[u8]) -> Self {
        Self::from_be_bytes([b[0], b[1], b[2], b[3]])
    }
    fn write_be(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_be_bytes());
    }
}

impl Pixel for i64 {
    const BITPIX: Bitpix = Bitpix::I64;
    fn from_be_bytes(b: &[u8]) -> Self {
        Self::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
    fn write_be(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_be_bytes());
    }
}

impl Pixel for f32 {
    const BITPIX: Bitpix = Bitpix::F32;
    fn from_be_bytes(b: &[u8]) -> Self {
        // Preserve NaN bit patterns (Sec.4.4.2.5).
        ieee::f32_from_be_bytes_preserving_nan(b)
    }
    fn write_be(self, out: &mut Vec<u8>) {
        // Preserve NaN bit patterns: write the raw 32-bit pattern BE.
        out.extend_from_slice(&self.to_bits().to_be_bytes());
    }
}

impl Pixel for f64 {
    const BITPIX: Bitpix = Bitpix::F64;
    fn from_be_bytes(b: &[u8]) -> Self {
        ieee::f64_from_be_bytes_preserving_nan(b)
    }
    fn write_be(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_bits().to_be_bytes());
    }
}

/// An image array decoded into memory.
///
/// This pairs a flat element vector with the axis lengths, in FITS
/// order. `axes[0]` is `NAXIS1`, the fastest-varying axis, so the
/// element at `(x, y)` sits at index `y * axes[0] + x`.
///
/// # Examples
///
/// ```
/// use fitsy::ImageData;
///
/// let img = ImageData::new(vec![1_i16, 2, 3, 4, 5, 6], vec![3, 2])?;
///
/// assert_eq!(img.axes(), &[3, 2]);
/// assert_eq!(img.as_slice().len(), 6);
///
/// // Row 1, column 2 is index 1 * 3 + 2.
/// assert_eq!(img.as_slice()[1 * 3 + 2], 6);
/// # Ok::<(), fitsy::FitsError>(())
/// ```
#[derive(Debug, Clone)]
pub struct ImageData<T> {
    data: Vec<T>,
    axes: Vec<u64>,
}

impl<T> ImageData<T> {
    /// Pair a flat element vector with its axis lengths.
    ///
    /// # Errors
    ///
    /// If the axis product does not equal `data.len()`, or overflows
    /// `u64`. An empty `axes` means `NAXIS = 0` and requires an empty
    /// `data`, since the empty product would otherwise demand one
    /// element (Sec.4.4.1.1).
    pub fn new(data: Vec<T>, axes: Vec<u64>) -> Result<Self> {
        // `NAXIS = 0` means no data (Sec.4.4.1.1), but the empty
        // product is 1, which would demand a one-element array.
        let expected: u64 = if axes.is_empty() {
            0
        } else {
            axes.iter()
                .try_fold(1_u64, |acc, &n| acc.checked_mul(n))
                .ok_or_else(|| FitsError::Data("axis product overflows u64".into()))?
        };
        if expected != data.len() as u64 {
            return Err(FitsError::Data(format!(
                "axis product {expected} does not match data length {}",
                data.len()
            )));
        }
        Ok(Self { data, axes })
    }

    /// Axis lengths in FITS order: `axes()[0]` is `NAXIS1`, the
    /// fastest-varying axis.
    #[must_use]
    pub fn axes(&self) -> &[u64] {
        &self.axes
    }

    /// The elements, flat, with the first axis varying fastest.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Consume the array and return its elements, dropping the shape.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}
