//! BITPIX encoding (Standard Sec.4.4.1.1) and the `Pixel` trait that
//! lets each supported in-memory type decode itself from the raw
//! big-endian byte stream.

use crate::data::ieee;
use crate::error::{FitsError, Result};

/// `BITPIX` value (Standard Sec.4.4.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bitpix {
    U8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl Bitpix {
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
    const BITPIX: Bitpix;
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

/// An image array decoded into memory: a flat element vector plus
/// axis lengths in FITS order (`NAXIS1` is the *fastest-varying*
/// axis, i.e. `axes[0]`).
#[derive(Debug, Clone)]
pub struct ImageData<T> {
    data: Vec<T>,
    axes: Vec<u64>,
}

impl<T> ImageData<T> {
    pub fn new(data: Vec<T>, axes: Vec<u64>) -> Result<Self> {
        let expected: u64 = axes.iter().product();
        if expected as usize != data.len() {
            return Err(FitsError::Data(format!(
                "axis product {expected} does not match data length {}",
                data.len()
            )));
        }
        Ok(Self { data, axes })
    }

    #[must_use]
    pub fn axes(&self) -> &[u64] {
        &self.axes
    }

    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}
