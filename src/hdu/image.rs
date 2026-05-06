//! Image HDUs (Standard Sec.7.1, Sec.3.3.1).
//!
//! Use [`ImageHdu::read_physical`] for the common case: it decodes
//! the raw integer or float pixels and applies `BZERO`/`BSCALE`,
//! returning `f64` values. Use [`ImageHdu::read_raw`] when you need
//! the native type without scaling.

use crate::data::encoding::{Bitpix, ImageData, Pixel};
use crate::data::scaling::Scaling;
use crate::error::{FitsError, Result};
use crate::header::Header;
use crate::io::block::pad_to_block;

/// An image HDU.
#[derive(Debug)]
pub struct ImageHdu<'a> {
    header: Header,
    data: &'a [u8],
    bitpix: Bitpix,
    axes: Vec<u64>,
    n_elements: u64,
}

impl<'a> ImageHdu<'a> {
    /// Construct from a parsed header and a slice covering the raw
    /// data section (no trailing padding).
    pub fn new(header: Header, data: &'a [u8]) -> Result<Self> {
        let bitpix = Bitpix::from_i64(header.bitpix()?)?;
        let axes = header.axes()?;
        let n_elements: u64 = if axes.is_empty() || axes.contains(&0) {
            0
        } else {
            axes.iter().product()
        };
        let needed = n_elements * bitpix.byte_size() as u64;
        if data.len() as u64 != needed {
            return Err(FitsError::Data(format!(
                "data slice {} bytes does not match expected {needed}",
                data.len()
            )));
        }
        Ok(Self {
            header,
            data,
            bitpix,
            axes,
            n_elements,
        })
    }

    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }

    #[must_use]
    pub fn bitpix(&self) -> Bitpix {
        self.bitpix
    }

    #[must_use]
    pub fn axes(&self) -> &[u64] {
        &self.axes
    }

    /// Number of pixels in the array. Zero for `NAXIS = 0` or any
    /// `NAXISn = 0`.
    #[must_use]
    pub fn n_elements(&self) -> u64 {
        self.n_elements
    }

    /// Raw data bytes (big-endian, unscaled).
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        self.data
    }

    /// Decode the array into native primitives without applying
    /// `BZERO`/`BSCALE`. The element type `T` must match `BITPIX`.
    pub fn read_raw<T: Pixel>(&self) -> Result<ImageData<T>> {
        if T::BITPIX != self.bitpix {
            return Err(FitsError::HduMismatch {
                expected: bitpix_name(T::BITPIX),
                found: bitpix_name(self.bitpix).into(),
            });
        }
        let bsize = self.bitpix.byte_size();
        let mut out = Vec::with_capacity(self.n_elements as usize);
        for chunk in self.data.chunks_exact(bsize) {
            out.push(T::from_be_bytes(chunk));
        }
        ImageData::new(out, self.axes.clone())
    }

    /// Decode the array into the native pixel type indicated by
    /// `BITPIX`, returning a [`ImagePixels`] enum so callers don't
    /// need to know the type at compile time. No scaling is applied.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use fitsy::{FitsError, FitsFile, Hdu, ImagePixels};
    ///
    /// let f = FitsFile::open("image.fits")?;
    /// let Hdu::Image(img) = f.hdu(0)? else {
    ///     return Err(FitsError::Header("HDU 0 is not an image".into()));
    /// };
    /// match img.read_raw_dyn()? {
    ///     ImagePixels::I16(d) => println!("i16, {} pixels", d.as_slice().len()),
    ///     ImagePixels::F32(d) => println!("f32, {} pixels", d.as_slice().len()),
    ///     other => println!("other dtype: {other:?}"),
    /// }
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    pub fn read_raw_dyn(&self) -> Result<ImagePixels> {
        Ok(match self.bitpix {
            Bitpix::U8 => ImagePixels::U8(self.read_raw::<u8>()?),
            Bitpix::I16 => ImagePixels::I16(self.read_raw::<i16>()?),
            Bitpix::I32 => ImagePixels::I32(self.read_raw::<i32>()?),
            Bitpix::I64 => ImagePixels::I64(self.read_raw::<i64>()?),
            Bitpix::F32 => ImagePixels::F32(self.read_raw::<f32>()?),
            Bitpix::F64 => ImagePixels::F64(self.read_raw::<f64>()?),
        })
    }

    /// Decode the array into `f64` and apply `BZERO`/`BSCALE` and
    /// `BLANK` per Sec.4.4.2.4-Sec.4.4.2.5.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use fitsy::{FitsError, FitsFile, Hdu};
    ///
    /// let f = FitsFile::open("image.fits")?;
    /// let Hdu::Image(img) = f.hdu(0)? else {
    ///     return Err(FitsError::Header("HDU 0 is not an image".into()));
    /// };
    /// let pixels = img.read_physical()?;
    /// assert_eq!(pixels.as_slice().len() as u64, img.axes().iter().product());
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    pub fn read_physical(&self) -> Result<ImageData<f64>> {
        let scaling = Scaling {
            bzero: self.header.bzero(),
            bscale: self.header.bscale(),
            blank: self.header.blank(),
        };
        let bsize = self.bitpix.byte_size();
        let mut out = Vec::with_capacity(self.n_elements as usize);
        for chunk in self.data.chunks_exact(bsize) {
            let v = match self.bitpix {
                Bitpix::U8 => scaling.apply_int(i64::from(<u8 as Pixel>::from_be_bytes(chunk))),
                Bitpix::I16 => scaling.apply_int(i64::from(<i16 as Pixel>::from_be_bytes(chunk))),
                Bitpix::I32 => scaling.apply_int(i64::from(<i32 as Pixel>::from_be_bytes(chunk))),
                Bitpix::I64 => scaling.apply_int(<i64 as Pixel>::from_be_bytes(chunk)),
                Bitpix::F32 => scaling.apply_real(f64::from(<f32 as Pixel>::from_be_bytes(chunk))),
                Bitpix::F64 => scaling.apply_real(<f64 as Pixel>::from_be_bytes(chunk)),
            };
            out.push(v);
        }
        ImageData::new(out, self.axes.clone())
    }

    /// Like [`Self::read_physical`] but returns `f32` instead of
    /// `f64`. Use this when memory is the constraint and the loss of
    /// precision is acceptable (e.g. visualization, single-precision
    /// downstream pipelines). Scaling is performed in `f64` and
    /// truncated only on the final store.
    pub fn read_physical_f32(&self) -> Result<ImageData<f32>> {
        let scaling = Scaling {
            bzero: self.header.bzero(),
            bscale: self.header.bscale(),
            blank: self.header.blank(),
        };
        let bsize = self.bitpix.byte_size();
        let mut out = Vec::with_capacity(self.n_elements as usize);
        for chunk in self.data.chunks_exact(bsize) {
            let v = match self.bitpix {
                Bitpix::U8 => scaling.apply_int(i64::from(<u8 as Pixel>::from_be_bytes(chunk))),
                Bitpix::I16 => scaling.apply_int(i64::from(<i16 as Pixel>::from_be_bytes(chunk))),
                Bitpix::I32 => scaling.apply_int(i64::from(<i32 as Pixel>::from_be_bytes(chunk))),
                Bitpix::I64 => scaling.apply_int(<i64 as Pixel>::from_be_bytes(chunk)),
                Bitpix::F32 => scaling.apply_real(f64::from(<f32 as Pixel>::from_be_bytes(chunk))),
                Bitpix::F64 => scaling.apply_real(<f64 as Pixel>::from_be_bytes(chunk)),
            };
            #[allow(
                clippy::cast_possible_truncation,
                reason = "documented precision loss is the point of this method"
            )]
            out.push(v as f32);
        }
        ImageData::new(out, self.axes.clone())
    }

    /// Number of bytes the data section occupies once padded to a
    /// 2880-byte block boundary.
    #[must_use]
    pub fn padded_data_size(&self) -> u64 {
        pad_to_block(self.n_elements * self.bitpix.byte_size() as u64)
    }

    /// Read a rectangular sub-array.
    ///
    /// `start` and `shape` are in FITS axis order -- element 0 is the
    /// `NAXIS1` (fastest-varying) axis. Both must have length
    /// `NAXIS`. The returned [`ImageData`] has the requested `shape`.
    /// Each row of the requested region is copied with a single byte
    /// range read; sub-pixel sub-axes are walked recursively, so the
    /// I/O is still bounded by the volume of the requested cuboid
    /// (not the full image).
    pub fn read_subarray<T: Pixel>(&self, start: &[u64], shape: &[u64]) -> Result<ImageData<T>> {
        if T::BITPIX != self.bitpix {
            return Err(FitsError::HduMismatch {
                expected: bitpix_name(T::BITPIX),
                found: bitpix_name(self.bitpix).into(),
            });
        }
        if start.len() != self.axes.len() || shape.len() != self.axes.len() {
            return Err(FitsError::Data(format!(
                "read_subarray: start/shape have length {}/{}, expected NAXIS = {}",
                start.len(),
                shape.len(),
                self.axes.len()
            )));
        }
        for (i, (&s, &n)) in start.iter().zip(shape.iter()).enumerate() {
            let axis = self.axes[i];
            if s.checked_add(n).is_none_or(|end| end > axis) {
                return Err(FitsError::Data(format!(
                    "read_subarray: axis {} (NAXIS{}) range {s}..{} out of bounds (length {axis})",
                    i,
                    i + 1,
                    s + n
                )));
            }
        }
        if shape.contains(&0) {
            return ImageData::new(Vec::new(), shape.to_vec());
        }
        let bsize = self.bitpix.byte_size();
        let total: usize = shape.iter().copied().product::<u64>() as usize;
        let mut out: Vec<T> = Vec::with_capacity(total);

        // Strides in elements for each axis (FITS order, NAXIS1 first).
        let mut strides: Vec<u64> = Vec::with_capacity(self.axes.len());
        let mut s = 1_u64;
        for &a in &self.axes {
            strides.push(s);
            s = s.saturating_mul(a);
        }

        let n1 = shape[0];
        let row_bytes = (n1 as usize) * bsize;

        // Recursively iterate axes 1..NAXIS, copying contiguous rows
        // of length n1 along axis 0.
        let mut idx = vec![0_u64; self.axes.len()];
        let outer_axes = self.axes.len();
        loop {
            // Axis-0 contribution to the flat element offset.
            let mut elem_off: u64 = start[0];
            for (ax, &io) in idx.iter().enumerate().skip(1) {
                elem_off += (start[ax] + io) * strides[ax];
            }
            let byte_off = (elem_off as usize) * bsize;
            let chunk = &self.data[byte_off..byte_off + row_bytes];
            for el in chunk.chunks_exact(bsize) {
                out.push(T::from_be_bytes(el));
            }
            if outer_axes == 1 {
                break;
            }
            // Increment the outer-axes counter (axis 1 is fastest in
            // the iteration; that matches contiguous order in memory).
            let mut ax = 1;
            loop {
                idx[ax] += 1;
                if idx[ax] < shape[ax] {
                    break;
                }
                idx[ax] = 0;
                ax += 1;
                if ax == outer_axes {
                    return ImageData::new(out, shape.to_vec());
                }
            }
        }
        ImageData::new(out, shape.to_vec())
    }

    /// Parse the WCS for the given alternate (`b' '` for the primary).
    pub fn wcs(&self, alt: char) -> Result<Option<crate::wcs::Wcs>> {
        crate::wcs::Wcs::from_header(&self.header, alt)
    }
}

fn bitpix_name(b: Bitpix) -> &'static str {
    match b {
        Bitpix::U8 => "u8",
        Bitpix::I16 => "i16",
        Bitpix::I32 => "i32",
        Bitpix::I64 => "i64",
        Bitpix::F32 => "f32",
        Bitpix::F64 => "f64",
    }
}

/// Pixels decoded into the native FITS dtype indicated by `BITPIX`.
///
/// Returned by [`ImageHdu::read_raw_dyn`] so callers can dispatch on
/// the pixel type without knowing it at compile time.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ImagePixels {
    U8(ImageData<u8>),
    I16(ImageData<i16>),
    I32(ImageData<i32>),
    I64(ImageData<i64>),
    F32(ImageData<f32>),
    F64(ImageData<f64>),
}

impl ImagePixels {
    /// The shape (axes in `NAXISn` order, fastest-first).
    #[must_use]
    pub fn axes(&self) -> &[u64] {
        match self {
            Self::U8(d) => d.axes(),
            Self::I16(d) => d.axes(),
            Self::I32(d) => d.axes(),
            Self::I64(d) => d.axes(),
            Self::F32(d) => d.axes(),
            Self::F64(d) => d.axes(),
        }
    }

    /// `BITPIX` for these pixels.
    #[must_use]
    pub fn bitpix(&self) -> Bitpix {
        match self {
            Self::U8(_) => Bitpix::U8,
            Self::I16(_) => Bitpix::I16,
            Self::I32(_) => Bitpix::I32,
            Self::I64(_) => Bitpix::I64,
            Self::F32(_) => Bitpix::F32,
            Self::F64(_) => Bitpix::F64,
        }
    }
}
