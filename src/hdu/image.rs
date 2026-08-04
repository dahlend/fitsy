//! Image HDUs (Standard Sec.7.1, Sec.3.3.1).
//!
//! # Purpose
//!
//! [`ImageHdu`] holds one image: its header, and a borrowed view of
//! its raw data bytes. It decodes those bytes on demand.
//!
//! # Layout
//!
//! Four read methods differ in the type they return and in whether
//! they scale the values:
//!
//! - [`ImageHdu::read_physical`] returns `f64` and applies `BZERO`,
//!   `BSCALE` and `BLANK`. This is the usual choice.
//! - [`ImageHdu::read_physical_f32`] does the same and returns `f32`.
//! - [`ImageHdu::read_raw`] returns the native type of `BITPIX` and
//!   applies no scaling. The caller names that type.
//! - [`ImageHdu::read_raw_dyn`] returns the native type inside the
//!   [`ImagePixels`] enum, so the caller does not name it.
//!
//! [`ImageHdu::read_subarray`] reads a rectangular region instead of
//! the whole array.
//!
//! # Design constraints
//!
//! An [`ImageHdu`] borrows its data bytes from the [`FitsFile`] that
//! produced it, so it cannot outlive that file.
//!
//! Scaling always runs in `f64`, including for
//! [`ImageHdu::read_physical_f32`]. Only the final store narrows the
//! value. This keeps the arithmetic identical between the two.
//!
//! [`FitsFile`]: crate::FitsFile

use crate::data::encoding::{Bitpix, ImageData, Pixel};
use crate::data::scaling::Scaling;
use crate::error::{FitsError, Result};
use crate::header::Header;
use crate::io::block::pad_to_block;

/// One image HDU.
///
/// This borrows its data section from the
/// [`FitsFile`](crate::FitsFile) that produced it, so it cannot
/// outlive that file. It reads `BITPIX` and the axis lengths from the
/// header at construction, and decodes pixels only when asked.
///
/// [`read_physical`](Self::read_physical) is the usual decoder. The
/// module documentation compares it with the other three.
///
/// # Examples
///
/// ```
/// # use fitsy::{FitsWriter, ImageBuilder};
/// # let (h, d) = ImageBuilder::new(vec![4_u64, 3], vec![7_i16; 12])?
/// #     .primary(true)
/// #     .build()?;
/// # let mut buf: Vec<u8> = Vec::new();
/// # FitsWriter::new(&mut buf).write_hdu(&h, &d)?;
/// use fitsy::{Bitpix, FitsFile, Hdu};
///
/// let file = FitsFile::from_bytes(buf)?;
/// let Hdu::Image(img) = file.hdu(0)? else {
///     panic!("HDU 0 is not an image");
/// };
///
/// assert_eq!(img.bitpix(), Bitpix::I16);
/// assert_eq!(img.n_elements(), 12);
/// assert_eq!(img.read_physical()?.as_slice()[0], 7.0);
/// # Ok::<(), fitsy::FitsError>(())
/// ```
#[derive(Debug, Clone)]
pub struct ImageHdu<'a> {
    header: Header,
    data: &'a [u8],
    bitpix: Bitpix,
    axes: Vec<u64>,
    n_elements: u64,
}

impl<'a> ImageHdu<'a> {
    /// Construct from a parsed header and the raw data section.
    ///
    /// The `data` slice must cover the data section without its
    /// trailing block padding.
    ///
    /// # Errors
    ///
    /// - [`FitsError::MissingMandatory`] when the header omits
    ///   `BITPIX` or `NAXIS`.
    /// - [`FitsError::Value`] when `BITPIX` holds a value outside the
    ///   six that Standard Sec.4.4.1.1 defines.
    /// - [`FitsError::Data`] when the pixel count or the byte count
    ///   overflows `u64`, or when `data.len()` does not equal the size
    ///   that the header declares.
    pub fn new(header: Header, data: &'a [u8]) -> Result<Self> {
        let bitpix = Bitpix::from_i64(header.bitpix()?)?;
        let axes = header.axes()?;
        let n_elements: u64 = if axes.is_empty() || axes.contains(&0) {
            0
        } else {
            axes.iter()
                .try_fold(1_u64, |acc, &a| acc.checked_mul(a))
                .ok_or_else(|| FitsError::Data("image pixel count overflows u64".into()))?
        };
        let needed = n_elements
            .checked_mul(bitpix.byte_size() as u64)
            .ok_or_else(|| FitsError::Data("image data size overflows u64".into()))?;
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
    /// The HDU's header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    #[must_use]
    /// Pixel encoding, from `BITPIX`.
    pub fn bitpix(&self) -> Bitpix {
        self.bitpix
    }

    #[must_use]
    /// `NAXISn` in FITS order, fastest-varying axis first.
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

    /// Decode the array into native primitives, with no scaling.
    ///
    /// The element type `T` must match the `BITPIX` of the HDU. This
    /// applies neither `BZERO` nor `BSCALE`, so the values are the
    /// ones the file stores.
    ///
    /// # Errors
    ///
    /// - [`FitsError::HduMismatch`] when `T` does not match `BITPIX`.
    /// - [`FitsError::Data`] when the decoded element count does not
    ///   match the axis product.
    pub fn read_raw<T: Pixel>(&self) -> Result<ImageData<T>> {
        if T::BITPIX != self.bitpix {
            return Err(FitsError::HduMismatch {
                expected: T::BITPIX.rust_type_name(),
                found: self.bitpix.rust_type_name().into(),
            });
        }
        let bsize = self.bitpix.byte_size();
        // `collect` from the `ExactSizeIterator` rather than pushing in a
        // loop: the per-push capacity check blocks vectorization and costs
        // roughly half the decode throughput.
        let out: Vec<T> = self
            .data
            .chunks_exact(bsize)
            .map(T::from_be_bytes)
            .collect();
        ImageData::new(out, self.axes.clone())
    }

    /// Decode into the native `BITPIX` type, wrapped in
    /// [`ImagePixels`].
    ///
    /// The caller dispatches on the returned variant instead of naming
    /// the element type at compile time. This applies no scaling.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when the decoded element count does not
    /// match the axis product.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fitsy::{FitsWriter, ImageBuilder};
    /// # let path = std::env::temp_dir().join("fitsy_doc_raw_dyn.fits");
    /// # let (h, d) = ImageBuilder::new(vec![4_u64, 3], vec![7_i16; 12])?
    /// #     .primary(true)
    /// #     .build()?;
    /// # let mut out = std::fs::File::create(&path)?;
    /// # FitsWriter::new(&mut out).write_hdu(&h, &d)?;
    /// use fitsy::{FitsError, FitsFile, Hdu, ImagePixels};
    ///
    /// let f = FitsFile::open(&path)?;
    /// let Hdu::Image(img) = f.hdu(0)? else {
    ///     return Err(FitsError::Header("HDU 0 is not an image".into()));
    /// };
    /// match img.read_raw_dyn()? {
    ///     ImagePixels::I16(d) => assert_eq!(d.as_slice().len(), 12),
    ///     other => panic!("expected i16 pixels, found {other:?}"),
    /// }
    /// # std::fs::remove_file(&path)?;
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

    /// Decode the array into `f64` in physical units.
    ///
    /// Each pixel becomes `BZERO + BSCALE * raw`, per Standard
    /// Sec.4.4.2.5. An integer image also maps each pixel equal to
    /// `BLANK` to `NaN`, per Sec.4.4.2.4. A float image carries no
    /// `BLANK` card, and its undefined pixels are already `NaN`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when the decoded element count does not
    /// match the axis product.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fitsy::{FitsWriter, ImageBuilder};
    /// # let path = std::env::temp_dir().join("fitsy_doc_physical.fits");
    /// # let (h, d) = ImageBuilder::new(vec![4_u64, 3], vec![2.5_f32; 12])?
    /// #     .primary(true)
    /// #     .build()?;
    /// # let mut out = std::fs::File::create(&path)?;
    /// # FitsWriter::new(&mut out).write_hdu(&h, &d)?;
    /// use fitsy::{FitsError, FitsFile, Hdu};
    ///
    /// let f = FitsFile::open(&path)?;
    /// let Hdu::Image(img) = f.hdu(0)? else {
    ///     return Err(FitsError::Header("HDU 0 is not an image".into()));
    /// };
    /// let pixels = img.read_physical()?;
    ///
    /// assert_eq!(
    ///     pixels.as_slice().len() as u64,
    ///     img.axes().iter().product::<u64>()
    /// );
    /// # std::fs::remove_file(&path)?;
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    pub fn read_physical(&self) -> Result<ImageData<f64>> {
        let scaling = Scaling {
            bzero: self.header.bzero(),
            bscale: self.header.bscale(),
            blank: self.header.blank(),
        };
        let out = self.scaled_pixels(&scaling, |v| v);
        ImageData::new(out, self.axes.clone())
    }

    /// Decode every pixel with `BZERO`/`BSCALE`/`BLANK` applied, passing
    /// each through `cast` on the way out.
    ///
    /// The `BITPIX` match is hoisted out of the loop so each arm is a
    /// single monomorphic pass that can vectorize, and `cast` is applied
    /// during the collect so a narrower output never has to be staged
    /// through a full-size `f64` buffer first.
    fn scaled_pixels<T>(&self, scaling: &Scaling, cast: impl Fn(f64) -> T) -> Vec<T> {
        match self.bitpix {
            Bitpix::U8 => self
                .data
                .iter()
                .map(|&b| cast(scaling.apply_int(i64::from(b))))
                .collect(),
            Bitpix::I16 => self
                .data
                .chunks_exact(2)
                .map(|c| cast(scaling.apply_int(i64::from(<i16 as Pixel>::from_be_bytes(c)))))
                .collect(),
            Bitpix::I32 => self
                .data
                .chunks_exact(4)
                .map(|c| cast(scaling.apply_int(i64::from(<i32 as Pixel>::from_be_bytes(c)))))
                .collect(),
            Bitpix::I64 => self
                .data
                .chunks_exact(8)
                .map(|c| cast(scaling.apply_int(<i64 as Pixel>::from_be_bytes(c))))
                .collect(),
            Bitpix::F32 => self
                .data
                .chunks_exact(4)
                .map(|c| cast(scaling.apply_real(f64::from(<f32 as Pixel>::from_be_bytes(c)))))
                .collect(),
            Bitpix::F64 => self
                .data
                .chunks_exact(8)
                .map(|c| cast(scaling.apply_real(<f64 as Pixel>::from_be_bytes(c))))
                .collect(),
        }
    }

    /// Decode the array into `f32` in physical units.
    ///
    /// This applies the same scaling as [`Self::read_physical`], and
    /// runs that arithmetic in `f64`. Only the final store narrows the
    /// value to `f32`. Use this method when memory is the constraint
    /// and the loss of precision is acceptable.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when the decoded element count does not
    /// match the axis product.
    pub fn read_physical_f32(&self) -> Result<ImageData<f32>> {
        let scaling = Scaling {
            bzero: self.header.bzero(),
            bscale: self.header.bscale(),
            blank: self.header.blank(),
        };
        #[allow(
            clippy::cast_possible_truncation,
            reason = "documented precision loss is the point of this method"
        )]
        let out = self.scaled_pixels(&scaling, |v| v as f32);
        ImageData::new(out, self.axes.clone())
    }

    /// Number of bytes the data section occupies once padded to a
    /// 2880-byte block boundary.
    #[must_use]
    pub fn padded_data_size(&self) -> u64 {
        pad_to_block(self.n_elements * self.bitpix.byte_size() as u64)
    }

    /// Read a rectangular sub-array, with no scaling.
    ///
    /// The `start` and `shape` arguments are in FITS axis order, where
    /// element 0 is `NAXIS1`, the fastest-varying axis. Both must have
    /// length `NAXIS`. The result carries the requested `shape`.
    ///
    /// This copies one contiguous row at a time, so the cost follows
    /// the requested region rather than the whole image.
    ///
    /// # Errors
    ///
    /// - [`FitsError::HduMismatch`] when `T` does not match `BITPIX`.
    /// - [`FitsError::Data`] when `start` or `shape` has the wrong
    ///   length, when the region escapes the array, or when an axis
    ///   stride overflows `u64`.
    pub fn read_subarray<T: Pixel>(&self, start: &[u64], shape: &[u64]) -> Result<ImageData<T>> {
        use crate::hdu::subarray::{checked_strides, next_subarray_index, validate_subarray_shape};

        if T::BITPIX != self.bitpix {
            return Err(FitsError::HduMismatch {
                expected: T::BITPIX.rust_type_name(),
                found: self.bitpix.rust_type_name().into(),
            });
        }
        validate_subarray_shape(&self.axes, start, shape)?;
        if shape.contains(&0) {
            return ImageData::new(Vec::new(), shape.to_vec());
        }
        let bsize = self.bitpix.byte_size();
        let total: usize = shape.iter().copied().product::<u64>() as usize;
        let mut out: Vec<T> = Vec::with_capacity(total);

        let strides = checked_strides(&self.axes)?;

        let n1 = shape[0];
        let row_bytes = (n1 as usize) * bsize;

        // Recursively iterate axes 1..NAXIS, copying contiguous rows
        // of length n1 along axis 0.
        let mut idx = vec![0_u64; self.axes.len()];
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
            if !next_subarray_index(&mut idx, shape) {
                break;
            }
        }
        ImageData::new(out, shape.to_vec())
    }

    /// Parse the WCS of this HDU for alternate descriptor `alt`.
    ///
    /// Pass `' '` for `alt` to select the primary description. The
    /// result is `Ok(None)` when the header carries no WCS for that
    /// descriptor.
    ///
    /// This function reads the header of this HDU alone. It resolves
    /// no `-TAB` lookup extension, because that needs the whole file.
    /// Call [`FitsFile::wcs`](crate::FitsFile::wcs) for that.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the header declares a WCS that the
    /// parser rejects, such as an unknown projection code.
    pub fn wcs(&self, alt: char) -> Result<Option<crate::wcs::Wcs>> {
        crate::wcs::Wcs::from_header(&self.header, alt)
    }
}

/// Pixels decoded into the native FITS dtype indicated by `BITPIX`.
///
/// Returned by [`ImageHdu::read_raw_dyn`] so callers can dispatch on
/// the pixel type without knowing it at compile time.
#[derive(Debug, Clone)]
pub enum ImagePixels {
    /// `BITPIX = 8`.
    U8(ImageData<u8>),
    /// `BITPIX = 16`.
    I16(ImageData<i16>),
    /// `BITPIX = 32`.
    I32(ImageData<i32>),
    /// `BITPIX = 64`.
    I64(ImageData<i64>),
    /// `BITPIX = -32`.
    F32(ImageData<f32>),
    /// `BITPIX = -64`.
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
