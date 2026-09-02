//! Compressed-input support (cargo feature `compression`).
//!
//! # Purpose
//!
//! This module handles two distinct conventions.
//!
//! First, whole-file gzip, in a `*.gz` file. The FITS file was
//! compressed with `gzip(1)`. [`maybe_gunzip`] inflates such a buffer
//! back to the underlying FITS bytes, and leaves any other buffer
//! unchanged.
//!
//! Second, FITS tile-compressed images, in a `*.fz` file (Pence &
//! Seaman 2010, Standard Sec.10). The image lives in a `BINTABLE` that
//! carries `ZIMAGE = T`, the `Z` geometry keywords, and a
//! variable-length `COMPRESSED_DATA` column holding one tile per row.
//! A tile may instead fall back to `GZIP_COMPRESSED_DATA` or to raw
//! `UNCOMPRESSED_DATA`.
//!
//! # Layout
//!
//! [`CompressedImageHdu`] wraps such a table.
//! [`CompressedImageHdu::as_image`] decodes it into an [`OwnedImage`].
//! [`compress_image_to_hdu`] runs the write path.
//!
//! Each submodule holds one codec: `rice`, `hcompress`, `plio` and
//! `quantize`. The gzip codecs use the `flate2` crate directly.
//!
//! # Design constraints
//!
//! The read path decodes each `ZCMPTYPE` of Standard Sec.10 Table 10.
//! Those are `GZIP_1`, `GZIP_2`, `RICE_1`, `PLIO_1` and `HCOMPRESS_1`
//! for 8-, 16-, 32- and 64-bit integer images, plus `NOCOMPRESS`,
//! whose tiles hold the pixel bytes verbatim. A float image decodes
//! through `NO_DITHER`, `SUBTRACTIVE_DITHER_1` or
//! `SUBTRACTIVE_DITHER_2` quantization, and losslessly through
//! `GZIP_1`, `GZIP_2` or `NOCOMPRESS`.
//!
//! The write path emits `GZIP_1`, `GZIP_2` or `RICE_1` tiles,
//! chosen through [`TileOpts`]. It writes a float image losslessly
//! by default. [`Quantize`] turns on lossy quantization, which
//! `RICE_1` requires before it can compress a float image.

mod hcompress;
mod plio;
mod quantize;
mod rice;

use std::io::Read;

use flate2::read::GzDecoder;

use crate::data::encoding::Bitpix;
use crate::error::{FitsError, Result};
use crate::hdu::bintable::{BinColumn, BinFieldKind, BinTableHdu, BinValue};
use crate::header::Header;
use crate::header::card::is_indexed;
use crate::header::value::Value;

pub use self::quantize::DitherMethod;
use self::quantize::NULL_VALUE;

/// gzip RFC 1952 magic bytes.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Inflate `buf` when it starts with the gzip magic bytes. Return it
/// unchanged otherwise.
///
/// # Errors
///
/// [`FitsError::Io`] when `buf` starts with the magic bytes but fails
/// to inflate as a gzip stream.
pub fn maybe_gunzip(buf: Vec<u8>) -> Result<Vec<u8>> {
    if buf.len() < 2 || buf[..2] != GZIP_MAGIC {
        return Ok(buf);
    }
    let mut out = Vec::with_capacity(buf.len() * 4);
    GzDecoder::new(buf.as_slice())
        .read_to_end(&mut out)
        .map_err(FitsError::Io)?;
    Ok(out)
}

/// A view over a tile-compressed image HDU.
///
/// This wraps the `BINTABLE` that carries the tiles and reads the `Z`
/// geometry keywords from its header. It borrows the data section from
/// the [`FitsFile`](crate::FitsFile) that produced it, so it cannot
/// outlive that file, and it decodes no tile until asked.
///
/// Call [`as_image`](Self::as_image) for a decoded [`OwnedImage`], or
/// [`synthetic_image_header`](Self::synthetic_image_header) for the
/// header alone.
#[derive(Debug, Clone)]
pub struct CompressedImageHdu<'a> {
    inner: BinTableHdu<'a>,
    /// Original (uncompressed) image `BITPIX`.
    bitpix: Bitpix,
    axes: Vec<u64>,
    tile: Vec<u64>,
    cmptype: CmpType,
    /// Bytes per pixel in the *decompressed tile buffer* -- always
    /// 4 for quantized float images (i32), otherwise the same as
    /// `bitpix.byte_size()`.
    internal_bp: usize,
    /// Quantization metadata when ZBITPIX < 0 and ZQUANTIZ is
    /// `NO_DITHER` / `SUBTRACTIVE_DITHER_*`. `None` for integer
    /// images and for lossless float compression (`ZQUANTIZ=NONE`).
    quantize: Option<QuantizeInfo>,
}

#[derive(Debug, Clone)]
struct QuantizeInfo {
    dither: DitherMethod,
    /// Per-tile seed offset from `ZDITHER0`. This defaults to 1.
    dither_seed: u32,
    /// Integer sentinel for NaN/Inf source pixels.
    blank: BlankSource,
    scale: ScaleSource,
    zero: ScaleSource,
}

/// Source of a per-tile scaling parameter (`ZSCALE` or `ZZERO`).
#[derive(Debug, Clone)]
enum ScaleSource {
    /// Constant value from a header keyword.
    Constant(f64),
    /// Slot index into [`BinTableHdu::columns`], one scalar per row.
    Column(usize),
}

/// Source of the `ZBLANK` undefined-pixel sentinel (Standard
/// Sec.10.2.2).
///
/// The sentinel may vary from tile to tile, so the Standard allows a
/// `ZBLANK` column beside the `ZBLANK` keyword. Sec.10.2.2 requires
/// the reader to use the column when a file carries both.
#[derive(Debug, Clone)]
enum BlankSource {
    /// Constant sentinel from the `ZBLANK` keyword. This holds
    /// [`NULL_VALUE`] when the file names no sentinel.
    Constant(i32),
    /// Slot index into [`BinTableHdu::columns`], one sentinel per row.
    Column(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CmpType {
    Gzip1,
    Gzip2,
    /// Pence et al. 2010 Sec.3.1. `blocksize` defaults to 32.
    Rice1 {
        blocksize: u32,
    },
    /// IRAF pixel-list run-length code (Sec.10.4 / Pence 2009 Appendix).
    Plio1,
    /// R. White's H-transform image compression
    /// (FITS standard Sec.10.4.5; only 16-/32-bit integer images).
    Hcompress1 {
        scale: i32,
        smooth: bool,
    },
    /// `NOCOMPRESS` -- tiles hold the pixel bytes verbatim (Table 10).
    /// Lossless, so legal for float images too.
    None,
}

impl<'a> CompressedImageHdu<'a> {
    /// Wrap a [`BinTableHdu`] whose header carries `ZIMAGE = T`.
    ///
    /// # Errors
    ///
    /// - [`FitsError::HduMismatch`] when the header does not carry
    ///   `ZIMAGE = T`.
    /// - [`FitsError::MissingMandatory`] when a mandatory `Z` keyword
    ///   is absent, such as `ZBITPIX`, `ZNAXIS` or `ZCMPTYPE`.
    /// - [`FitsError::Value`] when `ZBITPIX` holds an illegal value,
    ///   or when `ZCMPTYPE` names a compression type outside Standard
    ///   Sec.10 Table 10.
    /// - [`FitsError::Data`] when the tile geometry does not divide
    ///   the image, or when a required column is absent.
    pub fn from_bintable(inner: BinTableHdu<'a>) -> Result<Self> {
        let h = inner.header();
        if !matches!(h.first("ZIMAGE"), Some(Value::Logical(true))) {
            return Err(FitsError::HduMismatch {
                expected: "compressed image (ZIMAGE = T)",
                found: "BINTABLE without ZIMAGE".into(),
            });
        }
        let bitpix = Bitpix::from_i64(h.required_int("ZBITPIX")?)?;
        let znaxis = h.required_int("ZNAXIS")?;
        if !(0..=999).contains(&znaxis) {
            return Err(FitsError::Value {
                keyword: "ZNAXIS".into(),
                msg: format!("ZNAXIS={znaxis} out of range"),
            });
        }
        let mut axes = Vec::with_capacity(znaxis as usize);
        let mut tile = Vec::with_capacity(znaxis as usize);
        for i in 1..=znaxis {
            let n = h.required_int(&format!("ZNAXIS{i}"))?;
            if n < 0 {
                return Err(FitsError::Value {
                    keyword: format!("ZNAXIS{i}"),
                    msg: "must be >= 0".into(),
                });
            }
            axes.push(n as u64);
            // Default tile size (Pence & Seaman Sec.3): full first axis,
            // 1 along all other axes.
            let default = if i == 1 { n as u64 } else { 1 };
            let t = h
                .optional_int(&format!("ZTILE{i}"))
                .map(|v| {
                    if v < 0 {
                        Err(FitsError::Value {
                            keyword: format!("ZTILE{i}"),
                            msg: "must be >= 0".into(),
                        })
                    } else {
                        Ok(v as u64)
                    }
                })
                .transpose()?
                .unwrap_or(default);
            tile.push(t);
        }
        let cmptype_s =
            h.optional_string("ZCMPTYPE")
                .ok_or_else(|| FitsError::MissingMandatory {
                    keyword: "ZCMPTYPE".into(),
                })?;
        let cmptype_s = cmptype_s.trim();

        // Parse ZQUANTIZ first -- cmptype validation depends on the
        // *internal* pixel type (which is i32 for quantized floats).
        // Per FITS Standard Sec.10.4.3.4, when ZQUANTIZ is absent for a
        // float image, NO_DITHER is assumed (matches CFITSIO).
        let zquantiz = h.optional_string("ZQUANTIZ").map(|s| s.trim().to_owned());
        let is_float = matches!(bitpix, Bitpix::F32 | Bitpix::F64);
        let quant_method = match zquantiz.as_deref() {
            None => {
                if is_float {
                    Some(DitherMethod::NoDither)
                } else {
                    None
                }
            }
            Some("" | "NONE") => None,
            Some("NO_DITHER") => Some(DitherMethod::NoDither),
            Some("SUBTRACTIVE_DITHER_1") => Some(DitherMethod::Subtractive1),
            Some("SUBTRACTIVE_DITHER_2") => Some(DitherMethod::Subtractive2),
            Some(other) => {
                return Err(FitsError::NonStandard(format!(
                    "ZQUANTIZ=`{other}` not supported (expected NONE, NO_DITHER, \
                     SUBTRACTIVE_DITHER_1 or SUBTRACTIVE_DITHER_2)"
                )));
            }
        };
        if !is_float && quant_method.is_some() {
            return Err(FitsError::Header(format!(
                "ZQUANTIZ requires a floating-point ZBITPIX, got {}",
                bitpix.as_i64()
            )));
        }
        let quantize = if let Some(method) = quant_method {
            let scale = lookup_scale_source(h, &inner, "ZSCALE")?;
            let zero = lookup_scale_source(h, &inner, "ZZERO")?;
            let blank = lookup_blank_source(h, &inner)?;
            // cfitsio defaults the dither seed to 1 when ZDITHER0 is absent.
            let dither_seed = h
                .optional_int("ZDITHER0")
                .filter(|v| *v >= 0)
                .map_or(1, |v| v as u32);
            Some(QuantizeInfo {
                dither: method,
                dither_seed,
                blank,
                scale,
                zero,
            })
        } else {
            None
        };
        // Effective bitpix for cmptype validation: i32 when we'll be
        // decompressing into a 32-bit quantized integer buffer.
        let inner_bitpix = if quantize.is_some() {
            Bitpix::I32
        } else {
            bitpix
        };
        let internal_bp = inner_bitpix.byte_size();
        // Lossless float: only GZIP_1 / GZIP_2 are defined -- plus
        // NOCOMPRESS, which stores the IEEE bytes untouched and so
        // cannot lose anything.
        if is_float
            && quantize.is_none()
            && !matches!(cmptype_s, "GZIP_1" | "GZIP_2" | "NOCOMPRESS")
        {
            return Err(FitsError::NonStandard(format!(
                "lossless float compression requires ZCMPTYPE=GZIP_1, GZIP_2 or \
                 NOCOMPRESS, got {cmptype_s}"
            )));
        }

        let cmptype = match cmptype_s {
            "GZIP_1" => CmpType::Gzip1,
            "GZIP_2" => CmpType::Gzip2,
            "RICE_1" | "RICE_ONE" => {
                let blocksize = parse_rice_blocksize(h, inner_bitpix)?;
                CmpType::Rice1 { blocksize }
            }
            "PLIO_1" => {
                if !matches!(inner_bitpix, Bitpix::U8 | Bitpix::I16 | Bitpix::I32) {
                    return Err(FitsError::NonStandard(format!(
                        "PLIO_1 only supports 8/16/32-bit integer images, got effective \
                         BITPIX={}",
                        inner_bitpix.as_i64()
                    )));
                }
                CmpType::Plio1
            }
            "HCOMPRESS_1" => {
                if !matches!(
                    inner_bitpix,
                    Bitpix::U8 | Bitpix::I16 | Bitpix::I32 | Bitpix::I64
                ) {
                    return Err(FitsError::NonStandard(format!(
                        "HCOMPRESS_1 supports 8/16/32/64-bit integer images \
                         (or quantized floats); got effective BITPIX={}",
                        inner_bitpix.as_i64()
                    )));
                }
                if znaxis != 2 {
                    return Err(FitsError::NonStandard(format!(
                        "HCOMPRESS_1 requires a 2-D image, got ZNAXIS={znaxis}"
                    )));
                }
                let (scale, smooth) = parse_hcompress_params(h)?;
                CmpType::Hcompress1 { scale, smooth }
            }
            "NOCOMPRESS" => CmpType::None,
            other => {
                return Err(FitsError::NonStandard(format!(
                    "tile compression algorithm `{other}` is not supported (this build \
                     supports GZIP_1, GZIP_2, RICE_1, PLIO_1, HCOMPRESS_1 and NOCOMPRESS)"
                )));
            }
        };
        Ok(Self {
            inner,
            bitpix,
            axes,
            tile,
            cmptype,
            internal_bp,
            quantize,
        })
    }

    #[must_use]
    /// The HDU's header.
    pub fn header(&self) -> &Header {
        self.inner.header()
    }
    /// Borrow the underlying BINTABLE view (the on-disk
    /// representation of this compressed image HDU).
    #[must_use]
    pub fn as_bintable(&self) -> &BinTableHdu<'a> {
        &self.inner
    }
    /// Whether the original image was a primary array, from
    /// `ZSIMPLE` (Sec.10.2).
    ///
    /// A compressed image is a `BINTABLE`, so it sits in an extension
    /// slot. [`as_image`](Self::as_image) therefore yields an `IMAGE`
    /// extension header for every compressed HDU. This function
    /// reports the slot the image came from. A tool that rebuilds the
    /// original file layout needs it, as `fitsy funpack` does.
    #[must_use]
    pub fn was_primary(&self) -> bool {
        matches!(self.header().first("ZSIMPLE"), Some(Value::Logical(true)))
    }

    #[must_use]
    /// Pixel encoding of the *decompressed* image, from `ZBITPIX`.
    pub fn bitpix(&self) -> Bitpix {
        self.bitpix
    }
    /// Original image dimensions (`ZNAXISn`, fastest-varying first).
    #[must_use]
    /// `ZNAXISn` in FITS order, fastest-varying axis first.
    pub fn axes(&self) -> &[u64] {
        &self.axes
    }
    #[must_use]
    /// `ZTILEn` in FITS order, fastest-varying axis first.
    pub fn tile_shape(&self) -> &[u64] {
        &self.tile
    }

    /// Build a synthetic image-HDU header from the `Z` keywords.
    ///
    /// A caller then reads the WCS, the `BUNIT` and the rest through
    /// the same accessors an uncompressed image uses. This inverts the
    /// convention of Sec.10.4, rewriting each `Z` keyword to its image
    /// equivalent: `ZBITPIX` to `BITPIX`, `ZNAXISn` to `NAXISn`,
    /// `ZCTYPEn` to `CTYPEn`, and so on.
    ///
    /// # Errors
    ///
    /// [`FitsError::Header`] when a rewritten keyword is not a legal
    /// FITS keyword. [`FitsError::Value`] when a `Z` keyword holds a
    /// value of the wrong type.
    pub fn synthetic_image_header(&self) -> Result<Header> {
        synthesize_image_header(self.inner.header())
    }

    /// Decompress every tile and wrap the result as an
    /// [`OwnedImage`], which owns its pixel bytes.
    ///
    /// # Errors
    ///
    /// The conditions of [`Self::decompress`] and of
    /// [`Self::synthetic_image_header`].
    pub fn as_image(&self) -> Result<OwnedImage> {
        let bytes = self.decompress()?;
        let header = self.synthetic_image_header()?;
        OwnedImage::new(header, bytes)
    }

    /// Decompress every tile into one big-endian byte buffer, laid
    /// out as an uncompressed image HDU would be.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when a tile fails to decode, when a tile
    /// decodes to the wrong length, when the pixel count overflows
    /// `usize`, or when a quantized float tile carries no scale.
    /// [`FitsError::Io`] when a gzip tile fails to inflate.
    pub fn decompress(&self) -> Result<Vec<u8>> {
        let out_bp = self.bitpix.byte_size();
        let inner_bp = self.internal_bp;
        let n_pix: u64 = if self.axes.is_empty() {
            0
        } else {
            self.axes.iter().product()
        };
        let total = (n_pix as usize)
            .checked_mul(out_bp)
            .ok_or_else(|| FitsError::Data("decompressed size overflows usize".into()))?;
        let mut out = vec![0_u8; total];
        if total == 0 {
            return Ok(out);
        }

        let cols = TileColumns::from(&self.inner)?;
        let heap = self.inner.heap_bytes();
        let n_tiles = self.inner.n_rows();
        let expected = expected_tile_count(&self.axes, &self.tile);
        if n_tiles != expected {
            return Err(FitsError::Header(format!(
                "compressed image: {n_tiles} table rows, expected {expected} tiles \
                 from ZNAXIS/ZTILE"
            )));
        }

        let mut tile_buf = Vec::<u8>::new();
        let mut float_buf = Vec::<u8>::new();
        let mut indices = vec![0_u64; self.axes.len()];
        let mut extent = vec![0_u64; self.axes.len()];
        for row in 0..n_tiles {
            tile_index_from_row(row as u64, &self.axes, &self.tile, &mut indices);
            for i in 0..self.axes.len() {
                let t = effective_tile(self.axes[i], self.tile[i]);
                let origin = indices[i] * t;
                extent[i] = t.min(self.axes[i].saturating_sub(origin));
            }
            let tile_pixels: u64 = extent.iter().product();
            let tile_bytes_outer = (tile_pixels as usize) * out_bp;

            // Per Sec.10.4.1.3 the fallback columns (UNCOMPRESSED_DATA,
            // GZIP_COMPRESSED_DATA) carry pixels in the *original*
            // image format, never the quantized integer form. So for
            // those payloads we decode straight into `out_bp`-sized
            // bytes and skip dequantization entirely.
            let payload = cols.payload_for_row(&self.inner, heap, row)?;
            let payload_is_fallback = matches!(
                payload,
                TilePayload::Uncompressed(_) | TilePayload::GzipFallback(_)
            );
            let decode_bp = if payload_is_fallback || self.quantize.is_none() {
                out_bp
            } else {
                inner_bp
            };
            tile_buf.clear();
            tile_buf.resize((tile_pixels as usize) * decode_bp, 0);
            decompress_tile(
                payload,
                &mut tile_buf,
                self.cmptype,
                decode_bp,
                tile_pixels as u32,
            )?;

            let scattered: &[u8] = if let Some(q) = &self.quantize
                && !payload_is_fallback
            {
                // Quantized primary payload: convert i32 -> f32/f64.
                float_buf.clear();
                float_buf.resize(tile_bytes_outer, 0);
                let scale = q.scale.fetch(&self.inner, row)?;
                let zero = q.zero.fetch(&self.inner, row)?;
                let blank = q.blank.fetch(&self.inner, row)?;
                // cfitsio: seed = ZDITHER0 + (row + 1) - 1 = ZDITHER0 + row,
                // with `row` 0-based here (DitherWalker applies the final -1).
                let dither_seed = u64::from(q.dither_seed) + (row as u64);
                let dither_arg = match q.dither {
                    DitherMethod::NoDither => None,
                    other => Some((other, dither_seed)),
                };
                match self.bitpix {
                    Bitpix::F32 => quantize::unquantize_to_f32_be(
                        &tile_buf,
                        &mut float_buf,
                        scale,
                        zero,
                        blank,
                        dither_arg,
                    ),
                    Bitpix::F64 => quantize::unquantize_to_f64_be(
                        &tile_buf,
                        &mut float_buf,
                        scale,
                        zero,
                        blank,
                        dither_arg,
                    ),
                    _ => unreachable!("quantize is only set for float images"),
                }
                &float_buf
            } else {
                &tile_buf
            };

            scatter_tile(
                scattered, out_bp, &self.axes, &self.tile, &extent, &indices, &mut out,
            )?;
        }
        Ok(out)
    }
}

/// A decompressed image returned from
/// [`CompressedImageHdu::as_image`]. Owns its byte buffer.
#[derive(Debug, Clone)]
pub struct OwnedImage {
    header: Header,
    bytes: Vec<u8>,
    bitpix: Bitpix,
    axes: Vec<u64>,
}

impl OwnedImage {
    fn new(header: Header, bytes: Vec<u8>) -> Result<Self> {
        let bitpix = Bitpix::from_i64(header.bitpix()?)?;
        let axes = header.axes()?;
        Ok(Self {
            header,
            bytes,
            bitpix,
            axes,
        })
    }
    #[must_use]
    /// The HDU's header.
    pub fn header(&self) -> &Header {
        &self.header
    }
    #[must_use]
    /// Pixel encoding of the *decompressed* image, from `ZBITPIX`.
    pub fn bitpix(&self) -> Bitpix {
        self.bitpix
    }
    #[must_use]
    /// `ZNAXISn` in FITS order, fastest-varying axis first.
    pub fn axes(&self) -> &[u64] {
        &self.axes
    }
    /// Big-endian raw pixel bytes (`NAXISn` product * |BITPIX|/8).
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Read the WCS of this image for alternate descriptor `alt`.
    ///
    /// This function reads the recovered image header alone. It
    /// resolves no `-TAB` lookup extension, because that needs the
    /// whole file. Call [`FitsFile::wcs`](crate::FitsFile::wcs) for
    /// that.
    ///
    /// # Errors
    ///
    /// The conditions of
    /// [`Wcs::from_header`](crate::wcs::Wcs::from_header).
    pub fn wcs(&self, alt: char) -> Result<Option<crate::wcs::Wcs>> {
        crate::wcs::Wcs::from_header(&self.header, alt)
    }
}

// -- Tile dispatch --------------------------------------------------

/// Per-tile payload, classified by which BINTABLE column we read
/// from. `Compressed` is decoded with the HDU's ZCMPTYPE algorithm;
/// the other two are standard fallbacks (Sec.10.4.1).
#[derive(Debug, Clone, Copy)]
enum TilePayload<'a> {
    Compressed(&'a [u8]),
    /// `GZIP_COMPRESSED_DATA` -- always `GZIP_1` regardless of ZCMPTYPE.
    GzipFallback(&'a [u8]),
    /// `UNCOMPRESSED_DATA` -- raw big-endian pixels, no compression.
    Uncompressed(&'a [u8]),
}

/// References to the optional fallback columns, looked up once.
struct TileColumns<'a> {
    primary: &'a BinColumn,
    uncompressed: Option<&'a BinColumn>,
    gzip_fallback: Option<&'a BinColumn>,
}

impl<'a> TileColumns<'a> {
    fn from(bt: &'a BinTableHdu<'_>) -> Result<Self> {
        let primary = bt.column_by_name("COMPRESSED_DATA").ok_or_else(|| {
            FitsError::Header("compressed image: COMPRESSED_DATA column missing".into())
        })?;
        if !matches!(primary.format.kind, BinFieldKind::P | BinFieldKind::Q) {
            return Err(FitsError::Value {
                keyword: format!("TFORM{}", primary.index),
                msg: "COMPRESSED_DATA must be a P/Q variable-length array".into(),
            });
        }
        Ok(Self {
            primary,
            uncompressed: bt.column_by_name("UNCOMPRESSED_DATA"),
            gzip_fallback: bt.column_by_name("GZIP_COMPRESSED_DATA"),
        })
    }

    fn payload_for_row<'r>(
        &self,
        bt: &'r BinTableHdu<'_>,
        heap: &'r [u8],
        row: usize,
    ) -> Result<TilePayload<'r>> {
        let raw = bt.cell_bytes(row, self.primary)?;
        if let Some((off, len)) = heap_span(self.primary, raw)? {
            return Ok(TilePayload::Compressed(slice_heap(heap, off, len)?));
        }
        if let Some(fc) = self.uncompressed {
            let r = bt.cell_bytes(row, fc)?;
            if let Some((off, len)) = heap_span(fc, r)? {
                return Ok(TilePayload::Uncompressed(slice_heap(heap, off, len)?));
            }
        }
        if let Some(gc) = self.gzip_fallback {
            let r = bt.cell_bytes(row, gc)?;
            if let Some((off, len)) = heap_span(gc, r)? {
                return Ok(TilePayload::GzipFallback(slice_heap(heap, off, len)?));
            }
        }
        Err(FitsError::Data(format!(
            "compressed image: tile {row} has no payload in COMPRESSED_DATA, \
             UNCOMPRESSED_DATA or GZIP_COMPRESSED_DATA"
        )))
    }
}

/// Resolve a `P`/`Q` descriptor cell to a `(heap_offset, byte_len)`
/// span. Standard Sec.7.3.5: the descriptor's first field is the number
/// of *array elements*; convert to bytes via the column's inner VLA
/// element type. Returns `None` for an empty (zero-element) array.
fn heap_span(col: &BinColumn, raw: &[u8]) -> Result<Option<(usize, usize)>> {
    let (n, off) = parse_descriptor(col.format.kind, raw)?;
    if n == 0 {
        return Ok(None);
    }
    let elt = col.format.vla_kind.ok_or_else(|| {
        FitsError::Header(format!("TFORM{} is not a P/Q VLA descriptor", col.index))
    })?;
    let bytes = if matches!(elt, BinFieldKind::Bit) {
        // X within a VLA is a bit-packed array of `n` bits.
        n.div_ceil(8)
    } else {
        n.checked_mul(elt.element_bytes()).ok_or_else(|| {
            FitsError::Data(format!(
                "VLA byte length overflows usize (n={n}, elt={})",
                elt.element_bytes()
            ))
        })?
    };
    Ok(Some((off, bytes)))
}

fn parse_descriptor(kind: BinFieldKind, raw: &[u8]) -> Result<(usize, usize)> {
    crate::hdu::bintable::parse_vla_descriptor(kind, raw)
}

fn slice_heap(heap: &[u8], off: usize, n: usize) -> Result<&[u8]> {
    let end = off
        .checked_add(n)
        .ok_or_else(|| FitsError::Data("VLA descriptor overflows heap address".into()))?;
    if end > heap.len() {
        return Err(FitsError::Data(format!(
            "VLA descriptor [{off}, {end}) escapes heap (len={})",
            heap.len()
        )));
    }
    Ok(&heap[off..end])
}

fn decompress_tile(
    payload: TilePayload<'_>,
    out: &mut [u8],
    cmptype: CmpType,
    bp: usize,
    tile_pixels: u32,
) -> Result<()> {
    match payload {
        TilePayload::Uncompressed(bytes) => copy_tile(bytes, out, "UNCOMPRESSED_DATA"),
        TilePayload::GzipFallback(bytes) => inflate_into(bytes, out),
        TilePayload::Compressed(bytes) => match cmptype {
            // The tile holds the pixel bytes as they would appear in an
            // ordinary image: FITS byte order, no transform to undo.
            CmpType::None => copy_tile(bytes, out, "NOCOMPRESS"),
            CmpType::Gzip1 => inflate_into(bytes, out),
            CmpType::Gzip2 => {
                inflate_into(bytes, out)?;
                if bp > 1 {
                    unshuffle(out, bp)?;
                }
                Ok(())
            }
            CmpType::Rice1 { blocksize } => {
                rice::decompress_into(bp as u32, blocksize, tile_pixels, bytes, out)
            }
            CmpType::Plio1 => plio::decompress_into(bytes, bp, tile_pixels as usize, out),
            CmpType::Hcompress1 { scale, smooth } => hcompress::decompress_into(
                bytes,
                bp,
                tile_pixels as usize,
                hcompress::HcompressParams { scale, smooth },
                out,
            ),
        },
    }
}

/// A tile stored verbatim: the payload must be exactly the pixel bytes.
fn copy_tile(bytes: &[u8], out: &mut [u8], what: &str) -> Result<()> {
    if bytes.len() != out.len() {
        return Err(FitsError::Data(format!(
            "{what} tile is {} bytes, expected {}",
            bytes.len(),
            out.len()
        )));
    }
    out.copy_from_slice(bytes);
    Ok(())
}

/// Pence & Seaman 2010 Sec.3.3: `HCOMPRESS_1` carries `SCALE`
/// (integer quantization) and `SMOOTH` (0/1) via `ZNAMEn`/`ZVALn`.
/// `SCALE = 0` (or absent) means "trust the value embedded in the
/// payload".
fn parse_hcompress_params(h: &Header) -> Result<(i32, bool)> {
    let mut scale: i32 = 0;
    let mut smooth = false;
    for i in 1..=999_u32 {
        let Some(name) = h.optional_string(&format!("ZNAME{i}")) else {
            break;
        };
        match name.trim() {
            "SCALE" => {
                let v = h.required_int(&format!("ZVAL{i}"))?;
                if v < 0 || v > i64::from(i32::MAX) {
                    return Err(FitsError::Value {
                        keyword: format!("ZVAL{i}"),
                        msg: format!("HCOMPRESS SCALE out of range: {v}"),
                    });
                }
                scale = v as i32;
            }
            "SMOOTH" => {
                let v = h.required_int(&format!("ZVAL{i}"))?;
                smooth = v != 0;
            }
            // Forward-compat: ignore unknown parameters.
            _ => {}
        }
    }
    Ok((scale, smooth))
}

/// Resolve the `ZBLANK` sentinel: column first, then header keyword,
/// then the fallback.
///
/// Standard Sec.10.2.2 requires the reader to use the column when a
/// file carries both forms. A file may omit both. That states the
/// image holds no undefined pixel, and the fallback is then
/// [`NULL_VALUE`], the sentinel the dequantizer compares against.
fn lookup_blank_source(h: &Header, bt: &BinTableHdu<'_>) -> Result<BlankSource> {
    if let Some((slot, col)) = bt
        .columns()
        .iter()
        .enumerate()
        .find(|(_, c)| c.name.eq_ignore_ascii_case("ZBLANK"))
    {
        if !matches!(
            col.format.kind,
            BinFieldKind::Byte | BinFieldKind::I16 | BinFieldKind::I32 | BinFieldKind::I64
        ) {
            return Err(FitsError::Value {
                keyword: format!("TFORM(col {})", col.index),
                msg: "ZBLANK column must be a scalar integer".into(),
            });
        }
        return Ok(BlankSource::Column(slot));
    }
    Ok(BlankSource::Constant(
        h.optional_int("ZBLANK").map_or(NULL_VALUE, |v| v as i32),
    ))
}

impl BlankSource {
    /// Look up the sentinel for `row` (0-based).
    ///
    /// A null cell states that the tile holds no undefined pixel.
    /// That case returns [`NULL_VALUE`].
    fn fetch(&self, bt: &BinTableHdu<'_>, row: usize) -> Result<i32> {
        match self {
            Self::Constant(v) => Ok(*v),
            Self::Column(slot) => {
                let col = bt.columns().get(*slot).ok_or_else(|| {
                    FitsError::Header(format!("ZBLANK column slot {slot} no longer present"))
                })?;
                match bt.cell_value(row, col)? {
                    BinValue::Int(v) if v.len() == 1 => Ok(v[0].map_or(NULL_VALUE, |x| x as i32)),
                    other => Err(FitsError::Data(format!(
                        "expected scalar integer in ZBLANK column, got {other:?}"
                    ))),
                }
            }
        }
    }
}

/// Resolve a `ZSCALE` / `ZZERO` parameter: column first, then
/// header keyword. Pence & Seaman 2010 Sec.4.1 says one or the other
/// must be present for a quantized image.
fn lookup_scale_source(h: &Header, bt: &BinTableHdu<'_>, name: &str) -> Result<ScaleSource> {
    if let Some((slot, col)) = bt
        .columns()
        .iter()
        .enumerate()
        .find(|(_, c)| c.name.eq_ignore_ascii_case(name))
    {
        if !matches!(
            col.format.kind,
            BinFieldKind::F32 | BinFieldKind::F64 | BinFieldKind::I32 | BinFieldKind::I64
        ) {
            return Err(FitsError::Value {
                keyword: format!("TFORM(col {})", col.index),
                msg: format!("{name} column must be a scalar float/integer"),
            });
        }
        return Ok(ScaleSource::Column(slot));
    }
    if let Some(v) = h.optional_real(name) {
        return Ok(ScaleSource::Constant(v));
    }
    Err(FitsError::MissingMandatory {
        keyword: name.into(),
    })
}

impl ScaleSource {
    /// Look up the per-tile value for `row` (0-based).
    fn fetch(&self, bt: &BinTableHdu<'_>, row: usize) -> Result<f64> {
        match self {
            Self::Constant(v) => Ok(*v),
            Self::Column(slot) => {
                let col = bt.columns().get(*slot).ok_or_else(|| {
                    FitsError::Header(format!("scale column slot {slot} no longer present"))
                })?;
                match bt.cell_value(row, col)? {
                    BinValue::F32(v) if v.len() == 1 => Ok(f64::from(v[0])),
                    BinValue::F64(v) | BinValue::Float(v) if v.len() == 1 => Ok(v[0]),
                    BinValue::Int(v) if v.len() == 1 => Ok(v[0].map_or(0.0, |x| x as f64)),
                    other => Err(FitsError::Data(format!(
                        "expected scalar float in column {slot}, got {other:?}"
                    ))),
                }
            }
        }
    }
}

fn inflate_into(payload: &[u8], dst: &mut [u8]) -> Result<()> {
    let mut tmp = Vec::with_capacity(dst.len());
    GzDecoder::new(payload)
        .read_to_end(&mut tmp)
        .map_err(|e| FitsError::Data(format!("tile gzip inflate failed: {e}")))?;
    if tmp.len() != dst.len() {
        return Err(FitsError::Data(format!(
            "gzip tile inflated to {} bytes, expected {}",
            tmp.len(),
            dst.len()
        )));
    }
    dst.copy_from_slice(&tmp);
    Ok(())
}

/// Invert the `GZIP_2` byte-shuffle. Pixels were split into per-byte
/// planes (all most-significant bytes first, then next-most, ...)
/// before gzipping; restore big-endian pixel order.
fn unshuffle(buf: &mut [u8], bpp: usize) -> Result<()> {
    if !buf.len().is_multiple_of(bpp) {
        return Err(FitsError::Data(format!(
            "GZIP_2 tile size {} is not a multiple of {bpp}",
            buf.len()
        )));
    }
    let n = buf.len() / bpp;
    let mut tmp = vec![0_u8; buf.len()];
    for plane in 0..bpp {
        let plane_off = plane * n;
        for px in 0..n {
            tmp[px * bpp + plane] = buf[plane_off + px];
        }
    }
    buf.copy_from_slice(&tmp);
    Ok(())
}

/// Split `buf` into `bpp` byte planes, most significant plane first.
/// The write-side inverse of [`unshuffle`], for `GZIP_2` tiles.
fn shuffle(buf: &[u8], bpp: usize) -> Vec<u8> {
    let n = buf.len() / bpp;
    let mut out = vec![0_u8; buf.len()];
    for (i, chunk) in buf.chunks_exact(bpp).enumerate() {
        for (plane, &b) in chunk.iter().enumerate() {
            out[plane * n + i] = b;
        }
    }
    out
}

/// Walk the `ZNAMEn`/`ZVALn` pairs to extract the Rice block size.
/// Pence et al. 2010 Sec.3.1: `BLOCKSIZE` defaults to 32. `BYTEPIX`
/// must equal `|inner_bitpix|/8`; for quantized floats the inner
/// type is i32 so `BYTEPIX = 4` is correct.
fn parse_rice_blocksize(h: &Header, inner_bitpix: Bitpix) -> Result<u32> {
    if !matches!(inner_bitpix, Bitpix::U8 | Bitpix::I16 | Bitpix::I32) {
        return Err(FitsError::NonStandard(format!(
            "RICE_1 requires an effective integer pixel size of 1/2/4 bytes; \
             got effective BITPIX={}",
            inner_bitpix.as_i64()
        )));
    }
    let mut blocksize: u32 = 32;
    for i in 1..=999_u32 {
        let Some(name) = h.optional_string(&format!("ZNAME{i}")) else {
            break;
        };
        match name.trim() {
            "BLOCKSIZE" => {
                let v = h.required_int(&format!("ZVAL{i}"))?;
                if v <= 0 {
                    return Err(FitsError::Value {
                        keyword: format!("ZVAL{i}"),
                        msg: format!("RICE BLOCKSIZE must be > 0, got {v}"),
                    });
                }
                blocksize = v as u32;
            }
            "BYTEPIX" => {
                let v = h.required_int(&format!("ZVAL{i}"))?;
                let want = inner_bitpix.byte_size() as i64;
                if v != want {
                    return Err(FitsError::NonStandard(format!(
                        "RICE BYTEPIX={v} does not match effective |BITPIX|/8={want}"
                    )));
                }
            }
            // Ignore unknown parameters (forward-compat).
            _ => {}
        }
    }
    Ok(blocksize)
}

fn effective_tile(axis: u64, tile: u64) -> u64 {
    if tile == 0 { axis } else { tile }
}

fn expected_tile_count(axes: &[u64], tile: &[u64]) -> usize {
    let mut n: usize = 1;
    for (a, t) in axes.iter().zip(tile.iter()) {
        let t = effective_tile(*a, *t);
        if t == 0 {
            return 0;
        }
        n = n.saturating_mul(a.div_ceil(t) as usize);
    }
    n
}

/// Tile rows are stored in row-major order over the *tile grid*,
/// fastest-varying axis first. Decode `row` into per-axis tile
/// coordinates.
fn tile_index_from_row(row: u64, axes: &[u64], tile: &[u64], out: &mut [u64]) {
    let mut r = row;
    for i in 0..axes.len() {
        let t = effective_tile(axes[i], tile[i]);
        let n_along = if t == 0 { 1 } else { axes[i].div_ceil(t) };
        out[i] = r % n_along;
        r /= n_along;
    }
}

/// Copy a flat tile buffer into the right strided slot of the full
/// image buffer.
fn scatter_tile(
    tile_data: &[u8],
    bpp: usize,
    axes: &[u64],
    tile_full: &[u64],
    extent: &[u64],
    tile_idx: &[u64],
    out: &mut [u8],
) -> Result<()> {
    let ndim = axes.len();
    if ndim == 0 {
        return Ok(());
    }
    let tile_pixels: u64 = extent.iter().product();
    let needed = (tile_pixels as usize)
        .checked_mul(bpp)
        .ok_or_else(|| FitsError::Data("tile size overflows usize".into()))?;
    if tile_data.len() < needed {
        return Err(FitsError::Data(format!(
            "decompressed tile too short: {} bytes, need {needed}",
            tile_data.len()
        )));
    }
    // Per-axis origin of this tile in image coordinates, computed from
    // the un-clipped tile size (extent may be smaller at edges).
    let mut origin = vec![0_u64; ndim];
    for i in 0..ndim {
        let t = effective_tile(axes[i], tile_full[i]);
        origin[i] = tile_idx[i] * t;
    }
    // Image strides (bytes), fastest-varying axis first.
    let mut img_stride = vec![0_u64; ndim];
    let mut s: u64 = bpp as u64;
    for i in 0..ndim {
        img_stride[i] = s;
        s = s.saturating_mul(axes[i]);
    }
    let row_bytes = (extent[0] as usize) * bpp;
    let mut coord = vec![0_u64; ndim];
    let mut src = 0_usize;
    loop {
        let mut dst: u64 = 0;
        for i in 0..ndim {
            dst += (origin[i] + coord[i]) * img_stride[i];
        }
        let dst = dst as usize;
        out[dst..dst + row_bytes].copy_from_slice(&tile_data[src..src + row_bytes]);
        src += row_bytes;
        let mut carry = true;
        for i in 1..ndim {
            coord[i] += 1;
            if coord[i] < extent[i] {
                carry = false;
                break;
            }
            coord[i] = 0;
        }
        if carry {
            break;
        }
    }
    Ok(())
}

/// Append one tile of the full image buffer to `out`, row by row.
/// The write-side inverse of [`scatter_tile`].
///
/// The caller validates the geometry. `img` spans `axes` at `bpp`
/// bytes per pixel. `extent` fits inside the image at the position
/// `tile_idx` names.
///
/// # Panics
///
/// Panics when the geometry runs past the end of `img`.
fn gather_tile(
    img: &[u8],
    bpp: usize,
    axes: &[u64],
    tile_full: &[u64],
    extent: &[u64],
    tile_idx: &[u64],
    out: &mut Vec<u8>,
) {
    let ndim = axes.len();
    if ndim == 0 {
        return;
    }
    let mut origin = vec![0_u64; ndim];
    for i in 0..ndim {
        origin[i] = tile_idx[i] * effective_tile(axes[i], tile_full[i]);
    }
    // Image strides (bytes), fastest-varying axis first.
    let mut img_stride = vec![0_u64; ndim];
    let mut s: u64 = bpp as u64;
    for i in 0..ndim {
        img_stride[i] = s;
        s = s.saturating_mul(axes[i]);
    }
    let row_bytes = (extent[0] as usize) * bpp;
    let mut coord = vec![0_u64; ndim];
    loop {
        let mut src: u64 = 0;
        for i in 0..ndim {
            src += (origin[i] + coord[i]) * img_stride[i];
        }
        let src = src as usize;
        out.extend_from_slice(&img[src..src + row_bytes]);
        let mut carry = true;
        for i in 1..ndim {
            coord[i] += 1;
            if coord[i] < extent[i] {
                carry = false;
                break;
            }
            coord[i] = 0;
        }
        if carry {
            break;
        }
    }
}

// -- Synthetic image header ----------------------------------------

/// Indexed table keywords the compressed `BINTABLE` reserves for its
/// own columns, so an image card of the same name cannot be carried.
///
/// This is the set cfitsio and astropy reserve, the table-WCS forms
/// included, so a header compressed by any of the three loses the same
/// cards and no others.
const T_PREFIXES: &[&str] = &[
    "TTYPE", "TFORM", "TUNIT", "TDIM", "TSCAL", "TZERO", "TNULL", "TDISP", "TBCOL", "TCTYP",
    "TCUNI", "TCRPX", "TCRVL", "TCDLT", "TRPOS",
];

/// Indexed compression keywords reserved by Sec.10.2.
const Z_INDEXED: &[&str] = &["ZNAME", "ZVAL", "ZTILE", "ZNAXIS"];

/// Map a compressed-BINTABLE keyword to its synthetic-IMAGE form, or
/// `None` to drop it.
///
/// Sec.10.2 reserves a fixed set of `Z*` names and copies every other
/// image keyword verbatim -- there is no "strip the leading Z" rule, so
/// `ZP` and `ZD` pass through. The reserved names are either re-emitted
/// by `synthesize_image_header` or bookkeeping, so this only drops them.
fn z_to_image_keyword(k: &str) -> Option<String> {
    // Checksums cover the *compressed* bytes; ZHECKSUM/ZDATASUM cover a
    // pre-compression image that lossy tiles need not reproduce. Neither
    // survives -- callers recompute (`FitsWriter::with_checksums`).
    // The table's own structure and its checksums; the synthetic image
    // header re-emits what it needs from validated values.
    if crate::header::is_writer_owned_keyword(k) {
        return None;
    }
    // The same reserved set the write side applies, so the two
    // directions cannot come to disagree about what the convention
    // owns.
    if is_reserved_by_compression(k) {
        return None;
    }
    Some(k.to_string())
}

/// Map a source-image keyword to its compressed-table form, or `None`
/// to drop it. The write-side inverse of [`z_to_image_keyword`].
///
/// Sec.10.2 renames the structural keywords to `Z` forms. It copies
/// every other card verbatim. [`build_zimage_header`] re-emits the
/// dropped names from validated values. The rest are checksums, which
/// the writer recomputes.
fn image_to_z_keyword(k: &str) -> Option<&str> {
    match k {
        "SIMPLE" => return Some("ZSIMPLE"),
        "XTENSION" => return Some("ZTENSION"),
        "EXTEND" => return Some("ZEXTEND"),
        "BLOCKED" => return Some("ZBLOCKED"),
        "PCOUNT" => return Some("ZPCOUNT"),
        "GCOUNT" => return Some("ZGCOUNT"),
        "BLANK" => return Some("ZBLANK"),
        _ => {}
    }
    // The image's own structure and its checksums; `build_zimage_header`
    // re-emits the structural names in their `Z` forms above, and the
    // writer recomputes the checksums over the compressed bytes.
    if crate::header::is_writer_owned_keyword(k) {
        return None;
    }
    if is_reserved_by_compression(k) {
        return None;
    }
    Some(k)
}

/// True when the tiled-image convention reserves `k` for the
/// compressed `BINTABLE` itself, so an image card of that name cannot
/// be carried into it.
///
/// This is the same set cfitsio and astropy reserve. A card it names
/// is dropped by all three, which is what
/// [`reserved_keywords`] reports so a caller can be told.
#[must_use]
pub fn is_reserved_by_compression(k: &str) -> bool {
    const RESERVED: &[&str] = &[
        "TFIELDS", "THEAP", "ZIMAGE", "ZCMPTYPE", "ZQUANTIZ", "ZDITHER0", "ZMASKCMP", "ZSCALE",
        "ZZERO", "ZBLANK", "ZBITPIX", "ZNAXIS", "ZSIMPLE", "ZTENSION", "ZEXTEND", "ZBLOCKED",
        "ZPCOUNT", "ZGCOUNT", "ZHECKSUM", "ZDATASUM",
    ];
    RESERVED.contains(&k) || is_indexed(k, Z_INDEXED) || is_indexed(k, T_PREFIXES)
}

/// The cards of `header` that compressing it would not carry.
///
/// The tiled-image convention gives the compressed `BINTABLE` these
/// keywords for its own use, so an image card of the same name has
/// nowhere to go. cfitsio refuses such a header outright and astropy
/// drops the card with a warning; fitsy drops it, and this is what
/// lets a caller say so.
///
/// Structural cards are not listed: they are not lost but rewritten,
/// `BITPIX` as `ZBITPIX` and so on.
#[must_use]
pub fn reserved_keywords(header: &Header) -> Vec<String> {
    let mut out: Vec<String> = header
        .cards()
        .filter(|c| !c.is_commentary())
        .map(|c| c.keyword())
        .filter(|k| is_reserved_by_compression(k))
        .collect();
    out.dedup();
    out
}

fn synthesize_image_header(bt: &Header) -> Result<Header> {
    let mut out = Header::empty();
    let bitpix = bt.optional_int("ZBITPIX").unwrap_or(8);
    let znaxis = bt.optional_int("ZNAXIS").unwrap_or(0);
    // The decompressed view describes the HDU where it sits, which is
    // always an IMAGE extension: a compressed image is a BINTABLE, so
    // it can never occupy the primary slot. `ZSIMPLE = T` says the
    // *original* was a primary array, and
    // [`CompressedImageHdu::was_primary`] reports that separately.
    // Emitting `SIMPLE` here instead would hand every caller a header
    // that `FitsWriter::write_hdu` rejects in the slot it came from.
    out.push("XTENSION", Value::String("IMAGE".into()), None)?;
    out.push("BITPIX", Value::Integer(bitpix), None)?;
    out.push("NAXIS", Value::Integer(znaxis), None)?;
    for i in 1..=znaxis {
        if let Some(n) = bt.optional_int(&format!("ZNAXIS{i}")) {
            out.push(format!("NAXIS{i}"), Value::Integer(n), None)?;
        }
    }
    // Sec.7.1: an IMAGE extension always carries PCOUNT = 0 and
    // GCOUNT = 1, whatever ZPCOUNT/ZGCOUNT hold.
    out.push("PCOUNT", Value::Integer(0), None)?;
    out.push("GCOUNT", Value::Integer(1), None)?;

    // ZBLANK (Sec.10.2.4) -> BLANK in the synthetic IMAGE header so the
    // BLANK-aware accessors work after decompression. Only valid for
    // integer images (Sec.4.4.2.2); for quantized float images the
    // sentinel is consumed during dequantization and replaced by NaN.
    if bitpix > 0
        && let Some(blank) = bt.optional_int("ZBLANK")
    {
        out.push("BLANK", Value::Integer(blank), None)?;
    }

    // Map every other Z* and non-Z keyword to its image form. Skip
    // only the names emitted above.
    //
    // The test uses that record, not the contents of `out`. A repeated
    // source keyword is legal, and its second card is not in the
    // record, so this keeps it.
    let emitted: std::collections::HashSet<String> = out.cards().map(|c| c.keyword()).collect();
    for entry in bt.cards() {
        // A commentary card moves whole. COMMENT and HISTORY
        // survive a compression round trip.
        if entry.is_commentary() {
            out.splice(&entry);
            continue;
        }
        let keyword = entry.keyword();
        let Some(mapped) = z_to_image_keyword(&keyword) else {
            continue;
        };
        if emitted.contains(&mapped) {
            continue;
        }
        if mapped == keyword {
            // The name does not change, so the card moves whole and
            // is not re-encoded.
            out.splice(&entry);
        } else if let Some(v) = entry.value() {
            // The name changes, so the card is re-encoded. A value
            // the header cannot hold returns an error here. It is not
            // dropped.
            out.push(mapped, v, entry.comment().as_deref())?;
        }
    }
    Ok(out)
}

/// Tile-compress an image HDU into a `(Header, data)` pair describing
/// a tile-compressed `BINTABLE` (Pence & Seaman 2010, Standard
/// Sec.7.4).
///
/// The `header` and `data` arguments describe the uncompressed image,
/// as [`ImageBuilder`](crate::ImageBuilder) emits them. `data` holds
/// the big-endian byte payload. The structural cards move to their
/// `Z` forms, such as `BITPIX` to `ZBITPIX` and `SIMPLE` to
/// `ZSIMPLE`. Every other card copies into the table header. The
/// image that [`CompressedImageHdu::as_image`] restores therefore
/// keeps its WCS and the rest of its metadata.
///
/// The `opts` argument selects the codec, the tile shape and the
/// `EXTNAME` of the resulting table. Pass `&TileOpts::default()` for
/// `GZIP_1` with the default tile shape.
///
/// # Errors
///
/// - [`FitsError::MissingMandatory`] when `header` omits `BITPIX`,
///   `NAXIS` or a `NAXISn` card.
/// - [`FitsError::Data`] when `data.len()` does not match the size the
///   header implies, when the image has no axes, when `opts.tile` has
///   the wrong length or a zero entry, or when the compressed heap
///   exceeds the `P` descriptor range.
/// - [`FitsError::NonStandard`] when the codec does not apply to the
///   image, such as `RICE_1` on a floating-point or 64-bit image.
/// - [`FitsError::Header`] when a generated keyword is illegal.
/// - [`FitsError::Io`] when a tile fails to compress.
pub fn compress_image_to_hdu(
    header: &Header,
    data: &[u8],
    opts: &TileOpts,
) -> Result<(Header, Vec<u8>)> {
    let bitpix = header.bitpix()?;
    let axes = header.axes()?;
    if axes.is_empty() {
        return Err(FitsError::Data(
            "compress_image_to_hdu: cannot compress a 0-axis image".into(),
        ));
    }
    let bytes_per = (bitpix.unsigned_abs() / 8) as usize;
    let n_pixels: u64 = axes.iter().product();
    if data.len() != bytes_per * n_pixels as usize {
        return Err(FitsError::Data(format!(
            "compress_image_to_hdu: data is {} bytes; expected {} (BITPIX={bitpix}, n_pixels={n_pixels})",
            data.len(),
            bytes_per * n_pixels as usize,
        )));
    }
    if opts.quantize.is_some() && bitpix > 0 {
        return Err(FitsError::NonStandard(format!(
            "quantization applies to floating-point images; got BITPIX={bitpix}"
        )));
    }
    // Quantized tiles hold i32 samples whatever the source BITPIX,
    // which makes a quantized float image eligible for RICE_1.
    let rice_ok = opts.quantize.is_some() || matches!(bitpix, 8 | 16 | 32);
    if matches!(opts.codec, Codec::Rice1 { .. }) && !rice_ok {
        return Err(FitsError::NonStandard(format!(
            "RICE_1 writes integer images of 1, 2 or 4 bytes per pixel; got BITPIX={bitpix} \
             (quantization makes a float image eligible)"
        )));
    }
    let tile: Vec<u64> = if let Some(t) = &opts.tile {
        if t.len() != axes.len() {
            return Err(FitsError::Data(format!(
                "compress_image_to_hdu: tile rank {} does not match NAXIS {}",
                t.len(),
                axes.len()
            )));
        }
        if t.contains(&0) {
            return Err(FitsError::Data(
                "compress_image_to_hdu: tile dimensions must be >= 1".into(),
            ));
        }
        t.clone()
    } else {
        // Default tile shape (Pence & Seaman Sec.3): the full first
        // axis, 1 along every other axis.
        let mut t = vec![1_u64; axes.len()];
        t[0] = axes[0];
        t
    };

    let total_tiles = expected_tile_count(&axes, &tile);
    let mut tiles: Vec<TileOut> = Vec::with_capacity(total_tiles);
    let mut scales: Vec<(f64, f64)> = Vec::new();
    let mut indices = vec![0_u64; axes.len()];
    let mut extent = vec![0_u64; axes.len()];
    let mut tile_buf: Vec<u8> = Vec::new();
    for row in 0..total_tiles {
        tile_index_from_row(row as u64, &axes, &tile, &mut indices);
        for i in 0..axes.len() {
            let t = effective_tile(axes[i], tile[i]);
            extent[i] = t.min(axes[i].saturating_sub(indices[i] * t));
        }
        tile_buf.clear();
        gather_tile(
            data,
            bytes_per,
            &axes,
            &tile,
            &extent,
            &indices,
            &mut tile_buf,
        );
        if let Some(q) = &opts.quantize {
            let (out, scale) = quantize_and_compress(&tile_buf, bitpix, q, row, opts.codec)?;
            tiles.push(out);
            scales.push(scale);
        } else {
            tiles.push(TileOut::Primary(compress_tile(
                &tile_buf, opts.codec, bytes_per,
            )?));
        }
    }

    // One row per tile: a P descriptor per payload column, then the
    // per-tile ZSCALE / ZZERO under quantization. Payloads land in
    // the heap in row order, directly after the rows.
    let any_fallback = tiles.iter().any(|t| matches!(t, TileOut::Fallback(_)));
    let heap_size: usize = tiles.iter().map(|t| t.bytes().len()).sum();
    let max_primary = tiles
        .iter()
        .filter_map(|t| match t {
            TileOut::Primary(b) => Some(b.len()),
            TileOut::Fallback(_) => None,
        })
        .max()
        .unwrap_or(0);
    let max_fallback = tiles
        .iter()
        .filter_map(|t| match t {
            TileOut::Fallback(b) => Some(b.len()),
            TileOut::Primary(_) => None,
        })
        .max()
        .unwrap_or(0);
    if heap_size > i32::MAX as usize {
        return Err(FitsError::Data(format!(
            "compressed heap is {heap_size} bytes, which exceeds the P descriptor \
             range; use smaller tiles or split the image"
        )));
    }
    let quantized = opts.quantize.is_some();
    let row_bytes = 8 + if any_fallback { 8 } else { 0 } + if quantized { 16 } else { 0 };
    let mut out = Vec::with_capacity(tiles.len() * row_bytes + heap_size);
    let mut offset: usize = 0;
    for (row, t) in tiles.iter().enumerate() {
        let descriptor = |len: usize, off: usize| {
            let mut d = [0_u8; 8];
            d[..4].copy_from_slice(&(len as i32).to_be_bytes());
            d[4..].copy_from_slice(&(off as i32).to_be_bytes());
            d
        };
        let (primary, fallback) = match t {
            TileOut::Primary(b) => (descriptor(b.len(), offset), descriptor(0, 0)),
            TileOut::Fallback(b) => (descriptor(0, 0), descriptor(b.len(), offset)),
        };
        out.extend_from_slice(&primary);
        if any_fallback {
            out.extend_from_slice(&fallback);
        }
        if quantized {
            let (scale, zero) = scales[row];
            out.extend_from_slice(&scale.to_be_bytes());
            out.extend_from_slice(&zero.to_be_bytes());
        }
        offset += t.bytes().len();
    }
    for t in &tiles {
        out.extend_from_slice(t.bytes());
    }
    let h = build_zimage_header(
        header,
        bitpix,
        &axes,
        &tile,
        opts,
        total_tiles as u64,
        heap_size,
        TableColumnWidths {
            max_primary,
            max_fallback: any_fallback.then_some(max_fallback),
        },
    )?;
    Ok((h, out))
}

/// One compressed tile, keyed to the column that stores it.
enum TileOut {
    /// `COMPRESSED_DATA` -- the codec's output.
    Primary(Vec<u8>),
    /// `GZIP_COMPRESSED_DATA` -- lossless gzip of the original tile
    /// bytes, for a tile that could not be quantized.
    Fallback(Vec<u8>),
}

impl TileOut {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Primary(b) | Self::Fallback(b) => b,
        }
    }
}

/// Maximum payload length per column, for the `TFORMn` descriptors.
#[derive(Clone, Copy)]
struct TableColumnWidths {
    max_primary: usize,
    /// `Some` when any tile fell back to `GZIP_COMPRESSED_DATA`.
    max_fallback: Option<usize>,
}

/// Quantize one float tile and compress the samples, or fall back to
/// lossless gzip when the tile cannot be quantized.
///
/// The result pairs the payload with the tile's `(ZSCALE, ZZERO)`.
/// A fallback tile reports `(1.0, 0.0)`. The reader ignores that
/// pair, because a fallback tile holds unquantized pixels.
///
/// # Errors
///
/// The conditions of [`compress_tile`].
fn quantize_and_compress(
    tile_be: &[u8],
    bitpix: i64,
    q: &Quantize,
    row: usize,
    codec: Codec,
) -> Result<(TileOut, (f64, f64))> {
    let values: Vec<f64> = if bitpix == -32 {
        tile_be
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f64::from(f32::from_be_bytes(*c)))
            .collect()
    } else {
        tile_be
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_be_bytes(*c))
            .collect()
    };
    // The same per-tile seed rule the decoder applies: ZDITHER0 plus
    // the 0-based row number.
    let dither_arg = match q.dither {
        DitherMethod::NoDither => None,
        other => Some((other, u64::from(q.seed) + row as u64)),
    };
    match quantize::quantize_tile(&values, q.level, dither_arg) {
        Some((ints, scale, zero)) => {
            let be: Vec<u8> = ints.iter().flat_map(|v| v.to_be_bytes()).collect();
            Ok((
                TileOut::Primary(compress_tile(&be, codec, 4)?),
                (scale, zero),
            ))
        }
        None => Ok((
            TileOut::Fallback(compress_tile(tile_be, Codec::Gzip1, 1)?),
            (1.0, 0.0),
        )),
    }
}

/// Compress one tile's big-endian pixel bytes with `codec`.
///
/// # Errors
///
/// - [`FitsError::Io`] when a gzip stream fails to write.
/// - The conditions of [`rice::compress`] for [`Codec::Rice1`];
///   `compress_image_to_hdu` rejects an illegal pixel width up front.
fn compress_tile(tile_be: &[u8], codec: Codec, bytes_per: usize) -> Result<Vec<u8>> {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;
    let gzip = |bytes: &[u8]| -> Result<Vec<u8>> {
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(bytes).map_err(FitsError::Io)?;
        e.finish().map_err(FitsError::Io)
    };
    match codec {
        Codec::Gzip1 => gzip(tile_be),
        Codec::Gzip2 => gzip(&shuffle(tile_be, bytes_per)),
        Codec::Rice1 { blocksize } => rice::compress(bytes_per as u32, blocksize, tile_be),
    }
}

/// Build the complete BINTABLE header for a tile-compressed image:
/// the table structure, the `Z` geometry cards, then every card of
/// `src` carried through [`image_to_z_keyword`].
///
/// # Errors
///
/// [`FitsError::Header`] when a generated keyword is illegal.
#[allow(
    clippy::too_many_arguments,
    reason = "all parameters are required to build the tile-compressed FITS extension header"
)]
fn build_zimage_header(
    src: &Header,
    bitpix: i64,
    axes: &[u64],
    tile: &[u64],
    opts: &TileOpts,
    n_tiles: u64,
    heap_size: usize,
    widths: TableColumnWidths,
) -> Result<Header> {
    let quantized = opts.quantize.is_some();
    // Column layout: COMPRESSED_DATA, then GZIP_COMPRESSED_DATA when
    // any tile fell back, then the per-tile ZSCALE / ZZERO scalars.
    let n_fields = 1 + i64::from(widths.max_fallback.is_some()) + if quantized { 2 } else { 0 };
    let row_bytes =
        8 + if widths.max_fallback.is_some() { 8 } else { 0 } + if quantized { 16 } else { 0 };
    let mut h = Header::empty();
    h.push("XTENSION", Value::String("BINTABLE".into()), None)?;
    h.push("BITPIX", Value::Integer(8), None)?;
    h.push("NAXIS", Value::Integer(2), None)?;
    h.push(
        "NAXIS1",
        Value::Integer(row_bytes),
        Some("bytes per table row"),
    )?;
    h.push(
        "NAXIS2",
        Value::Integer(n_tiles as i64),
        Some("number of tiles"),
    )?;
    h.push("PCOUNT", Value::Integer(heap_size as i64), None)?;
    h.push("GCOUNT", Value::Integer(1), None)?;
    h.push("TFIELDS", Value::Integer(n_fields), None)?;
    let mut field = 0;
    let mut push_field = |h: &mut Header, name: &str, form: String| -> Result<()> {
        field += 1;
        h.push(format!("TTYPE{field}"), Value::String(name.into()), None)?;
        h.push(format!("TFORM{field}"), Value::String(form), None)?;
        Ok(())
    };
    push_field(
        &mut h,
        "COMPRESSED_DATA",
        format!("1PB({})", widths.max_primary),
    )?;
    if let Some(max_fallback) = widths.max_fallback {
        push_field(
            &mut h,
            "GZIP_COMPRESSED_DATA",
            format!("1PB({max_fallback})"),
        )?;
    }
    if quantized {
        push_field(&mut h, "ZSCALE", "1D".to_string())?;
        push_field(&mut h, "ZZERO", "1D".to_string())?;
    }
    h.push("ZIMAGE", Value::Logical(true), None)?;
    let cmp_comment = match opts.codec {
        Codec::Gzip1 => "gzip RFC 1952",
        Codec::Gzip2 => "gzip RFC 1952, byte planes shuffled",
        Codec::Rice1 { .. } => "Rice coding",
    };
    h.push(
        "ZCMPTYPE",
        Value::String(opts.codec.zcmptype().into()),
        Some(cmp_comment),
    )?;
    // Sec.10.2 keeps the original header's structural cards in `Z`
    // form, and a reader rebuilds that header by walking them in the
    // order it finds them. So they go in the order their image
    // counterparts had: `SIMPLE` or `XTENSION` first, then `BITPIX`,
    // `NAXIS`, `NAXISn`, then the rest. Emitting `ZSIMPLE` after
    // `ZBITPIX` rebuilds a header starting with `BITPIX`, which
    // cfitsio's `funpack` rejects as not a header at all.
    if matches!(src.first("SIMPLE"), Some(Value::Logical(true))) {
        h.push(
            "ZSIMPLE",
            Value::Logical(true),
            Some("conforms to FITS standard"),
        )?;
    } else if let Some(Value::String(kind)) = src.first("XTENSION") {
        h.push(
            "ZTENSION",
            Value::String(kind),
            Some("original extension type"),
        )?;
    }
    h.push("ZBITPIX", Value::Integer(bitpix), None)?;
    h.push("ZNAXIS", Value::Integer(axes.len() as i64), None)?;
    for (i, &n) in axes.iter().enumerate() {
        h.push(format!("ZNAXIS{}", i + 1), Value::Integer(n as i64), None)?;
    }
    // `EXTEND`, `PCOUNT` and `GCOUNT` follow `NAXISn` in the header
    // this came from, so their `Z` forms follow `ZNAXISn` here.
    if let Some(Value::Logical(extend)) = src.first("EXTEND") {
        h.push("ZEXTEND", Value::Logical(extend), None)?;
    }
    if let Some(Value::Integer(pcount)) = src.first("PCOUNT") {
        h.push("ZPCOUNT", Value::Integer(pcount), None)?;
    }
    if let Some(Value::Integer(gcount)) = src.first("GCOUNT") {
        h.push("ZGCOUNT", Value::Integer(gcount), None)?;
    }
    for (i, &t) in tile.iter().enumerate() {
        h.push(format!("ZTILE{}", i + 1), Value::Integer(t as i64), None)?;
    }
    if let Codec::Rice1 { blocksize } = opts.codec {
        // Quantized tiles hold i32 samples whatever the source BITPIX.
        let bytepix = if quantized {
            4
        } else {
            bitpix.unsigned_abs() as i64 / 8
        };
        h.push(
            "ZNAME1",
            Value::String("BLOCKSIZE".into()),
            Some("pixels per Rice block"),
        )?;
        h.push("ZVAL1", Value::Integer(i64::from(blocksize)), None)?;
        h.push(
            "ZNAME2",
            Value::String("BYTEPIX".into()),
            Some("bytes per pixel"),
        )?;
        h.push("ZVAL2", Value::Integer(bytepix), None)?;
    }
    if let Some(q) = &opts.quantize {
        h.push(
            "ZQUANTIZ",
            Value::String(q.dither.zquantiz().into()),
            Some("quantization dither method"),
        )?;
        h.push(
            "ZDITHER0",
            Value::Integer(i64::from(q.seed)),
            Some("dither seed offset"),
        )?;
        h.push(
            "ZBLANK",
            Value::Integer(i64::from(NULL_VALUE)),
            Some("undefined-pixel sentinel"),
        )?;
    } else if matches!(bitpix, -32 | -64) {
        // A lossless float image gzips the raw IEEE bytes directly.
        // Without ZQUANTIZ, readers default to NO_DITHER and demand
        // ZSCALE/ZZERO; ZQUANTIZ='NONE' says the tile bytes are raw
        // float pixels.
        h.push(
            "ZQUANTIZ",
            Value::String("NONE".into()),
            Some("no quantization (raw IEEE bytes)"),
        )?;
    }
    // The convention names this extension `COMPRESSED_IMAGE`, which is
    // what cfitsio and astropy write and what a reader looks for. A
    // name the caller chose wins, and so does one the source image
    // carried: that is the image's own name, not ours to replace.
    let extname = opts.extname.clone().or_else(|| match src.first("EXTNAME") {
        Some(Value::String(name)) => Some(name),
        _ => None,
    });
    h.push(
        "EXTNAME",
        Value::String(extname.unwrap_or_else(|| "COMPRESSED_IMAGE".to_string())),
        Some("name of this binary table extension"),
    )?;
    // Carry every other card of the source image, per Sec.10.2.
    //
    // Record the keywords this function emitted above. A source card
    // that maps to one of them is skipped: `opts.extname` wins over a
    // source `EXTNAME`, and the computed `ZBLANK` wins over the one a
    // source `BLANK` maps to.
    //
    // The test uses this record, not the contents of `h`. A repeated
    // source keyword is legal, and its second card is not in the
    // record, so this keeps it.
    let emitted: std::collections::HashSet<String> = h.cards().map(|c| c.keyword()).collect();
    for entry in src.cards() {
        if entry.is_commentary() {
            h.splice(&entry);
            continue;
        }
        let keyword = entry.keyword();
        let Some(mapped) = image_to_z_keyword(&keyword) else {
            continue;
        };
        if emitted.contains(mapped) {
            continue;
        }
        if mapped == keyword {
            // The name does not change, so the card moves whole.
            h.splice(&entry);
        } else if let Some(v) = entry.value() {
            // The name changes to its `Z` form, so the original
            // bytes no longer describe the card. Re-encode it.
            h.push(mapped, v, entry.comment().as_deref())?;
        }
    }
    Ok(h)
}

/// Tile compression codec for the write path.
///
/// The read path is independent of this type and decodes every
/// algorithm of Standard Sec.10 Table 10.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// `GZIP_1` -- gzip over the big-endian pixel bytes. Lossless for
    /// every `BITPIX`. The default.
    #[default]
    Gzip1,
    /// `GZIP_2` -- gzip over byte-plane-shuffled pixel bytes.
    /// Lossless for every `BITPIX`. Often smaller than `GZIP_1` on
    /// floating-point data, because the exponent bytes group
    /// together.
    Gzip2,
    /// `RICE_1` -- Rice coding (Pence et al. 2010 Sec.3.1). Integer
    /// images with 1, 2 or 4 bytes per pixel only.
    Rice1 {
        /// Pixels per Rice block. A reader assumes 32 when the file
        /// names no `BLOCKSIZE`, so 32 is the portable choice.
        blocksize: u32,
    },
}

impl Codec {
    /// The `ZCMPTYPE` value this codec writes.
    #[must_use]
    pub fn zcmptype(&self) -> &'static str {
        match self {
            Self::Gzip1 => "GZIP_1",
            Self::Gzip2 => "GZIP_2",
            Self::Rice1 { .. } => "RICE_1",
        }
    }

    /// A `RICE_1` codec with a block size of 32, the value a reader
    /// assumes when `BLOCKSIZE` is absent.
    #[must_use]
    pub fn rice() -> Self {
        Self::Rice1 { blocksize: 32 }
    }
}

/// Opt-in lossy quantization of a floating-point image
/// (Sec.10.4.4).
///
/// Quantization maps each tile's float pixels to 32-bit integers so
/// an integer codec such as `RICE_1` can compress them. The mapping
/// discards precision below the quantization step, which is the
/// estimated per-tile noise divided by [`level`](Self::level). A tile
/// with no measurable noise, or with a value range too wide for the
/// step, falls back to lossless gzip in a `GZIP_COMPRESSED_DATA`
/// column.
///
/// A non-finite pixel is recorded through `ZBLANK` and reads back as
/// NaN. Under [`DitherMethod::Subtractive2`] an exact zero reads back
/// as exact zero.
#[derive(Debug, Clone, Copy)]
pub struct Quantize {
    /// Quantization level `q`. The step is the estimated tile noise
    /// divided by `q`, so a larger level keeps more precision. The
    /// default is 4.
    pub level: f64,
    /// Dither method, written to `ZQUANTIZ`.
    /// [`DitherMethod::Subtractive1`] by default.
    pub dither: DitherMethod,
    /// `ZDITHER0` seed offset. 1 by default, so output is
    /// deterministic.
    pub seed: u32,
}

impl Default for Quantize {
    fn default() -> Self {
        Self {
            level: 4.0,
            dither: DitherMethod::Subtractive1,
            seed: 1,
        }
    }
}

impl Quantize {
    /// A quantization at `level` with the default dither and seed.
    #[must_use]
    pub fn level(level: f64) -> Self {
        Self {
            level,
            ..Self::default()
        }
    }
}

/// Per-call options for [`compress_image_to_hdu`] and
/// [`write_hdu_compressed`].
///
/// [`write_hdu_compressed`]: crate::FitsWriter::write_hdu_compressed
///
/// All fields are optional. The default value emits `GZIP_1` tiles,
/// tiles by `NAXIS1` rows (per Pence & Seaman Sec.3), and emits no
/// `EXTNAME` card.
#[derive(Debug, Default, Clone)]
pub struct TileOpts {
    /// Compression codec. [`Codec::Gzip1`] by default.
    pub codec: Codec,
    /// Tile shape in FITS axis order (`tile[0]` = `NAXIS1` direction).
    /// Length must equal `NAXIS`. `None` selects `(NAXIS1, 1, 1, ...)`.
    pub tile: Option<Vec<u64>>,
    /// `EXTNAME` to stamp on the resulting BINTABLE. A source-header
    /// `EXTNAME` card carries through when this is `None`.
    pub extname: Option<String>,
    /// Lossy quantization for a floating-point image. `None`, the
    /// default, compresses floats losslessly.
    pub quantize: Option<Quantize>,
}

impl TileOpts {
    /// Construct an options bag with the default codec and tiling and
    /// no `EXTNAME`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the compression codec.
    #[must_use]
    pub fn codec(mut self, codec: Codec) -> Self {
        self.codec = codec;
        self
    }

    /// Override the tile shape.
    #[must_use]
    pub fn tile(mut self, tile: impl Into<Vec<u64>>) -> Self {
        self.tile = Some(tile.into());
        self
    }

    /// Set the `EXTNAME` of the compressed BINTABLE.
    #[must_use]
    pub fn extname(mut self, name: impl Into<String>) -> Self {
        self.extname = Some(name.into());
        self
    }

    /// Quantize a floating-point image before compression. Lossy.
    #[must_use]
    pub fn quantize(mut self, quantize: Quantize) -> Self {
        self.quantize = Some(quantize);
        self
    }
}

impl<W: std::io::Write> crate::io::writer::FitsWriter<W> {
    /// Tile-compress an IMAGE HDU and stream it out as a BINTABLE.
    ///
    /// The `header` and `data` arguments describe the uncompressed
    /// image, as [`ImageBuilder`](crate::ImageBuilder) emits it. This
    /// re-encodes them per Sec.7.4 and writes them through
    /// [`write_hdu`](Self::write_hdu).
    ///
    /// The `opts` argument sets the codec, the tile shape and the
    /// `EXTNAME` of the resulting table. Pass `&TileOpts::default()`
    /// for `GZIP_1` with the default tile shape.
    ///
    /// # Errors
    ///
    /// The conditions of [`compress_image_to_hdu`] and of
    /// [`write_hdu`](Self::write_hdu).
    pub fn write_hdu_compressed(
        &mut self,
        header: &Header,
        data: &[u8],
        opts: &TileOpts,
    ) -> Result<()> {
        let (cz_h, cz_data) = compress_image_to_hdu(header, data, opts)?;
        self.write_hdu(&cz_h, &cz_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    #[test]
    fn maybe_gunzip_passes_through_plain_bytes() {
        let v = vec![b'S', b'I', b'M', b'P', b'L', b'E'];
        assert_eq!(maybe_gunzip(v.clone()).unwrap(), v);
    }

    #[test]
    fn maybe_gunzip_inflates_gzip() {
        let payload = b"hello fits".to_vec();
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(&payload).unwrap();
        let gz = e.finish().unwrap();
        assert_eq!(maybe_gunzip(gz).unwrap(), payload);
    }

    #[test]
    fn unshuffle_round_trips() {
        let pixels: [i32; 3] = [0x01020304, 0x05060708, 0x090a0b0c];
        let mut be = Vec::new();
        for p in &pixels {
            be.extend_from_slice(&p.to_be_bytes());
        }
        let bpp = 4;
        let n = pixels.len();
        let mut shuf = vec![0_u8; n * bpp];
        for (i, p) in pixels.iter().enumerate() {
            let bytes = p.to_be_bytes();
            for plane in 0..bpp {
                shuf[plane * n + i] = bytes[plane];
            }
        }
        unshuffle(&mut shuf, bpp).unwrap();
        assert_eq!(shuf, be);
    }

    #[test]
    fn parse_descriptor_rejects_short_p_cell() {
        assert!(parse_descriptor(BinFieldKind::P, &[0_u8; 4]).is_err());
    }

    #[test]
    fn parse_descriptor_rejects_short_q_cell() {
        assert!(parse_descriptor(BinFieldKind::Q, &[0_u8; 8]).is_err());
    }

    #[test]
    fn parse_descriptor_rejects_negative_fields() {
        let mut raw = [0_u8; 8];
        raw[..4].copy_from_slice(&(-1_i32).to_be_bytes());
        assert!(parse_descriptor(BinFieldKind::P, &raw).is_err());
    }

    #[test]
    fn decompress_tile_rice_matches_be_pixels() {
        let pixels: Vec<i16> = (0..32).map(|i| i * 17 - 100).collect();
        let payload = rice::encode_short(&pixels, 32);
        let mut out = vec![0_u8; pixels.len() * 2];
        decompress_tile(
            TilePayload::Compressed(&payload),
            &mut out,
            CmpType::Rice1 { blocksize: 32 },
            2,
            32,
        )
        .unwrap();
        let expected: Vec<u8> = pixels.iter().flat_map(|p| p.to_be_bytes()).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn decompress_tile_uncompressed_passthrough() {
        let bytes = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let mut out = vec![0_u8; 8];
        decompress_tile(
            TilePayload::Uncompressed(&bytes),
            &mut out,
            CmpType::Rice1 { blocksize: 32 },
            2,
            4,
        )
        .unwrap();
        assert_eq!(out, bytes);
    }

    #[test]
    fn decompress_tile_gzip_fallback_on_rice_hdu() {
        let pixels: [i16; 4] = [10, 20, 30, 40];
        let mut be = Vec::new();
        for p in &pixels {
            be.extend_from_slice(&p.to_be_bytes());
        }
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(&be).unwrap();
        let payload = e.finish().unwrap();
        let mut out = vec![0_u8; be.len()];
        decompress_tile(
            TilePayload::GzipFallback(&payload),
            &mut out,
            CmpType::Rice1 { blocksize: 32 },
            2,
            4,
        )
        .unwrap();
        assert_eq!(out, be);
    }
}
