//! Optional compressed-input support (cargo feature `compression`).
//!
//! Two distinct conventions are handled here:
//!
//! * **Whole-file gzip** (`*.gz`): the FITS file has been compressed
//!   with `gzip(1)`. [`maybe_gunzip`] inflates such a buffer back to
//!   the underlying FITS bytes, leaving any non-gzipped buffer
//!   unchanged.
//!
//! * **FITS tile-compressed images** (`*.fz`, Pence & Seaman 2010
//!   / FITS standard 2016 Sec.10). The image is stored as a `BINTABLE`
//!   carrying `ZIMAGE = T`, `ZBITPIX`, `ZNAXIS`, `ZNAXISn`,
//!   `ZTILEn`, `ZCMPTYPE`, plus a variable-length `COMPRESSED_DATA`
//!   column. Each row holds one tile. The standard also lets any
//!   tile fall back to GZIP via `GZIP_COMPRESSED_DATA`, or store
//!   raw pixels verbatim via `UNCOMPRESSED_DATA`.
//!
//! This crate currently implements `GZIP_1`, `GZIP_2`, `RICE_1`,
//! `PLIO_1` and `HCOMPRESS_1` for 8-/16-/32-/64-bit integer images,
//! plus `NO_DITHER` / `SUBTRACTIVE_DITHER_1` / `SUBTRACTIVE_DITHER_2`
//! quantized float images and lossless float compression via `GZIP_1`
//! / `GZIP_2`.

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
use crate::header::value::Value;

use self::quantize::{DitherMethod, NULL_VALUE};

/// gzip RFC 1952 magic bytes.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// If `buf` starts with the gzip magic, decompress it; otherwise
/// return it unchanged.
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

/// View over a tile-compressed image HDU.
#[derive(Debug)]
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
    /// Per-tile seed offset (`ZDITHER0`, default 0).
    dither_seed: u32,
    /// Integer sentinel for NaN/Inf source pixels.
    blank: i32,
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
}

impl<'a> CompressedImageHdu<'a> {
    /// Wrap a `BinTableHdu` whose header carries `ZIMAGE = T`.
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
            let blank = h.optional_int("ZBLANK").map_or(NULL_VALUE, |v| v as i32);
            let dither_seed = h
                .optional_int("ZDITHER0")
                .filter(|v| *v >= 0)
                .map_or(0, |v| v as u32);
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
        // Lossless float: only GZIP_1 / GZIP_2 are defined.
        if is_float && quantize.is_none() && !matches!(cmptype_s, "GZIP_1" | "GZIP_2") {
            return Err(FitsError::NonStandard(format!(
                "lossless float compression requires ZCMPTYPE=GZIP_1 or GZIP_2, \
                 got {cmptype_s}"
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
            other => {
                return Err(FitsError::NonStandard(format!(
                    "tile compression algorithm `{other}` is not supported \
                     (this build supports GZIP_1, GZIP_2, RICE_1, PLIO_1 and HCOMPRESS_1)"
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
    pub fn header(&self) -> &Header {
        self.inner.header()
    }
    /// Borrow the underlying BINTABLE view (the on-disk
    /// representation of this compressed image HDU).
    #[must_use]
    pub fn as_bintable(&self) -> &BinTableHdu<'a> {
        &self.inner
    }
    #[must_use]
    pub fn bitpix(&self) -> Bitpix {
        self.bitpix
    }
    /// Original image dimensions (`ZNAXISn`, fastest-varying first).
    #[must_use]
    pub fn axes(&self) -> &[u64] {
        &self.axes
    }
    #[must_use]
    pub fn tile_shape(&self) -> &[u64] {
        &self.tile
    }

    /// Build a synthetic IMAGE-HDU header from the `Z*` keywords so
    /// callers can look up WCS, BUNIT, etc. through the same accessors
    /// they use for an uncompressed image. Implements the inverse of
    /// the convention in Sec.10.4: each `ZNAME` keyword is rewritten to
    /// its image equivalent (`ZBITPIX -> BITPIX`, `ZNAXISn -> NAXISn`,
    /// `ZCTYPEn -> CTYPEn`, ...).
    pub fn synthetic_image_header(&self) -> Result<Header> {
        synthesise_image_header(self.inner.header())
    }

    /// Decompress and wrap as an [`crate::ImageHdu`]. The returned
    /// HDU borrows from this `CompressedImageHdu` for its bytes.
    pub fn as_image(&self) -> Result<OwnedImage> {
        let bytes = self.decompress()?;
        let header = self.synthetic_image_header()?;
        OwnedImage::new(header, bytes)
    }

    /// Decompress all tiles into one big-endian byte buffer matching
    /// the layout of an uncompressed image HDU.
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
                let dither_seed = u64::from(q.dither_seed) + (row as u64) + 1;
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
                        q.blank,
                        dither_arg,
                    ),
                    Bitpix::F64 => quantize::unquantize_to_f64_be(
                        &tile_buf,
                        &mut float_buf,
                        scale,
                        zero,
                        q.blank,
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
#[derive(Debug)]
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
    /// Big-endian raw pixel bytes (`NAXISn` product * |BITPIX|/8).
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Read the WCS from the synthetic header.
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
        TilePayload::Uncompressed(bytes) => {
            if bytes.len() != out.len() {
                return Err(FitsError::Data(format!(
                    "UNCOMPRESSED_DATA tile is {} bytes, expected {}",
                    bytes.len(),
                    out.len()
                )));
            }
            out.copy_from_slice(bytes);
            Ok(())
        }
        TilePayload::GzipFallback(bytes) => inflate_into(bytes, out),
        TilePayload::Compressed(bytes) => match cmptype {
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

/// Pence & Seaman 2010 Sec.3.3: `HCOMPRESS_1` carries `SCALE`
/// (integer quantisation) and `SMOOTH` (0/1) via `ZNAMEn`/`ZVALn`.
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

// -- Synthetic image header ----------------------------------------

/// Map a Z-prefixed compressed-image keyword to its image-HDU
/// equivalent per Sec.10.4. Returns `None` if the keyword should be
/// dropped (e.g. ZIMAGE itself, ZCMPTYPE, `COMPRESSED_DATA` bookkeeping).
fn z_to_image_keyword(k: &str) -> Option<String> {
    const T_PREFIXES: &[&str] = &[
        "TTYPE", "TFORM", "TUNIT", "TDIM", "TSCAL", "TZERO", "TNULL", "TDISP", "TBCOL",
    ];
    // Keywords that are bookkeeping for the BINTABLE container and
    // don't belong on the synthetic IMAGE.
    let drop = [
        "ZIMAGE", "ZCMPTYPE", "ZQUANTIZ", "ZDITHER0", "ZMASKCMP", "ZSIMPLE", "ZTENSION", "ZEXTEND",
        "ZBLOCKED", "ZPCOUNT", "ZGCOUNT", "ZHECKSUM", "ZDATASUM", "XTENSION", "BITPIX", "NAXIS",
        "NAXIS1", "NAXIS2", "PCOUNT", "GCOUNT", "TFIELDS", "EXTEND", "THEAP",
    ];
    if drop.contains(&k) {
        return None;
    }
    // ZNAMEn / ZVALn are algorithm parameters, not image keywords.
    for prefix in ["ZNAME", "ZVAL"] {
        if let Some(rest) = k.strip_prefix(prefix)
            && !rest.is_empty()
            && rest.chars().all(|c| c.is_ascii_digit())
        {
            return None;
        }
    }
    // TTYPEn / TFORMn / TUNITn / TDIMn / TSCALn / TZEROn / TNULLn /
    // TDISPn / TBCOLn are BINTABLE column descriptors -- drop them all.
    if T_PREFIXES.iter().any(|p| {
        k.starts_with(p) && {
            let rest = &k[p.len()..];
            !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
        }
    }) {
        return None;
    }
    // The actual mapping: strip a single leading `Z` for any keyword
    // beginning with one (ZBITPIX -> BITPIX, ZNAXIS -> NAXIS,
    // ZCTYPE1 -> CTYPE1, ...) and keep all non-Z keywords as-is.
    Some(if let Some(stripped) = k.strip_prefix('Z') {
        stripped.to_string()
    } else {
        k.to_string()
    })
}

fn synthesise_image_header(bt: &Header) -> Result<Header> {
    let mut out = Header::empty();
    let bitpix = bt.optional_int("ZBITPIX").unwrap_or(8);
    let znaxis = bt.optional_int("ZNAXIS").unwrap_or(0);
    out.push("XTENSION", Value::String("IMAGE".into()), None)?;
    out.push("BITPIX", Value::Integer(bitpix), None)?;
    out.push("NAXIS", Value::Integer(znaxis), None)?;
    for i in 1..=znaxis {
        if let Some(n) = bt.optional_int(&format!("ZNAXIS{i}")) {
            out.push(format!("NAXIS{i}"), Value::Integer(n), None)?;
        }
    }
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

    // Pass through every other Z* / non-Z keyword, mapped to its image
    // form. Skip anything we already emitted above.
    let already_emitted = |kw: &str| -> bool {
        kw == "XTENSION"
            || kw == "BITPIX"
            || kw == "NAXIS"
            || kw == "PCOUNT"
            || kw == "GCOUNT"
            || kw == "BLANK"
            || (kw.starts_with("NAXIS") && kw[5..].chars().all(|c| c.is_ascii_digit()))
    };
    for entry in bt.entries() {
        let Some(mapped) = z_to_image_keyword(&entry.keyword) else {
            continue;
        };
        if already_emitted(&mapped) {
            continue;
        }
        if let Some(v) = entry.value.clone() {
            // Per-keyword failures aren't fatal -- silently drop and
            // continue rather than abort the whole header. Non-finite
            // reals and oversized strings (which would need CONTINUE
            // and aren't carried by ZIMAGE conventions) are skipped
            // by the builder via Err.
            if matches!(v, Value::Real(r) if !r.is_finite()) {
                continue;
            }
            let _ = out.push(mapped, v, entry.comment.as_deref());
        }
    }
    Ok(out)
}

/// Tile-compress a regular IMAGE HDU into a `(Header, data)` pair
/// describing a tile-compressed BINTABLE (Pence & Seaman 2010 /
/// Standard Sec.7.4). Only `GZIP_1` is emitted.
///
/// * `bitpix`, `axes` describe the source image (`axes[0]` is
///   `NAXIS1`, the fastest-varying axis).
/// * `raw` is the source image's big-endian byte payload; its length
///   must equal `bytes_per_pixel(bitpix) * product(axes)`.
/// * `tile` is the tile shape in FITS axis order. `None` defaults to
///   `(NAXIS1, 1, 1, ...)` per Pence & Seaman Sec.3.
/// * `extname` becomes the `EXTNAME` card on the resulting BINTABLE.
pub fn compress_image_to_hdu(
    bitpix: i64,
    axes: &[u64],
    raw: &[u8],
    tile: Option<&[u64]>,
    extname: Option<&str>,
) -> Result<(Header, Vec<u8>)> {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    let bytes_per = (bitpix.unsigned_abs() / 8) as usize;
    let n_pixels: u64 = if axes.is_empty() {
        0
    } else {
        axes.iter().product()
    };
    if raw.len() != bytes_per * n_pixels as usize {
        return Err(FitsError::Data(format!(
            "compress_image_to_hdu: raw is {} bytes; expected {} (BITPIX={bitpix}, n_pixels={n_pixels})",
            raw.len(),
            bytes_per * n_pixels as usize,
        )));
    }

    if axes.is_empty() {
        return Err(FitsError::Data(
            "compress_image_to_hdu: cannot compress a 0-axis image".into(),
        ));
    }
    let tile: Vec<u64> = if let Some(t) = tile {
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
        t.to_vec()
    } else {
        let mut t = vec![1_u64; axes.len()];
        t[0] = axes[0];
        t
    };
    // Number of tiles per axis.
    let n_tiles_per_axis: Vec<u64> = axes
        .iter()
        .zip(tile.iter())
        .map(|(&a, &t)| a.div_ceil(t))
        .collect();
    let total_tiles: u64 = n_tiles_per_axis.iter().copied().product();

    // Strides (in elements) of the source image, FITS order.
    let mut strides = Vec::with_capacity(axes.len());
    let mut s: u64 = 1;
    for &a in axes {
        strides.push(s);
        s = s.saturating_mul(a);
    }

    // Walk tiles in FITS order (axis 0 fastest).
    let mut row_data: Vec<u8> = Vec::with_capacity(total_tiles as usize * 8);
    let mut heap: Vec<u8> = Vec::new();
    let mut tile_index = vec![0_u64; axes.len()];
    let mut max_compressed: u32 = 0;

    loop {
        // Compute this tile's start and shape (clipped to image).
        let mut tile_start: Vec<u64> = Vec::with_capacity(axes.len());
        let mut tile_shape: Vec<u64> = Vec::with_capacity(axes.len());
        for ax in 0..axes.len() {
            let start = tile_index[ax] * tile[ax];
            let end = (start + tile[ax]).min(axes[ax]);
            tile_start.push(start);
            tile_shape.push(end - start);
        }

        // Extract tile bytes by walking outer axes.
        let n0 = tile_shape[0] as usize;
        let row_bytes = n0 * bytes_per;
        let tile_pixels: usize = tile_shape.iter().copied().product::<u64>() as usize;
        let mut tile_buf: Vec<u8> = Vec::with_capacity(tile_pixels * bytes_per);

        let mut idx = vec![0_u64; axes.len()];
        let outer = axes.len();
        'inner: loop {
            let mut elem_off: u64 = tile_start[0];
            for ax in 1..outer {
                elem_off += (tile_start[ax] + idx[ax]) * strides[ax];
            }
            let byte_off = (elem_off as usize) * bytes_per;
            tile_buf.extend_from_slice(&raw[byte_off..byte_off + row_bytes]);
            if outer == 1 {
                break 'inner;
            }
            let mut ax = 1;
            loop {
                idx[ax] += 1;
                if idx[ax] < tile_shape[ax] {
                    break;
                }
                idx[ax] = 0;
                ax += 1;
                if ax == outer {
                    break 'inner;
                }
            }
        }

        // Gzip the tile.
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(&tile_buf).map_err(FitsError::Io)?;
        let compressed = e.finish().map_err(FitsError::Io)?;
        let count = compressed.len() as u32;
        let offset = heap.len() as u32;
        row_data.extend_from_slice(&(count as i32).to_be_bytes());
        row_data.extend_from_slice(&(offset as i32).to_be_bytes());
        heap.extend_from_slice(&compressed);
        if count > max_compressed {
            max_compressed = count;
        }

        // Advance tile index.
        if outer == 1 {
            tile_index[0] += 1;
            if tile_index[0] >= n_tiles_per_axis[0] {
                break;
            }
            continue;
        }
        let mut ax = 0;
        loop {
            tile_index[ax] += 1;
            if tile_index[ax] < n_tiles_per_axis[ax] {
                break;
            }
            tile_index[ax] = 0;
            ax += 1;
            if ax == outer {
                let mut h = Header::empty();
                finalise_zimage_header(
                    &mut h,
                    bitpix,
                    axes,
                    &tile,
                    total_tiles,
                    heap.len(),
                    max_compressed,
                    extname,
                )?;
                return Ok((h, concat(row_data, &heap)));
            }
        }
    }
    let mut h = Header::empty();
    finalise_zimage_header(
        &mut h,
        bitpix,
        axes,
        &tile,
        total_tiles,
        heap.len(),
        max_compressed,
        extname,
    )?;
    Ok((h, concat(row_data, &heap)))
}

fn concat(mut a: Vec<u8>, b: &[u8]) -> Vec<u8> {
    a.extend_from_slice(b);
    a
}

#[allow(
    clippy::too_many_arguments,
    reason = "all parameters are required to build the tile-compressed FITS extension header"
)]
fn finalise_zimage_header(
    h: &mut Header,
    bitpix: i64,
    axes: &[u64],
    tile: &[u64],
    n_tiles: u64,
    heap_size: usize,
    max_compressed: u32,
    extname: Option<&str>,
) -> Result<()> {
    h.push("XTENSION", Value::String("BINTABLE".into()), None)?;
    h.push("BITPIX", Value::Integer(8), None)?;
    h.push("NAXIS", Value::Integer(2), None)?;
    h.push(
        "NAXIS1",
        Value::Integer(8),
        Some("8 = sizeof(P descriptor)"),
    )?;
    h.push(
        "NAXIS2",
        Value::Integer(n_tiles as i64),
        Some("number of tiles"),
    )?;
    h.push("PCOUNT", Value::Integer(heap_size as i64), None)?;
    h.push("GCOUNT", Value::Integer(1), None)?;
    h.push("TFIELDS", Value::Integer(1), None)?;
    h.push("TTYPE1", Value::String("COMPRESSED_DATA".into()), None)?;
    h.push(
        "TFORM1",
        Value::String(format!("1PB({max_compressed})")),
        None,
    )?;
    h.push("ZIMAGE", Value::Logical(true), None)?;
    h.push(
        "ZCMPTYPE",
        Value::String("GZIP_1".into()),
        Some("gzip RFC 1952"),
    )?;
    h.push("ZBITPIX", Value::Integer(bitpix), None)?;
    h.push("ZNAXIS", Value::Integer(axes.len() as i64), None)?;
    // For floating-point images we gzip the raw IEEE bytes directly
    // (no quantization). Without ZQUANTIZ, readers default to
    // NO_DITHER and demand ZSCALE/ZZERO; ZQUANTIZ='NONE' tells them
    // the tile bytes are raw float pixels.
    if matches!(bitpix, -32 | -64) {
        h.push(
            "ZQUANTIZ",
            Value::String("NONE".into()),
            Some("no quantization (raw IEEE bytes)"),
        )?;
    }
    for (i, &n) in axes.iter().enumerate() {
        h.push(format!("ZNAXIS{}", i + 1), Value::Integer(n as i64), None)?;
    }
    for (i, &t) in tile.iter().enumerate() {
        h.push(format!("ZTILE{}", i + 1), Value::Integer(t as i64), None)?;
    }
    if let Some(name) = extname {
        h.push("EXTNAME", Value::String(name.to_string()), None)?;
    }
    Ok(())
}

/// Per-call options for [`FitsWriter::write_hdu_compressed`](crate::FitsWriter::write_hdu_compressed).
///
/// All fields are optional. The default value tiles by `NAXIS1` rows
/// (per Pence & Seaman Sec.3) and emits no `EXTNAME` card.
#[derive(Debug, Default, Clone)]
pub struct TileOpts {
    /// Tile shape in FITS axis order (`tile[0]` = `NAXIS1` direction).
    /// Length must equal `NAXIS`. `None` selects `(NAXIS1, 1, 1, ...)`.
    pub tile: Option<Vec<u64>>,
    /// `EXTNAME` to stamp on the resulting BINTABLE.
    pub extname: Option<String>,
}

impl TileOpts {
    /// Construct an options bag with default tiling and no `EXTNAME`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
}

impl<W: std::io::Write> crate::io::writer::FitsWriter<W> {
    /// Tile-compress an IMAGE HDU and stream it out as a BINTABLE.
    ///
    /// `header` and `data` describe the **uncompressed** image (the
    /// same `(Header, Vec<u8>)` pair an
    /// [`ImageBuilder`](crate::ImageBuilder) would emit). The writer
    /// re-encodes them per Standard Sec.7.4 (Pence & Seaman 2010) and
    /// writes the resulting tile-compressed BINTABLE through
    /// [`write_hdu`](Self::write_hdu).
    ///
    /// Only `GZIP_1` is emitted. See [`TileOpts`] for tile-shape and
    /// `EXTNAME` controls.
    pub fn write_hdu_compressed(
        &mut self,
        header: &Header,
        data: &[u8],
        opts: &TileOpts,
    ) -> Result<()> {
        let bitpix = header.bitpix()?;
        let axes = header.axes()?;
        let (cz_h, cz_data) = compress_image_to_hdu(
            bitpix,
            &axes,
            data,
            opts.tile.as_deref(),
            opts.extname.as_deref(),
        )?;
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
