//! Writer-side Python bindings: `image()`, `bintable()`,
//! `ascii_table()`, `compressed_image()` and `write()`.
//!
//! A caller builds one HDU specification per call to `image()`,
//! `bintable()`, `ascii_table()` or `compressed_image()`, then hands
//! the list of specifications to `write()`. Each factory function
//! covers a fixed, common set of dtypes and column kinds. A custom
//! `TFORM` code for a binary table column, or column scaling
//! (`TSCALn`/`TZEROn`), needs the Rust API instead.

use std::io::BufWriter;
use std::path::PathBuf;

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::hdu::ascii_table::AsciiFormat;
use crate::hdu::{AsciiColumnData, BinFieldKind};
use crate::header::Header;
use crate::{AsciiTableBuilder, BinTableBuilder, FitsWriter, ImageBuilder};

use super::IntoPyResult;
use super::header::header_from_py;

/// Opaque image HDU spec produced by :func:`image`.
///
/// Pass to :func:`write` as part of a list of builders.
#[pyclass(name = "ImageBuilder", module = "fitsy")]
#[derive(Debug)]
pub struct PyImageBuilder {
    pub(crate) header: Header,
    pub(crate) data: Vec<u8>,
}

/// Opaque BINTABLE HDU spec produced by :func:`bintable`.
#[pyclass(name = "BinTableBuilder", module = "fitsy")]
#[derive(Debug)]
pub struct PyBinTableBuilder {
    pub(crate) header: Header,
    pub(crate) data: Vec<u8>,
}

/// Opaque ASCII TABLE HDU spec produced by :func:`ascii_table`.
#[pyclass(name = "AsciiTableBuilder", module = "fitsy")]
#[derive(Debug)]
pub struct PyAsciiTableBuilder {
    pub(crate) header: Header,
    pub(crate) data: Vec<u8>,
}

/// Build an image HDU from an array.
///
/// Parameters
/// ----------
/// data : array-like
///   Image pixels. A numpy array of dtype ``bool``, ``int8``,
///   ``uint8``, ``int16``, ``uint16``, ``int32``, ``uint32``,
///   ``int64``, ``uint64``, ``float32`` or ``float64``, or anything
///   :func:`numpy.asarray` accepts (nested lists, tuples, objects
///   implementing ``__array__``). The returned HDU's ``NAXIS`` list
///   is the reverse of ``data.shape``, because numpy lists axes
///   slowest-first and FITS lists them fastest-first.
/// header : Header or mapping, optional
///   Extra header cards to merge in. A :class:`Header` or a
///   ``dict``. Values may be scalars or ``(value, comment)`` tuples.
///   Default ``None``, which adds no extra cards.
/// primary : bool, optional
///   Default ``True``. Builds the header with ``SIMPLE = T``. Set
///   to ``False`` to build an image extension header instead
///   (``XTENSION = 'IMAGE'`` with ``PCOUNT`` and ``GCOUNT``).
///
/// Returns
/// -------
/// ImageBuilder
///   Pass to :func:`write`.
///
/// Raises
/// ------
/// TypeError
///   If `data` is neither an array of one of the dtypes above nor
///   something :func:`numpy.asarray` accepts.
///
/// Notes
/// -----
/// ``int8``, ``uint16``, ``uint32`` and ``uint64`` have no direct
/// FITS type. fitsy stores each one under the FITS unsigned-integer
/// convention, which a conforming reader decodes back to the
/// original values. fitsy keeps a ``BZERO`` or ``BSCALE`` card that
/// `header` supplies, in place of the card it would compute. A
/// ``bool`` array is stored as ``BITPIX = 8``.
///
/// Pass ``primary=False`` to every :func:`image` call after the
/// first item passed to :func:`write`. :func:`write` raises
/// :class:`fitsy.FitsError` if a later HDU still declares
/// ``SIMPLE``, or if the first HDU does not.
///
/// A ``COMMENT``, ``HISTORY`` or blank-keyword card in `header` is
/// not copied to the built HDU; only cards that carry a value are.
#[pyfunction]
#[pyo3(signature = (data, header=None, primary=true))]
pub fn image(
    data: Bound<'_, PyAny>,
    header: Option<Bound<'_, PyAny>>,
    primary: bool,
) -> PyResult<PyImageBuilder> {
    // numpy axes are row-major (slowest first); FITS NAXIS is
    // fastest first. Reverse before handing to ImageBuilder.
    let extra = match header.as_ref() {
        Some(d) => header_from_py(d)?,
        None => Header::empty(),
    };
    let (h, bytes) = build_image(&data, primary, extra)?;
    Ok(PyImageBuilder {
        header: h,
        data: bytes,
    })
}

/// Tile-compress a numpy array into a ``BINTABLE`` HDU (``ZIMAGE``).
///
/// The result is a tile-compressed image extension: a ``BINTABLE``
/// with ``ZIMAGE = T`` whose rows each hold one compressed tile.
/// :func:`fitsy.open` reads it back as an image.
///
/// Parameters
/// ----------
/// data : array-like
///   Image pixels. A numpy array of dtype ``bool``, ``int8``,
///   ``uint8``, ``int16``, ``uint16``, ``int32``, ``uint32``,
///   ``int64``, ``uint64``, ``float32`` or ``float64``, or anything
///   :func:`numpy.asarray` accepts.
/// header : Header or mapping, optional
///   Extra cards merged into the synthesized image header before
///   compression. A :class:`Header` or a ``dict``. Default
///   ``None``, which adds no extra cards.
/// tile_shape : sequence of int, optional
///   Tile shape in FITS axis order (``tile_shape[0]`` is the
///   ``NAXIS1`` direction). Length must equal ``data.ndim``.
///   Default ``None``, which tiles as ``(NAXIS1, 1, 1, ...)`` --
///   one row per tile (Pence & Seaman 2010 Sec.3).
/// extname : str, optional
///   ``EXTNAME`` keyword on the resulting BINTABLE. Default
///   ``"COMPRESSED_IMAGE"``.
///
/// Returns
/// -------
/// BinTableBuilder
///   Pass to :func:`write`.
///
/// Raises
/// ------
/// TypeError
///   If `data` is neither an array of one of the dtypes above nor
///   something :func:`numpy.asarray` accepts.
/// FitsError
///   If `data` has no axes (a 0-D array). Also raised if
///   `tile_shape` is given and its length does not equal
///   ``data.ndim``, or if one of its entries is ``0``.
///
/// Notes
/// -----
/// fitsy emits only ``ZCMPTYPE = 'GZIP_1'`` compressed tiles.
///
/// fitsy ignores a structural keyword (such as ``BITPIX``), a
/// ``Z*`` keyword, and ``EXTNAME``, in `header`, keeping the value
/// it computes for the compressed HDU instead. Set ``EXTNAME``
/// through `extname`.
#[pyfunction]
#[pyo3(signature = (data, header=None, *, tile_shape=None, extname=None))]
pub fn compressed_image(
    data: Bound<'_, PyAny>,
    header: Option<Bound<'_, PyAny>>,
    tile_shape: Option<Vec<u64>>,
    extname: Option<String>,
) -> PyResult<PyBinTableBuilder> {
    use crate::Value;
    use crate::compression::compress_image_to_hdu;
    let extra = match header.as_ref() {
        Some(d) => header_from_py(d)?,
        None => Header::empty(),
    };
    // Build the uncompressed image first so we get correct BITPIX
    // and big-endian raw bytes; then hand off to the Rust compressor.
    let (img_header, raw) = build_image(&data, false, extra)?;
    let bitpix = match img_header.first("BITPIX") {
        Some(Value::Integer(i)) => *i,
        _ => {
            return Err(PyValueError::new_err(
                "compressed_image: BITPIX missing from synthesized image header",
            ));
        }
    };
    let naxis: i64 = match img_header.first("NAXIS") {
        Some(Value::Integer(i)) => *i,
        _ => 0,
    };
    let mut axes: Vec<u64> = Vec::with_capacity(naxis.max(0) as usize);
    for k in 1..=naxis {
        let key = format!("NAXIS{k}");
        let n = match img_header.first(&key) {
            Some(Value::Integer(i)) => *i,
            _ => 0,
        };
        axes.push(n.max(0) as u64);
    }
    let extname = extname.unwrap_or_else(|| "COMPRESSED_IMAGE".to_string());
    let (mut bin_header, bin_bytes) = compress_image_to_hdu(
        bitpix,
        &axes,
        &raw,
        tile_shape.as_deref(),
        Some(extname.as_str()),
    )
    .into_py_result()?;
    // Merge user-supplied non-structural cards into the BINTABLE
    // header so end users still see their EXPTIME, OBSERVER, etc.
    for entry in img_header.entries() {
        if let Some(v) = entry.value.as_ref() {
            let kw = entry.keyword.to_ascii_uppercase();
            if matches!(
                kw.as_str(),
                "SIMPLE"
                    | "BITPIX"
                    | "NAXIS"
                    | "EXTEND"
                    | "PCOUNT"
                    | "GCOUNT"
                    | "XTENSION"
                    | "ZIMAGE"
                    | "ZBITPIX"
                    | "ZNAXIS"
                    | "ZCMPTYPE"
                    | "ZTILE1"
                    | "ZTILE2"
                    | "ZTILE3"
                    | "ZTILE4"
                    | "EXTNAME"
            ) || (kw.starts_with("NAXIS") && kw[5..].chars().all(|c| c.is_ascii_digit()))
                || (kw.starts_with("ZNAXIS") && kw[6..].chars().all(|c| c.is_ascii_digit()))
                || (kw.starts_with("ZTILE") && kw[5..].chars().all(|c| c.is_ascii_digit()))
                || bin_header.first(&entry.keyword).is_some()
            {
                continue;
            }
            let _ = bin_header.set(&entry.keyword, v.clone(), entry.comment.as_deref());
        }
    }
    Ok(PyBinTableBuilder {
        header: bin_header,
        data: bin_bytes,
    })
}

/// Dispatch a numpy array to the matching [`ImageBuilder`] and
/// render it, merging in `extra`.
///
/// Handles all eleven supported dtypes: `bool`, `int8`, `uint8`,
/// `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`,
/// `float32` and `float64`. `bool` is promoted to `uint8`. Each of
/// `uint16`, `uint32`, `uint64` and `int8` is re-encoded into the
/// signed or unsigned FITS type of the same width, with a `BZERO`
/// offset applied through [`with_unsigned_scaling`]. `primary`
/// selects `SIMPLE` versus `XTENSION` on the built header; see
/// [`ImageBuilder::primary`].
///
/// # Errors
///
/// Returns [`PyTypeError`] if `arr` is neither an array of one of
/// the eleven dtypes above nor something [`super::as_native_ndarray`]
/// accepts. Returns the error from [`apply_extra_header`] if a card
/// in `extra` fails validation.
pub(crate) fn build_image(
    arr: &Bound<'_, PyAny>,
    primary: bool,
    extra: Header,
) -> PyResult<(Header, Vec<u8>)> {
    // Normalize to a native-byte-order ndarray before type dispatch.
    // `extract::<PyReadonlyArrayDyn<T>>()` below matches neither a
    // non-array sequence nor a byte-swapped dtype, so both would
    // otherwise fall through every arm below. Arrays that are already
    // native pass through untouched and without an interpreter call.
    let arr = &super::as_native_ndarray(arr, "image")?.into_any();

    // Try every supported numpy dtype in turn. PyO3/numpy 0.22 has
    // no single dynamic dispatch helper, so we fan out manually.
    macro_rules! try_dtype {
        ($t:ty, $build:expr) => {
            if let Ok(view) = arr.extract::<PyReadonlyArrayDyn<'_, $t>>() {
                let shape = view.shape().to_vec();
                let axes: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
                let pixels: Vec<$t> = view.as_array().iter().copied().collect();
                let b: ImageBuilder<$t> = ImageBuilder::new(axes, pixels)
                    .into_py_result()?
                    .primary(primary);
                let b = $build(b);
                return apply_extra_header(b, extra);
            }
        };
    }
    try_dtype!(u8, |b: ImageBuilder<u8>| b);
    try_dtype!(i16, |b: ImageBuilder<i16>| b);
    try_dtype!(i32, |b: ImageBuilder<i32>| b);
    try_dtype!(i64, |b: ImageBuilder<i64>| b);
    try_dtype!(f32, |b: ImageBuilder<f32>| b);
    try_dtype!(f64, |b: ImageBuilder<f64>| b);
    // numpy bool has no native FITS BITPIX; promote to u8 (BITPIX=8).
    if let Ok(view) = arr.extract::<PyReadonlyArrayDyn<'_, bool>>() {
        let shape = view.shape().to_vec();
        let axes: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
        let pixels: Vec<u8> = view.as_array().iter().map(|&b| u8::from(b)).collect();
        let b: ImageBuilder<u8> = ImageBuilder::new(axes, pixels)
            .into_py_result()?
            .primary(primary);
        return apply_extra_header(b, extra);
    }
    // FITS unsigned-int convention: pick the matching signed BITPIX
    // and emit BZERO=2^(N-1) with BSCALE=1. A conforming reader then
    // returns the original unsigned values. This is the inverse of
    // the read path in `src/python/hdu.rs::read_image_array`.
    if let Ok(view) = arr.extract::<PyReadonlyArrayDyn<'_, u16>>() {
        let shape = view.shape().to_vec();
        let axes: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
        let pixels: Vec<i16> = view
            .as_array()
            .iter()
            .map(|&x| (i32::from(x) - 32_768) as i16)
            .collect();
        let b: ImageBuilder<i16> = ImageBuilder::new(axes, pixels)
            .into_py_result()?
            .primary(primary);
        return apply_extra_header(b, with_unsigned_scaling(extra, 32_768.0_f64));
    }
    if let Ok(view) = arr.extract::<PyReadonlyArrayDyn<'_, u32>>() {
        let shape = view.shape().to_vec();
        let axes: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
        let pixels: Vec<i32> = view
            .as_array()
            .iter()
            .map(|&x| (i64::from(x) - 2_147_483_648) as i32)
            .collect();
        let b: ImageBuilder<i32> = ImageBuilder::new(axes, pixels)
            .into_py_result()?
            .primary(primary);
        return apply_extra_header(b, with_unsigned_scaling(extra, 2_147_483_648.0_f64));
    }
    if let Ok(view) = arr.extract::<PyReadonlyArrayDyn<'_, u64>>() {
        let shape = view.shape().to_vec();
        let axes: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
        let pixels: Vec<i64> = view
            .as_array()
            .iter()
            .map(|&x| x.wrapping_sub(0x8000_0000_0000_0000) as i64)
            .collect();
        let b: ImageBuilder<i64> = ImageBuilder::new(axes, pixels)
            .into_py_result()?
            .primary(primary);
        // 2^63 is not representable as i64. Emit BZERO as a
        // real-valued card, which the FITS convention permits.
        return apply_extra_header(
            b,
            with_unsigned_scaling(extra, 9_223_372_036_854_775_808.0_f64),
        );
    }
    if let Ok(view) = arr.extract::<PyReadonlyArrayDyn<'_, i8>>() {
        let shape = view.shape().to_vec();
        let axes: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
        let pixels: Vec<u8> = view
            .as_array()
            .iter()
            .map(|&x| (i16::from(x) + 128) as u8)
            .collect();
        let b: ImageBuilder<u8> = ImageBuilder::new(axes, pixels)
            .into_py_result()?
            .primary(primary);
        return apply_extra_header(b, with_unsigned_scaling(extra, -128.0_f64));
    }
    Err(PyTypeError::new_err(
        "image: unsupported numpy dtype \
         (expected bool/i8/u8/i16/u16/i32/u32/i64/u64/f32/f64)",
    ))
}

/// Add `BZERO = bzero` and `BSCALE = 1` to `extra`, for the FITS
/// unsigned-integer (or signed-byte) convention. Both are written as
/// real-valued cards, because `bzero` is an `f64`. Sets a card only
/// when `extra` does not already carry it, so a card the caller set
/// wins.
fn with_unsigned_scaling(mut extra: Header, bzero: f64) -> Header {
    if extra.first("BZERO").is_none() {
        let _ = extra.set(
            "BZERO",
            bzero,
            Some("offset for unsigned/signed convention"),
        );
    }
    if extra.first("BSCALE").is_none() {
        let _ = extra.set("BSCALE", 1.0_f64, None);
    }
    extra
}

/// Copy every value-bearing, non-structural card from `extra` onto
/// `builder`, then render the finished `(Header, data)` pair.
///
/// Drops `SIMPLE`, `BITPIX`, `NAXIS`, `NAXISn`, `EXTEND`, `PCOUNT`,
/// `GCOUNT` and `XTENSION` from `extra`: [`ImageBuilder::build`]
/// writes those itself. Also drops a commentary card (`COMMENT`,
/// `HISTORY`, or blank keyword), because `extra.entries()` carries
/// no value for one.
///
/// # Errors
///
/// Returns the error from [`ImageBuilder::build`] if a copied card
/// fails validation.
fn apply_extra_header<T>(builder: ImageBuilder<T>, extra: Header) -> PyResult<(Header, Vec<u8>)>
where
    T: crate::data::Pixel,
{
    let mut b = builder;
    for entry in extra.entries() {
        if let Some(v) = entry.value.as_ref() {
            // Skip structural keywords ImageBuilder writes itself.
            let kw = entry.keyword.to_ascii_uppercase();
            if matches!(
                kw.as_str(),
                "SIMPLE" | "BITPIX" | "NAXIS" | "EXTEND" | "PCOUNT" | "GCOUNT" | "XTENSION"
            ) {
                continue;
            }
            if kw.starts_with("NAXIS") && kw[5..].chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            b = b.card(entry.keyword.clone(), v.clone(), entry.comment.as_deref());
        }
    }
    b.build().into_py_result()
}

/// Build a ``BINTABLE`` HDU from a column dictionary.
///
/// Parameters
/// ----------
/// columns : dict[str, sequence]
///   One entry per column. All columns must share the same row
///   count. Supported value kinds:
///
///   - a numpy ``bool``, ``uint8``, ``int16``, ``int32``, ``int64``,
///     ``float32`` or ``float64`` array (1-D, or 2-D for a
///     fixed-repeat column)
///   - ``list[str]`` -> ``nA`` (right-padded to the longest string)
///   - ``list[complex]`` -> ``M`` (``C128``)
///   - ``list[list[float]]`` -> ``1PD`` variable-length column
///     (heap-stored, ``f64`` element type). This applies to any
///     nested numeric list, ragged or not. Pass a 2-D numpy array
///     to get a fixed-repeat column instead.
///   - a flat sequence of numbers (``[1, 2, 3]``, a tuple, a
///     ``range``), converted with :func:`numpy.asarray` and encoded
///     as the dtype numpy infers
/// units : dict[str, str], optional
///   Per-column ``TUNITn`` strings. Default ``None``, which adds no
///   ``TUNITn`` cards. A key that names no column in `columns` is
///   ignored.
/// extname : str, optional
///   ``EXTNAME`` keyword for this extension. Default ``None``,
///   which adds no ``EXTNAME`` card.
///
/// Returns
/// -------
/// BinTableBuilder
///   Pass to :func:`write`.
///
/// Raises
/// ------
/// ValueError
///   If a column's values do not match one of the kinds listed
///   above, or if the columns disagree on row count.
///
/// Notes
/// -----
/// A numpy array of ``int8``, ``uint16``, ``uint32`` or ``uint64``
/// is not one of the supported kinds; convert it first, for example
/// with ``arr.astype(numpy.int32)``.
#[pyfunction]
#[pyo3(signature = (columns, units=None, extname=None))]
pub fn bintable(
    columns: Bound<'_, PyDict>,
    units: Option<Bound<'_, PyDict>>,
    extname: Option<&str>,
) -> PyResult<PyBinTableBuilder> {
    let mut bt = BinTableBuilder::new();
    if let Some(name) = extname {
        bt.extname(name);
    }
    let mut n_rows: Option<usize> = None;
    let mut encs: Vec<(String, ColumnEncoding)> = Vec::new();
    for (k, v) in columns.iter() {
        let name: String = k.extract()?;
        let enc = encode_column(&v)
            .map_err(|e| PyValueError::new_err(format!("bintable column {name:?}: {e}")))?;
        let rows = enc.n_rows();
        if let Some(prev) = n_rows {
            if prev != rows {
                return Err(PyValueError::new_err(format!(
                    "column {name:?} has {rows} rows, expected {prev}"
                )));
            }
        } else {
            n_rows = Some(rows);
        }
        let unit: Option<String> = match units.as_ref() {
            Some(d) => match d.get_item(&name)? {
                Some(v) => Some(v.extract()?),
                None => None,
            },
            None => None,
        };
        match &enc {
            ColumnEncoding::Fixed {
                kind, repeat, tdim, ..
            } => {
                bt.add_column(
                    name.clone(),
                    *kind,
                    *repeat,
                    unit.as_deref(),
                    tdim.as_deref(),
                )
                .into_py_result()?;
            }
            ColumnEncoding::Vla {
                element,
                descriptor,
                ..
            } => {
                bt.add_vla_column(name.clone(), *descriptor, *element, unit.as_deref(), None)
                    .into_py_result()?;
            }
        }
        encs.push((name, enc));
    }
    let n = n_rows.unwrap_or(0);
    let row_size = bt.row_bytes();
    let mut buf = vec![0_u8; row_size * n];
    let mut heap = Vec::<u8>::new();
    let mut col_offset = 0_usize;
    for (_name, enc) in &encs {
        match enc {
            ColumnEncoding::Fixed {
                kind,
                repeat,
                row_bytes,
                ..
            } => {
                let cell = kind_byte_size(*kind) * repeat;
                for r in 0..n {
                    let dst = &mut buf[r * row_size + col_offset..r * row_size + col_offset + cell];
                    let src = &row_bytes[r * cell..(r + 1) * cell];
                    dst.copy_from_slice(src);
                }
                col_offset += cell;
            }
            ColumnEncoding::Vla {
                descriptor,
                payloads,
                element,
                ..
            } => {
                let elt_size = kind_byte_size(*element);
                let cell = if matches!(descriptor, BinFieldKind::Q) {
                    16
                } else {
                    8
                };
                for (r, payload) in payloads.iter().enumerate() {
                    let count = (payload.len() / elt_size) as u64;
                    let offset = heap.len() as u64;
                    let descr = if matches!(descriptor, BinFieldKind::Q) {
                        BinTableBuilder::q_descriptor(count, offset).to_vec()
                    } else {
                        BinTableBuilder::p_descriptor(count as u32, offset as u32).to_vec()
                    };
                    let dst = &mut buf[r * row_size + col_offset..r * row_size + col_offset + cell];
                    dst.copy_from_slice(&descr);
                    heap.extend_from_slice(payload);
                }
                col_offset += cell;
            }
        }
    }
    let (h, data) = bt.build_with_heap(n, buf, &heap).into_py_result()?;
    Ok(PyBinTableBuilder { header: h, data })
}

fn kind_byte_size(k: BinFieldKind) -> usize {
    // Several variants share the same byte size; clippy's
    // `match_same_arms` would have us collapse them, but the explicit
    // mapping is clearer at a FITS-spec glance.
    #[allow(
        clippy::match_same_arms,
        reason = "explicit per-kind sizes mirror the FITS table spec"
    )]
    match k {
        BinFieldKind::Logical | BinFieldKind::Byte => 1,
        BinFieldKind::I16 => 2,
        BinFieldKind::I32 | BinFieldKind::F32 => 4,
        BinFieldKind::I64 | BinFieldKind::F64 | BinFieldKind::C64 => 8,
        BinFieldKind::C128 => 16,
        BinFieldKind::Char => 1,
        BinFieldKind::Bit => 1,
        BinFieldKind::P => 8,
        BinFieldKind::Q => 16,
    }
}

/// Encoded result for one column. `Fixed` columns occupy contiguous
/// row bytes; `Vla` columns place a `P`/`Q` descriptor in the row
/// area and append the payloads to the heap.
enum ColumnEncoding {
    Fixed {
        kind: BinFieldKind,
        repeat: usize,
        row_bytes: Vec<u8>,
        n_rows: usize,
        /// `TDIMn` to stamp, if the cell is multi-dimensional.
        tdim: Option<String>,
    },
    Vla {
        descriptor: BinFieldKind,
        element: BinFieldKind,
        /// Per-row big-endian heap payload. The descriptor's `count`
        /// is `payload.len() / sizeof(element)`.
        payloads: Vec<Vec<u8>>,
        n_rows: usize,
    },
}

impl ColumnEncoding {
    fn n_rows(&self) -> usize {
        match self {
            Self::Fixed { n_rows, .. } | Self::Vla { n_rows, .. } => *n_rows,
        }
    }
}

/// Encode one numpy array column. Returns `Ok(None)` when `arr` is
/// not an array of a directly supported dtype, so the caller can try
/// the sequence-shaped encodings (VLA, string, complex) instead.
fn encode_array_column(arr: &Bound<'_, PyAny>) -> PyResult<Option<ColumnEncoding>> {
    macro_rules! try_scalar {
        ($t:ty, $kind:expr, $to_be:expr) => {
            if let Ok(view) = arr.extract::<PyReadonlyArrayDyn<'_, $t>>() {
                let shape = view.shape().to_vec();
                if shape.is_empty() {
                    return Err(PyValueError::new_err("column array must be 1-D or 2-D"));
                }
                let n = shape[0];
                let repeat: usize = shape.iter().skip(1).product::<usize>().max(1);
                let mut bytes = Vec::with_capacity(n * repeat * std::mem::size_of::<$t>());
                for v in view.as_array().iter() {
                    bytes.extend_from_slice(&($to_be(*v)));
                }
                return Ok(Some(ColumnEncoding::Fixed {
                    kind: $kind,
                    repeat,
                    row_bytes: bytes,
                    n_rows: n,
                    tdim: None,
                }));
            }
        };
    }
    try_scalar!(u8, BinFieldKind::Byte, |v: u8| [v]);
    try_scalar!(i16, BinFieldKind::I16, |v: i16| v.to_be_bytes());
    try_scalar!(i32, BinFieldKind::I32, |v: i32| v.to_be_bytes());
    try_scalar!(i64, BinFieldKind::I64, |v: i64| v.to_be_bytes());
    try_scalar!(f32, BinFieldKind::F32, |v: f32| v.to_bits().to_be_bytes());
    try_scalar!(f64, BinFieldKind::F64, |v: f64| v.to_bits().to_be_bytes());
    if let Ok(view) = arr.extract::<PyReadonlyArrayDyn<'_, bool>>() {
        let shape = view.shape().to_vec();
        if shape.is_empty() {
            return Err(PyValueError::new_err("column array must be 1-D or 2-D"));
        }
        let n = shape[0];
        let repeat: usize = shape.iter().skip(1).product::<usize>().max(1);
        let mut bytes = Vec::with_capacity(n * repeat);
        for v in &view.as_array() {
            bytes.push(if *v { b'T' } else { b'F' });
        }
        return Ok(Some(ColumnEncoding::Fixed {
            kind: BinFieldKind::Logical,
            repeat,
            row_bytes: bytes,
            n_rows: n,
            tdim: None,
        }));
    }
    Ok(None)
}

/// Reject anything outside the restricted ASCII set a FITS character
/// field is defined over.
///
/// Sec.3 defines a character string as decimal 32-126, and Sec.7.2.5
/// says an `Aw` field *shall* be composed of it. Writing other bytes
/// would emit a file the standard does not describe and that another
/// reader is free to reject, so fail rather than silently transcode or
/// truncate what the caller passed. This matches the header writer,
/// which refuses a non-ASCII string value for the same reason.
fn validate_fits_ascii(strings: impl IntoIterator<Item = impl AsRef<str>>) -> PyResult<()> {
    for s in strings {
        let s = s.as_ref();
        if let Some(b) = s.bytes().find(|b| !(0x20..=0x7E).contains(b)) {
            return Err(PyValueError::new_err(format!(
                "character columns are restricted to ASCII 32-126 (Standard Sec.7.2.5); \
                 found byte 0x{b:02X} in {s:?}"
            )));
        }
    }
    Ok(())
}

/// Encode one Python column to FITS bytes.
fn encode_column(arr: &Bound<'_, PyAny>) -> PyResult<ColumnEncoding> {
    if let Some(enc) = encode_array_column(arr)? {
        return Ok(enc);
    }

    // Variable-length f64 column: list[list[float]] / list[ndarray].
    // Detected before list[str] / list[complex] because nested lists
    // do not extract as those scalar types.
    if let Ok(rows) = arr.extract::<Vec<Vec<f64>>>() {
        let n = rows.len();
        let mut payloads = Vec::with_capacity(n);
        for row in rows {
            let mut p = Vec::with_capacity(row.len() * 8);
            for v in row {
                p.extend_from_slice(&v.to_bits().to_be_bytes());
            }
            payloads.push(p);
        }
        return Ok(ColumnEncoding::Vla {
            descriptor: BinFieldKind::P,
            element: BinFieldKind::F64,
            payloads,
            n_rows: n,
        });
    }

    // String column: list[str] -> nA, repeat = max length, padded
    // right with spaces.
    if let Ok(strings) = arr.extract::<Vec<String>>() {
        validate_fits_ascii(&strings)?;
        let n = strings.len();
        let max = strings.iter().map(String::len).max().unwrap_or(1).max(1);
        let mut bytes = vec![b' '; n * max];
        for (r, s) in strings.iter().enumerate() {
            let dst = &mut bytes[r * max..r * max + s.len()];
            dst.copy_from_slice(s.as_bytes());
        }
        return Ok(ColumnEncoding::Fixed {
            kind: BinFieldKind::Char,
            repeat: max,
            row_bytes: bytes,
            n_rows: n,
            tdim: None,
        });
    }

    // String-array column: list[list[str]] (or a 2-D unicode array) ->
    // `nA` plus a `TDIMn` of `(width, count)`. Sec.7.3.3.2: the first
    // TDIM axis is each string's width and the rest are the array
    // shape. Checked after the flat `list[str]` case, which a nested
    // list cannot match.
    if let Ok(rows) = arr.extract::<Vec<Vec<String>>>() {
        validate_fits_ascii(rows.iter().flatten())?;
        let n = rows.len();
        let per_row = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|r| r.len() != per_row) {
            return Err(PyValueError::new_err(
                "string-array column: every row must hold the same number of strings",
            ));
        }
        let width = rows
            .iter()
            .flatten()
            .map(String::len)
            .max()
            .unwrap_or(1)
            .max(1);
        let repeat = width * per_row;
        let mut bytes = vec![b' '; n * repeat];
        for (r, row) in rows.iter().enumerate() {
            for (j, s) in row.iter().enumerate() {
                let off = r * repeat + j * width;
                bytes[off..off + s.len()].copy_from_slice(s.as_bytes());
            }
        }
        return Ok(ColumnEncoding::Fixed {
            kind: BinFieldKind::Char,
            repeat,
            row_bytes: bytes,
            n_rows: n,
            tdim: Some(format!("({width},{per_row})")),
        });
    }

    // Complex column: list[complex] -> 1M (C128), one (re, im) pair
    // per row in big-endian f64s.
    if let Ok(list) = arr.cast::<PyList>() {
        // Check the first non-None element is a Python complex.
        if let Some(first) = list.iter().next()
            && first.cast::<pyo3::types::PyComplex>().is_ok()
        {
            let n = list.len();
            let mut bytes = Vec::with_capacity(n * 16);
            for item in list.iter() {
                let c = item.cast::<pyo3::types::PyComplex>().map_err(|_| {
                    PyValueError::new_err(
                        "complex column: expected list[complex] (mixed types found)",
                    )
                })?;
                bytes.extend_from_slice(&c.real().to_bits().to_be_bytes());
                bytes.extend_from_slice(&c.imag().to_bits().to_be_bytes());
            }
            return Ok(ColumnEncoding::Fixed {
                kind: BinFieldKind::C128,
                repeat: 1,
                row_bytes: bytes,
                n_rows: n,
                tdim: None,
            });
        }
    }

    // Last resort: a flat sequence of numbers (`[1, 2, 3]`, a tuple,
    // a `range`). This runs *after* every arm above, so the
    // established meanings of nested numeric lists (VLA), `list[str]`
    // and `list[complex]` are untouched -- only input that used to
    // raise reaches here, and only that input pays for the
    // conversion. `ascii_table` has always accepted plain sequences;
    // this closes the gap between the two.
    if let Ok(coerced) = super::as_native_ndarray(arr, "bintable")
        && let Some(enc) = encode_array_column(coerced.as_any())?
    {
        return Ok(enc);
    }

    Err(PyTypeError::new_err(
        "bintable: unsupported column type (use bool/u8/i16/i32/i64/f32/f64 numpy arrays \
         or sequences of numbers, list[str], list[complex], or list[list[float]] for VLA)",
    ))
}

/// Extract the `(Header, data)` pair from a builder object, so
/// `write()` can accept a heterogeneous list of HDU builders without
/// committing to a single Python class hierarchy.
///
/// # Errors
///
/// Returns [`PyTypeError`] if `item` is none of [`PyImageBuilder`],
/// [`PyBinTableBuilder`] or [`PyAsciiTableBuilder`].
fn extract_built(item: Bound<'_, PyAny>) -> PyResult<(Header, Vec<u8>)> {
    if let Ok(b) = item.extract::<PyRef<'_, PyImageBuilder>>() {
        return Ok((b.header.clone(), b.data.clone()));
    }
    if let Ok(b) = item.extract::<PyRef<'_, PyBinTableBuilder>>() {
        return Ok((b.header.clone(), b.data.clone()));
    }
    if let Ok(b) = item.extract::<PyRef<'_, PyAsciiTableBuilder>>() {
        return Ok((b.header.clone(), b.data.clone()));
    }
    Err(PyTypeError::new_err(
        "write: list items must be ImageBuilder, BinTableBuilder, or AsciiTableBuilder",
    ))
}

/// Write a sequence of HDU builders to disk.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///   Destination path.
/// hdus : list
///   Builders returned by :func:`image`, :func:`bintable`,
///   :func:`ascii_table` or :func:`compressed_image`. When the first
///   item is an image builder, it becomes the primary HDU, and it
///   must have been built with ``primary=True``. When the first item
///   is any other builder, fitsy writes an empty primary HDU before
///   it. A table-only file thus needs no placeholder image.
/// overwrite : bool, optional
///   Default ``False``, which raises :class:`fitsy.FitsError`
///   instead of truncating an existing file at `path`. ``True``
///   truncates and overwrites it.
/// checksums : bool, optional
///   Default ``False``. ``True`` computes and stamps ``CHECKSUM``
///   and ``DATASUM`` cards on every emitted HDU (FITS Checksum
///   Proposal).
///
/// Raises
/// ------
/// ValueError
///   If `hdus` is empty.
/// TypeError
///   If an item of `hdus` is not an :class:`ImageBuilder`,
///   :class:`BinTableBuilder` or :class:`AsciiTableBuilder`.
/// FitsError
///   If `path` cannot be opened for writing, for example because it
///   already exists and `overwrite` is ``False``. Also raised if an
///   HDU built with the wrong `primary` value reaches the writer --
///   a non-first image HDU built with ``primary=True``, or a first
///   image HDU built with ``primary=False``.
///
/// Notes
/// -----
/// When `checksums` is ``False``, fitsy writes a ``CHECKSUM`` or
/// ``DATASUM`` card already present in a builder's header verbatim,
/// unchanged.
///
/// Examples
/// --------
/// >>> import numpy as np, fitsy
/// >>> fitsy.write("out.fits", [
/// ...     fitsy.image(np.zeros((10, 10), dtype=np.float32)),
/// ... ])
#[pyfunction]
#[pyo3(signature = (path, hdus, overwrite=false, *, checksums=false))]
pub fn write(
    path: PathBuf,
    hdus: Bound<'_, PyList>,
    overwrite: bool,
    checksums: bool,
) -> PyResult<()> {
    if hdus.is_empty() {
        return Err(PyValueError::new_err(
            "fitsy.write: refusing to write a file with zero HDUs",
        ));
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true);
    if overwrite {
        opts.truncate(true);
    } else {
        opts.create_new(true);
    }
    let file = opts
        .open(&path)
        .map_err(|e| super::err_to_py(crate::error::FitsError::Io(e)))?;
    let mut w = FitsWriter::new(BufWriter::new(file));
    if checksums {
        w = w.with_checksums();
    }
    let mut emitted_primary = false;
    // If the caller's first HDU isn't an image builder, prepend an
    // empty primary so the output is a valid FITS file.
    if let Some(first) = hdus.iter().next()
        && first.extract::<PyRef<'_, PyImageBuilder>>().is_err()
    {
        let (h, d) = empty_primary_image();
        w.write_hdu(&h, &d).into_py_result()?;
        emitted_primary = true;
    }
    for item in hdus.iter() {
        let (mut h, data) = extract_built(item)?;
        if !emitted_primary {
            // The first emitted HDU must declare SIMPLE. An image
            // builder built with `primary=True` already does; see
            // `promote_to_primary` for why nothing else is done here.
            promote_to_primary(&mut h);
            emitted_primary = true;
        }
        w.write_hdu(&h, &data).into_py_result()?;
    }
    w.finish()
        .map_err(|e| super::err_to_py(crate::error::FitsError::Io(e)))?;
    Ok(())
}

/// Build a zero-axis primary HDU with no data, for `write()` to emit
/// when the caller's first HDU is not an image builder.
fn empty_primary_image() -> (Header, Vec<u8>) {
    use crate::Value;
    let mut h = Header::empty();
    let _ = h.set("SIMPLE", Value::Logical(true), Some("conforming FITS"));
    let _ = h.set("BITPIX", Value::Integer(8), None);
    let _ = h.set("NAXIS", Value::Integer(0), None);
    let _ = h.set("EXTEND", Value::Logical(true), None);
    (h, Vec::new())
}

/// No-op placeholder for the first emitted HDU's header.
///
/// An image builder built with `primary=True` already carries
/// `SIMPLE = T`; this function does nothing to it. A bintable or
/// ascii-table builder is never passed here: `write()` only reaches
/// this call when the first item is an image builder, and prepends
/// [`empty_primary_image`] otherwise. Rewriting a table header's
/// `XTENSION` into `SIMPLE` would still leave it missing `EXTEND`
/// and produce an invalid primary HDU, so this function does not
/// attempt it.
fn promote_to_primary(h: &mut Header) {
    let _ = h;
}

#[pymethods]
impl PyImageBuilder {
    fn __repr__(&self) -> String {
        format!("ImageBuilder(<{} bytes>)", self.data.len())
    }
}

#[pymethods]
impl PyBinTableBuilder {
    fn __repr__(&self) -> String {
        format!("BinTableBuilder(<{} bytes>)", self.data.len())
    }
}

#[pymethods]
impl PyAsciiTableBuilder {
    fn __repr__(&self) -> String {
        format!("AsciiTableBuilder(<{} bytes>)", self.data.len())
    }
}

/// Build an ASCII ``TABLE`` HDU from a column dictionary.
///
/// Parameters
/// ----------
/// columns : dict[str, sequence]
///   One entry per column. All columns must share the same row
///   count. Supported value kinds:
///
///   - ``list[str]`` -> ``A{maxlen}``
///   - ``list[int]``, ``list[Optional[int]]``, or a numpy integer
///     array -> ``I{w}`` by default. A ``None`` cell needs a
///     matching entry in `tnulls`.
///   - ``list[float]`` or a numpy float array -> ``E{w}.{d}`` by
///     default
/// formats : dict[str, str], optional
///   Per-column override for the auto-chosen ``TFORM`` code, such
///   as ``{"flux": "F10.3"}``. The format kind must match the
///   column's value kind: ``I`` for an integer column, ``F``/``E``/
///   ``D`` for a float column, ``A`` for a string column. Default
///   ``None``, which auto-chooses every column's format.
/// tnulls : dict[str, str], optional
///   ``TNULL`` sentinel string for a numeric column that holds an
///   undefined cell (``None`` for an integer column, ``nan`` for a
///   float column). Default ``None``, which sets no ``TNULL``.
/// units : dict[str, str], optional
///   Per-column ``TUNIT`` strings. Default ``None``, which adds no
///   ``TUNIT`` cards.
/// extname : str, optional
///   ``EXTNAME`` keyword. Default ``None``, which adds no
///   ``EXTNAME`` card.
///
/// Returns
/// -------
/// AsciiTableBuilder
///   Pass to :func:`write`.
///
/// Raises
/// ------
/// TypeError
///   If a column's values do not match ``list[str]``,
///   ``list[Optional[int]]``, or ``list[float]``, or if `formats`
///   gives a format kind that does not match the column's value
///   kind.
/// ValueError
///   If a string cell holds a byte outside ASCII 32-126 (Standard
///   Sec.7.2.5).
/// FitsError
///   If `formats` holds a string that is not a valid ``TFORM``
///   code, if the columns disagree on row count, if a rendered
///   cell or a ``TNULL`` sentinel does not fit the column's field
///   width, or if a numeric column holds an undefined cell with no
///   matching entry in `tnulls`.
#[pyfunction]
#[pyo3(signature = (columns, formats=None, tnulls=None, units=None, extname=None))]
pub fn ascii_table(
    py: Python<'_>,
    columns: Bound<'_, PyDict>,
    formats: Option<Bound<'_, PyDict>>,
    tnulls: Option<Bound<'_, PyDict>>,
    units: Option<Bound<'_, PyDict>>,
    extname: Option<&str>,
) -> PyResult<PyAsciiTableBuilder> {
    let _ = py;
    let mut bt = AsciiTableBuilder::new();
    if let Some(name) = extname {
        bt.extname(name);
    }
    for (k, v) in columns.iter() {
        let name: String = k.extract()?;
        let fmt_override: Option<String> = match formats.as_ref() {
            Some(d) => d.get_item(&name)?.map(|x| x.extract()).transpose()?,
            None => None,
        };
        let unit: Option<String> = match units.as_ref() {
            Some(d) => d.get_item(&name)?.map(|x| x.extract()).transpose()?,
            None => None,
        };
        let tnull: Option<String> = match tnulls.as_ref() {
            Some(d) => d.get_item(&name)?.map(|x| x.extract()).transpose()?,
            None => None,
        };
        let (data, format) = extract_ascii_column(&v, fmt_override.as_deref(), &name)?;
        bt.add_column(name.clone(), format, data).into_py_result()?;
        if let Some(u) = unit {
            bt.unit(u).into_py_result()?;
        }
        if let Some(tn) = tnull {
            bt.tnull(tn).into_py_result()?;
        }
    }
    let (h, data) = bt.build().into_py_result()?;
    Ok(PyAsciiTableBuilder { header: h, data })
}

/// Parse a `TFORM` code such as `"F10.3"` into an [`AsciiFormat`].
///
/// # Errors
///
/// Returns `fitsy.FitsError` if `s` is not a valid ASCII-table
/// `TFORM` code.
fn parse_ascii_format(s: &str) -> PyResult<AsciiFormat> {
    AsciiFormat::parse(s).map_err(super::err_to_py)
}

/// Infer one ASCII-table column's data and format from a Python
/// sequence. Tries a string column, then a nullable-integer column,
/// then a float column, in that order; `name` only labels an error.
///
/// # Errors
///
/// Returns [`PyTypeError`] if `arr` matches none of the three kinds,
/// or if `fmt_override` parses to a format kind that does not match
/// the inferred kind. Returns [`PyValueError`] if a string cell
/// holds a byte outside ASCII 32-126. Returns the error from
/// [`parse_ascii_format`] if `fmt_override` does not parse.
fn extract_ascii_column(
    arr: &Bound<'_, PyAny>,
    fmt_override: Option<&str>,
    name: &str,
) -> PyResult<(AsciiColumnData, AsciiFormat)> {
    // String column? Try list[str] / object array first.
    if let Ok(list) = arr.extract::<Vec<Option<String>>>() {
        // None entries are treated as empty string for A-columns.
        let strings: Vec<String> = list.into_iter().map(Option::unwrap_or_default).collect();
        validate_fits_ascii(&strings)?;
        let format = if let Some(s) = fmt_override {
            parse_ascii_format(s)?
        } else {
            let max = strings.iter().map(String::len).max().unwrap_or(1).max(1);
            AsciiFormat::A(max)
        };
        if !matches!(format, AsciiFormat::A(_)) {
            return Err(PyTypeError::new_err(format!(
                "ascii_table column {name:?}: string data needs an A format (got {format:?})"
            )));
        }
        return Ok((AsciiColumnData::Str(strings), format));
    }
    // Integer column? Try i64 first; nullable comes through as list.
    if let Ok(list) = arr.extract::<Vec<Option<i64>>>() {
        let format = if let Some(s) = fmt_override {
            parse_ascii_format(s)?
        } else {
            let w = list
                .iter()
                .filter_map(|x| x.as_ref())
                .map(|v| v.to_string().len())
                .max()
                .unwrap_or(1)
                .max(1);
            AsciiFormat::I(w + 1)
        };
        if !matches!(format, AsciiFormat::I(_)) {
            return Err(PyTypeError::new_err(format!(
                "ascii_table column {name:?}: integer data needs an I format (got {format:?})"
            )));
        }
        return Ok((AsciiColumnData::Int(list), format));
    }
    // Float column.
    if let Ok(list) = arr.extract::<Vec<f64>>() {
        let format = match fmt_override {
            Some(s) => parse_ascii_format(s)?,
            None => AsciiFormat::E(15, 7),
        };
        if !matches!(
            format,
            AsciiFormat::F(_, _) | AsciiFormat::E(_, _) | AsciiFormat::D(_, _)
        ) {
            return Err(PyTypeError::new_err(format!(
                "ascii_table column {name:?}: float data needs an F/E/D format (got {format:?})"
            )));
        }
        return Ok((AsciiColumnData::Float(list), format));
    }
    Err(PyTypeError::new_err(format!(
        "ascii_table column {name:?}: unsupported value type (use list[str], list[int|None], or list[float])"
    )))
}
