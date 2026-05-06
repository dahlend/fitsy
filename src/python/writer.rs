//! Writer-side wrappers: ``image()``, ``bintable()``, ``ascii_table()``,
//! ``write()``.
//!
//! The Python writing API is intentionally narrow. Users construct
//! HDU specifications via :func:`fitsy.image`,
//! :func:`fitsy.bintable`, or :func:`fitsy.ascii_table`, then hand
//! the list to :func:`fitsy.write`. Anything more elaborate (custom
//! TFORM, VLA columns, scaled storage) should drop down to Rust.

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
use super::header::header_from_dict;

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

/// Build an image HDU from a numpy array.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Image pixels (any supported FITS dtype: ``u8``, ``i16``,
///     ``i32``, ``i64``, ``f32``, ``f64``). The returned HDU's
///     NAXIS list is the *reverse* of ``data.shape`` (numpy is
///     row-major while FITS is fastest-axis-first).
/// header : dict, optional
///     Extra header cards to merge in. Values may be scalars or
///     ``(value, comment)`` tuples.
/// primary : bool, optional
///     If True (default) and this is the first HDU written, mark it
///     as the primary HDU. Subsequent calls become image extensions.
///
/// Returns
/// -------
/// ImageBuilder
///     Pass to :func:`write`.
#[pyfunction]
#[pyo3(signature = (data, header=None, primary=true))]
pub fn image(
    py: Python<'_>,
    data: Bound<'_, PyAny>,
    header: Option<Bound<'_, PyDict>>,
    primary: bool,
) -> PyResult<PyImageBuilder> {
    // numpy axes are row-major (slowest first); FITS NAXIS is
    // fastest first. Reverse before handing to ImageBuilder.
    let extra = match header.as_ref() {
        Some(d) => header_from_dict(d)?,
        None => Header::empty(),
    };
    let (h, bytes) = build_image(py, &data, primary, extra)?;
    Ok(PyImageBuilder {
        header: h,
        data: bytes,
    })
}

/// Tile-compress a numpy array into a `BINTABLE` HDU (`ZIMAGE`).
///
/// The result is identical in structure to a CFITSIO/`fpack`-produced
/// compressed image extension: a BINTABLE with `ZIMAGE = T` whose
/// rows hold one compressed tile each.  Astropy and `funpack` will
/// transparently decompress it; ``fitsy.open`` will too.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     Image pixels (any supported FITS dtype).
/// header : dict, optional
///     Extra cards merged into the synthesized image header before
///     compression.  Structural keywords (``BITPIX``, ``NAXIS*``,
///     etc.) are ignored.
/// tile_shape : sequence[int], optional
///     Tile shape in **FITS axis order** (`tile[0]` = `NAXIS1`
///     direction).  Length must equal `data.ndim`.  Default is
///     ``(NAXIS1, 1, 1, ...)`` per Pence & Seaman Sec.3 -- one row per
///     tile, which is the convention `fpack` uses.
/// extname : str, optional
///     `EXTNAME` keyword on the resulting BINTABLE.  Default
///     ``"COMPRESSED_IMAGE"``.
///
/// Notes
/// -----
/// The current Rust core only emits ``ZCMPTYPE = 'GZIP_1'``
/// compressed tiles.  The output is fully standards compliant and
/// readable by every FITS library; if you need RICE/HCOMPRESS use
/// the Rust API directly or run ``fpack -r`` on the output.
#[pyfunction]
#[pyo3(signature = (data, header=None, *, tile_shape=None, extname=None))]
pub fn compressed_image(
    py: Python<'_>,
    data: Bound<'_, PyAny>,
    header: Option<Bound<'_, PyDict>>,
    tile_shape: Option<Vec<u64>>,
    extname: Option<String>,
) -> PyResult<PyBinTableBuilder> {
    use crate::Value;
    use crate::compression::compress_image_to_hdu;
    let extra = match header.as_ref() {
        Some(d) => header_from_dict(d)?,
        None => Header::empty(),
    };
    // Build the uncompressed image first so we get correct BITPIX
    // and big-endian raw bytes; then hand off to the Rust compressor.
    let (img_header, raw) = build_image(py, &data, false, extra)?;
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

pub(crate) fn build_image(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    primary: bool,
    extra: Header,
) -> PyResult<(Header, Vec<u8>)> {
    // Normalize to native byte order before type dispatch. PyO3/numpy's
    // `extract::<PyReadonlyArrayDyn<T>>()` only matches arrays whose
    // dtype already carries the native endian mark, so a `>f4` or `>i2`
    // passed in from an external FITS tool would otherwise fall through
    // all arms and raise an "unsupported dtype" error.
    //
    // `arr.dtype.newbyteorder('=')` gives the same kind/itemsize dtype
    // but with the platform's native byte order. `astype(..., copy=False)`
    // returns `arr` unchanged when it is already native-endian (no alloc)
    // and a byte-swapped copy otherwise.
    let native_dtype = arr.getattr("dtype")?.call_method1("newbyteorder", ("=",))?;
    let arr_owned;
    let arr: &Bound<'_, PyAny> = {
        let kwargs = PyDict::new(py);
        kwargs.set_item("copy", false)?;
        let native = arr.call_method("astype", (native_dtype,), Some(&kwargs))?;
        arr_owned = native;
        &arr_owned
    };

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
    // and emit BZERO=2^(N-1) (BSCALE=1) so a conforming reader (us
    // and astropy) returns the original unsigned values. Mirrors
    // the inverse path in `src/python/hdu.rs::read_image_array`.
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
        // 2^63 is not representable as i64; emit as a real-valued
        // BZERO card (matches astropy and the FITS convention).
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

/// Inject `BZERO`/`BSCALE` cards for the FITS unsigned-int (or
/// signed-byte) convention. User-supplied cards in `extra` win --
/// we only set defaults if the keyword is absent.
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
///     One entry per column. All columns must share the same
///     row count. Supported value kinds:
///
///     - numpy ``bool``/``u8``/``i16``/``i32``/``i64``/``f32``/``f64``
///       arrays (1-D, or 2-D for fixed-repeat columns)
///     - ``list[str]`` -> ``nA`` (right-padded to the longest string)
///     - ``list[complex]`` -> ``M`` (``C128``)
///     - ``list[list[float]]`` -> ``1PD`` variable-length column
///       (heap-stored, ``f64`` element type)
/// units : dict[str, str], optional
///     Per-column ``TUNITn`` strings. Keys must match column names
///     in ``columns``; entries for unknown columns are ignored.
/// extname : str, optional
///     Sets the ``EXTNAME`` keyword for this extension.
///
/// Returns
/// -------
/// BinTableBuilder
///     Pass to :func:`write`.
#[pyfunction]
#[pyo3(signature = (columns, units=None, extname=None))]
pub fn bintable(
    py: Python<'_>,
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
        let enc = encode_column(py, &v)
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
            ColumnEncoding::Fixed { kind, repeat, .. } => {
                bt.add_column(name.clone(), *kind, *repeat, unit.as_deref(), None)
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

/// Encode one Python column to FITS bytes.
fn encode_column(_py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<ColumnEncoding> {
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
                return Ok(ColumnEncoding::Fixed {
                    kind: $kind,
                    repeat,
                    row_bytes: bytes,
                    n_rows: n,
                });
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
        return Ok(ColumnEncoding::Fixed {
            kind: BinFieldKind::Logical,
            repeat,
            row_bytes: bytes,
            n_rows: n,
        });
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
            });
        }
    }

    Err(PyTypeError::new_err(
        "bintable: unsupported column type (use bool/u8/i16/i32/i64/f32/f64 numpy arrays, list[str], list[complex], or list[list[float]] for VLA)",
    ))
}

/// Trait used by `write()` to accept a heterogeneous list of HDU
/// builders without committing to a single Python class hierarchy.
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
/// path : str or pathlib.Path
///     Destination path. Overwritten if it already exists.
/// hdus : list
///     Builders returned by :func:`image`, :func:`bintable`, or
///     :func:`ascii_table`. The first item must be an image (it
///     becomes the primary HDU); for table-only files pass
///     ``fitsy.image(np.zeros((0,)))`` first.
/// overwrite : bool, optional
///     If False (the default), raise :class:`FileExistsError`
///     rather than truncating an existing file at ``path``.
/// checksums : bool, optional
///     If True, compute and stamp ``CHECKSUM`` and ``DATASUM``
///     cards on every emitted HDU (FITS Checksum Proposal).
///     Defaults to False.
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
            // First emitted HDU must declare SIMPLE; builders that
            // started life as extensions get their XTENSION promoted.
            promote_to_primary(&mut h);
            emitted_primary = true;
        }
        w.write_hdu(&h, &data).into_py_result()?;
    }
    w.finish()
        .map_err(|e| super::err_to_py(crate::error::FitsError::Io(e)))?;
    Ok(())
}

fn empty_primary_image() -> (Header, Vec<u8>) {
    use crate::Value;
    let mut h = Header::empty();
    let _ = h.set("SIMPLE", Value::Logical(true), Some("conforming FITS"));
    let _ = h.set("BITPIX", Value::Integer(8), None);
    let _ = h.set("NAXIS", Value::Integer(0), None);
    let _ = h.set("EXTEND", Value::Logical(true), None);
    (h, Vec::new())
}

fn promote_to_primary(h: &mut Header) {
    // Image builders already produce a SIMPLE-headed primary; this
    // is a no-op for them. We deliberately do not try to promote a
    // bintable/ascii_table to primary since that produces an invalid
    // FITS file.
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
///     One entry per column. Supported value kinds:
///
///     - ``list[int]`` or numpy int array -> ``I{w}`` (use ``None``
///       cells for ``TNULL``; combine with ``tnulls={col: "-9999"}``)
///     - ``list[float]`` or numpy float array -> ``E{w}.{d}`` by default
///     - ``list[str]`` -> ``A{maxlen}``
/// formats : dict[str, str], optional
///     Per-column override for the auto-chosen ``TFORM``.
/// tnulls : dict[str, str], optional
///     ``TNULL`` sentinel string for integer columns containing
///     ``None``.
/// units : dict[str, str], optional
///     Per-column ``TUNIT`` strings.
/// extname : str, optional
///     Sets the ``EXTNAME`` keyword.
///
/// Returns
/// -------
/// AsciiTableBuilder
///     Pass to :func:`write`.
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

fn parse_ascii_format(s: &str) -> PyResult<AsciiFormat> {
    AsciiFormat::parse(s).map_err(super::err_to_py)
}

fn extract_ascii_column(
    arr: &Bound<'_, PyAny>,
    fmt_override: Option<&str>,
    name: &str,
) -> PyResult<(AsciiColumnData, AsciiFormat)> {
    // String column? Try list[str] / object array first.
    if let Ok(list) = arr.extract::<Vec<Option<String>>>() {
        // None entries are treated as empty string for A-columns.
        let strings: Vec<String> = list.into_iter().map(Option::unwrap_or_default).collect();
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
