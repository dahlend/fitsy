//! Module-level convenience functions: `getdata`, `getheader`,
//! `getval`, `setval`, `delval`, `info` and `append`.
//!
//! Each function opens `path`, performs one operation, then closes
//! the file. `setval` and `delval` rewrite the file. `append` writes
//! past the last HDU and rewrites nothing. The rest only read.
//!
//! Each call thus pays the open and parse cost once. A caller that
//! performs more than one operation on one file should use
//! `file::open` instead.

use std::path::PathBuf;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::IntoPyResult;

/// Return an HDU's pixel or column data.
///
/// Reads the `data` attribute first. Falls back to `to_dict()`,
/// which [`super::table::PyBinTable`] and
/// [`super::table::PyAsciiTable`] define.
///
/// # Errors
///
/// Returns the `to_dict()` error when both calls fail.
/// [`super::hdu::PyRandomGroups`] defines neither name, so this
/// function fails on one with an `AttributeError`.
///
/// The fallback catches every `data` failure, not only a missing
/// attribute. A pixel read error on an [`super::hdu::PyImageHdu`]
/// thus surfaces as an `AttributeError` about `to_dict`.
fn data_of<'py>(hdu: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    hdu.getattr("data").or_else(|_| hdu.call_method0("to_dict"))
}

/// Open `path` and resolve one HDU.
///
/// Opens the file in `mode`, then indexes into the returned
/// [`super::file::PyFitsFile`] with `ext`. `ext` selects HDU 0 when
/// absent.
///
/// # Errors
///
/// Returns `fitsy.FitsError` if `path` cannot be opened or parsed.
/// Returns a Python `IndexError` if `ext` is an out-of-range
/// integer, a `KeyError` if `ext` is a string that names no HDU, or
/// a `TypeError` if `ext` is some other type.
fn open_and_get<'py>(
    py: Python<'py>,
    path: PathBuf,
    ext: Option<Bound<'py, PyAny>>,
    mode: &str,
) -> PyResult<(Py<PyAny>, Bound<'py, PyAny>)> {
    let file = super::file::open(py, path, mode, false)?;
    let file_obj: Py<PyAny> = Py::new(py, file)?.into_any();
    let ext_key: Bound<'py, PyAny> = match ext {
        Some(e) => e,
        None => 0_i64.into_pyobject(py)?.into_any().into_any(),
    };
    let hdu = file_obj.bind(py).get_item(&ext_key)?;
    Ok((file_obj, hdu))
}

/// Read one HDU's data, and optionally its header, from `path`.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///   File to read.
/// ext : int or str, optional
///   HDU index or ``EXTNAME``. Default is HDU 0. When `ext` is
///   omitted and HDU 0 carries no data, fitsy reads HDU 1 instead.
/// header : bool, keyword-only, optional
///   Default ``False``. When ``True``, return ``(data, header)``
///   instead of only ``data``.
///
/// Returns
/// -------
/// numpy.ndarray or tuple
///   Pixel data for an image HDU. A read-only structured array, one
///   record per row, for a binary or ASCII table HDU. When `header`
///   is ``True``, returns ``(data, header)`` instead, paired with
///   the :class:`Header` of the HDU the data came from.
///
/// Raises
/// ------
/// FitsError
///   If `path` cannot be opened or parsed as FITS.
/// IndexError
///   If `ext` is an out-of-range integer. Also raised if the
///   selected HDU has no data, and, when `ext` was omitted, HDU 1
///   has no data either.
/// KeyError
///   If `ext` is a string that names no HDU.
/// TypeError
///   If `ext` is neither an int, a str, nor omitted.
///
/// Notes
/// -----
/// A random-groups primary HDU (``GROUPS = T``) has neither a data
/// array nor a dict form. Reading one with `ext` omitted raises
/// ``AttributeError`` instead of the exceptions above.
#[pyfunction]
#[pyo3(signature = (path, ext=None, *, header=false))]
pub fn getdata(
    py: Python<'_>,
    path: PathBuf,
    ext: Option<Bound<'_, PyAny>>,
    header: bool,
) -> PyResult<Py<PyAny>> {
    // Whether the caller pinned an HDU decides if the fallback below
    // is allowed: naming an empty HDU is an error, landing on one by
    // default is not. Captured before `ext` is consumed, because the
    // error below names the HDU the caller asked for -- which is the
    // only part of that message worth reading.
    let ext_given = match ext.as_ref() {
        Some(e) => Some(e.str()?.to_string()),
        None => None,
    };
    let (file, mut hdu) = open_and_get(py, path, ext, "readonly")?;
    let mut data = data_of(&hdu)?;
    if data.is_none() {
        if let Some(label) = ext_given {
            return Err(PyIndexError::new_err(format!("No data in HDU #{label}.")));
        }
        let bound = file.bind(py);
        if bound.len()? == 1 {
            return Err(PyIndexError::new_err(
                "No data in Primary HDU and no extension HDU found.",
            ));
        }
        hdu = bound.get_item(1_i64.into_pyobject(py)?.into_any())?;
        data = data_of(&hdu)?;
        if data.is_none() {
            return Err(PyIndexError::new_err(
                "No data in either Primary or first extension HDUs.",
            ));
        }
    }
    if header {
        let hdr = hdu.getattr("header")?;
        Ok(PyTuple::new(py, [data.unbind(), hdr.unbind()])?
            .into_any()
            .unbind())
    } else {
        Ok(data.unbind())
    }
}

/// Read one HDU's header from `path`.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///   File to read.
/// ext : int or str, optional
///   HDU index or ``EXTNAME``. Default is HDU 0.
///
/// Returns
/// -------
/// Header
///   Header of the selected HDU.
///
/// Raises
/// ------
/// FitsError
///   If `path` cannot be opened or parsed as FITS.
/// IndexError
///   If `ext` is an out-of-range integer.
/// KeyError
///   If `ext` is a string that names no HDU.
/// TypeError
///   If `ext` is neither an int, a str, nor omitted.
#[pyfunction]
#[pyo3(signature = (path, ext=None))]
pub fn getheader(
    py: Python<'_>,
    path: PathBuf,
    ext: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (_file, hdu) = open_and_get(py, path, ext, "readonly")?;
    Ok(hdu.getattr("header")?.unbind())
}

/// Read one header keyword from `path`.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///   File to read.
/// key : str
///   Header keyword to read.
/// ext : int or str, optional
///   HDU index or ``EXTNAME``. Default is HDU 0.
///
/// Returns
/// -------
/// bool, int, float, complex, str, or None
///   Value of the card. A :class:`HeaderCommentary` object for a
///   ``COMMENT``, ``HISTORY``, or blank keyword.
///
/// Raises
/// ------
/// FitsError
///   If `path` cannot be opened or parsed as FITS.
/// IndexError
///   If `ext` is an out-of-range integer.
/// KeyError
///   If `ext` is a string that names no HDU, or if `key` is absent
///   from the selected header.
/// TypeError
///   If `ext` is neither an int, a str, nor omitted.
#[pyfunction]
#[pyo3(signature = (path, key, ext=None))]
pub fn getval(
    py: Python<'_>,
    path: PathBuf,
    key: &str,
    ext: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let header = getheader(py, path, ext)?;
    let bound = header.bind(py);
    if !bound.contains(key)? {
        return Err(PyKeyError::new_err(format!("no header card {key:?}")));
    }
    Ok(bound.get_item(key)?.unbind())
}

/// Set one header keyword in `path`. Rewrites the file.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///   File to edit.
/// key : str
///   Header keyword to set.
/// value : bool, int, float, complex, str, or None, optional
///   New card value. Default ``None``, which writes a card with an
///   undefined value.
/// ext : int or str, keyword-only, optional
///   HDU index or ``EXTNAME``. Default is HDU 0.
/// comment : str, keyword-only, optional
///   New card comment. Default ``None``, which leaves an existing
///   card's comment unchanged.
///
/// Raises
/// ------
/// FitsError
///   If `path` cannot be opened or parsed as FITS, if `key` is not
///   a valid FITS keyword, or if the rewrite fails.
/// IndexError
///   If `ext` is an out-of-range integer.
/// KeyError
///   If `ext` is a string that names no HDU.
/// TypeError
///   If `ext` is neither an int, a str, nor omitted. Also raised if
///   `value` is a type fitsy cannot store in a header card.
/// ValueError
///   If `key` names a structural card managed by the writer, such
///   as ``BITPIX`` or ``NAXIS``.
#[pyfunction]
#[pyo3(signature = (path, key, value=None, *, ext=None, comment=None))]
pub fn setval(
    py: Python<'_>,
    path: PathBuf,
    key: &str,
    value: Option<Bound<'_, PyAny>>,
    ext: Option<Bound<'_, PyAny>>,
    comment: Option<&str>,
) -> PyResult<()> {
    let (file, hdu) = open_and_get(py, path.clone(), ext, "update")?;
    let header = hdu.getattr("header")?;
    let value: Py<PyAny> = match value {
        Some(v) => v.unbind(),
        None => py.None(),
    };
    let assign: Py<PyAny> = if let Some(c) = comment {
        PyTuple::new(py, [value, c.into_pyobject(py)?.into_any().unbind()])?
            .into_any()
            .unbind()
    } else {
        value
    };
    header.set_item(key, assign)?;
    file.bind(py).call_method0("flush").map(|_| ())
}

/// Remove one header keyword from `path`. Rewrites the file.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///   File to edit.
/// key : str
///   Header keyword to remove.
/// ext : int or str, keyword-only, optional
///   HDU index or ``EXTNAME``. Default is HDU 0.
///
/// Raises
/// ------
/// FitsError
///   If `path` cannot be opened or parsed as FITS, or if the
///   rewrite fails.
/// IndexError
///   If `ext` is an out-of-range integer.
/// KeyError
///   If `ext` is a string that names no HDU, or if `key` is absent
///   from the selected header. Call :func:`getval`, or test
///   ``key in header`` on an open file, to guard an optional card.
/// TypeError
///   If `ext` is neither an int, a str, nor omitted.
/// ValueError
///   If `key` names a structural card managed by the writer, such
///   as ``BITPIX`` or ``NAXIS``.
#[pyfunction]
#[pyo3(signature = (path, key, *, ext=None))]
pub fn delval(
    py: Python<'_>,
    path: PathBuf,
    key: &str,
    ext: Option<Bound<'_, PyAny>>,
) -> PyResult<()> {
    let (file, hdu) = open_and_get(py, path.clone(), ext, "update")?;
    let header = hdu.getattr("header")?;
    // Deliberately unguarded: `__delitem__` raises KeyError for a
    // missing card, which is the behavior callers expect. Silently
    // succeeding hid typo'd keywords.
    header.del_item(key)?;
    file.bind(py).call_method0("flush").map(|_| ())
}

/// Return a brief HDU summary table for `path`.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///   File to read.
///
/// Returns
/// -------
/// list of tuple
///   One ``(index, name, ver, kind, dims_or_n_rows)`` tuple per HDU,
///   in file order.
///
///   * ``index`` -- 0-based HDU position (``int``).
///   * ``name`` -- ``EXTNAME``, or an empty string if absent
///     (``str``).
///   * ``ver`` -- ``EXTVER``, or ``1`` if absent (``int``).
///   * ``kind`` -- wrapper class name: ``"ImageHdu"``,
///     ``"BinTable"``, ``"AsciiTable"``, or ``"RandomGroups"``.
///     ``"Unknown"`` in the unexpected case where fitsy cannot read
///     the HDU's Python type name.
///   * ``dims_or_n_rows`` -- axis lengths (``list`` of ``int``) for
///     an image HDU, row count (``int``) for a table HDU, or
///     ``None`` for a random-groups HDU.
///
/// Raises
/// ------
/// FitsError
///   If `path` cannot be opened or parsed as FITS.
#[pyfunction]
#[pyo3(signature = (path))]
pub fn info(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyList>> {
    let file = super::file::open(py, path, "readonly", false)?;
    let file_obj: Py<PyAny> = Py::new(py, file)?.into_any();
    let bound = file_obj.bind(py);
    let n: usize = bound.len()?;
    let out = PyList::empty(py);
    for i in 0..n {
        let i_obj = (i as i64).into_pyobject(py)?.into_any();
        let hdu = bound.get_item(&i_obj)?;
        let kind = hdu
            .get_type()
            .name()
            .map_or_else(|_| "Unknown".to_string(), |s| s.to_string());
        let header = hdu.getattr("header")?;
        let name: String = header
            .call_method1("get", ("EXTNAME", ""))?
            .extract()
            .unwrap_or_default();
        let ver: i64 = header
            .call_method1("get", ("EXTVER", 1))?
            .extract()
            .unwrap_or(1);
        let dims: Py<PyAny> = if let Ok(axes) = hdu.getattr("axes") {
            axes.unbind()
        } else if let Ok(n_rows) = hdu.getattr("n_rows") {
            n_rows.unbind()
        } else {
            py.None()
        };
        let tup = PyTuple::new(
            py,
            [
                (i as i64).into_pyobject(py)?.into_any().unbind(),
                name.into_pyobject(py)?.into_any().unbind(),
                ver.into_pyobject(py)?.into_any().unbind(),
                kind.into_pyobject(py)?.into_any().unbind(),
                dims,
            ],
        )?;
        out.append(tup)?;
    }
    Ok(out.unbind())
}

/// Append one image HDU to an existing FITS file.
///
/// Writes the new HDU directly after the last existing HDU. fitsy
/// rewrites no existing HDU.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///   FITS file to append to. The file must already exist and must
///   parse as FITS.
/// data : array-like
///   Image pixels for the new HDU. A numpy array of dtype
///   ``bool``, ``int8``, ``uint8``, ``int16``, ``uint16``,
///   ``int32``, ``uint32``, ``int64``, ``uint64``, ``float32`` or
///   ``float64``, or anything :func:`numpy.asarray` accepts.
/// header : Header or mapping, optional
///   Extra header cards for the new HDU. Default ``None``.
///
/// Raises
/// ------
/// FitsError
///   If `path` cannot be opened for append, or the write fails.
/// TypeError
///   If `data` is not an image array or array-like, or its dtype is
///   not one fitsy supports.
///
/// Notes
/// -----
/// fitsy writes the new HDU as an extension, with ``XTENSION =
/// 'IMAGE'``. fitsy parses the whole file first, to check it and to
/// find the offset of the append. Any bytes after the last HDU are
/// overwritten.
///
/// fitsy builds the new HDU with :func:`fitsy.image`. See that
/// function for the dtype rule, and for the ``BZERO`` and
/// ``BSCALE`` cards it adds for an unsigned dtype.
#[pyfunction]
#[pyo3(signature = (path, data, header=None))]
pub fn append(
    py: Python<'_>,
    path: PathBuf,
    data: Bound<'_, PyAny>,
    header: Option<Bound<'_, PyAny>>,
) -> PyResult<()> {
    // Build an extension-image HDU (primary=false ensures
    // XTENSION='IMAGE' and avoids SIMPLE=T being emitted).
    let builder = super::writer::image(data, header, false).map_err(|e| {
        PyTypeError::new_err(format!(
            "fitsy.append: data must be an image array or array-like ({e})"
        ))
    })?;

    // Release the GIL while doing the actual file I/O.
    py.detach(|| -> crate::error::Result<()> {
        let mut app = crate::FitsAppender::open(&path)?;
        app.append_hdu_parts(&builder.header, &builder.data)?;
        app.finish()?;
        Ok(())
    })
    .into_py_result()?;
    Ok(())
}
