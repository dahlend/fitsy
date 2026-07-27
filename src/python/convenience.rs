//! Module-level convenience functions matching `astropy.io.fits`:
//! `getdata`, `getheader`, `getval`, `setval`, `delval`, `info`, `append`.
//!
//! These mirror the astropy module API. They open the file, perform
//! one operation, and close. For repeated access prefer
//! ``with fitsy.open(...) as f``.

use std::path::PathBuf;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::IntoPyResult;

/// An HDU's data, falling back to its dict form for the kinds that
/// expose no `data` attribute.
fn data_of<'py>(hdu: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    hdu.getattr("data").or_else(|_| hdu.call_method0("to_dict"))
}

/// Open `path` and return one HDU resolved by `ext` (int or `EXTNAME` str).
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

/// Read one HDU's data (and optionally its header) from `path`.
///
/// Parameters
/// ----------
/// path : str | os.PathLike
///   File to read.
/// ext : int | str, optional
///   HDU index or ``EXTNAME``. When omitted, HDU 0 is read; if the
///   primary HDU carries no data the first extension is used
///   instead, matching ``astropy.io.fits.getdata``. The overwhelmingly
///   common layout -- an empty primary followed by the real data in
///   HDU 1 -- therefore works without naming an extension.
/// header : bool, keyword-only
///   When True, return ``(data, header)`` for whichever HDU the
///   data actually came from.
///
/// Raises
/// ------
/// IndexError
///   If the selected HDU has no data (and, when `ext` was omitted,
///   neither does the first extension).
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
/// Raises `KeyError` if the keyword is absent.
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

/// Set one header keyword in `path` (rewrites the file).
///
/// `value` defaults to `None`, which writes a card with an undefined
/// value -- the same default `astropy.io.fits.setval` uses.
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

/// Remove one header keyword from `path` (rewrites the file).
///
/// Raises
/// ------
/// KeyError
///   If the keyword is absent, matching `astropy.io.fits.delval`
///   and `fitsy.getval`. Guard with `getval` (or open the file and
///   test `key in header`) when the card is optional.
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
    // missing card, which is the behaviour callers expect. Silently
    // succeeding hid typo'd keywords.
    header.del_item(key)?;
    file.bind(py).call_method0("flush").map(|_| ())
}

/// Return a brief HDU summary table for `path`.
///
/// Returns a list of `(index, name, ver, kind, dims_or_n_rows)` tuples.
/// `kind` is the wrapper class name (`"ImageHdu"`, `"BinTable"`, ...).
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
/// The HDU is streamed in place at the end of the file -- existing
/// HDUs are not read or rewritten. The Python data array is
/// converted to a non-primary (`XTENSION = 'IMAGE   '`) HDU before
/// being written.  Matches `astropy.io.fits.append` semantics.
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
        app.append_hdu(&builder.header, &builder.data)?;
        app.finish()?;
        Ok(())
    })
    .into_py_result()?;
    Ok(())
}
