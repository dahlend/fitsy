//! Module-level convenience functions matching `astropy.io.fits`:
//! `getdata`, `getheader`, `getval`, `setval`, `delval`, `info`, `append`.
//!
//! These mirror the astropy module API. They open the file, perform
//! one operation, and close. For repeated access prefer
//! ``with fitsy.open(...) as f``.

use std::path::PathBuf;

use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::IntoPyResult;

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
#[pyfunction]
#[pyo3(signature = (path, ext=None, *, header=false))]
pub fn getdata(
    py: Python<'_>,
    path: PathBuf,
    ext: Option<Bound<'_, PyAny>>,
    header: bool,
) -> PyResult<Py<PyAny>> {
    let (_file, hdu) = open_and_get(py, path, ext, "readonly")?;
    let data = hdu
        .getattr("data")
        .or_else(|_| hdu.call_method0("to_dict"))?;
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
#[pyfunction]
#[pyo3(signature = (path, key, value, *, ext=None, comment=None))]
pub fn setval(
    py: Python<'_>,
    path: PathBuf,
    key: &str,
    value: Bound<'_, PyAny>,
    ext: Option<Bound<'_, PyAny>>,
    comment: Option<&str>,
) -> PyResult<()> {
    let (file, hdu) = open_and_get(py, path.clone(), ext, "update")?;
    let header = hdu.getattr("header")?;
    let assign: Py<PyAny> = if let Some(c) = comment {
        PyTuple::new(
            py,
            [value.unbind(), c.into_pyobject(py)?.into_any().unbind()],
        )?
        .into_any()
        .unbind()
    } else {
        value.unbind()
    };
    header.set_item(key, assign)?;
    file.bind(py).call_method0("flush").map(|_| ())
}

/// Remove one header keyword from `path` (rewrites the file).
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
    if header.contains(key)? {
        header.del_item(key)?;
    }
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
    header: Option<Bound<'_, PyDict>>,
) -> PyResult<()> {
    // Build an extension-image HDU (primary=false ensures
    // XTENSION='IMAGE' and avoids SIMPLE=T being emitted).
    let builder = super::writer::image(py, data, header, false).map_err(|e| {
        PyTypeError::new_err(format!(
            "fitsy.append: data must be a numpy image array ({e})"
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
