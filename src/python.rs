//! PyO3 bindings for fitsy.
//!
//! Built when the `python` feature is enabled.
//!
//! Design notes:
//! - Image pixel data is read with [`crate::ImageHdu::read_raw`] dispatched
//!   on `BITPIX`, then handed to numpy as an **owned** array in the
//!   machine's native byte order. FITS data is big-endian, so a copy
//!   is unavoidable on every modern host -- there is no zero-copy
//!   view to be had.
//! - Headers expose a dict-like Python interface returning native
//!   Python scalars (`int` / `float` / `str` / `bool` / `complex`).
//! - All `FitsError` variants map to a single custom Python exception
//!   `fitsy.FitsError` (subclass of `OSError` so `except OSError`
//!   still catches I/O issues).

// PyO3 macros expand to `unsafe` blocks; the crate-wide `deny(unsafe_code)`
// in lib.rs forbids them, so we opt out for this module only. Rust 2024
// also requires opting out of the `unsafe-op-in-unsafe-fn` lint that
// `#[pymethods]` would otherwise trip.
#![allow(unsafe_code, reason = "PyO3 macros expand to unsafe blocks")]
#![allow(
    unsafe_op_in_unsafe_fn,
    reason = "PyO3 macros generate unsafe blocks inside their own unsafe fns"
)]
// Some pyo3 releases emit cfg probes (e.g. `gil-refs`) that the
// compiler doesn't recognize; harmless, silence them.
#![allow(unexpected_cfgs, reason = "pyo3 emits cfg probes cargo cannot know")]
// PyO3 macros expand argument extraction through `From`/`Into` even
// when the source and target types coincide, which clippy flags as
// `useless_conversion`. There is nothing the user can do about it.
#![allow(
    clippy::useless_conversion,
    reason = "PyO3 macros emit From/Into round-trips users can't avoid"
)]
// `#[pyclass]` types and `#[pyfunction]`s are reachable only through
// the PyO3 module-registration macros, not through normal Rust paths.
// That makes the crate-wide `unreachable_pub` and `unnameable_types`
// lints fire on every binding; silence them for this module.
#![allow(
    unreachable_pub,
    reason = "PyO3 bindings are reached via the #[pymodule] registration"
)]
#![allow(
    unnameable_types,
    reason = "PyO3 #[pyclass] types are referenced only by the macro machinery"
)]
// PyO3 extracts arguments by value from Python; clippy's
// `needless_pass_by_value` would force an unergonomic `&[T]` API on
// every wrapper. Silence at the module level.
#![allow(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts owned values from Python; &[T] APIs aren't usable"
)]
// Docstrings here are Python prose (full of class/dtype names like
// `numpy.ndarray`, `BZERO`, etc.) that clippy's `doc_markdown` heuristic
// flags as missing backticks. Sphinx/Napoleon handles formatting.
#![allow(
    clippy::doc_markdown,
    reason = "docstrings here target Sphinx/Python, not rustdoc"
)]
// Same applies to clippy's `doc_link_with_quotes`, which mis-fires on
// Python subscript syntax like `hdr["KEY"]` in our docstring examples.
#![allow(
    clippy::doc_link_with_quotes,
    reason = "Python subscript syntax in docstring examples is not a doc link"
)]

use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyOSError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::FitsError as RustFitsError;

mod convenience;
mod diff;
mod file;
mod hdu;
mod header;
mod table;
mod wcs;
mod writer;

create_exception!(
    fitsy,
    FitsError,
    PyOSError,
    "FITS-level error raised by fitsy."
);

/// Translate a `FitsError` into a Python exception. All variants
/// flow through `fitsy.FitsError` so users can `except FitsError`
/// uniformly, and through `OSError` (its base) for code that does
/// not yet know about us.
pub(crate) fn err_to_py(e: RustFitsError) -> PyErr {
    FitsError::new_err(e.to_string())
}

/// Convenience wrapper to convert any `Result<T, FitsError>` into
/// a `PyResult<T>`.
pub(crate) trait IntoPyResult<T> {
    fn into_py_result(self) -> PyResult<T>;
}

impl<T> IntoPyResult<T> for Result<T, RustFitsError> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(err_to_py)
    }
}

/// Coerce a Python object into a numpy array whose dtype is in the
/// platform's native byte order.
///
/// Both properties are preconditions for the typed
/// `PyReadonlyArrayDyn<T>` extraction the writer paths rely on:
/// that extraction only matches real arrays, and only when the
/// dtype already carries the native endian mark, so a `>f4` from
/// another FITS tool or a plain ``[[1.0, 2.0], ...]`` would
/// otherwise fall through every dtype arm.
///
/// The common case -- an ndarray already in native order -- costs
/// two C-level checks (`PyArray_Check` and `PyArray_ISNBO`) and no
/// interpreter round trip, and returns *the same object*, so the
/// "array is stored by reference" contract of
/// :class:`ImageHdu` still holds. Only byte-swapped arrays and
/// non-array sequences pay for `astype`/`asarray`.
///
/// `context` prefixes any error message; pass the user-facing
/// function name (`"image"`, `"ImageHdu"`, ...).
pub(crate) fn as_native_ndarray<'py>(
    obj: &Bound<'py, PyAny>,
    context: &str,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    let py = obj.py();
    // `is_native_byteorder()` is `None` for dtypes where byte order
    // does not apply (`u8`, `i8`, `bool`), which needs no rewrite
    // either -- only an explicit `Some(false)` does.
    let arr: Bound<'py, PyAny> = match obj.cast::<PyUntypedArray>() {
        Ok(arr) if !matches!(arr.dtype().is_native_byteorder(), Some(false)) => {
            return Ok(arr.clone());
        }
        Ok(_) => obj.clone(),
        // Not an array: a list of lists, a tuple, or anything
        // exposing `__array__` / the buffer protocol.
        Err(_) => py
            .import("numpy")?
            .call_method1("asarray", (obj,))
            .map_err(|e| {
                PyTypeError::new_err(format!(
                    "{context}: expected a numpy array or an array-like sequence ({e})"
                ))
            })?,
    };
    let arr = arr.cast_into::<PyUntypedArray>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{context}: expected a numpy array or an array-like sequence"
        ))
    })?;
    if !matches!(arr.dtype().is_native_byteorder(), Some(false)) {
        return Ok(arr);
    }
    // `dtype.newbyteorder('=')` yields the same kind/itemsize in the
    // platform's order; `astype(..., copy=False)` returns the array
    // untouched when it already matches and a swapped copy otherwise.
    let native_dtype = arr.dtype().as_any().call_method1("newbyteorder", ("=",))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", false)?;
    arr.as_any()
        .call_method("astype", (native_dtype,), Some(&kwargs))?
        .cast_into::<PyUntypedArray>()
        .map_err(|_| PyTypeError::new_err(format!("{context}: byte-order conversion failed")))
}

/// The native module entry point. `maturin` builds this as
/// `fitsy.fitsy` (see `module-name` in `pyproject.toml`); the shim
/// `python/fitsy/__init__.py` re-exports the symbols registered below
/// into the top-level `fitsy` namespace. That shim exists so the
/// package is a real directory, which is what lets the type stubs and
/// the `py.typed` marker ship inside it.
#[pymodule]
fn fitsy(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("FitsError", py.get_type::<FitsError>())?;

    m.add_class::<file::PyFitsFile>()?;
    m.add_class::<header::PyHeader>()?;
    m.add_class::<header::PyHeaderCommentary>()?;
    m.add_class::<hdu::PyImageHdu>()?;
    m.add_class::<hdu::PyImageSection>()?;
    m.add_class::<hdu::PyRandomGroups>()?;
    m.add_class::<table::PyBinTable>()?;
    m.add_class::<table::PyAsciiTable>()?;
    m.add_class::<wcs::PyWcs>()?;
    m.add_class::<wcs::PyWcsFit>()?;
    m.add_class::<writer::PyImageBuilder>()?;
    m.add_class::<writer::PyBinTableBuilder>()?;
    m.add_class::<writer::PyAsciiTableBuilder>()?;
    m.add_class::<diff::PyFitsDiff>()?;

    m.add_function(wrap_pyfunction!(file::open, m)?)?;
    m.add_function(wrap_pyfunction!(writer::image, m)?)?;
    m.add_function(wrap_pyfunction!(writer::bintable, m)?)?;
    m.add_function(wrap_pyfunction!(writer::ascii_table, m)?)?;
    m.add_function(wrap_pyfunction!(writer::write, m)?)?;
    m.add_function(wrap_pyfunction!(writer::compressed_image, m)?)?;
    m.add_function(wrap_pyfunction!(wcs::fit_wcs, m)?)?;
    m.add_function(wrap_pyfunction!(diff::diff, m)?)?;
    m.add_function(wrap_pyfunction!(convenience::getdata, m)?)?;
    m.add_function(wrap_pyfunction!(convenience::getheader, m)?)?;
    m.add_function(wrap_pyfunction!(convenience::getval, m)?)?;
    m.add_function(wrap_pyfunction!(convenience::setval, m)?)?;
    m.add_function(wrap_pyfunction!(convenience::delval, m)?)?;
    m.add_function(wrap_pyfunction!(convenience::info, m)?)?;
    m.add_function(wrap_pyfunction!(convenience::append, m)?)?;
    Ok(())
}
