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
// pyo3 0.22 emits cfg(gil-refs) probes that the compiler doesn't
// recognize; harmless, silence them.
#![allow(unexpected_cfgs, reason = "pyo3 0.22 probes cfg(gil-refs)")]
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

use pyo3::create_exception;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

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

/// The native module entry point. `maturin` builds this as
/// `fitsy._fitsy`; the pure-Python `fitsy/__init__.py` re-exports
/// the symbols listed below into the top-level `fitsy` namespace.
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
