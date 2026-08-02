//! PyO3 bindings for fitsy.
//!
//! This module is compiled when the `python` feature is enabled.
//! It builds the `fitsy` Python wheel through maturin.
//!
//! # Purpose
//!
//! This module is a translation layer. It holds no FITS logic.
//! Each type here wraps a type from the Rust API of the crate.
//! Each function here converts between Rust values and Python objects.
//!
//! Add new FITS behavior to the core crate. Add only the conversion
//! to this module. If you must add logic here, the core crate is
//! usually missing a function.
//!
//! # Layout
//!
//! This file holds the module setup. It contains the lint allowances,
//! the exception type, the shared helpers and the module registration.
//! Each submodule holds one part of the Python API:
//!
//! - `file` -- `open()` and `FitsFile`. This is the HDU list, the
//!   update mode and the write-back paths.
//! - `hdu` -- `ImageHdu`, `_ImageSection` and `RandomGroups`.
//! - `header` -- `Header` and `HeaderCommentary`.
//! - `table` -- `BinTable` and `AsciiTable`.
//! - `wcs` -- `Wcs`, `WcsFit` and `fit_wcs()`.
//! - `writer` -- the `image()`, `bintable()`, `ascii_table()` and
//!   `compressed_image()` factories, the `write()` function, and the
//!   three builder classes that they return.
//! - `diff` -- `FitsDiff` and `diff()`.
//! - `convenience` -- the one-shot module functions, such as
//!   `getdata()` and `getheader()`.
//!
//! A new class or function needs three steps to become visible:
//! 1. Write it in the correct submodule.
//! 2. Register it in the `fitsy()` function at the end of this file.
//! 3. Declare it in the type stub `python/fitsy/__init__.pyi`.
//!
//! Step 3 is easy to forget. Users and type checkers do not see a name
//! that is absent from the stub.
//!
//! # Design constraints
//!
//! Three constraints control most of the code in this module.
//!
//! First, lifetimes. The Rust API hands out borrowed views. For example,
//! [`crate::ImageHdu`] borrows its [`crate::FitsFile`]. A `#[pyclass]`
//! must be `'static`, so it cannot hold such a view. The bindings keep
//! an `Arc<FitsFile>` and an index instead. See `ReadBinding` in `hdu`.
//! Each operation makes the borrowed view again from those two values.
//!
//! Second, shared mutation. Python can hold many references to one
//! object, and it can use that object from more than one thread.
//! Mutable state thus sits behind `Arc<Mutex<...>>`. An `ImageHdu` and
//! its section share one pixel array in this way.
//!
//! Third, laziness. `FitsFile` does not read pixel data at open time.
//! An `ImageHdu` reads its pixels at the first access to `data`.
//! Python edits a numpy array without telling the bindings, so the
//! bindings cannot see such an edit. `DirtyFlags` in `file` records
//! that an array went out to Python. On flush, the code compares the
//! array with the file and rewrites only after a true difference.
//!
//! # Image data
//!
//! The read code is in `hdu`. The dtype of the returned array depends
//! on the scaling keywords, not on `BITPIX` alone:
//!
//! - fitsy reads pixel data from a [`crate::ImageHdu`] into a numpy array.
//! - The read path is selected from `BZERO`, `BSCALE` and `BLANK` first.
//!   `BITPIX` selects the width in each path.
//! - If `BZERO` is 0, `BSCALE` is 1 and there is no `BLANK`, fitsy returns
//!   the raw pixels. The dtype follows `BITPIX`.
//! - If `BSCALE` is 1, `BZERO` is the standard integer offset and there
//!   is no `BLANK`, fitsy returns the matching unsigned dtype. `BITPIX`
//!   8 with `BZERO` -128 is the signed-byte form of this convention and
//!   returns `int8`.
//! - For all other scaling, fitsy applies `BSCALE`, `BZERO` and `BLANK`,
//!   and returns floats. `BITPIX` 8, 16 and -32 give `float32`.
//!   All other values give `float64`.
//! - The data is returned to numpy as an owned array in the native byte order.
//! - FITS files store data in big-endian order.
//! - fitsy writes the pixels into the numpy buffer and changes the byte order
//!   in that same buffer. It makes no more copies for the byte order.
//! - `BITPIX` 8 data needs no byte-order change.
//!
//! # Headers
//!
//! The header code is in `header`. Cards convert to Python as follows:
//!
//! - The header interface uses a Python dict-like format.
//! - Values are returned as native Python types: `int`, `float`,
//!   `str`, `bool`, `complex`.
//! - An undefined value is returned as `None`.
//! - `COMMENT`, `HISTORY` and blank keywords are returned as a
//!   `HeaderCommentary` object.
//!
//! # Errors
//!
//! Use `into_py_result()` on each `Result` that comes from the core
//! crate. Raise a Python exception directly for bad Python-level input:
//!
//! - All `FitsError` types map to one Python exception: `fitsy.FitsError`.
//! - `fitsy.FitsError` is a subclass of `OSError`.
//! - Code that uses `except OSError` will also catch these errors.
//! - The bindings also raise `ValueError`, `TypeError`, `KeyError`,
//!   `IndexError` and `RuntimeError` for bad Python-level input.
//! - These exceptions are not subclasses of `OSError`. Do not use
//!   `except OSError` to catch all errors from fitsy.

// PyO3 macros expand to `unsafe` blocks.
// The crate-wide `deny(unsafe_code)` in lib.rs does not allow this.
// This module allows `unsafe_code` to permit the PyO3 macros.
// Rust 2024 also requires allowing `unsafe_op_in_unsafe_fn`.
// The `#[pymodule]`, `#[pyfunction]` and `#[pymethods]` macros
// trigger this lint.
#![allow(unsafe_code, reason = "PyO3 macros expand to unsafe blocks")]
#![allow(
    unsafe_op_in_unsafe_fn,
    reason = "PyO3 macros generate unsafe blocks inside their own unsafe fns"
)]
// Some PyO3 releases emit cfg probes that the compiler does not recognize.
// These probes are harmless. This attribute suppresses the warning.
#![allow(unexpected_cfgs, reason = "pyo3 emits cfg probes cargo cannot know")]
// PyO3 macros extract arguments through `From`/`Into` conversions.
// This occurs even when the source and target types are the same type.
// Clippy flags this as `useless_conversion`. You cannot avoid this.
#![allow(
    clippy::useless_conversion,
    reason = "PyO3 macros emit From/Into round-trips users can't avoid"
)]
// `#[pyclass]` types and `#[pyfunction]` items are only reachable
// through the PyO3 module-registration macros, not through normal Rust paths.
// This causes the `unreachable_pub` and `unnameable_types` lints to fire
// on every binding. This module suppresses both lints.
#![allow(
    unreachable_pub,
    reason = "PyO3 bindings are reached via the #[pymodule] registration"
)]
#![allow(
    unnameable_types,
    reason = "PyO3 #[pyclass] types are referenced only by the macro machinery"
)]
// PyO3 extracts arguments by value from Python.
// The `needless_pass_by_value` lint would require a `&[T]` API for
// every wrapper. A `&[T]` API is not usable here. This lint is
// suppressed at the module level.
#![allow(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts owned values from Python; &[T] APIs aren't usable"
)]
// Docstrings in this module target Sphinx and Python users.
// They contain class and dtype names such as `numpy.ndarray` and `BZERO`.
// The `doc_markdown` lint flags these as missing backticks.
// Sphinx and Napoleon handle the formatting. This lint is suppressed.
#![allow(
    clippy::doc_markdown,
    reason = "docstrings here target Sphinx/Python, not rustdoc"
)]
// Docstring examples use Python subscript syntax such as `hdr["KEY"]`.
// The `doc_link_with_quotes` lint incorrectly treats these as doc links.
// This lint is suppressed.
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

/// Convert a core-crate error into a Python exception.
///
/// Every variant of [`RustFitsError`] maps to the one Python exception
/// `fitsy.FitsError`. The message of the exception is the `Display`
/// form of `e`. The FITS error kind is thus kept in the text, not in
/// the exception type.
///
/// `fitsy.FitsError` is a subclass of `OSError`, so Python code that
/// catches `OSError` also catches these errors.
///
/// Prefer [`IntoPyResult::into_py_result`] at call sites. Use this
/// function directly only where there is no `Result` to map.
pub(crate) fn err_to_py(e: RustFitsError) -> PyErr {
    FitsError::new_err(e.to_string())
}

/// Convert a fallible core-crate call into a [`PyResult`].
///
/// Apply this trait to every `Result` that crosses into the bindings.
/// It keeps the error mapping in one place and keeps the call sites
/// free of a repeated `map_err(err_to_py)`.
pub(crate) trait IntoPyResult<T> {
    /// Keep the success value and map the error with [`err_to_py`].
    ///
    /// # Errors
    ///
    /// Returns `fitsy.FitsError` if `self` is an [`Err`].
    fn into_py_result(self) -> PyResult<T>;
}

impl<T> IntoPyResult<T> for Result<T, RustFitsError> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(err_to_py)
    }
}

/// Convert a Python object into a numpy array in the native byte order.
///
/// Apply this function to all user array data that enters the bindings.
/// Two groups of callers need it:
/// - The writer extracts data with `PyReadonlyArrayDyn<T>`. That
///   extraction accepts only a true array, and only when the dtype
///   carries the native byte-order mark. A `>f4` array from another
///   FITS tool matches no dtype arm.
/// - [`hdu::PyImageHdu`] keeps the array that this function returns.
///
/// The behavior depends on `obj`:
/// - A numpy array that is already in the native byte order is
///   returned as the same object, with no copy. `ImageHdu` thus holds
///   the array of the caller by reference, and later in-place edits
///   still reach `FitsFile.writeto`.
/// - A numpy array in the opposite byte order is converted with
///   `astype`, which copies. `ImageHdu` does not see later edits of
///   the initial array.
/// - Any other object goes through `numpy.asarray` first, which
///   copies. This accepts a list, a tuple, and any object with
///   `__array__` or the buffer protocol.
///
/// `context` names the caller and prefixes every error message. Pass
/// the user-facing function, class or attribute name, such as
/// `"image"`, `"ImageHdu"` or `"ImageHdu.data"`.
///
/// # Errors
///
/// Returns [`PyTypeError`] if `obj` is neither an array nor a sequence
/// that `numpy.asarray` accepts, and if the byte-order conversion
/// returns a non-array. Returns the original Python exception if the
/// `numpy` import, `newbyteorder` or `astype` call fails.
pub(crate) fn as_native_ndarray<'py>(
    obj: &Bound<'py, PyAny>,
    context: &str,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    let py = obj.py();
    // `is_native_byteorder()` returns `None` for dtypes where byte order
    // does not apply, such as `u8`, `i8`, and `bool`.
    // These dtypes do not need a byte-order conversion.
    // Only `Some(false)` means the byte order must be changed.
    let arr: Bound<'py, PyAny> = match obj.cast::<PyUntypedArray>() {
        Ok(arr) if !matches!(arr.dtype().is_native_byteorder(), Some(false)) => {
            return Ok(arr.clone());
        }
        Ok(_) => obj.clone(),
        // The input is not a numpy array.
        // It may be a list, a tuple, or any object that supports
        // `__array__` or the buffer protocol.
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
    // `newbyteorder('=')` returns the same dtype kind and item size
    // in the platform byte order.
    // The byte order is known to differ at this point.
    // `astype(..., copy=False)` thus always returns a converted copy.
    let native_dtype = arr.dtype().as_any().call_method1("newbyteorder", ("=",))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", false)?;
    arr.as_any()
        .call_method("astype", (native_dtype,), Some(&kwargs))?
        .cast_into::<PyUntypedArray>()
        .map_err(|_| PyTypeError::new_err(format!("{context}: byte-order conversion failed")))
}

// The native module entry point.
//
// The doc comment below is not a Rust comment for maintainers: PyO3
// puts it on the module object as `fitsy.__doc__`, which is what
// `help(fitsy)` prints. Keep it in numpydoc format and keep it
// user-facing.
//
// `maturin` builds this module as `fitsy.fitsy`. See `module-name` in
// `pyproject.toml`. The shim at `python/fitsy/__init__.py` re-exports
// the symbols below into the top-level `fitsy` namespace, and also
// re-exports this `__doc__`. The shim is required so the package is a
// real directory, which is required to ship the type stubs and the
// `py.typed` marker.
/// Fast FITS file I/O, WCS transforms and image compression.
///
/// Open a file with :func:`fitsy.open`. The returned :class:`FitsFile`
/// behaves like a list of HDUs. Each HDU carries a :class:`Header`,
/// and an image HDU reads its pixels into a :class:`numpy.ndarray` on
/// the first access to ``.data``.
///
/// To write a file, describe each HDU with :func:`fitsy.image`,
/// :func:`fitsy.bintable`, :func:`fitsy.ascii_table` or
/// :func:`fitsy.compressed_image`, then pass the list to
/// :func:`fitsy.write`.
///
/// For a single operation on a single path, use the module-level
/// functions :func:`fitsy.getdata`, :func:`fitsy.getheader`,
/// :func:`fitsy.getval`, :func:`fitsy.setval`, :func:`fitsy.delval`,
/// :func:`fitsy.info` and :func:`fitsy.append`.
///
/// All FITS-level failures raise :class:`fitsy.FitsError`, which is a
/// subclass of :class:`OSError`. An invalid Python argument raises the
/// usual built-in exception, such as :class:`TypeError`,
/// :class:`KeyError` or :class:`IndexError`.
///
/// Attributes
/// ----------
/// __version__ : str
///   Version of the installed fitsy package.
/// FitsError : type
///   Exception class for all FITS-level failures.
///
/// Examples
/// --------
/// Read the pixels and one keyword of the primary HDU:
///
/// >>> import fitsy
/// >>> with fitsy.open("image.fits") as f:
/// ...     data = f[0].data
/// ...     width = f[0].header["NAXIS1"]
///
/// Write a new file:
///
/// >>> import numpy as np, fitsy
/// >>> fitsy.write("out.fits", [
/// ...     fitsy.image(np.zeros((10, 10), dtype=np.float32)),
/// ... ])
///
/// Convert a pixel position to a sky position:
///
/// >>> with fitsy.open("image.fits") as f:
/// ...     wcs = f[0].wcs()
/// ...     ra, dec = wcs.pixel_to_celestial(512.0, 512.0)
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
