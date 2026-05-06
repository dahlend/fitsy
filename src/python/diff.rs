//! Python wrapper for `fitsy::diff::FitsDiff`.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::IntoPyResult;
use crate::diff::{DiffOptions, FitsDiff};

/// Compare two FITS files and return a diff report.
///
/// Parameters
/// ----------
/// a, b : str | os.PathLike
///     Paths to the two files to compare.
/// rtol : float, optional
///     Relative tolerance for floating-point comparisons.  Default
///     ``0.0`` (exact equality).
/// max_diffs : int, optional
///     Maximum number of differences recorded per category before
///     truncation. Default 10.
/// ignore_keywords : list[str], optional
///     Header keywords to ignore (case-insensitive). Defaults to
///     ``["CHECKSUM", "DATASUM", "DATE"]``.
///
/// Returns
/// -------
/// FitsDiff
///     Diff object. Use ``str(diff)`` for the report; ``diff.identical``
///     is True when the files match.
#[pyfunction]
#[pyo3(signature = (a, b, *, rtol=0.0, max_diffs=10, ignore_keywords=None))]
pub fn diff(
    a: PathBuf,
    b: PathBuf,
    rtol: f64,
    max_diffs: usize,
    ignore_keywords: Option<Vec<String>>,
) -> PyResult<PyFitsDiff> {
    let mut opts = DiffOptions {
        relative_tolerance: rtol,
        max_diffs,
        ..Default::default()
    };
    if let Some(kw) = ignore_keywords {
        opts.ignore_keywords = kw;
    }
    let inner = FitsDiff::open(&a, &b, opts).into_py_result()?;
    Ok(PyFitsDiff { inner })
}

/// Result of comparing two FITS files.
#[pyclass(name = "FitsDiff", module = "fitsy")]
#[derive(Debug)]
pub struct PyFitsDiff {
    pub(crate) inner: FitsDiff,
}

#[pymethods]
impl PyFitsDiff {
    /// True when both files have the same number of HDUs and every
    /// HDU is byte-equivalent under the configured options.
    #[getter]
    fn identical(&self) -> bool {
        self.inner.is_identical()
    }

    /// ``(n_a, n_b)`` HDU counts.
    #[getter]
    fn hdu_counts(&self) -> (usize, usize) {
        self.inner.hdu_counts
    }

    /// Number of HDUs that have at least one difference.
    #[getter]
    fn diff_hdu_count(&self) -> usize {
        self.inner.hdus.iter().filter(|h| !h.is_empty()).count()
    }

    /// List of HDU indices that contain differences.
    fn diff_hdu_indices(&self, py: Python<'_>) -> Py<PyList> {
        let out = PyList::empty(py);
        for (i, h) in self.inner.hdus.iter().enumerate() {
            if !h.is_empty() {
                let _ = out.append(i);
            }
        }
        out.unbind()
    }

    /// Multi-line text report (mirrors astropy ``FITSDiff.report``).
    fn report(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        self.report()
    }

    fn __repr__(&self) -> String {
        format!(
            "FitsDiff(hdu_counts={:?}, identical={})",
            self.inner.hdu_counts,
            self.inner.is_identical()
        )
    }

    fn __bool__(&self) -> bool {
        // Truthy when non-identical (i.e. diff exists).
        !self.inner.is_identical()
    }
}

/// Internal stub so ValueError gets re-exported.
#[pyfunction]
#[pyo3(signature = ())]
pub fn _diff_module_marker() {
    let _ = PyValueError::new_err("");
}
