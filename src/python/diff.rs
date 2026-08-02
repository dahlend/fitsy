//! `PyFitsDiff` -- Python wrapper for [`crate::diff::FitsDiff`].

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyList;

use super::IntoPyResult;
use crate::diff::{DiffOptions, FitsDiff};

/// Compare two FITS files and return the differences.
///
/// Parameters
/// ----------
/// a, b : str or os.PathLike
///   Paths to the two files to compare.
/// rtol : float, optional
///   Relative tolerance for floating-point comparisons. Default
///   ``0.0`` (exact equality).
/// atol : float, optional
///   Absolute tolerance for floating-point comparisons. Default
///   ``0.0``.
/// max_diffs : int, optional
///   Maximum number of data differences recorded per HDU. Default
///   10. Counting continues past this limit. The true count appears
///   in the text report. Header differences are never truncated.
/// ignore_keywords : sequence of str, optional
///   Header keywords to ignore, case-insensitive. Default
///   ``["CHECKSUM", "DATASUM", "DATE"]``.
///
/// Returns
/// -------
/// FitsDiff
///   The comparison result. Call ``str(diff)`` to get the text
///   report. :attr:`FitsDiff.identical` is True when the files
///   match.
///
/// Raises
/// ------
/// FitsError
///   If either file cannot be opened or parsed as FITS.
///
/// Notes
/// -----
/// fitsy combines the two tolerances as ``|a - b| <= atol + rtol *
/// |b|``. Both default to ``0.0``, which requires exact equality.
/// A relative tolerance alone cannot reconcile values that straddle
/// zero. Two values that are both NaN compare equal at any
/// tolerance.
///
/// The tolerances apply to every floating-point value fitsy
/// compares: header card values, image pixels, and table cells.
/// fitsy compares image pixels in physical units, with ``BZERO``
/// and ``BSCALE`` applied and ``BLANK`` as NaN. It reports an image
/// difference by pixel number and a table difference as
/// ``COLUMN[row]``.
///
/// A random-groups HDU, or an HDU of an extension type fitsy does
/// not recognize, has no decoded form. fitsy compares its raw bytes
/// instead. That comparison does not use ``rtol`` or ``atol``.
///
/// When the two HDUs at one index have different types, fitsy
/// reports the type difference and compares neither the headers nor
/// the data of that HDU.
#[pyfunction]
#[pyo3(signature = (a, b, *, rtol=0.0, atol=0.0, max_diffs=10, ignore_keywords=None))]
pub fn diff(
    a: PathBuf,
    b: PathBuf,
    rtol: f64,
    atol: f64,
    max_diffs: usize,
    ignore_keywords: Option<Vec<String>>,
) -> PyResult<PyFitsDiff> {
    let mut opts = DiffOptions {
        relative_tolerance: rtol,
        absolute_tolerance: atol,
        max_diffs,
        ..Default::default()
    };
    if let Some(kw) = ignore_keywords {
        opts.ignore_keywords = kw;
    }
    let inner = FitsDiff::open(&a, &b, opts).into_py_result()?;
    Ok(PyFitsDiff { inner })
}

/// Result of comparing two FITS files. See :func:`diff`.
#[pyclass(name = "FitsDiff", module = "fitsy")]
#[derive(Debug)]
pub struct PyFitsDiff {
    pub(crate) inner: FitsDiff,
}

#[pymethods]
impl PyFitsDiff {
    /// True when both files have the same number of HDUs and no
    /// compared HDU has a reported difference. Differences are
    /// judged with the tolerances and ignored keywords passed to
    /// :func:`diff`.
    #[getter]
    fn identical(&self) -> bool {
        self.inner.is_identical()
    }

    /// HDU counts of the two files, as ``(n_a, n_b)``. When the
    /// counts differ, only the shared prefix of HDUs is compared.
    #[getter]
    fn hdu_counts(&self) -> (usize, usize) {
        self.inner.hdu_counts
    }

    /// Number of compared HDUs with a reported difference.
    #[getter]
    fn diff_hdu_count(&self) -> usize {
        self.inner.hdus.iter().filter(|h| !h.is_empty()).count()
    }

    /// Return the index of every compared HDU with a difference.
    ///
    /// Returns
    /// -------
    /// list of int
    ///   Indices in ascending order. An index counts from the
    ///   primary HDU, which is index ``0``.
    fn diff_hdu_indices(&self, py: Python<'_>) -> Py<PyList> {
        let out = PyList::empty(py);
        for (i, h) in self.inner.hdus.iter().enumerate() {
            if !h.is_empty() {
                let _ = out.append(i);
            }
        }
        out.unbind()
    }

    /// Render the differences as text.
    ///
    /// Returns
    /// -------
    /// str
    ///   A multi-line report. The report names each HDU with a
    ///   difference, then lists the header differences and the data
    ///   differences of that HDU. ``str(diff)`` returns the same
    ///   text.
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
        // Truthy means the files differ: bool(diff) answers whether
        // a difference exists, not whether the files match.
        !self.inner.is_identical()
    }
}
