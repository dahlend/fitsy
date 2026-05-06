//! `PyBinTable` / `PyAsciiTable` -- table HDU wrappers.
//!
//! Tables are read into owned Python data eagerly. Scalar columns
//! return 1-D numpy arrays; fixed-repeat columns return `(n_rows,
//! repeat)` 2-D arrays; VLA and string columns return Python lists
//! (the cells have non-uniform shape, so a single ndarray is not
//! the right answer).

use numpy::IntoPyArray;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::hdu::{AsciiCell, AsciiTableHdu, BinTableHdu, BinValue};

/// Convert an `Option<i64>` column to a Python object: a plain
/// `numpy.int64` array if there are no nulls, otherwise a
/// `numpy.ma.MaskedArray` so callers don't lose the integer dtype.
fn nullable_int_to_py(py: Python<'_>, v: &[Option<i64>]) -> PyResult<Py<PyAny>> {
    if v.iter().all(Option::is_some) {
        let plain: Vec<i64> = v.iter().map(|x| x.unwrap()).collect();
        return Ok(plain.into_pyarray(py).into_any().unbind());
    }
    // Fill masked positions with 0 so the underlying buffer is valid;
    // the mask hides them. numpy.ma is part of the standard numpy
    // distribution, so this import is always safe when numpy is.
    let data: Vec<i64> = v.iter().map(|x| x.unwrap_or(0)).collect();
    let mask: Vec<bool> = v.iter().map(Option::is_none).collect();
    let np_ma = py.import("numpy.ma")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("data", data.into_pyarray(py))?;
    kwargs.set_item("mask", mask.into_pyarray(py))?;
    let arr = np_ma.call_method("array", (), Some(&kwargs))?;
    Ok(arr.unbind())
}

use super::IntoPyResult;
use super::header::PyHeader;

/// Binary table HDU (``BINTABLE``).
///
/// Returned by :meth:`FitsFile.hdu` (or ``file[i]``) when the HDU
/// kind is ``BINTABLE``. Columns are decoded eagerly:
///
/// - Scalar numeric columns return 1-D :class:`numpy.ndarray`.
/// - Fixed-repeat columns return ``(n_rows, repeat)`` 2-D arrays.
/// - Variable-length and string columns return Python lists.
///
/// Examples
/// --------
/// >>> with fitsy.open("catalog.fits") as f:
/// ...     tbl = f[1]
/// ...     ra = tbl["RA"]   # numpy array
/// ...     name = tbl["NAME"]  # list[str]
#[pyclass(name = "BinTable", module = "fitsy")]
#[derive(Debug)]
pub struct PyBinTable {
    pub(crate) header: PyHeader,
    pub(crate) n_rows: usize,
    pub(crate) column_names: Vec<String>,
    /// Pre-decoded columns. One `PyColumn` per BinColumn, in
    /// declaration order.
    columns: Vec<PyColumn>,
    /// Raw on-disk data bytes (rows + heap), captured at load time
    /// so that :meth:`FitsFile.writeto` can re-emit the table
    /// byte-for-byte. Column-level Python edits are *not* round
    /// tripped in v0.1; reconstruct the table with
    /// :func:`fitsy.bintable` to change column data.
    pub(crate) raw: Vec<u8>,
}

/// Owned column data. We materialise everything up front because
/// column access is the dominant Python idiom and lazy decode
/// would mean keeping `BinTableHdu<'a>` alive in the Python wrapper.
#[derive(Debug)]
enum PyColumn {
    /// Float-like scalars (any numeric BinValue flattened to f64
    /// row-by-row, including scaled ints). Length = `n_rows`.
    F64(Vec<f64>),
    /// Integer scalars (B/I/J/K with no scaling). `None` => TNULL.
    I64(Vec<Option<i64>>),
    /// Strings (TFORM=`A`). One per row.
    Str(Vec<String>),
    /// Booleans. `None` => undefined logical (FITS `'\0'`).
    Bool(Vec<Option<bool>>),
    /// Anything else (vector cells, complex, bits, VLA, ...) is held
    /// as a list of per-cell Python objects built lazily. The
    /// optional `Vec<usize>` is the column's `TDIMn` shape in FITS
    /// order (fastest-varying first); when present, numeric cells
    /// are reshaped accordingly when handed to Python.
    Generic(Vec<BinValue>, Option<Vec<usize>>),
}

impl PyBinTable {
    pub(crate) fn from_table(t: &BinTableHdu<'_>, header: PyHeader) -> PyResult<Self> {
        let n_rows = t.n_rows();
        let mut column_names = Vec::with_capacity(t.columns().len());
        let mut columns = Vec::with_capacity(t.columns().len());
        for col in t.columns() {
            column_names.push(col.name.clone());
            columns.push(decode_column(t, col, n_rows)?);
        }
        Ok(Self {
            header,
            n_rows,
            column_names,
            columns,
            raw: t.data_bytes().to_vec(),
        })
    }

    /// Clone the underlying `Header` (for serialization).
    pub(crate) fn header_clone(&self) -> crate::Header {
        self.header.lock().clone()
    }

    /// Reconstruct a `PyBinTable` from a builder snapshot. Columns
    /// are not re-decoded -- the table behaves as a raw byte blob
    /// (column accessors return KeyError).
    pub(crate) fn from_built_bytes(header: crate::Header, bytes: Vec<u8>) -> Self {
        Self {
            header: PyHeader::from_header_with(&header, false),
            n_rows: 0,
            column_names: Vec::new(),
            columns: Vec::new(),
            raw: bytes,
        }
    }
}

/// Mark a numpy array as read-only by clearing its `WRITEABLE` flag.
/// Silently no-ops for non-array values (lists, strings).
fn freeze_if_array(py: Python<'_>, obj: &Py<PyAny>) {
    let _ = obj.bind(py).call_method1("setflags", ((), false));
}

fn decode_column(
    t: &BinTableHdu<'_>,
    col: &crate::hdu::BinColumn,
    n_rows: usize,
) -> PyResult<PyColumn> {
    let tdim = || col.tdim.clone();
    // Peek at the first row to decide on a representation. Empty
    // tables fall back to `Generic` (an empty list).
    if n_rows == 0 {
        return Ok(PyColumn::Generic(Vec::new(), tdim()));
    }
    let first = t.cell_value(0, col).into_py_result()?;
    match first {
        BinValue::F64(v) if v.len() == 1 => {
            let mut out = Vec::with_capacity(n_rows);
            out.push(v[0]);
            for r in 1..n_rows {
                if let BinValue::F64(v) = t.cell_value(r, col).into_py_result()? {
                    out.push(*v.first().unwrap_or(&f64::NAN));
                } else {
                    return Ok(PyColumn::Generic(reread_all(t, col, n_rows)?, tdim()));
                }
            }
            Ok(PyColumn::F64(out))
        }
        BinValue::F32(v) if v.len() == 1 => {
            let mut out = Vec::with_capacity(n_rows);
            out.push(f64::from(v[0]));
            for r in 1..n_rows {
                if let BinValue::F32(v) = t.cell_value(r, col).into_py_result()? {
                    out.push(f64::from(*v.first().unwrap_or(&f32::NAN)));
                } else {
                    return Ok(PyColumn::Generic(reread_all(t, col, n_rows)?, tdim()));
                }
            }
            Ok(PyColumn::F64(out))
        }
        BinValue::Float(v) if v.len() == 1 => {
            let mut out = Vec::with_capacity(n_rows);
            out.push(v[0]);
            for r in 1..n_rows {
                if let BinValue::Float(v) = t.cell_value(r, col).into_py_result()? {
                    out.push(*v.first().unwrap_or(&f64::NAN));
                } else {
                    return Ok(PyColumn::Generic(reread_all(t, col, n_rows)?, tdim()));
                }
            }
            Ok(PyColumn::F64(out))
        }
        BinValue::Int(v) if v.len() == 1 => {
            let mut out = Vec::with_capacity(n_rows);
            out.push(v[0]);
            for r in 1..n_rows {
                if let BinValue::Int(v) = t.cell_value(r, col).into_py_result()? {
                    out.push(v.into_iter().next().unwrap_or(None));
                } else {
                    return Ok(PyColumn::Generic(reread_all(t, col, n_rows)?, tdim()));
                }
            }
            Ok(PyColumn::I64(out))
        }
        BinValue::Logical(v) if v.len() == 1 => {
            let mut out = Vec::with_capacity(n_rows);
            out.push(v[0]);
            for r in 1..n_rows {
                if let BinValue::Logical(v) = t.cell_value(r, col).into_py_result()? {
                    out.push(v.into_iter().next().unwrap_or(None));
                } else {
                    return Ok(PyColumn::Generic(reread_all(t, col, n_rows)?, tdim()));
                }
            }
            Ok(PyColumn::Bool(out))
        }
        BinValue::Str(s) => {
            let mut out = Vec::with_capacity(n_rows);
            out.push(s);
            for r in 1..n_rows {
                if let BinValue::Str(s) = t.cell_value(r, col).into_py_result()? {
                    out.push(s);
                } else {
                    return Ok(PyColumn::Generic(reread_all(t, col, n_rows)?, tdim()));
                }
            }
            Ok(PyColumn::Str(out))
        }
        _ => Ok(PyColumn::Generic(reread_all(t, col, n_rows)?, tdim())),
    }
}

fn reread_all(
    t: &BinTableHdu<'_>,
    col: &crate::hdu::BinColumn,
    n_rows: usize,
) -> PyResult<Vec<BinValue>> {
    let mut out = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        out.push(t.cell_value(r, col).into_py_result()?);
    }
    Ok(out)
}

// -- repr helpers ------------------------------------------------------------

fn extname_opt(h: &PyHeader) -> Option<String> {
    h.lock()
        .entries()
        .iter()
        .find(|e| e.keyword == "EXTNAME")
        .and_then(|e| e.value.as_ref())
        .and_then(|v| match v {
            crate::header::Value::String(s) => {
                let t = s.trim().to_string();
                if t.is_empty() { None } else { Some(t) }
            }
            _ => None,
        })
}

fn fmt_f64(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    let abs = x.abs();
    if abs == 0.0 || (0.001..1_000_000.0).contains(&abs) {
        let raw = format!("{x:.6}");
        let trimmed = raw.trim_end_matches('0').trim_end_matches('.');
        trimmed.to_string()
    } else {
        format!("{x:.4e}")
    }
}

fn bin_column_preview(col: &PyColumn, n: usize) -> (&'static str, Vec<String>) {
    match col {
        PyColumn::F64(v) => {
            let cells = v[..n.min(v.len())].iter().map(|&x| fmt_f64(x)).collect();
            ("float64", cells)
        }
        PyColumn::I64(v) => {
            let cells = v[..n.min(v.len())]
                .iter()
                .map(|x| match x {
                    Some(i) => i.to_string(),
                    None => "--".to_string(),
                })
                .collect();
            ("int64", cells)
        }
        PyColumn::Str(v) => {
            let cells = v[..n.min(v.len())]
                .iter()
                .map(|s| {
                    let s = s.trim();
                    let chars: Vec<char> = s.chars().collect();
                    if chars.len() > 12 {
                        format!("{}\u{2026}", chars[..12].iter().collect::<String>())
                    } else {
                        s.to_string()
                    }
                })
                .collect();
            ("str", cells)
        }
        PyColumn::Bool(v) => {
            let cells = v[..n.min(v.len())]
                .iter()
                .map(|x| match x {
                    Some(true) => "True".to_string(),
                    Some(false) => "False".to_string(),
                    None => "--".to_string(),
                })
                .collect();
            ("bool", cells)
        }
        PyColumn::Generic(v, shape) => {
            let cells = v[..n.min(v.len())]
                .iter()
                .map(|c| generic_cell_summary(c, shape.as_deref()))
                .collect();
            ("object", cells)
        }
    }
}

/// Compact one-cell summary for a vector / non-scalar `BinValue`.
/// Shows element type and shape so the repr is informative for
/// vector / VLA / array-valued columns. When `tdim` is provided
/// and matches the cell length, the multi-dim shape is shown
/// (in C order, i.e. reversed from FITS).
fn generic_cell_summary(v: &BinValue, tdim: Option<&[usize]>) -> String {
    fn shape_str(n: usize, tdim: Option<&[usize]>) -> String {
        if let Some(s) = tdim {
            let prod: usize = s.iter().product();
            if prod == n && s.len() >= 2 {
                let parts: Vec<String> = s.iter().rev().map(ToString::to_string).collect();
                return parts.join("\u{00d7}");
            }
        }
        n.to_string()
    }
    match v {
        BinValue::F64(x) | BinValue::Float(x) => format!("f64[{}]", shape_str(x.len(), tdim)),
        BinValue::F32(x) => format!("f32[{}]", shape_str(x.len(), tdim)),
        BinValue::Int(x) => format!("i64[{}]", shape_str(x.len(), tdim)),
        BinValue::Uint(x) => format!("u64[{}]", shape_str(x.len(), tdim)),
        BinValue::Logical(x) => format!("bool[{}]", shape_str(x.len(), tdim)),
        BinValue::Str(s) => {
            let chars: Vec<char> = s.chars().collect();
            if chars.len() > 12 {
                format!("\"{}\u{2026}\"", chars[..12].iter().collect::<String>())
            } else {
                format!("\"{s}\"")
            }
        }
        BinValue::C64(x) => format!("c64[{}]", shape_str(x.len(), tdim)),
        BinValue::C128(x) => format!("c128[{}]", shape_str(x.len(), tdim)),
        BinValue::Bits(_, count) => format!("bits[{count}]"),
        BinValue::Vla(inner) => format!("vla<{}>", generic_cell_summary(inner, None)),
    }
}

fn ascii_column_preview(col: &AsciiPyColumn, n: usize) -> (&'static str, Vec<String>) {
    match col {
        AsciiPyColumn::F64(v) => {
            let cells = v[..n.min(v.len())].iter().map(|&x| fmt_f64(x)).collect();
            ("float64", cells)
        }
        AsciiPyColumn::Str(v) => {
            let cells = v[..n.min(v.len())]
                .iter()
                .map(|s| {
                    let s = s.trim();
                    let chars: Vec<char> = s.chars().collect();
                    if chars.len() > 12 {
                        format!("{}\u{2026}", chars[..12].iter().collect::<String>())
                    } else {
                        s.to_string()
                    }
                })
                .collect();
            ("str", cells)
        }
    }
}

#[pymethods]
impl PyBinTable {
    /// The HDU header.
    #[getter]
    fn header(&self) -> PyHeader {
        self.header.clone()
    }

    /// Number of rows in the table.
    #[getter]
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// List of column names in declaration order.
    #[getter]
    fn column_names(&self) -> Vec<String> {
        self.column_names.clone()
    }

    /// Return one column by name, or one row by integer/slice
    /// (``table[name]`` / ``table[i]`` / ``table[i:j]``).
    ///
    /// - ``str`` key: column accessor (numpy array or list).
    /// - ``int`` key: a row dict ``{col_name: value, ...}``.
    /// - ``slice`` key: a list of row dicts.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If a string key names no column.
    /// IndexError
    ///     If an integer key is out of range.
    fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(name) = key.extract::<String>() {
            return self.column(py, &name);
        }
        if let Ok(slice) = key.cast::<pyo3::types::PySlice>() {
            let idx = slice.indices(self.n_rows as isize)?;
            let mut out: Vec<Py<PyAny>> = Vec::new();
            let mut i = idx.start;
            let stop = idx.stop;
            let step = idx.step;
            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                out.push(self.row(py, i)?);
                i += step;
            }
            return Ok(PyList::new(py, out)?.into_any().unbind());
        }
        if let Ok(i) = key.extract::<isize>() {
            let n = self.n_rows as isize;
            let r = if i < 0 { i + n } else { i };
            if r < 0 || r >= n {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "row {i} out of range (n_rows = {})",
                    self.n_rows
                )));
            }
            return self.row(py, r);
        }
        Err(PyKeyError::new_err(
            "BinTable index must be a column name (str), a row index (int), or a slice",
        ))
    }

    /// Build a row dict for row index `r`.
    fn row(&self, py: Python<'_>, r: isize) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (i, name) in self.column_names.iter().enumerate() {
            let val: Py<PyAny> = match &self.columns[i] {
                PyColumn::F64(v) => v[r as usize].into_py_any(py)?,
                PyColumn::I64(v) => match v[r as usize] {
                    Some(x) => x.into_py_any(py)?,
                    None => py.None(),
                },
                PyColumn::Str(v) => v[r as usize].clone().into_py_any(py)?,
                PyColumn::Bool(v) => match v[r as usize] {
                    Some(b) => b.into_py_any(py)?,
                    None => py.None(),
                },
                PyColumn::Generic(v, shape) => {
                    bin_value_to_py(py, &v[r as usize], shape.as_deref())?
                }
            };
            dict.set_item(name, val)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Pre-decoded columns assembled into a numpy structured array
    /// (one record per row, dtype follows column types).
    ///
    /// Variable-length and string columns are exposed via
    /// ``object`` dtype; numeric scalars retain native dtype.
    /// Returned array is read-only.
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let np = py.import("numpy")?;
        let arrays = PyList::empty(py);
        let names = PyList::empty(py);
        for (i, name) in self.column_names.iter().enumerate() {
            let arr = match &self.columns[i] {
                PyColumn::F64(v) => v.clone().into_pyarray(py).into_any().unbind(),
                PyColumn::I64(v) => nullable_int_to_py(py, v)?,
                PyColumn::Str(v) => {
                    // numpy.array of strings -> unicode dtype
                    np.call_method1("array", (v.clone(),))?.unbind()
                }
                PyColumn::Bool(v) => {
                    let plain: Vec<bool> = v.iter().map(|x| x.unwrap_or(false)).collect();
                    plain.into_pyarray(py).into_any().unbind()
                }
                PyColumn::Generic(v, shape) => {
                    let lst = generic_to_pylist(py, v, shape.as_deref());
                    np.call_method1("array", (lst, "object"))?.unbind()
                }
            };
            arrays.append(arr)?;
            names.append(name)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("names", names)?;
        let rec = np.getattr("rec")?;
        let result = rec.call_method("fromarrays", (arrays,), Some(&kwargs))?;
        Ok(result.unbind())
    }

    /// Column accessor; equivalent to ``table[name]``.
    fn column(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let idx = self
            .column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| PyKeyError::new_err(format!("no column {name:?}")))?;
        let obj: Py<PyAny> = match &self.columns[idx] {
            PyColumn::F64(v) => v.clone().into_pyarray(py).into_any().unbind(),
            PyColumn::I64(v) => nullable_int_to_py(py, v)?,
            PyColumn::Str(v) => v.clone().into_py_any(py)?,
            PyColumn::Bool(v) => v.clone().into_py_any(py)?,
            PyColumn::Generic(v, shape) => generic_to_pylist(py, v, shape.as_deref()).into_any(),
        };
        freeze_if_array(py, &obj);
        Ok(obj)
    }

    /// Number of rows: ``len(table)``.
    fn __len__(&self) -> usize {
        self.n_rows
    }

    /// Membership test: ``name in table``.
    fn __contains__(&self, name: &str) -> bool {
        self.column_names.iter().any(|n| n == name)
    }

    /// Iterate column names: ``for name in table``.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let names = slf.column_names.clone();
        Ok(PyList::new(py, names)?.call_method0("__iter__")?.unbind())
    }

    /// Materialise every column as a plain ``dict[str, ndarray | list]``.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for name in &self.column_names {
            dict.set_item(name, self.column(py, name)?)?;
        }
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        use std::fmt::Write as _;
        const MAX_COLS: usize = 6;
        const MAX_ROWS: usize = 5;
        let n_cols = self.column_names.len();
        let n_rows = self.n_rows;

        let title = match extname_opt(&self.header).as_deref() {
            Some(name) => format!("BinTable(\"{name}\", {n_rows} rows \u{00d7} {n_cols} cols)"),
            None => format!("BinTable({n_rows} rows \u{00d7} {n_cols} cols)"),
        };

        if n_cols == 0 || n_rows == 0 {
            return title;
        }

        let show_cols = n_cols.min(MAX_COLS);
        let show_rows = n_rows.min(MAX_ROWS);

        let mut dtypes: Vec<&'static str> = Vec::with_capacity(show_cols);
        let mut cells: Vec<Vec<String>> = Vec::with_capacity(show_cols);
        for ci in 0..show_cols {
            let (dtype, col_cells) = bin_column_preview(&self.columns[ci], show_rows);
            dtypes.push(dtype);
            cells.push(col_cells);
        }

        let widths: Vec<usize> = (0..show_cols)
            .map(|ci| {
                let max_cell = cells[ci].iter().map(String::len).max().unwrap_or(0);
                self.column_names[ci]
                    .len()
                    .max(dtypes[ci].len())
                    .max(max_cell)
            })
            .collect();

        let col_sep = "  ";
        let indent = "  ";
        let more_tag = if n_cols > MAX_COLS {
            format!("{col_sep}\u{2026} +{} more cols", n_cols - MAX_COLS)
        } else {
            String::new()
        };

        let name_row: String = (0..show_cols)
            .map(|ci| format!("{:<width$}", self.column_names[ci], width = widths[ci]))
            .collect::<Vec<_>>()
            .join(col_sep);

        let dtype_row: String = (0..show_cols)
            .map(|ci| format!("{:<width$}", dtypes[ci], width = widths[ci]))
            .collect::<Vec<_>>()
            .join(col_sep);

        let mut out = title;
        out.push('\n');
        out.push_str(indent);
        out.push_str(&name_row);
        out.push_str(&more_tag);
        out.push('\n');
        out.push_str(indent);
        out.push_str(&dtype_row);

        #[allow(
            clippy::needless_range_loop,
            reason = "cells is column-major; ri indexes all columns simultaneously"
        )]
        for ri in 0..show_rows {
            out.push('\n');
            out.push_str(indent);
            let row: String = (0..show_cols)
                .map(|ci| {
                    let w = widths[ci];
                    let cell = &cells[ci][ri];
                    match &self.columns[ci] {
                        PyColumn::Str(_) => format!("{cell:<w$}"),
                        _ => format!("{cell:>w$}"),
                    }
                })
                .collect::<Vec<_>>()
                .join(col_sep);
            out.push_str(&row);
        }

        if n_rows > MAX_ROWS {
            out.push('\n');
            let _ = write!(out, "{indent}\u{2026} {} more rows", n_rows - MAX_ROWS);
        }

        out
    }
}

/// Convert one BinValue cell to a single Python object using the
/// same dtype rules as `generic_to_pylist`.
fn bin_value_to_py(
    py: Python<'_>,
    cell: &BinValue,
    shape: Option<&[usize]>,
) -> PyResult<Py<PyAny>> {
    let lst = generic_to_pylist(py, std::slice::from_ref(cell), shape);
    let bound = lst.bind(py);
    Ok(bound.get_item(0)?.unbind())
}

fn generic_to_pylist(py: Python<'_>, cells: &[BinValue], shape: Option<&[usize]>) -> Py<PyList> {
    // Reshape numeric cell arrays per `TDIMn` when present. FITS
    // dimension order is fastest-varying first (FORTRAN/column-major),
    // so we reverse to get C-order for numpy and only apply the
    // shape when the element count matches (a sanity check that
    // protects against malformed `TDIMn` keywords).
    let c_shape: Option<Vec<usize>> = shape.map(|s| {
        let mut v = s.to_vec();
        v.reverse();
        v
    });
    let reshape = |arr: Py<PyAny>, n: usize| -> Py<PyAny> {
        let Some(target) = c_shape.as_ref() else {
            return arr;
        };
        let prod: usize = target.iter().product();
        if prod != n || target.len() < 2 {
            return arr;
        }
        match arr.bind(py).call_method1("reshape", (target.clone(),)) {
            Ok(r) => r.unbind(),
            Err(_) => arr,
        }
    };

    let list = PyList::empty(py);
    for c in cells {
        // Each numeric variant calls a different `into_pyarray`
        // monomorphisation; clippy flags the bodies as identical but
        // they aren't (different element types).
        #[allow(
            clippy::match_same_arms,
            reason = "arms differ by generic element type, not body text"
        )]
        let obj: Py<PyAny> = match c {
            BinValue::F64(v) => {
                let n = v.len();
                reshape(v.clone().into_pyarray(py).into_any().unbind(), n)
            }
            BinValue::F32(v) => {
                let n = v.len();
                reshape(v.clone().into_pyarray(py).into_any().unbind(), n)
            }
            BinValue::Float(v) => {
                let n = v.len();
                reshape(v.clone().into_pyarray(py).into_any().unbind(), n)
            }
            BinValue::Int(v) => {
                let n = v.len();
                let arr = nullable_int_to_py(py, v).unwrap_or_else(|_| py.None());
                if arr.is_none(py) {
                    arr
                } else {
                    reshape(arr, n)
                }
            }
            BinValue::Uint(v) => {
                let n = v.len();
                let arr = if v.iter().all(Option::is_some) {
                    let plain: Vec<u64> = v.iter().map(|x| x.unwrap()).collect();
                    plain.into_pyarray(py).into_any().unbind()
                } else {
                    let lifted: Vec<Option<i64>> = v.iter().map(|x| x.map(|u| u as i64)).collect();
                    nullable_int_to_py(py, &lifted).unwrap_or_else(|_| py.None())
                };
                if arr.is_none(py) {
                    arr
                } else {
                    reshape(arr, n)
                }
            }
            BinValue::Logical(v) => v.clone().into_py_any(py).unwrap_or_else(|_| py.None()),
            BinValue::Str(s) => s.clone().into_py_any(py).unwrap_or_else(|_| py.None()),
            BinValue::C64(v) => v.clone().into_py_any(py).unwrap_or_else(|_| py.None()),
            BinValue::C128(v) => v.clone().into_py_any(py).unwrap_or_else(|_| py.None()),
            BinValue::Bits(bytes, count) => {
                // Unpack X-format bits (MSB first) into a numpy bool array
                // of length `count`, matching astropy's representation.
                let mut bits: Vec<bool> = Vec::with_capacity(*count);
                for i in 0..*count {
                    let byte = bytes.get(i / 8).copied().unwrap_or(0);
                    let bit = (byte >> (7 - (i % 8))) & 1;
                    bits.push(bit != 0);
                }
                bits.into_pyarray(py).into_any().unbind()
            }
            BinValue::Vla(inner) => {
                // Convert the inner BinValue (already decoded from heap)
                // by routing through a single-cell list.
                let inner_list = generic_to_pylist(py, std::slice::from_ref(inner), None);
                inner_list
                    .bind(py)
                    .get_item(0)
                    .map_or_else(|_| py.None(), Bound::unbind)
            }
        };
        list.append(obj).expect("append");
    }
    list.into()
}

// ---------------------------------------------------------------------
// ASCII table
// ---------------------------------------------------------------------

/// ASCII ``TABLE`` HDU.
///
/// Returned by :meth:`FitsFile.hdu` when the HDU kind is
/// ``TABLE``. Columns where every cell parses as numeric are
/// returned as :class:`numpy.ndarray` (with ``NaN`` for null
/// cells); mixed columns fall back to ``list[str]``.
#[pyclass(name = "AsciiTable", module = "fitsy")]
#[derive(Debug)]
pub struct PyAsciiTable {
    pub(crate) header: PyHeader,
    pub(crate) n_rows: usize,
    pub(crate) column_names: Vec<String>,
    /// One column per AsciiColumn, eagerly decoded. Mixed columns
    /// (`I` with TNULL holes) are lifted to f64 with NaN for null,
    /// matching the BinTable convention above.
    columns: Vec<AsciiPyColumn>,
    /// Raw on-disk data bytes captured at load time, used by
    /// :meth:`FitsFile.writeto` to round-trip the table.
    pub(crate) raw: Vec<u8>,
}

#[derive(Debug)]
enum AsciiPyColumn {
    F64(Vec<f64>),
    Str(Vec<String>),
}

impl PyAsciiTable {
    pub(crate) fn from_table(t: &AsciiTableHdu<'_>, header: PyHeader) -> PyResult<Self> {
        let n_rows = t.n_rows();
        let mut column_names = Vec::with_capacity(t.columns().len());
        let mut columns = Vec::with_capacity(t.columns().len());
        for col in t.columns() {
            column_names.push(col.name.clone());
            // Probe: if all rows are Int/Float, use F64; else Str.
            let mut all_numeric = true;
            let mut numeric: Vec<f64> = Vec::with_capacity(n_rows);
            let mut strings: Vec<String> = Vec::with_capacity(n_rows);
            for r in 0..n_rows {
                let cell = t.cell_value(r, col).into_py_result()?;
                match cell {
                    Some(AsciiCell::Int(i)) => numeric.push(i as f64),
                    Some(AsciiCell::Float(f)) => numeric.push(f),
                    Some(AsciiCell::Str(s)) => {
                        all_numeric = false;
                        strings.push(s);
                    }
                    None => numeric.push(f64::NAN),
                }
            }
            if all_numeric {
                columns.push(AsciiPyColumn::F64(numeric));
            } else {
                // We may not have collected strings for every row;
                // re-walk to get a uniform list.
                let mut all = Vec::with_capacity(n_rows);
                for r in 0..n_rows {
                    let cell = t.cell_value(r, col).into_py_result()?;
                    all.push(match cell {
                        Some(AsciiCell::Str(s)) => s,
                        Some(AsciiCell::Int(i)) => i.to_string(),
                        Some(AsciiCell::Float(f)) => f.to_string(),
                        None => String::new(),
                    });
                }
                columns.push(AsciiPyColumn::Str(all));
            }
            let _ = strings;
        }
        Ok(Self {
            header,
            n_rows,
            column_names,
            columns,
            raw: t.data_bytes().to_vec(),
        })
    }

    pub(crate) fn header_clone(&self) -> crate::Header {
        self.header.lock().clone()
    }

    /// Reconstruct a `PyAsciiTable` from a builder snapshot. Columns
    /// are not re-decoded -- accessors return `KeyError`.
    pub(crate) fn from_built_bytes(header: crate::Header, bytes: Vec<u8>) -> Self {
        Self {
            header: PyHeader::from_header_with(&header, false),
            n_rows: 0,
            column_names: Vec::new(),
            columns: Vec::new(),
            raw: bytes,
        }
    }
}

#[pymethods]
impl PyAsciiTable {
    /// The HDU header.
    #[getter]
    fn header(&self) -> PyHeader {
        self.header.clone()
    }

    /// Number of rows in the table.
    #[getter]
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// List of column names in declaration order.
    #[getter]
    fn column_names(&self) -> Vec<String> {
        self.column_names.clone()
    }

    /// Return one column by name, or one row by integer/slice.
    ///
    /// - ``str`` key: column accessor.
    /// - ``int`` key: a row dict ``{col_name: value, ...}``.
    /// - ``slice`` key: a list of row dicts.
    fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(name) = key.extract::<String>() {
            return self.column(py, &name);
        }
        if let Ok(slice) = key.cast::<pyo3::types::PySlice>() {
            let idx = slice.indices(self.n_rows as isize)?;
            let mut out: Vec<Py<PyAny>> = Vec::new();
            let mut i = idx.start;
            let stop = idx.stop;
            let step = idx.step;
            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                out.push(self.row(py, i)?);
                i += step;
            }
            return Ok(PyList::new(py, out)?.into_any().unbind());
        }
        if let Ok(i) = key.extract::<isize>() {
            let n = self.n_rows as isize;
            let r = if i < 0 { i + n } else { i };
            if r < 0 || r >= n {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "row {i} out of range (n_rows = {})",
                    self.n_rows
                )));
            }
            return self.row(py, r);
        }
        Err(PyKeyError::new_err(
            "AsciiTable index must be a column name (str), a row index (int), or a slice",
        ))
    }

    /// Build a row dict for row index `r`.
    fn row(&self, py: Python<'_>, r: isize) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (i, name) in self.column_names.iter().enumerate() {
            let val: Py<PyAny> = match &self.columns[i] {
                AsciiPyColumn::F64(v) => v[r as usize].into_py_any(py)?,
                AsciiPyColumn::Str(v) => v[r as usize].clone().into_py_any(py)?,
            };
            dict.set_item(name, val)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// All columns assembled into a numpy structured array.
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let np = py.import("numpy")?;
        let arrays = PyList::empty(py);
        let names = PyList::empty(py);
        for (i, name) in self.column_names.iter().enumerate() {
            let arr: Py<PyAny> = match &self.columns[i] {
                AsciiPyColumn::F64(v) => v.clone().into_pyarray(py).into_any().unbind(),
                AsciiPyColumn::Str(v) => np.call_method1("array", (v.clone(),))?.unbind(),
            };
            arrays.append(arr)?;
            names.append(name)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("names", names)?;
        let rec = np.getattr("rec")?;
        let result = rec.call_method("fromarrays", (arrays,), Some(&kwargs))?;
        Ok(result.unbind())
    }

    /// Column accessor; equivalent to ``table[name]``.
    fn column(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let idx = self
            .column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| PyKeyError::new_err(format!("no column {name:?}")))?;
        let obj: Py<PyAny> = match &self.columns[idx] {
            AsciiPyColumn::F64(v) => v.clone().into_pyarray(py).into_any().unbind(),
            AsciiPyColumn::Str(v) => v.clone().into_py_any(py)?,
        };
        freeze_if_array(py, &obj);
        Ok(obj)
    }

    /// Number of rows: ``len(table)``.
    fn __len__(&self) -> usize {
        self.n_rows
    }

    /// Membership test: ``name in table``.
    fn __contains__(&self, name: &str) -> bool {
        self.column_names.iter().any(|n| n == name)
    }

    /// Iterate column names: ``for name in table``.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let names = slf.column_names.clone();
        Ok(PyList::new(py, names)?.call_method0("__iter__")?.unbind())
    }

    /// Materialise every column as a plain ``dict[str, ndarray | list]``.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for name in &self.column_names {
            dict.set_item(name, self.column(py, name)?)?;
        }
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        use std::fmt::Write as _;
        const MAX_COLS: usize = 6;
        const MAX_ROWS: usize = 5;
        let n_cols = self.column_names.len();
        let n_rows = self.n_rows;

        let title = match extname_opt(&self.header).as_deref() {
            Some(name) => format!("AsciiTable(\"{name}\", {n_rows} rows \u{00d7} {n_cols} cols)"),
            None => format!("AsciiTable({n_rows} rows \u{00d7} {n_cols} cols)"),
        };

        if n_cols == 0 || n_rows == 0 {
            return title;
        }

        let show_cols = n_cols.min(MAX_COLS);
        let show_rows = n_rows.min(MAX_ROWS);

        let mut dtypes: Vec<&'static str> = Vec::with_capacity(show_cols);
        let mut cells: Vec<Vec<String>> = Vec::with_capacity(show_cols);
        for ci in 0..show_cols {
            let (dtype, col_cells) = ascii_column_preview(&self.columns[ci], show_rows);
            dtypes.push(dtype);
            cells.push(col_cells);
        }

        let widths: Vec<usize> = (0..show_cols)
            .map(|ci| {
                let max_cell = cells[ci].iter().map(String::len).max().unwrap_or(0);
                self.column_names[ci]
                    .len()
                    .max(dtypes[ci].len())
                    .max(max_cell)
            })
            .collect();

        let col_sep = "  ";
        let indent = "  ";
        let more_tag = if n_cols > MAX_COLS {
            format!("{col_sep}\u{2026} +{} more cols", n_cols - MAX_COLS)
        } else {
            String::new()
        };

        let name_row: String = (0..show_cols)
            .map(|ci| format!("{:<width$}", self.column_names[ci], width = widths[ci]))
            .collect::<Vec<_>>()
            .join(col_sep);

        let dtype_row: String = (0..show_cols)
            .map(|ci| format!("{:<width$}", dtypes[ci], width = widths[ci]))
            .collect::<Vec<_>>()
            .join(col_sep);

        let mut out = title;
        out.push('\n');
        out.push_str(indent);
        out.push_str(&name_row);
        out.push_str(&more_tag);
        out.push('\n');
        out.push_str(indent);
        out.push_str(&dtype_row);

        #[allow(
            clippy::needless_range_loop,
            reason = "cells is column-major; ri indexes all columns simultaneously"
        )]
        for ri in 0..show_rows {
            out.push('\n');
            out.push_str(indent);
            let row: String = (0..show_cols)
                .map(|ci| {
                    let w = widths[ci];
                    let cell = &cells[ci][ri];
                    match &self.columns[ci] {
                        AsciiPyColumn::Str(_) => format!("{cell:<w$}"),
                        AsciiPyColumn::F64(_) => format!("{cell:>w$}"),
                    }
                })
                .collect::<Vec<_>>()
                .join(col_sep);
            out.push_str(&row);
        }

        if n_rows > MAX_ROWS {
            out.push('\n');
            let _ = write!(out, "{indent}\u{2026} {} more rows", n_rows - MAX_ROWS);
        }

        out
    }
}
