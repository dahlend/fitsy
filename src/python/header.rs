//! `PyHeader` -- dict-like view of a fitsy `Header`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::header::{Header, Value};

/// Convert a fitsy `Value` to a native Python object.
fn value_to_py(py: Python<'_>, v: &Value) -> Py<PyAny> {
    use pyo3::IntoPyObjectExt;
    match v {
        Value::Logical(b) => b.into_py_any(py).unwrap(),
        Value::Integer(i) => i.into_py_any(py).unwrap(),
        Value::Real(f) => f.into_py_any(py).unwrap(),
        // Complex types: emit `complex(re, im)`. The integer form
        // is rare enough that flattening to f64 is fine.
        Value::ComplexInteger(r, i) => {
            pyo3::types::PyComplex::from_doubles(py, *r as f64, *i as f64)
                .into_py_any(py)
                .unwrap()
        }
        Value::ComplexReal(r, i) => pyo3::types::PyComplex::from_doubles(py, *r, *i)
            .into_py_any(py)
            .unwrap(),
        Value::String(s) => s.into_py_any(py).unwrap(),
        Value::Undefined => py.None(),
    }
}

/// Normalize a user-supplied keyword for header lookup.
///
/// FITS reserves uppercase ASCII for regular keywords and writes
/// only uppercase to disk; case-sensitive lookup turns minor
/// typos (``hdr["bitpix"]``) into silent ``None``/``KeyError`` and
/// is the source of frequent bugs. Match astropy and fold every
/// lookup to uppercase.
fn norm_key(key: &str) -> String {
    key.to_ascii_uppercase()
}

/// True if `key` is a structural / layout card whose value is
/// derived from the HDU's data array (image) or column descriptors
/// (table). User edits to these cards are silently overwritten by
/// :meth:`FitsFile.writeto`, which is a footgun -- we reject the
/// mutation up front instead.
fn is_layout_card(key: &str) -> bool {
    let k = key.trim();
    if matches!(
        k,
        "SIMPLE"
            | "BITPIX"
            | "NAXIS"
            | "EXTEND"
            | "PCOUNT"
            | "GCOUNT"
            | "XTENSION"
            | "END"
            | "GROUPS"
    ) {
        return true;
    }
    // NAXIS{n}: 1-3 ASCII digits.
    if let Some(rest) = k.strip_prefix("NAXIS")
        && !rest.is_empty()
        && rest.len() <= 3
        && rest.bytes().all(|b| b.is_ascii_digit())
    {
        return true;
    }
    false
}

/// Dict-like view of a FITS header.
///
/// A ``Header`` is an owned snapshot that is *shared* with the
/// parent HDU wrapper. Cloning the Python object yields another
/// handle to the same underlying header, so a mutation made
/// through one handle (for example ``hdu.header['FOO'] = 'bar'``)
/// is visible through every other handle.
///
/// Headers obtained from a read-only :class:`FitsFile`
/// (:func:`fitsy.open` with ``mode='readonly'``, the default) raise
/// :class:`ValueError` from every mutating method. Open the file
/// with ``mode='update'`` to enable in-memory edits.
///
/// Examples
/// --------
/// >>> with fitsy.open("image.fits") as f:
/// ...     hdr = f[0].header
/// ...     bitpix = hdr["BITPIX"]
/// ...     for key in hdr:
/// ...         print(key, hdr[key])
#[pyclass(name = "Header", module = "fitsy", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyHeader {
    pub(crate) inner: Arc<Mutex<Header>>,
    pub(crate) read_only: bool,
    /// Optional back-pointer to the parent `FitsFile`'s dirty flag.
    /// Set when the header was materialized from a file opened with
    /// `mode='update'`; `None` for standalone headers (built in
    /// Python or attached to a builder). When `Some`, every header
    /// mutation flips the bit so `flush()` / `__exit__` know they
    /// must rewrite the file.
    pub(crate) dirty: Option<Arc<AtomicBool>>,
}

impl PyHeader {
    pub(crate) fn from_header_with(h: &Header, read_only: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(h.clone())),
            read_only,
            dirty: None,
        }
    }

    /// Construct an empty, writable header.
    pub(crate) fn empty() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Header::empty())),
            read_only: false,
            dirty: None,
        }
    }

    /// Lock the inner `Header`. Panics only if a previous panic
    /// poisoned the mutex; we surface that as a normal lock since
    /// fitsy's header methods do not themselves panic.
    pub(crate) fn lock(&self) -> MutexGuard<'_, Header> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn ensure_writable(&self) -> PyResult<()> {
        if self.read_only {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "header is read-only; reopen the file with `mode='update'` to enable mutations",
            ));
        }
        if let Some(flag) = &self.dirty {
            flag.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Merge value cards from another `PyHeader` or a Python
    /// `Mapping` into `self`. Internal helper shared by the
    /// `update` pymethod and HDU constructors.
    pub(crate) fn update_from(&self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        // Fast path: another PyHeader -> copy value cards directly,
        // preserving comments.
        if let Ok(other_header) = other.extract::<PyRef<'_, Self>>() {
            let entries: Vec<(String, Value, Option<String>)> = other_header
                .lock()
                .entries()
                .iter()
                .filter(|e| e.value.is_some())
                .map(|e| {
                    (
                        e.keyword.clone(),
                        e.value.clone().unwrap(),
                        e.comment.clone(),
                    )
                })
                .collect();
            let mut me = self.lock();
            for (k, v, c) in entries {
                me.set(&k, v, c.as_deref()).map_err(super::err_to_py)?;
            }
            return Ok(());
        }
        // Mapping path: iterate (key, value).
        let items = other.call_method0("items").map_err(|_| {
            PyTypeError::new_err(
                "header.update: argument must be a Header or a mapping with .items()",
            )
        })?;
        let iter = items.try_iter()?;
        for item in iter {
            let pair = item?;
            let key: String = pair.get_item(0)?.extract()?;
            let value = pair.get_item(1)?;
            let (val, comment) = parse_setitem_value(&value)?;
            self.lock()
                .set(&key, val, comment.as_deref())
                .map_err(super::err_to_py)?;
        }
        Ok(())
    }
}

#[pymethods]
impl PyHeader {
    /// Test whether ``key`` is present (``key in header``).
    fn __contains__(&self, key: &str) -> bool {
        self.lock().contains(&norm_key(key))
    }

    /// Look up the value of a card (``header[key]``).
    ///
    /// Parameters
    /// ----------
    /// key : str or tuple[str, int]
    ///     A keyword (case-insensitive). Pass ``(keyword, n)`` to
    ///     fetch the n-th occurrence of a duplicated keyword
    ///     (negative indices count from the end).
    ///
    /// Returns
    /// -------
    /// bool, int, float, complex, str, None, or HeaderCommentary
    ///     Regular value cards return the native Python scalar
    ///     matching the FITS value type. Undefined values come
    ///     through as ``None``. Indexing with ``"COMMENT"``,
    ///     ``"HISTORY"`` or ``""`` returns a list-like
    ///     :class:`HeaderCommentary` view of every text body.
    ///     For a value card with multiple occurrences, only the
    ///     first value is returned -- use ``header[(key, n)]`` or
    ///     :meth:`cards` to access the rest.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If no card with that keyword is present.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        use crate::header::CardKind;
        use pyo3::IntoPyObjectExt;

        // Resolve (key, n)-tuple form: return the n-th occurrence's
        // value (astropy idiom for disambiguating duplicate keywords).
        if let Ok(tup) = key.cast::<PyTuple>()
            && tup.len() == 2
        {
            let kw_obj = tup.get_item(0)?;
            let n_obj = tup.get_item(1)?;
            let kw_str: String = kw_obj.extract()?;
            let n: isize = n_obj.extract()?;
            let k = norm_key(&kw_str);
            let header = self.lock();
            let matches: Vec<&crate::header::HeaderEntry> =
                header.entries().iter().filter(|e| e.keyword == k).collect();
            if matches.is_empty() {
                return Err(PyKeyError::new_err(kw_str));
            }
            let idx = if n < 0 { matches.len() as isize + n } else { n };
            if idx < 0 || (idx as usize) >= matches.len() {
                return Err(PyKeyError::new_err(format!(
                    "{kw_str}[{n}]: only {} occurrence(s) present",
                    matches.len()
                )));
            }
            let e = matches[idx as usize];
            if matches!(e.kind, CardKind::Commentary) {
                return e.commentary.clone().unwrap_or_default().into_py_any(py);
            }
            return Ok(match e.value.as_ref() {
                Some(v) => value_to_py(py, v),
                None => py.None(),
            });
        }

        let kw_str: String = key.extract()?;
        let k = norm_key(&kw_str);
        let header = self.lock();

        // Commentary keywords (COMMENT, HISTORY, blank-keyword) get
        // a list-like view object that prints newline-joined and
        // supports ``len()`` / iteration / indexing -- mirroring
        // astropy's ``_HeaderCommentaryCards``.
        if matches!(k.as_str(), "COMMENT" | "HISTORY" | "") {
            let texts: Vec<String> = header
                .entries()
                .iter()
                .filter(|e| matches!(e.kind, CardKind::Commentary) && e.keyword == k)
                .map(|e| e.commentary.clone().unwrap_or_default())
                .collect();
            if texts.is_empty() {
                return Err(PyKeyError::new_err(kw_str));
            }
            return Ok(Py::new(py, PyHeaderCommentary { lines: texts })?.into_any());
        }

        // Regular value cards: return the first occurrence's value
        // (astropy's documented behavior). Use ``header[(key, n)]``
        // or ``header.cards(key)`` for the rest.
        match header.entries().iter().find(|e| e.keyword == k) {
            None => Err(PyKeyError::new_err(kw_str)),
            Some(e) => Ok(match e.value.as_ref() {
                Some(v) => value_to_py(py, v),
                None => py.None(),
            }),
        }
    }

    /// Set or append a value card (``header[key] = value``).
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Keyword (1-8 ASCII chars, or HIERARCH form).
    /// value : bool, int, float, str, None, or tuple
    ///     A bare scalar, or a ``(value, comment)`` tuple where
    ///     ``comment`` is a string or ``None``.
    ///
    /// Notes
    /// -----
    /// If a card with this keyword already exists, its value (and
    /// comment, if supplied) is updated in place; otherwise a new
    /// card is appended.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the header is read-only.
    fn __setitem__(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_writable()?;
        let k = norm_key(key);
        if is_layout_card(&k) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "header[{key:?}] is a structural card managed by the writer; \
                 it is recomputed from the data array on writeto. Use \
                 `hdu.data = new_array.astype(...)` to change BITPIX/NAXIS."
            )));
        }
        let (val, comment) = parse_setitem_value(value)?;
        self.lock()
            .set(&k, val, comment.as_deref())
            .map_err(super::err_to_py)?;
        Ok(())
    }

    /// Remove every value card with the given keyword (``del header[key]``).
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If no card matches.
    /// ValueError
    ///     If the header is read-only.
    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        self.ensure_writable()?;
        let k = norm_key(key);
        if is_layout_card(&k) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "header[{key:?}] is a structural card and cannot be deleted; \
                 it is recomputed from the data array on writeto."
            )));
        }
        let removed = self.lock().remove(&k);
        if removed == 0 {
            Err(PyKeyError::new_err(key.to_string()))
        } else {
            Ok(())
        }
    }

    /// Append a commentary card.
    ///
    /// Parameters
    /// ----------
    /// kind : {'COMMENT', 'HISTORY', ''}
    ///     Commentary kind. The empty string emits a blank-keyword
    ///     commentary card.
    /// text : str
    ///     Commentary text. Long lines are split across multiple
    ///     80-byte cards on serialization.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``kind`` is not one of the recognized values.
    /// ValueError
    ///     If the header is read-only.
    fn add_commentary(&mut self, kind: &str, text: &str) -> PyResult<()> {
        use crate::header::CommentaryKind;
        self.ensure_writable()?;
        let k = match kind.to_ascii_uppercase().as_str() {
            "COMMENT" => CommentaryKind::Comment,
            "HISTORY" => CommentaryKind::History,
            "" => CommentaryKind::Blank,
            other => {
                return Err(PyTypeError::new_err(format!(
                    "commentary kind must be 'COMMENT', 'HISTORY', or '' (got {other:?})"
                )));
            }
        };
        self.lock().push_commentary(k, text);
        Ok(())
    }

    /// Set a header card with optional positional placement.
    ///
    /// Equivalent to ``astropy.io.fits.Header.set``: if the keyword
    /// already exists its value (and comment, if supplied) are
    /// updated in place; otherwise a new card is appended unless
    /// ``before`` or ``after`` is given, in which case the new card
    /// is inserted at that position.
    ///
    /// Parameters
    /// ----------
    /// keyword : str
    ///     Card keyword. May be a HIERARCH name.
    /// value : Any, optional
    ///     New value. If omitted (and the card already exists), only
    ///     the comment is updated.
    /// comment : str, optional
    ///     New comment. ``None`` leaves the existing comment intact
    ///     when updating, or emits no comment when inserting.
    /// before : str, optional
    ///     Insert the new card immediately before the first card
    ///     whose keyword equals this. Ignored if `keyword` already
    ///     exists.
    /// after : str, optional
    ///     Insert the new card immediately after the first card
    ///     whose keyword equals this. Ignored if `keyword` already
    ///     exists. Mutually exclusive with `before`.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If both `before` and `after` are supplied, or if the
    ///     header is read-only.
    /// KeyError
    ///     If the named `before`/`after` card does not exist.
    #[pyo3(signature = (keyword, value=None, comment=None, *, before=None, after=None))]
    fn set(
        &mut self,
        keyword: &str,
        value: Option<Bound<'_, PyAny>>,
        comment: Option<&str>,
        before: Option<&str>,
        after: Option<&str>,
    ) -> PyResult<()> {
        self.ensure_writable()?;
        if before.is_some() && after.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Header.set: 'before' and 'after' are mutually exclusive",
            ));
        }
        let k = norm_key(keyword);
        if is_layout_card(&k) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Header.set: {keyword:?} is a structural card managed by the writer"
            )));
        }
        let mut h = self.lock();
        // If the card already exists, just update value/comment in place.
        if h.contains(&k) {
            let val = match value {
                Some(v) => py_to_value(&v)?,
                None => h
                    .first(&k)
                    .cloned()
                    .ok_or_else(|| PyKeyError::new_err(keyword.to_string()))?,
            };
            h.set(&k, val, comment).map_err(super::err_to_py)?;
            return Ok(());
        }
        // Inserting a new card.
        let val = match value {
            Some(v) => py_to_value(&v)?,
            None => Value::Undefined,
        };
        if let Some(after_k) = after {
            let after_k = norm_key(after_k);
            if !h.contains(&after_k) {
                return Err(PyKeyError::new_err(after_k));
            }
            h.set_after(&after_k, k, val, comment)
                .map_err(super::err_to_py)?;
        } else if let Some(before_k) = before {
            let before_k = norm_key(before_k);
            if !h.contains(&before_k) {
                return Err(PyKeyError::new_err(before_k));
            }
            h.set_before(&before_k, k, val, comment)
                .map_err(super::err_to_py)?;
        } else {
            h.push(k, val, comment).map_err(super::err_to_py)?;
        }
        Ok(())
    }

    /// Insert a value card at a specified position.
    ///
    /// Parameters
    /// ----------
    /// position : int or str
    ///     Integer index (0 = first card), or the keyword of an
    ///     existing card; in the latter case the new card is
    ///     inserted before/after it depending on `after`.
    /// keyword : str
    ///     Card keyword.
    /// value : Any, optional
    ///     Card value. ``None`` records an undefined-value card.
    /// comment : str, optional
    ///     Optional inline comment.
    /// after : bool, optional
    ///     When `position` is a keyword, set ``after=True`` to insert
    ///     the new card just after that card rather than before it.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If `position` is a keyword that does not exist.
    /// ValueError
    ///     If the header is read-only.
    #[pyo3(signature = (position, keyword, value=None, comment=None, *, after=false))]
    fn insert(
        &mut self,
        position: Bound<'_, PyAny>,
        keyword: &str,
        value: Option<Bound<'_, PyAny>>,
        comment: Option<&str>,
        after: bool,
    ) -> PyResult<()> {
        self.ensure_writable()?;
        let k = norm_key(keyword);
        if is_layout_card(&k) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Header.insert: {keyword:?} is a structural card managed by the writer"
            )));
        }
        let val = match value {
            Some(v) => py_to_value(&v)?,
            None => Value::Undefined,
        };
        let mut h = self.lock();
        if let Ok(idx) = position.extract::<usize>() {
            h.insert(idx, k, val, comment).map_err(super::err_to_py)?;
        } else if let Ok(anchor_str) = position.extract::<String>() {
            let anchor = norm_key(&anchor_str);
            if !h.contains(&anchor) {
                return Err(PyKeyError::new_err(anchor));
            }
            if after {
                h.set_after(&anchor, k, val, comment)
                    .map_err(super::err_to_py)?;
            } else {
                h.set_before(&anchor, k, val, comment)
                    .map_err(super::err_to_py)?;
            }
        } else {
            return Err(PyTypeError::new_err(
                "Header.insert: position must be int or str",
            ));
        }
        Ok(())
    }

    /// Rename every value card whose keyword equals `oldname` to
    /// use `newname`.
    ///
    /// Parameters
    /// ----------
    /// oldname : str
    ///     Existing keyword.
    /// newname : str
    ///     Replacement keyword. Must be a valid FITS or HIERARCH
    ///     keyword.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If no card with `oldname` exists.
    /// ValueError
    ///     If the header is read-only or `newname` is not a valid
    ///     keyword.
    fn rename_keyword(&mut self, oldname: &str, newname: &str) -> PyResult<()> {
        self.ensure_writable()?;
        let old = norm_key(oldname);
        let new = norm_key(newname);
        if is_layout_card(&old) || is_layout_card(&new) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Header.rename_keyword: cannot rename a structural card",
            ));
        }
        let renamed = self
            .lock()
            .rename_keyword(&old, &new)
            .map_err(super::err_to_py)?;
        if renamed == 0 {
            return Err(PyKeyError::new_err(oldname.to_string()));
        }
        Ok(())
    }

    /// Merge another header (or a ``str``-keyed mapping) into this one.
    ///
    /// For each ``(key, value)`` in ``other``, behaves like
    /// ``self[key] = value``: existing keywords are overwritten in
    /// place, new keywords are appended.
    ///
    /// Commentary cards (``COMMENT``, ``HISTORY``, blank-keyword) are
    /// **not** copied; use :meth:`add_commentary` if you want to
    /// transfer them explicitly.
    ///
    /// Parameters
    /// ----------
    /// other : Header or Mapping[str, Any]
    ///     The values to merge in. ``Header`` instances copy their
    ///     value cards; mappings are iterated in declaration order.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the header is read-only.
    /// TypeError
    ///     If ``other`` is neither a ``Header`` nor a string-keyed
    ///     mapping.
    fn update(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_writable()?;
        self.update_from(other)
    }

    /// Number of populated cards (``len(header)``).
    ///
    /// The trailing ``END`` card and blank padding cards are
    /// excluded.
    fn __len__(&self) -> usize {
        self.lock().entries().len()
    }

    /// Iterate over keyword strings in declaration order.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<HeaderKeyIter>> {
        let keys: Vec<String> = slf
            .lock()
            .entries()
            .iter()
            .map(|e| e.keyword.clone())
            .collect();
        Py::new(slf.py(), HeaderKeyIter { keys, pos: 0 })
    }

    /// Non-raising lookup (``header.get(key, default=None)``).
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Keyword to look up.
    /// default : object, optional
    ///     Value to return if ``key`` is absent. Defaults to ``None``.
    ///
    /// Returns
    /// -------
    /// object
    ///     The matching value, or ``default`` if absent.
    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> Py<PyAny> {
        let k = norm_key(key);
        if let Some(v) = self.lock().first(&k) {
            value_to_py(py, v)
        } else {
            default.unwrap_or_else(|| py.None())
        }
    }

    /// All keywords in declaration order.
    ///
    /// Duplicates are kept, matching FITS semantics where
    /// ``HISTORY`` and ``COMMENT`` cards repeat.
    fn keys(&self) -> Vec<String> {
        self.lock()
            .entries()
            .iter()
            .map(|e| e.keyword.clone())
            .collect()
    }

    /// All ``(keyword, value)`` pairs in declaration order.
    ///
    /// Commentary cards (COMMENT, HISTORY, blank) report ``None``
    /// for the value.
    ///
    /// Returns
    /// -------
    /// list of tuple
    fn items(&self, py: Python<'_>) -> Py<PyList> {
        use pyo3::IntoPyObjectExt;
        let list = PyList::empty(py);
        for e in self.lock().entries() {
            let v = e
                .value
                .as_ref()
                .map_or_else(|| py.None(), |v| value_to_py(py, v));
            let tup = PyTuple::new(py, [e.keyword.clone().into_py_any(py).unwrap(), v])
                .expect("PyTuple::new");
            list.append(tup).expect("append");
        }
        list.into()
    }

    /// Inline comment for the first card with this keyword.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Keyword (case-insensitive match).
    ///
    /// Returns
    /// -------
    /// str or None
    ///     The comment text, or ``None`` if no such card exists or
    /// the matching card has no inline comment.
    fn comment(&self, key: &str) -> Option<String> {
        self.lock()
            .entries()
            .iter()
            .find(|e| e.keyword.eq_ignore_ascii_case(key))
            .and_then(|e| e.comment.clone())
    }

    /// Plain ``dict`` view of the header.
    ///
    /// Comments are dropped; duplicate keywords are deduplicated by
    /// last-seen value. Convenience for ad-hoc work; round-trip
    /// fidelity requires :meth:`items`.
    fn to_dict(&self, py: Python<'_>) -> Py<PyDict> {
        let d = PyDict::new(py);
        for e in self.lock().entries() {
            if let Some(v) = e.value.as_ref() {
                d.set_item(&e.keyword, value_to_py(py, v)).expect("set");
            }
        }
        d.into()
    }

    /// Return every card matching ``key`` as a list of
    /// ``(value, comment)`` tuples, in declaration order.
    ///
    /// Useful when a keyword appears more than once and you need
    /// programmatic access to every occurrence (the indexed
    /// accessor only returns the first). Commentary cards yield
    /// ``(text, None)``. Returns an empty list if no match is found.
    fn cards(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyList>> {
        use crate::header::CardKind;
        use pyo3::IntoPyObjectExt;

        let k = norm_key(key);
        let header = self.lock();
        let list = PyList::empty(py);
        for e in header.entries().iter().filter(|e| e.keyword == k) {
            let value: Py<PyAny> = if matches!(e.kind, CardKind::Commentary) {
                e.commentary.clone().unwrap_or_default().into_py_any(py)?
            } else {
                match e.value.as_ref() {
                    Some(v) => value_to_py(py, v),
                    None => py.None(),
                }
            };
            let comment: Py<PyAny> = match e.comment.as_ref() {
                Some(c) => c.clone().into_py_any(py)?,
                None => py.None(),
            };
            let tup = PyTuple::new(py, [value, comment])?;
            list.append(tup)?;
        }
        Ok(list.unbind())
    }

    /// Serialize the header as a single string of 80-character FITS
    /// cards (no separators, terminated by ``END`` and padded to a
    /// 2880-byte block). Mirrors ``astropy.io.fits.Header.tostring``
    /// so this object can be fed to ``astropy.wcs.WCS`` via
    /// ``fits.Header.fromstring(h.tostring())``.
    fn tostring(&self) -> PyResult<String> {
        let bytes = self.lock().to_bytes().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialize header: {e}"))
        })?;
        String::from_utf8(bytes).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("non-ASCII header bytes: {e}"))
        })
    }

    /// Raw FITS bytes (``bytes(header)``). Same content as
    /// :meth:`tostring` but returned as ``bytes``.
    fn __bytes__(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let bytes = self.lock().to_bytes().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialize header: {e}"))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
    }

    fn __repr__(&self) -> String {
        // Mirror astropy.io.fits.Header.__repr__: render every card
        // as a fixed 80-character FITS string, one per line, in
        // declaration order. Padding cards (trailing blanks added to
        // round up to a 2880-byte block) and the closing ``END``
        // card are omitted. Falls back to a one-line summary if
        // serialization fails (e.g. malformed values).
        let header = self.lock();
        if let Ok(bytes) = header.to_bytes() {
            let mut out = String::with_capacity(bytes.len() + bytes.len() / 80);
            for chunk in bytes.chunks(80) {
                let card = String::from_utf8_lossy(chunk);
                let trimmed = card.trim_end();
                if trimmed.is_empty() || trimmed == "END" {
                    continue;
                }
                if !out.is_empty() {
                    out.push('\n');
                }
                // Astropy keeps the 80-char card padded with
                // trailing spaces; preserve that.
                out.push_str(card.as_ref().trim_end_matches('\0'));
            }
            out
        } else {
            let n = header.entries().len();
            if self.read_only {
                format!("Header(<{n} cards, read-only>)")
            } else {
                format!("Header(<{n} cards>)")
            }
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// True when this header was obtained from a read-only file.
    ///
    /// In that case, mutating methods raise :class:`ValueError`.
    #[getter]
    fn read_only(&self) -> bool {
        self.read_only
    }
}

#[pyclass]
#[derive(Debug)]
pub struct HeaderKeyIter {
    keys: Vec<String>,
    pos: usize,
}

#[pymethods]
impl HeaderKeyIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<String> {
        if slf.pos < slf.keys.len() {
            let i = slf.pos;
            slf.pos += 1;
            Some(slf.keys[i].clone())
        } else {
            None
        }
    }
}

/// List-like view of every commentary card body that shares a
/// keyword (``COMMENT``, ``HISTORY``, blank-keyword). Returned by
/// :meth:`fitsy.Header.__getitem__` for those keywords; mirrors
/// ``astropy.io.fits.header._HeaderCommentaryCards``.
///
/// - ``len(view)`` -- number of cards
/// - ``view[i]``   -- text body of the i-th card
/// - ``str(view)`` / ``repr(view)`` -- newline-joined bodies
/// - iterable
#[pyclass(name = "HeaderCommentary", module = "fitsy", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyHeaderCommentary {
    lines: Vec<String>,
}

#[pymethods]
impl PyHeaderCommentary {
    fn __len__(&self) -> usize {
        self.lines.len()
    }

    fn __getitem__(&self, mut idx: isize) -> PyResult<String> {
        let n = self.lines.len() as isize;
        if idx < 0 {
            idx += n;
        }
        if idx < 0 || idx >= n {
            return Err(PyKeyError::new_err(format!(
                "index {idx} out of range (len={n})"
            )));
        }
        Ok(self.lines[idx as usize].clone())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        Ok(PyList::new(py, &slf.lines)?
            .call_method0("__iter__")?
            .unbind())
    }

    fn __str__(&self) -> String {
        self.lines.join("\n")
    }

    fn __repr__(&self) -> String {
        self.lines.join("\n")
    }
}

/// Build a `Header` from a Python `dict[str, value]`. Used by the
/// writer wrappers. Comments are not supported via dict; callers
/// who need them should use the lower-level builder.
pub(crate) fn header_from_dict(d: &Bound<'_, PyDict>) -> PyResult<Header> {
    let mut h = Header::empty();
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        let val = py_to_value(&v)?;
        h.push(key, val, None).map_err(super::err_to_py)?;
    }
    Ok(h)
}

/// Parse the right-hand side of `header[key] = ...`. Accepts either
/// a bare scalar or a `(value, comment_str)` tuple.
fn parse_setitem_value(v: &Bound<'_, PyAny>) -> PyResult<(Value, Option<String>)> {
    if let Ok(t) = v.cast::<PyTuple>()
        && t.len() == 2
    {
        let val = py_to_value(&t.get_item(0)?)?;
        let comment_obj = t.get_item(1)?;
        let comment: Option<String> = if comment_obj.is_none() {
            None
        } else {
            Some(comment_obj.extract()?)
        };
        return Ok((val, comment));
    }
    Ok((py_to_value(v)?, None))
}

/// Best-effort `PyAny` -> `Value` coercion.
fn py_to_value(v: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(b) = v.extract::<bool>() {
        return Ok(Value::Logical(b));
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(Value::Integer(i));
    }
    if let Ok(f) = v.extract::<f64>() {
        return Ok(Value::Real(f));
    }
    // Python `complex` -> FITS Value::ComplexReal so headers with
    // complex cards round-trip through `header[k] = read[k]`.
    // Whole-number reals collapse to ComplexInteger so reads of
    // integer-typed cards stay integer-typed on rewrite.
    if let Ok(c) = v.cast::<pyo3::types::PyComplex>() {
        let re: f64 = c.getattr("real")?.extract()?;
        let im: f64 = c.getattr("imag")?.extract()?;
        if re.fract() == 0.0
            && im.fract() == 0.0
            && re.abs() <= i64::MAX as f64
            && im.abs() <= i64::MAX as f64
        {
            return Ok(Value::ComplexInteger(re as i64, im as i64));
        }
        return Ok(Value::ComplexReal(re, im));
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(Value::String(s));
    }
    if v.is_none() {
        return Ok(Value::Undefined);
    }
    Err(PyTypeError::new_err(format!(
        "cannot convert {:?} into a FITS header value",
        v.get_type().name()?,
    )))
}
