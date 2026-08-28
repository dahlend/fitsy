//! `PyHeader` / `PyHeaderCommentary` -- dict-like view of a fitsy
//! `Header` and the list-like view returned for a commentary
//! keyword.
//!
//! A card's FITS value converts to a native Python scalar: logical
//! to `bool`, integer to `int`, real to `float`, complex to
//! `complex`, string to `str`, and an undefined value to `None`. The
//! reverse conversion, in [`py_to_value`], accepts the same six
//! Python types. A `COMMENT`, `HISTORY` or blank-keyword card holds
//! no value; reading one through `header[key]` returns a
//! [`PyHeaderCommentary`] instead.
//!
//! [`is_layout_card`] rejects every mutation of a structural card --
//! the keywords an HDU writer recomputes from the data array or
//! column descriptors on `writeto`. A header obtained from a
//! read-only file additionally rejects every mutating method with
//! `ValueError`.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::header::{CARD_SIZE, Header, Level, Value};
use crate::io::block::BLOCK_SIZE;

/// Convert a fitsy `Value` to a native Python object.
///
/// `Logical` becomes `bool`, `Integer` becomes `int`, `Real` becomes
/// `float`, `ComplexInteger` and `ComplexReal` both become
/// `complex`, `String` and `Unparsed` both become `str`, and
/// `Undefined` becomes `None`. `Unparsed` holds a card whose value
/// field matched no standard FITS type during lenient parsing; its
/// raw text is exposed verbatim rather than dropped.
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
        // Unparsed text is preserved verbatim by lenient parsing; expose
        // the raw string so the card is at least inspectable from Python.
        Value::String(s) | Value::Unparsed(s) => s.into_py_any(py).unwrap(),
        Value::Undefined => py.None(),
    }
}

/// Normalize a user-supplied keyword for header lookup.
///
/// FITS reserves uppercase ASCII for regular keywords and writes
/// only uppercase to disk; case-sensitive lookup turns minor
/// typos (``hdr["bitpix"]``) into silent ``None``/``KeyError`` and
/// is the source of frequent bugs. Fold every lookup to uppercase
/// instead.
fn norm_key(key: &str) -> String {
    key.to_ascii_uppercase()
}

/// True if `key` is a structural layout card. The value of such a
/// card comes from the data array of an image HDU, or from the
/// column descriptors of a table HDU.
///
/// `FitsFile::writeto` recomputes these cards and overwrites any
/// user edit without a report. The bindings reject the mutation at
/// the call site instead, so the caller sees the failure.
///
/// The rejected set is `SIMPLE`, `BITPIX`, `NAXIS`, `EXTEND`,
/// `PCOUNT`, `GCOUNT`, `XTENSION`, `END`, `GROUPS`, and `NAXISn` for
/// `n` of 1 to 3 ASCII digits.
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
/// A ``Header`` is shared with its parent HDU: copying the Python
/// object gives another handle to the same header, so an edit through
/// one is visible through all of them.
///
/// Headers from a read-only :class:`FitsFile` raise
/// :class:`ValueError` from every mutating method: :meth:`__setitem__`,
/// :meth:`__delitem__`, :meth:`set`, :meth:`insert`,
/// :meth:`add_commentary`, :meth:`rename_keyword`, and :meth:`update`.
/// Open the file with ``mode='update'`` to allow in-memory edits.
///
/// Notes
/// -----
/// A card's value converts to a native Python scalar: a logical to
/// ``bool``, an integer to ``int``, a real to ``float``, a complex
/// value to ``complex``, and a string (including one assembled from
/// ``CONTINUE`` cards) to ``str``. An undefined value converts to
/// ``None``. Writing a value back accepts the same six Python types;
/// any other type raises :class:`TypeError`. A ``COMMENT``,
/// ``HISTORY`` or blank-keyword card holds no value of its own;
/// ``header[key]`` returns a :class:`HeaderCommentary` for one of
/// these three keywords instead.
///
/// A keyword is matched case-insensitively: ``hdr["bitpix"]`` and
/// ``hdr["BITPIX"]`` name the same card. A hyphenated keyword such
/// as ``MJD-OBS`` also matches a card some writers store with an
/// underscore instead (``MJD_OBS``).
///
/// :meth:`__setitem__`, :meth:`__delitem__`, :meth:`set`,
/// :meth:`insert` and :meth:`rename_keyword` reject a structural
/// card: ``SIMPLE``, ``BITPIX``, ``NAXIS``, ``EXTEND``, ``PCOUNT``,
/// ``GCOUNT``, ``XTENSION``, ``END``, ``GROUPS``, or ``NAXISn``. An
/// HDU writer recomputes these from the data array or column
/// descriptors, so a direct edit would be silently overwritten.
/// Constructing a :class:`Header` from a mapping, and
/// :meth:`update`, do not apply this check: a mapping or another
/// :class:`Header` carrying a structural keyword is accepted
/// unchanged.
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
    pub(crate) dirty: Option<Arc<crate::python::file::DirtyFlags>>,
}

impl PyHeader {
    /// Wrap a clone of `h` in a new `PyHeader`. `read_only` controls
    /// whether the mutating pymethods later reject an edit.
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

    /// Parse a header from a buffer of concatenated 80-byte cards.
    ///
    /// The buffer is normalized before parsing so that both a full
    /// 2880-byte block dump (the output of `Header::to_bytes`) and a
    /// bare card fragment are accepted: a partial trailing card is
    /// padded to 80 bytes, an `END` card is appended when absent (it is
    /// the header/data delimiter the parser requires), and the whole is
    /// padded to a 2880-byte block boundary. Backs `fromstring` /
    /// `frombytes`.
    fn from_card_bytes(data: &[u8], lenient: bool) -> PyResult<Self> {
        let mut buf = data.to_vec();
        // Pad a partial trailing card out to a full 80 bytes.
        while !buf.len().is_multiple_of(CARD_SIZE) {
            buf.push(b' ');
        }
        // The parser requires an END card to delimit the header; append
        // one when the caller's text has none (e.g. a single-card snippet).
        let has_end = buf
            .as_chunks::<CARD_SIZE>()
            .0
            .iter()
            .any(|card| is_end_card(card, lenient));
        if !has_end {
            let mut end = [b' '; CARD_SIZE];
            end[..3].copy_from_slice(b"END");
            buf.extend_from_slice(&end);
        }
        // Only whole 2880-byte blocks are scanned; pad up to the boundary.
        while !buf.len().is_multiple_of(BLOCK_SIZE) {
            buf.push(b' ');
        }
        let (header, _) = Header::parse_with(&buf, 0, lenient).map_err(super::err_to_py)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(header)),
            read_only: false,
            dirty: None,
        })
    }

    /// Lock the inner `Header`. Panics only if a previous panic
    /// poisoned the mutex; we surface that as a normal lock since
    /// fitsy's header methods do not themselves panic.
    pub(crate) fn lock(&self) -> MutexGuard<'_, Header> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Reject a mutation on a read-only header; otherwise mark the
    /// parent file dirty when one is attached.
    ///
    /// Every mutating pymethod calls this first, before touching the
    /// underlying `Header`.
    ///
    /// # Errors
    ///
    /// Returns a Python `ValueError` if `self.read_only` is `true`.
    fn ensure_writable(&self) -> PyResult<()> {
        if self.read_only {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "header is read-only; reopen the file with `mode='update'` to enable mutations",
            ));
        }
        if let Some(flag) = &self.dirty {
            flag.definite.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Merge value cards from another `PyHeader` or a Python
    /// `Mapping` into `self`. Internal helper shared by the
    /// `update` pymethod and HDU constructors.
    ///
    /// Unlike `__setitem__`, this does not call [`is_layout_card`]:
    /// a structural keyword present in `other` is copied into `self`
    /// unchecked.
    ///
    /// Both branches reach `Header::set` with an upper-case keyword.
    /// The `PyHeader` fast path reuses `other`'s own entries, whose
    /// keywords are already upper case because every write path that
    /// produced them enforced that. The mapping branch calls
    /// [`norm_key`], as [`header_from_py`] and `__setitem__` do.
    ///
    /// # Errors
    ///
    /// Returns a Python `TypeError` if `other` is neither a
    /// `PyHeader` nor an object with an `.items()` method, or if one
    /// of its values cannot convert to a FITS value ([`py_to_value`]).
    /// Returns `fitsy.FitsError` if a copied keyword fails
    /// validation -- too long, or an invalid character.
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
            // `Header::set` requires an uppercase keyword. Normalize
            // here so a mapping key behaves the same as `header[key]`
            // and as a key passed to the constructor.
            self.lock()
                .set(&norm_key(&key), val, comment.as_deref())
                .map_err(super::err_to_py)?;
        }
        Ok(())
    }
}

#[pymethods]
impl PyHeader {
    /// Construct a new, writable header.
    ///
    /// Parameters
    /// ----------
    /// mapping : Header or mapping, optional
    ///   Initial cards. A :class:`Header` is deep-copied, including
    ///   commentary and structural keywords. A ``dict`` -- or anything
    ///   with ``.items()`` -- is inserted key by key, folding keywords
    ///   to upper case and accepting a ``(value, comment)`` tuple.
    ///   Omit for an empty header.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///   If `mapping` is neither a :class:`Header` nor an object
    ///   with an ``.items()`` method, or if one of its values is not
    ///   a ``bool``, ``int``, ``float``, ``complex``, ``str`` or
    ///   ``None``.
    /// FitsError
    ///   If a keyword in `mapping` exceeds 8 characters or contains
    ///   an invalid character (a HIERARCH keyword is exempt from the
    ///   length limit).
    ///
    /// Notes
    /// -----
    /// The result is standalone and writable: attach it to an HDU or
    /// serialize it with :meth:`tostring`. Use :meth:`fromstring` /
    /// :meth:`frombytes` to parse existing card text.
    ///
    /// A ``dict`` entry does not go through the same structural-card
    /// check as ``header[key] = value``: a mapping that carries
    /// ``SIMPLE`` or another structural keyword is accepted here.
    ///
    /// Examples
    /// --------
    /// >>> h = fitsy.Header()
    /// >>> h = fitsy.Header({"OBJECT": "M31", "EXPTIME": (30.0, "s")})
    #[new]
    #[pyo3(signature = (mapping=None))]
    fn py_new(mapping: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(obj) = mapping else {
            return Ok(Self::empty());
        };
        let core = header_from_py(obj)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(core)),
            read_only: false,
            dirty: None,
        })
    }

    /// Test whether ``key`` is present (``key in header``).
    ///
    /// `key` is matched case-insensitively.
    fn __contains__(&self, key: &str) -> bool {
        self.lock().contains(&norm_key(key))
    }

    /// Look up the value of a card (``header[key]``).
    ///
    /// Parameters
    /// ----------
    /// key : str or tuple[str, int]
    ///   A keyword (case-insensitive). Pass ``(keyword, n)`` to
    ///   fetch the n-th occurrence of a duplicated keyword
    ///   (negative indices count from the end).
    ///
    /// Returns
    /// -------
    /// bool, int, float, complex, str, None, or HeaderCommentary
    ///   The native Python scalar for the card's FITS type, or ``None``
    ///   for an undefined value. For a plain ``str`` `key` of
    ///   ``"COMMENT"``, ``"HISTORY"`` or ``""``, a list-like
    ///   :class:`HeaderCommentary` of every text body with that
    ///   keyword. For a ``(key, n)`` tuple naming one of those three
    ///   keywords, the single ``str`` text body of the n-th card
    ///   instead. A duplicated value keyword returns its first
    ///   value with a plain ``str`` `key`; use ``header[(key, n)]``
    ///   or :meth:`cards` for the rest.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///   If no card with that keyword is present, or if a
    ///   ``(keyword, n)`` tuple names fewer than ``n`` occurrences.
    /// TypeError
    ///   If `key` is not a ``str`` or a 2-element tuple.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        use crate::header::CardKind;
        use pyo3::IntoPyObjectExt;

        // Resolve (key, n)-tuple form: return the n-th occurrence's
        // value, to disambiguate a duplicated keyword.
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
        // supports len() / iteration / indexing over every card
        // sharing that keyword.
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

        // Regular value cards: return the first occurrence's value.
        // Use ``header[(key, n)]`` or ``header.cards(key)`` for the rest.
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
    ///   Keyword (1-8 ASCII chars, or HIERARCH form).
    /// value : bool, int, float, complex, str, None, or tuple
    ///   A bare scalar, or a ``(value, comment)`` tuple where
    ///   ``value`` is one of the same scalar types and ``comment``
    ///   is a ``str`` or ``None``.
    ///
    /// Notes
    /// -----
    /// If a card with this keyword already exists, its value is
    /// replaced. Its comment is replaced too when `value` is a
    /// ``(value, comment)`` tuple whose ``comment`` is not ``None``;
    /// a bare `value`, or a tuple with a ``None`` comment, leaves the
    /// existing comment untouched. If no card with this keyword
    /// exists, a new one is appended.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the header is read-only, or if `key` names a structural
    ///   card (see the class Notes for the full list).
    /// TypeError
    ///   If `value` is not one of the accepted types, or is a tuple
    ///   whose second element is not a ``str`` or ``None``.
    /// FitsError
    ///   If `key` exceeds 8 characters or contains an invalid
    ///   character (a HIERARCH keyword is exempt from the length
    ///   limit).
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
    /// `key` is matched case-insensitively. Commentary cards
    /// (``COMMENT``, ``HISTORY``, blank-keyword) are never removed by
    /// this method, even when `key` names one of those keywords.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///   If no value card matches.
    /// ValueError
    ///   If the header is read-only, or if `key` names a structural
    ///   card (see the class Notes for the full list).
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
    ///   Commentary kind. The empty string emits a blank-keyword
    ///   commentary card.
    /// text : str
    ///   Commentary text. Long lines are split across multiple
    ///   80-byte cards on serialization.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///   If ``kind`` is not one of the recognized values.
    /// ValueError
    ///   If the header is read-only.
    ///
    /// Notes
    /// -----
    /// `text` is not checked for non-ASCII bytes here. A non-ASCII
    /// byte only raises, as :class:`ValueError`, when the header is
    /// later serialized with :meth:`tostring`, ``bytes(header)``, or
    /// a file write.
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
    /// If `keyword` already exists, its value is replaced (or kept,
    /// if `value` is omitted) and its comment is replaced when
    /// `comment` is given. Otherwise a new card is appended, unless
    /// `before` or `after` is given, in which case the new card is
    /// inserted at that position.
    ///
    /// Parameters
    /// ----------
    /// keyword : str
    ///   Card keyword. May be a HIERARCH name.
    /// value : bool, int, float, complex, str, or None, optional
    ///   New value. If omitted and the card already exists, only
    ///   the comment is updated and the existing value is kept. If
    ///   omitted and the card does not exist, the new card is
    ///   inserted with an undefined value. Default ``None``.
    /// comment : str, optional
    ///   New comment. ``None`` leaves the existing comment intact
    ///   when updating, or emits no comment when inserting. Default
    ///   ``None``.
    /// before : str, optional
    ///   Insert the new card immediately before the first card
    ///   whose keyword equals this. Ignored if `keyword` already
    ///   exists. Default ``None``.
    /// after : str, optional
    ///   Insert the new card immediately after the first card
    ///   whose keyword equals this. Ignored if `keyword` already
    ///   exists. Mutually exclusive with `before`. Default ``None``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If both `before` and `after` are supplied, if the header
    ///   is read-only, or if `keyword` names a structural card (see
    ///   the class Notes for the full list).
    /// KeyError
    ///   If the named `before`/`after` card does not exist.
    /// TypeError
    ///   If `value` is not one of the accepted types.
    /// FitsError
    ///   If `keyword` exceeds 8 characters or contains an invalid
    ///   character (a HIERARCH keyword is exempt from the length
    ///   limit).
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
    /// A duplicate: if `keyword` already has a card, a second card
    /// with the same keyword is inserted rather than replacing it.
    ///
    /// Parameters
    /// ----------
    /// position : int or str
    ///   Integer index (0 = first card; an index at or past the
    ///   current card count appends at the end), or the keyword of
    ///   an existing card, in which case the new card is inserted
    ///   before or after it depending on `after`.
    /// keyword : str
    ///   Card keyword. May be a HIERARCH name.
    /// value : bool, int, float, complex, str, or None, optional
    ///   Card value. ``None`` (the default) records an
    ///   undefined-value card.
    /// comment : str, optional
    ///   Inline comment. Default ``None``, which emits no comment.
    /// after : bool, optional
    ///   When `position` is a keyword, set ``after=True`` to insert
    ///   the new card just after that card rather than before it.
    ///   Default ``False``. Ignored when `position` is an integer.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///   If `position` is a keyword that does not exist.
    /// TypeError
    ///   If `position` is neither ``int`` nor ``str``, or if `value`
    ///   is not one of the accepted types.
    /// ValueError
    ///   If the header is read-only, or if `keyword` names a
    ///   structural card (see the class Notes for the full list).
    /// FitsError
    ///   If `keyword` exceeds 8 characters or contains an invalid
    ///   character (a HIERARCH keyword is exempt from the length
    ///   limit).
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
    ///   Existing keyword.
    /// newname : str
    ///   Replacement keyword. Must be a valid FITS or HIERARCH
    ///   keyword.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the header is read-only, or if `oldname` or `newname`
    ///   names a structural card (see the class Notes for the full
    ///   list). Checked before `newname` validity and before
    ///   `oldname` is looked up.
    /// FitsError
    ///   If `newname` exceeds 8 characters or contains an invalid
    ///   character (a HIERARCH keyword is exempt from the length
    ///   limit). Checked before `oldname` is looked up, so this can
    ///   fire even when `oldname` does not exist.
    /// KeyError
    ///   If no card with `oldname` exists.
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
    /// For each ``(key, value)`` in ``other``, an existing keyword's
    /// value is overwritten in place and a new keyword is appended.
    /// A mapping value may be a bare scalar or a ``(value, comment)``
    /// tuple, as for :meth:`__setitem__`. Unlike :meth:`__setitem__`,
    /// a structural keyword (see the class Notes) is copied unchecked
    /// rather than rejected.
    ///
    /// Commentary cards (``COMMENT``, ``HISTORY``, blank-keyword) are
    /// **not** copied; use :meth:`add_commentary` if you want to
    /// transfer them explicitly.
    ///
    /// Parameters
    /// ----------
    /// other : Header or Mapping[str, Any]
    ///   The values to merge in. ``Header`` instances copy their
    ///   value cards; mappings are iterated in declaration order.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the header is read-only.
    /// TypeError
    ///   If ``other`` is neither a ``Header`` nor an object with an
    ///   ``.items()`` method, or if one of its values is not a
    ///   ``bool``, ``int``, ``float``, ``complex``, ``str`` or
    ///   ``None``.
    /// FitsError
    ///   If a keyword copied from `other` exceeds 8 characters or
    ///   contains an invalid character.
    ///
    /// Notes
    /// -----
    /// A mapping key is folded to upper case, as
    /// ``header[key] = value`` folds its key.
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
    ///   Keyword to look up (case-insensitive).
    /// default : object, optional
    ///   Value to return if ``key`` is absent. Defaults to ``None``.
    ///
    /// Returns
    /// -------
    /// object
    ///   The matching value, or ``default`` if absent.
    ///
    /// Notes
    /// -----
    /// Unlike ``header[key]``, this method never returns a
    /// :class:`HeaderCommentary`: ``"COMMENT"``, ``"HISTORY"`` and
    /// ``""`` have no value card to match, so ``get`` on one of
    /// these three keywords always returns `default`.
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
    ///   Keyword (case-insensitive match).
    ///
    /// Returns
    /// -------
    /// str or None
    ///   The comment text, or ``None`` if no such card exists or
    ///   the matching card has no inline comment.
    fn comment(&self, key: &str) -> Option<String> {
        self.lock()
            .entries()
            .iter()
            .find(|e| e.keyword.eq_ignore_ascii_case(key))
            .and_then(|e| e.comment.clone())
    }

    /// Plain ``dict`` view of the header.
    ///
    /// Inline comments are dropped. A commentary card (``COMMENT``,
    /// ``HISTORY``, blank-keyword) carries no value and is omitted
    /// entirely, not even as a ``None`` entry. A duplicated value
    /// keyword is deduplicated to its last-seen value. Convenience
    /// for ad-hoc work; round-trip fidelity requires :meth:`items`.
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
    /// 2880-byte block). The text round-trips through
    /// :meth:`fromstring`.
    ///
    /// Returns
    /// -------
    /// str
    ///   The serialized header text.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If a card cannot be serialized: a non-finite (``NaN`` or
    ///   infinite) real value, or a string or commentary card
    ///   holding a byte outside printable ASCII. These are accepted
    ///   without checking by :meth:`__setitem__`, :meth:`set`,
    ///   :meth:`insert` and :meth:`add_commentary`, and rejected only
    ///   here.
    fn tostring(&self) -> PyResult<String> {
        let bytes = self.lock().to_bytes().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialize header: {e}"))
        })?;
        String::from_utf8(bytes).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("non-ASCII header bytes: {e}"))
        })
    }

    /// Parse a header from a string of concatenated 80-character FITS
    /// cards -- the inverse of :meth:`tostring`.
    ///
    /// The text is read as raw card images with no separators. It need
    /// not be block-aligned or carry an ``END`` card: a partial final
    /// card is space-padded and an ``END`` is appended when absent, so
    /// both a full 2880-byte dump and a bare ``"OBJECT  = 'M31'"``
    /// fragment parse.
    ///
    /// Parameters
    /// ----------
    /// data : str
    ///   Header cards as ASCII text. Use :meth:`frombytes` for a raw
    ///   ``bytes`` buffer.
    /// lenient : bool, keyword-only, optional
    ///   Tolerate non-conforming values and structure (default
    ///   ``True``, matching :func:`fitsy.open`). Pass ``False`` to
    ///   require strict Standard conformance.
    ///
    /// Returns
    /// -------
    /// Header
    ///   A new, writable header.
    ///
    /// Raises
    /// ------
    /// FitsError
    ///   If `data` does not parse as FITS header cards -- for
    ///   example an unrecognized value, or (only when `lenient` is
    ///   ``False``) a non-ASCII byte in a value field.
    #[staticmethod]
    #[pyo3(signature = (data, *, lenient=true))]
    fn fromstring(data: &str, lenient: bool) -> PyResult<Self> {
        Self::from_card_bytes(data.as_bytes(), lenient)
    }

    /// Parse a header from raw FITS bytes -- the inverse of
    /// ``bytes(header)``. Behaves like :meth:`fromstring` but takes a
    /// ``bytes`` buffer (for example, header blocks read straight from
    /// a file), including its `Raises` conditions.
    #[staticmethod]
    #[pyo3(signature = (data, *, lenient=true))]
    fn frombytes(data: &[u8], lenient: bool) -> PyResult<Self> {
        Self::from_card_bytes(data, lenient)
    }

    /// Raw FITS bytes (``bytes(header)``). Same content as
    /// :meth:`tostring` but returned as ``bytes``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   Under the same conditions as :meth:`tostring`.
    fn __bytes__(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyBytes>> {
        let bytes = self.lock().to_bytes().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialize header: {e}"))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &bytes).unbind())
    }

    fn __repr__(&self) -> String {
        // Render every card as a fixed 80-character FITS string, one
        // per line, in declaration order. Padding cards (trailing
        // blanks added to round up to a 2880-byte block) and the
        // closing END card are omitted. Falls back to a one-line
        // summary if serialization fails (e.g. malformed values).
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
                // Keep the 80-char card padded with trailing spaces;
                // only a NUL fill byte (see Card::parse) is trimmed.
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

    // -- Time accessors -------------------------------------------------------

    /// Creation date of this HDU (``DATE``), always UTC, as an ISO-8601 string.
    ///
    /// Returns ``None`` if the keyword is absent.
    #[getter]
    fn date(&self) -> Option<String> {
        let h = self.lock();
        h.date().map(|dt| {
            if dt.hour == 0 && dt.minute == 0 && dt.second == 0 && dt.frac_second == 0.0 {
                format!("{:04}-{:02}-{:02}", dt.year, dt.month, dt.day)
            } else {
                format!(
                    "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                    dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
                )
            }
        })
    }

    /// Active time scale (``TIMESYS``), trimmed and upper-cased.
    ///
    /// Returns ``"UTC"`` when the keyword is absent, per the FITS standard
    /// default. The value is read back verbatim, so a header with an
    /// unrecognized or malformed ``TIMESYS`` still returns that text;
    /// this getter does not validate it against the WCS Paper IV
    /// Table 1 time scales that :attr:`mjd_obs_utc` and its siblings
    /// understand.
    #[getter]
    fn time_sys(&self) -> String {
        self.lock().time_sys()
    }

    /// Reference epoch as MJD. Reads ``MJDREFI``+``MJDREFF`` -> ``MJDREF`` ->
    /// ``JDREFI``+``JDREFF`` -> ``JDREF`` -> ``DATEREF``. Zero point for
    /// relative time values in the HDU.
    #[getter]
    fn mjd_ref(&self) -> Option<f64> {
        self.lock().mjd_ref()
    }

    /// Time unit for numeric time values (``TIMEUNIT``), lower-cased.
    ///
    /// Returns ``"s"`` when the keyword is absent (FITS standard default).
    #[getter]
    fn time_unit(&self) -> String {
        self.lock().time_unit()
    }

    /// Effective exposure time in seconds, excluding dead time.
    ///
    /// This reads ``XPOSURE``, scaled by ``TIMEUNIT`` or by a per-card
    /// ``[unit]`` annotation when one is present. It falls back to
    /// ``EXPTIME``, which predates the standard and is always in
    /// seconds, when ``XPOSURE`` is absent.
    ///
    /// Returns
    /// -------
    /// float or None
    ///   The exposure time in seconds. ``None`` when neither keyword
    ///   is present, and when ``XPOSURE`` is present but its unit is
    ///   not a recognized time unit.
    #[getter]
    fn time_exposure(&self) -> Option<f64> {
        self.lock().time_exposure()
    }

    /// Wall-clock elapsed time in seconds (``TELAPSE``), including dead time.
    /// ``None`` if absent or ``TIMEUNIT`` is unrecognized.
    #[getter]
    fn time_elapsed(&self) -> Option<f64> {
        self.lock().time_elapsed()
    }

    /// Start of the observation as UTC MJD.
    ///
    /// Tries, in order: ``MJD-BEG``; ``DATE-BEG``; ``TSTART`` added
    /// to the reference epoch and ``TIMEOFFS``; ``UTSTART`` combined
    /// with the date from ``DATE-OBS``. The first three are
    /// converted from ``TIMESYS`` to UTC; ``UTSTART`` is UTC
    /// already.
    #[getter]
    fn mjd_begin_utc(&self) -> Option<f64> {
        self.lock().mjd_begin_utc()
    }

    /// End of the observation as UTC MJD.
    ///
    /// Tries, in order: ``MJD-END``; ``DATE-END``; ``TSTOP`` added
    /// to the reference epoch and ``TIMEOFFS``; ``UTSTOP`` combined
    /// with the date from ``DATE-OBS``. The first three are
    /// converted from ``TIMESYS`` to UTC; ``UTSTOP`` is UTC already.
    #[getter]
    fn mjd_end_utc(&self) -> Option<f64> {
        self.lock().mjd_end_utc()
    }

    /// Average/mid time of the observation as UTC MJD.
    ///
    /// Reads ``MJD-AVG`` or ``DATE-AVG`` and converts from
    /// ``TIMESYS``; falls back to the midpoint of
    /// :attr:`mjd_begin_utc` and :attr:`mjd_end_utc` when neither is
    /// present.
    #[getter]
    fn mjd_avg_utc(&self) -> Option<f64> {
        self.lock().mjd_avg_utc()
    }

    /// Observation start converted to UTC MJD, regardless of ``TIMESYS``.
    ///
    /// Handles the full set of time scales defined in WCS Paper IV:
    /// ``UTC``, ``GMT``, ``TAI``, ``TT``/``TDT``/``ET``, ``GPS``,
    /// ``TCG``, ``TDB``, and ``TCB``.  Barycentric/geocentric scales
    /// are reduced to TT via the linear relations in Sec.3.1.2 before the
    /// leap-second table is applied.
    ///
    /// Returns
    /// -------
    /// float or None
    ///   UTC MJD of the observation start, or ``None`` if the observation
    ///   time is absent or the time scale cannot be reduced to UTC
    ///   (e.g. ``LOCAL``, ``UT1``).
    #[getter]
    fn mjd_obs_utc(&self) -> Option<f64> {
        self.lock().mjd_obs_utc()
    }

    // -- Observatory location -------------------------------------------------

    /// Observatory location as ITRS/ECEF Cartesian ``(x, y, z)`` in meters.
    /// Reads ``OBSGEO-X/Y/Z`` directly; falls back to geodetic keywords
    /// converted via WGS84.
    #[getter]
    fn obs_ecef(&self) -> Option<(f64, f64, f64)> {
        let g = self.lock().obs_ecef()?;
        Some((g.x, g.y, g.z))
    }

    /// Observatory geodetic coordinates ``(lat_deg, lon_deg, alt_m)`` on the
    /// WGS84 ellipsoid.
    ///
    /// Tries ``OBSGEO-B/L/H`` first, then non-standard variants
    /// (``SITELAT``, ``SITELONG``, ``SITEELEV``, etc.). ``None`` if neither
    /// latitude nor longitude is present.
    #[getter]
    fn obs_geodetic(&self) -> Option<(f64, f64, f64)> {
        let g = self.lock().obs_geodetic()?;
        Some((g.lat, g.lon, g.alt))
    }

    /// Orbit ephemeris file (``OBSORBIT``): URI, URL, or name.
    #[getter]
    fn obs_orbit(&self) -> Option<String> {
        self.lock().obs_orbit()
    }

    // -- Unit helpers ---------------------------------------------------------

    /// Unit string for a keyword's ``[unit]`` comment annotation.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///   Keyword to look up (case-insensitive).
    ///
    /// Returns
    /// -------
    /// str or None
    ///   The unit text, or ``None`` if the keyword is absent or its
    ///   comment carries no ``[unit]`` annotation.
    fn unit_for(&self, key: &str) -> Option<String> {
        self.lock().keyword_unit(key)
    }

    /// Value of `key` converted to the canonical unit for its
    /// physical dimension: meters for length, seconds for time,
    /// degrees for angle, and so on.
    ///
    /// Reads the source unit from the keyword's ``[unit]`` comment
    /// annotation and applies the conversion factor.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///   Keyword to look up (case-insensitive).
    ///
    /// Returns
    /// -------
    /// float or None
    ///   The converted value, or ``None`` if the keyword is absent,
    ///   non-numeric, carries no ``[unit]`` annotation, or the
    ///   annotation is not a recognized unit.
    fn value_in_si(&self, key: &str) -> Option<f64> {
        self.lock().real_in_canonical(key)
    }

    /// Check the header for deprecated, non-standard, or missing keywords.
    ///
    /// Parameters
    /// ----------
    /// fix : bool, optional
    ///   When ``True``, every suggested fix is applied to the returned
    ///   header copy. Defaults to ``False``.
    /// warn : bool, optional
    ///   When ``True`` (the default), each issue is emitted as a Python
    ///   :mod:`warnings` warning prefixed with ``[warning]`` or
    ///   ``[error]``. Set to ``False`` to suppress all output.
    ///
    /// Returns
    /// -------
    /// Header
    ///   A new independent snapshot of this header (fixed when
    ///   ``fix=True``, otherwise an unmodified clone).
    #[pyo3(signature = (fix = false, warn = true))]
    fn validate(&self, py: Python<'_>, fix: bool, warn: bool) -> PyResult<Py<Self>> {
        let (diags, fixed_hdr) = self.lock().validate(fix);
        if warn {
            let warnings = py.import("warnings")?;
            for d in diags {
                let level = match d.level {
                    Level::Warning => "warning",
                    Level::Error => "error",
                };
                let msg = format!("[{level}] {}: {}", d.keyword, d.message);
                warnings.call_method1("warn", (msg,))?;
            }
        }
        Py::new(
            py,
            Self {
                inner: Arc::new(Mutex::new(fixed_hdr)),
                read_only: false,
                dirty: None,
            },
        )
    }
}

/// Iterator over keyword strings, returned by ``iter(header)``.
///
/// Snapshots the keyword list at the time ``iter(header)`` was
/// called; a later edit to the header does not extend or shrink an
/// iterator already in progress.
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

    /// Next keyword, or ``None`` at the end of iteration.
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
/// ``header[key]`` for one of those three keywords.
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
    /// Number of cards: ``len(view)``.
    fn __len__(&self) -> usize {
        self.lines.len()
    }

    /// Text body of the i-th card: ``view[i]``.
    ///
    /// Parameters
    /// ----------
    /// idx : int
    ///   Index, accepts a negative value counting from the end.
    ///
    /// Returns
    /// -------
    /// str
    ///   The card's text body.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///   If `idx` is outside ``range(-len(view), len(view))``.
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

    /// Iterate over each card's text body, in declaration order.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        Ok(PyList::new(py, &slf.lines)?
            .call_method0("__iter__")?
            .unbind())
    }

    /// Every text body, joined with ``"\n"``.
    fn __str__(&self) -> String {
        self.lines.join("\n")
    }

    fn __repr__(&self) -> String {
        self.lines.join("\n")
    }
}

/// Convert a Python ``header=`` argument into a core [`Header`].
///
/// Accepts a [`PyHeader`] (cloned, preserving every card, including
/// commentary and structural keywords), a `dict`, or any object
/// exposing `.items()`. A mapping entry's keyword is folded to upper
/// case and its value is a bare scalar or a `(value, comment)` tuple,
/// as [`parse_setitem_value`] accepts. Unlike `header[key] = value`,
/// a structural keyword in a mapping entry is not rejected. Shared by
/// the `Header(...)` constructor and the writer builders.
///
/// # Errors
///
/// Returns a Python `TypeError` if `obj` is neither a [`PyHeader`]
/// nor an object with an `.items()` method, or if one of its values
/// does not convert through [`py_to_value`]. Returns `fitsy.FitsError`
/// if a keyword fails validation (too long, or an invalid character).
pub(crate) fn header_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Header> {
    // Another Header: clone every card (values, commentary, structural).
    if let Ok(h) = obj.extract::<PyRef<'_, PyHeader>>() {
        return Ok(h.lock().clone());
    }
    // Mapping: insert each entry as though by ``header[key] = value``.
    let mut header = Header::empty();
    let items = obj
        .call_method0("items")
        .map_err(|_| PyTypeError::new_err("header must be a Header or a mapping with .items()"))?;
    let iter = items.try_iter()?;
    for item in iter {
        let pair = item?;
        let key: String = pair.get_item(0)?.extract()?;
        let value = pair.get_item(1)?;
        let (val, comment) = parse_setitem_value(&value)?;
        header
            .set(&norm_key(&key), val, comment.as_deref())
            .map_err(super::err_to_py)?;
    }
    Ok(header)
}

/// True if `card` (an 80-byte slice) is an `END` card: the keyword
/// field is `END` followed by spaces. Case-insensitive in lenient mode,
/// matching the card scanner's folding of a lower-case `end` terminator.
/// Used by `from_card_bytes` to avoid appending a duplicate `END`.
fn is_end_card(card: &[u8], lenient: bool) -> bool {
    if card.len() < CARD_SIZE {
        return false;
    }
    let head = &card[..3];
    let is_end = if lenient {
        head.eq_ignore_ascii_case(b"END")
    } else {
        head == b"END"
    };
    // Bytes 3..8 (the rest of the keyword field) must be spaces, so an
    // 8-char keyword like `ENDIAN` is not mistaken for the terminator.
    is_end && card[3..8].iter().all(|&b| b == b' ')
}

/// Parse the right-hand side of `header[key] = ...`. Accepts either
/// a bare scalar, converted through [`py_to_value`] with no comment,
/// or a 2-element `(value, comment)` tuple where `comment` is a
/// Python `str` or `None`.
///
/// A tuple of any other length is not treated as `(value, comment)`;
/// it falls through to [`py_to_value`] on the whole object, which
/// then fails with the same `TypeError` as any other unsupported
/// type.
///
/// # Errors
///
/// Returns a Python `TypeError` if `v` (or a 2-tuple's first
/// element) does not convert through [`py_to_value`], or if a
/// 2-tuple's second element is not a `str` or `None`.
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

/// Convert a Python scalar to a FITS [`Value`].
///
/// Tries, in order: `bool` to [`Value::Logical`]; `int` to
/// [`Value::Integer`] (an integer outside the `i64` range falls
/// through to the next case rather than erroring here); `float` to
/// [`Value::Real`]; `complex` to [`Value::ComplexInteger`] when both
/// parts are whole numbers within the `i64` range, otherwise
/// [`Value::ComplexReal`]; `str` to [`Value::String`]; and `None` to
/// [`Value::Undefined`]. A numpy scalar (`numpy.int64`,
/// `numpy.float64`, `numpy.bool_`, ...) converts through the same
/// arms, because it supports the same `__index__` / `__float__`
/// protocols PyO3 extracts through.
///
/// # Errors
///
/// Returns a Python `TypeError` naming `v`'s type if `v` matches
/// none of the above -- for example a `list`, `dict`, or `tuple`.
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
