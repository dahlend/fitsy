//! `PyFitsFile` -- top-level reader/writer.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::{FitsFile, FitsUpdater};

use super::IntoPyResult;
use super::hdu::PyImageHdu;
use super::header::PyHeader;
use super::table::{PyAsciiTable, PyBinTable};
use super::wcs::PyWcs;

/// Extract EXTNAME from a `PyHeader`, returning an empty string if absent.
fn extname_from_header(h: &PyHeader) -> String {
    h.lock()
        .optional_string("EXTNAME")
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Returns true when the header declares at least one non-empty `CTYPEi`,
/// which is the minimal indicator that a WCS is present.  This is a fast
/// keyword scan -- no full WCS parse -- so repr stays cheap.
fn header_has_wcs(h: &PyHeader) -> bool {
    h.lock().cards().any(|c| {
        c.keyword().starts_with("CTYPE")
            && matches!(
                c.value(),
                Some(crate::header::Value::String(ref s)) if !s.trim().is_empty()
            )
    })
}

/// Parse a FITS file open mode into the read-only flag.
///
/// Recognizes four mode strings:
///
/// * `"readonly"` (default) -- header mutation and `writeto` raise.
/// * `"denywrite"` -- a stricter readonly mode. No other process may
///   open the file for writing under its intended semantics; fitsy
///   does not enforce that OS-level lock, and treats it exactly like
///   `"readonly"`.
/// * `"update"` -- read/write; edits are kept on `writeto`.
/// * `"append"` and `"ostream"` -- not implemented; use `fitsy.write`
///   for output-only work.
///
/// # Errors
///
/// Returns [`PyValueError`] if `mode` is none of the four strings
/// above.
fn parse_mode(mode: &str) -> PyResult<bool> {
    match mode {
        "readonly" | "denywrite" => Ok(true),
        "update" => Ok(false),
        "append" | "ostream" => Err(PyValueError::new_err(format!(
            "fitsy.open: mode {mode:?} is not supported; use `fitsy.write(path, hdus)` to \
             create a new file, or open with mode='readonly' and call `writeto(new_path)` to \
             save a modified copy"
        ))),
        other => Err(PyValueError::new_err(format!(
            "fitsy.open: mode must be 'readonly', 'denywrite', or 'update'; got {other:?}"
        ))),
    }
}

/// Open a FITS file by path.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///   Filesystem path to the FITS file.
/// mode : {'readonly', 'denywrite', 'update'}, optional
///   ``'readonly'`` (default) opens read-only. Every mutation --
///   a header edit, a pixel edit, :meth:`FitsFile.append`,
///   ``del file[i]``, and :meth:`FitsFile.writeto` back onto the
///   same path -- raises :class:`ValueError`. :meth:`FitsFile.writeto`
///   to a different path still works, and copies the file unchanged.
///
///   ``'denywrite'`` behaves exactly like ``'readonly'``. fitsy does
///   not take an OS-level write lock for this mode.
///
///   ``'update'`` opens read/write. Header edits and image-pixel
///   in-place edits (``hdu.data[...] = x``) are preserved on the next
///   :meth:`FitsFile.flush`, :meth:`FitsFile.close`, or a clean
///   ``__exit__``. Table column data is read-only in this release;
///   reconstruct the table with :func:`fitsy.bintable` to change
///   column values.
///
///   ``'append'`` and ``'ostream'`` are recognized but not
///   implemented; use :func:`fitsy.write` for output-only work.
/// lenient : bool, optional
///   Tolerate common non-conforming headers so real-world files load.
///   **Default True.** Pass ``lenient=False`` to require strict FITS
///   conformance.
///
///   Stray bytes in free-text *comments* are always sanitized to
///   spaces, even when ``lenient=False``.
///
///   Leniency also accepts non-conforming *values*:
///
///   * ``SIMPLE = F`` primary headers;
///   * non-ASCII bytes in string values, sanitized to spaces;
///   * lower-case or otherwise malformed keywords;
///   * values matching no standard type, kept verbatim as a string so
///     the rest of the file still loads;
///   * stray bytes after ``END``, a lower-case ``end``, and broken
///     ``CONTINUE`` chains.
///
///   A present ``END``, block alignment and the declared data size are
///   enforced in every mode.
///
/// Returns
/// -------
/// FitsFile
///   A read-only or read/write handle depending on ``mode``.
///
/// Raises
/// ------
/// ValueError
///   If ``mode`` is not one of the recognized values.
/// FitsError
///   On parse failures or I/O errors.
///
/// Examples
/// --------
/// >>> import fitsy
/// >>> with fitsy.open("image.fits") as f:
/// ...     img = f[0]
/// ...     print(img.axes)
#[pyfunction]
#[pyo3(signature = (path, mode="readonly", lenient=true))]
pub fn open(_py: Python<'_>, path: PathBuf, mode: &str, lenient: bool) -> PyResult<PyFitsFile> {
    let read_only = parse_mode(mode)?;
    let inner = FitsFile::open_with(&path, lenient).into_py_result()?;
    let n = inner.len();
    // Lazy: just record one Pending slot per HDU. Each slot is
    // materialized into a Python object only when first accessed.
    let slots: Vec<HduSlot> = (0..n).map(HduSlot::Pending).collect();
    let filename = path.file_name().map(|n| n.to_string_lossy().into_owned());
    let original_path = std::fs::canonicalize(&path).ok();
    // Open a writable file handle alongside the read-only one when
    // the user asked for `mode='update'`. Patch writes via
    // `hdu.section[a:b] = arr` go through this updater via
    // positional `pwrite` (O(patch)); the read-only `inner` keeps
    // serving header / `data` reads.
    let updater = if read_only {
        None
    } else {
        Some(Arc::new(Mutex::new(
            FitsUpdater::open_with(&path, lenient).into_py_result()?,
        )))
    };
    Ok(PyFitsFile {
        state: Mutex::new(FileState {
            file: Some(Arc::new(inner)),
            slots,
        }),
        read_only,
        filename,
        original_path,
        updater,
        dirty: Arc::new(DirtyFlags::default()),
        stamp_checksums: AtomicBool::new(false),
    })
}

/// One HDU position. Either Pending (still living in `FileState.file`
/// at the recorded original index) or Materialized (decoded into a
/// Python wrapper that owns its data; the original `FitsFile` is no
/// longer required for it).
#[derive(Debug)]
enum HduSlot {
    Pending(usize),
    Materialized(Py<PyAny>),
}

/// Mirror of [`HduSlot`], taken as a snapshot by both write paths.
///
/// Those paths are [`PyFitsFile::writeto`] and the in-place rewrite
/// behind [`PyFitsFile::flush`]. A snapshot lets each one classify and
/// re-frame the slot list without holding the state lock across a
/// Python callback.
#[derive(Debug)]
enum WritetoSlot {
    Pending(usize),
    Materialized(Py<PyAny>),
}

/// Create a sibling temp file next to `target`, opened
/// `O_CREAT|O_EXCL`.
///
/// The name is `<basename>.fitsy-tmp.<pid>.<nanos>`. The `caller`
/// argument names the Python method in the error message.
///
/// The unpredictable suffix avoids the race a fixed
/// `<path>.fitsy-tmp` name carries. Under a fixed name, an attacker or
/// a stale file from a killed process can create the path first. The
/// call then fails, or follows a symlink. `O_EXCL` turns a collision
/// into a retry rather than an overwrite, and a lost race draws a
/// fresh timestamp. Sixteen attempts cover nanosecond granularity.
///
/// # Errors
///
/// The last [`std::io::Error`], mapped through
/// [`err_to_py`](super::err_to_py), when every attempt fails. The
/// usual cause is a parent directory that is missing or not
/// writable.
fn create_sibling_temp(
    target: &std::path::Path,
    caller: &str,
) -> PyResult<(PathBuf, std::fs::File)> {
    use std::fs::OpenOptions;

    let parent = target.parent().unwrap_or_else(|| std::path::Path::new("."));
    let basename = target.file_name().map_or_else(
        || std::ffi::OsString::from("fitsy-out"),
        std::ffi::OsStr::to_os_string,
    );
    let mut last_err: Option<std::io::Error> = None;
    for _ in 0..16 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let pid = std::process::id();
        let mut name = basename.clone();
        name.push(format!(".fitsy-tmp.{pid}.{nanos:08x}"));
        let candidate = parent.join(&name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(f) => return Ok((candidate, f)),
            Err(e) => last_err = Some(e),
        }
    }
    Err(super::err_to_py(crate::error::FitsError::Io(
        last_err.unwrap_or_else(|| {
            std::io::Error::other(format!("{caller}: could not create temp file"))
        }),
    )))
}

/// Internal mutable state shared between threads. Holding `file`
/// alongside `slots` lets `materialize_at` pull data on demand
/// while still allowing other threads to mutate the slot list.
#[derive(Debug)]
struct FileState {
    /// The on-disk file (still required while any Pending slot
    /// exists, and -- in the lazy-data design -- while any
    /// materialized image HDU still has unrealized pixel data).
    /// Held as `Arc` so each [`PyImageHdu`] that needs lazy reads
    /// can keep its own clone.
    file: Option<Arc<FitsFile>>,
    slots: Vec<HduSlot>,
}

/// Owning, ordered, mutable list of HDUs.
///
/// Each slot is a typed object -- :class:`ImageHdu`, :class:`BinTable`,
/// :class:`AsciiTable`, or :class:`RandomGroups` -- owning its own
/// header and data, and it outlives the file handle.
///
/// In ``mode='readonly'`` (the default), every mutation raises
/// :class:`ValueError`: a header edit, ``hdu.data[...] = x``,
/// ``f.append(hdu)``, ``del f[i]``, and :meth:`writeto` back onto the
/// source path. :meth:`writeto` to a different path still works and
/// copies the file unchanged.
///
/// In ``mode='update'``, the same edits are held in memory and reach
/// the source file on the next :meth:`flush`, :meth:`close`, or a
/// clean ``__exit__``. A pixel patch through
/// ``f[i].section[a:b] = arr`` is written immediately instead, and
/// does not wait for :meth:`flush`.
///
/// Build a new, in-memory file with the ``FitsFile()`` constructor
/// (no path), then :meth:`append` each HDU and call :meth:`writeto`
/// to create a file from nothing.
///
/// Use the :func:`open` factory to load an existing file rather than
/// constructing this class directly.
///
/// Notes
/// -----
/// A :class:`RandomGroups` HDU writes back the header and the data
/// section it was read with. The bindings expose no way to edit
/// either, so a write reproduces the source HDU byte for byte.
///
/// Examples
/// --------
/// >>> with fitsy.open("image.fits", mode="update") as f:
/// ...     f[0].data[0, 0] = 42.0
/// ...     # changes flushed automatically on __exit__
/// >>> with fitsy.open("image.fits") as f:    # readonly
/// ...     f.writeto("copy.fits")             # unmodified copy
#[pyclass(name = "FitsFile", module = "fitsy")]
#[derive(Debug)]
pub struct PyFitsFile {
    state: Mutex<FileState>,
    pub(crate) read_only: bool,
    /// Display name (filename, not full path) for `__repr__`.
    pub(crate) filename: Option<String>,
    /// Canonicalized backing path, when opened from disk. Used by
    /// `writeto` to detect the "write to ourselves" case (which
    /// would invalidate the read handle and the writable updater).
    pub(crate) original_path: Option<PathBuf>,
    /// Writable file handle (for `pwrite`), present only when
    /// opened with `mode='update'`. Image HDUs receive a clone of
    /// this `Arc` during materialization so that `hdu.section[...]
    /// = arr` performs O(patch) in-place writes.
    pub(crate) updater: Option<Arc<Mutex<FitsUpdater>>>,
    /// Set whenever a non-pixel-patch mutation happens (header
    /// edit, `set_data`, structural mutation). On `flush()` /
    /// `__exit__` (clean exit) in `mode='update'`, a true value
    /// triggers a rewrite-via-temp+rename of the original file.
    /// Pixel patches via `hdu.section[a:b] = arr` write through
    /// `pwrite` directly and do **not** flip this bit.
    pub(crate) dirty: Arc<DirtyFlags>,
    /// When true, the next `writeto` / `flush` will compute and
    /// stamp `CHECKSUM` / `DATASUM` cards on every emitted HDU
    /// via [`crate::FitsWriter::with_checksums`]. Toggled on by
    /// [`add_checksums`](Self::add_checksums); stays on for the
    /// lifetime of the file.
    pub(crate) stamp_checksums: AtomicBool,
}

/// Rewrite bookkeeping shared between a `FitsFile` and the HDU
/// wrappers it hands out.
#[derive(Debug, Default)]
pub(crate) struct DirtyFlags {
    /// A mutation that definitely needs a rewrite.
    pub(crate) definite: AtomicBool,
    /// A writeable pixel array was handed to Python. numpy edits are
    /// invisible from here, so we cannot tell whether it was modified;
    /// `flush` compares the cache against the file and only rewrites if
    /// it actually differs. That keeps a read-only pass over `.data` in
    /// `mode='update'` from costing a full rewrite.
    pub(crate) handed_out: AtomicBool,
}

impl PyFitsFile {
    fn lock_state(&self) -> std::sync::MutexGuard<'_, FileState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Materialize the slot at `slot_idx` into a live Python HDU
    /// wrapper, replacing the `Pending` placeholder. Returns a new
    /// owned reference (clone of the cached one).
    ///
    /// # Errors
    ///
    /// Returns [`PyIndexError`] if `slot_idx` is out of range.
    /// Returns [`PyValueError`] if the backing file was already
    /// dropped (by [`Self::close`] or a prior rewrite) while this
    /// slot was still `Pending`. Returns the Python exception mapped
    /// from `fitsy.FitsError` if the HDU fails to parse, and
    /// [`PyTypeError`] if it is a kind the bindings do not wrap.
    fn materialize_at(&self, py: Python<'_>, slot_idx: usize) -> PyResult<Py<PyAny>> {
        let mut st = self.lock_state();
        let n = st.slots.len();
        if slot_idx >= n {
            return Err(PyIndexError::new_err(format!(
                "HDU index {slot_idx} out of range"
            )));
        }
        if let HduSlot::Materialized(p) = &st.slots[slot_idx] {
            return Ok(p.clone_ref(py));
        }
        let HduSlot::Pending(file_idx) = st.slots[slot_idx] else {
            unreachable!("checked above");
        };
        let file = st.file.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "FitsFile: backing file dropped while a slot is still pending \
                 (internal invariant violated)",
            )
        })?;
        // Wire mutations through to the file-level dirty flag so
        // `flush()` / `__exit__` know to rewrite. Pixel patches via
        // `section[a:b] = arr` go through `update_binding` instead
        // and are persisted by `pwrite` without flipping the bit.
        let dirty_flag = self.updater.as_ref().map(|_| self.dirty.clone());
        // Header-only fast path for plain image HDUs: avoids
        // populating `FitsFile`'s per-HDU data cache, which would
        // otherwise hold the raw image bytes resident for the
        // lifetime of the file. Lazy `data` / `section` reads go
        // through `read_data_owned` / `read_image_subarray_be`
        // which never touch that cache.
        if let Some(wrapped) = self.try_image_fast_path(py, file_idx, file, dirty_flag.clone())? {
            st.slots[slot_idx] = HduSlot::Materialized(wrapped.clone_ref(py));
            return Ok(wrapped);
        }
        let h = file.hdu(file_idx).into_py_result()?;
        let mut header = PyHeader::from_header_with(h.header(), self.read_only);
        header.dirty.clone_from(&dirty_flag);
        let wrapped = wrap_hdu(
            py,
            file_idx,
            h,
            header,
            self.read_only,
            self.updater.as_ref(),
            dirty_flag,
            file.clone(),
        )?;
        st.slots[slot_idx] = HduSlot::Materialized(wrapped.clone_ref(py));
        Ok(wrapped)
    }

    /// Try the header-only fast path for a plain image HDU.
    ///
    /// `Some` when the HDU at `file_idx` is a plain image -- not
    /// random groups, not tile-compressed -- and `None` when the
    /// caller should fall back to `file.hdu(i)`.
    ///
    /// This path skips the per-HDU data cache, reading pixels only on
    /// demand, so opening a file and reading header properties keeps
    /// no pixel bytes resident.
    ///
    /// # Errors
    ///
    /// Returns the Python exception mapped from `fitsy.FitsError` if
    /// the header cannot be parsed, or if `NAXIS`/`NAXISn`/`BITPIX`
    /// is missing or invalid.
    fn try_image_fast_path(
        &self,
        py: Python<'_>,
        file_idx: usize,
        file: &Arc<FitsFile>,
        dirty_flag: Option<Arc<DirtyFlags>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        use crate::Value;
        use crate::data::Bitpix;
        let header = file.parsed_header(file_idx).into_py_result()?;
        // Detect plain-image kind without reading data.
        let is_image = if file_idx == 0 {
            // Primary: an image unless it is random-groups (Sec.6).
            // This is the predicate the core reader dispatches on.
            !header.is_random_groups()
        } else {
            matches!(
                header.first("XTENSION"),
                Some(Value::String(s)) if s == "IMAGE"
            )
        };
        if !is_image {
            return Ok(None);
        }
        // ZIMAGE-tagged BINTABLEs are tile-compressed images, but
        // they have XTENSION='BINTABLE' so the check above already
        // rejects them. Plain images cannot have ZIMAGE.
        let axes = header.axes().into_py_result()?;
        let bitpix_i = header.bitpix().into_py_result()?;
        let bitpix = Bitpix::from_i64(bitpix_i).into_py_result()?;
        let mut py_header = PyHeader::from_header_with(&header, self.read_only);
        py_header.dirty.clone_from(&dirty_flag);
        let mut py_img = PyImageHdu {
            header: py_header,
            bitpix,
            axes: axes.clone(),
            read_only: self.read_only,
            data: Arc::new(Mutex::new(None)),
            read_binding: Some(crate::python::hdu::ReadBinding {
                file: file.clone(),
                hdu_idx: file_idx,
                axes,
            }),
            update_binding: None,
            dirty: dirty_flag,
            wcs_file: Some(file.clone()),
        };
        if let Some(u) = self.updater.as_ref() {
            let generation = u.lock().map_or(u64::MAX, |g| g.generation());
            py_img.update_binding = Some(super::hdu::UpdateBinding {
                updater: u.clone(),
                hdu_idx: file_idx,
                generation,
            });
        }
        Ok(Some(Py::new(py, py_img)?.into_any()))
    }

    /// Force every slot to be materialized; used by `__iter__`
    /// and `__repr__`. (`writeto` / `flush` deliberately do NOT
    /// call this -- they stream untouched Pending slots through
    /// `hdu_raw_padded` / `write_raw_padded` to avoid loading
    /// multi-GB images that the user never edited.)
    fn materialize_all(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let n = self.lock_state().slots.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(self.materialize_at(py, i)?);
        }
        Ok(out)
    }

    /// Did any HDU that handed a writeable array to Python actually
    /// come out different from the file?
    ///
    /// numpy edits are invisible to us, so the only exact answer is to
    /// re-read and compare. That costs one sequential read instead of a
    /// full rewrite, and lets a read-only pass over `.data` in update
    /// mode finish without touching the file. Anything we cannot verify
    /// counts as changed.
    fn handed_out_data_changed(&self, py: Python<'_>) -> bool {
        let snapshot: Vec<Py<PyAny>> = {
            let st = self.lock_state();
            st.slots
                .iter()
                .filter_map(|s| match s {
                    HduSlot::Materialized(p) => Some(p.clone_ref(py)),
                    HduSlot::Pending(_) => None,
                })
                .collect()
        };
        for obj in snapshot {
            let bound = obj.bind(py);
            let Ok(img) = bound.extract::<PyRef<'_, PyImageHdu>>() else {
                continue; // only image HDUs hand out a pixel array
            };
            if !img.data_matches_source(py) {
                return true;
            }
        }
        false
    }

    /// Rewrite the backing file to absorb the in-memory edits that
    /// could not be satisfied by an in-place pixel-patch (header
    /// edits, structural mutations, `set_data`, fancy slice writes).
    ///
    /// Streams raw bytes for slots the user never touched
    /// (`HduSlot::Pending`); only re-encodes materialized slots.
    /// On success, drops the original `FitsFile` and `FitsUpdater`
    /// and re-opens them against the freshly written file so that
    /// further mutations and pixel-patches keep working.
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] if `self.original_path` is `None`
    /// (an in-memory file has nothing to rewrite), if the resulting
    /// HDU list would be empty, or if a mutex was poisoned by an
    /// earlier panic. Returns [`PyTypeError`] if a materialized HDU
    /// slot cannot be encoded (see [`encode_hdu`]). Returns the
    /// Python exception mapped from `fitsy.FitsError` on an I/O
    /// failure or an encoding error from the core crate.
    /// Snapshot the slot list into owned [`WritetoSlot`] values.
    ///
    /// This holds the state lock for the copy alone. The caller
    /// encodes after the lock is released. Encoding a materialized
    /// slot calls back into Python. Holding the lock across that call
    /// can deadlock against another thread that touches this file.
    fn snapshot_slots(&self, py: Python<'_>) -> Vec<WritetoSlot> {
        let st = self.lock_state();
        st.slots
            .iter()
            .map(|s| match s {
                HduSlot::Pending(i) => WritetoSlot::Pending(*i),
                HduSlot::Materialized(p) => WritetoSlot::Materialized(p.clone_ref(py)),
            })
            .collect()
    }

    fn persist_full_rewrite(&self, py: Python<'_>) -> PyResult<()> {
        use std::io::{BufWriter, Write};

        let original_path = self.original_path.clone().ok_or_else(|| {
            PyValueError::new_err(
                "FitsFile.flush: cannot rewrite an in-memory file; use writeto(path)",
            )
        })?;

        // Make sure any in-flight `pwrite` patches are durable
        // before we start reading the source bytes.
        if let Some(updater) = self.updater.as_ref() {
            let guard = updater
                .lock()
                .map_err(|_| PyValueError::new_err("FitsFile.flush: updater mutex poisoned"))?;
            guard.flush().into_py_result()?;
        }

        // Snapshot slot states under the lock; release the lock
        // before doing the actual encoding (which may need to call
        // back into Python).
        let snapshot = self.snapshot_slots(py);
        if snapshot.is_empty() {
            return Err(PyValueError::new_err(
                "FitsFile.flush: refusing to rewrite a file with zero HDUs",
            ));
        }

        let (tmp_path, tmp_file) = create_sibling_temp(&original_path, "FitsFile.flush")?;

        let write_result: PyResult<()> = (|| {
            let mut bw = BufWriter::new(tmp_file);
            // Emit each slot. For Pending slots we copy the raw
            // header+padded-data bytes from the source file; for
            // Materialized slots we re-encode through `FitsWriter`.
            //
            // Pre-pass: any `Pending` slot whose source-file role
            // doesn't match its destination role (a Pending source
            // primary that's now an extension after `insert(0, ...)`,
            // or a Pending source extension that's now the primary
            // after `del f[0]`) must be materialized so we can
            // re-encode it with the correct SIMPLE / XTENSION
            // framing. Streaming the raw bytes would otherwise
            // produce an invalid FITS file (two primaries, or a
            // primary that starts with `XTENSION`).
            let mut snapshot = snapshot;
            let stamping = self.stamp_checksums.load(Ordering::Relaxed);
            for (dst_idx, slot) in snapshot.iter_mut().enumerate() {
                if let WritetoSlot::Pending(file_idx) = slot {
                    let needs_reframe = (*file_idx == 0) ^ (dst_idx == 0);
                    if needs_reframe || stamping {
                        let materialized = self.materialize_at(py, dst_idx)?;
                        *slot = WritetoSlot::Materialized(materialized);
                    }
                }
            }
            let mut writer = crate::FitsWriter::new(&mut bw);
            if self.stamp_checksums.load(Ordering::Relaxed) {
                writer = writer.with_checksums();
            }
            let mut emitted_primary = false;
            // Determine if we need to synthesize an empty primary:
            // only when the first emitted HDU isn't an image-like
            // (BinTable / AsciiTable can't be a primary).
            let needs_synth_primary = matches!(
                snapshot.first(),
                Some(WritetoSlot::Materialized(p)) if !is_image_like(py, p)
            );
            if needs_synth_primary {
                let (h, d) = empty_primary_header_and_bytes();
                writer.write_hdu(&h, &d).into_py_result()?;
                emitted_primary = true;
            }
            for slot in &snapshot {
                match slot {
                    WritetoSlot::Pending(file_idx) => {
                        let st = self.lock_state();
                        let file = st.file.as_ref().ok_or_else(|| {
                            PyValueError::new_err(
                                "FitsFile.flush: backing file dropped before persist",
                            )
                        })?;
                        let raw = file
                            .hdu_raw_padded(*file_idx)
                            .into_py_result()?
                            .ok_or_else(|| {
                                PyValueError::new_err(format!(
                                    "FitsFile.flush: source HDU {file_idx} out of range",
                                ))
                            })?;
                        // Drop straight into the writer's underlying
                        // sink, bypassing re-encoding entirely.
                        writer
                            .write_raw_padded(&raw)
                            .map_err(|e| super::err_to_py(crate::error::FitsError::Io(e)))?;
                        emitted_primary = true;
                    }
                    WritetoSlot::Materialized(p) => {
                        let is_primary = !emitted_primary;
                        let (header, data) = encode_hdu(py, p, is_primary)?;
                        writer.write_hdu(&header, &data).into_py_result()?;
                        emitted_primary = true;
                    }
                }
            }
            writer
                .finish()
                .map_err(|e| super::err_to_py(crate::error::FitsError::Io(e)))?;
            bw.flush()
                .map_err(|e| super::err_to_py(crate::error::FitsError::Io(e)))?;
            // fsync the data + directory entry before rename so a
            // crash mid-rename leaves either the old or the fully
            // written new file -- never a truncated mix.
            bw.get_ref()
                .sync_all()
                .map_err(|e| super::err_to_py(crate::error::FitsError::Io(e)))?;
            Ok(())
        })();
        if let Err(e) = write_result {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }

        // Drop the old read-only file + writable updater before
        // the atomic rename so neither holds stale handles.
        {
            let mut st = self.lock_state();
            st.file = None;
        }
        // The updater is shared via Arc; replace the inner Mutex
        // contents with a fresh one after rename.
        std::fs::rename(&tmp_path, &original_path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp_path);
            super::err_to_py(crate::error::FitsError::Io(e))
        })?;

        // Re-open the source so future mutations and pixel patches
        // keep working. Slots that were `Pending` before are still
        // `Pending` against the same indices in the new file (which
        // is a byte-for-byte rewrite of those HDUs anyway).
        let new_inner = FitsFile::open(&original_path).into_py_result()?;
        {
            let mut st = self.lock_state();
            st.file = Some(Arc::new(new_inner));
        }
        if let Some(updater) = self.updater.as_ref() {
            let new_updater = FitsUpdater::open(&original_path).into_py_result()?;
            let mut guard = updater
                .lock()
                .map_err(|_| PyValueError::new_err("FitsFile.flush: updater mutex poisoned"))?;
            // `replace_with` bumps the generation tag so any cached
            // `(arc, hdu_idx)` UpdateBindings held by Python wrappers
            // become stale and refuse the fast in-place pwrite path.
            // The next write through them flips the dirty bit and
            // takes the safe rewrite path instead.
            guard.replace_with(new_updater);
        }
        Ok(())
    }
}

#[pymethods]
impl PyFitsFile {
    /// Construct an empty in-memory file (zero HDUs). Use
    /// :func:`fitsy.open` to load from disk.
    #[new]
    fn py_new() -> Self {
        Self {
            state: Mutex::new(FileState {
                file: None,
                slots: Vec::new(),
            }),
            read_only: false,
            filename: None,
            original_path: None,
            updater: None,
            dirty: Arc::new(DirtyFlags::default()),
            stamp_checksums: AtomicBool::new(false),
        }
    }

    /// Number of HDUs (``len(file)``).
    fn __len__(&self) -> usize {
        self.lock_state().slots.len()
    }

    /// True when the file was opened read-only.
    #[getter]
    fn read_only(&self) -> bool {
        self.read_only
    }

    /// Return the ``i``-th HDU (``file[i]``), the first HDU named
    /// ``EXTNAME`` (``file["NAME"]``), or the HDU matching both
    /// ``EXTNAME`` and ``EXTVER`` (``file["NAME", ver]``).
    ///
    /// Parameters
    /// ----------
    /// key : int or str or tuple[str, int]
    ///   An HDU index, an ``EXTNAME`` string, or an
    ///   ``(EXTNAME, EXTVER)`` tuple. A negative integer index counts
    ///   from the end.
    ///
    /// Returns
    /// -------
    /// ImageHdu or BinTable or AsciiTable or RandomGroups
    ///   The matching HDU.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///   If `key` is an ``int`` outside the valid range.
    /// KeyError
    ///   If `key` is a ``str`` or an ``(EXTNAME, EXTVER)`` tuple and
    ///   no HDU matches.
    /// TypeError
    ///   If `key` is a tuple whose length is not 2 or whose elements
    ///   do not convert to ``(str, int)``, or if `key` is none of
    ///   ``int``, ``str``, or a 2-element tuple.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // (EXTNAME, EXTVER) tuple lookup.
        if let Ok(tup) = key.cast::<pyo3::types::PyTuple>()
            && tup.len() == 2
        {
            let name: String = tup.get_item(0)?.extract().map_err(|_| {
                PyTypeError::new_err("HDU tuple key must be (EXTNAME: str, EXTVER: int)")
            })?;
            let ver: i64 = tup.get_item(1)?.extract().map_err(|_| {
                PyTypeError::new_err("HDU tuple key must be (EXTNAME: str, EXTVER: int)")
            })?;
            return self.hdu_by_name(py, &name, Some(ver));
        }
        if let Ok(name) = key.extract::<String>() {
            return self.hdu_by_name(py, &name, None);
        }
        let i: isize = key.extract().map_err(|_| {
            PyTypeError::new_err(
                "HDU index must be an int, an EXTNAME string, or a (name, ver) tuple",
            )
        })?;
        let n = self.lock_state().slots.len() as isize;
        let idx = if i < 0 { i + n } else { i };
        if idx < 0 || idx >= n {
            return Err(PyIndexError::new_err(format!("HDU index {i} out of range")));
        }
        self.materialize_at(py, idx as usize)
    }

    /// Replace ``file[i]`` (``file[i] = value``).
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///   HDU index to replace. Accepts a negative index, counting
    ///   from the end.
    /// value : ImageHdu or BinTable or AsciiTable
    ///   The new HDU. This also accepts a builder, meaning an
    ///   :class:`fitsy.ImageBuilder`, a :class:`fitsy.BinTableBuilder`
    ///   or an :class:`fitsy.AsciiTableBuilder`, as returned by
    ///   :func:`fitsy.image`, :func:`fitsy.bintable` or
    ///   :func:`fitsy.ascii_table`. A builder is promoted to a live,
    ///   independently editable HDU instance.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the file was opened read-only.
    /// TypeError
    ///   If `value` is not an HDU instance or a builder.
    /// IndexError
    ///   If `i` is outside the valid range.
    ///
    /// Notes
    /// -----
    /// Marks the file dirty, and invalidates every cached in-place
    /// pixel-patch binding, before `value` or `i` is checked. Even a
    /// call that raises still forces the next write to be a full
    /// rewrite.
    fn __setitem__(&self, py: Python<'_>, i: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_writable()?;
        self.dirty.definite.store(true, Ordering::Release);
        self.invalidate_bindings();
        let hdu = coerce_to_hdu(py, &value)?;
        let mut st = self.lock_state();
        let n = st.slots.len() as isize;
        let idx = if i < 0 { i + n } else { i };
        if idx < 0 || idx >= n {
            return Err(PyIndexError::new_err(format!("HDU index {i} out of range")));
        }
        st.slots[idx as usize] = HduSlot::Materialized(hdu);
        Ok(())
    }

    /// Remove ``file[i]`` (``del file[i]``).
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///   HDU index to remove. Accepts a negative index, counting
    ///   from the end.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the file was opened read-only.
    /// IndexError
    ///   If `i` is outside the valid range.
    ///
    /// Notes
    /// -----
    /// Marks the file dirty, and invalidates every cached in-place
    /// pixel-patch binding, before `i` is checked. A call that
    /// raises `IndexError` still forces the next write to be a full
    /// rewrite.
    fn __delitem__(&self, i: isize) -> PyResult<()> {
        self.ensure_writable()?;
        self.dirty.definite.store(true, Ordering::Release);
        self.invalidate_bindings();
        let mut st = self.lock_state();
        let n = st.slots.len() as isize;
        let idx = if i < 0 { i + n } else { i };
        if idx < 0 || idx >= n {
            return Err(PyIndexError::new_err(format!("HDU index {i} out of range")));
        }
        st.slots.remove(idx as usize);
        Ok(())
    }

    /// Iterate over HDUs in declaration order.
    ///
    /// Materializes every pending slot up front, so the iterator's
    /// snapshot is stable against a concurrent edit.
    ///
    /// Returns
    /// -------
    /// Iterator[ImageHdu | BinTable | AsciiTable | RandomGroups]
    ///
    /// Raises
    /// ------
    /// FitsError
    ///   If an HDU fails to parse from the source file.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<HduIter>> {
        let snapshot = slf.materialize_all(slf.py())?;
        Py::new(
            slf.py(),
            HduIter {
                items: snapshot,
                pos: 0,
            },
        )
    }

    /// Append an HDU at the end.
    ///
    /// Parameters
    /// ----------
    /// value : ImageHdu or BinTable or AsciiTable
    ///   The new HDU. This also accepts a builder, meaning an
    ///   :class:`fitsy.ImageBuilder`, a :class:`fitsy.BinTableBuilder`
    ///   or an :class:`fitsy.AsciiTableBuilder`, as returned by
    ///   :func:`fitsy.image`, :func:`fitsy.bintable` or
    ///   :func:`fitsy.ascii_table`. A builder is promoted to a live,
    ///   independently editable HDU instance.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the file was opened read-only.
    /// TypeError
    ///   If `value` is not an HDU instance or a builder.
    ///
    /// Notes
    /// -----
    /// Marks the file dirty, and invalidates every cached in-place
    /// pixel-patch binding, before `value` is checked. A call that
    /// raises `TypeError` still forces the next write to be a full
    /// rewrite.
    fn append(&self, py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_writable()?;
        self.dirty.definite.store(true, Ordering::Release);
        self.invalidate_bindings();
        let hdu = coerce_to_hdu(py, &value)?;
        self.lock_state().slots.push(HduSlot::Materialized(hdu));
        Ok(())
    }

    /// Insert an HDU at position ``i``.
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///   Target position. A negative index counts from the end.
    ///   Clamped into ``[0, len(file)]``, so an out-of-range value
    ///   inserts at the nearer end instead of raising.
    /// value : ImageHdu or BinTable or AsciiTable
    ///   The new HDU. This also accepts a builder, meaning an
    ///   :class:`fitsy.ImageBuilder`, a :class:`fitsy.BinTableBuilder`
    ///   or an :class:`fitsy.AsciiTableBuilder`, as returned by
    ///   :func:`fitsy.image`, :func:`fitsy.bintable` or
    ///   :func:`fitsy.ascii_table`. A builder is promoted to a live,
    ///   independently editable HDU instance.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the file was opened read-only.
    /// TypeError
    ///   If `value` is not an HDU instance or a builder.
    ///
    /// Notes
    /// -----
    /// Marks the file dirty, and invalidates every cached in-place
    /// pixel-patch binding, before `value` is checked. A call that
    /// raises `TypeError` still forces the next write to be a full
    /// rewrite.
    fn insert(&self, py: Python<'_>, i: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_writable()?;
        self.dirty.definite.store(true, Ordering::Release);
        self.invalidate_bindings();
        let hdu = coerce_to_hdu(py, &value)?;
        let mut st = self.lock_state();
        let n = st.slots.len() as isize;
        let idx = if i < 0 { (i + n).max(0) } else { i.min(n) };
        st.slots.insert(idx as usize, HduSlot::Materialized(hdu));
        Ok(())
    }

    /// Return the ``i``-th HDU. Equivalent to ``file[i]`` for
    /// non-negative integer ``i``.
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///   HDU index. Unlike ``file[i]``, does not accept a negative
    ///   index.
    ///
    /// Returns
    /// -------
    /// ImageHdu or BinTable or AsciiTable or RandomGroups
    ///   The matching HDU.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///   If `i` is at least ``len(file)``.
    /// OverflowError
    ///   If `i` is negative.
    fn hdu(&self, py: Python<'_>, i: usize) -> PyResult<Py<PyAny>> {
        if i >= self.lock_state().slots.len() {
            return Err(PyIndexError::new_err(format!("HDU index {i} out of range")));
        }
        self.materialize_at(py, i)
    }

    /// Return the first HDU with matching ``EXTNAME``.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///   Value of the ``EXTNAME`` keyword to match.
    /// ver : int, optional
    ///   Value of the ``EXTVER`` keyword to also require. Default
    ///   ``None``, which matches on `name` alone, regardless of
    ///   ``EXTVER``. When `ver` is given, an HDU with no ``EXTVER``
    ///   card is treated as ``EXTVER=1``.
    ///
    /// Returns
    /// -------
    /// ImageHdu or BinTable or AsciiTable or RandomGroups
    ///   The matching HDU.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///   If no HDU matches `name` (and `ver`, when given).
    ///
    /// Notes
    /// -----
    /// Materializes each HDU, in order, until a match is found or
    /// the list is exhausted.
    #[pyo3(signature = (name, ver=None))]
    fn hdu_by_name(&self, py: Python<'_>, name: &str, ver: Option<i64>) -> PyResult<Py<PyAny>> {
        use pyo3::exceptions::PyKeyError;
        let n = self.lock_state().slots.len();
        for i in 0..n {
            let h = self.materialize_at(py, i)?;
            let bound = h.bind(py);
            let header = bound.getattr("header")?;
            let extname: Option<String> = header
                .call_method1("get", ("EXTNAME",))
                .ok()
                .and_then(|v| v.extract().ok());
            if extname.as_deref() == Some(name) {
                if let Some(want) = ver {
                    let got: i64 = header
                        .call_method1("get", ("EXTVER", 1))
                        .ok()
                        .and_then(|v| v.extract().ok())
                        .unwrap_or(1);
                    if got != want {
                        continue;
                    }
                }
                return Ok(h);
            }
        }
        Err(PyKeyError::new_err(format!(
            "no HDU with EXTNAME={name:?}{}",
            ver.map(|v| format!(", EXTVER={v}")).unwrap_or_default()
        )))
    }

    /// Resolve the WCS for the given HDU index.
    ///
    /// Parameters
    /// ----------
    /// i : int, optional
    ///   HDU index. Default 0 (primary HDU). Does not accept a
    ///   negative index.
    /// alt : str, optional
    ///   Single ASCII character. ``' '`` (default) selects the
    ///   primary WCS description; ``'A'`` through ``'Z'`` select
    ///   alternate descriptions.
    ///
    /// Returns
    /// -------
    /// Wcs or None
    ///   ``None`` if HDU `i`'s header carries no WCS for `alt`.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///   If `i` is at least ``len(file)``.
    /// OverflowError
    ///   If `i` is negative.
    /// ValueError
    ///   If `alt` is not exactly one character.
    /// FitsError
    ///   If `alt` is not ``' '`` or one of ``'A'``-``'Z'``, if the
    ///   header carries a malformed WCS, or if a ``-TAB`` axis
    ///   cannot be resolved (see Notes).
    ///
    /// Notes
    /// -----
    /// A ``-TAB`` axis (Paper III Sec.6) stores its coordinate
    /// array in a sibling BINTABLE. The ``PSi_0`` / ``PVi_1`` cards
    /// name that table. This method loads the table from the file
    /// this handle was opened from. A handle built in memory (the
    /// ``FitsFile()`` constructor) has no file to search. A
    /// ``-TAB`` axis then raises here, not later at transform time.
    /// Use ``fitsy.Wcs(f[i].header)`` to inspect such a header
    /// without the lookup table.
    #[pyo3(signature = (i=0, alt=' '))]
    fn wcs(&self, py: Python<'_>, i: usize, alt: char) -> PyResult<Option<PyWcs>> {
        let hdu = self.hdu(py, i)?;
        let bound = hdu.bind(py);
        let header: PyHeader = bound.getattr("header")?.extract()?;
        let wcs = crate::wcs::Wcs::from_header(&header.lock(), alt).into_py_result()?;
        let Some(mut wcs) = wcs else { return Ok(None) };
        if !wcs.tab_specs.is_empty() {
            // `hdu()` above took and released the state lock; safe to
            // re-take it here.
            let Some(file) = self.lock_state().file.clone() else {
                return Err(super::err_to_py(crate::error::FitsError::Wcs(
                    "WCS has a -TAB axis, but this FitsFile carries no file \
                     to load the lookup table from; open the file with \
                     fitsy.open, or use fitsy.Wcs(f[i].header) for \
                     header-only inspection"
                        .into(),
                )));
            };
            wcs.resolve_tab(&file).into_py_result()?;
        }
        Ok(Some(PyWcs::from(wcs)))
    }

    /// Write the file (with all in-memory edits) to ``path``.
    ///
    /// Each HDU is re-emitted from its current Python state:
    ///
    /// - :class:`ImageHdu` -- pixel data is encoded from the live
    ///   numpy array (so ``hdu.data[...] = x`` round-trips);
    ///   ``BITPIX`` and ``NAXIS*`` are recomputed from the array.
    /// - :class:`BinTable`, :class:`AsciiTable` -- data bytes are
    ///   re-emitted as captured at load time (column edits do
    ///   *not* round-trip in this release).
    /// - :class:`RandomGroups` -- header and data section are
    ///   re-emitted as captured at load time.
    ///
    /// An HDU slot that is still ``Pending`` (never accessed)
    /// streams through unchanged, whatever its kind.
    ///
    /// If the first HDU is not an image, an empty primary image HDU
    /// (``NAXIS = 0``) is automatically prepended so the output is a
    /// valid FITS file.
    ///
    /// The on-disk source file (if any) is never modified, *except*
    /// when ``path`` resolves to the same file the handle was opened
    /// from -- a self-write requires update mode and triggers an
    /// in-place rewrite (alias for :meth:`flush`).
    ///
    /// Parameters
    /// ----------
    /// path : str or os.PathLike
    ///   Destination path.
    /// overwrite : bool, optional
    ///   If False (default), raise :class:`FileExistsError` when
    ///   ``path`` already exists. Set to True to replace it.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the file contains zero HDUs, or if ``path`` resolves
    ///   to the source file and the handle is read-only.
    /// FileExistsError
    ///   If ``path`` exists and ``overwrite`` is False.
    /// TypeError
    ///   If an HDU slot holds an object that is none of
    ///   :class:`ImageHdu`, :class:`BinTable`, :class:`AsciiTable`
    ///   or :class:`RandomGroups`.
    /// FitsError
    ///   On I/O failure.
    #[pyo3(signature = (path, overwrite=false))]
    fn writeto(&self, py: Python<'_>, path: PathBuf, overwrite: bool) -> PyResult<()> {
        use pyo3::exceptions::PyFileExistsError;
        use std::io::BufWriter;
        // Writing back over our own backing file is allowed only via
        // `flush()`'s drop+rewrite+reopen path, because doing so
        // through a sibling-rename would unmap the live `inner` and
        // `updater` mappings out from under any held `hdu.section`
        // bindings. Detect the self-write case and dispatch
        // accordingly: with `overwrite=False` we honor the
        // FileExistsError contract; with `overwrite=True` we behave
        // exactly like `flush()` (rewrite, atomic rename, reopen).
        let writes_to_self = self
            .original_path
            .as_ref()
            .and_then(|orig| std::fs::canonicalize(&path).ok().map(|t| &t == orig))
            .unwrap_or(false);
        if writes_to_self {
            // A self-write is a mutation of the source file; only
            // permitted in update mode. Writing to a different path
            // from a readonly handle is allowed, and is the standard
            // way to save a copy.
            self.ensure_writable()?;
            if !overwrite {
                return Err(PyFileExistsError::new_err(format!(
                    "FitsFile.writeto: {} already exists; pass overwrite=True to replace",
                    path.display(),
                )));
            }
            // Force a full rewrite even if no edits are pending so
            // the on-disk bytes match what `materialize_all + encode`
            // would produce.
            self.dirty.definite.store(true, Ordering::Release);
            return self.persist_full_rewrite(py);
        }
        if !overwrite && path.exists() {
            return Err(PyFileExistsError::new_err(format!(
                "FitsFile.writeto: {} already exists; pass overwrite=True to replace",
                path.display(),
            )));
        }
        let (tmp, tmp_file) = create_sibling_temp(&path, "FitsFile.writeto")?;
        let write_result: PyResult<()> = (|| {
            let mut w = crate::FitsWriter::new(BufWriter::new(tmp_file));
            if self.stamp_checksums.load(Ordering::Relaxed) {
                w = w.with_checksums();
            }
            // Snapshot slots WITHOUT forcing every Pending HDU to
            // materialize. Pending slots whose source-file role
            // (primary vs extension) matches the destination role
            // can be streamed straight from the source via
            // `hdu_raw_padded`; everything else falls back to
            // materialize + re-encode. This keeps writeto() at
            // O(materialized + raw bytes) RSS instead of O(file).
            let snapshot = self.snapshot_slots(py);
            if snapshot.is_empty() {
                return Err(PyValueError::new_err(
                    "FitsFile.writeto: refusing to write a file with zero HDUs",
                ));
            }
            // Pre-pass: a Pending slot whose file index doesn't
            // match its destination index in the (primary vs
            // extension) sense must be re-framed; force-materialize
            // those.
            let mut snapshot = snapshot;
            let stamping = self.stamp_checksums.load(Ordering::Relaxed);
            for (dst_idx, slot) in snapshot.iter_mut().enumerate() {
                if let WritetoSlot::Pending(file_idx) = slot {
                    // Stamping checksums requires re-encoding via
                    // `write_hdu`; raw-streamed Pending slots
                    // bypass that path.
                    let needs_reframe = (*file_idx == 0) ^ (dst_idx == 0);
                    if needs_reframe || stamping {
                        let materialized = self.materialize_at(py, dst_idx)?;
                        *slot = WritetoSlot::Materialized(materialized);
                    }
                }
            }
            // Synthesize an empty primary if the first emitted HDU
            // can't legally be a primary.
            let mut emitted_primary = false;
            let needs_synth_primary = match snapshot.first() {
                Some(WritetoSlot::Materialized(p)) => !is_image_like(py, p),
                // Pending slots that survived the reframe pass are
                // already at their original primary/extension
                // position, so they're safe to stream as-is.
                _ => false,
            };
            if needs_synth_primary {
                let (h, d) = empty_primary_header_and_bytes();
                w.write_hdu(&h, &d).into_py_result()?;
                emitted_primary = true;
            }
            for slot in &snapshot {
                match slot {
                    WritetoSlot::Pending(file_idx) => {
                        let st = self.lock_state();
                        let file = st.file.as_ref().ok_or_else(|| {
                            PyValueError::new_err(
                                "FitsFile.writeto: backing file dropped before write",
                            )
                        })?;
                        let raw = file
                            .hdu_raw_padded(*file_idx)
                            .into_py_result()?
                            .ok_or_else(|| {
                                PyValueError::new_err(format!(
                                    "FitsFile.writeto: source HDU {file_idx} out of range",
                                ))
                            })?;
                        w.write_raw_padded(&raw)
                            .map_err(|e| super::err_to_py(crate::error::FitsError::Io(e)))?;
                        emitted_primary = true;
                    }
                    WritetoSlot::Materialized(p) => {
                        let is_primary = !emitted_primary;
                        let (header, data) = encode_hdu(py, p, is_primary)?;
                        w.write_hdu(&header, &data).into_py_result()?;
                        emitted_primary = true;
                    }
                }
            }
            w.finish()
                .map_err(|e| super::err_to_py(crate::error::FitsError::Io(e)))?;
            Ok(())
        })();
        if let Err(e) = write_result {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            super::err_to_py(crate::error::FitsError::Io(e))
        })
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        // --- Collect per-HDU display rows ---
        struct Row {
            name: String,
            type_str: &'static str,
            details: String,
            has_wcs: bool,
        }
        use std::fmt::Write as _;

        let Ok(hdus) = self.materialize_all(py) else {
            // If a slot fails to materialize just report the count.
            return format!(
                "FitsFile({:?}, {} HDUs)",
                self.filename.as_deref().unwrap_or("<memory>"),
                self.lock_state().slots.len(),
            );
        };
        let n = hdus.len();
        let hdu_word = if n == 1 { "HDU" } else { "HDUs" };
        let fname = self.filename.as_deref().unwrap_or("<memory>");

        if n == 0 {
            return format!("FitsFile({fname:?}, 0 HDUs)");
        }

        let rows: Vec<Row> = hdus
            .iter()
            .map(|hdu| {
                let bound = hdu.bind(py);

                if let Ok(img) = bound.cast::<PyImageHdu>() {
                    let img = img.borrow();
                    let dtype = crate::python::hdu::bitpix_numpy_dtype(img.bitpix);
                    let details = {
                        // Use the cached `axes` snapshot so the
                        // info table never triggers a lazy data
                        // read (the eager pixel materialization
                        // would defeat the lazy design).
                        let dims: Vec<String> = img.axes.iter().map(ToString::to_string).collect();
                        if dims.is_empty() {
                            dtype.to_string()
                        } else {
                            // Left-pad dtype to 7 chars so "uint8" aligns
                            // with "float32".
                            format!("{dtype:<7}  {}", dims.join(" \u{00d7} "))
                        }
                    };
                    let name = extname_from_header(&img.header);
                    let has_wcs = header_has_wcs(&img.header);
                    Row {
                        name,
                        type_str: "Image",
                        details,
                        has_wcs,
                    }
                } else if let Ok(tbl) = bound.cast::<PyBinTable>() {
                    let tbl = tbl.borrow();
                    let n_cols = tbl.column_names.len();
                    let details = format!("{} rows x {n_cols} cols", tbl.n_rows);
                    let name = extname_from_header(&tbl.header);
                    Row {
                        name,
                        type_str: "BinTable",
                        details,
                        has_wcs: false,
                    }
                } else if let Ok(tbl) = bound.cast::<PyAsciiTable>() {
                    let tbl = tbl.borrow();
                    let n_cols = tbl.column_names.len();
                    let details = format!("{} rows x {n_cols} cols", tbl.n_rows);
                    let name = extname_from_header(&tbl.header);
                    Row {
                        name,
                        type_str: "AsciiTable",
                        details,
                        has_wcs: false,
                    }
                } else {
                    Row {
                        name: String::new(),
                        type_str: "Unknown",
                        details: String::new(),
                        has_wcs: false,
                    }
                }
            })
            .collect();

        // Dynamic column widths for clean alignment.
        let idx_w = if n >= 100 {
            3
        } else if n >= 10 {
            2
        } else {
            1
        };
        let name_w = rows.iter().map(|r| r.name.len()).max().unwrap_or(0).max(4);
        let type_w = rows.iter().map(|r| r.type_str.len()).max().unwrap_or(0);

        let mut out = format!("FitsFile({fname:?}, {n} {hdu_word})\n");
        for (i, row) in rows.iter().enumerate() {
            let wcs_tag = if row.has_wcs { "  WCS" } else { "" };
            let _ = writeln!(
                out,
                "  [{i:>idx_w$}] {name:<name_w$}  {tp:<type_w$}  {det}{wcs_tag}",
                name = row.name,
                tp = row.type_str,
                det = row.details,
            );
        }
        out.trim_end().to_string()
    }

    /// Verify per-HDU ``CHECKSUM`` and ``DATASUM`` cards.
    ///
    /// Streams the data section of each HDU directly from disk in
    /// fixed-size chunks (no full materialization) and compares
    /// against the values stored in the HDU header. HDUs that have
    /// neither card are reported with both fields ``None``; HDUs
    /// that only have one of the two are reported with the missing
    /// field ``None`` and the present one as ``True`` / ``False``.
    ///
    /// Returns
    /// -------
    /// list[dict]
    ///   One dict per HDU, in file order. Keys:
    ///
    ///   * ``hdu`` -- 0-based HDU index (``int``).
    ///   * ``checksum_ok`` -- ``True``/``False``/``None``.
    ///   * ``datasum_ok`` -- ``True``/``False``/``None``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the file has no backing path -- built with the
    ///   ``FitsFile()`` constructor rather than :func:`fitsy.open` --
    ///   or if :meth:`close` was already called.
    /// FitsError
    ///   On an I/O failure, or if a header fails to parse.
    ///
    /// Notes
    /// -----
    /// Reads the header and data bytes currently on disk, in both
    /// ``mode='readonly'`` and ``mode='update'``. An in-memory edit
    /// that has not yet reached the file through :meth:`flush`,
    /// :meth:`close`, or a clean ``__exit__`` is not reflected.
    fn verify_checksums(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyBool, PyDict, PyList};
        let st = self.lock_state();
        let file = st.file.as_ref().ok_or_else(|| {
            PyValueError::new_err("FitsFile.verify_checksums: file already closed")
        })?;
        let reports = file.verify_checksums().into_py_result()?;
        let list = PyList::empty(py);
        for r in reports {
            let d = PyDict::new(py);
            d.set_item("hdu", r.hdu)?;
            let chk: Py<PyAny> = match r.checksum_ok {
                Some(b) => PyBool::new(py, b).to_owned().into_any().unbind(),
                None => py.None(),
            };
            d.set_item("checksum_ok", chk)?;
            let dsm: Py<PyAny> = match r.datasum_ok {
                Some(b) => PyBool::new(py, b).to_owned().into_any().unbind(),
                None => py.None(),
            };
            d.set_item("datasum_ok", dsm)?;
            list.append(d)?;
        }
        Ok(list.into_any().unbind())
    }

    /// Enable ``CHECKSUM`` / ``DATASUM`` stamping on every HDU that
    /// :meth:`writeto` emits, and on every HDU that :meth:`flush`
    /// rewrites.
    ///
    /// When enabled, every HDU written gains freshly computed
    /// ``CHECKSUM`` and ``DATASUM`` cards (per the FITS Checksum
    /// Proposal). An existing placeholder card in the header is
    /// overwritten in place; a missing one is inserted. The flag
    /// stays on for the lifetime of the ``FitsFile`` object; there
    /// is no way to turn it back off or to stamp only one HDU.
    ///
    /// Notes
    /// -----
    /// This does not stamp anything immediately. fitsy computes each
    /// value during a write, when the final byte layout of the HDU
    /// is known. Both :meth:`writeto` and :meth:`flush` then stamp
    /// every HDU. This call marks the file as needing a rewrite, so
    /// a :meth:`flush` with no other pending edit still writes the
    /// cards. To check the result, call :meth:`verify_checksums` on
    /// the written file.
    fn add_checksums(&self) {
        self.stamp_checksums.store(true, Ordering::Relaxed);
        // Stamping happens during a full rewrite. Without this flag a
        // `flush()` with no other pending edit takes the fsync-only
        // path, and the cards never reach the file.
        self.dirty.definite.store(true, Ordering::Release);
    }

    /// Flush pending edits to disk.
    ///
    /// A no-op when the file was opened ``mode='readonly'`` or
    /// ``mode='denywrite'``.
    ///
    /// In ``mode='update'``, a mutation that an in-place pixel patch
    /// cannot satisfy -- a header edit, ``hdu.data = new_array``,
    /// :meth:`append`, ``del file[i]``, a fancy or dtype-mismatched
    /// ``section[...]`` write, or an edit on a tile-compressed image
    /// -- rewrites the whole file through a sibling temp file and an
    /// atomic rename. A slot the caller never touched is streamed
    /// byte-for-byte from the original file, with no decode or
    /// re-encode. Reading ``hdu.data`` alone does not by itself force
    /// a rewrite: fitsy re-reads that HDU's data section from disk
    /// and compares it against the cached array, and rewrites only if
    /// the two differ.
    ///
    /// Mixing modes: if you issue an in-place ``section[...]`` patch
    /// and then a non-patch mutation in the same session, the patch
    /// reaches disk first, through ``pwrite``, and the subsequent
    /// ``flush()`` then performs a full rewrite that includes the
    /// patched bytes, by streaming the already-patched source file.
    /// The patch is not lost.
    ///
    /// Crash safety: an in-place patch uses ``pwrite`` with no undo
    /// journal; a process death mid-patch can leave the file with
    /// some rows updated and others not. The full-rewrite path is
    /// crash-safe, because it writes to a sibling temp file and
    /// renames atomically once the bytes are durable. The parent
    /// directory is not separately ``fsync``\ ed, so a power loss
    /// between the rename and the next directory commit can, in
    /// theory, leave the rename invisible after reboot on a
    /// non-journaling filesystem. A stale ``.fitsy-tmp.*`` sibling
    /// left by a crashed rewrite is harmless and may be deleted.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///   If a rewrite is needed and an HDU slot holds an object that
    ///   is none of the four wrapper classes; see :meth:`writeto`.
    /// ValueError
    ///   If a rewrite is needed and would leave zero HDUs, or if an
    ///   internal lock was poisoned by an earlier panic.
    /// FitsError
    ///   On an I/O failure, or if an HDU cannot be encoded for write.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        if self.updater.is_none() {
            return Ok(());
        }
        if self.dirty.definite.swap(false, Ordering::AcqRel) {
            self.dirty.handed_out.store(false, Ordering::Release);
            self.persist_full_rewrite(py)?;
        } else if self.dirty.handed_out.swap(false, Ordering::AcqRel)
            && self.handed_out_data_changed(py)
        {
            self.persist_full_rewrite(py)?;
        } else if let Some(updater) = self.updater.as_ref() {
            let guard = updater
                .lock()
                .map_err(|_| PyValueError::new_err("FitsFile: updater mutex poisoned"))?;
            guard.flush().into_py_result()?;
        }
        Ok(())
    }

    /// Flush pending edits (if any) and release the source file
    /// handle.
    ///
    /// After ``close()``, the slot list and any HDU wrapper Python
    /// already holds remain usable as in-memory data, but the
    /// underlying file handle is dropped: reading a still-``Pending``
    /// slot then raises :class:`ValueError`. :meth:`writeto` always
    /// visits every slot, so it raises the same way if any slot is
    /// still ``Pending``. A later :meth:`flush` call raises only if
    /// it decides a rewrite is needed and a ``Pending`` slot remains;
    /// with nothing left to write, it is a successful no-op.
    ///
    /// Idempotent: calling ``close()`` more than once is safe.
    ///
    /// Raises
    /// ------
    /// TypeError, ValueError, FitsError
    ///   Under the same conditions as :meth:`flush`, which this
    ///   method calls first.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        // Best-effort flush; surface errors so the caller sees them.
        self.flush(py)?;
        // Drop the read-only file handle. The writable updater is
        // held via `Arc` clones inside materialized `PyImageHdu`
        // objects; bumping its generation invalidates the fast
        // in-place patch path so subsequent writes through stale
        // wrappers can no longer reach the file (they fall through
        // to the dirty-flag path, which raises on the next flush
        // because `state.file` is gone).
        self.invalidate_bindings();
        let mut st = self.lock_state();
        st.file = None;
        Ok(())
    }

    /// Context-manager entry.
    ///
    /// Returns
    /// -------
    /// FitsFile
    ///   ``self``.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context-manager exit.
    ///
    /// On a clean exit -- no exception in flight -- calls
    /// :meth:`flush` and lets any error it raises propagate.
    ///
    /// On exit due to an in-flight exception, makes any already
    /// applied in-place pixel patch durable, but does not rewrite the
    /// file for a pending header or full-array edit; any error from
    /// that best-effort step is discarded.
    ///
    /// Returns
    /// -------
    /// bool
    ///   Always ``False``. An in-flight exception is never
    ///   suppressed.
    #[pyo3(signature = (exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Option<Py<PyAny>>,
        _exc_val: Option<Py<PyAny>>,
        _exc_tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        // If we are unwinding because of an in-flight Python
        // exception, do not mask it: best-effort flush, swallow any
        // secondary error.
        if exc_type.is_some() {
            if let Some(updater) = self.updater.as_ref()
                && let Ok(guard) = updater.lock()
            {
                let _ = guard.flush();
            }
            return Ok(false);
        }
        // Clean exit: persist any dirty edits and surface errors.
        self.flush(py)?;
        Ok(false)
    }
}

impl PyFitsFile {
    /// Reject a mutation on a file opened read-only.
    ///
    /// Every mutating pymethod calls this first, before touching the
    /// slot list.
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] if `self.read_only` is `true`.
    fn ensure_writable(&self) -> PyResult<()> {
        if self.read_only {
            Err(PyValueError::new_err(
                "FitsFile: opened read-only; reopen with mode='update' to enable mutations",
            ))
        } else {
            Ok(())
        }
    }

    /// Mark every cached `UpdateBinding` as stale by bumping the
    /// updater's generation tag. Call after any structural mutation
    /// (`del`/`insert`/`append`/`__setitem__`) so previously-issued
    /// `(arc, hdu_idx)` bindings refuse the fast in-place pwrite
    /// path instead of patching what is now a different HDU.
    fn invalidate_bindings(&self) {
        if let Some(updater) = self.updater.as_ref()
            && let Ok(mut g) = updater.lock()
        {
            g.bump_generation();
        }
    }
}

/// Iterator over HDUs in a `PyFitsFile`.
#[pyclass(name = "_HduIter", module = "fitsy")]
#[derive(Debug)]
pub struct HduIter {
    items: Vec<Py<PyAny>>,
    pos: usize,
}

#[pymethods]
impl HduIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Py<PyAny>> {
        let py = slf.py();
        let i = slf.pos;
        if i >= slf.items.len() {
            return None;
        }
        slf.pos += 1;
        Some(slf.items[i].clone_ref(py))
    }
}

/// Coerce a Python value into a live HDU instance suitable for
/// storage in `PyFitsFile.hdus`. Builders are promoted to live
/// `ImageHdu` / `BinTable` / `AsciiTable` instances so callers can
/// inspect and edit them after `append`/`insert`.
///
/// # Errors
///
/// Returns [`PyTypeError`] if `v` is neither an `ImageHdu`,
/// `BinTable`, or `AsciiTable` instance, nor a builder returned by
/// `fitsy.image`, `fitsy.bintable`, or `fitsy.ascii_table`.
fn coerce_to_hdu(py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use crate::python::writer::{PyAsciiTableBuilder, PyBinTableBuilder, PyImageBuilder};
    if v.extract::<PyRef<'_, PyImageHdu>>().is_ok()
        || v.extract::<PyRef<'_, PyBinTable>>().is_ok()
        || v.extract::<PyRef<'_, PyAsciiTable>>().is_ok()
    {
        return Ok(v.clone().unbind());
    }
    if let Ok(b) = v.extract::<PyRef<'_, PyImageBuilder>>() {
        let header = b.header.clone();
        let data = b.data.clone();
        drop(b);
        let img = PyImageHdu::from_built_bytes(py, header, data, false)?;
        return Ok(Py::new(py, img)?.into_any());
    }
    if let Ok(b) = v.extract::<PyRef<'_, PyBinTableBuilder>>() {
        let py_t = PyBinTable::from_built_bytes(b.header.clone(), b.data.clone());
        return Ok(Py::new(py, py_t)?.into_any());
    }
    if let Ok(b) = v.extract::<PyRef<'_, PyAsciiTableBuilder>>() {
        let py_t = PyAsciiTable::from_built_bytes(b.header.clone(), b.data.clone());
        return Ok(Py::new(py, py_t)?.into_any());
    }
    Err(PyTypeError::new_err(
        "expected an ImageHdu / BinTable / AsciiTable instance or a builder",
    ))
}

/// True when the HDU can serve as the primary HDU, so `writeto`
/// does not have to prepend an empty one.
///
/// An image HDU and an image builder both qualify. So does a
/// random-groups HDU: Standard Sec.6 puts random groups records in
/// the primary HDU only, and its header already carries `SIMPLE`,
/// never `XTENSION`. Prepending an empty primary before one would
/// push it into extension position, where it cannot be written.
fn is_image_like(py: Python<'_>, hdu: &Py<PyAny>) -> bool {
    use crate::python::writer::PyImageBuilder;
    let b = hdu.bind(py);
    b.extract::<PyRef<'_, PyImageHdu>>().is_ok()
        || b.extract::<PyRef<'_, PyImageBuilder>>().is_ok()
        || b.extract::<PyRef<'_, super::hdu::PyRandomGroups>>().is_ok()
}

/// Build an empty primary image header (`NAXIS = 0`) for the
/// auto-prepend case.
fn empty_primary_header_and_bytes() -> (crate::Header, Vec<u8>) {
    use crate::Value;
    let mut h = crate::Header::empty();
    let _ = h.set("SIMPLE", Value::Logical(true), Some("conforming FITS"));
    let _ = h.set("BITPIX", Value::Integer(8), None);
    let _ = h.set("NAXIS", Value::Integer(0), None);
    let _ = h.set("EXTEND", Value::Logical(true), None);
    (h, Vec::new())
}

/// Encode one HDU's current Python state into header + bytes
/// for serialization.
///
/// Handles every HDU kind the bindings wrap: [`PyImageHdu`],
/// [`PyBinTable`], [`PyAsciiTable`] and
/// [`super::hdu::PyRandomGroups`]. Only an image HDU re-encodes its
/// pixels. The other three keep the data section they were loaded
/// with, so each hands back its own bytes unchanged.
///
/// Called only for a `HduSlot::Materialized` /
/// `WritetoSlot::Materialized` slot; a `Pending` slot streams its
/// source bytes directly instead and never reaches this function.
///
/// # Errors
///
/// Returns [`PyTypeError`] if `hdu` is none of the four wrapper
/// types. Otherwise, propagates the error from
/// [`PyImageHdu::encode`] for an image HDU.
fn encode_hdu(
    py: Python<'_>,
    hdu: &Py<PyAny>,
    is_primary: bool,
) -> PyResult<(crate::Header, Vec<u8>)> {
    let bound = hdu.bind(py);
    if let Ok(img) = bound.extract::<PyRef<'_, PyImageHdu>>() {
        return img.encode(py, is_primary);
    }
    if let Ok(t) = bound.extract::<PyRef<'_, PyBinTable>>() {
        return Ok((t.header_clone(), t.raw.clone()));
    }
    if let Ok(t) = bound.extract::<PyRef<'_, PyAsciiTable>>() {
        return Ok((t.header_clone(), t.raw.clone()));
    }
    // A random-groups HDU owns its header and its big-endian data
    // section, and the bindings expose no way to edit either. Write
    // both back as they were read.
    if let Ok(rg) = bound.extract::<PyRef<'_, super::hdu::PyRandomGroups>>() {
        return Ok((rg.header_clone(), rg.data_clone()));
    }
    Err(PyTypeError::new_err(
        "FitsFile: HDU slot has unsupported type",
    ))
}

/// Build the Python HDU wrapper for `hdu`, dispatching on its kind.
///
/// `i` is `hdu`'s index in `file`; a plain image HDU keeps a
/// [`super::hdu::ReadBinding`] to `file` at that index so `data` and
/// `section` can pread fresh bytes later. `updater`, when `Some`,
/// additionally attaches a [`super::hdu::UpdateBinding`] so an
/// in-place `section[a:b] = arr` patch can reach the file.
///
/// # Errors
///
/// Returns the Python exception mapped from `fitsy.FitsError` if
/// decompressing a tile-compressed image fails. Returns
/// [`PyTypeError`] if `hdu` is a kind the Python bindings do not
/// wrap: a conforming extension with an unrecognized `XTENSION`.
fn wrap_hdu(
    py: Python<'_>,
    i: usize,
    hdu: crate::Hdu<'_>,
    header: PyHeader,
    read_only: bool,
    updater: Option<&Arc<Mutex<FitsUpdater>>>,
    dirty_flag: Option<Arc<DirtyFlags>>,
    file: Arc<FitsFile>,
) -> PyResult<Py<PyAny>> {
    use crate::Hdu;
    match hdu {
        Hdu::Image(img) => {
            let mut py_img = PyImageHdu::from_image(py, &img, header, read_only);
            // Attach the lazy-read source so `data` / `section`
            // can pread fresh bytes on demand. Skipping this
            // would force the only path to be eager `from_image`
            // materialization (which we removed).
            py_img.read_binding = Some(super::hdu::ReadBinding {
                file: file.clone(),
                hdu_idx: i,
                axes: img.axes().to_vec(),
            });
            py_img.wcs_file = Some(file);
            py_img.dirty.clone_from(&dirty_flag);
            if let Some(u) = updater {
                let generation = u.lock().map_or(u64::MAX, |g| g.generation());
                py_img.update_binding = Some(super::hdu::UpdateBinding {
                    updater: u.clone(),
                    hdu_idx: i,
                    generation,
                });
            }
            Ok(Py::new(py, py_img)?.into_any())
        }
        Hdu::BinTable(t) => {
            let py_t = PyBinTable::from_table(&t, header)?;
            Ok(Py::new(py, py_t)?.into_any())
        }
        Hdu::AsciiTable(t) => {
            let py_t = PyAsciiTable::from_table(&t, header)?;
            Ok(Py::new(py, py_t)?.into_any())
        }
        #[cfg(feature = "compression")]
        Hdu::CompressedImage(c) => {
            // Decompress on read: the BINTABLE / ZIMAGE wrapper is
            // hidden from Python and replaced with the synthetic
            // image view (BITPIX/NAXISn rewritten from Z*).
            let _ = header;
            let owned = c.as_image().into_py_result()?;
            let mut py_img = PyImageHdu::from_built_bytes(
                py,
                owned.header().clone(),
                owned.raw_bytes().to_vec(),
                read_only,
            )?;
            // No `update_binding`: tile-compressed images cannot be
            // patched in place. Mutations fall through the cache +
            // dirty path so `flush()` rewrites the file.
            py_img.dirty = dirty_flag;
            // The decode above already read the pixels, so this path
            // sets no `read_binding`. `wcs()` still needs the file to
            // resolve a `-TAB` axis against a sibling BINTABLE.
            py_img.wcs_file = Some(file);
            Ok(Py::new(py, py_img)?.into_any())
        }
        Hdu::RandomGroups(rg) => {
            let py_rg = super::hdu::PyRandomGroups::from_hdu(&rg, header);
            Ok(Py::new(py, py_rg)?.into_any())
        }
        Hdu::Conforming(h) => Err(PyTypeError::new_err(format!(
            "HDU has XTENSION={:?}, which is not supported by the \
             Python wrapper. Use the Rust API for raw access.",
            h.xtension(),
        ))),
        #[allow(
            unreachable_patterns,
            reason = "Hdu is #[non_exhaustive]; needed for forward compatibility"
        )]
        _ => Err(PyTypeError::new_err("HDU kind is not wrapped for Python")),
    }
}
