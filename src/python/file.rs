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
        .entries()
        .iter()
        .find(|e| e.keyword == "EXTNAME")
        .and_then(|e| e.value.as_ref())
        .and_then(|v| match v {
            crate::header::Value::String(s) => Some(s.trim().to_string()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Returns true when the header declares at least one non-empty `CTYPEi`,
/// which is the minimal indicator that a WCS is present.  This is a fast
/// keyword scan -- no full WCS parse -- so repr stays cheap.
fn header_has_wcs(h: &PyHeader) -> bool {
    h.lock().entries().iter().any(|e| {
        e.keyword.starts_with("CTYPE")
            && matches!(
                e.value.as_ref(),
                Some(crate::header::Value::String(s)) if !s.trim().is_empty()
            )
    })
}

/// Parse a FITS file open mode into the read-only flag.
///
/// We follow astropy's `fits.open` naming so users coming from astropy
/// don't have to relearn the vocabulary:
///
/// * `"readonly"` (default) -- read-only handle; header mutation and
///   `writeto` raise.
/// * `"update"` -- read/write handle; in-place edits are preserved on
///   `writeto`.
///
/// Astropy's `"append"`, `"denywrite"` and `"ostream"` are not
/// implemented; pass `fitsy.write` instead for output-only workflows.
fn parse_mode(mode: &str) -> PyResult<bool> {
    match mode {
        // 'denywrite' is astropy's stricter readonly (no other
        // process may open the file for writing). We don't enforce
        // the OS-level lock today, but we honour the read-only
        // intent so existing astropy code keeps working.
        "readonly" | "denywrite" => Ok(true),
        "update" => Ok(false),
        "append" | "ostream" => Err(PyValueError::new_err(format!(
            "fitsy.open: mode {mode:?} is not supported; use `fitsy.write(path, hdus)` to \
             create a new file, or open with mode='readonly' and call `writeto(new_path)` to \
             save a modified copy"
        ))),
        other => Err(PyValueError::new_err(format!(
            "fitsy.open: mode must be 'readonly', 'denywrite', or 'update' (astropy convention); \
             got {other:?}"
        ))),
    }
}

/// Open a FITS file by path.
///
/// Parameters
/// ----------
/// path : str or os.PathLike
///     Filesystem path to the FITS file.
/// mode : {'readonly', 'update'}, optional
///     ``'readonly'`` (default) opens read-only; header mutations and
///     :meth:`FitsFile.writeto` raise :class:`ValueError`. Matches
///     astropy's ``fits.open`` mode of the same name.
///
///     ``'update'`` opens read/write. Header edits and image-pixel
///     in-place edits (``hdu.data[...] = x``) are preserved on
///     :meth:`FitsFile.writeto`. Table column data is read-only
///     in this release; reconstruct the table with
///     :func:`fitsy.bintable` to change column values.
/// lenient : bool, optional
///     When True, accept ``SIMPLE = F`` primary headers (non-standard
///     FITS files written by some pipelines). Does **not** downgrade
///     any other validation errors to warnings. Default False.
///
/// Returns
/// -------
/// FitsFile
///     A read-only or read/write handle depending on ``mode``.
///
/// Raises
/// ------
/// ValueError
///     If ``mode`` is not one of the recognized values.
/// FitsError
///     On parse failures or I/O errors.
///
/// Examples
/// --------
/// >>> import fitsy
/// >>> with fitsy.open("image.fits") as f:
/// ...     img = f[0]
/// ...     print(img.axes)
#[pyfunction]
#[pyo3(signature = (path, mode="readonly", lenient=false))]
pub fn open(_py: Python<'_>, path: PathBuf, mode: &str, lenient: bool) -> PyResult<PyFitsFile> {
    let read_only = parse_mode(mode)?;
    let inner = crate::FitsOpenOptions::new()
        .lenient(lenient)
        .open(&path)
        .into_py_result()?;
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
        dirty: Arc::new(AtomicBool::new(false)),
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

/// Mirror of [`HduSlot`] used as a snapshot inside
/// [`PyFitsFile::writeto`] (and the rewrite path) so the slot list
/// can be classified and re-framed without holding the state lock
/// across Python-callback work.
#[derive(Debug)]
enum WritetoSlot {
    Pending(usize),
    Materialized(Py<PyAny>),
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
/// astropy parity: ``FitsFile`` behaves like astropy's
/// ``astropy.io.fits.HDUList``. Slots are typed Python
/// objects (:class:`ImageHdu` / :class:`BinTable` /
/// :class:`AsciiTable`); they own their header and data and
/// survive after the file handle is dropped.
///
/// In ``mode='readonly'`` (the default), in-place edits
/// (``f[0].data[...] = x``, ``f[0].header["K"] = v``,
/// ``f.append(hdu)``, ``del f[i]``) are kept in memory and
/// preserved on the next :meth:`writeto`; the on-disk source
/// file is never modified.
///
/// In ``mode='update'``, those same edits are written back to
/// the source file on :meth:`flush`, :meth:`close`, or clean
/// ``__exit__``. Pixel patches via ``f[i].section[...] = arr``
/// are written immediately via positional ``pwrite``.
///
/// Use the module-level :func:`open` factory rather than
/// constructing this class directly.
///
/// Examples
/// --------
/// >>> with fitsy.open("image.fits", mode="update") as f:
/// ...     f[0].data[0, 0] = 42.0
/// ...     # changes flushed automatically on __exit__
/// >>> with fitsy.open("image.fits") as f:    # readonly
/// ...     f[0].header["OBSERVER"] = "you"
/// ...     f.writeto("edited.fits")           # original untouched
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
    pub(crate) dirty: Arc<AtomicBool>,
    /// When true, the next `writeto` / `flush` will compute and
    /// stamp `CHECKSUM` / `DATASUM` cards on every emitted HDU
    /// via [`crate::FitsWriter::with_checksums`]. Toggled on by
    /// [`add_checksums`](Self::add_checksums); stays on for the
    /// lifetime of the file.
    pub(crate) stamp_checksums: AtomicBool,
}

impl PyFitsFile {
    fn lock_state(&self) -> std::sync::MutexGuard<'_, FileState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Materialize the slot at `slot_idx` into a live Python HDU
    /// wrapper, replacing the `Pending` placeholder. Returns a new
    /// owned reference (clone of the cached one).
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
    /// Returns `Some(py_image_hdu)` when the HDU at `file_idx` is a
    /// plain image (primary or `XTENSION='IMAGE'`, not random
    /// groups, not a tile-compressed image). Returns `None`
    /// otherwise so the caller can fall back to the generic
    /// `file.hdu(i)` dispatch.
    ///
    /// This avoids the per-HDU data cache (the bytes are read
    /// lazily on demand via `read_data_owned` /
    /// `read_image_subarray_be`), so `fitsy.open(...)` followed by
    /// iteration over header-only properties costs only the
    /// already-cached header parses -- no pixel bytes resident.
    fn try_image_fast_path(
        &self,
        py: Python<'_>,
        file_idx: usize,
        file: &Arc<FitsFile>,
        dirty_flag: Option<Arc<AtomicBool>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        use crate::Value;
        use crate::data::Bitpix;
        let header = file.parsed_header(file_idx).into_py_result()?;
        // Detect plain-image kind without reading data.
        let is_image = if file_idx == 0 {
            // Primary: image unless it's random-groups
            // (NAXIS1==0, NAXIS>=2, GROUPS=T).
            let naxis = header.naxis().unwrap_or(0);
            let naxis1 = header.first("NAXIS1").and_then(|v| match v {
                Value::Integer(i) => Some(*i),
                _ => None,
            });
            let groups = matches!(header.first("GROUPS"), Some(Value::Logical(true)));
            !(naxis1 == Some(0) && naxis >= 2 && groups)
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

    /// Rewrite the backing file to absorb the in-memory edits that
    /// could not be satisfied by an in-place pixel-patch (header
    /// edits, structural mutations, `set_data`, fancy slice writes).
    ///
    /// Streams raw bytes for slots the user never touched
    /// (`HduSlot::Pending`); only re-encodes materialised slots.
    /// On success, drops the original `FitsFile` and `FitsUpdater`
    /// and re-opens them against the freshly written file so that
    /// further mutations and pixel-patches keep working.
    fn persist_full_rewrite(&self, py: Python<'_>) -> PyResult<()> {
        use std::fs::OpenOptions;
        use std::io::{BufWriter, Write};
        enum SlotKind {
            Pending(usize),
            Materialized(Py<PyAny>),
        }

        let original_path = self.original_path.clone().ok_or_else(|| {
            PyValueError::new_err(
                "FitsFile.flush: cannot rewrite an in-memory file; use writeto(path)",
            )
        })?;
        let parent = original_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let basename = original_path.file_name().map_or_else(
            || std::ffi::OsString::from("fitsy-out"),
            std::ffi::OsStr::to_os_string,
        );

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
        let snapshot: Vec<SlotKind> = {
            let st = self.lock_state();
            st.slots
                .iter()
                .map(|s| match s {
                    HduSlot::Pending(i) => SlotKind::Pending(*i),
                    HduSlot::Materialized(p) => SlotKind::Materialized(p.clone_ref(py)),
                })
                .collect()
        };
        if snapshot.is_empty() {
            return Err(PyValueError::new_err(
                "FitsFile.flush: refusing to rewrite a file with zero HDUs",
            ));
        }

        // Open a sibling temp file with O_CREAT|O_EXCL.
        let (tmp_path, tmp_file) = {
            let mut chosen: Option<(PathBuf, std::fs::File)> = None;
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
                    Ok(f) => {
                        chosen = Some((candidate, f));
                        break;
                    }
                    Err(e) => last_err = Some(e),
                }
            }
            chosen.ok_or_else(|| {
                super::err_to_py(crate::error::FitsError::Io(last_err.unwrap_or_else(|| {
                    std::io::Error::other("FitsFile.flush: could not create temp file")
                })))
            })?
        };

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
                if let SlotKind::Pending(file_idx) = slot {
                    let needs_reframe = (*file_idx == 0) ^ (dst_idx == 0);
                    if needs_reframe || stamping {
                        let materialized = self.materialize_at(py, dst_idx)?;
                        *slot = SlotKind::Materialized(materialized);
                    }
                }
            }
            let mut writer = crate::FitsWriter::new(&mut bw);
            if self.stamp_checksums.load(Ordering::Relaxed) {
                writer = writer.with_checksums();
            }
            let mut emitted_primary = false;
            // Determine if we need to synthesise an empty primary:
            // only when the first emitted HDU isn't an image-like
            // (BinTable / AsciiTable can't be a primary).
            let needs_synth_primary = matches!(
                snapshot.first(),
                Some(SlotKind::Materialized(p)) if !is_image_like(py, p)
            );
            if needs_synth_primary {
                let (h, d) = empty_primary_header_and_bytes();
                writer.write_hdu(&h, &d).into_py_result()?;
                emitted_primary = true;
            }
            for slot in &snapshot {
                match slot {
                    SlotKind::Pending(file_idx) => {
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
                    SlotKind::Materialized(p) => {
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
        let new_inner = crate::FitsOpenOptions::new()
            .open(&original_path)
            .into_py_result()?;
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
            dirty: Arc::new(AtomicBool::new(false)),
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

    /// Return the ``i``-th HDU (``file[i]``) or the first HDU named
    /// ``EXTNAME`` (``file["NAME"]``).
    ///
    /// Negative integer indices count from the end (Python convention).
    /// Tuple keys ``file["NAME", ver]`` select by ``EXTNAME`` and
    /// ``EXTVER`` (astropy parity).
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // astropy-style ("EXTNAME", EXTVER) tuple lookup.
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

    /// Replace ``file[i]``. Accepts an HDU instance or a builder.
    fn __setitem__(&self, py: Python<'_>, i: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_writable()?;
        self.dirty.store(true, Ordering::Release);
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

    /// Remove ``file[i]``.
    fn __delitem__(&self, i: isize) -> PyResult<()> {
        self.ensure_writable()?;
        self.dirty.store(true, Ordering::Release);
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

    /// Iterate over HDUs in declaration order. Materializes any
    /// pending slots up front so the iterator's snapshot is stable
    /// against concurrent edits.
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

    /// Append an HDU at the end. Accepts an HDU instance or a builder.
    fn append(&self, py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_writable()?;
        self.dirty.store(true, Ordering::Release);
        self.invalidate_bindings();
        let hdu = coerce_to_hdu(py, &value)?;
        self.lock_state().slots.push(HduSlot::Materialized(hdu));
        Ok(())
    }

    /// Insert an HDU at position ``i``. Accepts an HDU instance or a builder.
    fn insert(&self, py: Python<'_>, i: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_writable()?;
        self.dirty.store(true, Ordering::Release);
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
    ///     Value of the ``EXTNAME`` keyword to match.
    /// ver : int, optional
    ///     If given, also require matching ``EXTVER`` (default 1
    ///     when the keyword is absent).
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If no HDU matches.
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
    ///     HDU index. Default 0 (primary HDU).
    /// alt : str, optional
    ///     Single ASCII character. ``' '`` (default) selects the
    ///     primary WCS description.
    ///
    /// Notes
    /// -----
    /// Only the target HDU's header is consulted; ``-TAB`` axis
    /// tables stored in sibling HDUs are not currently resolved.
    #[pyo3(signature = (i=0, alt=' '))]
    fn wcs(&self, py: Python<'_>, i: usize, alt: char) -> PyResult<Option<PyWcs>> {
        let hdu = self.hdu(py, i)?;
        let bound = hdu.bind(py);
        let header: PyHeader = bound.getattr("header")?.extract()?;
        let wcs = crate::wcs::Wcs::from_header(&header.lock(), alt).into_py_result()?;
        Ok(wcs.map(PyWcs::from))
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
    ///     Destination path.
    /// overwrite : bool, optional
    ///     If False (default), raise :class:`FileExistsError` when
    ///     ``path`` already exists. Set to True to replace it.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the file contains zero HDUs, or if ``path`` resolves
    ///     to the source file and the handle is read-only.
    /// FileExistsError
    ///     If ``path`` exists and ``overwrite`` is False.
    /// FitsError
    ///     On I/O failure.
    #[pyo3(signature = (path, overwrite=false))]
    fn writeto(&self, py: Python<'_>, path: PathBuf, overwrite: bool) -> PyResult<()> {
        use pyo3::exceptions::PyFileExistsError;
        use std::fs::{File, OpenOptions};
        use std::io::BufWriter;
        // Writing back over our own backing file is allowed only via
        // `flush()`'s drop+rewrite+reopen path, because doing so
        // through a sibling-rename would unmap the live `inner` and
        // `updater` mappings out from under any held `hdu.section`
        // bindings. Detect the self-write case and dispatch
        // accordingly: with `overwrite=False` we honour the
        // FileExistsError contract; with `overwrite=True` we behave
        // exactly like `flush()` (rewrite, atomic rename, reopen).
        let writes_to_self = self
            .original_path
            .as_ref()
            .and_then(|orig| std::fs::canonicalize(&path).ok().map(|t| &t == orig))
            .unwrap_or(false);
        if writes_to_self {
            // A self-write is a mutation of the source file; only
            // permitted in update mode. Astropy parity: writing to
            // a *different* path from a readonly handle is allowed
            // and is the canonical "save edits to a copy" workflow.
            self.ensure_writable()?;
            if !overwrite {
                return Err(PyFileExistsError::new_err(format!(
                    "FitsFile.writeto: {} already exists; pass overwrite=True to replace",
                    path.display(),
                )));
            }
            // Force a full rewrite even if no edits are pending so
            // the on-disk bytes match what `materialize_all + encode`
            // would produce -- matches astropy semantics.
            self.dirty.store(true, Ordering::Release);
            return self.persist_full_rewrite(py);
        }
        if !overwrite && path.exists() {
            return Err(PyFileExistsError::new_err(format!(
                "FitsFile.writeto: {} already exists; pass overwrite=True to replace",
                path.display(),
            )));
        }
        // Sibling temp file with O_CREAT|O_EXCL + unpredictable
        // suffix. Avoids the predictable `<path>.fitsy-tmp` race
        // where an attacker (or a stale leftover) could pre-create
        // the path and have us either fail or follow a symlink.
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let basename = path.file_name().map_or_else(
            || std::ffi::OsString::from("fitsy-out"),
            std::ffi::OsStr::to_os_string,
        );
        let (tmp, tmp_file) = {
            let mut last_err: Option<std::io::Error> = None;
            let mut chosen: Option<(PathBuf, File)> = None;
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
                    Ok(f) => {
                        chosen = Some((candidate, f));
                        break;
                    }
                    Err(e) => last_err = Some(e),
                }
            }
            chosen.ok_or_else(|| {
                super::err_to_py(crate::error::FitsError::Io(last_err.unwrap_or_else(|| {
                    std::io::Error::other("FitsFile.writeto: could not create temp file")
                })))
            })?
        };
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
            let snapshot: Vec<WritetoSlot> = {
                let st = self.lock_state();
                st.slots
                    .iter()
                    .map(|s| match s {
                        HduSlot::Pending(i) => WritetoSlot::Pending(*i),
                        HduSlot::Materialized(p) => WritetoSlot::Materialized(p.clone_ref(py)),
                    })
                    .collect()
            };
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
            // Synthesise an empty primary if the first emitted HDU
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
                    let dtype = match img.bitpix {
                        crate::data::Bitpix::U8 => "uint8",
                        crate::data::Bitpix::I16 => "int16",
                        crate::data::Bitpix::I32 => "int32",
                        crate::data::Bitpix::I64 => "int64",
                        crate::data::Bitpix::F32 => "float32",
                        crate::data::Bitpix::F64 => "float64",
                    };
                    let details = {
                        // Use the cached `axes` snapshot so the
                        // info table never triggers a lazy data
                        // read (the eager pixel materialisation
                        // would defeat the lazy design).
                        let dims: Vec<String> = img.axes.iter().map(ToString::to_string).collect();
                        if dims.is_empty() {
                            dtype.to_string()
                        } else {
                            // Left-pad dtype to 7 chars so "uint8" aligns with "float32".
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
    /// fixed-size chunks (no full materialisation) and compares
    /// against the values stored in the HDU header. HDUs that have
    /// neither card are reported with both fields ``None``; HDUs
    /// that only have one of the two are reported with the missing
    /// field ``None`` and the present one as ``True`` / ``False``.
    ///
    /// Returns
    /// -------
    /// list[dict]
    ///     One dict per HDU, in file order. Keys:
    ///
    ///     * ``hdu`` -- 0-based HDU index (``int``).
    ///     * ``checksum_ok`` -- ``True``/``False``/``None``.
    ///     * ``datasum_ok`` -- ``True``/``False``/``None``.
    ///
    /// Notes
    /// -----
    /// Works on any file (read-only or update), and on in-memory
    /// `FitsFile` objects whose data is already resident.
    /// Astropy parity: equivalent to iterating
    /// ``[hdu.verify_checksum() for hdu in hdul]`` but without
    /// reading the data into memory.
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

    /// Enable ``CHECKSUM`` / ``DATASUM`` stamping on the next
    /// :meth:`writeto` or :meth:`flush`.
    ///
    /// When called, every HDU emitted by the next write will gain
    /// freshly computed ``CHECKSUM`` and ``DATASUM`` cards (per
    /// the FITS Checksum Proposal). Existing placeholder cards in
    /// each header are overwritten in place; missing ones are
    /// inserted. The flag stays on for the lifetime of the
    /// ``FitsFile`` object, matching astropy's semantics where
    /// ``hdu.add_checksum()`` permanently mutates the HDU.
    ///
    /// Notes
    /// -----
    /// This does not stamp anything immediately -- the actual
    /// computation happens during the next write, when the final
    /// byte layout of each HDU is known. To verify checksums on
    /// the resulting file, call :meth:`verify_checksums` after
    /// the write.
    ///
    /// Astropy parity: equivalent to calling
    /// ``hdu.add_checksum()`` on every HDU in the list. There is
    /// no per-HDU variant in fitsy because checksums must be
    /// computed against the final on-disk byte layout.
    fn add_checksums(&self) {
        self.stamp_checksums.store(true, Ordering::Relaxed);
    }

    /// Flush pending edits to disk.
    ///
    ///   ``f.append(...)``, ``del f[i]``, fancy / dtype-mismatched
    ///   patches, edits on a tile-compressed image), rewrites the
    ///   file via a sibling temp file + atomic ``rename``. Slots
    ///   the user never touched are streamed byte-for-byte from
    ///   the original file (no decode/re-encode).
    ///
    /// Mixing modes: if you issue an in-place ``section[...]``
    /// patch and then a non-patch mutation in the same session,
    /// the patch reaches disk first (via ``pwrite``) and the
    /// subsequent ``flush()`` then performs a full rewrite that
    /// includes the patched bytes by streaming the (already
    /// patched) source file. Patches are not lost.
    ///
    /// Crash safety: in-place patches use ``pwrite`` with no undo
    /// journal; a process death mid-patch can leave the file with
    /// some rows updated and others not (this matches astropy's
    /// mmap-backed update mode). The full-rewrite path is
    /// crash-safe because it writes to a sibling temp file and
    /// renames atomically once the bytes are durable. Note that
    /// the parent directory is not separately ``fsync``\ ed, so a
    /// power loss between rename and the next directory commit can
    /// theoretically leave the rename invisible after reboot on
    /// non-journaling filesystems. Stale ``.fitsy-tmp.*`` siblings
    /// from a crashed rewrite are harmless and may be deleted.
    ///
    /// A no-op for read-only files.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        if self.updater.is_none() {
            return Ok(());
        }
        if self.dirty.swap(false, Ordering::AcqRel) {
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
    /// After ``close()`` the slot list and any HDU wrappers Python
    /// already holds remain usable as in-memory data, but the
    /// underlying file handle is dropped so any ``Pending`` slot
    /// that has not yet been materialized will fail to load.
    /// Subsequent ``flush()`` / ``writeto()`` calls also raise.
    ///
    /// Idempotent: calling ``close()`` more than once is safe.
    /// Astropy parity: matches ``HDUList.close()``.
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

    /// Context-manager entry. Returns ``self``.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

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

/// True when the HDU is an image (or image builder) and can serve
/// as the primary HDU.
fn is_image_like(py: Python<'_>, hdu: &Py<PyAny>) -> bool {
    use crate::python::writer::PyImageBuilder;
    let b = hdu.bind(py);
    b.extract::<PyRef<'_, PyImageHdu>>().is_ok() || b.extract::<PyRef<'_, PyImageBuilder>>().is_ok()
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
    Err(PyTypeError::new_err(
        "FitsFile.writeto: HDU slot has unsupported type",
    ))
}

fn wrap_hdu(
    py: Python<'_>,
    i: usize,
    hdu: crate::Hdu<'_>,
    header: PyHeader,
    read_only: bool,
    updater: Option<&Arc<Mutex<FitsUpdater>>>,
    dirty_flag: Option<Arc<AtomicBool>>,
    file: Arc<FitsFile>,
) -> PyResult<Py<PyAny>> {
    use crate::Hdu;
    match hdu {
        Hdu::Image(img) => {
            let mut py_img = PyImageHdu::from_image(py, &img, header, read_only);
            // Attach the lazy-read source so `data` / `section`
            // can pread fresh bytes on demand. Skipping this
            // would force the only path to be eager `from_image`
            // materialisation (which we removed).
            py_img.read_binding = Some(super::hdu::ReadBinding {
                file,
                hdu_idx: i,
                axes: img.axes().to_vec(),
            });
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
            // image view (BITPIX/NAXISn rewritten from Z*). Mirrors
            // astropy's transparent CompImageHDU behaviour.
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
