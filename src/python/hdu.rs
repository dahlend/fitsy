//! `PyImageHdu` -- image HDU with lazy numpy data.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::ImageHdu;
use crate::data::Bitpix;

use super::IntoPyResult;
use super::header::PyHeader;
use super::wcs::PyWcs;

/// Per-HDU writable-file binding shared between [`PyImageHdu`] and
/// [`PyImageSection`]. Cloned cheaply (just two `Arc` bumps).
#[derive(Debug, Clone)]
pub(crate) struct UpdateBinding {
    pub(crate) updater: Arc<Mutex<crate::FitsUpdater>>,
    pub(crate) hdu_idx: usize,
    /// Generation counter snapshot at binding time. The fast
    /// in-place pwrite path checks this against the live updater
    /// generation before writing -- if the file has been rewritten
    /// (or the slot list has been mutated) the binding's
    /// `hdu_idx` may no longer point at the original HDU, so we
    /// refuse the patch and let the slow rewrite path take over.
    pub(crate) generation: u64,
}

/// Per-HDU read-from-disk binding: lets [`PyImageHdu`] and
/// [`PyImageSection`] pull pixel bytes from the parent
/// [`crate::FitsFile`] on demand without ever materialising the
/// whole data section. Cheap to clone (just an `Arc` bump).
#[derive(Debug, Clone)]
pub(crate) struct ReadBinding {
    /// Backing FITS file. Shared with the parent
    /// :class:`PyFitsFile`'s `state.file`. Holding a clone here
    /// keeps the source bytes reachable for as long as any
    /// materialised image HDU might want to lazy-load its data.
    pub(crate) file: Arc<crate::FitsFile>,
    /// Index of this HDU in `file`.
    pub(crate) hdu_idx: usize,
    /// Image axes in **FITS order** (`NAXIS1` fastest). Cached so
    /// section reads/writes don't have to reparse the header.
    pub(crate) axes: Vec<u64>,
}

/// Image HDU with lazy numpy data.
///
/// Returned by :meth:`FitsFile.hdu` (or ``file[i]``) when the HDU
/// kind is an image. The pixel data is **not** read at materialisation
/// time; the first access to ``hdu.data`` (or any operation that
/// needs the full array, e.g. :meth:`FitsFile.writeto`) reads it
/// from disk via positional ``pread``. Subsequent accesses return
/// the same array, and in-place mutation (``hdu.data[0, 0] = 42``)
/// is preserved on the next :meth:`FitsFile.writeto`.
///
/// For workloads that operate on rectangular sub-regions of an
/// image larger than RAM, use :attr:`section` -- ``hdu.section[a:b]``
/// reads only the requested bytes, and ``hdu.section[a:b] = arr``
/// writes only the touched bytes (via ``pwrite``) without
/// materialising the full array in memory.
///
/// Examples
/// --------
/// >>> with fitsy.open("image.fits") as f:
/// ...     img = f[0]
/// ...     print(img.bitpix, img.axes, img.data.shape)
#[pyclass(name = "ImageHdu", module = "fitsy")]
#[derive(Debug)]
pub struct PyImageHdu {
    pub(crate) header: PyHeader,
    pub(crate) bitpix: Bitpix,
    /// Image axes in **FITS order** (NAXIS1 fastest). Empty when
    /// ``NAXIS == 0``. Used to decide between "data not yet
    /// materialised" and "no data section to materialise".
    pub(crate) axes: Vec<u64>,
    /// Whether this HDU was opened in read-only mode. Materialised
    /// numpy arrays are frozen (``WRITEABLE`` flag cleared) when set.
    pub(crate) read_only: bool,
    /// The pixel array. Shared with [`PyImageSection`] via `Arc`
    /// so that section reads/writes observe each other's mutations.
    /// `None` means either:
    ///   - ``NAXIS == 0`` (then `axes` is empty), or
    ///   - the data has not yet been read from disk (then `axes` is
    ///     non-empty and `read_binding` is `Some`).
    pub(crate) data: Arc<Mutex<Option<Py<PyAny>>>>,
    /// Lazy-read source. `Some` whenever the HDU was materialised
    /// from a `FitsFile` (i.e. read from disk or from a byte buffer);
    /// `None` only for HDUs constructed in memory by the user
    /// (e.g. via :func:`fitsy.image`).
    pub(crate) read_binding: Option<ReadBinding>,
    /// In-place patch-write binding. Set only when the parent
    /// :class:`FitsFile` was opened with `mode='update'` AND the
    /// HDU is an uncompressed image. When present, `section[a:b] =
    /// arr` writes through the file via positional ``pwrite``.
    pub(crate) update_binding: Option<UpdateBinding>,
    /// Optional back-pointer to the parent `FitsFile`'s dirty
    /// flag. `Some` when the HDU was materialized from a file
    /// opened with `mode='update'`. Mutations that *cannot* be
    /// satisfied by a fast in-place pwrite patch (whole-array
    /// reassignment, fancy / negative-step `section` writes, edits
    /// on a compressed image, etc.) flip the bit so `flush()` /
    /// `__exit__` know to rewrite the file. Pure pixel-patch
    /// writes via ``pwrite`` leave the bit alone.
    pub(crate) dirty: Option<Arc<AtomicBool>>,
}

impl PyImageHdu {
    /// Construct a lazy `PyImageHdu` from a parsed `Hdu::Image`
    /// view. **Does not read pixel data.** The caller is expected
    /// to attach a `read_binding` immediately afterwards so future
    /// `data` accesses can lazy-load from disk.
    pub(crate) fn from_image(
        _py: Python<'_>,
        img: &ImageHdu<'_>,
        header: PyHeader,
        read_only: bool,
    ) -> Self {
        let axes: Vec<u64> = img.axes().to_vec();
        let bitpix = img.bitpix();
        Self {
            header,
            bitpix,
            axes,
            read_only,
            data: Arc::new(Mutex::new(None)),
            read_binding: None,
            update_binding: None,
            dirty: None,
        }
    }

    /// Lock and inspect the cached `data` slot. Returns a clone of
    /// the materialised array if any. Does **not** trigger a
    /// lazy load.
    fn data_if_loaded(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        let g = self.data.lock().unwrap_or_else(PoisonError::into_inner);
        g.as_ref().map(|a| a.clone_ref(py))
    }

    /// Set or clear the cached `data` slot.
    fn store_data(&self, value: Option<Py<PyAny>>) {
        let mut g = self.data.lock().unwrap_or_else(PoisonError::into_inner);
        *g = value;
    }

    /// Lazy data accessor. If the array has already been
    /// materialised, return a clone. Otherwise read it from the
    /// `read_binding`, decode into a numpy array (applying
    /// BSCALE/BZERO/BLANK and byteswapping to native order),
    /// cache it, and return a clone. Returns `None` when
    /// ``NAXIS == 0`` or any axis is zero (no data section).
    fn ensure_data(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if let Some(arr) = self.data_if_loaded(py) {
            return Ok(Some(arr));
        }
        if self.axes.is_empty() || self.axes.contains(&0) {
            return Ok(None);
        }
        let Some(binding) = self.read_binding.as_ref() else {
            // No source and no cached data. The HDU was
            // constructed in memory by the user with no data; treat
            // as empty rather than erroring.
            return Ok(None);
        };
        let bytes = binding
            .file
            .read_data_owned(binding.hdu_idx)
            .into_py_result()?;
        let user_header = self.header.lock().clone();
        let img = ImageHdu::new(user_header, &bytes).into_py_result()?;
        let arr = read_pixels(py, &img, self.bitpix, &self.axes)?;
        if self.read_only {
            freeze_array(py, &arr)?;
        }
        let cloned = arr.clone_ref(py);
        self.store_data(Some(arr));
        Ok(Some(cloned))
    }

    /// Reconstruct from a builder snapshot (header + raw bytes).
    /// Decodes the bytes back into a numpy array using the header's
    /// `BITPIX`/`NAXIS*` so the appended HDU is fully editable.
    pub(crate) fn from_built_bytes(
        py: Python<'_>,
        header: crate::Header,
        bytes: Vec<u8>,
        read_only: bool,
    ) -> PyResult<Self> {
        use crate::Value;
        let bitpix_i = header
            .first("BITPIX")
            .and_then(|v| match v {
                Value::Integer(i) => Some(*i),
                _ => None,
            })
            .unwrap_or(8);
        let bitpix = Bitpix::from_i64(bitpix_i).map_err(super::err_to_py)?;
        let naxis: i64 = header
            .first("NAXIS")
            .and_then(|v| match v {
                Value::Integer(i) => Some(*i),
                _ => None,
            })
            .unwrap_or(0);
        let mut axes: Vec<u64> = Vec::with_capacity(naxis.max(0) as usize);
        for k in 1..=naxis {
            let key = format!("NAXIS{k}");
            let n = header
                .first(&key)
                .and_then(|v| match v {
                    Value::Integer(i) => Some(*i),
                    _ => None,
                })
                .unwrap_or(0);
            axes.push(n.max(0) as u64);
        }
        let header = PyHeader::from_header_with(&header, read_only);
        let data = if axes.is_empty() || axes.contains(&0) {
            None
        } else {
            let shape: Vec<usize> = axes.iter().rev().map(|&n| n as usize).collect();
            let arr = decode_be_to_array(py, bitpix, &bytes, &shape);
            if read_only {
                freeze_array(py, &arr)?;
            }
            Some(arr)
        };
        Ok(Self {
            header,
            bitpix,
            axes,
            read_only,
            data: Arc::new(Mutex::new(data)),
            read_binding: None,
            update_binding: None,
            dirty: None,
        })
    }

    /// Encode this HDU's current state into header + data bytes
    /// for writing. Re-stamps BITPIX/NAXIS from the live array.
    /// Lazy: triggers a `data` materialisation if the user never
    /// touched it, so the encoded bytes reflect on-disk reality.
    pub(crate) fn encode(
        &self,
        py: Python<'_>,
        is_primary: bool,
    ) -> PyResult<(crate::Header, Vec<u8>)> {
        use crate::python::writer::build_image;
        let user_header = self.header.lock().clone();
        match self.ensure_data(py)? {
            Some(arr) => build_image(py, arr.bind(py), is_primary, user_header),
            None => Ok((empty_image_header(is_primary, user_header), Vec::new())),
        }
    }

    /// Re-stamp `BITPIX` + `NAXIS*` cards in the header to match the
    /// current pixel array. Removes any leftover higher-axis cards.
    fn restamp_layout(&self, axes: &[u64]) {
        use crate::Value;
        let mut h = self.header.lock();
        let _ = h.set("BITPIX", Value::Integer(self.bitpix.as_i64()), None);
        let _ = h.set("NAXIS", Value::Integer(axes.len() as i64), None);
        for (i, n) in axes.iter().enumerate() {
            let key = format!("NAXIS{}", i + 1);
            let _ = h.set(&key, Value::Integer(*n as i64), None);
        }
        // Drop trailing NAXISk cards from any prior larger array.
        let mut k = axes.len() + 1;
        loop {
            let key = format!("NAXIS{k}");
            if h.first(&key).is_some() {
                h.remove(&key);
                k += 1;
            } else {
                break;
            }
        }
    }
}

/// Mark a numpy array as read-only by clearing its `WRITEABLE` flag.
fn freeze_array(py: Python<'_>, arr: &Py<PyAny>) -> PyResult<()> {
    arr.bind(py).call_method1("setflags", ((), false))?;
    Ok(())
}

fn bitpix_numpy_dtype(b: Bitpix) -> &'static str {
    match b {
        Bitpix::U8 => "uint8",
        Bitpix::I16 => "int16",
        Bitpix::I32 => "int32",
        Bitpix::I64 => "int64",
        Bitpix::F32 => "float32",
        Bitpix::F64 => "float64",
    }
}

/// Decode big-endian raw pixel bytes into a numpy array.
fn decode_be_to_array(py: Python<'_>, bitpix: Bitpix, bytes: &[u8], shape: &[usize]) -> Py<PyAny> {
    fn dec<T: crate::data::Pixel>(bytes: &[u8]) -> Vec<T> {
        bytes
            .chunks_exact(size_of::<T>())
            .map(T::from_be_bytes)
            .collect()
    }
    match bitpix {
        Bitpix::U8 => to_array(py, dec::<u8>(bytes), shape),
        Bitpix::I16 => to_array(py, dec::<i16>(bytes), shape),
        Bitpix::I32 => to_array(py, dec::<i32>(bytes), shape),
        Bitpix::I64 => to_array(py, dec::<i64>(bytes), shape),
        Bitpix::F32 => to_array(py, dec::<f32>(bytes), shape),
        Bitpix::F64 => to_array(py, dec::<f64>(bytes), shape),
    }
}

/// Build a NAXIS=0 image header (no data section).
fn empty_image_header(is_primary: bool, user: crate::Header) -> crate::Header {
    use crate::Value;
    let mut h = crate::Header::empty();
    if is_primary {
        let _ = h.set("SIMPLE", Value::Logical(true), Some("conforming FITS"));
        let _ = h.set("BITPIX", Value::Integer(8), None);
        let _ = h.set("NAXIS", Value::Integer(0), None);
        let _ = h.set("EXTEND", Value::Logical(true), None);
    } else {
        let _ = h.set("XTENSION", Value::String("IMAGE".into()), None);
        let _ = h.set("BITPIX", Value::Integer(8), None);
        let _ = h.set("NAXIS", Value::Integer(0), None);
        let _ = h.set("PCOUNT", Value::Integer(0), None);
        let _ = h.set("GCOUNT", Value::Integer(1), None);
    }
    for entry in user.entries() {
        let kw = entry.keyword.to_ascii_uppercase();
        if matches!(
            kw.as_str(),
            "SIMPLE" | "BITPIX" | "NAXIS" | "EXTEND" | "PCOUNT" | "GCOUNT" | "XTENSION" | "END"
        ) {
            continue;
        }
        if kw.starts_with("NAXIS") && kw[5..].chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if let Some(v) = entry.value.as_ref() {
            let _ = h.set(&entry.keyword, v.clone(), entry.comment.as_deref());
        }
    }
    h
}

fn read_pixels(
    py: Python<'_>,
    img: &ImageHdu<'_>,
    bitpix: Bitpix,
    axes: &[u64],
) -> PyResult<Py<PyAny>> {
    // numpy expects row-major shape (slowest first); FITS NAXISn
    // is fastest-first. Reverse.
    let shape: Vec<usize> = axes.iter().rev().map(|&n| n as usize).collect();
    let header = img.header();
    let bzero = header.bzero();
    let bscale = header.bscale();
    let blank = header.blank();
    let identity = bzero == 0.0 && bscale == 1.0 && blank.is_none();
    if identity {
        return read_raw_to_array(py, img, bitpix, &shape);
    }
    // Special unsigned/signed integer reinterpretations (BSCALE=1,
    // BZERO=2^(N-1) or -2^(N-1)). astropy returns the corresponding
    // unsigned (or int8) dtype rather than promoting to float.
    if bscale == 1.0 && blank.is_none() {
        match bitpix {
            Bitpix::I16 if (bzero - 32_768.0).abs() < f64::EPSILON => {
                let raw = img.read_raw::<i16>().into_py_result()?.into_vec();
                let conv: Vec<u16> = raw
                    .into_iter()
                    .map(|x| (i32::from(x) + 32_768) as u16)
                    .collect();
                return Ok(to_array(py, conv, &shape));
            }
            Bitpix::I32 if (bzero - 2_147_483_648.0).abs() < 1.0 => {
                let raw = img.read_raw::<i32>().into_py_result()?.into_vec();
                let conv: Vec<u32> = raw
                    .into_iter()
                    .map(|x| (i64::from(x) + 2_147_483_648) as u32)
                    .collect();
                return Ok(to_array(py, conv, &shape));
            }
            Bitpix::I64 if (bzero - 9_223_372_036_854_775_808.0).abs() < 4096.0 => {
                let raw = img.read_raw::<i64>().into_py_result()?.into_vec();
                let conv: Vec<u64> = raw
                    .into_iter()
                    .map(|x| (x as u64).wrapping_add(0x8000_0000_0000_0000))
                    .collect();
                return Ok(to_array(py, conv, &shape));
            }
            Bitpix::U8 if (bzero + 128.0).abs() < f64::EPSILON => {
                let raw = img.read_raw::<u8>().into_py_result()?.into_vec();
                let conv: Vec<i8> = raw
                    .into_iter()
                    .map(|x| (i16::from(x) - 128) as i8)
                    .collect();
                return Ok(to_array(py, conv, &shape));
            }
            _ => {}
        }
    }
    // General case: apply BSCALE/BZERO/BLANK and return floats.
    if matches!(bitpix, Bitpix::F32) {
        let arr = img.read_physical_f32().into_py_result()?.into_vec();
        Ok(to_array(py, arr, &shape))
    } else {
        let arr = img.read_physical().into_py_result()?.into_vec();
        Ok(to_array(py, arr, &shape))
    }
}

fn read_raw_to_array(
    py: Python<'_>,
    img: &ImageHdu<'_>,
    bitpix: Bitpix,
    shape: &[usize],
) -> PyResult<Py<PyAny>> {
    Ok(match bitpix {
        Bitpix::U8 => to_array(py, img.read_raw::<u8>().into_py_result()?.into_vec(), shape),
        Bitpix::I16 => to_array(
            py,
            img.read_raw::<i16>().into_py_result()?.into_vec(),
            shape,
        ),
        Bitpix::I32 => to_array(
            py,
            img.read_raw::<i32>().into_py_result()?.into_vec(),
            shape,
        ),
        Bitpix::I64 => to_array(
            py,
            img.read_raw::<i64>().into_py_result()?.into_vec(),
            shape,
        ),
        Bitpix::F32 => to_array(
            py,
            img.read_raw::<f32>().into_py_result()?.into_vec(),
            shape,
        ),
        Bitpix::F64 => to_array(
            py,
            img.read_raw::<f64>().into_py_result()?.into_vec(),
            shape,
        ),
    })
}

#[pymethods]
impl PyImageHdu {
    /// Construct a new image HDU from a numpy array.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     Pixel data. Dtype must be one of ``bool``, ``int8``,
    ///     ``uint8``, ``int16``, ``uint16``, ``int32``, ``uint32``,
    ///     ``int64``, ``uint64``, ``float32``, ``float64``. Unsigned
    ///     integers (and ``int8``) are encoded with the standard
    ///     ``BZERO`` convention so they round-trip via the reader.
    ///     The array is stored by reference; in-place mutation later\n    ///     is preserved on :meth:`FitsFile.writeto`.
    /// header : Header or Mapping[str, Any], optional
    ///     Initial header. Layout cards (``BITPIX``, ``NAXIS*``)\n    ///     are recomputed from the array on write.
    /// name : str, optional
    ///     Convenience: sets the ``EXTNAME`` card.
    #[new]
    #[pyo3(signature = (data, header=None, name=None))]
    fn py_new(
        py: Python<'_>,
        data: Py<PyAny>,
        header: Option<Py<PyAny>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let bitpix = bitpix_from_array(py, &data)?;
        let header = build_header(py, header, name)?;
        let shape: Vec<usize> = data.bind(py).getattr("shape")?.extract()?;
        let axes: Vec<u64> = shape.into_iter().rev().map(|n| n as u64).collect();
        Ok(Self {
            header,
            bitpix,
            axes,
            read_only: false,
            data: Arc::new(Mutex::new(Some(data))),
            read_binding: None,
            update_binding: None,
            dirty: None,
        })
    }

    /// The HDU header (see :class:`Header`).
    #[getter]
    fn header(&self) -> PyHeader {
        self.header.clone()
    }

    /// Image axes in **NAXIS order**: ``[NAXIS1, NAXIS2, ...]``.
    ///
    /// When the pixel data has been materialised, the axes are
    /// reported from the live numpy array shape (reversed, since
    /// numpy is row-major while FITS lists fastest-varying first).
    /// Otherwise the axes recorded at HDU-open time are returned --
    /// this is the lazy path that does **not** trigger a data read.
    #[getter]
    fn axes(&self, py: Python<'_>) -> PyResult<Vec<u64>> {
        if let Some(arr) = self.data_if_loaded(py) {
            let shape: Vec<usize> = arr.bind(py).getattr("shape")?.extract()?;
            return Ok(shape.into_iter().rev().map(|n| n as u64).collect());
        }
        Ok(self.axes.clone())
    }

    /// FITS ``BITPIX`` value (e.g. ``-32`` for ``f32``).
    #[getter]
    fn bitpix(&self) -> i64 {
        self.bitpix.as_i64()
    }

    /// Pixel data as a numpy array.
    ///
    /// Materialises the array on first access by reading the data
    /// section from disk, byteswapping into native order, and
    /// applying ``BSCALE``/``BZERO``/``BLANK`` scaling. Subsequent
    /// accesses return the same array, and in-place mutation
    /// (``hdu.data[...] = x``) is preserved by the next
    /// :meth:`FitsFile.writeto`.
    ///
    /// For images that do not fit in RAM, prefer :attr:`section`
    /// -- ``hdu.section[a:b]`` reads only the requested bytes
    /// without materialising the full array.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray or None
    ///     ``None`` when the HDU has no data section
    ///     (``NAXIS == 0``).
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.ensure_data(py)?.unwrap_or_else(|| py.None()))
    }

    /// Slicing accessor that mirrors :class:`numpy.ndarray`
    /// indexing. ``hdu.section[a:b, c:d]`` reads only the
    /// requested region from disk -- no full-image materialisation.
    ///
    /// In ``mode='update'``, ``hdu.section[a:b] = arr`` writes only
    /// the touched bytes back via positional ``pwrite``, again
    /// without materialising the full image. This is the supported
    /// way to read or patch sub-regions of an image bigger than
    /// available RAM.
    ///
    /// In-place writes require contiguous slicing (``start:stop``
    /// with step 1) on an HDU with identity scaling
    /// (``BSCALE=1``, ``BZERO=0``, no ``BLANK``). Anything else
    /// (fancy indexing, negative steps, scaled HDUs) raises a
    /// ``ValueError`` -- assign through ``hdu.data[...]`` to
    /// trigger a full-file rewrite instead.
    ///
    /// If ``hdu.data`` has already been accessed (and is therefore
    /// resident in memory), reads and writes go through the
    /// in-memory array for consistency with subsequent
    /// ``hdu.data`` accesses.
    ///
    /// Returns
    /// -------
    /// _ImageSection
    ///     Slicing proxy. Use ``section[i, j, k]`` exactly like
    ///     ``data[i, j, k]``.
    #[getter]
    fn section(slf: PyRef<'_, Self>) -> PyImageSection {
        PyImageSection {
            bitpix: slf.bitpix,
            axes: slf.axes.clone(),
            read_only: slf.read_only,
            header: slf.header.clone(),
            data: slf.data.clone(),
            read_binding: slf.read_binding.clone(),
            update_binding: slf.update_binding.clone(),
            dirty: slf.dirty.clone(),
        }
    }

    /// Replace the pixel array. Dtype must be a supported FITS
    /// type. The header's ``BITPIX`` and ``NAXIS*`` cards are
    /// updated immediately to match the new array.
    #[setter]
    fn set_data(&mut self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        if let Some(flag) = &self.dirty {
            flag.store(true, Ordering::Release);
        }
        // Reassigning `data` changes BITPIX/NAXISn, so the cached
        // writable-file binding (which encodes the on-disk byte
        // offsets and pixel layout from before) no longer matches
        // the in-memory image. Drop it; future patches go through
        // the slow rewrite path (which re-encodes everything from
        // the current cache).
        self.update_binding = None;
        // Reassigning `data` invalidates the lazy-read source: the
        // bytes on disk no longer correspond to the in-memory
        // image. Drop the read binding so subsequent accesses
        // never silently re-read stale on-disk bytes.
        self.read_binding = None;
        if value.is_none(py) {
            self.store_data(None);
            self.axes.clear();
            self.restamp_layout(&[]);
            return Ok(());
        }
        self.bitpix = bitpix_from_array(py, &value)?;
        let shape: Vec<usize> = value.bind(py).getattr("shape")?.extract()?;
        let axes: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
        self.axes.clone_from(&axes);
        self.store_data(Some(value));
        self.restamp_layout(&axes);
        Ok(())
    }

    /// Resolve the WCS for this HDU.
    ///
    /// Parameters
    /// ----------
    /// alt : str, optional
    ///     Single ASCII character. ``' '`` (default) selects the
    ///     primary description; ``'A'`` through ``'Z'`` select
    ///     alternate descriptions.
    ///
    /// Returns
    /// -------
    /// Wcs or None
    ///     ``None`` if the header carries no WCS for ``alt``.
    ///
    /// Notes
    /// -----
    /// Only this HDU's header is consulted; ``-TAB`` axis tables
    /// stored in sibling HDUs are not resolved.
    #[pyo3(signature = (alt=' '))]
    fn wcs(&self, alt: char) -> PyResult<Option<PyWcs>> {
        let header = self.header.lock();
        let wcs = crate::wcs::Wcs::from_header(&header, alt).into_py_result()?;
        Ok(wcs.map(PyWcs::from))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let axes = self.axes(py).unwrap_or_default();
        let dtype = bitpix_numpy_dtype(self.bitpix);
        // Render axes as ``(N1, N2, ...)`` matching numpy's
        // ``.shape`` so users immediately recognize the layout.
        let shape = if axes.is_empty() {
            "()".to_string()
        } else {
            let mut s = String::from("(");
            for (i, a) in axes.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&a.to_string());
            }
            s.push(')');
            s
        };
        let header = self.header.lock();
        let name = header
            .entries()
            .iter()
            .find(|e| e.keyword == "EXTNAME")
            .and_then(|e| e.value.as_ref())
            .and_then(|v| match v {
                crate::header::Value::String(s) => Some(s.trim().to_string()),
                _ => None,
            })
            .filter(|s| !s.is_empty());
        match name {
            Some(n) => format!("ImageHdu(name={n:?}, dtype={dtype:?}, shape={shape})"),
            None => format!("ImageHdu(dtype={dtype:?}, shape={shape})"),
        }
    }
}

/// Infer `BITPIX` from a numpy array's dtype.
fn bitpix_from_array(py: Python<'_>, arr: &Py<PyAny>) -> PyResult<Bitpix> {
    let bound = arr.bind(py);
    let dtype = bound.getattr("dtype")?;
    let kind: char = dtype
        .getattr("kind")?
        .extract::<String>()?
        .chars()
        .next()
        .unwrap_or('?');
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    Ok(match (kind, itemsize) {
        // 'b' (numpy bool) and 1-byte int/uint all map to BITPIX=8.
        // BZERO offsets map signed/unsigned to the same BITPIX storage.
        ('b', _) | ('u' | 'i', 1) => Bitpix::U8,
        ('i' | 'u', 2) => Bitpix::I16,
        ('i' | 'u', 4) => Bitpix::I32,
        ('i' | 'u', 8) => Bitpix::I64,
        ('f', 4) => Bitpix::F32,
        ('f', 8) => Bitpix::F64,
        _ => {
            let name: String = dtype.str()?.extract()?;
            return Err(PyTypeError::new_err(format!(
                "ImageHdu: unsupported numpy dtype {name:?}; expected one of \
                 bool, int8, uint8, int16, uint16, int32, uint32, \
                 int64, uint64, float32, float64"
            )));
        }
    })
}

/// Build a `PyHeader` from an optional Python header (Header or
/// Mapping) plus an optional EXTNAME shortcut.
fn build_header(
    py: Python<'_>,
    src: Option<Py<PyAny>>,
    name: Option<String>,
) -> PyResult<PyHeader> {
    let header = if let Some(obj) = src {
        let bound = obj.bind(py);
        if let Ok(h) = bound.extract::<PyHeader>() {
            // Deep-clone so the new HDU owns an independent header.
            // Reusing the Arc would alias edits back into the source
            // HDU and could launder a read-only header into a
            // writable one, sidestepping the mode='readonly' guard.
            PyHeader::from_header_with(&h.lock(), false)
        } else {
            let new = PyHeader::empty();
            new.update_from(bound)?;
            new
        }
    } else {
        PyHeader::empty()
    };
    if let Some(n) = name {
        header
            .lock()
            .set("EXTNAME", crate::Value::String(n), None)
            .map_err(super::err_to_py)?;
    }
    Ok(header)
}

fn to_array<T>(py: Python<'_>, data: Vec<T>, shape: &[usize]) -> Py<PyAny>
where
    T: numpy::Element,
{
    let arr = data.into_pyarray(py);
    arr.reshape(shape.to_vec())
        .expect(
            "internal invariant: read_raw element count must equal NAXIS product; \
             this is a fitsy bug, please report",
        )
        .into_any()
        .unbind()
}

// =====================================================================
// Random Groups HDU (legacy radio-interferometry format).
// =====================================================================

use crate::hdu::random_groups::RandomGroupsHdu;

/// Random-groups primary HDU (legacy format; see Standard Sec.6).
///
/// Read-only Python view: groups are decoded on demand through
/// :meth:`group`. The HDU does **not** participate in
/// :meth:`FitsFile.writeto`; round-tripping random-groups files
/// through Python is intentionally not supported (use the Rust API).
#[pyclass(name = "RandomGroups", module = "fitsy")]
#[derive(Debug)]
pub struct PyRandomGroups {
    pub(crate) header: PyHeader,
    pub(crate) bitpix: Bitpix,
    pub(crate) n_groups: u64,
    pub(crate) n_params: u64,
    pub(crate) data_per_group: u64,
    /// Owned data section (big-endian).
    pub(crate) data: Vec<u8>,
}

impl PyRandomGroups {
    pub(crate) fn from_hdu(rg: &RandomGroupsHdu<'_>, header: PyHeader) -> Self {
        Self {
            header,
            bitpix: rg.bitpix(),
            n_groups: rg.n_groups(),
            n_params: rg.pcount(),
            data_per_group: rg.data_per_group(),
            data: rg.raw_bytes().to_vec(),
        }
    }
}

#[pymethods]
impl PyRandomGroups {
    /// HDU header.
    #[getter]
    fn header(&self) -> PyHeader {
        self.header.clone()
    }

    /// `BITPIX` value.
    #[getter]
    fn bitpix(&self) -> i64 {
        self.bitpix.as_i64()
    }

    /// Number of groups (`GCOUNT`).
    #[getter]
    fn n_groups(&self) -> u64 {
        self.n_groups
    }

    /// Number of parameters per group (`PCOUNT`).
    #[getter]
    fn n_params(&self) -> u64 {
        self.n_params
    }

    /// Number of data values per group (`prod(NAXIS2..NAXISn)`).
    #[getter]
    fn data_per_group(&self) -> u64 {
        self.data_per_group
    }

    /// Decode group `i` (0-based) as `(parameters, data)` numpy
    /// arrays. Both arrays use the HDU's BITPIX dtype; `BSCALE`,
    /// `BZERO`, `PSCALn`, `PZEROn` are **not** applied.
    fn group(&self, py: Python<'_>, i: u64) -> PyResult<Py<pyo3::types::PyTuple>> {
        if i >= self.n_groups {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "group {i} out of range (n_groups = {})",
                self.n_groups
            )));
        }
        let bsize = self.bitpix.byte_size();
        let group_elements = (self.n_params + self.data_per_group) as usize;
        let group_bytes = group_elements * bsize;
        let off = (i as usize) * group_bytes;
        let slab = &self.data[off..off + group_bytes];
        let p_bytes = (self.n_params as usize) * bsize;
        let (params_be, data_be) = slab.split_at(p_bytes);
        let p_shape = vec![self.n_params as usize];
        let d_shape = vec![self.data_per_group as usize];
        let params = decode_be_to_array(py, self.bitpix, params_be, &p_shape);
        let data = decode_be_to_array(py, self.bitpix, data_be, &d_shape);
        let tup = pyo3::types::PyTuple::new(py, [params, data])?;
        Ok(tup.unbind())
    }

    fn __len__(&self) -> usize {
        self.n_groups as usize
    }

    fn __repr__(&self) -> String {
        format!(
            "RandomGroups(n_groups={}, n_params={}, data_per_group={}, bitpix={})",
            self.n_groups,
            self.n_params,
            self.data_per_group,
            self.bitpix.as_i64()
        )
    }
}

// =====================================================================
// PySection -- slicing proxy for ImageHdu.section
// =====================================================================

/// Slicing proxy returned by :attr:`ImageHdu.section`.
///
/// Routes ``__getitem__`` and ``__setitem__`` through the parent
/// HDU's lazy state:
///
/// * If ``hdu.data`` has already been materialised, slicing and
///   patching go through the in-memory numpy array (with patches
///   *also* mirrored to disk via ``pwrite`` in update mode).
/// * Otherwise, ``section[a:b]`` reads only the requested bytes
///   from disk via ``pread``, and ``section[a:b] = arr`` writes
///   only the touched bytes via ``pwrite`` -- the full image is
///   never resident in memory.
#[pyclass(name = "_ImageSection", module = "fitsy")]
#[derive(Debug)]
pub struct PyImageSection {
    /// Pixel encoding -- needed by `__setitem__` to convert the user's
    /// numpy array to big-endian bytes for the patch write, and by
    /// `__getitem__` to decode pread'd bytes back to native order.
    pub(crate) bitpix: Bitpix,
    /// Image axes in **FITS order** (NAXIS1 fastest). Empty when
    /// ``NAXIS == 0``.
    pub(crate) axes: Vec<u64>,
    /// Whether to freeze freshly materialised arrays.
    pub(crate) read_only: bool,
    /// Snapshot of the parent HDU's header (cheap clone of the
    /// shared `Arc<Mutex<Header>>`). Needed for lazy reads so we
    /// can apply BSCALE/BZERO/BLANK scaling without holding a
    /// back-pointer to the parent HDU.
    pub(crate) header: PyHeader,
    /// Shared cache of the materialised pixel array (same `Arc` as
    /// the parent :class:`PyImageHdu`'s `data`). Lets section
    /// reads/writes observe and update the parent's view.
    pub(crate) data: Arc<Mutex<Option<Py<PyAny>>>>,
    /// Lazy-read source. `Some` when the parent HDU was opened
    /// from disk (or a byte buffer), enabling region-only reads.
    pub(crate) read_binding: Option<ReadBinding>,
    /// `Some` only when the parent file was opened with
    /// `mode='update'`. When present, `section[a:b] = arr` performs
    /// an O(patch) in-place write to the on-disk file.
    pub(crate) update_binding: Option<UpdateBinding>,
    /// Optional back-pointer to the parent `FitsFile`'s dirty
    /// flag. `Some` when the parent was opened with
    /// `mode='update'`. Patches that cannot take the fast in-place
    /// pwrite path (compressed image, fancy index, dtype mismatch)
    /// fall back to mutating the cached numpy array and flip this
    /// bit so `flush()` rewrites the file.
    pub(crate) dirty: Option<Arc<AtomicBool>>,
}

impl PyImageSection {
    fn data_if_loaded(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        let g = self.data.lock().unwrap_or_else(PoisonError::into_inner);
        g.as_ref().map(|a| a.clone_ref(py))
    }

    fn store_data(&self, value: Py<PyAny>) {
        let mut g = self.data.lock().unwrap_or_else(PoisonError::into_inner);
        *g = Some(value);
    }

    /// Materialise the full pixel array (same as `PyImageHdu::ensure_data`)
    /// and cache it in the shared `data` slot.
    fn ensure_data(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if let Some(arr) = self.data_if_loaded(py) {
            return Ok(Some(arr));
        }
        if !self.has_data() {
            return Ok(None);
        }
        let Some(binding) = self.read_binding.as_ref() else {
            return Ok(None);
        };
        let bytes = binding
            .file
            .read_data_owned(binding.hdu_idx)
            .into_py_result()?;
        let user_header = self.header.lock().clone();
        let img = ImageHdu::new(user_header, &bytes).into_py_result()?;
        let arr = read_pixels(py, &img, self.bitpix, &self.axes)?;
        if self.read_only {
            freeze_array(py, &arr)?;
        }
        let cloned = arr.clone_ref(py);
        self.store_data(arr);
        Ok(Some(cloned))
    }

    /// Indicate whether this section has an active data section.
    fn has_data(&self) -> bool {
        !self.axes.is_empty() && !self.axes.contains(&0)
    }
}

#[pymethods]
impl PyImageSection {
    fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.has_data() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "section: HDU has no data section (NAXIS == 0)",
            ));
        }
        // Fast path: data already loaded -> just slice it.
        if let Some(arr) = self.data_if_loaded(py) {
            return Ok(arr.bind(py).get_item(&key)?.unbind());
        }
        // Lazy path: try to read only the requested region.
        if let Some(binding) = self.read_binding.as_ref() {
            // Region read returns raw big-endian bytes; we only
            // know how to decode them in the identity-scaling case
            // (BSCALE=1, BZERO=0, BLANK absent). For non-identity
            // scaling, fall through to the materialise-then-slice
            // path so `read_pixels` applies the conversion.
            let scaling_identity = {
                let h = self.header.lock();
                h.bzero() == 0.0 && h.bscale() == 1.0 && h.blank().is_none()
            };
            if scaling_identity {
                let np_shape: Vec<usize> = self.axes.iter().rev().map(|&n| n as usize).collect();
                if let Some((np_start, np_region_shape, squeeze)) =
                    parse_region_key(&key, &np_shape)?
                {
                    let fits_start: Vec<u64> = np_start.iter().rev().map(|&n| n as u64).collect();
                    let fits_shape: Vec<u64> =
                        np_region_shape.iter().rev().map(|&n| n as u64).collect();
                    let bytes = binding
                        .file
                        .read_image_subarray_be(
                            binding.hdu_idx,
                            &binding.axes,
                            &fits_start,
                            &fits_shape,
                            self.bitpix,
                        )
                        .into_py_result()?;
                    let arr = decode_be_to_array(py, self.bitpix, &bytes, &np_region_shape);
                    // Apply numpy integer-index squeeze semantics:
                    // axes selected with a plain integer collapse.
                    let final_shape: Vec<usize> = np_region_shape
                        .iter()
                        .zip(squeeze.iter())
                        .filter_map(|(n, sq)| if *sq { None } else { Some(*n) })
                        .collect();
                    if final_shape.len() != np_region_shape.len() {
                        let reshaped = arr.bind(py).call_method1("reshape", (final_shape,))?;
                        return Ok(reshaped.unbind());
                    }
                    return Ok(arr);
                }
            }
        }
        // Fallback: materialise the whole array, then slice. Used
        // when the key is something `parse_region_key` doesn't
        // understand (fancy indexing, negative steps, ...).
        let Some(arr) = self.ensure_data(py)? else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "section: HDU has no data section (NAXIS == 0)",
            ));
        };
        Ok(arr.bind(py).get_item(&key)?.unbind())
    }

    /// Assign a patch into the image.
    ///
    /// Mirrors astropy's ``hdu.section[...] = value``. The fast
    /// path -- triggered when the file was opened with
    /// ``mode='update'`` and the key is a tuple of
    /// ``slice(start, stop)`` (step 1) and non-negative integers --
    /// writes only the affected pixel bytes through the file via
    /// positional ``pwrite`` (O(patch), no full-image rewrite). If
    /// the data array has already been materialised in memory the
    /// patch is also mirrored into it for consistency.
    ///
    /// All other writes (compressed images, fancy indexing,
    /// negative steps, dtype casts that change the underlying byte
    /// representation) fall back to mutating the cached numpy
    /// array and flag the file as dirty; the next
    /// :meth:`FitsFile.flush` (or clean ``__exit__``) rewrites the
    /// file via a sibling temp file + atomic rename.
    fn __setitem__(
        &mut self,
        py: Python<'_>,
        key: Bound<'_, PyAny>,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if !self.has_data() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "section: HDU has no data section (NAXIS == 0)",
            ));
        }
        // Try the fast in-place pwrite path: only available when
        // the parent file is in update mode AND the key describes
        // a contiguous rectangular region AND the HDU has identity
        // scaling (BSCALE=1, BZERO=0, no BLANK). When the parent
        // is in update mode but the patch shape is incompatible we
        // raise rather than silently falling back to the cache +
        // O(file) rewrite path -- the silent fallback would be a
        // performance trap (see CHANGELOG / docs).
        if let Some(binding) = self.update_binding.clone() {
            let np_shape: Vec<usize> = self.axes.iter().rev().map(|&n| n as usize).collect();
            let scaling_identity = {
                let h = self.header.lock();
                h.bzero() == 0.0 && h.bscale() == 1.0 && h.blank().is_none()
            };
            if !scaling_identity {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "section[...] = value: cannot patch in place because this HDU has \
                     non-identity scaling (BSCALE != 1, BZERO != 0, or BLANK is set). \
                     Assign through `hdu.data[...] = value` to materialize the array, \
                     apply the scaled write in memory, and persist via the next \
                     `flush()` (which rewrites the file).",
                ));
            }
            let parsed = parse_region_key(&key, &np_shape)?;
            let Some((np_start, np_region_shape, _squeeze)) = parsed else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "section[...] = value: this indexing pattern (fancy indexing, \
                     negative steps, an `Ellipsis` or boolean mask, ...) is not \
                     supported by the in-place patch path used by `mode='update'`. \
                     Either narrow the key to contiguous slices `start:stop` (step 1) \
                     and non-negative integers, or assign through `hdu.data[...]` to \
                     opt into a full-file rewrite on the next `flush()`.",
                ));
            };
            let fits_start: Vec<u64> = np_start.iter().rev().map(|&n| n as u64).collect();
            let fits_shape: Vec<u64> = np_region_shape.iter().rev().map(|&n| n as u64).collect();
            // The empty-region case (a slice yielding zero pixels)
            // is a no-op for a contiguous patch; bail out cleanly.
            if fits_start.is_empty() || fits_shape.contains(&0) {
                return Ok(());
            }
            let np = py.import("numpy")?;
            let dtype_str = bitpix_numpy_dtype(self.bitpix);
            let target = np.call_method1("empty", (np_region_shape.clone(), dtype_str))?;
            target.set_item((), &value)?;
            let bytes_value = target.call_method0("tobytes")?;
            let raw: Vec<u8> = bytes_value.extract()?;
            let bsize = self.bitpix.byte_size();
            let expected_elems: u64 = fits_shape.iter().product();
            if raw.len() as u64 != expected_elems * bsize as u64 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "section[...] = value: encoded value has {} bytes but the \
                     selected region requires {} ({} elements x {} bytes/elem). \
                     This usually means `value`'s shape doesn't broadcast to the \
                     region shape.",
                    raw.len(),
                    expected_elems * bsize as u64,
                    expected_elems,
                    bsize,
                )));
            }
            // Mirror into the cached array (if any) so subsequent
            // in-memory reads see the patch. If the array hasn't
            // been materialised yet we simply skip the mirror --
            // the next lazy load will read fresh bytes (with this
            // patch) from disk.
            if let Some(arr) = self.data_if_loaded(py) {
                arr.bind(py).set_item(&key, &target)?;
            }
            let bitpix = self.bitpix;
            let hdu_idx = binding.hdu_idx;
            let snapshot_gen = binding.generation;
            let updater_arc = binding.updater.clone();
            let res: PyResult<crate::error::Result<()>> = py.detach(move || {
                let Ok(mut updater) = updater_arc.lock() else {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "FitsFile: writable file lock was poisoned by an \
                         earlier panic; reopen the file to recover",
                    ));
                };
                // Refuse the fast path if the file has been
                // rewritten (or the slot list mutated) since this
                // binding was issued -- the cached `hdu_idx` may
                // now point at a different HDU. The outer fallback
                // path will re-encode the whole file from the live
                // cache, which is safe.
                if updater.generation() != snapshot_gen {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "FitsFile: this HDU's writable-file binding is stale \
                         (the file was rewritten or restructured). The pixel \
                         write has been mirrored into the cached array and \
                         will be persisted on the next flush().",
                    ));
                }
                Ok(write_patch_be(
                    &mut updater,
                    hdu_idx,
                    &fits_start,
                    &fits_shape,
                    bitpix,
                    &raw,
                ))
            });
            match res {
                Ok(inner) => {
                    inner.into_py_result()?;
                    return Ok(());
                }
                Err(_stale) => {
                    // Fast path refused due to stale binding (file
                    // was structurally mutated since this HDU
                    // wrapper was issued). If the array is loaded
                    // it was already mutated above; if not,
                    // materialise + apply patch so subsequent
                    // encodes see it. This keeps already-issued
                    // wrappers usable across structural mutations
                    // at the cost of a one-time materialisation.
                    let Some(arr) = self.ensure_data(py)? else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "section: HDU has no data section (NAXIS == 0)",
                        ));
                    };
                    arr.bind(py).set_item(&key, &target)?;
                    if let Some(flag) = &self.dirty {
                        flag.store(true, Ordering::Release);
                    }
                    self.update_binding = None;
                    return Ok(());
                }
            }
        }
        // Fallback: write through to the cached array and mark the
        // parent file dirty (so `flush()` rewrites it). Used for
        // readonly files (where it's an in-memory edit), compressed
        // images, fancy indexing, and dtype mismatches. Forces a
        // materialisation if the array hasn't been loaded yet.
        let Some(arr) = self.ensure_data(py)? else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "section: HDU has no data section (NAXIS == 0)",
            ));
        };
        arr.bind(py).set_item(&key, &value)?;
        if let Some(flag) = &self.dirty {
            flag.store(true, Ordering::Release);
        }
        Ok(())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        if !self.has_data() {
            return Ok("_ImageSection(<no data>)".into());
        }
        if let Some(arr) = self.data_if_loaded(py) {
            let shape: Vec<usize> = arr.bind(py).getattr("shape")?.extract()?;
            return Ok(format!("_ImageSection(shape={shape:?})"));
        }
        let shape: Vec<usize> = self.axes.iter().rev().map(|&n| n as usize).collect();
        Ok(format!("_ImageSection(shape={shape:?}, lazy)"))
    }
}

// ---------------------------------------------------------------------
// Helpers for PyImageSection::__setitem__
// ---------------------------------------------------------------------

/// Parse a numpy-style indexing key into `(start, region_shape)`
/// (numpy / C order).
///
/// Returns `Ok(None)` for keys that contain anything other than
/// `slice(start, stop)` (with step 1 or absent) or non-negative
/// integers -- those force the slow fallback path.
fn parse_region_key(
    key: &Bound<'_, PyAny>,
    np_shape: &[usize],
) -> PyResult<Option<(Vec<usize>, Vec<usize>, Vec<bool>)>> {
    use pyo3::types::{PySlice, PyTuple};

    let py = key.py();
    let key_tuple: Vec<Bound<'_, PyAny>> = if let Ok(tup) = key.cast::<PyTuple>() {
        tup.iter().collect()
    } else {
        vec![key.clone()]
    };
    if key_tuple.len() > np_shape.len() {
        return Ok(None);
    }

    let mut start = Vec::with_capacity(np_shape.len());
    let mut shape = Vec::with_capacity(np_shape.len());
    // `squeeze[axis] = true` if the user supplied an integer for
    // that axis (numpy semantics: integer indexing collapses the
    // axis, slice indexing preserves it).
    let mut squeeze = Vec::with_capacity(np_shape.len());

    for (axis, idx) in key_tuple.iter().enumerate() {
        let axis_len = np_shape[axis];
        // Try integer first.
        if let Ok(i) = idx.extract::<isize>() {
            let pos = if i < 0 { axis_len as isize + i } else { i };
            if pos < 0 || (pos as usize) >= axis_len {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "section: axis {axis} index {i} out of bounds (len {axis_len})"
                )));
            }
            start.push(pos as usize);
            shape.push(1);
            squeeze.push(true);
            continue;
        }
        // Otherwise must be a slice with step 1.
        if let Ok(s) = idx.cast::<PySlice>() {
            let indices = s.indices(axis_len as isize)?;
            if indices.step != 1 {
                return Ok(None);
            }
            if indices.start < 0 || indices.stop < indices.start {
                return Ok(None);
            }
            start.push(indices.start as usize);
            shape.push((indices.stop - indices.start) as usize);
            squeeze.push(false);
            continue;
        }
        // Anything else (Ellipsis, ndarray index, list, etc) -> fallback.
        let _ = py;
        return Ok(None);
    }
    // Implicit trailing axes: full range.
    for &len in &np_shape[key_tuple.len()..] {
        start.push(0);
        shape.push(len);
        squeeze.push(false);
    }
    Ok(Some((start, shape, squeeze)))
}

/// Decode `raw` (native-endian numpy bytes for `bitpix`) into the
/// matching primitive slice and call
/// [`crate::FitsUpdater::write_image_subarray`].
fn write_patch_be(
    updater: &mut crate::FitsUpdater,
    hdu_idx: usize,
    fits_start: &[u64],
    fits_shape: &[u64],
    bitpix: Bitpix,
    raw: &[u8],
) -> crate::error::Result<()> {
    match bitpix {
        Bitpix::U8 => updater.write_image_subarray::<u8>(hdu_idx, fits_start, fits_shape, raw),
        Bitpix::I16 => {
            let pix: Vec<i16> = raw
                .chunks_exact(2)
                .map(|c| i16::from_ne_bytes([c[0], c[1]]))
                .collect();
            updater.write_image_subarray::<i16>(hdu_idx, fits_start, fits_shape, &pix)
        }
        Bitpix::I32 => {
            let pix: Vec<i32> = raw
                .chunks_exact(4)
                .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            updater.write_image_subarray::<i32>(hdu_idx, fits_start, fits_shape, &pix)
        }
        Bitpix::I64 => {
            let pix: Vec<i64> = raw
                .chunks_exact(8)
                .map(|c| i64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            updater.write_image_subarray::<i64>(hdu_idx, fits_start, fits_shape, &pix)
        }
        Bitpix::F32 => {
            let pix: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            updater.write_image_subarray::<f32>(hdu_idx, fits_start, fits_shape, &pix)
        }
        Bitpix::F64 => {
            let pix: Vec<f64> = raw
                .chunks_exact(8)
                .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            updater.write_image_subarray::<f64>(hdu_idx, fits_start, fits_shape, &pix)
        }
    }
}
