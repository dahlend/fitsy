//! `PyImageHdu`, `PyImageSection` and `PyRandomGroups` -- image HDU
//! bindings with lazy numpy data.
//!
//! [`PyImageHdu`] is the main wrapper: `data` decodes pixels on the
//! first access and caches the result. [`PyImageSection`] (Python
//! `_ImageSection`) is the slicing proxy behind `ImageHdu.section`;
//! it reads or patches a sub-region without materializing the whole
//! array. [`PyRandomGroups`] wraps the legacy Random Groups primary
//! HDU (Standard Sec.6) and is read-only.
//!
//! [`ReadBinding`] and [`UpdateBinding`] are the two lazy-I/O handles
//! an image HDU or section carries. A `ReadBinding` lets `data` or
//! `section` pull fresh bytes from the source file. An
//! `UpdateBinding` lets `section[a:b] = arr` patch bytes back into a
//! file opened with `mode='update'`. Both are `None` for an HDU
//! built in memory, and for a tile-compressed image: that case is
//! decoded to a plain array eagerly, when the HDU is materialized,
//! rather than through this lazy path.
//!
//! The dtype dispatch for a pixel read -- which keywords choose the
//! returned array's dtype -- is documented on `PyImageHdu::data`,
//! and must agree with the "Image data" section of [`crate::python`].

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};

use numpy::{
    IntoPyArray, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
};
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
/// [`crate::FitsFile`] on demand without ever materializing the
/// whole data section. Cheap to clone (just an `Arc` bump).
#[derive(Debug, Clone)]
pub(crate) struct ReadBinding {
    /// Backing FITS file. Shared with the parent
    /// [`super::file::PyFitsFile`]'s `state.file`. Holding a clone
    /// here keeps the source bytes reachable for as long as any
    /// materialized image HDU might want to lazy-load its data.
    pub(crate) file: Arc<crate::FitsFile>,
    /// Index of this HDU in `file`.
    pub(crate) hdu_idx: usize,
    /// Image axes in **FITS order** (`NAXIS1` fastest). Cached so
    /// section reads and writes do not reparse the header.
    pub(crate) axes: Vec<u64>,
}

/// Image HDU with lazy numpy data.
///
/// Returned by :meth:`FitsFile.hdu` (or ``file[i]``) for an image HDU.
/// Pixels are read on the first access to ``hdu.data``, not before,
/// except for a tile-compressed image, which is decoded in full as
/// soon as the HDU is materialized. Later accesses return the same
/// array, and in-place edits like ``hdu.data[0, 0] = 42`` are kept on
/// the next :meth:`FitsFile.writeto`.
///
/// For images larger than RAM, use :attr:`section`:
/// ``hdu.section[a:b]`` reads only those bytes and
/// ``hdu.section[a:b] = arr`` writes only those, never materializing
/// the whole array. A tile-compressed image gains nothing from
/// :attr:`section`, since its array is already resident by the time
/// ``hdu.section`` can be used.
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
    /// materialized" and "no data section to materialize".
    pub(crate) axes: Vec<u64>,
    /// Whether this HDU was opened in read-only mode. Materialized
    /// numpy arrays are frozen (``WRITEABLE`` flag cleared) when set.
    pub(crate) read_only: bool,
    /// The pixel array. Shared with [`PyImageSection`] via `Arc`
    /// so that section reads/writes observe each other's mutations.
    /// `None` means either:
    ///   - ``NAXIS == 0`` (then `axes` is empty), or
    ///   - the data has not yet been read from disk (then `axes` is
    ///   non-empty and `read_binding` is `Some`).
    pub(crate) data: Arc<Mutex<Option<Py<PyAny>>>>,
    /// Lazy-read source. `Some` when the HDU was materialized from a
    /// plain (non-compressed) image HDU in a `FitsFile` -- read from
    /// disk or from a byte buffer -- enabling a later `data` or
    /// `section` access to pull pixel bytes on demand. `None` for an
    /// HDU built in memory by the user (e.g. via the `ImageHdu`
    /// constructor), and also for a tile-compressed image, whose
    /// pixels are decoded once, eagerly, in `from_built_bytes`.
    pub(crate) read_binding: Option<ReadBinding>,
    /// In-place patch-write binding. Set only when the parent
    /// [`super::file::PyFitsFile`] was opened with `mode='update'`
    /// AND the HDU is an uncompressed image. When present,
    /// `section[a:b] = arr` writes through the file via positional
    /// ``pwrite``.
    pub(crate) update_binding: Option<UpdateBinding>,
    /// Optional back-pointer to the parent `FitsFile`'s dirty
    /// flag. `Some` when the HDU was materialized from a file
    /// opened with `mode='update'`. Mutations that *cannot* be
    /// satisfied by a fast in-place pwrite patch (whole-array
    /// reassignment, fancy / negative-step `section` writes, edits
    /// on a compressed image, etc.) flip the bit so `flush()` /
    /// `__exit__` know to rewrite the file. Pure pixel-patch
    /// writes via ``pwrite`` leave the bit alone.
    pub(crate) dirty: Option<Arc<super::file::DirtyFlags>>,
    /// Backing file that `wcs()` reads a `-TAB` lookup table from.
    /// `Some` for every HDU that comes from a `FitsFile`. This
    /// includes a tile-compressed image, which leaves `read_binding`
    /// `None`. This is a separate field from `read_binding` because
    /// the two go stale at different times. A `data` reassignment
    /// drops `read_binding`, because the on-disk pixel bytes no
    /// longer match the image. The same edit leaves the sibling
    /// `-TAB` table valid.
    pub(crate) wcs_file: Option<Arc<crate::FitsFile>>,
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
            wcs_file: None,
        }
    }

    /// Lock and inspect the cached `data` slot. Returns a clone of
    /// the materialized array if any. Does **not** trigger a
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
    /// materialized, return a clone. Otherwise read it from the
    /// `read_binding`, decode into a numpy array (applying
    /// BSCALE/BZERO/BLANK and byteswapping to native order),
    /// cache it, and return a clone. Returns `None` when `NAXIS`
    /// is 0 or any axis is zero (no data section).
    ///
    /// # Errors
    ///
    /// Returns the Python exception mapped from `fitsy.FitsError` if
    /// the pixel bytes cannot be read from the source file, or if the
    /// data section no longer matches the header's `BITPIX`/`NAXISn`
    /// layout.
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
        let header = header_with_layout(&self.header.lock(), &self.axes, self.bitpix);
        let arr = if header.bzero() == 0.0 && header.bscale() == 1.0 && header.blank().is_none() {
            // Identity scaling: read straight into numpy's buffer so the
            // image is never staged through an intermediate `Vec`.
            let shape: Vec<usize> = self.axes.iter().rev().map(|&n| n as usize).collect();
            file_to_array(py, &binding.file, binding.hdu_idx, self.bitpix, &shape)?
        } else {
            let bytes = binding
                .file
                .read_data_owned(binding.hdu_idx)
                .into_py_result()?;
            let img = ImageHdu::new(header, &bytes).into_py_result()?;
            read_pixels(py, &img, self.bitpix, &self.axes)?
        };
        if self.read_only {
            freeze_array(py, &arr)?;
        }
        let cloned = arr.clone_ref(py);
        self.store_data(Some(arr));
        Ok(Some(cloned))
    }

    /// Reconstruct from a builder snapshot (header + raw bytes).
    /// Decodes the bytes back into a numpy array using the header's
    /// `BITPIX`/`NAXIS*` so the appended HDU is fully editable. The
    /// array is decoded eagerly, unlike `from_image`; there is no
    /// byte buffer left to lazy-read from afterwards.
    ///
    /// # Errors
    ///
    /// Returns the Python exception mapped from `fitsy.FitsError` if
    /// `header`'s `BITPIX` is not a valid FITS pixel encoding.
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
            wcs_file: None,
        })
    }

    /// Encode this HDU's current state into header + data bytes
    /// for writing. Re-stamps BITPIX/NAXIS from the live array.
    /// Lazy: triggers a `data` materialization if the user never
    /// touched it, so the encoded bytes reflect on-disk reality.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`Self::ensure_data`]. Also returns
    /// the Python exception mapped from `fitsy.FitsError` if the
    /// pixel array cannot be re-encoded, for example because its
    /// axis count overflows the FITS `NAXIS` limit.
    pub(crate) fn encode(
        &self,
        py: Python<'_>,
        is_primary: bool,
    ) -> PyResult<(crate::Header, Vec<u8>)> {
        use crate::python::writer::build_image;
        let user_header = self.header.lock().clone();
        match self.ensure_data(py)? {
            Some(arr) => build_image(arr.bind(py), is_primary, user_header),
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

/// Clone `header`, overwriting its layout cards with the geometry the
/// HDU actually has.
///
/// Layout cards on an attached header are advisory: the write path
/// re-stamps `BITPIX`/`NAXIS*` from the HDU's own `axes`/`bitpix`.
/// That is what makes `ImageHdu(new_data, header=other_hdu.header)`
/// work -- the donor header describes a different image.
///
/// The read path must follow the same rule, since [`crate::ImageHdu`]
/// takes its geometry from the header it is given, and a stale
/// `NAXISn` would abort a read of perfectly good bytes. `BZERO`,
/// `BSCALE` and `BLANK` are left alone: those cards are the user's.
fn header_with_layout(header: &crate::Header, axes: &[u64], bitpix: Bitpix) -> crate::Header {
    use crate::Value;
    let mut h = header.clone();
    let _ = h.set("BITPIX", Value::Integer(bitpix.as_i64()), None);
    let _ = h.set("NAXIS", Value::Integer(axes.len() as i64), None);
    for (i, n) in axes.iter().enumerate() {
        let _ = h.set(&format!("NAXIS{}", i + 1), Value::Integer(*n as i64), None);
    }
    let mut k = axes.len() + 1;
    while h.first(&format!("NAXIS{k}")).is_some() {
        h.remove(&format!("NAXIS{k}"));
        k += 1;
    }
    h
}

/// Mark a numpy array as read-only by clearing its `WRITEABLE` flag.
///
/// # Errors
///
/// Returns the Python exception if `arr`'s `setflags` call fails.
fn freeze_array(py: Python<'_>, arr: &Py<PyAny>) -> PyResult<()> {
    arr.bind(py).call_method1("setflags", ((), false))?;
    Ok(())
}

/// The numpy dtype name for a `BITPIX` value.
///
/// Returned as a string rather than a numpy type object, because the
/// callers pass it straight to `dtype()` on the Python side.
pub(super) fn bitpix_numpy_dtype(b: Bitpix) -> &'static str {
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
    // `N` is the pixel width in bytes. The `const` block rejects an
    // `N` that does not match `T` at compile time, so the width
    // cannot desync from the type it decodes.
    fn dec<const N: usize, T: crate::data::Pixel>(bytes: &[u8]) -> Vec<T> {
        const { assert!(N == size_of::<T>(), "chunk width must equal the pixel size") };
        bytes
            .as_chunks::<N>()
            .0
            .iter()
            .map(|c| T::from_be_bytes(c))
            .collect()
    }
    match bitpix {
        Bitpix::U8 => to_array(py, dec::<1, u8>(bytes), shape),
        Bitpix::I16 => to_array(py, dec::<2, i16>(bytes), shape),
        Bitpix::I32 => to_array(py, dec::<4, i32>(bytes), shape),
        Bitpix::I64 => to_array(py, dec::<8, i64>(bytes), shape),
        Bitpix::F32 => to_array(py, dec::<4, f32>(bytes), shape),
        Bitpix::F64 => to_array(py, dec::<8, f64>(bytes), shape),
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

/// Decode `img`'s pixels into a numpy array, choosing the dtype from
/// `BZERO`, `BSCALE` and `BLANK` -- not from `BITPIX` alone.
///
/// Three cases, tried in order:
///
/// 1. Identity scaling (`BZERO` 0, `BSCALE` 1, no `BLANK`): the raw
///    bytes are byte-swapped and returned as-is. The dtype follows
///    `BITPIX` directly.
/// 2. The standard unsigned-integer (or signed-byte) convention
///    (`BSCALE` 1, `BZERO` exactly `2^(N-1)` or, for `BITPIX = 8`,
///    `-128`, no `BLANK`): the stored ints are reinterpreted in the
///    matching unsigned (or `int8`) type of the same width, with no
///    promotion to float.
/// 3. Anything else: `BSCALE`, `BZERO` and `BLANK` are applied via
///    [`ImageHdu::read_physical`] / [`ImageHdu::read_physical_f32`],
///    and the result is floats. An undefined pixel -- one whose
///    stored integer value equals `BLANK`, or a float pixel that was
///    already `NaN` -- becomes `NaN`. `BITPIX` 8, 16 and -32 scale to
///    `f32`; `BITPIX` 32, 64 and -64 scale to `f64`.
///
/// This must agree with the "Image data" section of [`crate::python`].
///
/// # Errors
///
/// Returns the Python exception mapped from `fitsy.FitsError` if the
/// underlying [`ImageHdu`] read fails, for example because `T` from a
/// generic read does not match `bitpix`.
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
    // BZERO=2^(N-1) or -2^(N-1)). fitsy returns the corresponding
    // unsigned (or int8) dtype in this case, instead of the general
    // float path below.
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
    //
    // Width choice: BITPIX 8, 16 and -32 scale to float32; BITPIX 32,
    // 64 and -64 scale to float64. float32 represents every u8/i16
    // value exactly, so the narrow path costs no fidelity on the raw
    // values while halving the array -- and scaled int16 is one of
    // the most common layouts in instrument data.
    if matches!(bitpix, Bitpix::F32 | Bitpix::U8 | Bitpix::I16) {
        let arr = img.read_physical_f32().into_py_result()?.into_vec();
        Ok(to_array(py, arr, &shape))
    } else {
        let arr = img.read_physical().into_py_result()?.into_vec();
        Ok(to_array(py, arr, &shape))
    }
}

/// Identity-scaling decode of `img`'s already-loaded bytes: no
/// `BZERO`/`BSCALE`/`BLANK` applied, dtype follows `bitpix` directly.
///
/// # Errors
///
/// Returns a Python `ValueError` if `img`'s raw byte length does not
/// match `bitpix` and `shape` -- an internal consistency check that
/// should not fail given a correctly constructed [`ImageHdu`].
fn read_raw_to_array(
    py: Python<'_>,
    img: &ImageHdu<'_>,
    bitpix: Bitpix,
    shape: &[usize],
) -> PyResult<Py<PyAny>> {
    raw_bytes_to_array(py, img.raw_bytes(), bitpix, shape)
}

#[pymethods]
impl PyImageHdu {
    /// Construct a new image HDU from an array.
    ///
    /// Parameters
    /// ----------
    /// data : array-like
    ///   Pixel data, of dtype ``bool``, ``int8``, ``uint8``,
    ///   ``int16``, ``uint16``, ``int32``, ``uint32``, ``int64``,
    ///   ``uint64``, ``float32`` or ``float64``. Unsigned integers and
    ///   ``int8`` use the standard ``BZERO`` convention, so they
    ///   round-trip.
    ///
    ///   A numpy array in native byte order is stored by reference, so
    ///   in-place edits reach :meth:`FitsFile.writeto`. Anything else
    ///   is converted once via :func:`numpy.asarray`, and later edits
    ///   to the original are not seen.
    /// header : Header or Mapping[str, Any], optional
    ///   Initial header. Layout cards (``BITPIX``, ``NAXIS*``)
    ///   are recomputed from the array on write. Default ``None``,
    ///   which starts from an empty header.
    /// name : str, optional
    ///   Convenience: sets the ``EXTNAME`` card. Default ``None``,
    ///   which sets no ``EXTNAME``.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///   If `data` is neither an array of one of the dtypes above nor
    ///   something :func:`numpy.asarray` accepts, or if `header` is
    ///   neither a :class:`Header` nor an object with an ``.items()``
    ///   method.
    /// FitsError
    ///   If `header` carries a keyword that exceeds 8 characters or
    ///   contains an invalid character.
    #[new]
    #[pyo3(signature = (data, header=None, name=None))]
    fn py_new(
        py: Python<'_>,
        data: Bound<'_, PyAny>,
        header: Option<Py<PyAny>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let data = crate::python::as_native_ndarray(&data, "ImageHdu")?;
        let bitpix = bitpix_from_array(&data)?;
        let header = build_header(py, header, name)?;
        let axes: Vec<u64> = data.shape().iter().rev().map(|&n| n as u64).collect();
        Ok(Self {
            header,
            bitpix,
            axes,
            read_only: false,
            data: Arc::new(Mutex::new(Some(data.into_any().unbind()))),
            read_binding: None,
            update_binding: None,
            dirty: None,
            wcs_file: None,
        })
    }

    /// The HDU header (see :class:`Header`).
    #[getter]
    fn header(&self) -> PyHeader {
        self.header.clone()
    }

    /// Image axes in **NAXIS order**: ``[NAXIS1, NAXIS2, ...]``.
    ///
    /// When the pixel data has been materialized, the axes are
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
    /// Materializes the array on first access by reading the data
    /// section from disk, byteswapping into native order, and
    /// applying ``BSCALE``/``BZERO``/``BLANK`` scaling. Subsequent
    /// accesses return the same array, and in-place mutation
    /// (``hdu.data[...] = x``) is preserved by the next
    /// :meth:`FitsFile.writeto`, and, in ``mode='update'``, by
    /// :meth:`FitsFile.flush`.
    ///
    /// For images that do not fit in RAM, prefer :attr:`section`
    /// -- ``hdu.section[a:b]`` reads only the requested bytes
    /// without materializing the full array.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray or None
    ///   ``None`` when the HDU has no data section (``NAXIS == 0``).
    ///   Otherwise, the dtype depends on ``BZERO``, ``BSCALE`` and
    ///   ``BLANK``, not on ``BITPIX`` alone:
    ///
    ///   - If ``BZERO`` is 0, ``BSCALE`` is 1 and ``BLANK`` is
    ///     absent, the raw pixels are returned. The dtype follows
    ///     ``BITPIX``.
    ///   - If ``BSCALE`` is 1 and ``BZERO`` is the standard integer
    ///     offset, the matching unsigned dtype is returned.
    ///     ``BITPIX`` 8 with ``BZERO`` -128 returns ``int8``.
    ///   - For all other scaling, ``BSCALE``, ``BZERO`` and
    ///     ``BLANK`` are applied, and the result is floats. A pixel
    ///     whose stored value matches ``BLANK`` -- or that was
    ///     already ``nan`` in a floating-point image -- becomes
    ///     ``nan``. ``BITPIX`` 8, 16 and -32 give ``float32``. All
    ///     other values give ``float64``.
    ///
    ///   An array obtained from a read-only :class:`FitsFile` has its
    ///   ``WRITEABLE`` flag cleared; assigning into it raises
    ///   :class:`ValueError`.
    ///
    /// Raises
    /// ------
    /// FitsError
    ///   If the pixel bytes cannot be read from the source file.
    ///
    /// Notes
    /// -----
    /// A tile-compressed image is decoded in full when the HDU is
    /// materialized, not on this first ``.data`` access; by the time
    /// Python code can reach ``hdu.data``, the array already exists.
    ///
    /// Reading ``.data`` in ``mode='update'`` does not, by itself,
    /// force the file to be rewritten. The next
    /// :meth:`FitsFile.flush` compares this array's bytes against the
    /// file and rewrites only if they differ.
    #[getter]
    fn data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Some(arr) = self.ensure_data(py)? else {
            return Ok(py.None());
        };
        // Handing a writeable array to Python: numpy in-place edits
        // (`hdu.data[i] = v`) are invisible from here, so the cache has
        // to be written back on flush. This lives in the getter rather
        // than in `ensure_data` so it also covers HDUs whose pixels were
        // materialized eagerly (tile-compressed images), and so the
        // internal `encode` read does not mark the file dirty.
        // Read-only handles get a frozen array and never need it.
        //
        // This records only "may have changed": `flush` compares the
        // cache against the file, so merely reading `.data` in update
        // mode does not cost a rewrite.
        if !self.read_only
            && let Some(flag) = &self.dirty
        {
            flag.handed_out.store(true, Ordering::Release);
        }
        Ok(arr)
    }

    /// Whether the cached pixel array still matches the bytes on disk.
    ///
    /// Returns `false` when that cannot be established (no read source,
    /// or a scaled HDU whose cache is not a straight image of the data
    /// section), so the caller falls back to rewriting.
    pub(crate) fn data_matches_source(&self, py: Python<'_>) -> bool {
        let Some(arr) = self.data_if_loaded(py) else {
            return true; // never materialized, so nothing to write back
        };
        let Some(binding) = self.read_binding.as_ref() else {
            return false;
        };
        let header = header_with_layout(&self.header.lock(), &self.axes, self.bitpix);
        if header.bzero() != 0.0 || header.bscale() != 1.0 || header.blank().is_some() {
            return false;
        }
        let bound = arr.bind(py);
        let (file, idx) = (&binding.file, binding.hdu_idx);
        match self.bitpix {
            Bitpix::U8 => cache_matches::<u8>(bound, file, idx),
            Bitpix::I16 => cache_matches::<i16>(bound, file, idx),
            Bitpix::I32 => cache_matches::<i32>(bound, file, idx),
            Bitpix::I64 => cache_matches::<i64>(bound, file, idx),
            Bitpix::F32 => cache_matches::<f32>(bound, file, idx),
            Bitpix::F64 => cache_matches::<f64>(bound, file, idx),
        }
    }

    /// Slicing accessor that mirrors :class:`numpy.ndarray`
    /// indexing. ``hdu.section[a:b, c:d]`` reads only the
    /// requested region from disk -- no full-image materialization.
    ///
    /// In ``mode='update'``, ``hdu.section[a:b] = arr`` writes only
    /// the touched bytes back via positional ``pwrite``, again
    /// without materializing the full image. This is the supported
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
    ///   Slicing proxy. Use ``section[i, j, k]`` exactly like
    ///   ``data[i, j, k]``.
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

    /// Replace the pixel array.
    ///
    /// Accepts any array-like value; its dtype must resolve to a
    /// supported FITS type. The header's ``BITPIX`` and ``NAXIS*``
    /// cards are updated immediately to match the new array. Passing
    /// ``None`` clears the data section instead (``NAXIS`` becomes 0).
    ///
    /// Setting this attribute drops any lazy-read or in-place-write
    /// binding this HDU held: a later ``hdu.section[a:b] = arr``
    /// patch re-encodes the whole file rather than writing in place,
    /// and a later ``hdu.data`` read never falls back to on-disk
    /// bytes from before the assignment.
    ///
    /// Parameters
    /// ----------
    /// value : array-like or None
    ///   New pixel data, of one of the dtypes :class:`ImageHdu`
    ///   accepts, or ``None`` to remove the data section.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///   If `value` is neither an array of a supported dtype nor
    ///   something :func:`numpy.asarray` accepts.
    #[setter]
    fn set_data(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        if let Some(flag) = &self.dirty {
            flag.definite.store(true, Ordering::Release);
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
        if value.is_none() {
            self.store_data(None);
            self.axes.clear();
            self.restamp_layout(&[]);
            return Ok(());
        }
        let value = crate::python::as_native_ndarray(&value, "ImageHdu.data")?;
        self.bitpix = bitpix_from_array(&value)?;
        let axes: Vec<u64> = value.shape().iter().rev().map(|&n| n as u64).collect();
        self.axes.clone_from(&axes);
        self.store_data(Some(value.into_any().unbind()));
        self.restamp_layout(&axes);
        Ok(())
    }

    /// Resolve the WCS for this HDU.
    ///
    /// Parameters
    /// ----------
    /// alt : str, optional
    ///   Single ASCII character. ``' '`` (default) selects the
    ///   primary description; ``'A'`` through ``'Z'`` select
    ///   alternate descriptions.
    ///
    /// Returns
    /// -------
    /// Wcs or None
    ///   ``None`` if the header carries no WCS for ``alt``.
    ///
    /// Raises
    /// ------
    /// FitsError
    ///   If `alt` is not ``' '`` or one of ``'A'``-``'Z'``, if the
    ///   header carries a malformed WCS, or if a ``-TAB`` axis
    ///   cannot be resolved (see Notes).
    ///
    /// Notes
    /// -----
    /// A ``-TAB`` axis (Paper III Sec.6) stores its coordinate
    /// array in a sibling BINTABLE. The ``PSi_0`` / ``PVi_1`` cards
    /// name that table. An HDU from :func:`fitsy.open` keeps a
    /// handle to its file. This method resolves the lookup through
    /// that handle, exactly as :meth:`fitsy.FitsFile.wcs` does. An
    /// HDU built in memory has no file to search. A ``-TAB`` axis
    /// then raises here, not later at transform time. Use
    /// ``fitsy.Wcs(hdu.header)`` to inspect such a header without
    /// the lookup table.
    #[pyo3(signature = (alt=' '))]
    fn wcs(&self, alt: char) -> PyResult<Option<PyWcs>> {
        let wcs = {
            let header = self.header.lock();
            crate::wcs::Wcs::from_header(&header, alt).into_py_result()?
        };
        let Some(mut wcs) = wcs else { return Ok(None) };
        if !wcs.tab_specs.is_empty() {
            let Some(file) = self.wcs_file.as_ref() else {
                return Err(super::err_to_py(crate::error::FitsError::Wcs(
                    "WCS has a -TAB axis, but this HDU carries no file to \
                     load the lookup table from; read the HDU from \
                     fitsy.open, or use fitsy.Wcs(hdu.header) for \
                     header-only inspection"
                        .into(),
                )));
            };
            wcs.resolve_tab(file).into_py_result()?;
        }
        Ok(Some(PyWcs::from(wcs)))
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

/// Infer `BITPIX` from an array's dtype. `kind`/`itemsize` come
/// straight off the descriptor struct, so no Python attribute
/// lookups are involved.
///
/// # Errors
///
/// Returns a Python `TypeError` if `arr`'s dtype is not `bool`,
/// `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`,
/// `uint64`, `float32` or `float64`.
fn bitpix_from_array(arr: &Bound<'_, PyUntypedArray>) -> PyResult<Bitpix> {
    let dtype = arr.dtype();
    let kind = dtype.kind();
    let itemsize = dtype.itemsize();
    Ok(match (kind, itemsize) {
        // 'b' (numpy bool) and 1-byte int/uint all map to BITPIX=8.
        // BZERO offsets map signed/unsigned to the same BITPIX storage.
        (b'b', _) | (b'u' | b'i', 1) => Bitpix::U8,
        (b'i' | b'u', 2) => Bitpix::I16,
        (b'i' | b'u', 4) => Bitpix::I32,
        (b'i' | b'u', 8) => Bitpix::I64,
        (b'f', 4) => Bitpix::F32,
        (b'f', 8) => Bitpix::F64,
        _ => {
            let name: String = dtype.as_any().str()?.extract()?;
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
///
/// # Errors
///
/// Returns a Python `TypeError` if `src` is neither a [`PyHeader`]
/// nor an object with an `.items()` method, or if one of its values
/// does not convert to a FITS value. Returns the Python exception
/// mapped from `fitsy.FitsError` if a keyword in `src` fails
/// validation.
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

/// A pixel type that can be byte-swapped from big-endian in place.
///
/// The float impls swap through the same-width unsigned integer so the
/// bit pattern -- including a NaN payload (Sec.4.4.2.5) -- is carried
/// through untouched. `from_be` is a no-op on big-endian hosts.
trait SwapBe: numpy::Element + bytemuck::Pod {
    fn swap_be_in_place(slice: &mut [Self]);
}

impl SwapBe for u8 {
    fn swap_be_in_place(_: &mut [Self]) {}
}

/// Every width swaps through its unsigned view: the signed spelling
/// (`i16::from_be`) costs about a third of the throughput because it
/// does not vectorize as well, and for floats it is the only way to
/// move the bits without going through a float value at all.
macro_rules! swap_be {
    ($($t:ty => $u:ty),*) => {$(
        impl SwapBe for $t {
            fn swap_be_in_place(slice: &mut [Self]) {
                for x in bytemuck::cast_slice_mut::<$t, $u>(slice) {
                    *x = <$u>::from_be(*x);
                }
            }
        }
    )*};
}
swap_be!(i16 => u16, i32 => u32, i64 => u64, f32 => u32, f64 => u64);

/// Allocate a numpy array of `shape`, let `fill` write the raw
/// big-endian pixels straight into its buffer, then swap in place.
///
/// The obvious spelling -- read into a `Vec<u8>`, decode into a
/// `Vec<T>`, hand that to `into_pyarray` -- allocates the image three
/// times and fills it twice. Going through numpy's own buffer does one
/// allocation and one pass, which also halves the peak footprint.
///
/// # Errors
///
/// Returns whatever `fill` returns.
fn alloc_swapped<T: SwapBe>(
    py: Python<'_>,
    shape: &[usize],
    fill: impl FnOnce(&mut [u8]) -> PyResult<()>,
) -> PyResult<Py<PyAny>> {
    let n: usize = shape.iter().product();
    let arr = numpy::PyArray1::<T>::zeros(py, n, false);
    {
        let mut rw = arr.readwrite();
        let dst = rw.as_slice_mut()?;
        fill(bytemuck::cast_slice_mut(dst))?;
        T::swap_be_in_place(dst);
    }
    Ok(arr
        .reshape(shape.to_vec())
        .expect(
            "internal invariant: element count must equal NAXIS product; \
             this is a fitsy bug, please report",
        )
        .into_any()
        .unbind())
}

/// Dispatch `alloc_swapped` on `BITPIX`.
macro_rules! by_bitpix {
    ($py:expr, $bitpix:expr, $shape:expr, $fill:expr) => {
        match $bitpix {
            Bitpix::U8 => alloc_swapped::<u8>($py, $shape, $fill),
            Bitpix::I16 => alloc_swapped::<i16>($py, $shape, $fill),
            Bitpix::I32 => alloc_swapped::<i32>($py, $shape, $fill),
            Bitpix::I64 => alloc_swapped::<i64>($py, $shape, $fill),
            Bitpix::F32 => alloc_swapped::<f32>($py, $shape, $fill),
            Bitpix::F64 => alloc_swapped::<f64>($py, $shape, $fill),
        }
    };
}

/// Identity-scaling decode straight from the file: no intermediate
/// buffer, so the image is allocated once and touched once.
///
/// # Errors
///
/// Returns the Python exception mapped from `fitsy.FitsError` if the
/// pixel bytes cannot be read from `file`.
fn file_to_array(
    py: Python<'_>,
    file: &crate::FitsFile,
    hdu_idx: usize,
    bitpix: Bitpix,
    shape: &[usize],
) -> PyResult<Py<PyAny>> {
    by_bitpix!(py, bitpix, shape, |dst: &mut [u8]| file
        .read_data_into(hdu_idx, dst)
        .into_py_result())
}

/// Re-read HDU `hdu_idx` and compare it byte-for-byte with the pixels
/// numpy is holding.
///
/// Streamed through a small fixed buffer rather than a full-size one so
/// verifying does not double the peak footprint of a large image, and
/// so a difference near the start exits without reading the rest.
/// Compared as raw bytes rather than as `T`, which keeps the answer
/// exact for NaN payloads; the buffer is allocated as `[T]` so its byte
/// view is always correctly aligned.
fn cache_matches<T: SwapBe>(
    arr: &Bound<'_, PyAny>,
    file: &crate::FitsFile,
    hdu_idx: usize,
) -> bool {
    /// Chunk size in elements, sized so the buffer stays around 1 MiB.
    const CHUNK_BYTES: usize = 1 << 20;

    let Ok(typed) = arr.cast::<numpy::PyArrayDyn<T>>() else {
        return false;
    };
    let ro = typed.readonly();
    let Ok(cached) = ro.as_slice() else {
        return false; // non-contiguous: cannot compare cheaply
    };
    let elem = size_of::<T>();
    let per_chunk = (CHUNK_BYTES / elem).max(1);
    let cached_bytes: &[u8] = bytemuck::cast_slice(cached);
    let mut scratch: Vec<T> = vec![T::zeroed(); per_chunk.min(cached.len().max(1))];
    let mut done = 0_usize;
    while done < cached.len() {
        let n = per_chunk.min(cached.len() - done);
        let buf = &mut scratch[..n];
        let byte_off = done * elem;
        if file
            .read_data_range_into(hdu_idx, byte_off as u64, bytemuck::cast_slice_mut(buf))
            .is_err()
        {
            return false;
        }
        T::swap_be_in_place(buf);
        if bytemuck::cast_slice::<T, u8>(buf) != &cached_bytes[byte_off..byte_off + n * elem] {
            return false;
        }
        done += n;
    }
    true
}

/// Identity-scaling decode from bytes already in memory.
///
/// # Errors
///
/// Returns a Python `ValueError` if `raw`'s length does not match
/// `bitpix` and `shape`.
fn raw_bytes_to_array(
    py: Python<'_>,
    raw: &[u8],
    bitpix: Bitpix,
    shape: &[usize],
) -> PyResult<Py<PyAny>> {
    by_bitpix!(py, bitpix, shape, |dst: &mut [u8]| {
        if dst.len() != raw.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "pixel buffer is {} bytes, data section is {}",
                dst.len(),
                raw.len()
            )));
        }
        dst.copy_from_slice(raw);
        Ok(())
    })
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
/// :meth:`group`. Indexing the file to reach this HDU (for example,
/// ``file[0]``) materializes it in memory.
///
/// This class exposes no way to edit the header or the groups.
/// :meth:`FitsFile.writeto` and :meth:`FitsFile.flush` write back
/// the header and the data section the HDU was read with, so the
/// written HDU matches the source byte for byte.
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
    /// Clone the underlying `Header`, for serialization.
    pub(crate) fn header_clone(&self) -> crate::Header {
        self.header.lock().clone()
    }

    /// Clone the data section, for serialization.
    ///
    /// The bytes stay in the big-endian FITS order they were read
    /// in. The bindings expose no way to edit them, so writing them
    /// back reproduces the source HDU.
    pub(crate) fn data_clone(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Snapshot a borrowed random-groups HDU into an owned Python
    /// object. The data section is copied, so the result outlives the
    /// [`FitsFile`](crate::FitsFile) it came from.
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

    /// Decode one group as ``(parameters, data)`` numpy arrays.
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///   Group index, 0-based. Does not accept a negative index.
    ///
    /// Returns
    /// -------
    /// tuple of numpy.ndarray
    ///   ``(parameters, data)``. Both arrays share the HDU's
    ///   ``BITPIX`` dtype and are read-only. ``parameters`` has
    ///   length :attr:`n_params`; ``data`` has length
    ///   :attr:`data_per_group`. Neither ``BSCALE``/``BZERO`` nor
    ///   ``PSCALn``/``PZEROn`` is applied -- both arrays hold the
    ///   stored, unscaled values.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///   If `i` is not less than :attr:`n_groups`.
    /// OverflowError
    ///   If `i` is negative.
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
        // Decoded fresh on every call and with no write-back path, so an
        // edit could never persist. Freeze both, matching the table
        // accessors, so a write raises instead of vanishing.
        freeze_array(py, &params)?;
        freeze_array(py, &data)?;
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
/// * If ``hdu.data`` has already been materialized, slicing and
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
    /// Whether to freeze freshly materialized arrays.
    pub(crate) read_only: bool,
    /// Snapshot of the parent HDU's header (cheap clone of the
    /// shared `Arc<Mutex<Header>>`). Needed for lazy reads so we
    /// can apply BSCALE/BZERO/BLANK scaling without holding a
    /// back-pointer to the parent HDU.
    pub(crate) header: PyHeader,
    /// Shared cache of the materialized pixel array (same `Arc` as
    /// the parent [`PyImageHdu`]'s `data`). Lets section
    /// reads/writes observe and update the parent's view.
    pub(crate) data: Arc<Mutex<Option<Py<PyAny>>>>,
    /// Lazy-read source. `Some` when the parent HDU was opened from
    /// disk (or a byte buffer) as a plain image, enabling
    /// region-only reads. `None` for an HDU built in memory and for
    /// a tile-compressed image, matching the parent [`PyImageHdu`]'s
    /// `read_binding` field.
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
    pub(crate) dirty: Option<Arc<super::file::DirtyFlags>>,
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

    /// Materialize the full pixel array (same as `PyImageHdu::ensure_data`)
    /// and cache it in the shared `data` slot.
    ///
    /// # Errors
    ///
    /// Returns the Python exception mapped from `fitsy.FitsError` if
    /// the pixel bytes cannot be read from the source file, or if the
    /// data section no longer matches the header's `BITPIX`/`NAXISn`
    /// layout.
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
        let header = header_with_layout(&self.header.lock(), &self.axes, self.bitpix);
        let img = ImageHdu::new(header, &bytes).into_py_result()?;
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
    /// Read a region: ``section[key]``.
    ///
    /// Parameters
    /// ----------
    /// key : int, slice, or tuple of int/slice
    ///   Indexed like ``hdu.data[key]``. When the pixel array has not
    ///   yet been loaded into memory, a `key` built only from a
    ///   contiguous ``slice(start, stop)`` (step 1, or omitted) and
    ///   non-negative integers reads only the requested region from
    ///   disk. Any other form -- a negative step, ``Ellipsis``, fancy
    ///   indexing, a boolean mask -- loads the full array into memory
    ///   first, then slices it.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///   The selected region. Its dtype is the one :attr:`ImageHdu.data`
    ///   would have for this HDU.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the HDU has no data section (``NAXIS == 0``).
    /// IndexError
    ///   If an integer entry in `key` is out of bounds for its axis.
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
            // scaling, fall through to the materialize-then-slice
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
        // Fallback: materialize the whole array, then slice. Used
        // when the key is something `parse_region_key` doesn't
        // understand (fancy indexing, negative steps, ...).
        let Some(arr) = self.ensure_data(py)? else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "section: HDU has no data section (NAXIS == 0)",
            ));
        };
        Ok(arr.bind(py).get_item(&key)?.unbind())
    }

    /// Assign a patch into the image: ``section[key] = value``.
    ///
    /// The fast path -- triggered when the file was opened with
    /// ``mode='update'`` and the key is a tuple of
    /// ``slice(start, stop)`` (step 1) and non-negative integers --
    /// writes only the affected pixel bytes through the file via
    /// positional ``pwrite`` (O(patch), no full-image rewrite). If
    /// the data array has already been materialized in memory the
    /// patch is also mirrored into it for consistency.
    ///
    /// All other writes (compressed images, fancy indexing,
    /// negative steps, dtype casts that change the underlying byte
    /// representation) fall back to mutating the cached numpy
    /// array and flag the file as dirty; the next
    /// :meth:`FitsFile.flush` (or clean ``__exit__``) rewrites the
    /// file via a sibling temp file + atomic rename.
    ///
    /// Parameters
    /// ----------
    /// key : int, slice, or tuple of int/slice
    ///   Region to patch, indexed like ``hdu.data[key] = value``.
    /// value : array-like
    ///   Replacement pixels. Must broadcast to the shape `key`
    ///   selects.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the HDU has no data section (``NAXIS == 0``); if the
    ///   file is open with ``mode='update'`` and this HDU has
    ///   non-identity scaling (``BSCALE != 1``, ``BZERO != 0``, or
    ///   ``BLANK`` is set); if the file is open with
    ///   ``mode='update'`` and `key` is not built only from
    ///   contiguous ``slice(start, stop)`` (step 1) and non-negative
    ///   integers; if `value` does not broadcast to the region `key`
    ///   selects; or if the parent :class:`FitsFile` is read-only, in
    ///   which case the cached array is frozen.
    /// IndexError
    ///   If an integer entry in `key` is out of bounds for its axis.
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
            // been materialized yet we simply skip the mirror --
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
                    // materialize + apply patch so subsequent
                    // encodes see it. This keeps already-issued
                    // wrappers usable across structural mutations
                    // at the cost of a one-time materialization.
                    let Some(arr) = self.ensure_data(py)? else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "section: HDU has no data section (NAXIS == 0)",
                        ));
                    };
                    arr.bind(py).set_item(&key, &target)?;
                    if let Some(flag) = &self.dirty {
                        flag.definite.store(true, Ordering::Release);
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
        // materialization if the array hasn't been loaded yet.
        let Some(arr) = self.ensure_data(py)? else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "section: HDU has no data section (NAXIS == 0)",
            ));
        };
        arr.bind(py).set_item(&key, &value)?;
        if let Some(flag) = &self.dirty {
            flag.definite.store(true, Ordering::Release);
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
///
/// # Errors
///
/// Returns a Python `IndexError` if an integer entry in `key` is out
/// of bounds for its axis in `np_shape`.
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
///
/// # Errors
///
/// Returns a [`crate::FitsError`] if the underlying patch write
/// fails, for example an I/O error writing to the file.
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
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| i16::from_ne_bytes(*c))
                .collect();
            updater.write_image_subarray::<i16>(hdu_idx, fits_start, fits_shape, &pix)
        }
        Bitpix::I32 => {
            let pix: Vec<i32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_ne_bytes(*c))
                .collect();
            updater.write_image_subarray::<i32>(hdu_idx, fits_start, fits_shape, &pix)
        }
        Bitpix::I64 => {
            let pix: Vec<i64> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_ne_bytes(*c))
                .collect();
            updater.write_image_subarray::<i64>(hdu_idx, fits_start, fits_shape, &pix)
        }
        Bitpix::F32 => {
            let pix: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_ne_bytes(*c))
                .collect();
            updater.write_image_subarray::<f32>(hdu_idx, fits_start, fits_shape, &pix)
        }
        Bitpix::F64 => {
            let pix: Vec<f64> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_ne_bytes(*c))
                .collect();
            updater.write_image_subarray::<f64>(hdu_idx, fits_start, fits_shape, &pix)
        }
    }
}
