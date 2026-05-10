//! In-place patch updates for image HDUs.
//!
//! [`FitsUpdater`] opens an existing FITS file read/write and
//! exposes [`FitsUpdater::write_image_subarray`] for writing a
//! rectangular patch into one image HDU's data section without
//! touching the rest of the file.
//!
//! This is the primitive behind ``hdu.data[a:b, c:d] = patch``
//! semantics under ``mode='update'``: small edits to large files
//! cost O(patch), not O(file). Patch writes go through positional
//! `pwrite` (`FileExt::write_at`) so no `unsafe` is involved and
//! external truncation surfaces as an `Err` instead of `SIGBUS`.
//!
//! Like astropy's mmap-backed update path, patches are NOT crash-safe:
//! a process death mid-write can leave a torn file with no automatic
//! recovery. Callers that need atomicity should write to a temp file
//! and rename, or take an external snapshot before patching.

use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::Hdu;
use crate::data::encoding::{Bitpix, Pixel};
use crate::error::{FitsError, Result};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

/// Image-HDU pixel layout cached at open time.
#[derive(Debug, Clone)]
struct ImageMeta {
    /// File-byte offset of the unpadded pixel data.
    data_offset: u64,
    /// Axis lengths in FITS order (NAXIS1 first = fastest-varying).
    axes: Vec<u64>,
    /// Pixel encoding from `BITPIX`.
    bitpix: Bitpix,
}

/// Updater for in-place pixel patch writes.
///
/// Open with [`Self::open`], call [`Self::write_image_subarray`] for
/// each patch, and [`Self::flush`] (or drop) when done.
///
/// # Concurrency
///
/// Backed by a writable file handle and `pwrite`. The caller must
/// ensure no other process or thread mutates the file concurrently.
///
/// # Safety
///
/// Resizing an HDU is **not** supported (it would invalidate the
/// offsets of every following HDU). Patch writes are bounds-checked
/// against the cached axis lengths and against the file length.
#[derive(Debug)]
pub struct FitsUpdater {
    file: File,
    /// File length cached at open time. Used to bounds-check writes
    /// without an extra `metadata()` call per patch.
    len: u64,
    /// One entry per HDU. `None` for non-image HDUs (we only support
    /// image patches today).
    images: Vec<Option<ImageMeta>>,
    /// Monotonically increasing tag bumped on every reopen / replace
    /// of the inner state. Callers that cache `(updater, hdu_idx)`
    /// across rewrites can record the generation at the time the
    /// binding was issued and refuse the patch when the generation
    /// has advanced. See `FitsFile.persist_full_rewrite`.
    ///
    /// Only consulted by the Python wrapper today; pure-Rust
    /// `FitsUpdater` users have no shared cached bindings to
    /// invalidate.
    #[cfg(feature = "python")]
    generation: u64,
}

impl FitsUpdater {
    /// Open `path` for in-place updates.
    ///
    /// Parses the file once to discover HDU layouts, then memory-maps
    /// it read/write.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, false)
    }

    /// As [`Self::open`], but propagates the `lenient` flag through
    /// to the read-only probe so that non-conforming files (e.g.
    /// `SIMPLE = F`) can be patched in place when the caller has
    /// already opted into lenient parsing.
    pub fn open_with(path: impl AsRef<Path>, lenient: bool) -> Result<Self> {
        let path = path.as_ref();
        let probe = crate::FitsOpenOptions::new().lenient(lenient).open(path)?;
        let n = probe.len();
        let mut images = Vec::with_capacity(n);
        for i in 0..n {
            let entry = match probe.hdu(i)? {
                Hdu::Image(img) => {
                    let data_offset = probe.data_offset(i).ok_or_else(|| {
                        FitsError::Header(format!("missing data span for HDU {i}"))
                    })?;
                    Some(ImageMeta {
                        data_offset,
                        axes: img.axes().to_vec(),
                        bitpix: img.bitpix(),
                    })
                }
                _ => None,
            };
            images.push(entry);
        }
        drop(probe);
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.metadata()?.len();
        // Sanity-check that the file is at least as large as the
        // greatest (data_offset + data_size) we will ever poke.
        for (i, m) in images.iter().enumerate() {
            if let Some(meta) = m {
                let elems: u64 = meta
                    .axes
                    .iter()
                    .try_fold(1_u64, |acc, &a| acc.checked_mul(a))
                    .ok_or_else(|| FitsError::Data(format!("HDU {i} pixel count overflows u64")))?;
                let bytes = elems
                    .checked_mul(meta.bitpix.byte_size() as u64)
                    .and_then(|b| meta.data_offset.checked_add(b))
                    .ok_or_else(|| FitsError::Data(format!("HDU {i} data extent overflows u64")))?;
                if bytes > len {
                    return Err(FitsError::Data(format!(
                        "FitsUpdater: HDU {i} extends to byte {bytes} but the file is only {len} bytes long"
                    )));
                }
            }
        }
        Ok(Self {
            file,
            len,
            images,
            #[cfg(feature = "python")]
            generation: 0,
        })
    }

    /// Opaque tag that changes whenever the updater's backing file is
    /// replaced (e.g. after `FitsFile.flush()` rewrites the file).
    /// Callers that cache `(Arc<Mutex<FitsUpdater>>, hdu_idx)`
    /// bindings across rewrites should record this at binding time
    /// and re-check before each write -- a mismatch means the slot
    /// indices may have shifted.
    #[cfg(feature = "python")]
    #[must_use]
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Replace this updater's file handle and HDU layout with
    /// `fresh`'s, preserving (and bumping) the generation counter so
    /// existing `Arc<Mutex<FitsUpdater>>` clones see a strictly
    /// increasing tag. Used by the `FitsFile` rewrite path so that
    /// any cached `(arc, hdu_idx)` bindings are invalidated by the
    /// bump rather than silently writing into a re-numbered HDU's
    /// bytes.
    #[cfg(feature = "python")]
    pub(crate) fn replace_with(&mut self, fresh: Self) {
        let next = self.generation.saturating_add(1);
        self.file = fresh.file;
        self.len = fresh.len;
        self.images = fresh.images;
        self.generation = next;
    }

    /// Bump the generation tag without changing the file or layout.
    /// Used by `FitsFile` after structural mutations
    /// (`del`/`insert`/`append`/`__setitem__`) that re-number slots
    /// but do not rewrite the file -- any existing `UpdateBinding`
    /// with the previous tag is now pointing at the wrong HDU and
    /// must refuse the fast path.
    #[cfg(feature = "python")]
    pub(crate) fn bump_generation(&mut self) {
        self.generation = self.generation.saturating_add(1);
    }

    /// Number of HDUs in the file.
    #[must_use]
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// `true` when the file contains zero HDUs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Axis lengths of image HDU `i` in FITS order, or `None` if
    /// the HDU is not an image (or `i` is out of range).
    #[must_use]
    pub fn image_axes(&self, i: usize) -> Option<&[u64]> {
        self.images
            .get(i)
            .and_then(|m| m.as_ref().map(|m| m.axes.as_slice()))
    }

    /// `BITPIX` of image HDU `i`, or `None` if not an image.
    #[must_use]
    pub fn image_bitpix(&self, i: usize) -> Option<Bitpix> {
        self.images
            .get(i)
            .and_then(|m| m.as_ref().map(|m| m.bitpix))
    }

    /// Write a rectangular pixel patch into image HDU `i`.
    ///
    /// `start` and `shape` are in FITS axis order (element 0 is the
    /// `NAXIS1` / fastest-varying axis). Both must have length
    /// `NAXIS`. Values are encoded big-endian and copied row by row;
    /// only the touched byte range is written.
    ///
    /// `pixels` must contain `shape.iter().product()` elements in
    /// C order with the `NAXIS1` axis varying fastest (matches the
    /// in-memory layout of a numpy array of shape `shape` reversed).
    ///
    /// # Crash safety
    ///
    /// None. A crash mid-patch can leave the file with a torn write
    /// (some rows updated, others not). This matches astropy's
    /// mmap-backed update mode. Callers that need atomicity should
    /// snapshot the file before patching or use a temp-file + rename
    /// rewrite via `FitsFile.flush()` instead.
    pub fn write_image_subarray<T: Pixel>(
        &mut self,
        i: usize,
        start: &[u64],
        shape: &[u64],
        pixels: &[T],
    ) -> Result<()> {
        use crate::hdu::subarray::{checked_strides, next_subarray_index, validate_subarray_shape};

        let meta = self
            .images
            .get(i)
            .and_then(|m| m.as_ref())
            .ok_or_else(|| {
                FitsError::Data(format!(
                    "FitsUpdater: HDU {i} is not an image (or out of range)"
                ))
            })?
            .clone();
        if T::BITPIX != meta.bitpix {
            return Err(FitsError::HduMismatch {
                expected: bitpix_name(T::BITPIX),
                found: bitpix_name(meta.bitpix).into(),
            });
        }
        validate_subarray_shape(&meta.axes, start, shape)?;
        if shape.contains(&0) {
            return Ok(());
        }
        let expected: u64 = shape
            .iter()
            .try_fold(1_u64, |acc, &n| acc.checked_mul(n))
            .ok_or_else(|| FitsError::Data("shape product overflows u64".into()))?;
        if pixels.len() as u64 != expected {
            return Err(FitsError::Data(format!(
                "pixels.len() = {} but shape implies {expected} elements",
                pixels.len(),
            )));
        }
        let bsize = meta.bitpix.byte_size();

        let strides = checked_strides(&meta.axes)?;

        let n1 = shape[0];
        let row_elems = n1 as usize;
        let row_bytes = row_elems * bsize;

        // ---- Pass 1: compute the byte offset of every patch row.
        // Done eagerly so that bounds checks fail BEFORE we touch
        // the file -- partial writes don't leave the file in an
        // indeterminate state on bad input.
        let n_rows: u64 = shape[1..].iter().product::<u64>().max(1);
        let n_rows_usize = usize::try_from(n_rows)
            .map_err(|_| FitsError::Data("row count overflows usize".into()))?;
        let mut row_offsets: Vec<u64> = Vec::with_capacity(n_rows_usize);
        {
            let mut idx = vec![0_u64; meta.axes.len()];
            loop {
                let mut elem_off: u64 = start[0];
                for (ax, &io) in idx.iter().enumerate().skip(1) {
                    let s = start[ax]
                        .checked_add(io)
                        .and_then(|v| v.checked_mul(strides[ax]))
                        .and_then(|v| elem_off.checked_add(v))
                        .ok_or_else(|| FitsError::Data("element offset overflows u64".into()))?;
                    elem_off = s;
                }
                let byte_off = elem_off
                    .checked_mul(bsize as u64)
                    .and_then(|v| meta.data_offset.checked_add(v))
                    .ok_or_else(|| FitsError::Data("byte offset overflows u64".into()))?;
                let end = byte_off
                    .checked_add(row_bytes as u64)
                    .ok_or_else(|| FitsError::Data("byte range overflows u64".into()))?;
                if end > self.len {
                    return Err(FitsError::Data(format!(
                        "byte range {byte_off}..{end} exceeds file length {}",
                        self.len
                    )));
                }
                row_offsets.push(byte_off);

                if !next_subarray_index(&mut idx, shape) {
                    break;
                }
            }
        }

        // ---- Pass 2: pre-encode every row of the patch into one
        // contiguous big-endian buffer, then issue the data pwrites
        // row by row.
        let total_bytes = row_offsets
            .len()
            .checked_mul(row_bytes)
            .ok_or_else(|| FitsError::Data("total byte count overflows usize".to_string()))?;
        let mut new_bytes = Vec::with_capacity(total_bytes);
        for px in pixels {
            px.write_be(&mut new_bytes);
        }
        debug_assert_eq!(
            new_bytes.len(),
            total_bytes,
            "encoded patch buffer must equal rows * row_bytes"
        );

        for (i, &off) in row_offsets.iter().enumerate() {
            let chunk = &new_bytes[i * row_bytes..(i + 1) * row_bytes];
            pwrite_all(&self.file, off, chunk)?;
        }
        Ok(())
    }

    /// Force a `fsync` of the data pages to disk.
    ///
    /// After a successful `flush` the patches issued via
    /// [`Self::write_image_subarray`] since the last flush are
    /// considered durable. There is no transactional rollback: a
    /// crash before `flush` returns may still leave the file with
    /// some patched rows committed and others not.
    pub fn flush(&self) -> Result<()> {
        self.file.sync_data().map_err(FitsError::Io)?;
        Ok(())
    }
}

#[cfg(unix)]
fn pwrite_all(file: &File, mut off: u64, mut buf: &[u8]) -> Result<()> {
    while !buf.is_empty() {
        match file.write_at(buf, off) {
            Ok(0) => {
                return Err(FitsError::Io(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "pwrite returned 0 bytes",
                )));
            }
            Ok(n) => {
                off += n as u64;
                buf = &buf[n..];
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(FitsError::Io(e)),
        }
    }
    Ok(())
}

#[cfg(windows)]
fn pwrite_all(file: &File, mut off: u64, mut buf: &[u8]) -> Result<()> {
    while !buf.is_empty() {
        match file.seek_write(buf, off) {
            Ok(0) => {
                return Err(FitsError::Io(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "seek_write returned 0 bytes",
                )));
            }
            Ok(n) => {
                off += n as u64;
                buf = &buf[n..];
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(FitsError::Io(e)),
        }
    }
    Ok(())
}

fn bitpix_name(b: Bitpix) -> &'static str {
    match b {
        Bitpix::U8 => "u8",
        Bitpix::I16 => "i16",
        Bitpix::I32 => "i32",
        Bitpix::I64 => "i64",
        Bitpix::F32 => "f32",
        Bitpix::F64 => "f64",
    }
}
