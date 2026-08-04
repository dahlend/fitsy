//! In-place patch updates for image HDUs.
//!
//! # Purpose
//!
//! [`FitsUpdater`] opens an existing FITS file for reading and
//! writing. [`FitsUpdater::write_image_subarray`] writes a rectangular
//! patch into the data section of one image HDU and leaves the rest of
//! the file untouched. A small edit to a large file therefore costs
//! the size of the patch.
//!
//! This backs the Python `hdu.data[a:b, c:d] = patch` under
//! `mode='update'`.
//!
//! # Design constraints
//!
//! Writes go through positional `pwrite`. The module holds no
//! `unsafe` code, and a truncated file returns an error rather than
//! raising SIGBUS.
//!
//! A patch is not crash-safe. A process that dies mid-write leaves the
//! file torn, with some rows updated and others not. A caller that
//! needs atomicity takes a snapshot first, or writes to a temporary
//! file and renames it.
//!
//! An HDU cannot change size here. A resize would move every later
//! HDU, and the cached offsets would then be wrong.

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
/// The caller must ensure nothing else mutates the file meanwhile.
///
/// # Limits
///
/// This type cannot resize an HDU. A resize would move every later
/// HDU, and the cached offsets would then be wrong. Each patch is
/// bounds-checked against the cached axis lengths and the file
/// length.
#[derive(Debug)]
pub struct FitsUpdater {
    file: File,
    /// File length cached at open time. Used to bounds-check writes
    /// without an extra `metadata()` call per patch.
    len: u64,
    /// One entry per HDU. `None` for non-image HDUs (we only support
    /// image patches today).
    images: Vec<Option<ImageMeta>>,
    /// Bumped whenever the inner state is replaced. A caller caching
    /// `(updater, hdu_idx)` across rewrites records the tag and
    /// refuses the patch once it advances.
    ///
    /// Only the Python wrapper needs this; pure-Rust users have no
    /// cached bindings to invalidate.
    #[cfg(feature = "python")]
    generation: u64,
}

impl FitsUpdater {
    /// Open `path` for in-place updates.
    ///
    /// This parses the file once for its HDU layout, then keeps a
    /// read/write handle. It parses headers leniently, as
    /// [`FitsFile::open`](crate::FitsFile::open) does. Call
    /// [`Self::open_with`] with `lenient = false` for strict parsing.
    ///
    /// # Errors
    ///
    /// The conditions of [`Self::open_with`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, true)
    }

    /// Open `path` for in-place updates, with explicit control over
    /// lenient parsing.
    ///
    /// The `lenient` flag reaches the read-only probe that reads the
    /// layout. A caller that has opted into lenient parsing can
    /// therefore patch a non-conforming file, such as one with
    /// `SIMPLE = F`.
    ///
    /// # Errors
    ///
    /// - [`FitsError::Io`] when `path` cannot be opened for reading
    ///   and writing.
    /// - The conditions of
    ///   [`FitsFile::open_with`](crate::FitsFile::open_with), because
    ///   this reads the layout first.
    /// - [`FitsError::Data`] when an image HDU declares a pixel count
    ///   that overflows `u64`.
    /// - [`FitsError::Header`] when an image HDU reports no data
    ///   offset.
    pub fn open_with(path: impl AsRef<Path>, lenient: bool) -> Result<Self> {
        let path = path.as_ref();
        let probe = crate::FitsFile::open_with(path, lenient)?;
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
    /// `start` and `shape` are in FITS axis order (element 0 is
    /// `NAXIS1`, the fastest-varying axis) and must both have length
    /// `NAXIS`. Only the touched byte range is written.
    ///
    /// The `pixels` argument holds `shape.iter().product()` elements
    /// in C order, with `NAXIS1` varying fastest. This is the layout
    /// of a numpy array whose shape is the reverse of `shape`.
    ///
    /// # Crash safety
    ///
    /// A patch is not atomic. A process that dies mid-patch leaves
    /// some rows updated and others not. Take a snapshot first when
    /// you need atomicity.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] in five cases:
    ///
    /// - HDU `i` is not an image, or lies out of range.
    /// - `start` or `shape` has the wrong length.
    /// - The region escapes the array.
    /// - `pixels.len()` does not match the product of `shape`.
    /// - A byte offset overflows `u64`.
    ///
    /// [`FitsError::HduMismatch`] when `T` does not match the `BITPIX`
    /// of the HDU. [`FitsError::Io`] when the write fails.
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
                expected: T::BITPIX.rust_type_name(),
                found: meta.bitpix.rust_type_name().into(),
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
    /// After this returns, every patch that
    /// [`Self::write_image_subarray`] issued since the last flush is
    /// durable. There is no rollback. A crash before this returns can
    /// still leave some patched rows committed and others not.
    ///
    /// # Errors
    ///
    /// [`FitsError::Io`] when the `fsync` fails.
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
