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
//! HDU, and the cached offsets would then be wrong. This rules out a
//! tile-compressed image, whose tiles change byte length when they
//! are rewritten.

use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::Hdu;
use crate::data::encoding::{Bitpix, Pixel};
use crate::error::{FitsError, Result};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

/// Encode physical values into the big-endian stored form of `meta`.
///
/// Each value is inverted through `BZERO`, `BSCALE` and `BLANK`, then
/// narrowed to the type `BITPIX` names. An integer that does not fit
/// that type is an error rather than a wrapped value.
fn encode_physical(meta: &ImageMeta, pixels: &[f64]) -> Result<Vec<u8>> {
    let s = &meta.scaling;
    let mut be = Vec::with_capacity(pixels.len() * meta.bitpix.byte_size());
    macro_rules! narrow {
        ($t:ty, $v:expr) => {{
            let raw = s.unapply_int($v)?;
            let fitted = <$t>::try_from(raw).map_err(|_| {
                FitsError::Data(format!(
                    "physical value {} scales to {raw}, outside the range of {}",
                    $v,
                    stringify!($t)
                ))
            })?;
            fitted.write_be(&mut be);
        }};
    }
    for &v in pixels {
        match meta.bitpix {
            Bitpix::U8 => narrow!(u8, v),
            Bitpix::I16 => narrow!(i16, v),
            Bitpix::I32 => narrow!(i32, v),
            Bitpix::I64 => {
                s.unapply_int(v)?.write_be(&mut be);
            }
            Bitpix::F32 => {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "BITPIX = -32 stores a single-precision pixel"
                )]
                (s.unapply_real(v)? as f32).write_be(&mut be);
            }
            Bitpix::F64 => s.unapply_real(v)?.write_be(&mut be),
        }
    }
    Ok(be)
}

/// Whether one HDU can take a patch, and why not when it cannot.
#[derive(Debug, Clone)]
enum Patchable {
    /// An image HDU, with its pixel layout cached.
    Image(ImageMeta),
    /// A tile-compressed image. A rewritten tile does not keep its
    /// byte length, so a patch would resize the HDU.
    #[cfg(feature = "compression")]
    Compressed,
    /// Any other HDU.
    No,
}

/// Image-HDU pixel layout cached at open time.
#[derive(Debug, Clone)]
struct ImageMeta {
    /// File-byte offset of the unpadded pixel data.
    data_offset: u64,
    /// Axis lengths in FITS order (NAXIS1 first = fastest-varying).
    axes: Vec<u64>,
    /// Pixel encoding from `BITPIX`.
    bitpix: Bitpix,
    /// `BZERO`, `BSCALE` and `BLANK`, cached at open.
    scaling: crate::data::Scaling,
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
///
/// [`write_image_subarray_physical`](Self::write_image_subarray_physical)
/// writes in the units the header describes, inverting `BZERO`,
/// `BSCALE` and `BLANK`.
/// [`write_image_subarray`](Self::write_image_subarray) writes stored
/// values, the counterpart of
/// [`ImageHdu::read_raw`](crate::ImageHdu::read_raw).
///
/// A tile-compressed image takes no patch either. A rewritten tile
/// does not keep its byte length.
/// [`write_image_subarray`](Self::write_image_subarray) reports this.
/// [`image_axes`](Self::image_axes) and
/// [`image_bitpix`](Self::image_bitpix) return `None` for such an HDU.
/// Decompress the file first, with
/// [`FitsFile::write_decompressed`](crate::FitsFile::write_decompressed).
#[derive(Debug)]
pub struct FitsUpdater {
    file: File,
    /// File length cached at open time. Used to bounds-check writes
    /// without an extra `metadata()` call per patch.
    len: u64,
    /// One entry per HDU. Not `Patchable::Image` for HDUs that take
    /// no patch (we only support
    /// image patches today).
    images: Vec<Patchable>,
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
                    let h = img.header();
                    Patchable::Image(ImageMeta {
                        data_offset,
                        axes: img.axes().to_vec(),
                        bitpix: img.bitpix(),
                        scaling: crate::data::Scaling {
                            bzero: h.bzero(),
                            bscale: h.bscale(),
                            blank: h.blank(),
                        },
                    })
                }
                #[cfg(feature = "compression")]
                Hdu::CompressedImage(_) => Patchable::Compressed,
                _ => Patchable::No,
            };
            images.push(entry);
        }
        drop(probe);
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.metadata()?.len();
        // Sanity-check that the file is at least as large as the
        // greatest (data_offset + data_size) we will ever poke.
        for (i, m) in images.iter().enumerate() {
            if let Patchable::Image(meta) = m {
                let elems = crate::data::encoding::axis_product(&meta.axes)
                    .map_err(|_| FitsError::Data(format!("HDU {i} pixel count overflows u64")))?;
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
        match self.images.get(i) {
            Some(Patchable::Image(m)) => Some(m.axes.as_slice()),
            _ => None,
        }
    }

    /// `BITPIX` of image HDU `i`, or `None` if not an image.
    #[must_use]
    pub fn image_bitpix(&self, i: usize) -> Option<Bitpix> {
        match self.images.get(i) {
            Some(Patchable::Image(m)) => Some(m.bitpix),
            _ => None,
        }
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
    /// These are stored values, the ones
    /// [`ImageHdu::read_raw`](crate::ImageHdu::read_raw) returns. When
    /// the header declares `BZERO`, `BSCALE` or `BLANK`, a stored
    /// value is not the value a reader sees, and
    /// [`Self::write_image_subarray_physical`] is the method that
    /// takes the units the header describes.
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
    /// - HDU `i` is a tile-compressed image. Such an HDU takes no
    ///   in-place patch, because a rewritten tile does not keep its
    ///   byte length.
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
        let meta = self.patchable_image(i)?;
        if T::BITPIX != meta.bitpix {
            return Err(FitsError::HduMismatch {
                expected: T::BITPIX.rust_type_name(),
                found: meta.bitpix.rust_type_name().into(),
            });
        }
        Self::validate_patch(&meta, start, shape, pixels.len())?;
        if shape.contains(&0) {
            return Ok(());
        }
        let mut be = Vec::with_capacity(pixels.len() * meta.bitpix.byte_size());
        for px in pixels {
            px.write_be(&mut be);
        }
        self.patch_bytes(&meta, start, shape, &be)
    }

    /// Write a rectangular pixel patch in physical units.
    ///
    /// The `pixels` argument holds `shape.iter().product()` values in
    /// the units the header describes, the ones
    /// [`ImageHdu::read_physical`](crate::ImageHdu::read_physical)
    /// returns. This inverts `BZERO`, `BSCALE` and `BLANK` before it
    /// writes, so a caller reads and writes the same numbers whatever
    /// the file stores. An integer image rounds to the nearest stored
    /// value, and a `NaN` writes the `BLANK` sentinel.
    ///
    /// [`Self::write_image_subarray`] is the raw counterpart, for a
    /// caller holding stored values.
    ///
    /// # Crash safety
    ///
    /// The conditions of [`Self::write_image_subarray`].
    ///
    /// # Errors
    ///
    /// The conditions of [`Self::write_image_subarray`], and
    /// [`FitsError::Data`] when a value does not fit the stored type,
    /// when a `NaN` reaches an integer image that declares no `BLANK`,
    /// or when `BSCALE` is zero.
    pub fn write_image_subarray_physical(
        &mut self,
        i: usize,
        start: &[u64],
        shape: &[u64],
        pixels: &[f64],
    ) -> Result<()> {
        let meta = self.patchable_image(i)?;
        Self::validate_patch(&meta, start, shape, pixels.len())?;
        if shape.contains(&0) {
            return Ok(());
        }
        let be = encode_physical(&meta, pixels)?;
        self.patch_bytes(&meta, start, shape, &be)
    }

    /// Write `be`, the big-endian encoding of a patch, into the file.
    ///
    /// The `be` argument holds the patch pixels in order. Rows run
    /// along `NAXIS1`, so this splits `be` into one row per write.
    fn patch_bytes(
        &mut self,
        meta: &ImageMeta,
        start: &[u64],
        shape: &[u64],
        be: &[u8],
    ) -> Result<()> {
        use crate::hdu::subarray::{checked_strides, next_subarray_index};

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

        // ---- Pass 2: issue the data pwrites, one row at a time.
        let total_bytes = row_offsets
            .len()
            .checked_mul(row_bytes)
            .ok_or_else(|| FitsError::Data("total byte count overflows usize".to_string()))?;
        debug_assert_eq!(
            be.len(),
            total_bytes,
            "encoded patch buffer must equal rows * row_bytes"
        );

        for (i, &off) in row_offsets.iter().enumerate() {
            let chunk = &be[i * row_bytes..(i + 1) * row_bytes];
            pwrite_all(&self.file, off, chunk)?;
        }
        Ok(())
    }

    /// The cached layout of image HDU `i`, or the reason there is
    /// none.
    fn patchable_image(&self, i: usize) -> Result<ImageMeta> {
        match self.images.get(i) {
            Some(Patchable::Image(m)) => Ok(m.clone()),
            #[cfg(feature = "compression")]
            Some(Patchable::Compressed) => Err(FitsError::Data(format!(
                "FitsUpdater: HDU {i} is a tile-compressed image, which takes \
                 no in-place patch: a rewritten tile does not keep its byte \
                 length. Decompress the file first, with \
                 `FitsFile::write_decompressed`"
            ))),
            _ => Err(FitsError::Data(format!(
                "FitsUpdater: HDU {i} is not an image (or out of range)"
            ))),
        }
    }

    /// Check a patch region and its element count against the HDU.
    fn validate_patch(
        meta: &ImageMeta,
        start: &[u64],
        shape: &[u64],
        n_pixels: usize,
    ) -> Result<()> {
        crate::hdu::subarray::validate_subarray_shape(&meta.axes, start, shape)?;
        if shape.contains(&0) {
            return Ok(());
        }
        let expected = crate::data::encoding::shape_product(shape)?;
        if n_pixels as u64 != expected {
            return Err(FitsError::Data(format!(
                "pixels.len() = {n_pixels} but shape implies {expected} elements",
            )));
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
