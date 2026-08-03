//! [`FitsFile`], the top-level reader (Standard Sec.3.4).
//!
//! # Purpose
//!
//! [`FitsFile`] owns the bytes of one FITS file. It parses every
//! header when it opens the file, and it hands out a borrowed view of
//! each HDU.
//!
//! Open a file with [`FitsFile::open`]. Reach one HDU by index with
//! [`FitsFile::hdu`] or by name with [`FitsFile::hdu_by_name`]. Walk
//! every HDU with [`FitsFile::iter`].
//!
//! # Design constraints
//!
//! Four facts explain the shape of this module.
//!
//! First, a data section loads on first use. [`FitsFile::open`] reads
//! the headers and records the byte span of each data section. It
//! reads no pixel bytes and no table bytes. The first call to
//! [`FitsFile::hdu`] reads that one data section and caches it until
//! the [`FitsFile`] drops. Opening a 50 GB file therefore costs the
//! headers alone, and reading HDU 3 costs HDU 3 alone.
//!
//! Second, [`FitsFile::from_bytes`] holds the whole buffer in memory.
//! It adds no further cache, because the bytes are already present.
//!
//! Third, parsing is lenient by default. [`FitsFile::open`] accepts
//! the deviations that real files contain. [`FitsFile::open_with`]
//! takes a `lenient` flag, and `false` requires strict conformance.
//! The flag is stored on the [`FitsFile`], so each later re-parse of a
//! header applies the same rule.
//!
//! Fourth, a trailing region that is not an HDU ends the scan. Some
//! capture programs append zero bytes or vendor metadata after the
//! last HDU. The scan stops at an all-zero region, and at any later
//! region that does not begin with an `XTENSION` card. Neither case is
//! an error.

use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Seek, SeekFrom};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::OnceLock;

#[cfg(all(unix, not(target_arch = "wasm32")))]
use std::os::unix::fs::FileExt;
#[cfg(all(windows, not(target_arch = "wasm32")))]
use std::os::windows::fs::FileExt;

use crate::data::encoding::Bitpix;
use crate::error::{FitsError, Result};
use crate::hdu::ascii_table::AsciiTableHdu;
use crate::hdu::bintable::BinTableHdu;
use crate::hdu::image::ImageHdu;
use crate::hdu::kind::{ConformingHdu, Hdu};
use crate::header::Header;
use crate::header::value::Value;
use crate::io::block::pad_to_block;
// `BLOCK_SIZE` is only referenced by the on-disk reader, which is excluded on wasm.
#[cfg(not(target_arch = "wasm32"))]
use crate::io::block::BLOCK_SIZE;
use crate::io::source::ByteSource;

/// Top-level FITS file. `Send` and `Sync`: the lazy data cache is a
/// `OnceLock`, so `&FitsFile` can be shared across threads.
///
/// # Examples
///
/// ```
/// # use fitsy::{FitsWriter, ImageBuilder};
/// # let (h, d) = ImageBuilder::new(vec![4_u64, 3], vec![7_i16; 12])?
/// #     .primary(true)
/// #     .build()?;
/// # let mut buf: Vec<u8> = Vec::new();
/// # FitsWriter::new(&mut buf).write_hdu(&h, &d)?;
/// use fitsy::{FitsFile, Hdu};
///
/// let file = FitsFile::from_bytes(buf)?;
/// assert_eq!(file.len(), 1);
///
/// let Hdu::Image(img) = file.hdu(0)? else {
///     panic!("HDU 0 is not an image");
/// };
/// assert_eq!(img.axes(), &[4, 3]);
/// # Ok::<(), fitsy::FitsError>(())
/// ```
#[derive(Debug)]
pub struct FitsFile {
    backing: Backing,
    /// Byte spans for each HDU.
    hdu_spans: Vec<HduSpan>,
    /// Owned header bytes (already padded through the END card) for
    /// each HDU. Headers are small and always loaded eagerly so that
    /// [`hdu`] / [`header_inherited`] / iteration over HDU metadata
    /// never incur disk I/O.
    header_bytes: Vec<Vec<u8>>,
    /// Lazy per-HDU data section cache. Populated on the first
    /// access via `pread`. Empty for in-memory backings (the bytes
    /// live in `Backing::InMemory` instead) -- so on wasm, where the
    /// on-disk backing is excluded, it is written but never read.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    data_cache: Vec<OnceLock<Vec<u8>>>,
    /// Map `EXTNAME` (trimmed) -> sorted list of HDU indices that
    /// declare it. Built once at open time so [`hdu_by_name`] is
    /// O(log n + k) instead of O(n).
    extname_index: BTreeMap<String, Vec<usize>>,
    /// Whether headers were opened in lenient mode. Retained so the
    /// per-HDU re-parses done by [`hdu`], [`parsed_header`], etc. apply
    /// the same leniency as the initial open.
    lenient: bool,
}

#[derive(Debug)]
enum Backing {
    /// Whole-file in-memory buffer (used by `from_bytes`).
    InMemory(ByteSource),
    /// On-disk file; data sections are loaded lazily.
    #[cfg(not(target_arch = "wasm32"))]
    OnDisk(File),
}

#[derive(Debug, Clone, Copy)]
struct HduSpan {
    header_end: u64,
    data_logical_len: u64,
}

impl FitsFile {
    /// Open `path` and parse its HDU headers.
    ///
    /// This function reads no data section. Each one loads on first
    /// access. It parses headers leniently, so a real-world file
    /// loads. Call [`FitsFile::open_with`] with `lenient = false` to
    /// require strict conformance.
    ///
    /// A `path` that names a gzipped file is inflated whole into
    /// memory, because a compressed stream has no seekable layout.
    /// This needs the `compression` feature.
    ///
    /// # Errors
    ///
    /// - [`FitsError::Io`] when `path` cannot be opened or read.
    /// - [`FitsError::Block`], [`FitsError::Card`] or
    ///   [`FitsError::Value`] when a header violates the block, card
    ///   or value structure of the format.
    /// - [`FitsError::MissingMandatory`] when the primary header omits
    ///   `SIMPLE`, or an extension header omits `XTENSION`.
    /// - [`FitsError::Header`] when the file holds no HDU.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fitsy::{FitsWriter, ImageBuilder};
    /// # let path = std::env::temp_dir().join("fitsy_doc_open.fits");
    /// # let (h, d) = ImageBuilder::new(vec![2_u64, 2], vec![0.0_f32; 4])?
    /// #     .primary(true)
    /// #     .build()?;
    /// # let mut out = std::fs::File::create(&path)?;
    /// # FitsWriter::new(&mut out).write_hdu(&h, &d)?;
    /// use fitsy::FitsFile;
    ///
    /// let f = FitsFile::open(&path)?;
    /// assert_eq!(f.len(), 1);
    /// # std::fs::remove_file(&path)?;
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, true)
    }

    /// Open `path` with explicit control over lenient parsing.
    ///
    /// A `lenient` value of `true` is what [`FitsFile::open`] uses. It
    /// tolerates each of these:
    ///
    /// - `SIMPLE = F` in the primary header.
    /// - A non-ASCII byte in a string value. The parser replaces it
    ///   with a space.
    /// - A lower-case keyword. The parser folds it to upper case.
    /// - A value that matches no standard type. The parser keeps it as
    ///   [`crate::header::Value::Unparsed`].
    /// - A stray byte after the `END` card.
    /// - A lower-case `end` card.
    /// - A broken `CONTINUE` chain.
    ///
    /// A `lenient` value of `false` rejects each of them. Three rules
    /// hold under either value:
    ///
    /// - The header must contain an `END` card.
    /// - The file must align to the 2880-byte block.
    /// - The data section must have the size the header declares.
    ///
    /// The flag is stored on the [`FitsFile`], so each per-HDU
    /// re-parse applies the same rule.
    ///
    /// # Errors
    ///
    /// The same conditions as [`FitsFile::open`]. A `lenient` value of
    /// `false` adds [`FitsError::Card`], [`FitsError::Value`] and
    /// [`FitsError::Header`] for each deviation listed above.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_with(path: impl AsRef<Path>, lenient: bool) -> Result<Self> {
        let path = path.as_ref();
        // gzip files cannot be `pread`'d in place, so detect the
        // magic bytes and fall back to a full read + decompress
        // through `ByteSource::from_vec`.
        #[cfg(feature = "compression")]
        {
            let mut probe = [0_u8; 2];
            let mut f = File::open(path)?;
            let n = f.read(&mut probe)?;
            if n == 2 && probe == [0x1f, 0x8b] {
                use std::io::Read;
                f.seek(SeekFrom::Start(0))?;
                let mut buf = Vec::new();
                f.read_to_end(&mut buf)?;
                return Self::from_source(ByteSource::from_vec(buf)?, lenient);
            }
        }
        let file = File::open(path)?;
        Self::from_file(file, lenient)
    }

    /// Build a `FitsFile` from an in-memory buffer.
    ///
    /// The `FitsFile` holds the whole buffer for its lifetime. It
    /// parses headers leniently, as [`FitsFile::open`] does. Call
    /// [`FitsFile::from_bytes_with`] with `lenient = false` to require
    /// strict conformance.
    ///
    /// # Errors
    ///
    /// The same conditions as [`FitsFile::open`], except that no
    /// [`FitsError::Io`] arises from opening a path. A gzipped `buf`
    /// still yields [`FitsError::Io`] when it fails to inflate.
    pub fn from_bytes(buf: Vec<u8>) -> Result<Self> {
        Self::from_source(ByteSource::from_vec(buf)?, true)
    }

    /// Build a `FitsFile` from an in-memory buffer, with explicit
    /// control over lenient parsing.
    ///
    /// The `buf` argument holds the whole file, and the `FitsFile`
    /// takes ownership of it. [`FitsFile::open_with`] lists what the
    /// `lenient` flag tolerates.
    ///
    /// # Errors
    ///
    /// The same conditions as [`FitsFile::from_bytes`].
    pub fn from_bytes_with(buf: Vec<u8>, lenient: bool) -> Result<Self> {
        Self::from_source(ByteSource::from_vec(buf)?, lenient)
    }

    fn from_source(src: ByteSource, lenient: bool) -> Result<Self> {
        let bytes = src.as_bytes();
        let total = bytes.len() as u64;
        let mut hdu_spans = Vec::new();
        let mut header_bytes_per_hdu = Vec::new();
        let mut cursor: u64 = 0;
        let mut is_first = true;

        while cursor < total {
            // Several CCD capture programs append a run of zero bytes after
            // the final HDU (sometimes not even padded to the 2880-byte block
            // size). An all-zero region can never be a valid HDU -- a header
            // must begin with a keyword -- so treat an all-zero tail as
            // padding and stop rather than failing the whole read.
            if bytes[cursor as usize..total as usize]
                .iter()
                .all(|&b| b == 0)
            {
                break;
            }
            // Some capture programs (e.g. ZWO ASI Studio) append vendor
            // metadata or thumbnail blobs after the last HDU's data section.
            // A conforming extension must begin with an XTENSION card
            // (Standard Sec.7.1.3); if the bytes here don't, treat them as
            // trailing junk and stop.
            if !is_first && !looks_like_extension_start(&bytes[cursor as usize..total as usize]) {
                break;
            }
            let header_start = cursor;
            let (header, header_blocks_bytes) = Header::parse_with(bytes, cursor, lenient)?;
            let header_end = header_start + header_blocks_bytes;

            if is_first {
                require_simple_t(&header, lenient)?;
            } else {
                require_xtension(&header)?;
            }

            let data_logical_len = data_section_size(&header)?;
            let data_padded = pad_to_block(data_logical_len);
            // Saturate: a wrapped sum would pass the check below.
            let data_end = header_end.saturating_add(data_padded);
            if data_end > total {
                return Err(FitsError::Block {
                    offset: header_end,
                    msg: format!(
                        "HDU data section requires {data_padded} bytes but only {} remain",
                        total - header_end
                    ),
                });
            }

            header_bytes_per_hdu.push(bytes[header_start as usize..header_end as usize].to_vec());
            hdu_spans.push(HduSpan {
                header_end,
                data_logical_len,
            });

            cursor = data_end;
            is_first = false;
        }

        Self::finish_open(
            Backing::InMemory(src),
            hdu_spans,
            header_bytes_per_hdu,
            lenient,
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn from_file(mut file: File, lenient: bool) -> Result<Self> {
        let total = file.metadata()?.len();
        let mut hdu_spans = Vec::new();
        let mut header_bytes_per_hdu: Vec<Vec<u8>> = Vec::new();
        let mut cursor: u64 = 0;
        let mut is_first = true;

        while cursor < total {
            // See `from_source`: tolerate an all-zero trailing region emitted
            // by some non-standard writers instead of failing the read.
            if remaining_is_zero(&file, cursor, total)? {
                break;
            }
            // See `from_source`: also tolerate non-zero trailing junk that
            // is not a conforming extension start.
            if !is_first && !next_is_extension(&file, cursor, total)? {
                break;
            }
            let header_start = cursor;
            let header_buf = read_header_blocks(&mut file, cursor, total, lenient)?;
            let (header, header_blocks_bytes) = Header::parse_with(&header_buf, 0, lenient)?;
            let header_end = header_start + header_blocks_bytes;
            // Truncate the header buffer to the exact block-padded
            // length the parser consumed (it may have read one extra
            // block while looking for END).
            let mut header_owned = header_buf;
            header_owned.truncate(header_blocks_bytes as usize);

            if is_first {
                require_simple_t(&header, lenient)?;
            } else {
                require_xtension(&header)?;
            }

            let data_logical_len = data_section_size(&header)?;
            let data_padded = pad_to_block(data_logical_len);
            // Saturate: a wrapped sum would pass the check below.
            let data_end = header_end.saturating_add(data_padded);
            if data_end > total {
                return Err(FitsError::Block {
                    offset: header_end,
                    msg: format!(
                        "HDU data section requires {data_padded} bytes but only {} remain",
                        total - header_end
                    ),
                });
            }

            header_bytes_per_hdu.push(header_owned);
            hdu_spans.push(HduSpan {
                header_end,
                data_logical_len,
            });

            cursor = data_end;
            is_first = false;
        }

        Self::finish_open(
            Backing::OnDisk(file),
            hdu_spans,
            header_bytes_per_hdu,
            lenient,
        )
    }

    fn finish_open(
        backing: Backing,
        hdu_spans: Vec<HduSpan>,
        header_bytes: Vec<Vec<u8>>,
        lenient: bool,
    ) -> Result<Self> {
        if hdu_spans.is_empty() {
            return Err(FitsError::Header("file contains no HDU".into()));
        }
        let mut extname_index: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, hb) in header_bytes.iter().enumerate() {
            if let Ok((header, _)) = Header::parse_with(hb, 0, lenient)
                && let Some(Value::String(s)) = header.first("EXTNAME")
            {
                extname_index
                    .entry(s.trim_end().to_string())
                    .or_default()
                    .push(i);
            }
        }
        let n = hdu_spans.len();
        Ok(Self {
            backing,
            hdu_spans,
            header_bytes,
            data_cache: (0..n).map(|_| OnceLock::new()).collect(),
            extname_index,
            lenient,
        })
    }

    /// Borrow the unpadded data bytes for HDU `i`. For on-disk
    /// backings this triggers a `pread` on first call and caches
    /// the result.
    fn data_bytes(&self, i: usize) -> Result<&[u8]> {
        let span = &self.hdu_spans[i];
        let logical = span.data_logical_len as usize;
        match &self.backing {
            Backing::InMemory(src) => {
                let start = span.header_end as usize;
                Ok(&src.as_bytes()[start..start + logical])
            }
            #[cfg(not(target_arch = "wasm32"))]
            Backing::OnDisk(file) => {
                let cell = &self.data_cache[i];
                if let Some(buf) = cell.get() {
                    return Ok(&buf[..logical]);
                }
                // Read padded bytes (so other accessors that need
                // the padded view -- checksum verification,
                // hdu_raw_padded -- can reuse the same cache).
                let padded = pad_to_block(span.data_logical_len) as usize;
                let mut buf = vec![0_u8; padded];
                pread_exact(file, span.header_end, &mut buf)?;
                let _ = cell.set(buf);
                Ok(&cell.get().expect("OnceLock just set")[..logical])
            }
        }
    }

    /// Borrow the data section padded out to the next 2880-byte
    /// boundary. Same caching strategy as [`data_bytes`].
    #[cfg(feature = "python")]
    fn data_padded_bytes(&self, i: usize) -> Result<&[u8]> {
        let span = &self.hdu_spans[i];
        let padded = pad_to_block(span.data_logical_len) as usize;
        match &self.backing {
            Backing::InMemory(src) => {
                let start = span.header_end as usize;
                Ok(&src.as_bytes()[start..start + padded])
            }
            #[cfg(not(target_arch = "wasm32"))]
            Backing::OnDisk(_) => {
                // `data_bytes` populates the cache with padded bytes.
                let _ = self.data_bytes(i)?;
                Ok(&self.data_cache[i].get().expect("just populated")[..padded])
            }
        }
    }

    /// Read the unpadded data section for HDU `i` into an owned
    /// `Vec<u8>`. This bypasses the per-HDU cache. Use it for bytes
    /// that the caller consumes once and then drops.
    ///
    /// An in-memory backing still copies, so that the return type
    /// stays uniform. Call [`FitsFile::hdu`] there to avoid the copy.
    #[cfg(feature = "python")]
    pub(crate) fn read_data_owned(&self, i: usize) -> Result<Vec<u8>> {
        let span = self.hdu_spans.get(i).ok_or_else(|| {
            FitsError::Header(format!("HDU index {i} out of range (len = {})", self.len()))
        })?;
        let logical = span.data_logical_len as usize;
        match &self.backing {
            Backing::InMemory(src) => {
                let start = span.header_end as usize;
                Ok(src.as_bytes()[start..start + logical].to_vec())
            }
            #[cfg(not(target_arch = "wasm32"))]
            Backing::OnDisk(file) => {
                let padded = pad_to_block(span.data_logical_len) as usize;
                let mut buf = vec![0_u8; padded];
                pread_exact(file, span.header_end, &mut buf)?;
                buf.truncate(logical);
                Ok(buf)
            }
        }
    }

    /// Read the unpadded data section for HDU `i` into `dst`. The
    /// length of `dst` must equal the logical data length. This
    /// bypasses the cache.
    ///
    /// A caller that already owns a correctly sized buffer avoids the
    /// intermediate `Vec` that
    /// [`read_data_owned`](Self::read_data_owned) allocates. That
    /// halves both the copy count and the peak memory. The Python
    /// reader passes the numpy allocation here for that reason.
    #[cfg(feature = "python")]
    pub(crate) fn read_data_into(&self, i: usize, dst: &mut [u8]) -> Result<()> {
        let logical = self
            .hdu_spans
            .get(i)
            .ok_or_else(|| {
                FitsError::Header(format!("HDU index {i} out of range (len = {})", self.len()))
            })?
            .data_logical_len as usize;
        if dst.len() != logical {
            return Err(FitsError::Data(format!(
                "read_data_into: destination is {} bytes, data section is {logical}",
                dst.len()
            )));
        }
        self.read_data_range_into(i, 0, dst)
    }

    /// Fill `dst` from the data section of HDU `i`, starting `offset`
    /// bytes in. Lets a caller stream the section through a small fixed
    /// buffer instead of materializing it whole.
    #[cfg(feature = "python")]
    pub(crate) fn read_data_range_into(&self, i: usize, offset: u64, dst: &mut [u8]) -> Result<()> {
        let span = self.hdu_spans.get(i).ok_or_else(|| {
            FitsError::Header(format!("HDU index {i} out of range (len = {})", self.len()))
        })?;
        if offset
            .checked_add(dst.len() as u64)
            .is_none_or(|end| end > span.data_logical_len)
        {
            return Err(FitsError::Data(format!(
                "read_data_range_into: [{offset}, +{}) escapes the {}-byte data section",
                dst.len(),
                span.data_logical_len
            )));
        }
        let start = span.header_end.checked_add(offset).ok_or_else(|| {
            FitsError::Data("read_data_range_into: offset overflows the file".into())
        })?;
        match &self.backing {
            Backing::InMemory(src) => {
                let s = start as usize;
                dst.copy_from_slice(&src.as_bytes()[s..s + dst.len()]);
                Ok(())
            }
            #[cfg(not(target_arch = "wasm32"))]
            Backing::OnDisk(file) => pread_exact(file, start, dst),
        }
    }

    /// Read a rectangular sub-region of image HDU `i` from disk into a
    /// big-endian buffer. This does not touch the cache.
    ///
    /// The `axes`, `start` and `shape` arguments are in FITS order,
    /// where `NAXIS1` varies fastest. The result holds
    /// `prod(shape) * bitpix.byte_size()` big-endian bytes. The caller
    /// performs the byte-order change.
    ///
    /// This backs the Python `section[a:b]` read. A small tile out of
    /// a large image therefore costs only that tile.
    #[cfg(feature = "python")]
    pub(crate) fn read_image_subarray_be(
        &self,
        i: usize,
        axes: &[u64],
        start: &[u64],
        shape: &[u64],
        bitpix: Bitpix,
    ) -> Result<Vec<u8>> {
        use crate::hdu::subarray::{checked_strides, next_subarray_index, validate_subarray_shape};

        let span = self.hdu_spans.get(i).ok_or_else(|| {
            FitsError::Header(format!("HDU index {i} out of range (len = {})", self.len()))
        })?;
        validate_subarray_shape(axes, start, shape)?;
        let bsize = bitpix.byte_size();
        let total_elems: u64 = shape
            .iter()
            .try_fold(1_u64, |acc, &n| acc.checked_mul(n))
            .ok_or_else(|| FitsError::Data("shape product overflows u64".into()))?;
        let total_bytes = (total_elems as usize)
            .checked_mul(bsize)
            .ok_or_else(|| FitsError::Data("total bytes overflows usize".into()))?;
        let mut out = vec![0_u8; total_bytes];
        if total_elems == 0 {
            return Ok(out);
        }

        let strides = checked_strides(axes)?;

        let n1 = shape[0];
        let row_elems = n1 as usize;
        let row_bytes = row_elems * bsize;
        let data_offset = span.header_end;
        let mut idx = vec![0_u64; axes.len()];
        let mut dst_row_start: usize = 0;
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
                .and_then(|v| data_offset.checked_add(v))
                .ok_or_else(|| FitsError::Data("byte offset overflows u64".into()))?;

            let dst = &mut out[dst_row_start..dst_row_start + row_bytes];
            match &self.backing {
                Backing::InMemory(src) => {
                    let src_bytes = src.as_bytes();
                    let start_byte = byte_off as usize;
                    let end_byte = start_byte + row_bytes;
                    if end_byte > src_bytes.len() {
                        return Err(FitsError::Data(format!(
                            "row at byte {byte_off}..{end_byte} exceeds buffer length {}",
                            src_bytes.len()
                        )));
                    }
                    dst.copy_from_slice(&src_bytes[start_byte..end_byte]);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backing::OnDisk(file) => {
                    pread_exact(file, byte_off, dst)?;
                }
            }

            dst_row_start += row_bytes;
            if !next_subarray_index(&mut idx, shape) {
                break;
            }
        }
        Ok(out)
    }

    /// Number of HDUs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hdu_spans.len()
    }

    #[must_use]
    /// True when the file holds no HDUs.
    pub fn is_empty(&self) -> bool {
        self.hdu_spans.is_empty()
    }

    /// Parse and return the header for HDU `i`.
    ///
    /// This function reads no data section and caches nothing. It
    /// costs less than [`hdu`](Self::hdu) when the caller needs only
    /// header-level information, such as the axes, the `BITPIX` value,
    /// or the kind of the HDU.
    ///
    /// # Errors
    ///
    /// - [`FitsError::Header`] when `i` is not a valid HDU index.
    /// - [`FitsError::Block`], [`FitsError::Card`] or
    ///   [`FitsError::Value`] when the header fails to parse.
    pub fn parsed_header(&self, i: usize) -> Result<Header> {
        let _ = self.hdu_spans.get(i).ok_or_else(|| {
            FitsError::Header(format!("HDU index {i} out of range (len = {})", self.len()))
        })?;
        let (header, _) = Header::parse_with(&self.header_bytes[i], 0, self.lenient)?;
        Ok(header)
    }

    /// Total number of bytes occupied by all HDUs (each header and
    /// data section padded to the 2880-byte block boundary). This is
    /// the offset where new HDUs would be written by
    /// [`FitsAppender`](crate::FitsAppender).
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        match self.hdu_spans.last() {
            Some(span) => span.header_end + pad_to_block(span.data_logical_len),
            None => 0,
        }
    }

    /// Byte offset (from file start) of the data section for HDU `i`.
    /// Returns `None` if `i` is out of range. Used by
    /// [`FitsUpdater`](crate::FitsUpdater) to locate pixel bytes for
    /// in-place patch writes.
    #[must_use]
    pub fn data_offset(&self, i: usize) -> Option<u64> {
        self.hdu_spans.get(i).map(|s| s.header_end)
    }

    /// Length in bytes of the unpadded data section for HDU `i`.
    /// Returns `None` if `i` is out of range.
    #[must_use]
    pub fn data_logical_len(&self, i: usize) -> Option<u64> {
        self.hdu_spans.get(i).map(|s| s.data_logical_len)
    }

    /// Raw header + data bytes for HDU `i`, padded to the 2880-byte
    /// FITS block boundary. Suitable for streaming an untouched HDU
    /// directly into a writer when persisting modifications without
    /// re-encoding. Returns `None` if `i` is out of range. Loads the
    /// data section from disk if it has not been read yet.
    #[cfg(feature = "python")]
    pub(crate) fn hdu_raw_padded(&self, i: usize) -> Result<Option<Vec<u8>>> {
        if i >= self.hdu_spans.len() {
            return Ok(None);
        }
        let header = &self.header_bytes[i];
        let data = self.data_padded_bytes(i)?;
        let mut out = Vec::with_capacity(header.len() + data.len());
        out.extend_from_slice(header);
        out.extend_from_slice(data);
        Ok(Some(out))
    }

    /// Borrow the HDU at index `i`. Index 0 is the primary HDU.
    ///
    /// This call reads the data section of HDU `i` if no earlier call
    /// has read it, and caches those bytes.
    ///
    /// The returned variant follows the header. Index 0 gives
    /// [`Hdu::RandomGroups`] when the header declares `GROUPS = T`
    /// with `NAXIS1 = 0`, and [`Hdu::Image`] otherwise. A later index
    /// dispatches on `XTENSION`: `IMAGE` gives [`Hdu::Image`], `TABLE`
    /// gives [`Hdu::AsciiTable`], and `BINTABLE` gives
    /// [`Hdu::BinTable`]. A `BINTABLE` that also declares `ZIMAGE = T`
    /// gives [`Hdu::CompressedImage`] under the `compression` feature.
    /// Any other `XTENSION` value gives [`Hdu::Conforming`].
    ///
    /// # Errors
    ///
    /// - [`FitsError::Header`] when `i` is not a valid HDU index.
    /// - [`FitsError::Io`] when the data section fails to read.
    /// - [`FitsError::MissingMandatory`] when an extension header
    ///   omits `XTENSION`.
    /// - [`FitsError::Data`] or [`FitsError::Header`] when the HDU
    ///   constructor rejects the header. A `BINTABLE` with a `TFORM`
    ///   value that does not parse is one such case.
    pub fn hdu(&self, i: usize) -> Result<Hdu<'_>> {
        let _ = self.hdu_spans.get(i).ok_or_else(|| {
            FitsError::Header(format!("HDU index {i} out of range (len = {})", self.len()))
        })?;
        let (header, _) = Header::parse_with(&self.header_bytes[i], 0, self.lenient)?;
        let data = self.data_bytes(i)?;

        if i == 0 {
            // Random Groups primary HDU (Standard Sec.6): NAXIS1 = 0,
            // NAXIS >= 2, GROUPS = T.
            if is_random_groups(&header) {
                return Ok(Hdu::RandomGroups(
                    crate::hdu::random_groups::RandomGroupsHdu::new(header, data)?,
                ));
            }
            return Ok(Hdu::Image(ImageHdu::new(header, data)?));
        }

        let xtension = match header.first("XTENSION") {
            Some(Value::String(s)) => s.clone(),
            _ => {
                return Err(FitsError::MissingMandatory {
                    keyword: "XTENSION".into(),
                });
            }
        };

        if xtension == "IMAGE" {
            Ok(Hdu::Image(ImageHdu::new(header, data)?))
        } else if xtension == "TABLE" {
            Ok(Hdu::AsciiTable(AsciiTableHdu::new(header, data)?))
        } else if xtension == "BINTABLE" {
            let bt = BinTableHdu::new(header, data)?;
            #[cfg(feature = "compression")]
            {
                if matches!(bt.header().first("ZIMAGE"), Some(Value::Logical(true))) {
                    return Ok(Hdu::CompressedImage(
                        crate::compression::CompressedImageHdu::from_bintable(bt)?,
                    ));
                }
            }
            Ok(Hdu::BinTable(bt))
        } else {
            Ok(Hdu::Conforming(ConformingHdu::new(header, data, xtension)))
        }
    }

    /// Iterate every HDU in file order.
    ///
    /// Each item wraps a failure in [`FitsError::InHdu`], which carries
    /// the index of the HDU that failed. The inner error is the one
    /// that [`FitsFile::hdu`] reports.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fitsy::{FitsWriter, ImageBuilder};
    /// # let path = std::env::temp_dir().join("fitsy_doc_iter.fits");
    /// # let (h, d) = ImageBuilder::new(vec![2_u64, 2], vec![0.0_f32; 4])?
    /// #     .primary(true)
    /// #     .build()?;
    /// # let mut out = std::fs::File::create(&path)?;
    /// # FitsWriter::new(&mut out).write_hdu(&h, &d)?;
    /// use fitsy::{FitsFile, Hdu};
    ///
    /// let f = FitsFile::open(&path)?;
    /// for (i, hdu) in f.iter().enumerate() {
    ///     match hdu? {
    ///         Hdu::Image(img) => println!("#{i}: image {:?}", img.axes()),
    ///         Hdu::BinTable(t) => println!("#{i}: table {}", t.n_rows()),
    ///         _ => println!("#{i}: other"),
    ///     }
    /// }
    /// # std::fs::remove_file(&path)?;
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = Result<Hdu<'_>>> {
        (0..self.len()).map(move |i| {
            self.hdu(i).map_err(|e| FitsError::InHdu {
                index: i,
                source: Box::new(e),
            })
        })
    }

    /// Return the header of HDU `i`, with primary-HDU keywords merged
    /// in when the extension declares `INHERIT = T`.
    ///
    /// The merge adds a value card from the primary header only when
    /// the extension header does not already contain that keyword. It
    /// skips every commentary card, and it skips every structural
    /// keyword, such as `BITPIX`, `NAXIS` and `TFIELDS`. An index of
    /// 0, or an extension without `INHERIT = T`, returns the header of
    /// that HDU unchanged.
    ///
    /// # Errors
    ///
    /// The same conditions as [`FitsFile::parsed_header`].
    pub fn header_inherited(&self, i: usize) -> Result<Header> {
        let _ = self.hdu_spans.get(i).ok_or_else(|| {
            FitsError::Header(format!("HDU index {i} out of range (len = {})", self.len()))
        })?;
        let (mut header, _) = Header::parse_with(&self.header_bytes[i], 0, self.lenient)?;
        if i == 0 {
            return Ok(header);
        }
        if !matches!(header.first("INHERIT"), Some(Value::Logical(true))) {
            return Ok(header);
        }
        let (primary, _) = Header::parse_with(&self.header_bytes[0], 0, self.lenient)?;
        header.merge_inherited(&primary);
        Ok(header)
    }

    /// Parse the WCS of HDU `i` for alternate descriptor `alt`, after
    /// merging inherited primary-HDU keywords.
    ///
    /// Pass `' '` for `alt` to select the primary description. The
    /// result is `Ok(None)` when the merged header carries no WCS for
    /// that descriptor. The call is equivalent to
    /// `Wcs::from_header(&self.header_inherited(i)?, alt)`.
    ///
    /// This function does not resolve a `-TAB` lookup table. Use
    /// [`FitsFile::wcs`] when the description needs one.
    ///
    /// # Errors
    ///
    /// - The conditions of [`FitsFile::header_inherited`].
    /// - [`FitsError::Wcs`] when the header declares a WCS that the
    ///   parser rejects, such as an unknown projection code.
    pub fn wcs_inherited(&self, i: usize, alt: char) -> Result<Option<crate::wcs::Wcs>> {
        let header = self.header_inherited(i)?;
        crate::wcs::Wcs::from_header(&header, alt)
    }

    /// Look up an extension by `EXTNAME` and return the first match.
    ///
    /// Standard Sec.4.4.2.6 makes `EXTNAME` case-sensitive once
    /// trailing spaces are trimmed, so `name` must match exactly. A
    /// `ver` of `Some(v)` also requires `EXTVER = v`. An extension
    /// without an `EXTVER` card counts as version 1. A `ver` of `None`
    /// accepts any version.
    ///
    /// # Errors
    ///
    /// - [`FitsError::Header`] when no HDU matches `name`, or when no
    ///   HDU matches both `name` and `ver`.
    /// - The conditions of [`FitsFile::hdu`] for the matched HDU.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fitsy::{BinFieldKind, BinTableBuilder, FitsWriter, ImageBuilder};
    /// # let path = std::env::temp_dir().join("fitsy_doc_by_name.fits");
    /// # let empty: Vec<f32> = Vec::new();
    /// # let (ph, pd) = ImageBuilder::new(Vec::<u64>::new(), empty)?
    /// #     .primary(true)
    /// #     .build()?;
    /// # let mut b = BinTableBuilder::new();
    /// # b.add_column("TIME", BinFieldKind::F64, 1, None, None)?;
    /// # b.extname("EVENTS");
    /// # let mut rows = Vec::new();
    /// # for v in [1.0_f64, 2.0, 3.0] {
    /// #     rows.extend_from_slice(&v.to_be_bytes());
    /// # }
    /// # let (th, td) = b.build(3, rows)?;
    /// # let mut w = FitsWriter::new(std::fs::File::create(&path)?);
    /// # w.write_hdu(&ph, &pd)?;
    /// # w.write_hdu(&th, &td)?;
    /// # w.finish()?;
    /// use fitsy::{FitsFile, Hdu};
    ///
    /// let f = FitsFile::open(&path)?;
    /// let Hdu::BinTable(events) = f.hdu_by_name("EVENTS", None)? else {
    ///     panic!("the EVENTS extension is not a binary table");
    /// };
    /// assert_eq!(events.n_rows(), 3);
    /// # std::fs::remove_file(&path)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn hdu_by_name(&self, name: &str, ver: Option<i64>) -> Result<Hdu<'_>> {
        let candidates: &[usize] = self.extname_index.get(name).map_or(&[], Vec::as_slice);
        for &i in candidates {
            if let Some(want) = ver {
                let (header, _) = Header::parse_with(&self.header_bytes[i], 0, self.lenient)?;
                let have = match header.first("EXTVER") {
                    Some(Value::Integer(v)) => *v,
                    _ => 1,
                };
                if have != want {
                    continue;
                }
            }
            return self.hdu(i);
        }
        Err(FitsError::Header(match ver {
            Some(v) => format!("no HDU with EXTNAME = `{name}` and EXTVER = {v}"),
            None => format!("no HDU with EXTNAME = `{name}`"),
        }))
    }

    /// Iterator that transparently decompresses tile-compressed image
    /// HDUs. Each `Hdu::CompressedImage` is materialized as an
    /// [`OwnedImage`](crate::OwnedImage); all other HDUs are yielded
    /// as `Decompressed::Hdu(_)` unchanged.
    #[cfg(feature = "compression")]
    pub fn iter_decompressed(&self) -> impl Iterator<Item = Result<Decompressed<'_>>> {
        self.iter().map(|r| {
            r.and_then(|h| match h {
                Hdu::CompressedImage(c) => c.as_image().map(Decompressed::Image),
                other => Ok(Decompressed::Hdu(other)),
            })
        })
    }

    /// Re-serialize every HDU and write the result to `path`.
    ///
    /// The output is valid FITS. It is not byte-identical to the
    /// source, because number formatting, comment padding and
    /// `CONTINUE` chunking can differ. Call [`std::fs::copy`] for an
    /// exact copy.
    ///
    /// # Errors
    ///
    /// - [`FitsError::Io`] when `path` cannot be created or written.
    ///   An `overwrite` value of `false` gives
    ///   [`std::io::ErrorKind::AlreadyExists`] when `path` exists.
    /// - The conditions of [`FitsFile::hdu`] for each HDU, because
    ///   this function re-reads every HDU before it writes.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write(&self, path: impl AsRef<Path>, overwrite: bool) -> Result<()> {
        use crate::io::writer::FitsWriter;
        use std::fs::OpenOptions;
        use std::io::BufWriter;

        let mut opts = OpenOptions::new();
        opts.write(true).create(true);
        if overwrite {
            opts.truncate(true);
        } else {
            opts.create_new(true);
        }
        let file = opts.open(path.as_ref())?;
        let mut w = FitsWriter::new(BufWriter::new(file));
        for i in 0..self.len() {
            let hdu = self.hdu(i)?;
            w.write_hdu(hdu.header(), hdu.data_bytes())?;
        }
        w.finish()?;
        Ok(())
    }

    /// Return HDU `i` as an image, decompressing it when needed.
    ///
    /// An [`Hdu::Image`] comes back as [`ImageOrOwned::Borrowed`]. An
    /// [`Hdu::CompressedImage`] is decoded first and comes back as
    /// [`ImageOrOwned::Owned`].
    ///
    /// # Errors
    ///
    /// - [`FitsError::HduMismatch`] when HDU `i` is neither an image
    ///   nor a tile-compressed image.
    /// - [`FitsError::Data`] when a tile fails to decompress.
    /// - The conditions of [`FitsFile::hdu`].
    #[cfg(feature = "compression")]
    pub fn image(&self, i: usize) -> Result<ImageOrOwned<'_>> {
        match self.hdu(i)? {
            Hdu::Image(img) => Ok(ImageOrOwned::Borrowed(img)),
            Hdu::CompressedImage(c) => Ok(ImageOrOwned::Owned(c.as_image()?)),
            other => Err(FitsError::HduMismatch {
                expected: "IMAGE or compressed-IMAGE",
                found: format!("{other:?}").chars().take(64).collect(),
            }),
        }
    }

    /// Parse the WCS of HDU `i` for alternate descriptor `alt`.
    ///
    /// Pass `' '` for `alt` to select the primary description. This
    /// function loads any `-TAB` lookup extension from the same file,
    /// so the returned [`Wcs`](crate::wcs::Wcs) needs no further
    /// setup. The result is `Ok(None)` when the header carries no WCS
    /// for that descriptor.
    ///
    /// This function does not merge inherited primary-HDU keywords.
    /// Call [`FitsFile::wcs_inherited`] when the extension declares
    /// `INHERIT = T`.
    ///
    /// # Errors
    ///
    /// - [`FitsError::HduMismatch`] when HDU `i` is not image-shaped.
    /// - [`FitsError::Wcs`] when the header declares a WCS that the
    ///   parser rejects, or when a `-TAB` extension named by the
    ///   header is absent or has the wrong shape.
    /// - The conditions of [`FitsFile::parsed_header`].
    pub fn wcs(&self, i: usize, alt: char) -> Result<Option<crate::wcs::Wcs>> {
        // A WCS is entirely in the header, and headers are already
        // loaded; `hdu(i)` would read the data section too. Tile
        // compression is the exception: the real header is only
        // recovered by decoding the image.
        let header = self.parsed_header(i)?;
        let header = match hdu_kind(&header, i) {
            HduKind::Image => header,
            #[cfg(feature = "compression")]
            HduKind::CompressedImage => match self.hdu(i)? {
                Hdu::CompressedImage(c) => c.as_image()?.header().clone(),
                // `ZIMAGE = T` but not decodable: use the raw header.
                _ => header,
            },
            HduKind::Other(kind) => {
                return Err(FitsError::HduMismatch {
                    expected: "IMAGE",
                    found: kind,
                });
            }
        };
        let Some(mut wcs) = crate::wcs::Wcs::from_header(&header, alt)? else {
            return Ok(None);
        };
        if !wcs.tab_specs.is_empty() {
            wcs.resolve_tab(self)?;
        }
        Ok(Some(wcs))
    }

    /// Parse the pixel-list WCS of binary-table HDU `i` (Standard
    /// Sec.8.2, Table 22).
    ///
    /// A pixel list uses the `TCTYPn` and `TCRVLn` keyword family to
    /// georeference scalar coordinate columns. An event list from a
    /// high-energy instrument carries this form.
    ///
    /// The result is `Ok(None)` when the table carries no pixel-list
    /// keyword for `alt`. The `colax` field of the result names the
    /// column that feeds each axis.
    ///
    /// # Errors
    ///
    /// - [`FitsError::HduMismatch`] when HDU `i` is not a `BINTABLE`.
    /// - [`FitsError::Wcs`] when the keywords describe a WCS that the
    ///   parser rejects, or when a `-TAB` extension is absent.
    /// - The conditions of [`FitsFile::parsed_header`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use fitsy::FitsFile;
    ///
    /// let f = FitsFile::open("acis_evt2.fits")?;
    /// if let Some(t) = f.pixel_list_wcs(1, ' ')? {
    ///     println!("sky columns {:?}", t.colax);
    ///     let sky = t.wcs.pixel_to_world(&[4096.0, 4096.0])?;
    ///     println!("{sky:?}");
    /// }
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    pub fn pixel_list_wcs(&self, i: usize, alt: char) -> Result<Option<crate::wcs::TableWcs>> {
        let header = self.bintable_header(i)?;
        let Some(mut t) = crate::wcs::TableWcs::from_pixel_list(&header, alt)? else {
            return Ok(None);
        };
        if !t.wcs.tab_specs.is_empty() {
            t.wcs.resolve_tab(self)?;
        }
        Ok(Some(t))
    }

    /// Parse the WCS of the image held in the vector cells of column
    /// `column` of binary-table HDU `i`.
    ///
    /// This is the `iCTYPn` form of Standard Sec.8.2, Table 22. The
    /// `column` argument is 1-based, as the `TFORMn` numbering is. The
    /// result is `Ok(None)` when the table carries no such keyword for
    /// `alt`.
    ///
    /// # Errors
    ///
    /// The same conditions as [`FitsFile::pixel_list_wcs`].
    pub fn column_wcs(
        &self,
        i: usize,
        column: usize,
        alt: char,
    ) -> Result<Option<crate::wcs::TableWcs>> {
        let header = self.bintable_header(i)?;
        let Some(mut t) = crate::wcs::TableWcs::from_table_column(&header, column, alt)? else {
            return Ok(None);
        };
        if !t.wcs.tab_specs.is_empty() {
            t.wcs.resolve_tab(self)?;
        }
        Ok(Some(t))
    }

    /// Header of HDU `i`, rejecting anything that is not a binary
    /// table. Table 22's table-resident WCS forms are only defined for
    /// `BINTABLE`.
    fn bintable_header(&self, i: usize) -> Result<Header> {
        let header = self.parsed_header(i)?;
        match header.optional_string("XTENSION") {
            Some("BINTABLE") => Ok(header),
            other => Err(FitsError::HduMismatch {
                expected: "BINTABLE",
                found: other.unwrap_or("primary HDU").to_string(),
            }),
        }
    }

    /// Verify the `CHECKSUM` and `DATASUM` cards of every HDU.
    ///
    /// The result holds one [`ChecksumReport`] per HDU, in file order.
    /// An HDU that declares neither card reports `None` for both
    /// verdicts, and this function skips its data scan.
    ///
    /// An on-disk file streams through a 1 MiB buffer and populates no
    /// cache, so verifying a large file does not hold it in memory.
    ///
    /// # Errors
    ///
    /// - [`FitsError::Io`] when a data section fails to read.
    /// - [`FitsError::Block`], [`FitsError::Card`] or
    ///   [`FitsError::Value`] when a header fails to re-parse.
    pub fn verify_checksums(&self) -> Result<Vec<ChecksumReport>> {
        // Streaming chunk size for the on-disk path. 1 MiB is a
        // good balance between syscall overhead and peak RSS.
        // Only the on-disk path uses it, so it is absent on wasm.
        #[cfg(not(target_arch = "wasm32"))]
        const CHUNK: usize = 1 << 20;

        let mut out = Vec::with_capacity(self.hdu_spans.len());
        for i in 0..self.hdu_spans.len() {
            let header_bytes: &[u8] = &self.header_bytes[i];
            let (header, _) = Header::parse_with(header_bytes, 0, self.lenient)?;
            let checksum_card = match header.first("CHECKSUM") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            };
            let datasum_card = match header.first("DATASUM") {
                Some(Value::String(s)) => Some(s.clone()),
                Some(Value::Integer(n)) => Some(n.to_string()),
                _ => None,
            };

            // Skip the (potentially large) data scan entirely when
            // the HDU declares neither sum.
            let need_sum = checksum_card.is_some() || datasum_card.is_some();
            let span = &self.hdu_spans[i];
            let padded = pad_to_block(span.data_logical_len);

            let data_sum: u32 = if need_sum {
                match &self.backing {
                    Backing::InMemory(src) => {
                        let start = span.header_end as usize;
                        let bytes = &src.as_bytes()[start..start + padded as usize];
                        crate::checksum::checksum_bytes(bytes)
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    Backing::OnDisk(file) => {
                        // Stream the padded data section in fixed-size
                        // chunks, accumulating the 1's-complement sum
                        // via `checksum_combine`. This avoids touching
                        // `data_cache` so the verified bytes are never
                        // held resident.
                        let mut acc: u32 = 0;
                        let mut off = span.header_end;
                        let end = span.header_end + padded;
                        let mut buf = vec![0_u8; CHUNK];
                        while off < end {
                            let want = ((end - off) as usize).min(CHUNK);
                            let dst = &mut buf[..want];
                            pread_exact(file, off, dst)?;
                            acc = crate::checksum::checksum_combine(
                                acc,
                                crate::checksum::checksum_bytes(dst),
                            );
                            off += want as u64;
                        }
                        acc
                    }
                }
            } else {
                0
            };

            let datasum_ok = datasum_card.as_deref().map(|stored| {
                let want: u32 = match stored.trim().trim_matches('\'').trim().parse() {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                data_sum == want
            });
            let checksum_ok = checksum_card.as_deref().map(|_| {
                let header_sum = crate::checksum::checksum_bytes(header_bytes);
                let combined = crate::checksum::checksum_combine(header_sum, data_sum);
                combined == 0xFFFF_FFFF
            });

            out.push(ChecksumReport {
                hdu: i,
                checksum_ok,
                datasum_ok,
            });
        }
        Ok(out)
    }
}

/// Output of [`FitsFile::image`]: either a borrowed plain `ImageHdu`
/// or an owned decompressed image.
#[cfg(feature = "compression")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ImageOrOwned<'a> {
    /// An image HDU read in place, borrowing the file buffer.
    Borrowed(ImageHdu<'a>),
    /// An image decompressed into a fresh allocation.
    Owned(crate::compression::OwnedImage),
}

/// Per-HDU result returned by [`FitsFile::verify_checksums`]. A
/// `None` means the corresponding keyword was absent (FITS standard
/// permits omitting either independently).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ChecksumReport {
    /// Zero-based index of the HDU this report covers.
    pub hdu: usize,
    /// Whether `CHECKSUM` matched, or `None` if the card is absent.
    pub checksum_ok: Option<bool>,
    /// Whether `DATASUM` matched, or `None` if the card is absent.
    pub datasum_ok: Option<bool>,
}

/// One item from [`FitsFile::iter_decompressed`].
///
/// A tile-compressed image arrives as [`Decompressed::Image`], already
/// decoded into an [`OwnedImage`](crate::OwnedImage). Every other HDU
/// arrives unchanged as [`Decompressed::Hdu`].
#[cfg(feature = "compression")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Decompressed<'a> {
    /// A regular HDU, returned untouched.
    Hdu(Hdu<'a>),
    /// A tile-compressed image HDU that has been fully decompressed.
    Image(crate::compression::OwnedImage),
}

/// HDU kind, decided from the header alone.
#[derive(Debug)]
enum HduKind {
    Image,
    #[cfg(feature = "compression")]
    CompressedImage,
    Other(String),
}

fn hdu_kind(header: &Header, index: usize) -> HduKind {
    if index == 0 {
        return if is_random_groups(header) {
            HduKind::Other("RANDOM-GROUPS".into())
        } else {
            HduKind::Image
        };
    }
    match header.first("XTENSION") {
        Some(Value::String(s)) if s == "IMAGE" => HduKind::Image,
        Some(Value::String(s)) if s == "BINTABLE" => {
            #[cfg(feature = "compression")]
            if matches!(header.first("ZIMAGE"), Some(Value::Logical(true))) {
                return HduKind::CompressedImage;
            }
            HduKind::Other(s.clone())
        }
        Some(Value::String(s)) => HduKind::Other(s.clone()),
        _ => HduKind::Other("unknown".into()),
    }
}

fn require_simple_t(h: &Header, lenient: bool) -> Result<()> {
    match h.first("SIMPLE") {
        Some(Value::Logical(true)) => Ok(()),
        Some(Value::Logical(false)) if lenient => Ok(()),
        Some(Value::Logical(false)) => Err(FitsError::NonStandard(
            "SIMPLE = F (file does not conform to FITS); lenient parsing \
             (the default) reads it anyway -- drop `.lenient(false)`"
                .into(),
        )),
        Some(_) => Err(FitsError::Value {
            keyword: "SIMPLE".into(),
            msg: "SIMPLE must be Logical".into(),
        }),
        None => Err(FitsError::MissingMandatory {
            keyword: "SIMPLE".into(),
        }),
    }
}

fn require_xtension(h: &Header) -> Result<()> {
    match h.first("XTENSION") {
        Some(Value::String(_)) => Ok(()),
        _ => Err(FitsError::MissingMandatory {
            keyword: "XTENSION".into(),
        }),
    }
}

/// Detect a Random Groups primary HDU per Standard Sec.6:
/// `NAXIS1 = 0`, `NAXIS >= 2`, and `GROUPS = T`.
fn is_random_groups(h: &Header) -> bool {
    let Ok(naxis) = h.naxis() else { return false };
    if naxis < 2 {
        return false;
    }
    let Ok(n1) = h.naxisn(1) else { return false };
    if n1 != 0 {
        return false;
    }
    matches!(h.first("GROUPS"), Some(Value::Logical(true)))
}

/// Read 2880-byte blocks from `cursor` until an `END` card appears,
/// and return the whole block-aligned header buffer.
///
/// A missing `END` card is a hard error, matching [`Header::parse_with`]:
/// `END` is the only header/data delimiter, so it is required in every
/// mode.
#[cfg(not(target_arch = "wasm32"))]
fn read_header_blocks(file: &mut File, cursor: u64, total: u64, lenient: bool) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(BLOCK_SIZE * 2);
    let mut at = cursor;
    file.seek(SeekFrom::Start(cursor))?;
    loop {
        if at + BLOCK_SIZE as u64 > total {
            return Err(FitsError::Block {
                offset: at,
                msg: "header truncated before END card".into(),
            });
        }
        let mut block = [0_u8; BLOCK_SIZE];
        file.read_exact(&mut block)?;
        buf.extend_from_slice(&block);
        // Scan this block for the END card.
        if block_contains_end(&block, lenient) {
            return Ok(buf);
        }
        at += BLOCK_SIZE as u64;
    }
}

/// True if every byte in `start..total` is zero. Reads positionally in
/// block-sized chunks and short-circuits on the first non-zero byte, so the
/// common case (a real extension header follows) costs a single block read.
/// Used to tolerate the all-zero trailing padding some non-standard writers
/// append after the final HDU.
#[cfg(not(target_arch = "wasm32"))]
fn remaining_is_zero(file: &File, start: u64, total: u64) -> Result<bool> {
    let mut off = start;
    let mut buf = [0_u8; BLOCK_SIZE];
    while off < total {
        let want = ((total - off) as usize).min(BLOCK_SIZE);
        pread_exact(file, off, &mut buf[..want])?;
        if buf[..want].iter().any(|&b| b != 0) {
            return Ok(false);
        }
        off += want as u64;
    }
    Ok(true)
}

#[cfg(not(target_arch = "wasm32"))]
fn block_contains_end(block: &[u8], lenient: bool) -> bool {
    use crate::header::card::{CARD_SIZE, Card};
    // Detect the END card with the *same* logic the header parser uses
    // (`Card::parse_with`), so the on-disk reader and the in-memory
    // parser never disagree about where a header ends. In particular this
    // inherits the parser's handling of NUL-padded END bodies and, in
    // lenient mode, a lower-case/mixed-case `end` keyword (folded to
    // `END`). A card that fails to parse is simply "not END" for the
    // purposes of this scan; its error, if any, surfaces during the full
    // parse of the header buffer.
    block
        .chunks_exact(CARD_SIZE)
        .any(|c| Card::parse_with(c, 0, lenient).is_ok_and(|card| card.is_end()))
}

/// True when `probe` could be the first card of a conforming
/// extension HDU. Standard Sec.7.1.3 requires `XTENSION` in columns 1
/// to 8. Bytes that do not start that way are trailing junk, so the
/// scan stops rather than parsing them as a header.
fn looks_like_extension_start(probe: &[u8]) -> bool {
    use crate::header::card::CARD_SIZE;
    probe.len() >= CARD_SIZE && probe.starts_with(b"XTENSION")
}

/// On-disk variant of [`looks_like_extension_start`]: positional read of
/// the keyword bytes at `cursor`. Returns `Ok(false)` when there is not
/// enough room for a full card or the keyword is not `XTENSION`.
#[cfg(not(target_arch = "wasm32"))]
fn next_is_extension(file: &File, cursor: u64, total: u64) -> Result<bool> {
    use crate::header::card::CARD_SIZE;
    if total - cursor < CARD_SIZE as u64 {
        return Ok(false);
    }
    let mut probe = [0_u8; 8];
    pread_exact(file, cursor, &mut probe)?;
    Ok(&probe == b"XTENSION")
}

/// Positional read filling `buf` exactly. Loops over short reads
/// and retries on `EINTR`.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn pread_exact(file: &File, mut off: u64, mut buf: &mut [u8]) -> Result<()> {
    while !buf.is_empty() {
        match file.read_at(buf, off) {
            Ok(0) => {
                return Err(FitsError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "pread reached EOF before filling buffer (file truncated?)",
                )));
            }
            Ok(n) => {
                off += n as u64;
                let tmp = buf;
                buf = &mut tmp[n..];
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(FitsError::Io(e)),
        }
    }
    Ok(())
}

#[cfg(all(windows, not(target_arch = "wasm32")))]
fn pread_exact(file: &File, mut off: u64, mut buf: &mut [u8]) -> Result<()> {
    while !buf.is_empty() {
        match file.seek_read(buf, off) {
            Ok(0) => {
                return Err(FitsError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "seek_read reached EOF before filling buffer",
                )));
            }
            Ok(n) => {
                off += n as u64;
                let tmp = buf;
                buf = &mut tmp[n..];
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(FitsError::Io(e)),
        }
    }
    Ok(())
}

/// Data section size in bytes (Standard Sec.4.4.1.1, Sec.6, Sec.7).
///
/// Two modes share the formula
/// `|BITPIX|/8 * GCOUNT * (PCOUNT + prod NAXIS{start..=naxis})`:
///
/// - Random Groups (Sec.6): NAXIS1 = 0 is the marker, so the product
///   starts at NAXIS2; an empty data axis still leaves PCOUNT bytes
///   per group.
/// - Generic conforming extension (Sec.7.1.3): product starts at
///   NAXIS1; an empty data axis means zero bytes total.
pub(crate) fn data_section_size(h: &Header) -> Result<u64> {
    let bitpix = Bitpix::from_i64(h.bitpix()?)?;
    let naxis = h.naxis()?;
    if naxis == 0 {
        return Ok(0);
    }

    let rg = is_random_groups(h);
    let start_axis = if rg { 2 } else { 1 };
    let mut prod: u64 = 1;
    for i in start_axis..=naxis {
        let n = h.naxisn(i)?;
        if n == 0 {
            if rg {
                prod = 0;
                break;
            }
            return Ok(0);
        }
        prod = prod
            .checked_mul(n)
            .ok_or_else(|| FitsError::Data("axis product overflows u64".into()))?;
    }

    // Sec.7.1.3 makes both non-negative. Casting a negative straight to
    // `u64` produced a huge value that failed the overflow check below
    // and reported "data size overflows", naming neither the keyword
    // nor the real problem.
    let count = |key: &str, default: i64| -> Result<u64> {
        let v = h.optional_int(key).unwrap_or(default);
        u64::try_from(v).map_err(|_| FitsError::Value {
            keyword: key.into(),
            msg: format!("{key} must be non-negative, got {v}"),
        })
    };
    let pcount = count("PCOUNT", 0)?;
    let gcount = count("GCOUNT", 1)?;

    let bytes_per_elem = bitpix.byte_size() as u64;
    let total = bytes_per_elem
        .checked_mul(gcount)
        .and_then(|v| v.checked_mul(pcount.checked_add(prod)?))
        .ok_or_else(|| FitsError::Data("data size overflows u64".into()))?;
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::card::CARD_SIZE;
    use crate::io::block::BLOCK_SIZE;

    fn pad_card(s: &str) -> [u8; CARD_SIZE] {
        let mut c = [b' '; CARD_SIZE];
        c[..s.len()].copy_from_slice(s.as_bytes());
        c
    }

    fn build_simple_no_data() -> Vec<u8> {
        let cards = [
            pad_card("SIMPLE  =                    T"),
            pad_card("BITPIX  =                    8"),
            pad_card("NAXIS   =                    0"),
            pad_card("END"),
        ];
        let mut buf = Vec::new();
        for c in &cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(b' ');
        }
        buf
    }

    #[test]
    fn empty_primary_hdu_round_trip() {
        let bytes = build_simple_no_data();
        let f = FitsFile::from_bytes(bytes.clone()).unwrap();
        assert_eq!(f.len(), 1);
        match f.hdu(0).unwrap() {
            Hdu::Image(img) => {
                assert_eq!(img.bitpix(), Bitpix::U8);
                assert_eq!(img.n_elements(), 0);
                assert_eq!(img.raw_bytes().len(), 0);
            }
            other => panic!("expected image, got {other:?}"),
        }
    }

    #[test]
    fn primary_with_image_data() {
        // BITPIX=16, NAXIS=2, NAXIS1=2, NAXIS2=3 -> 6 i16 values = 12 bytes
        let header_cards = [
            pad_card("SIMPLE  =                    T"),
            pad_card("BITPIX  =                   16"),
            pad_card("NAXIS   =                    2"),
            pad_card("NAXIS1  =                    2"),
            pad_card("NAXIS2  =                    3"),
            pad_card("END"),
        ];
        let mut buf = Vec::new();
        for c in &header_cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(b' ');
        }
        // Data block.
        let pixels: [i16; 6] = [1, 2, 3, 4, 5, 6];
        let data_start = buf.len();
        for &p in &pixels {
            buf.extend_from_slice(&p.to_be_bytes());
        }
        while (buf.len() - data_start) % BLOCK_SIZE != 0 {
            buf.push(0);
        }
        let f = FitsFile::from_bytes(buf).unwrap();
        let Hdu::Image(img) = f.hdu(0).unwrap() else {
            panic!("expected image");
        };
        assert_eq!(img.axes(), &[2_u64, 3]);
        let raw: Vec<i16> = img.read_raw::<i16>().unwrap().into_vec();
        assert_eq!(raw, pixels);
        let phys = img.read_physical().unwrap().into_vec();
        assert_eq!(
            phys,
            pixels.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn trailing_zero_padding_is_tolerated() {
        // Several CCD capture programs append a run of zero bytes after the
        // final HDU, sometimes not even padded to the 2880-byte block size
        // (e.g. ZWO ASI camera captures). The primary HDU must still read.
        let mut buf = build_simple_no_data();
        // Non-block-aligned trailing zeros, as seen in the wild.
        buf.extend(std::iter::repeat_n(0_u8, 12345));
        let f = FitsFile::from_bytes(buf).unwrap();
        assert_eq!(f.len(), 1);
        match f.hdu(0).unwrap() {
            Hdu::Image(img) => assert_eq!(img.n_elements(), 0),
            other => panic!("expected image, got {other:?}"),
        }
    }

    /// SIMPLE primary HDU whose single header block is NUL-padded after
    /// the END card instead of space-padded (a non-conforming form emitted
    /// by some capture software).
    fn build_simple_nul_padded_header() -> Vec<u8> {
        let cards = [
            pad_card("SIMPLE  =                    T"),
            pad_card("BITPIX  =                    8"),
            pad_card("NAXIS   =                    0"),
            pad_card("END"),
        ];
        let mut buf = Vec::new();
        for c in &cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(0); // NUL pad rather than space pad
        }
        buf
    }

    #[test]
    fn nul_padded_header_block_is_tolerated() {
        let bytes = build_simple_nul_padded_header();
        let f = FitsFile::from_bytes(bytes).unwrap();
        assert_eq!(f.len(), 1);
        match f.hdu(0).unwrap() {
            Hdu::Image(img) => assert_eq!(img.n_elements(), 0),
            other => panic!("expected image, got {other:?}"),
        }
    }

    #[test]
    fn trailing_non_zero_junk_is_tolerated() {
        // Some CCD capture programs (ZWO ASI Studio is a known case) append
        // a block of vendor metadata / thumbnail data after the final HDU.
        // The bytes are not all-zero and do not begin with `XTENSION`, so
        // they cannot be a conforming extension; the primary HDU must still
        // read.
        let mut buf = build_simple_no_data();
        // 87 blocks of non-zero junk, mirroring the real-world ZWO files.
        let junk_len = 87 * BLOCK_SIZE + 879;
        buf.extend((0..junk_len).map(|i| (i & 0xff) as u8 | 1));
        let f = FitsFile::from_bytes(buf).unwrap();
        assert_eq!(f.len(), 1);
        match f.hdu(0).unwrap() {
            Hdu::Image(img) => assert_eq!(img.n_elements(), 0),
            other => panic!("expected image, got {other:?}"),
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn trailing_non_zero_junk_on_disk() {
        // On-disk path equivalent of `trailing_non_zero_junk_is_tolerated`.
        let mut bytes = build_simple_no_data();
        let junk_len = 87 * BLOCK_SIZE + 879;
        bytes.extend((0..junk_len).map(|i| (i & 0xff) as u8 | 1));
        let dir = std::env::temp_dir();
        let path = dir.join("fitsy_trailing_junk_test.fits");
        std::fs::write(&path, &bytes).unwrap();
        let f = FitsFile::open(&path).unwrap();
        assert_eq!(f.len(), 1);
        match f.hdu(0).unwrap() {
            Hdu::Image(img) => assert_eq!(img.n_elements(), 0),
            other => panic!("expected image, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn nul_padded_header_and_trailing_zeros_on_disk() {
        // Exercises the on-disk `from_file` path (the in-memory tests do
        // not): a NUL-padded header block followed by a non-block-aligned
        // run of trailing zero bytes, both of which must be tolerated.
        let mut bytes = build_simple_nul_padded_header();
        bytes.extend(std::iter::repeat_n(0_u8, 4321));
        let dir = std::env::temp_dir();
        let path = dir.join("fitsy_nul_padded_test.fits");
        std::fs::write(&path, &bytes).unwrap();
        let f = FitsFile::open(&path).unwrap();
        assert_eq!(f.len(), 1);
        match f.hdu(0).unwrap() {
            Hdu::Image(img) => assert_eq!(img.n_elements(), 0),
            other => panic!("expected image, got {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn loaders_agree_across_malformed_headers() {
        // Regression guard for loader consistency: the in-memory
        // (`from_source`) and on-disk (`from_file`) paths must reach the
        // same accept/reject verdict for the same bytes, in both strict and
        // lenient modes. This is what stops the two code paths (notably the
        // on-disk END scan vs. the card parser) from silently drifting.
        fn block(cards: &[&[u8]]) -> Vec<u8> {
            let mut buf = Vec::new();
            for c in cards {
                let mut card = [b' '; CARD_SIZE];
                let n = c.len().min(CARD_SIZE);
                card[..n].copy_from_slice(&c[..n]);
                buf.extend_from_slice(&card);
            }
            while buf.len() % BLOCK_SIZE != 0 {
                buf.push(b' ');
            }
            buf
        }
        let primary = |extra: &[&[u8]]| -> Vec<u8> {
            let base: [&[u8]; 3] = [
                b"SIMPLE  =                    T",
                b"BITPIX  =                    8",
                b"NAXIS   =                    0",
            ];
            block(&base.iter().chain(extra).copied().collect::<Vec<_>>())
        };

        let mut junk_after_end = primary(&[b"END"]);
        junk_after_end[4 * CARD_SIZE] = b'X'; // stray byte in the post-END fill

        // A valid IMAGE extension (header-only, NAXIS=0) appended to a
        // primary, exercising the extension-detection pre-check on both
        // loaders.
        let mut two_hdu = primary(&[b"EXTEND  =                    T", b"END"]);
        two_hdu.extend_from_slice(&block(&[
            b"XTENSION= 'IMAGE   '",
            b"BITPIX  =                    8",
            b"NAXIS   =                    0",
            b"PCOUNT  =                    0",
            b"GCOUNT  =                    1",
            b"END",
        ]));
        // Same, but the second unit's keyword is lower-case `xtension`:
        // neither loader recognizes it as an extension, so both stop at 1.
        let mut lower_xtension = primary(&[b"END"]);
        lower_xtension.extend_from_slice(&block(&[
            b"xtension= 'IMAGE   '",
            b"BITPIX  =                    8",
            b"NAXIS   =                    0",
            b"END",
        ]));

        let cases: [(&str, Vec<u8>); 9] = [
            ("valid", primary(&[b"END"])),
            ("lowercase_end", primary(&[b"end"])),
            ("mixed_case_end", primary(&[b"eNd"])),
            ("missing_end", primary(&[])),
            ("junk_after_end", junk_after_end),
            (
                "broken_continue",
                primary(&[b"OBJECT  = 'ab'", b"CONTINUE  ' cd'", b"END"]),
            ),
            (
                "bad_value",
                primary(&[b"EXPTIME =              12.3.4.5", b"END"]),
            ),
            ("two_hdu", two_hdu),
            ("lower_xtension", lower_xtension),
        ];

        let path = std::env::temp_dir().join("fitsy_loader_consistency.fits");
        for (name, bytes) in &cases {
            for lenient in [false, true] {
                let mem =
                    FitsFile::from_source(ByteSource::from_vec(bytes.clone()).unwrap(), lenient);
                std::fs::write(&path, bytes).unwrap();
                let disk = FitsFile::from_file(File::open(&path).unwrap(), lenient);
                assert_eq!(
                    mem.is_ok(),
                    disk.is_ok(),
                    "loader verdict mismatch for `{name}` at lenient={lenient}: \
                     in-memory={} on-disk={}",
                    mem.is_ok(),
                    disk.is_ok()
                );
                // When both accept, they must agree on the HDU count too --
                // this is what catches an extension-detection divergence.
                if let (Ok(m), Ok(d)) = (&mem, &disk) {
                    assert_eq!(
                        m.len(),
                        d.len(),
                        "HDU-count mismatch for `{name}` at lenient={lenient}"
                    );
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    fn build_simple_f_no_data() -> Vec<u8> {
        let cards = [
            pad_card("SIMPLE  =                    F"),
            pad_card("BITPIX  =                    8"),
            pad_card("NAXIS   =                    0"),
            pad_card("END"),
        ];
        let mut buf = Vec::new();
        for c in &cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(b' ');
        }
        buf
    }

    #[test]
    fn simple_f_strict_rejected() {
        // Parsing is lenient by default now, so strict must be requested
        // explicitly to reject a `SIMPLE = F` file.
        let bytes = build_simple_f_no_data();
        let err = FitsFile::from_bytes_with(bytes, false).unwrap_err();
        assert!(matches!(err, FitsError::NonStandard(_)), "got {err:?}");
    }

    #[test]
    fn simple_f_accepted_by_default() {
        // The default (lenient) path reads a `SIMPLE = F` file.
        let bytes = build_simple_f_no_data();
        let f = FitsFile::from_bytes(bytes).unwrap();
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn simple_f_lenient_accepted() {
        let bytes = build_simple_f_no_data();
        let f = FitsFile::from_bytes_with(bytes, true).unwrap();
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn inherit_merges_primary_keywords() {
        // Primary HDU with OBJECT keyword.
        let primary_cards = [
            pad_card("SIMPLE  =                    T"),
            pad_card("BITPIX  =                    8"),
            pad_card("NAXIS   =                    0"),
            pad_card("EXTEND  =                    T"),
            pad_card("OBJECT  = 'NGC1234 '"),
            pad_card("OBSERVER= 'me      '"),
            pad_card("END"),
        ];
        let ext_cards = [
            pad_card("XTENSION= 'IMAGE   '"),
            pad_card("BITPIX  =                    8"),
            pad_card("NAXIS   =                    0"),
            pad_card("PCOUNT  =                    0"),
            pad_card("GCOUNT  =                    1"),
            pad_card("INHERIT =                    T"),
            pad_card("OBSERVER= 'override'"),
            pad_card("END"),
        ];
        let mut buf = Vec::new();
        for c in &primary_cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(b' ');
        }
        for c in &ext_cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(b' ');
        }
        let f = FitsFile::from_bytes(buf).unwrap();
        let merged = f.header_inherited(1).unwrap();
        // OBJECT inherited from primary.
        assert!(matches!(
            merged.first("OBJECT"),
            Some(Value::String(s)) if s == "NGC1234"
        ));
        // OBSERVER kept from extension (already present).
        assert!(matches!(
            merged.first("OBSERVER"),
            Some(Value::String(s)) if s == "override"
        ));
        // Structural keyword from primary NOT inherited.
        // (Extension has its own BITPIX/NAXIS, INHERIT itself stays.)
        assert!(matches!(
            merged.first("INHERIT"),
            Some(Value::Logical(true))
        ));
    }

    #[test]
    fn wcs_inherited_pulls_wcs_from_primary() {
        // Primary HDU carries the WCS keywords; extension carries only
        // image data + INHERIT = T. `wcs_inherited(1, b' ')` must
        // return the primary's WCS.
        let primary_cards = [
            pad_card("SIMPLE  =                    T"),
            pad_card("BITPIX  =                    8"),
            pad_card("NAXIS   =                    0"),
            pad_card("EXTEND  =                    T"),
            pad_card("CTYPE1  = 'RA---TAN'"),
            pad_card("CTYPE2  = 'DEC--TAN'"),
            pad_card("CRPIX1  =                 50.0"),
            pad_card("CRPIX2  =                 50.0"),
            pad_card("CRVAL1  =              83.6331"),
            pad_card("CRVAL2  =              22.0145"),
            pad_card("CDELT1  =          -2.78E-04"),
            pad_card("CDELT2  =           2.78E-04"),
            pad_card("END"),
        ];
        let ext_cards = [
            pad_card("XTENSION= 'IMAGE   '"),
            pad_card("BITPIX  =                    8"),
            pad_card("NAXIS   =                    2"),
            pad_card("NAXIS1  =                  100"),
            pad_card("NAXIS2  =                  100"),
            pad_card("PCOUNT  =                    0"),
            pad_card("GCOUNT  =                    1"),
            pad_card("INHERIT =                    T"),
            pad_card("END"),
        ];
        let mut buf = Vec::new();
        for c in &primary_cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(b' ');
        }
        for c in &ext_cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(b' ');
        }
        // Pad data unit (100*100 = 10000 bytes).
        let data_start = buf.len();
        buf.extend(std::iter::repeat_n(0_u8, 100 * 100));
        while (buf.len() - data_start) % BLOCK_SIZE != 0 {
            buf.push(0);
        }
        let f = FitsFile::from_bytes(buf).unwrap();
        // Without inheritance the extension has no WCS:
        let Hdu::Image(img) = f.hdu(1).unwrap() else {
            panic!("not image");
        };
        assert!(img.wcs(' ').unwrap().is_none(), "no WCS on extension");
        // With inheritance the primary's WCS is used:
        let wcs = f
            .wcs_inherited(1, ' ')
            .unwrap()
            .expect("inherited WCS present");
        // CRPIX1/2 = 50 in the FITS header (1-based). The Wcs API
        // is 0-based, so the reference pixel is at (49, 49).
        let world = wcs.pixel_to_world(&[49.0, 49.0]).unwrap();
        assert!((world[0] - 83.6331).abs() < 1e-9);
        assert!((world[1] - 22.0145).abs() < 1e-9);
    }

    #[test]
    fn inherit_false_does_not_merge() {
        let primary_cards = [
            pad_card("SIMPLE  =                    T"),
            pad_card("BITPIX  =                    8"),
            pad_card("NAXIS   =                    0"),
            pad_card("EXTEND  =                    T"),
            pad_card("OBJECT  = 'NGC1234 '"),
            pad_card("END"),
        ];
        let ext_cards = [
            pad_card("XTENSION= 'IMAGE   '"),
            pad_card("BITPIX  =                    8"),
            pad_card("NAXIS   =                    0"),
            pad_card("PCOUNT  =                    0"),
            pad_card("GCOUNT  =                    1"),
            pad_card("END"),
        ];
        let mut buf = Vec::new();
        for c in &primary_cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(b' ');
        }
        for c in &ext_cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(b' ');
        }
        let f = FitsFile::from_bytes(buf).unwrap();
        let merged = f.header_inherited(1).unwrap();
        assert!(merged.first("OBJECT").is_none());
    }
}
