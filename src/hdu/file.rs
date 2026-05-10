//! [`FitsFile`]: top-level reader (Standard Sec.3.4).
//!
//! Open a file with [`FitsFile::open`]. Access individual HDUs by
//! index with [`FitsFile::hdu`], by name with [`FitsFile::hdu_by_name`],
//! or iterate all of them with [`FitsFile::iter`].
//!
//! For lenient parsing (e.g. `SIMPLE = F`) use [`FitsOpenOptions`].
//!
//! # Memory model
//!
//! `FitsFile::open` does **not** read pixel/table bytes up front.
//! It scans the file's headers (small, bounded by the FITS 2880-byte
//! block size) and records the byte offset and length of each HDU's
//! data section. The data bytes for HDU `i` are loaded from disk on
//! the first call to `hdu(i)` (or related accessors) via positional
//! reads (`pread`) and cached in memory until the file is dropped.
//! Files opened with [`FitsFile::from_bytes`] keep the entire buffer
//! in memory (suited to tests and small in-memory workflows).
//!
//! This is critical for very large multi-extension files: opening a
//! 50-GB mosaic costs only the headers, and a caller that only reads
//! HDU 3 only pays for HDU 3's data section.

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
use crate::io::block::{BLOCK_SIZE, pad_to_block};
use crate::io::source::ByteSource;

/// Top-level FITS file. `Send` but not `Sync`; for concurrent access
/// each thread or task should open its own handle.
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
    /// live in `Backing::InMemory` instead).
    data_cache: Vec<OnceLock<Vec<u8>>>,
    /// Map `EXTNAME` (trimmed) -> sorted list of HDU indices that
    /// declare it. Built once at open time so [`hdu_by_name`] is
    /// O(log n + k) instead of O(n).
    extname_index: BTreeMap<String, Vec<usize>>,
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
    /// Open `path` and parse its HDU headers. Pixel/table data
    /// sections are **not** read up front; each HDU's data is loaded
    /// on demand the first time it is accessed.
    ///
    /// For non-default options (lenient parsing) use
    /// [`FitsOpenOptions`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use fitsy::FitsFile;
    ///
    /// let f = FitsFile::open("image.fits")?;
    /// println!("{} HDUs", f.len());
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, false)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn open_with(path: impl AsRef<Path>, lenient: bool) -> Result<Self> {
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

    /// Build a `FitsFile` from an in-memory buffer. The whole buffer
    /// is retained for the life of the `FitsFile`.
    pub fn from_bytes(buf: Vec<u8>) -> Result<Self> {
        Self::from_source(ByteSource::from_vec(buf)?, false)
    }

    fn from_source(src: ByteSource, lenient: bool) -> Result<Self> {
        let bytes = src.as_bytes();
        let total = bytes.len() as u64;
        let mut hdu_spans = Vec::new();
        let mut header_bytes_per_hdu = Vec::new();
        let mut cursor: u64 = 0;
        let mut is_first = true;

        while cursor < total {
            let header_start = cursor;
            let (header, header_blocks_bytes) = Header::parse(bytes, cursor)?;
            let header_end = header_start + header_blocks_bytes;

            if is_first {
                require_simple_t(&header, lenient)?;
            } else {
                require_xtension(&header)?;
            }

            let data_logical_len = data_section_size(&header)?;
            let data_padded = pad_to_block(data_logical_len);
            let data_end = header_end + data_padded;
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

        Self::finish_open(Backing::InMemory(src), hdu_spans, header_bytes_per_hdu)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn from_file(mut file: File, lenient: bool) -> Result<Self> {
        let total = file.metadata()?.len();
        let mut hdu_spans = Vec::new();
        let mut header_bytes_per_hdu: Vec<Vec<u8>> = Vec::new();
        let mut cursor: u64 = 0;
        let mut is_first = true;

        while cursor < total {
            let header_start = cursor;
            let header_buf = read_header_blocks(&mut file, cursor, total)?;
            let (header, header_blocks_bytes) = Header::parse(&header_buf, 0)?;
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
            let data_end = header_end + data_padded;
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

        Self::finish_open(Backing::OnDisk(file), hdu_spans, header_bytes_per_hdu)
    }

    fn finish_open(
        backing: Backing,
        hdu_spans: Vec<HduSpan>,
        header_bytes: Vec<Vec<u8>>,
    ) -> Result<Self> {
        if hdu_spans.is_empty() {
            return Err(FitsError::Header("file contains no HDU".into()));
        }
        let mut extname_index: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, hb) in header_bytes.iter().enumerate() {
            if let Ok((header, _)) = Header::parse(hb, 0)
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

    /// Read the unpadded data section for HDU `i` into a freshly
    /// allocated owned `Vec<u8>`, **bypassing the per-HDU cache**.
    /// Use this when the caller will consume the bytes once and
    /// then drop them -- it avoids holding the data resident in
    /// the cache for the lifetime of the [`FitsFile`].
    ///
    /// For in-memory backings this still copies (to keep the
    /// return type uniform); callers that hold a [`FitsFile`]
    /// opened with [`FitsFile::from_bytes`] can use
    /// [`FitsFile::hdu`] instead to avoid the copy.
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

    /// Read a contiguous rectangular sub-region of the pixel array
    /// for image HDU `i` directly from disk into a freshly allocated
    /// big-endian byte buffer. **Does not touch the cache.**
    ///
    /// `axes` is the full image shape in FITS order (NAXIS1 fastest).
    /// `start` and `shape` describe the region to read in the same
    /// FITS order. The returned bytes are big-endian, length =
    /// `prod(shape) * bitpix.byte_size()`. Caller is responsible for
    /// byteswapping into native order.
    ///
    /// Used by the lazy `section[a:b]` Python read path so that
    /// reading a small tile out of a huge image only pays for the
    /// tile bytes, not the whole HDU.
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
            .ok_or_else(|| FitsError::Data(format!("shape product overflows u64")))?;
        let total_bytes = (total_elems as usize)
            .checked_mul(bsize)
            .ok_or_else(|| FitsError::Data(format!("total bytes overflows usize")))?;
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
                    .ok_or_else(|| {
                        FitsError::Data(format!("element offset overflows u64"))
                    })?;
                elem_off = s;
            }
            let byte_off = elem_off
                .checked_mul(bsize as u64)
                .and_then(|v| data_offset.checked_add(v))
                .ok_or_else(|| FitsError::Data(format!("byte offset overflows u64")))?;

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
    pub fn is_empty(&self) -> bool {
        self.hdu_spans.is_empty()
    }

    /// Parse and return the header for HDU `i` **without** reading
    /// or caching its data section. Cheaper than [`hdu`](Self::hdu)
    /// when the caller only needs header-level information (axes,
    /// `BITPIX`, kind detection).
    pub fn parsed_header(&self, i: usize) -> Result<Header> {
        let _ = self.hdu_spans.get(i).ok_or_else(|| {
            FitsError::Header(format!("HDU index {i} out of range (len = {})", self.len()))
        })?;
        let (header, _) = Header::parse(&self.header_bytes[i], 0)?;
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

    /// Borrow the `i`-th HDU (0 = primary).
    pub fn hdu(&self, i: usize) -> Result<Hdu<'_>> {
        let _ = self.hdu_spans.get(i).ok_or_else(|| {
            FitsError::Header(format!("HDU index {i} out of range (len = {})", self.len()))
        })?;
        let (header, _) = Header::parse(&self.header_bytes[i], 0)?;
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

    /// Iterator over all HDUs. Errors produced while parsing HDU
    /// `i` are wrapped in [`FitsError::InHdu`] so callers can
    /// identify the offending HDU without re-walking the file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use fitsy::{FitsFile, Hdu};
    ///
    /// let f = FitsFile::open("multi.fits")?;
    /// for (i, hdu) in f.iter().enumerate() {
    ///     match hdu? {
    ///         Hdu::Image(img) => println!("#{i}: image {:?}", img.axes()),
    ///         Hdu::BinTable(t) => println!("#{i}: bintable {} rows", t.n_rows()),
    ///         _ => println!("#{i}: other"),
    ///     }
    /// }
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

    /// Borrow the `i`-th HDU's header, with primary-HDU keywords
    /// merged in when the extension declares `INHERIT = T` (Goddard /
    /// IRAF convention). Structural keywords are never inherited.
    /// For the primary HDU, returns its own header verbatim.
    pub fn header_inherited(&self, i: usize) -> Result<Header> {
        let _ = self.hdu_spans.get(i).ok_or_else(|| {
            FitsError::Header(format!("HDU index {i} out of range (len = {})", self.len()))
        })?;
        let (mut header, _) = Header::parse(&self.header_bytes[i], 0)?;
        if i == 0 {
            return Ok(header);
        }
        if !matches!(header.first("INHERIT"), Some(Value::Logical(true))) {
            return Ok(header);
        }
        let (primary, _) = Header::parse(&self.header_bytes[0], 0)?;
        header.merge_inherited(&primary);
        Ok(header)
    }

    /// Parse the WCS for HDU `i` and alternate `alt` (`b' '` for the
    /// primary description), with primary-HDU keywords merged in if
    /// the extension declares `INHERIT = T`. Returns `Ok(None)` if
    /// the (possibly inherited) header carries no recognizable WCS.
    ///
    /// This is a convenience shortcut equivalent to
    /// `Wcs::from_header(&self.header_inherited(i)?, alt)`.
    pub fn wcs_inherited(&self, i: usize, alt: char) -> Result<Option<crate::wcs::Wcs>> {
        let header = self.header_inherited(i)?;
        crate::wcs::Wcs::from_header(&header, alt)
    }

    /// Look up an extension by `EXTNAME`, optionally restricting to a
    /// specific `EXTVER`. Returns the first matching HDU. Per the
    /// FITS standard Sec.4.4.2.6, `EXTNAME` is case-sensitive after
    /// trimming trailing spaces; `EXTVER` defaults to 1 when absent.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use fitsy::{FitsFile, Hdu};
    ///
    /// let f = FitsFile::open("hst.fits")?;
    /// let Hdu::BinTable(events) = f.hdu_by_name("EVENTS", None)? else {
    ///     panic!("EVENTS extension is not a binary table");
    /// };
    /// println!("{} events", events.n_rows());
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    pub fn hdu_by_name(&self, name: &str, ver: Option<i64>) -> Result<Hdu<'_>> {
        let candidates: &[usize] = self.extname_index.get(name).map_or(&[], Vec::as_slice);
        for &i in candidates {
            if let Some(want) = ver {
                let (header, _) = Header::parse(&self.header_bytes[i], 0)?;
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
    /// HDUs. Each `Hdu::CompressedImage` is materialised as an
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
    /// The output is valid FITS but is **not** guaranteed to be
    /// byte-identical to the source: number formatting, comment
    /// padding, and the order of `CONTINUE` chunks may differ. For
    /// a byte-exact copy use [`std::fs::copy`] (or read the original
    /// bytes via your own `std::fs::read`).
    ///
    /// `overwrite = false` returns [`std::io::ErrorKind::AlreadyExists`]
    /// if the destination already exists.
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

    /// Convenience accessor for an image HDU. Returns the image
    /// directly (auto-decompressing tile-compressed images when the
    /// `compression` feature is enabled). Errors if HDU `i` is not an
    /// image-shaped HDU.
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

    /// Parse the WCS for HDU `i`, alternate code `alt` (`' '` for
    /// the primary). Resolves any `-TAB` axes by loading their
    /// referenced binary-table extensions from this file. The
    /// returned `Wcs` is fully self-contained: forward and inverse
    /// maps work with no further setup.
    ///
    /// Returns `Ok(None)` when the HDU has no WCS for that
    /// alternate. Errors when the HDU is not image-shaped.
    pub fn wcs(&self, i: usize, alt: char) -> Result<Option<crate::wcs::Wcs>> {
        let header = match self.hdu(i)? {
            Hdu::Image(img) => img.header().clone(),
            #[cfg(feature = "compression")]
            Hdu::CompressedImage(c) => c.as_image()?.header().clone(),
            other => {
                return Err(FitsError::HduMismatch {
                    expected: "IMAGE",
                    found: format!("{other:?}").chars().take(64).collect(),
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

    /// Verify `CHECKSUM` and `DATASUM` for every HDU that carries
    /// them. HDUs without either keyword are skipped. Returns the
    /// per-HDU verdicts in HDU order.
    ///
    /// Streams each data section through the checksum accumulator
    /// in 1-MiB chunks (for on-disk backings) without populating
    /// the per-HDU data cache, so verifying a 50-GB file does not
    /// pin the entire file in RAM.
    pub fn verify_checksums(&self) -> Result<Vec<ChecksumReport>> {
        // Streaming chunk size for the on-disk path. 1 MiB is a
        // good balance between syscall overhead and peak RSS.
        const CHUNK: usize = 1 << 20;

        let mut out = Vec::with_capacity(self.hdu_spans.len());
        for i in 0..self.hdu_spans.len() {
            let header_bytes: &[u8] = &self.header_bytes[i];
            let (header, _) = Header::parse(header_bytes, 0)?;
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
#[derive(Debug)]
#[non_exhaustive]
pub enum ImageOrOwned<'a> {
    Borrowed(ImageHdu<'a>),
    Owned(crate::compression::OwnedImage),
}

/// Builder for opening FITS files with non-default options.
///
/// ```ignore
/// use fitsy::FitsOpenOptions;
///
/// let f = FitsOpenOptions::new()
///     .lenient(true)
///     .open("legacy.fits")?;
/// # Ok::<(), fitsy::FitsError>(())
/// ```
///
/// The shortcut constructors on [`FitsFile`] (`open`, `from_bytes`)
/// cover the strict-mode common cases without going through the
/// builder.
#[derive(Debug, Default, Clone, Copy)]
pub struct FitsOpenOptions {
    lenient: bool,
}

impl FitsOpenOptions {
    /// New options with all flags off (matches [`FitsFile::open`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// If `true`, accept `SIMPLE = F` headers. The Standard
    /// (Sec.3.4.1) calls these "non-standard FITS-like files"; many
    /// legacy IRAF and survey products use this form. All other
    /// validation rules apply unchanged.
    #[must_use]
    pub fn lenient(mut self, lenient: bool) -> Self {
        self.lenient = lenient;
        self
    }

    /// Open the file at `path` using the configured options.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(self, path: impl AsRef<Path>) -> Result<FitsFile> {
        FitsFile::from_source(ByteSource::read_file(path)?, self.lenient)
    }

    /// Build a `FitsFile` from an in-memory buffer using the
    /// configured `lenient` flag.
    pub fn from_bytes(self, buf: Vec<u8>) -> Result<FitsFile> {
        FitsFile::from_source(ByteSource::from_vec(buf)?, self.lenient)
    }
}

/// Per-HDU result returned by [`FitsFile::verify_checksums`]. A
/// `None` means the corresponding keyword was absent (FITS standard
/// permits omitting either independently).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ChecksumReport {
    pub hdu: usize,
    pub checksum_ok: Option<bool>,
    pub datasum_ok: Option<bool>,
}

/// Output element of [`FitsFile::iter_decompressed`].
#[cfg(feature = "compression")]
#[derive(Debug)]
#[non_exhaustive]
pub enum Decompressed<'a> {
    /// A regular HDU, returned untouched.
    Hdu(Hdu<'a>),
    /// A tile-compressed image HDU that has been fully decompressed.
    Image(crate::compression::OwnedImage),
}

fn require_simple_t(h: &Header, lenient: bool) -> Result<()> {
    match h.first("SIMPLE") {
        Some(Value::Logical(true)) => Ok(()),
        Some(Value::Logical(false)) if lenient => Ok(()),
        Some(Value::Logical(false)) => Err(FitsError::NonStandard(
            "SIMPLE = F (file does not conform to FITS); use \
             FitsOpenOptions::new().lenient(true) to read anyway"
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

/// Read FITS header blocks from a file starting at `cursor`. Reads
/// 2880-byte blocks one at a time until an `END` card is found, then
/// returns the entire block-aligned header buffer (including the
/// terminating block, with whatever trailing space padding it
/// contained).
#[cfg(not(target_arch = "wasm32"))]
fn read_header_blocks(file: &mut File, cursor: u64, total: u64) -> Result<Vec<u8>> {
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
        // Scan this block for END card.
        if block_contains_end(&block) {
            return Ok(buf);
        }
        at += BLOCK_SIZE as u64;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn block_contains_end(block: &[u8]) -> bool {
    use crate::header::card::CARD_SIZE;
    block
        .chunks_exact(CARD_SIZE)
        .any(|c| c.starts_with(b"END") && c[3..].iter().all(|&b| b == b' '))
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
fn data_section_size(h: &Header) -> Result<u64> {
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

    let pcount = h.optional_int("PCOUNT").unwrap_or(0) as u64;
    let gcount = h.optional_int("GCOUNT").unwrap_or(1) as u64;

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
        let bytes = build_simple_f_no_data();
        let err = FitsFile::from_bytes(bytes).unwrap_err();
        assert!(matches!(err, FitsError::NonStandard(_)), "got {err:?}");
    }

    #[test]
    fn simple_f_lenient_accepted() {
        let bytes = build_simple_f_no_data();
        let f = FitsOpenOptions::new()
            .lenient(true)
            .from_bytes(bytes)
            .unwrap();
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
