//! Streaming HDU appender (Standard Sec.3.1, Sec.4.4).
//!
//! [`FitsAppender`] seeks past the last HDU of an existing file and
//! writes further extension HDUs there. It copies nothing.
//!
//! Every appended HDU must be an extension. A header that declares
//! `SIMPLE` rather than `XTENSION` is an error, because a file holds
//! one primary HDU and it already has one.
//!
//! # Example
//!
//! ```
//! # use fitsy::FitsWriter;
//! # let path = std::env::temp_dir().join("fitsy_doc_append.fits");
//! # let (h, d) = fitsy::ImageBuilder::new(vec![4_u64, 4], vec![0_u8; 16])?
//! #     .primary(true)
//! #     .build()?;
//! # let mut out = std::fs::File::create(&path)?;
//! # FitsWriter::new(&mut out).write_hdu(&h, &d)?;
//! # drop(out);
//! use fitsy::{FitsAppender, ImageBuilder};
//!
//! let pixels = vec![0_u8; 32 * 32];
//! let (header, data) = ImageBuilder::new(vec![32_u64, 32], pixels)?
//!     .build()?;
//!
//! let mut app = FitsAppender::open(&path)?;
//! app.append_hdu(&header, &data)?;
//! assert_eq!(app.finish()?, 2);
//! # std::fs::remove_file(&path)?;
//! # Ok::<(), fitsy::FitsError>(())
//! ```

use std::fs::OpenOptions;
use std::io::{BufWriter, Seek, SeekFrom};
use std::path::Path;

use crate::error::{FitsError, Result};
use crate::hdu::file::FitsFile;
use crate::header::Header;
use crate::io::writer::FitsWriter;

/// Streaming appender that adds HDUs to the end of an existing
/// FITS file without copying its contents.
///
/// Call [`finish`](Self::finish) when done: it is what syncs the
/// writes and trims any tail the new HDUs did not cover. Dropping the
/// appender instead flushes but does neither.
#[derive(Debug)]
pub struct FitsAppender {
    inner: FitsWriter<BufWriter<std::fs::File>>,
    initial_hdu_count: usize,
}

impl FitsAppender {
    /// Open `path` for append.
    ///
    /// This parses the whole file, to validate it and to find the byte
    /// offset just past the padded data of the last HDU. It then
    /// re-opens the file for reading and writing.
    ///
    /// Opening changes no byte of the file. Bytes after the last HDU
    /// survive until an append writes over them.
    ///
    /// # Errors
    ///
    /// - The conditions of
    ///   [`FitsFile::open`](crate::FitsFile::open), because this
    ///   parses the file first.
    /// - [`FitsError::Io`] when `path` cannot be re-opened for
    ///   writing, or when the seek fails.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        // Parse the file to discover its HDU count and end-of-data
        // offset.
        let f = FitsFile::open(path)?;
        let initial_hdu_count = f.len();
        let end = f.byte_len();
        drop(f);

        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        file.seek(SeekFrom::Start(end))?;

        let inner = FitsWriter::with_hdu_count(BufWriter::new(file), initial_hdu_count);
        Ok(Self {
            inner,
            initial_hdu_count,
        })
    }

    /// Append one HDU.
    ///
    /// The `header` argument describes the HDU, and the `data`
    /// argument holds its raw data section with no padding.
    ///
    /// This applies the same validation as
    /// [`FitsWriter::write_hdu`]. The header must declare `XTENSION`,
    /// because a file holds one primary HDU and it already has one.
    ///
    /// # Errors
    ///
    /// The conditions of [`FitsWriter::write_hdu`].
    pub fn append_hdu(&mut self, header: &Header, data: &[u8]) -> Result<()> {
        self.inner.write_hdu(header, data)
    }

    /// Number of HDUs that existed before this appender was opened.
    #[must_use]
    pub fn initial_hdu_count(&self) -> usize {
        self.initial_hdu_count
    }

    /// Number of HDUs in the file after the appends so far. This is
    /// the initial count plus each successful `append_hdu` call.
    #[must_use]
    pub fn hdu_count(&self) -> usize {
        self.inner.hdu_count()
    }

    /// Flush, sync and close. The result is the number of HDUs now in
    /// the file.
    ///
    /// After an append, this truncates the file to the end of the last
    /// HDU, so no fragment of the previous tail survives. Without an
    /// append, this changes no byte of the file.
    ///
    /// # Errors
    ///
    /// [`FitsError::Io`] when the flush, the truncation or the sync
    /// fails.
    pub fn finish(self) -> Result<usize> {
        let n = self.inner.hdu_count();
        let appended = n > self.initial_hdu_count;
        let buf = self.inner.finish().map_err(FitsError::Io)?;
        let mut file = buf
            .into_inner()
            .map_err(|e| FitsError::Io(e.into_error()))?;
        if appended {
            let end = file.stream_position()?;
            file.set_len(end)?;
        }
        file.sync_data().map_err(FitsError::Io)?;
        Ok(n)
    }
}
