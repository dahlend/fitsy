//! Streaming HDU appender (Standard Sec.3.1, Sec.4.4).
//!
//! [`FitsAppender`] seeks past the last HDU of an existing file and
//! writes further extension HDUs in place, copying nothing.
//!
//! Every appended HDU must be an extension: appending a primary HDU
//! is an error.
//!
//! # Example
//!
//! ```no_run
//! use fitsy::{FitsAppender, ImageBuilder};
//!
//! let pixels = vec![0u8; 32 * 32];
//! let (header, data) = ImageBuilder::new(vec![32u64, 32], pixels)?
//!     .build()?;
//!
//! let mut app = FitsAppender::open("existing.fits")?;
//! app.append_hdu(&header, &data)?;
//! app.finish()?;
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
    /// Open `path` for append. The file is parsed in full to
    /// validate it and to locate the byte offset just after the
    /// last HDU's padded data; the file is then re-opened in
    /// read/write mode for streaming writes.
    ///
    /// Opening does not modify the file. Any bytes after the last HDU
    /// are kept until something is appended over them.
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

    /// Append a single HDU. Same validation as
    /// [`FitsWriter::write_hdu`]; the HDU must declare `XTENSION`
    /// (primary HDUs cannot be appended).
    pub fn append_hdu(&mut self, header: &Header, data: &[u8]) -> Result<()> {
        self.inner.write_hdu(header, data)
    }

    /// Number of HDUs that existed before this appender was opened.
    #[must_use]
    pub fn initial_hdu_count(&self) -> usize {
        self.initial_hdu_count
    }

    /// Number of HDUs in the file after the appends performed so
    /// far (initial count plus successful `append_hdu` calls).
    #[must_use]
    pub fn hdu_count(&self) -> usize {
        self.inner.hdu_count()
    }

    /// Flush, sync, and close. Returns the number of HDUs now in
    /// the file.
    ///
    /// If anything was appended the file is truncated to the end of
    /// the last HDU, so no fragment of the old tail survives. If
    /// nothing was, the file is left untouched.
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
