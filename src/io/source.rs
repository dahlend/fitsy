//! Owning, in-memory FITS byte buffer, used by [`crate::FitsFile`].
//!
//! FITS data is big-endian, so a little-endian host changes the byte
//! order on read. That step allocates a fresh buffer, so a memory map
//! would yield no zero-copy read. Owning the bytes also keeps this
//! layer free of `unsafe`, and it returns an error rather than raising
//! SIGBUS when the file is truncated.

#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Read;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use crate::error::Result;
use crate::io::block::BLOCK_SIZE;

/// Owning, in-memory FITS buffer.
#[derive(Debug)]
pub struct ByteSource {
    buf: Vec<u8>,
}

impl ByteSource {
    /// Open `path` and read its whole contents into memory.
    ///
    /// Under the `compression` feature, a file that starts with the
    /// gzip magic bytes inflates here.
    ///
    /// # Errors
    ///
    /// - [`crate::FitsError::Io`] when `path` cannot be opened or
    ///   read, or when a gzip stream fails to inflate.
    /// - [`crate::FitsError::Block`] when the file is empty.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut f = fs::File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        Self::from_vec(buf)
    }

    /// Wrap an in-memory buffer.
    ///
    /// Under the `compression` feature, a buffer that starts with the
    /// gzip magic bytes inflates here.
    ///
    /// # Errors
    ///
    /// - [`crate::FitsError::Io`] when a gzip stream fails to inflate.
    /// - [`crate::FitsError::Block`] when `buf` is empty.
    pub fn from_vec(buf: Vec<u8>) -> Result<Self> {
        #[cfg(feature = "compression")]
        let buf = crate::compression::maybe_gunzip(buf)?;
        let buf = Self::pad_to_block(buf)?;
        Ok(Self { buf })
    }

    #[must_use]
    /// The whole buffer, zero-padded to a block boundary.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    fn validate_nonempty(n: usize) -> Result<()> {
        if n == 0 {
            return Err(crate::error::FitsError::Block {
                offset: 0,
                msg: "empty file".into(),
            });
        }
        Ok(())
    }

    /// Zero-pad an in-memory buffer to a multiple of `BLOCK_SIZE`.
    ///
    /// The standard ends every HDU on a 2880-byte boundary. Some
    /// observatory files truncate the last padding block instead, so
    /// this restores it rather than rejecting the file.
    fn pad_to_block(mut buf: Vec<u8>) -> Result<Vec<u8>> {
        Self::validate_nonempty(buf.len())?;
        let rem = buf.len() % BLOCK_SIZE;
        if rem != 0 {
            buf.resize(buf.len() + (BLOCK_SIZE - rem), 0);
        }
        Ok(buf)
    }
}
