//! Owning, in-memory FITS byte buffer used by [`crate::FitsFile`].
//!
//! FITS stores numeric data big-endian, so on every modern
//! little-endian host the read path must byteswap to deliver a
//! native-dtype `numpy.ndarray`. Byteswapping inherently allocates a
//! fresh buffer, so there is no zero-copy advantage to be had from
//! mapping file bytes -- the bytes get copied on the way out
//! regardless. Owning the bytes keeps the I/O layer free of
//! `unsafe` and immune to SIGBUS from external truncation.

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
    /// Open `path` and read its entire contents into memory.
    /// If the `compression` feature is enabled and the file begins
    /// with the gzip magic, it is transparently inflated.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut f = fs::File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        Self::from_vec(buf)
    }

    /// Wrap an in-memory buffer. If the `compression` feature is
    /// enabled and the buffer begins with the gzip magic, it is
    /// transparently inflated.
    pub fn from_vec(buf: Vec<u8>) -> Result<Self> {
        #[cfg(feature = "compression")]
        let buf = crate::compression::maybe_gunzip(buf)?;
        let buf = Self::pad_to_block(buf)?;
        Ok(Self { buf })
    }

    #[must_use]
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
    /// FITS requires HDUs to end on a 2880-byte boundary, but real
    /// observatory files (e.g. CFHT/MegaCam) sometimes truncate the
    /// last padding block. CFITSIO and astropy silently pad-on-read,
    /// so we do too.
    fn pad_to_block(mut buf: Vec<u8>) -> Result<Vec<u8>> {
        Self::validate_nonempty(buf.len())?;
        let rem = buf.len() % BLOCK_SIZE;
        if rem != 0 {
            buf.resize(buf.len() + (BLOCK_SIZE - rem), 0);
        }
        Ok(buf)
    }
}
