//! Sequential HDU writer (Standard Sec.3.1, Sec.3.3, Sec.4.4).
//!
//! # Purpose
//!
//! [`FitsWriter`] streams one HDU at a time to any
//! [`std::io::Write`]. [`write()`] wraps it for the common case of
//! dumping a list of HDUs to a path.
//!
//! # Layout
//!
//! Each call to [`FitsWriter::write_hdu`] runs three steps:
//!
//! 1. It checks the mandatory keywords. The first HDU must declare
//!    `SIMPLE = T`, and each later HDU must declare `XTENSION`.
//! 2. It renders the header with [`Header::to_bytes`], which pads to a
//!    2880-byte boundary and emits an `END` card.
//! 3. It writes the data section verbatim, then pads to the next block
//!    boundary. The fill byte is an ASCII space for
//!    `XTENSION = 'TABLE   '`, and a zero byte otherwise (Standard
//!    Sec.3.3.1, Sec.3.3.2).
//!
//! # Design constraints
//!
//! [`FitsWriter::with_checksums`] makes the writer compute and stamp a
//! `CHECKSUM` and a `DATASUM` card on every HDU. Without it, a
//! `CHECKSUM` or `DATASUM` card already in the supplied header goes
//! out verbatim, and the writer computes neither.

use std::io::{self, Write};

use crate::error::{FitsError, Result};
use crate::hdu::HduBytes;
use crate::header::Header;
use crate::header::value::Value;
use crate::io::block::{BLOCK_SIZE, pad_to_block};

/// Streaming writer for a sequence of HDUs.
///
/// # Examples
///
/// ```
/// use fitsy::{FitsWriter, ImageBuilder};
///
/// let hdu = ImageBuilder::new(vec![2_u64, 2], vec![1.0_f32; 4])?
///     .primary(true)
///     .build()?;
///
/// let mut buf: Vec<u8> = Vec::new();
/// let mut w = FitsWriter::new(&mut buf);
/// w.write_hdu(&hdu)?;
/// assert_eq!(w.hdu_count(), 1);
/// w.finish()?;
///
/// // Every HDU is padded to the 2880-byte block.
/// assert_eq!(buf.len() % 2880, 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct FitsWriter<W: Write> {
    inner: W,
    hdu_count: usize,
    stamp_checksums: bool,
}

impl<W: Write> FitsWriter<W> {
    /// Wrap an arbitrary writer.
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            hdu_count: 0,
            stamp_checksums: false,
        }
    }

    /// Wrap a writer that already sits just past a sequence of
    /// `hdu_count` HDUs.
    ///
    /// The `inner` argument is that writer, positioned at the byte
    /// after the last HDU. The `hdu_count` argument is how many HDUs
    /// precede that point.
    ///
    /// The next call to [`write_hdu`](Self::write_hdu) validates its
    /// header as an extension, so it requires `XTENSION` and rejects
    /// `SIMPLE`. [`FitsAppender`](crate::FitsAppender) uses this.
    pub fn with_hdu_count(inner: W, hdu_count: usize) -> Self {
        Self {
            inner,
            hdu_count,
            stamp_checksums: false,
        }
    }

    /// Compute and stamp a `CHECKSUM` and a `DATASUM` card on every
    /// HDU that this writer emits.
    ///
    /// The header passed to [`write_hdu`](Self::write_hdu) needs no
    /// placeholder card. The writer appends one when it is absent.
    #[must_use]
    pub fn with_checksums(mut self) -> Self {
        self.stamp_checksums = true;
        self
    }

    /// Write one HDU: the header bytes, then the padded data bytes.
    ///
    /// The `hdu` argument is anything that pairs a header with a data
    /// section: what a builder returns, an [`Hdu`](crate::Hdu) read
    /// from a file, or a `(Header, Vec<u8>)` pair. The padding to the
    /// next 2880-byte block is added here.
    ///
    /// # Errors
    ///
    /// [`FitsError::Header`] in five cases:
    ///
    /// - The first HDU does not declare `SIMPLE = T`.
    /// - A later HDU does not declare `XTENSION`.
    /// - `BITPIX`, `NAXIS` or a `NAXISn` card is absent.
    /// - An extension omits `PCOUNT` or `GCOUNT`.
    /// - `data.len()` does not match the size those keywords imply.
    ///
    /// [`FitsError::Io`] when the write fails.
    pub fn write_hdu(&mut self, hdu: &impl HduBytes) -> Result<()> {
        self.write_hdu_parts(hdu.header(), hdu.data_bytes())
    }

    /// Write one HDU from a header and a data section held apart.
    ///
    /// [`write_hdu`](Self::write_hdu) is the usual call. Use this one
    /// when the two are not paired in a type.
    ///
    /// # Errors
    ///
    /// The conditions of [`write_hdu`](Self::write_hdu).
    pub fn write_hdu_parts(&mut self, header: &Header, data: &[u8]) -> Result<()> {
        let is_primary = self.hdu_count == 0;
        validate_mandatory(header, is_primary)?;
        validate_data_size(header, data.len())?;

        let mut header_bytes = if self.stamp_checksums {
            // Inject placeholders if missing, then serialize.
            let mut tmp = header.clone();
            if !tmp.contains("DATASUM") {
                tmp.push(
                    "DATASUM",
                    Value::String("0".into()),
                    Some("data unit checksum"),
                )?;
            }
            if !tmp.contains("CHECKSUM") {
                tmp.push(
                    "CHECKSUM",
                    Value::String("0000000000000000".into()),
                    Some("HDU checksum"),
                )?;
            }
            tmp.to_bytes()
        } else {
            header.to_bytes()
        };
        debug_assert!(
            header_bytes.len().is_multiple_of(BLOCK_SIZE),
            "header must be block-aligned ({} bytes)",
            header_bytes.len()
        );

        let pad_byte = pad_byte_for(header);
        let padded_len = pad_to_block(data.len() as u64) as usize;
        let mut padded_data = Vec::with_capacity(padded_len);
        padded_data.extend_from_slice(data);
        padded_data.resize(padded_len, pad_byte);

        if self.stamp_checksums {
            crate::checksum::stamp_checksum(&mut header_bytes, &padded_data)
                .map_err(|e| FitsError::Header(format!("checksum stamp failed: {e}")))?;
        }

        self.inner.write_all(&header_bytes)?;
        self.inner.write_all(&padded_data)?;

        self.hdu_count += 1;
        Ok(())
    }

    /// Number of HDUs written so far.
    pub fn hdu_count(&self) -> usize {
        self.hdu_count
    }

    /// Append a raw, already-padded HDU, meaning its header followed
    /// by its padded data.
    ///
    /// The Python `FitsFile.flush()` uses this to stream an untouched
    /// HDU from the source file without decoding and re-encoding it.
    /// The caller guarantees that `bytes` frames one complete HDU.
    ///
    /// # Errors
    ///
    /// [`io::Error`] when `bytes.len()` is not a multiple of 2880, or
    /// when the write fails.
    pub fn write_raw_padded(&mut self, bytes: &[u8]) -> io::Result<()> {
        if !bytes.len().is_multiple_of(BLOCK_SIZE) {
            return Err(io::Error::other(format!(
                "write_raw_padded: HDU bytes ({}) not block-aligned",
                bytes.len()
            )));
        }
        self.inner.write_all(bytes)?;
        self.hdu_count += 1;
        Ok(())
    }

    /// Flush buffered output and return the inner writer.
    ///
    /// # Errors
    ///
    /// [`io::Error`] when the flush fails.
    pub fn finish(mut self) -> io::Result<W> {
        self.inner.flush()?;
        Ok(self.inner)
    }
}

/// Write a sequence of HDUs to `path` in one call.
///
/// This wraps [`FitsWriter`] for the common case: build a list of
/// HDUs, then write them to a file. Each tuple is a `(header, data)`
/// pair, as
/// [`ImageBuilder::build`](crate::ImageBuilder::build),
/// [`BinTableBuilder::build`](crate::BinTableBuilder::build) and
/// [`AsciiTableBuilder::build`](crate::AsciiTableBuilder::build)
/// produce.
///
/// An `overwrite` value of `false` fails when `path` exists. Pass
/// `true` to replace the file.
///
/// # Errors
///
/// - [`FitsError::Header`] when `hdus` is empty, or when an HDU fails
///   the checks that [`FitsWriter::write_hdu`] applies.
/// - [`FitsError::Io`] when `path` cannot be created or written. An
///   `overwrite` value of `false` gives
///   [`std::io::ErrorKind::AlreadyExists`] when `path` exists.
///
/// # Example
///
/// ```
/// use fitsy::{ImageBuilder, write};
///
/// # let path = std::env::temp_dir().join("fitsy_doc_write_fn.fits");
/// let pixels: Vec<f32> = vec![0.0; 64 * 64];
/// let img = ImageBuilder::new(vec![64_u64, 64], pixels)?
///     .primary(true)
///     .build()?;
/// write(&path, &[img], true)?;
///
/// # assert_eq!(std::fs::metadata(&path)?.len() % 2880, 0);
/// # std::fs::remove_file(&path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write(
    path: impl AsRef<std::path::Path>,
    hdus: &[impl HduBytes],
    overwrite: bool,
) -> Result<()> {
    use std::fs::OpenOptions;
    use std::io::BufWriter;

    if hdus.is_empty() {
        return Err(FitsError::Header(
            "fitsy::write: cannot write a FITS file with zero HDUs".into(),
        ));
    }
    let mut opts = OpenOptions::new();
    opts.write(true).create(true);
    if overwrite {
        opts.truncate(true);
    } else {
        opts.create_new(true);
    }
    let file = opts.open(path.as_ref())?;
    let mut writer = FitsWriter::new(BufWriter::new(file));
    for hdu in hdus {
        writer.write_hdu(hdu)?;
    }
    writer.finish()?;
    Ok(())
}

fn validate_mandatory(header: &Header, is_primary: bool) -> Result<()> {
    if is_primary {
        match header.first("SIMPLE") {
            Some(Value::Logical(true)) => {}
            Some(Value::Logical(false)) => {
                return Err(FitsError::Header(
                    "primary HDU has SIMPLE = F (non-conforming files cannot be written)".into(),
                ));
            }
            _ => {
                return Err(FitsError::Header(
                    "primary HDU header is missing SIMPLE = T".into(),
                ));
            }
        }
    } else {
        match header.first("XTENSION") {
            Some(Value::String(_)) => {}
            _ => {
                return Err(FitsError::Header(
                    "extension HDU header is missing XTENSION".into(),
                ));
            }
        }
        // Sec.7.1.1: every conforming extension must declare PCOUNT and
        // GCOUNT (even when both are zero/one). Accept what the lenient
        // reader accepts (integral reals like `PCOUNT = 0.`) so every
        // openable file is also writable.
        if header.optional_int("PCOUNT").is_none() {
            return Err(FitsError::Header(
                "extension HDU header is missing or has non-integer PCOUNT".into(),
            ));
        }
        if header.optional_int("GCOUNT").is_none() {
            return Err(FitsError::Header(
                "extension HDU header is missing or has non-integer GCOUNT".into(),
            ));
        }
    }

    // Sec.4.4.1.1: BITPIX, NAXIS, and NAXIS1..NAXISn are mandatory in
    // every HDU (including the primary). We accept only the values
    // BITPIX = +/-8, +/-16, +/-32, +/-64.
    let Some(Value::Integer(bitpix)) = header.first("BITPIX") else {
        return Err(FitsError::Header(
            "HDU header is missing or has non-integer BITPIX".into(),
        ));
    };
    if !matches!(bitpix, 8 | 16 | 32 | 64 | -32 | -64) {
        return Err(FitsError::Header(format!(
            "BITPIX = {bitpix} is not one of 8, 16, 32, 64, -32, -64"
        )));
    }
    let naxis = match header.first("NAXIS") {
        Some(Value::Integer(n)) if n >= 0 => n as usize,
        Some(Value::Integer(n)) => {
            return Err(FitsError::Header(format!("NAXIS = {n} is negative")));
        }
        _ => {
            return Err(FitsError::Header(
                "HDU header is missing or has non-integer NAXIS".into(),
            ));
        }
    };
    for i in 1..=naxis {
        let key = format!("NAXIS{i}");
        match header.first(&key) {
            Some(Value::Integer(n)) if n >= 0 => {}
            Some(Value::Integer(n)) => {
                return Err(FitsError::Header(format!("{key} = {n} is negative")));
            }
            _ => {
                return Err(FitsError::Header(format!(
                    "HDU header is missing or has non-integer {key}"
                )));
            }
        }
    }
    Ok(())
}

/// Verify that `data_len` matches what the header declares.
///
/// Uses the same Sec.4.4.1.1/Sec.6/Sec.7 formula as the reader
/// (`data_section_size`), including the Random Groups `NAXIS1 = 0`
/// convention -- everything the reader opens must round-trip through
/// the writer.
fn validate_data_size(header: &Header, data_len: usize) -> Result<()> {
    // Malformed BITPIX/NAXIS is already reported by validate_mandatory.
    let Ok(expected) = crate::hdu::file::data_section_size(header) else {
        return Ok(());
    };
    if expected != data_len as u64 {
        return Err(FitsError::Header(format!(
            "data section is {data_len} bytes but header declares {expected} bytes"
        )));
    }
    Ok(())
}

/// Per Standard Sec.3.3.2 ASCII tables pad with spaces; everything else
/// pads with zeroes.
fn pad_byte_for(header: &Header) -> u8 {
    if let Some(Value::String(x)) = header.first("XTENSION")
        && x.trim_end() == "TABLE"
    {
        return b' ';
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::value::Value;

    fn primary(naxis: i64, axes: &[i64]) -> Header {
        let mut h = Header::empty();
        h.push("SIMPLE", Value::Logical(true), None).unwrap();
        h.push("BITPIX", Value::Integer(8), None).unwrap();
        h.push("NAXIS", Value::Integer(naxis), None).unwrap();
        for (i, n) in axes.iter().enumerate() {
            h.push(format!("NAXIS{}", i + 1), Value::Integer(*n), None)
                .unwrap();
        }
        h
    }

    #[test]
    fn write_empty_primary() {
        let h = primary(0, &[]);
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu_parts(&h, &[]).unwrap();
        w.finish().unwrap();
        assert_eq!(buf.len(), BLOCK_SIZE);
        assert_eq!(&buf[..6], b"SIMPLE");
    }

    #[test]
    fn primary_without_simple_rejected() {
        let h = Header::empty();
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        assert!(w.write_hdu_parts(&h, &[]).is_err());
    }

    #[test]
    fn writes_data_and_pads_to_block() {
        let h = primary(1, &[7]);
        let data = vec![0xAA_u8; 7];
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu_parts(&h, &data).unwrap();
        w.finish().unwrap();
        // Header (1 block) + data block.
        assert_eq!(buf.len(), 2 * BLOCK_SIZE);
        assert_eq!(&buf[BLOCK_SIZE..BLOCK_SIZE + 7], &data[..]);
        // Padding is zero.
        assert!(buf[BLOCK_SIZE + 7..].iter().all(|&b| b == 0));
    }

    #[test]
    fn ascii_table_pads_with_spaces() {
        // First HDU must be primary.
        let primary_h = primary(0, &[]);
        let mut h = Header::empty();
        h.push("XTENSION", Value::String("TABLE".into()), None)
            .unwrap();
        h.push("BITPIX", Value::Integer(8), None).unwrap();
        h.push("NAXIS", Value::Integer(2), None).unwrap();
        h.push("NAXIS1", Value::Integer(3), None).unwrap();
        h.push("NAXIS2", Value::Integer(1), None).unwrap();
        h.push("PCOUNT", Value::Integer(0), None).unwrap();
        h.push("GCOUNT", Value::Integer(1), None).unwrap();
        h.push("TFIELDS", Value::Integer(0), None).unwrap();

        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu_parts(&primary_h, &[]).unwrap();
        w.write_hdu_parts(&h, b"abc").unwrap();
        w.finish().unwrap();
        // Last 2880 - 3 bytes should be ASCII spaces.
        let tail = &buf[buf.len() - (BLOCK_SIZE - 3)..];
        assert!(tail.iter().all(|&b| b == b' '));
    }

    #[test]
    fn extension_without_xtension_rejected() {
        let primary_h = primary(0, &[]);
        let bogus = Header::empty();
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu_parts(&primary_h, &[]).unwrap();
        assert!(w.write_hdu_parts(&bogus, &[]).is_err());
    }

    #[test]
    fn checksum_stamping_round_trips() {
        let h = primary(1, &[7]);
        let data = vec![0xAA_u8; 7];
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf).with_checksums();
        w.write_hdu_parts(&h, &data).unwrap();
        w.finish().unwrap();
        // Verify via the high-level reader.
        let parsed = crate::FitsFile::from_bytes(buf).unwrap();
        let report = parsed.verify_checksums().unwrap();
        assert_eq!(report.len(), 1);
        // checksum + datasum must both be present and verify.
        let r = &report[0];
        assert_eq!(r.checksum_ok, Some(true), "CHECKSUM did not verify: {r:?}");
        assert_eq!(r.datasum_ok, Some(true), "DATASUM did not verify: {r:?}");
    }

    #[test]
    fn rejects_data_size_mismatch() {
        // Header declares NAXIS1 = 10 (10 bytes), data only 7.
        let h = primary(1, &[10]);
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        let err = w.write_hdu_parts(&h, &[0_u8; 7]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("data section"), "got: {msg}");
    }

    #[test]
    fn rejects_extension_missing_pcount_gcount() {
        let primary_h = primary(0, &[]);
        let mut h = Header::empty();
        h.push("XTENSION", Value::String("IMAGE".into()), None)
            .unwrap();
        h.push("BITPIX", Value::Integer(8), None).unwrap();
        h.push("NAXIS", Value::Integer(0), None).unwrap();
        // PCOUNT + GCOUNT deliberately omitted.
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu_parts(&primary_h, &[]).unwrap();
        let err = w.write_hdu_parts(&h, &[]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("PCOUNT"), "got: {msg}");
    }

    #[test]
    fn rejects_invalid_bitpix() {
        let mut h = Header::empty();
        h.push("SIMPLE", Value::Logical(true), None).unwrap();
        h.push("BITPIX", Value::Integer(7), None).unwrap();
        h.push("NAXIS", Value::Integer(0), None).unwrap();
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        let err = w.write_hdu_parts(&h, &[]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("BITPIX"), "got: {msg}");
    }
}
