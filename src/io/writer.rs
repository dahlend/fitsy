//! Sequential HDU writer (Standard Sec.3.1, Sec.3.3, Sec.4.4).
//!
//! [`FitsWriter`] streams one HDU at a time to any [`std::io::Write`]:
//!
//! 1. The header is rendered with [`Header::to_bytes`] (which always
//!    pads to a 2880-byte boundary and emits an `END` card).
//! 2. The data section is written verbatim, then padded to the next
//!    block boundary with the appropriate fill byte: ASCII space
//!    (`0x20`) for `XTENSION = 'TABLE   '`, otherwise zero bytes
//!    (Standard Sec.3.3.1, Sec.3.3.2).
//! 3. Mandatory keywords are sanity-checked: the first HDU must
//!    declare `SIMPLE = T`; subsequent HDUs must declare `XTENSION`.
//!
//! The writer can optionally compute and stamp `CHECKSUM`/`DATASUM`
//! cards on every HDU it emits -- see [`FitsWriter::with_checksums`].
//! When that mode is off, any `CHECKSUM`/`DATASUM` cards already in
//! the supplied header are emitted verbatim.

use std::io::{self, Write};

use crate::error::{FitsError, Result};
use crate::header::Header;
use crate::header::value::Value;
use crate::io::block::{BLOCK_SIZE, pad_to_block};

/// Streaming writer for a sequence of HDUs.
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

    /// Wrap a writer that is positioned just after an existing
    /// sequence of `hdu_count` HDUs. The next call to
    /// [`write_hdu`](Self::write_hdu) will be validated as an
    /// extension HDU (i.e. `XTENSION` required, `SIMPLE` rejected).
    /// Used by [`FitsAppender`](crate::FitsAppender).
    pub fn with_hdu_count(inner: W, hdu_count: usize) -> Self {
        Self {
            inner,
            hdu_count,
            stamp_checksums: false,
        }
    }

    /// Enable automatic computation and stamping of `CHECKSUM` and
    /// `DATASUM` cards on every HDU written. The header passed to
    /// [`write_hdu`](Self::write_hdu) does not need to contain
    /// placeholders -- the writer appends them itself if absent.
    #[must_use]
    pub fn with_checksums(mut self) -> Self {
        self.stamp_checksums = true;
        self
    }

    /// Write a single HDU. The header bytes and the padded data
    /// bytes are written in that order.
    ///
    /// `data` is the raw data section (no padding). The writer pads
    /// it to the next 2880-byte block on its own.
    ///
    /// The first HDU must be the primary HDU (`SIMPLE = T`);
    /// subsequent HDUs must declare `XTENSION`. The writer also
    /// verifies that `BITPIX`, `NAXIS`, `NAXISn`, and (for extensions)
    /// `PCOUNT`/`GCOUNT` are present and that `data.len()` matches
    /// the size implied by `NAXIS*` x `BITPIX` (+ heap, for
    /// `BINTABLE`).
    pub fn write_hdu(&mut self, header: &Header, data: &[u8]) -> Result<()> {
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
            tmp.to_bytes()?
        } else {
            header.to_bytes()?
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
                .map_err(|e| FitsError::Header(format!("stamp_checksum: {e}")))?;
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

    /// Append a raw, already-padded HDU (header + padded data) to
    /// the output. Used internally by `FitsFile.flush()` to stream
    /// untouched HDUs straight from the source file without a
    /// decode/encode round-trip. The caller is responsible for
    /// ensuring `bytes` is a complete, validly framed HDU.
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

    /// Flush any buffered output and return the inner writer.
    pub fn finish(mut self) -> io::Result<W> {
        self.inner.flush()?;
        Ok(self.inner)
    }
}

/// Write a sequence of HDUs to `path` in one call.
///
/// Convenience wrapper around [`FitsWriter`] for the common case of
/// "build a list of HDUs, dump them to a file." Each tuple is a
/// `(header, data)` pair as produced by
/// [`ImageBuilder::build`](crate::ImageBuilder::build),
/// [`BinTableBuilder::build`](crate::BinTableBuilder::build), or
/// [`AsciiTableBuilder::build`](crate::AsciiTableBuilder::build).
///
/// `overwrite = false` (the default in the Python wrapper) returns
/// an `io::Error::AlreadyExists` if the destination file is already
/// present; pass `true` to clobber.
///
/// # Example
///
/// ```no_run
/// use fitsy::{ImageBuilder, write};
///
/// let pixels: Vec<f32> = vec![0.0; 64 * 64];
/// let img = ImageBuilder::new(vec![64u64, 64], pixels)?
///     .primary(true)
///     .build()?;
/// write("out.fits", &[img], false)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write(
    path: impl AsRef<std::path::Path>,
    hdus: &[(Header, Vec<u8>)],
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
    for (header, data) in hdus {
        writer.write_hdu(header, data)?;
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
        // GCOUNT (even when both are zero/one).
        if !matches!(header.first("PCOUNT"), Some(Value::Integer(_))) {
            return Err(FitsError::Header(
                "extension HDU header is missing PCOUNT".into(),
            ));
        }
        if !matches!(header.first("GCOUNT"), Some(Value::Integer(_))) {
            return Err(FitsError::Header(
                "extension HDU header is missing GCOUNT".into(),
            ));
        }
    }

    // Sec.4.4.1.1: BITPIX, NAXIS, and NAXIS1..NAXISn are mandatory in
    // every HDU (including the primary). We accept only the values
    // BITPIX = +/-8, +/-16, +/-32, +/-64.
    let bitpix = match header.first("BITPIX") {
        Some(Value::Integer(b)) => *b,
        _ => {
            return Err(FitsError::Header(
                "HDU header is missing or has non-integer BITPIX".into(),
            ));
        }
    };
    if !matches!(bitpix, 8 | 16 | 32 | 64 | -32 | -64) {
        return Err(FitsError::Header(format!(
            "BITPIX = {bitpix} is not one of 8, 16, 32, 64, -32, -64"
        )));
    }
    let naxis = match header.first("NAXIS") {
        Some(Value::Integer(n)) if *n >= 0 => *n as usize,
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
            Some(Value::Integer(n)) if *n >= 0 => {}
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
/// Reproduces the Sec.4.4.1.1/Sec.7 data-section formula:
///   `|BITPIX|/8 * GCOUNT * (PCOUNT + Pi NAXISn)` for `NAXIS > 0`,
///   `0` for `NAXIS = 0` or any `NAXISn = 0`. The primary HDU is
///   treated as `PCOUNT=0, GCOUNT=1` per Sec.4.4.1.1.
fn validate_data_size(header: &Header, data_len: usize) -> Result<()> {
    let bitpix = match header.first("BITPIX") {
        Some(Value::Integer(b)) => *b,
        // Already reported by validate_mandatory.
        _ => return Ok(()),
    };
    let naxis = match header.first("NAXIS") {
        Some(Value::Integer(n)) if *n >= 0 => *n as usize,
        _ => return Ok(()),
    };
    let pcount = match header.first("PCOUNT") {
        Some(Value::Integer(p)) if *p >= 0 => *p as u64,
        _ => 0,
    };
    let gcount = match header.first("GCOUNT") {
        Some(Value::Integer(g)) if *g >= 1 => *g as u64,
        _ => 1,
    };
    let mut prod: u64 = u64::from(naxis != 0);
    for i in 1..=naxis {
        let n = match header.first(&format!("NAXIS{i}")) {
            Some(Value::Integer(n)) if *n >= 0 => *n as u64,
            _ => return Ok(()),
        };
        if n == 0 {
            prod = 0;
            break;
        }
        prod = prod
            .checked_mul(n)
            .ok_or_else(|| FitsError::Header("NAXISn product overflowed u64".into()))?;
    }
    let bytes_per_elt = bitpix.unsigned_abs() / 8;
    let expected = if prod == 0 {
        0
    } else {
        bytes_per_elt
            .checked_mul(gcount)
            .and_then(|v| v.checked_mul(prod + pcount))
            .ok_or_else(|| FitsError::Header("data size overflowed u64".into()))?
    };
    if expected != data_len as u64 {
        return Err(FitsError::Header(format!(
            "data section is {data_len} bytes but header declares {expected} bytes \
             (BITPIX={bitpix}, NAXIS={naxis}, PCOUNT={pcount}, GCOUNT={gcount})"
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
        w.write_hdu(&h, &[]).unwrap();
        w.finish().unwrap();
        assert_eq!(buf.len(), BLOCK_SIZE);
        assert_eq!(&buf[..6], b"SIMPLE");
    }

    #[test]
    fn primary_without_simple_rejected() {
        let h = Header::empty();
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        assert!(w.write_hdu(&h, &[]).is_err());
    }

    #[test]
    fn writes_data_and_pads_to_block() {
        let h = primary(1, &[7]);
        let data = vec![0xAA_u8; 7];
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu(&h, &data).unwrap();
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
        w.write_hdu(&primary_h, &[]).unwrap();
        w.write_hdu(&h, b"abc").unwrap();
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
        w.write_hdu(&primary_h, &[]).unwrap();
        assert!(w.write_hdu(&bogus, &[]).is_err());
    }

    #[test]
    fn checksum_stamping_round_trips() {
        let h = primary(1, &[7]);
        let data = vec![0xAA_u8; 7];
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf).with_checksums();
        w.write_hdu(&h, &data).unwrap();
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
        let err = w.write_hdu(&h, &[0_u8; 7]).unwrap_err();
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
        w.write_hdu(&primary_h, &[]).unwrap();
        let err = w.write_hdu(&h, &[]).unwrap_err();
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
        let err = w.write_hdu(&h, &[]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("BITPIX"), "got: {msg}");
    }
}
