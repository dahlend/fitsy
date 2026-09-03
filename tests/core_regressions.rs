//! Regression tests for core I/O fixes: each of these reproduced a real
//! bug (scaled TNULL ignored, CONTINUE quote-split corruption, Random
//! Groups unwritable, lenient-read/strict-write asymmetry, tail-byte
//! rejection in `Header::parse`).

use fitsy::io::writer::FitsWriter;
use fitsy::{BinValue, FitsFile, Hdu, Header, Value};

/// Re-serialize every HDU of `f` to bytes (the same loop as
/// `FitsFile::write`, without touching the filesystem).
fn write_to_bytes(f: &FitsFile) -> fitsy::error::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    for i in 0..f.len() {
        let hdu = f.hdu(i)?;
        w.write_hdu(&hdu)?;
    }
    w.finish()?;
    Ok(buf)
}

const CARD: usize = 80;
const BLOCK: usize = 2880;

fn pad_card(s: &str) -> [u8; CARD] {
    let mut b = [b' '; CARD];
    assert!(s.len() <= CARD, "card too long: {s}");
    b[..s.len()].copy_from_slice(s.as_bytes());
    b
}

fn pad_to_block(buf: &mut Vec<u8>, fill: u8) {
    while !buf.len().is_multiple_of(BLOCK) {
        buf.push(fill);
    }
}

fn empty_primary() -> Vec<u8> {
    let cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
        "END",
    ];
    let mut buf = Vec::new();
    for c in cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    buf
}

/// TNULL marks the *stored* value of an undefined cell even when the
/// column is scaled (Sec.7.3.3.2): the sentinel must surface as NaN, not
/// be scaled into a plausible physical number.
#[test]
fn scaled_int_column_honors_tnull() {
    let mut buf = empty_primary();
    let cards = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                    4",
        "NAXIS2  =                    2",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    2",
        "TTYPE1  = 'V       '",
        "TFORM1  = '1I      '",
        "TSCAL1  =                  2.0",
        "TZERO1  =                 10.0",
        "TNULL1  =                 -999",
        "TTYPE2  = 'W       '",
        "TFORM2  = '1I      '",
        "END",
    ];
    for c in cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    for v in [-999_i16, 0, 7, 0] {
        buf.extend_from_slice(&v.to_be_bytes());
    }
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::BinTable(t) = f.hdu(1).unwrap() else {
        panic!("expected BINTABLE");
    };
    let col = t.column_by_name("V").unwrap();
    let BinValue::Float(row0) = t.cell_value(0, col).unwrap() else {
        panic!("expected scaled column to decode as Float");
    };
    assert!(row0[0].is_nan(), "TNULL sentinel decoded as {}", row0[0]);
    let BinValue::Float(row1) = t.cell_value(1, col).unwrap() else {
        panic!("expected scaled column to decode as Float");
    };
    assert_eq!(row1[0], 10.0 + 2.0 * 7.0);
}

/// A long string whose escaped `''` would straddle a CONTINUE chunk
/// boundary must not be split mid-escape: `to_bytes` output has to
/// re-parse to the exact original value in both strict and lenient mode.
#[test]
fn continue_chunking_never_splits_quote_escape() {
    // A quote at every offset near the chunk boundary, and a pathological
    // all-quotes value.
    let mut values: Vec<String> = (60..=72)
        .map(|n| format!("{}'{}", "x".repeat(n), "y".repeat(50)))
        .collect();
    values.push("'".repeat(150));
    for val in values {
        let mut h = Header::empty();
        h.push("SVAL", Value::String(val.clone()), None).unwrap();
        let bytes = h.to_bytes();
        for lenient in [false, true] {
            let (parsed, _) = Header::parse_with(&bytes, 0, lenient)
                .unwrap_or_else(|e| panic!("reparse (lenient={lenient}) failed: {e}"));
            match parsed.first("SVAL") {
                Some(Value::String(s)) => {
                    assert_eq!(s, val, "value corrupted (lenient={lenient})");
                }
                other => panic!("SVAL parsed as {other:?}"),
            }
        }
    }
}

/// Every Random Groups file the reader accepts must also be writable:
/// the writer's size check has to use the Sec.6 formula (NAXIS1 = 0 is
/// the RG marker, not an empty data section).
#[test]
fn random_groups_round_trips_through_writer() {
    let cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    2",
        "NAXIS1  =                    0",
        "NAXIS2  =                    3",
        "GROUPS  =                    T",
        "PCOUNT  =                    2",
        "GCOUNT  =                    4",
        "END",
    ];
    let mut buf = Vec::new();
    for c in cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    // 4 groups x (2 params + 3 data) x 2 bytes = 40 bytes.
    for i in 0..20_i16 {
        buf.extend_from_slice(&i.to_be_bytes());
    }
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf.clone()).expect("reader accepts RG");
    let out = write_to_bytes(&f).expect("writer must accept RG too");
    let f2 = FitsFile::from_bytes(out).unwrap();
    assert_eq!(f2.len(), 1);
}

/// A header the lenient reader accepts (`PCOUNT = 0.` as a real) must be
/// writable, not rejected as "missing PCOUNT".
#[test]
fn lenient_real_pcount_is_writable() {
    let mut buf = empty_primary();
    let cards = [
        "XTENSION= 'IMAGE   '",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "PCOUNT  =                   0.",
        "GCOUNT  =                    1",
        "END",
    ];
    for c in cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');

    let f = FitsFile::from_bytes(buf).expect("lenient open");
    write_to_bytes(&f).expect("write must accept what the lenient reader accepted");
}

/// `Header::parse` on a buffer with a non-block-multiple tail after a
/// valid header must succeed -- only full blocks are scanned.
#[test]
fn header_parse_tolerates_partial_trailing_block() {
    let mut buf = empty_primary();
    Header::parse(&buf, 0).expect("exact block parses");
    buf.extend_from_slice(b"0123456789");
    Header::parse(&buf, 0).expect("valid header + stray tail bytes parses");
}

/// 6700417 * 2753074036095 == `u64::MAX`, so rounding the data section up
/// to a block boundary used to overflow instead of erroring.
#[test]
fn absurd_naxis_product_is_rejected_without_panicking() {
    let mut buf = Vec::new();
    for c in [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =              6700417",
        "NAXIS2  =        2753074036095",
        "END",
    ] {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    assert!(
        FitsFile::from_bytes(buf).is_err(),
        "a data section larger than the file must be an error"
    );
}
