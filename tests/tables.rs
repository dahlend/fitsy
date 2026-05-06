//! End-to-end Phase 3 tests: build synthetic ASCII and Binary table
//! HDUs, parse them through `FitsFile`, and verify column decoding +
//! VLA heap dereferencing.

use fitsy::{AsciiCell, BinValue, FitsFile, Hdu};

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

#[test]
fn ascii_table_round_trip() {
    // Two columns: NAME (A8) at TBCOL=1, MAG (F8.3) at TBCOL=10.
    // Row width = 17 bytes, 3 rows.
    let row_size: usize = 17;
    let n_rows: usize = 3;
    let header_cards = [
        "XTENSION= 'TABLE   '",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        &format!("NAXIS1  = {row_size:>20}"),
        &format!("NAXIS2  = {n_rows:>20}"),
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    2",
        "TTYPE1  = 'NAME    '",
        "TFORM1  = 'A8      '",
        "TBCOL1  =                    1",
        "TTYPE2  = 'MAG     '",
        "TFORM2  = 'F8.3    '",
        "TBCOL2  =                   10",
        "END",
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');

    // Rows: "ALPHA   " + " " + "  12.345"
    //       "BETA    " + " " + "  -9.876"
    //       "GAMMA   " + " " + "   0.000"
    let rows: [(&str, &str); 3] = [
        ("ALPHA   ", "  12.345"),
        ("BETA    ", "  -9.876"),
        ("GAMMA   ", "   0.000"),
    ];
    for (name, mag) in &rows {
        assert_eq!(name.len(), 8);
        assert_eq!(mag.len(), 8);
        buf.extend_from_slice(name.as_bytes());
        buf.push(b' '); // 1-byte gap (TBCOL2 = 10)
        buf.extend_from_slice(mag.as_bytes());
    }
    pad_to_block(&mut buf, b' ');

    let f = FitsFile::from_bytes(buf).unwrap();
    assert_eq!(f.len(), 2);
    let Hdu::AsciiTable(t) = f.hdu(1).unwrap() else {
        panic!("expected ASCII table");
    };
    assert_eq!(t.n_rows(), 3);
    assert_eq!(t.row_size(), row_size);
    assert_eq!(t.columns().len(), 2);

    let name_col = t.column_by_name("NAME").unwrap().clone();
    let mag_col = t.column_by_name("MAG").unwrap().clone();

    let want = [("ALPHA", 12.345), ("BETA", -9.876), ("GAMMA", 0.0)];
    for (i, (wname, wmag)) in want.iter().enumerate() {
        let n = t.cell_value(i, &name_col).unwrap().unwrap();
        let m = t.cell_value(i, &mag_col).unwrap().unwrap();
        match n {
            AsciiCell::Str(s) => assert_eq!(s.trim(), *wname),
            other => panic!("name not string: {other:?}"),
        }
        match m {
            AsciiCell::Float(v) => {
                assert!((v - *wmag).abs() < 1e-9, "row {i}: got {v}, want {wmag}");
            }
            other => panic!("mag not float: {other:?}"),
        }
    }
}

#[test]
fn bintable_fixed_columns_round_trip() {
    // Two columns: ID (1J = i32), FLUX (3E = three f32). Row = 4+12 = 16 bytes.
    let row_size: usize = 16;
    let n_rows: usize = 2;
    let header_cards = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        &format!("NAXIS1  = {row_size:>20}"),
        &format!("NAXIS2  = {n_rows:>20}"),
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    2",
        "TTYPE1  = 'ID      '",
        "TFORM1  = '1J      '",
        "TTYPE2  = 'FLUX    '",
        "TFORM2  = '3E      '",
        "END",
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');

    let mut data = Vec::new();
    for (id, flux) in [
        (42_i32, [1.0_f32, 2.0, 3.0]),
        (-7_i32, [10.5_f32, -1.25, 0.0]),
    ] {
        data.extend_from_slice(&id.to_be_bytes());
        for v in flux {
            data.extend_from_slice(&v.to_be_bytes());
        }
    }
    assert_eq!(data.len(), row_size * n_rows);
    buf.extend_from_slice(&data);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::BinTable(t) = f.hdu(1).unwrap() else {
        panic!("expected BINTABLE");
    };
    assert_eq!(t.n_rows(), 2);

    let id_col = t.column_by_name("ID").unwrap().clone();
    let flux_col = t.column_by_name("FLUX").unwrap().clone();

    match t.cell_value(0, &id_col).unwrap() {
        BinValue::Int(v) => assert_eq!(v, vec![Some(42)]),
        other => panic!("id row 0: {other:?}"),
    }
    match t.cell_value(1, &id_col).unwrap() {
        BinValue::Int(v) => assert_eq!(v, vec![Some(-7)]),
        other => panic!("id row 1: {other:?}"),
    }
    match t.cell_value(0, &flux_col).unwrap() {
        BinValue::F32(v) => assert_eq!(v, vec![1.0, 2.0, 3.0]),
        other => panic!("flux row 0: {other:?}"),
    }
    match t.cell_value(1, &flux_col).unwrap() {
        BinValue::F32(v) => assert_eq!(v, vec![10.5, -1.25, 0.0]),
        other => panic!("flux row 1: {other:?}"),
    }
}

#[test]
fn bintable_vla_with_heap() {
    // One VLA column `PE(99)` = `[i32 nelem, i32 offset]` per row.
    // 2 rows. Heap holds two f32 arrays.
    let row_size: usize = 8;
    let n_rows: usize = 2;
    // Row 0 -> 3 floats at offset 0; Row 1 -> 2 floats at offset 12.
    let row0 = [1.5_f32, 2.5, 3.5];
    let row1 = [-1.0_f32, 7.25];
    let mut heap = Vec::new();
    let r0_off = heap.len() as i32;
    for v in row0 {
        heap.extend_from_slice(&v.to_be_bytes());
    }
    let r1_off = heap.len() as i32;
    for v in row1 {
        heap.extend_from_slice(&v.to_be_bytes());
    }
    let pcount = heap.len();

    let header_cards = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        &format!("NAXIS1  = {row_size:>20}"),
        &format!("NAXIS2  = {n_rows:>20}"),
        &format!("PCOUNT  = {pcount:>20}"),
        "GCOUNT  =                    1",
        "TFIELDS =                    1",
        "TTYPE1  = 'WAVE    '",
        "TFORM1  = 'PE(99)  '",
        "END",
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');

    // Row table: 2 descriptors (n,offset) as i32 BE.
    let mut data = Vec::new();
    data.extend_from_slice(&(row0.len() as i32).to_be_bytes());
    data.extend_from_slice(&r0_off.to_be_bytes());
    data.extend_from_slice(&(row1.len() as i32).to_be_bytes());
    data.extend_from_slice(&r1_off.to_be_bytes());
    assert_eq!(data.len(), row_size * n_rows);
    buf.extend_from_slice(&data);
    buf.extend_from_slice(&heap);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::BinTable(t) = f.hdu(1).unwrap() else {
        panic!("expected BINTABLE");
    };
    let col = t.column_by_name("WAVE").unwrap().clone();

    let v0 = t.cell_value(0, &col).unwrap();
    let BinValue::Vla(inner) = v0 else {
        panic!("not vla: {v0:?}");
    };
    let BinValue::F32(values) = *inner else {
        panic!("inner not f32");
    };
    assert_eq!(values, vec![1.5, 2.5, 3.5]);

    let v1 = t.cell_value(1, &col).unwrap();
    let BinValue::Vla(inner) = v1 else {
        panic!("not vla: {v1:?}");
    };
    let BinValue::F32(values) = *inner else {
        panic!("inner not f32");
    };
    assert_eq!(values, vec![-1.0, 7.25]);
}

#[test]
fn bintable_string_and_logical_columns() {
    // Two columns: NAME (8A), FLAG (1L). Row = 9 bytes.
    let row_size: usize = 9;
    let n_rows: usize = 2;
    let header_cards = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        &format!("NAXIS1  = {row_size:>20}"),
        &format!("NAXIS2  = {n_rows:>20}"),
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    2",
        "TTYPE1  = 'NAME    '",
        "TFORM1  = '8A      '",
        "TTYPE2  = 'FLAG    '",
        "TFORM2  = '1L      '",
        "END",
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');

    let mut data = Vec::new();
    data.extend_from_slice(b"hello   ");
    data.push(b'T');
    data.extend_from_slice(b"world!  ");
    data.push(b'F');
    assert_eq!(data.len(), row_size * n_rows);
    buf.extend_from_slice(&data);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::BinTable(t) = f.hdu(1).unwrap() else {
        panic!("expected BINTABLE");
    };
    let name = t.column_by_name("NAME").unwrap().clone();
    let flag = t.column_by_name("FLAG").unwrap().clone();

    match t.cell_value(0, &name).unwrap() {
        BinValue::Str(s) => assert_eq!(s, "hello"),
        other => panic!("not str: {other:?}"),
    }
    match t.cell_value(1, &name).unwrap() {
        BinValue::Str(s) => assert_eq!(s, "world!"),
        other => panic!("not str: {other:?}"),
    }
    match t.cell_value(0, &flag).unwrap() {
        BinValue::Logical(v) => assert_eq!(v, vec![Some(true)]),
        other => panic!("not logical: {other:?}"),
    }
    match t.cell_value(1, &flag).unwrap() {
        BinValue::Logical(v) => assert_eq!(v, vec![Some(false)]),
        other => panic!("not logical: {other:?}"),
    }
}

#[test]
fn bintable_unsigned_k_preserves_full_u64() {
    // K column with TZERO=2^63, TSCAL=1: stored signed i64 must be
    // reinterpreted as u64 (Standard Sec.11.3.1) without going through
    // f64. Exercises a value > 2^53 so any f64 round-trip would lose
    // precision.
    let row_size: usize = 8;
    let n_rows: usize = 2;
    let header_cards = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        &format!("NAXIS1  = {row_size:>20}"),
        &format!("NAXIS2  = {n_rows:>20}"),
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    1",
        "TTYPE1  = 'COUNT   '",
        "TFORM1  = '1K      '",
        "TSCAL1  =                  1.0",
        "TZERO1  = 9223372036854775808.",
        "END",
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');

    // Pick u64 values that exceed 2^53; reinterpret as i64 for storage.
    let want: [u64; 2] = [u64::MAX - 7, (1_u64 << 60) | 0xDEAD_BEEF_CAFE];
    let mut data = Vec::new();
    for &u in &want {
        let signed = u.wrapping_sub(0x8000_0000_0000_0000) as i64;
        data.extend_from_slice(&signed.to_be_bytes());
    }
    buf.extend_from_slice(&data);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::BinTable(t) = f.hdu(1).unwrap() else {
        panic!("expected BINTABLE");
    };
    let col = t.column_by_name("COUNT").unwrap().clone();
    for (row, &expected) in want.iter().enumerate() {
        match t.cell_value(row, &col).unwrap() {
            BinValue::Uint(v) => assert_eq!(v, vec![Some(expected)], "row {row}"),
            other => panic!("row {row}: expected Uint, got {other:?}"),
        }
    }
}

#[test]
fn bintable_signed_byte_via_tzero_minus_128() {
    // B + TZERO=-128, TSCAL=1: signed i8 convention.
    let row_size: usize = 1;
    let n_rows: usize = 3;
    let header_cards = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        &format!("NAXIS1  = {row_size:>20}"),
        &format!("NAXIS2  = {n_rows:>20}"),
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    1",
        "TTYPE1  = 'S       '",
        "TFORM1  = '1B      '",
        "TSCAL1  =                  1.0",
        "TZERO1  =               -128.0",
        "END",
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    // Stored u8 = signed_value + 128. -128 -> 0, 0 -> 128, 127 -> 255.
    buf.extend_from_slice(&[0_u8, 128_u8, 255_u8]);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::BinTable(t) = f.hdu(1).unwrap() else {
        panic!("expected BINTABLE");
    };
    let col = t.column_by_name("S").unwrap().clone();
    let want = [-128_i64, 0, 127];
    for (row, &w) in want.iter().enumerate() {
        match t.cell_value(row, &col).unwrap() {
            BinValue::Int(v) => assert_eq!(v, vec![Some(w)], "row {row}"),
            other => panic!("row {row}: expected Int, got {other:?}"),
        }
    }
}

#[test]
fn bintable_negative_pcount_rejected() {
    let row_size: usize = 1;
    let n_rows: usize = 1;
    let header_cards = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        &format!("NAXIS1  = {row_size:>20}"),
        &format!("NAXIS2  = {n_rows:>20}"),
        "PCOUNT  =                   -1",
        "GCOUNT  =                    1",
        "TFIELDS =                    1",
        "TFORM1  = '1B      '",
        "END",
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    buf.push(0);
    pad_to_block(&mut buf, 0);

    // Either FitsFile::from_bytes (computing data_section_size) or
    // BinTableHdu::new should reject this.
    let res = FitsFile::from_bytes(buf);
    assert!(res.is_err(), "expected error, got Ok");
}

#[test]
fn ascii_table_tnull_only_for_integer_columns() {
    // I-column with TNULL="****", F-column with all blanks -> None.
    let row_size: usize = 12;
    let n_rows: usize = 2;
    let header_cards = [
        "XTENSION= 'TABLE   '",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        &format!("NAXIS1  = {row_size:>20}"),
        &format!("NAXIS2  = {n_rows:>20}"),
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    2",
        "TFORM1  = 'I4      '",
        "TBCOL1  =                    1",
        "TNULL1  = '****    '",
        "TFORM2  = 'F8.2    '",
        "TBCOL2  =                    5",
        "END",
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    // Row 0: I=" 123", F="   42.50"
    // Row 1: I="****", F="        " (all blanks -> undefined for F)
    buf.extend_from_slice(b" 123   42.50");
    buf.extend_from_slice(b"****        ");
    pad_to_block(&mut buf, b' ');

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::AsciiTable(t) = f.hdu(1).unwrap() else {
        panic!("expected ASCII table");
    };
    let icol = &t.columns()[0].clone();
    let fcol = &t.columns()[1].clone();
    assert!(icol.tnull.is_some(), "TNULL must apply to I-format");
    assert!(fcol.tnull.is_none(), "TNULL must NOT apply to F-format");
    assert_eq!(t.cell_value(0, icol).unwrap(), Some(AsciiCell::Int(123)),);
    assert_eq!(t.cell_value(1, icol).unwrap(), None, "TNULL match -> None");
    assert_eq!(t.cell_value(0, fcol).unwrap(), Some(AsciiCell::Float(42.5)),);
    assert_eq!(t.cell_value(1, fcol).unwrap(), None, "all blanks -> None");
}

#[test]
fn bintable_vla_repeat_must_be_zero_or_one() {
    use fitsy::BinFormat;
    // Outer repeat 2 for P/Q is illegal per Sec.7.3.5.
    assert!(BinFormat::parse("2PE(10)").is_err());
    assert!(BinFormat::parse("PE(10)").is_ok());
    assert!(BinFormat::parse("0PE").is_ok());
    assert!(BinFormat::parse("1PE").is_ok());
}
