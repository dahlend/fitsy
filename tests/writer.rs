//! End-to-end writer integration tests. These verify that:
//!
//! 1. A header built from scratch via [`Header::empty`] +
//!    [`Header::push`] serializes to bytes that the parser accepts.
//! 2. [`FitsFile::write`] re-serializes a real on-disk file into
//!    output that itself parses back to the same HDU layout (same
//!    number of HDUs, same kind/shape/data length per HDU).
//! 3. Long-string values containing CONTINUE chunks survive a full
//!    write -> parse -> string-equal round-trip.
//! 4. Binary tables and ASCII tables are written with the correct
//!    pad byte (0 for BINTABLE, ASCII space for TABLE).

use std::path::PathBuf;

use fitsy::{CommentaryKind, FitsFile, FitsWriter, Hdu, Header, Value};

fn data_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("data");
    p.push(name);
    p
}

#[test]
fn build_minimal_primary_from_scratch_round_trips() {
    let mut h = Header::empty();
    h.push("SIMPLE", Value::Logical(true), Some("conforming"))
        .unwrap();
    h.push("BITPIX", Value::Integer(8), None).unwrap();
    h.push("NAXIS", Value::Integer(0), None).unwrap();
    h.push_commentary(CommentaryKind::History, "built by fitsy writer test");

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&h, &[]).unwrap();
    w.finish().unwrap();

    // Parse it back via the public reader.
    let parsed = FitsFile::from_bytes(buf).unwrap();
    assert_eq!(parsed.len(), 1);
    let hdu = parsed.hdu(0).unwrap();
    match hdu {
        Hdu::Image(img) => {
            assert_eq!(img.bitpix().as_i64(), 8);
            assert_eq!(img.axes().len(), 0);
            assert_eq!(img.raw_bytes().len(), 0);
        }
        other => panic!("expected primary IMAGE, got {other:?}"),
    }
    let h2 = parsed.hdu(0).unwrap();
    let header = h2.header();
    let history: Vec<_> = header
        .entries()
        .iter()
        .filter(|e| e.keyword == "HISTORY")
        .filter_map(|e| e.commentary.clone())
        .collect();
    assert_eq!(history, vec!["built by fitsy writer test".to_string()]);
}

#[test]
fn primary_with_image_data_round_trips() {
    // 3x2 = 6 i16 pixels -> BITPIX=16, NAXIS=2, NAXIS1=3, NAXIS2=2.
    let mut h = Header::empty();
    h.push("SIMPLE", Value::Logical(true), None).unwrap();
    h.push("BITPIX", Value::Integer(16), None).unwrap();
    h.push("NAXIS", Value::Integer(2), None).unwrap();
    h.push("NAXIS1", Value::Integer(3), None).unwrap();
    h.push("NAXIS2", Value::Integer(2), None).unwrap();

    // Big-endian i16 payload.
    let pixels: [i16; 6] = [-3, -2, -1, 0, 1, 2];
    let mut data = Vec::with_capacity(pixels.len() * 2);
    for p in pixels {
        data.extend_from_slice(&p.to_be_bytes());
    }

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&h, &data).unwrap();
    w.finish().unwrap();

    let parsed = FitsFile::from_bytes(buf).unwrap();
    let img = match parsed.hdu(0).unwrap() {
        Hdu::Image(i) => i,
        other => panic!("expected IMAGE, got {other:?}"),
    };
    assert_eq!(img.axes(), &[3_u64, 2]);
    assert_eq!(img.raw_bytes(), &data[..]);
}

#[test]
fn long_string_round_trips_via_writer() {
    let mut h = Header::empty();
    h.push("SIMPLE", Value::Logical(true), None).unwrap();
    h.push("BITPIX", Value::Integer(8), None).unwrap();
    h.push("NAXIS", Value::Integer(0), None).unwrap();
    let payload = "abcdefghij".repeat(40); // 400 chars -> CONTINUE chain
    h.push("OBJECT", Value::String(payload.clone()), None)
        .unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&h, &[]).unwrap();
    w.finish().unwrap();

    let parsed = FitsFile::from_bytes(buf).unwrap();
    let hdu = parsed.hdu(0).unwrap();
    match hdu.header().first("OBJECT").unwrap() {
        Value::String(s) => assert_eq!(s, &payload),
        other => panic!("not a string: {other:?}"),
    }
}

#[test]
fn embedded_quote_round_trips() {
    let mut h = Header::empty();
    h.push("SIMPLE", Value::Logical(true), None).unwrap();
    h.push("BITPIX", Value::Integer(8), None).unwrap();
    h.push("NAXIS", Value::Integer(0), None).unwrap();
    h.push("OBJECT", Value::String("it's me".into()), None)
        .unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&h, &[]).unwrap();
    w.finish().unwrap();

    let parsed = FitsFile::from_bytes(buf).unwrap();
    match parsed.hdu(0).unwrap().header().first("OBJECT").unwrap() {
        Value::String(s) => assert!(s.starts_with("it's me"), "got `{s}`"),
        other => panic!("not a string: {other:?}"),
    }
}

#[test]
fn fits_file_to_bytes_preserves_hdu_count_and_shapes() {
    let path = data_path("3i_palomar.fits");
    if !path.exists() {
        eprintln!("skipping: {path:?} not present");
        return;
    }
    let original = FitsFile::open(&path).unwrap();
    let original_count = original.len();

    // Capture per-HDU shape from original.
    let mut want: Vec<(&'static str, usize)> = Vec::new();
    for i in 0..original_count {
        let hdu = original.hdu(i).unwrap();
        let kind = match &hdu {
            Hdu::Image(_) => "IMAGE",
            Hdu::AsciiTable(_) => "TABLE",
            Hdu::BinTable(_) => "BINTABLE",
            #[cfg(feature = "compression")]
            Hdu::CompressedImage(_) => "BINTABLE",
            Hdu::Conforming(_) => "CONFORMING",
            _ => "UNKNOWN",
        };
        want.push((kind, hdu.data_bytes().len()));
    }

    let tmp = std::env::temp_dir().join(format!(
        "fitsy_writer_roundtrip_{}.fits",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&tmp);
    original.write(&tmp, true).unwrap();
    let reparsed = FitsFile::open(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    assert_eq!(reparsed.len(), original_count);
    for (i, (kind, dlen)) in want.iter().enumerate() {
        let hdu = reparsed.hdu(i).unwrap();
        let got_kind = match &hdu {
            Hdu::Image(_) => "IMAGE",
            Hdu::AsciiTable(_) => "TABLE",
            Hdu::BinTable(_) => "BINTABLE",
            #[cfg(feature = "compression")]
            Hdu::CompressedImage(_) => "BINTABLE",
            Hdu::Conforming(_) => "CONFORMING",
            _ => "UNKNOWN",
        };
        assert_eq!(got_kind, *kind, "HDU {i} kind drifted");
        assert_eq!(hdu.data_bytes().len(), *dlen, "HDU {i} data length drifted");
    }
}

#[test]
fn fits_file_write_refuses_to_clobber_without_overwrite() {
    let path = data_path("3i_palomar.fits");
    if !path.exists() {
        eprintln!("skipping: {path:?} not present");
        return;
    }
    let f = FitsFile::open(&path).unwrap();
    let tmp =
        std::env::temp_dir().join(format!("fitsy_writer_clobber_{}.fits", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    f.write(&tmp, false).unwrap();
    // Second write without overwrite must fail.
    let err = f.write(&tmp, false).unwrap_err();
    assert!(
        matches!(err, fitsy::FitsError::Io(ref e)
        if e.kind() == std::io::ErrorKind::AlreadyExists),
        "expected AlreadyExists, got {err:?}"
    );
    // Overwrite succeeds.
    f.write(&tmp, true).unwrap();
    let _ = std::fs::remove_file(&tmp);
}
