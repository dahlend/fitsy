//! End-to-end tests for the optional `compression` feature: whole-file
//! gzip auto-decompress and `.fz` tile-compressed image HDUs.

#![cfg(feature = "compression")]

use fitsy::data::encoding::Bitpix;
use fitsy::{FitsFile, Hdu};
use flate2::Compression;
use flate2::write::GzEncoder;
use std::io::Write;

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
fn open_gzipped_file_in_memory() {
    // Build a trivial valid FITS file...
    let bytes = empty_primary();
    // ...gzip it...
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(&bytes).unwrap();
    let gz = e.finish().unwrap();
    // ...and confirm `from_bytes` transparently inflates it.
    let f = FitsFile::from_bytes(gz).unwrap();
    assert_eq!(f.len(), 1);
    let Hdu::Image(img) = f.hdu(0).unwrap() else {
        panic!("expected image");
    };
    assert_eq!(img.n_elements(), 0);
}

#[test]
fn fz_gzip1_single_tile_round_trip() {
    // Pixels: 4x3 i16 image, deterministic content.
    let nx: usize = 4;
    let ny: usize = 3;
    let original: Vec<i16> = (0..(nx * ny) as i16).map(|i| i * 100 - 250).collect();

    // Pack to big-endian bytes, gzip them as a single tile.
    let mut be = Vec::new();
    for &p in &original {
        be.extend_from_slice(&p.to_be_bytes());
    }
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(&be).unwrap();
    let tile_payload = e.finish().unwrap();

    // BINTABLE with one column COMPRESSED_DATA = 1PB(maxlen). One row.
    // Row size = 8 bytes (one P descriptor).
    let row_size: usize = 8;
    let n_rows: usize = 1;
    let pcount = tile_payload.len();
    let header_cards: Vec<String> = vec![
        "XTENSION= 'BINTABLE'".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    2".into(),
        format!("NAXIS1  = {row_size:>20}"),
        format!("NAXIS2  = {n_rows:>20}"),
        format!("PCOUNT  = {pcount:>20}"),
        "GCOUNT  =                    1".into(),
        "TFIELDS =                    1".into(),
        "TTYPE1  = 'COMPRESSED_DATA'".into(),
        format!("TFORM1  = '1PB({pcount:<3})'"),
        "ZIMAGE  =                    T".into(),
        "ZBITPIX =                   16".into(),
        "ZNAXIS  =                    2".into(),
        format!("ZNAXIS1 = {nx:>20}"),
        format!("ZNAXIS2 = {ny:>20}"),
        format!("ZTILE1  = {nx:>20}"),
        format!("ZTILE2  = {ny:>20}"),
        "ZCMPTYPE= 'GZIP_1  '".into(),
        "END".into(),
    ];

    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');

    // Single row: P descriptor (n=pcount, off=0) followed by heap.
    let n_be = (pcount as i32).to_be_bytes();
    let off_be = 0_i32.to_be_bytes();
    buf.extend_from_slice(&n_be);
    buf.extend_from_slice(&off_be);
    buf.extend_from_slice(&tile_payload);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    assert_eq!(f.len(), 2);
    let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
        panic!("expected compressed image");
    };
    assert_eq!(c.bitpix(), Bitpix::I16);
    assert_eq!(c.axes(), &[nx as u64, ny as u64]);
    let raw = c.decompress().unwrap();
    assert_eq!(raw.len(), nx * ny * 2);
    // Decode and compare.
    let decoded: Vec<i16> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(decoded, original);
}

#[test]
fn fz_gzip2_byte_shuffled_round_trip() {
    // Build the same image, but byte-shuffle before gzip and tag GZIP_2.
    let nx: usize = 4;
    let ny: usize = 3;
    let original: Vec<i32> = (0..(nx * ny) as i32)
        .map(|i| i * 1_000_000 - 2_500_000)
        .collect();

    let bpp = 4;
    let n = nx * ny;
    let mut shuf = vec![0_u8; n * bpp];
    for (i, p) in original.iter().enumerate() {
        let bytes = p.to_be_bytes();
        for plane in 0..bpp {
            shuf[plane * n + i] = bytes[plane];
        }
    }
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(&shuf).unwrap();
    let tile_payload = e.finish().unwrap();

    let row_size: usize = 8;
    let n_rows: usize = 1;
    let pcount = tile_payload.len();
    let header_cards: Vec<String> = vec![
        "XTENSION= 'BINTABLE'".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    2".into(),
        format!("NAXIS1  = {row_size:>20}"),
        format!("NAXIS2  = {n_rows:>20}"),
        format!("PCOUNT  = {pcount:>20}"),
        "GCOUNT  =                    1".into(),
        "TFIELDS =                    1".into(),
        "TTYPE1  = 'COMPRESSED_DATA'".into(),
        format!("TFORM1  = '1PB({pcount:<3})'"),
        "ZIMAGE  =                    T".into(),
        "ZBITPIX =                   32".into(),
        "ZNAXIS  =                    2".into(),
        format!("ZNAXIS1 = {nx:>20}"),
        format!("ZNAXIS2 = {ny:>20}"),
        format!("ZTILE1  = {nx:>20}"),
        format!("ZTILE2  = {ny:>20}"),
        "ZCMPTYPE= 'GZIP_2  '".into(),
        "END".into(),
    ];

    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');

    buf.extend_from_slice(&(pcount as i32).to_be_bytes());
    buf.extend_from_slice(&0_i32.to_be_bytes());
    buf.extend_from_slice(&tile_payload);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
        panic!("expected compressed image");
    };
    let raw = c.decompress().unwrap();
    let decoded: Vec<i32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(decoded, original);
}

#[test]
fn fz_multi_tile_row_strip_round_trip() {
    // 4x3 image, tiled as 4x1 row strips -> 3 rows in the BINTABLE.
    let nx: usize = 4;
    let ny: usize = 3;
    let original: Vec<i16> = (0..(nx * ny) as i16).map(|i| i + 1).collect();

    // Compress each row separately.
    let mut per_row_payload: Vec<Vec<u8>> = Vec::new();
    for row in 0..ny {
        let mut be = Vec::new();
        for x in 0..nx {
            be.extend_from_slice(&original[row * nx + x].to_be_bytes());
        }
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(&be).unwrap();
        per_row_payload.push(e.finish().unwrap());
    }
    let pcount: usize = per_row_payload.iter().map(Vec::len).sum();
    let row_size: usize = 8;
    let n_rows: usize = ny;
    let max_payload = per_row_payload.iter().map(Vec::len).max().unwrap();
    let header_cards: Vec<String> = vec![
        "XTENSION= 'BINTABLE'".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    2".into(),
        format!("NAXIS1  = {row_size:>20}"),
        format!("NAXIS2  = {n_rows:>20}"),
        format!("PCOUNT  = {pcount:>20}"),
        "GCOUNT  =                    1".into(),
        "TFIELDS =                    1".into(),
        "TTYPE1  = 'COMPRESSED_DATA'".into(),
        format!("TFORM1  = '1PB({max_payload:<3})'"),
        "ZIMAGE  =                    T".into(),
        "ZBITPIX =                   16".into(),
        "ZNAXIS  =                    2".into(),
        format!("ZNAXIS1 = {nx:>20}"),
        format!("ZNAXIS2 = {ny:>20}"),
        format!("ZTILE1  = {nx:>20}"),
        "ZTILE2  =                    1".into(),
        "ZCMPTYPE= 'GZIP_1  '".into(),
        "END".into(),
    ];

    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');

    // Row table: one P descriptor per row.
    let mut heap_off: i32 = 0;
    for payload in &per_row_payload {
        buf.extend_from_slice(&(payload.len() as i32).to_be_bytes());
        buf.extend_from_slice(&heap_off.to_be_bytes());
        heap_off += payload.len() as i32;
    }
    // Heap.
    for payload in &per_row_payload {
        buf.extend_from_slice(payload);
    }
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
        panic!("expected compressed image");
    };
    let raw = c.decompress().unwrap();
    let decoded: Vec<i16> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(decoded, original);
}

#[test]
fn unsupported_cmptype_is_explicit() {
    // ZCMPTYPE = NOSUCH_1 -> explicit error, not a panic.
    let row_size: usize = 8;
    let pcount: usize = 1; // dummy
    let header_cards: Vec<String> = vec![
        "XTENSION= 'BINTABLE'".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    2".into(),
        format!("NAXIS1  = {row_size:>20}"),
        "NAXIS2  =                    1".into(),
        format!("PCOUNT  = {pcount:>20}"),
        "GCOUNT  =                    1".into(),
        "TFIELDS =                    1".into(),
        "TTYPE1  = 'COMPRESSED_DATA'".into(),
        "TFORM1  = '1PB(1)  '".into(),
        "ZIMAGE  =                    T".into(),
        "ZBITPIX =                   16".into(),
        "ZNAXIS  =                    2".into(),
        "ZNAXIS1 =                    1".into(),
        "ZNAXIS2 =                    1".into(),
        "ZCMPTYPE= 'NOSUCH_1'".into(),
        "END".into(),
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    buf.extend_from_slice(&1_i32.to_be_bytes());
    buf.extend_from_slice(&0_i32.to_be_bytes());
    buf.push(0);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let err = f.hdu(1).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("NOSUCH_1"), "got: {msg}");
}

/// Build a single-tile compressed BINTABLE for a 2x2 quantized float
/// image and verify the unquantized values come out as expected.
#[test]
fn fz_quantized_float_no_dither_round_trip() {
    let nx: usize = 2;
    let ny: usize = 2;
    // i32 quantized values; ZSCALE=2.0, ZZERO=100.0 -> floats below.
    let q: [i32; 4] = [10, -5, 0, 7];
    let expected: [f32; 4] = [120.0, 90.0, 100.0, 114.0];

    let mut be = Vec::new();
    for v in q {
        be.extend_from_slice(&v.to_be_bytes());
    }
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(&be).unwrap();
    let tile_payload = e.finish().unwrap();

    let row_size: usize = 8;
    let n_rows: usize = 1;
    let pcount = tile_payload.len();
    let header_cards: Vec<String> = vec![
        "XTENSION= 'BINTABLE'".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    2".into(),
        format!("NAXIS1  = {row_size:>20}"),
        format!("NAXIS2  = {n_rows:>20}"),
        format!("PCOUNT  = {pcount:>20}"),
        "GCOUNT  =                    1".into(),
        "TFIELDS =                    1".into(),
        "TTYPE1  = 'COMPRESSED_DATA'".into(),
        format!("TFORM1  = '1PB({pcount:<3})'"),
        "ZIMAGE  =                    T".into(),
        "ZBITPIX =                  -32".into(),
        "ZNAXIS  =                    2".into(),
        format!("ZNAXIS1 = {nx:>20}"),
        format!("ZNAXIS2 = {ny:>20}"),
        format!("ZTILE1  = {nx:>20}"),
        format!("ZTILE2  = {ny:>20}"),
        "ZCMPTYPE= 'GZIP_1  '".into(),
        "ZQUANTIZ= 'NO_DITHER'".into(),
        "ZSCALE  =                  2.0".into(),
        "ZZERO   =                100.0".into(),
        "END".into(),
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    buf.extend_from_slice(&(pcount as i32).to_be_bytes());
    buf.extend_from_slice(&0_i32.to_be_bytes());
    buf.extend_from_slice(&tile_payload);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
        panic!("expected compressed image");
    };
    assert_eq!(c.bitpix(), Bitpix::F32);
    let raw = c.decompress().unwrap();
    let decoded: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for (got, want) in decoded.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-5, "got {got} want {want}");
    }
}

/// Regression: a quantized F64 image whose tile fell back to
/// `GZIP_COMPRESSED_DATA` carries raw f64 pixels in physical units,
/// not quantized i32s. The decoder must size its buffer for f64
/// and skip dequantization for that tile.
#[test]
fn fz_quantized_f64_gzip_fallback_round_trip() {
    let nx: usize = 2;
    let ny: usize = 2;
    let pixels: [f64; 4] = [1.5, -2.25, 3.125e10, 0.0];

    // Fallback payload = raw big-endian f64 pixels, gzipped.
    let mut be = Vec::new();
    for v in pixels {
        be.extend_from_slice(&v.to_be_bytes());
    }
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(&be).unwrap();
    let fallback = e.finish().unwrap();

    // Two columns: COMPRESSED_DATA (empty 1Pi) + GZIP_COMPRESSED_DATA (1PB).
    let row_size: usize = 8 + 8;
    let pcount = fallback.len();
    let header_cards: Vec<String> = vec![
        "XTENSION= 'BINTABLE'".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    2".into(),
        format!("NAXIS1  = {row_size:>20}"),
        "NAXIS2  =                    1".into(),
        format!("PCOUNT  = {pcount:>20}"),
        "GCOUNT  =                    1".into(),
        "TFIELDS =                    2".into(),
        "TTYPE1  = 'COMPRESSED_DATA'".into(),
        "TFORM1  = '1PI(0)  '".into(),
        "TTYPE2  = 'GZIP_COMPRESSED_DATA'".into(),
        format!("TFORM2  = '1PB({pcount:<3})'"),
        "ZIMAGE  =                    T".into(),
        "ZBITPIX =                  -64".into(),
        "ZNAXIS  =                    2".into(),
        format!("ZNAXIS1 = {nx:>20}"),
        format!("ZNAXIS2 = {ny:>20}"),
        format!("ZTILE1  = {nx:>20}"),
        format!("ZTILE2  = {ny:>20}"),
        "ZCMPTYPE= 'RICE_1  '".into(),
        "ZQUANTIZ= 'NO_DITHER'".into(),
        "ZSCALE  =                  2.0".into(),
        "ZZERO   =                100.0".into(),
        "END".into(),
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');

    // Row: empty COMPRESSED_DATA descriptor (n=0), then the gzip
    // fallback descriptor pointing at the heap.
    buf.extend_from_slice(&0_i32.to_be_bytes()); // n
    buf.extend_from_slice(&0_i32.to_be_bytes()); // off
    buf.extend_from_slice(&(pcount as i32).to_be_bytes());
    buf.extend_from_slice(&0_i32.to_be_bytes());
    buf.extend_from_slice(&fallback);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
        panic!("expected compressed image");
    };
    assert_eq!(c.bitpix(), Bitpix::F64);
    let raw = c.decompress().unwrap();
    let decoded: Vec<f64> = raw
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    for (got, want) in decoded.iter().zip(pixels.iter()) {
        assert_eq!(got.to_bits(), want.to_bits(), "got {got} want {want}");
    }
}

/// Regression: the synthetic IMAGE header for a quantized float
/// image must NOT carry a `BLANK` card (forbidden on float images
/// per Sec.4.4.2.2; the sentinel is consumed during dequantization).
#[test]
fn synthetic_header_drops_blank_for_float_image() {
    let nx: usize = 2;
    let ny: usize = 1;
    let q: [i32; 2] = [10, 20];
    let mut be = Vec::new();
    for v in q {
        be.extend_from_slice(&v.to_be_bytes());
    }
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(&be).unwrap();
    let payload = e.finish().unwrap();

    let row_size: usize = 8;
    let pcount = payload.len();
    let header_cards: Vec<String> = vec![
        "XTENSION= 'BINTABLE'".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    2".into(),
        format!("NAXIS1  = {row_size:>20}"),
        "NAXIS2  =                    1".into(),
        format!("PCOUNT  = {pcount:>20}"),
        "GCOUNT  =                    1".into(),
        "TFIELDS =                    1".into(),
        "TTYPE1  = 'COMPRESSED_DATA'".into(),
        format!("TFORM1  = '1PB({pcount:<3})'"),
        "ZIMAGE  =                    T".into(),
        "ZBITPIX =                  -32".into(),
        "ZNAXIS  =                    2".into(),
        format!("ZNAXIS1 = {nx:>20}"),
        format!("ZNAXIS2 = {ny:>20}"),
        format!("ZTILE1  = {nx:>20}"),
        format!("ZTILE2  = {ny:>20}"),
        "ZCMPTYPE= 'GZIP_1  '".into(),
        "ZQUANTIZ= 'NO_DITHER'".into(),
        "ZSCALE  =                  1.0".into(),
        "ZZERO   =                  0.0".into(),
        "ZBLANK  =          -2147483647".into(),
        "END".into(),
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    buf.extend_from_slice(&(pcount as i32).to_be_bytes());
    buf.extend_from_slice(&0_i32.to_be_bytes());
    buf.extend_from_slice(&payload);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
        panic!("expected compressed image");
    };
    let synth = c.synthetic_image_header().unwrap();
    assert!(
        synth.optional_int("BLANK").is_none(),
        "synthetic float header must not carry BLANK"
    );
}

/// Regression: only the Sec.10.2 reserved `Z*` names are rewritten.
/// `ZP`/`ZD` are ordinary keywords; `ZTILE` is geometry, not `TILE`.
#[test]
fn synthetic_header_preserves_non_reserved_z_keywords() {
    let nx: usize = 2;
    let ny: usize = 1;
    let pixels: [i16; 2] = [7, 9];
    let mut be = Vec::new();
    for v in pixels {
        be.extend_from_slice(&v.to_be_bytes());
    }
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(&be).unwrap();
    let payload = e.finish().unwrap();

    let row_size: usize = 8;
    let pcount = payload.len();
    let header_cards: Vec<String> = vec![
        "XTENSION= 'BINTABLE'".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    2".into(),
        format!("NAXIS1  = {row_size:>20}"),
        "NAXIS2  =                    1".into(),
        format!("PCOUNT  = {pcount:>20}"),
        "GCOUNT  =                    1".into(),
        "TFIELDS =                    1".into(),
        "TTYPE1  = 'COMPRESSED_DATA'".into(),
        format!("TFORM1  = '1PB({pcount:<3})'"),
        "ZIMAGE  =                    T".into(),
        "ZBITPIX =                   16".into(),
        "ZNAXIS  =                    2".into(),
        format!("ZNAXIS1 = {nx:>20}"),
        format!("ZNAXIS2 = {ny:>20}"),
        format!("ZTILE1  = {nx:>20}"),
        format!("ZTILE2  = {ny:>20}"),
        "ZCMPTYPE= 'GZIP_1  '".into(),
        // The wrapper's own sums, over the *compressed* bytes.
        "CHECKSUM= 'Z7HYd4GWZ4GWb4GW'".into(),
        "DATASUM = '1541178127'".into(),
        // Ordinary image keywords that merely begin with `Z`.
        "ZP      =                 25.3".into(),
        "ZD      =                 12.5".into(),
        "CTYPE1  = 'RA---TAN'".into(),
        "END".into(),
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    buf.extend_from_slice(&(pcount as i32).to_be_bytes());
    buf.extend_from_slice(&0_i32.to_be_bytes());
    buf.extend_from_slice(&payload);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
        panic!("expected compressed image");
    };
    let synth = c.synthetic_image_header().unwrap();

    assert_eq!(synth.optional_real("ZP"), Some(25.3), "ZP must survive");
    assert_eq!(synth.optional_real("ZD"), Some(12.5), "ZD must survive");
    assert!(!synth.contains("P"), "ZP must not be rewritten to P");
    assert!(!synth.contains("D"), "ZD must not be rewritten to D");
    // Compression geometry stays out of the image header.
    assert!(!synth.contains("TILE1"), "ZTILE1 must not leak as TILE1");
    assert!(!synth.contains("TILE2"), "ZTILE2 must not leak as TILE2");
    assert!(!synth.contains("ZTILE1"));
    // These cover the compressed bytes, not the decompressed image.
    assert!(
        !synth.contains("CHECKSUM"),
        "the BINTABLE's CHECKSUM must not describe the decompressed image"
    );
    assert!(
        !synth.contains("DATASUM"),
        "the BINTABLE's DATASUM must not describe the decompressed image"
    );
    // The structural rewrite still works.
    assert_eq!(synth.bitpix().unwrap(), 16);
    assert_eq!(synth.axes().unwrap(), vec![nx as u64, ny as u64]);
    assert_eq!(
        synth.optional_string("CTYPE1"),
        Some("RA---TAN".to_string())
    );
}

/// Standard Sec.10.2.2 requires the reader to use the `ZBLANK`
/// column. This applies when an image carries both forms.
///
/// The image holds two tiles. Each tile marks a different quantized
/// value as undefined. Only a per-tile column can express that. The
/// keyword names a third value, which matches no pixel. A reader that
/// consults the keyword therefore returns four defined pixels, and
/// fails the assertions below.
#[test]
fn fz_zblank_column_overrides_keyword() {
    let nx: usize = 2;
    let ny: usize = 2;
    // ZSCALE=2.0, ZZERO=100.0 -> physical = q * 2 + 100.
    let tiles: [[i32; 2]; 2] = [[10, -5], [0, 7]];
    // Per-tile sentinel: tile 0 blanks -5, tile 1 blanks 7.
    let zblank_per_tile: [i32; 2] = [-5, 7];

    let mut payloads: Vec<Vec<u8>> = Vec::new();
    for tile in &tiles {
        let mut be = Vec::new();
        for v in tile {
            be.extend_from_slice(&v.to_be_bytes());
        }
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(&be).unwrap();
        payloads.push(e.finish().unwrap());
    }

    // Row: COMPRESSED_DATA descriptor (8 bytes) + ZBLANK i32 (4 bytes).
    let row_size: usize = 12;
    let n_rows: usize = 2;
    let heap: usize = payloads.iter().map(Vec::len).sum();
    let header_cards: Vec<String> = vec![
        "XTENSION= 'BINTABLE'".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    2".into(),
        format!("NAXIS1  = {row_size:>20}"),
        format!("NAXIS2  = {n_rows:>20}"),
        format!("PCOUNT  = {heap:>20}"),
        "GCOUNT  =                    1".into(),
        "TFIELDS =                    2".into(),
        "TTYPE1  = 'COMPRESSED_DATA'".into(),
        "TFORM1  = '1PB     '".into(),
        "TTYPE2  = 'ZBLANK  '".into(),
        "TFORM2  = '1J      '".into(),
        format!("THEAP   = {:>20}", row_size * n_rows),
        "ZIMAGE  =                    T".into(),
        "ZBITPIX =                  -32".into(),
        "ZNAXIS  =                    2".into(),
        format!("ZNAXIS1 = {nx:>20}"),
        format!("ZNAXIS2 = {ny:>20}"),
        format!("ZTILE1  = {nx:>20}"),
        "ZTILE2  =                    1".into(),
        "ZCMPTYPE= 'GZIP_1  '".into(),
        "ZQUANTIZ= 'NO_DITHER'".into(),
        "ZSCALE  =                  2.0".into(),
        "ZZERO   =                100.0".into(),
        // Matches no quantized value. A reader that uses the keyword
        // instead of the column blanks nothing.
        "ZBLANK  =                 -999".into(),
        "END".into(),
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');

    // Row area: descriptor then sentinel, one row per tile.
    let mut offset = 0_i32;
    for (p, z) in payloads.iter().zip(zblank_per_tile.iter()) {
        buf.extend_from_slice(&(p.len() as i32).to_be_bytes());
        buf.extend_from_slice(&offset.to_be_bytes());
        buf.extend_from_slice(&z.to_be_bytes());
        offset += p.len() as i32;
    }
    for p in &payloads {
        buf.extend_from_slice(p);
    }
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
        panic!("expected compressed image");
    };
    let raw = c.decompress().unwrap();
    let decoded: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    assert_eq!(decoded.len(), 4);
    assert!((decoded[0] - 120.0).abs() < 1e-5, "got {}", decoded[0]);
    assert!(decoded[1].is_nan(), "tile 0 sentinel -5 must read as NaN");
    assert!((decoded[2] - 100.0).abs() < 1e-5, "got {}", decoded[2]);
    assert!(decoded[3].is_nan(), "tile 1 sentinel 7 must read as NaN");
}

/// A null cell in the `ZBLANK` column means the tile holds no
/// undefined pixel. Every pixel of that tile stays defined.
#[test]
fn fz_zblank_column_null_cell_blanks_nothing() {
    let nx: usize = 2;
    let ny: usize = 1;
    let q: [i32; 2] = [10, -5];

    let mut be = Vec::new();
    for v in q {
        be.extend_from_slice(&v.to_be_bytes());
    }
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(&be).unwrap();
    let payload = e.finish().unwrap();

    let row_size: usize = 12;
    let n_rows: usize = 1;
    let heap = payload.len();
    let header_cards: Vec<String> = vec![
        "XTENSION= 'BINTABLE'".into(),
        "BITPIX  =                    8".into(),
        "NAXIS   =                    2".into(),
        format!("NAXIS1  = {row_size:>20}"),
        format!("NAXIS2  = {n_rows:>20}"),
        format!("PCOUNT  = {heap:>20}"),
        "GCOUNT  =                    1".into(),
        "TFIELDS =                    2".into(),
        "TTYPE1  = 'COMPRESSED_DATA'".into(),
        "TFORM1  = '1PB     '".into(),
        "TTYPE2  = 'ZBLANK  '".into(),
        "TFORM2  = '1J      '".into(),
        // The null marker for the ZBLANK column.
        "TNULL2  =            -32768".into(),
        format!("THEAP   = {:>20}", row_size * n_rows),
        "ZIMAGE  =                    T".into(),
        "ZBITPIX =                  -32".into(),
        "ZNAXIS  =                    2".into(),
        format!("ZNAXIS1 = {nx:>20}"),
        format!("ZNAXIS2 = {ny:>20}"),
        format!("ZTILE1  = {nx:>20}"),
        "ZTILE2  =                    1".into(),
        "ZCMPTYPE= 'GZIP_1  '".into(),
        "ZQUANTIZ= 'NO_DITHER'".into(),
        "ZSCALE  =                  2.0".into(),
        "ZZERO   =                100.0".into(),
        "END".into(),
    ];
    let mut buf = empty_primary();
    for c in &header_cards {
        buf.extend_from_slice(&pad_card(c));
    }
    pad_to_block(&mut buf, b' ');
    buf.extend_from_slice(&(payload.len() as i32).to_be_bytes());
    buf.extend_from_slice(&0_i32.to_be_bytes());
    buf.extend_from_slice(&(-32768_i32).to_be_bytes());
    buf.extend_from_slice(&payload);
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
        panic!("expected compressed image");
    };
    let raw = c.decompress().unwrap();
    let decoded: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!((decoded[0] - 120.0).abs() < 1e-5, "got {}", decoded[0]);
    assert!((decoded[1] - 90.0).abs() < 1e-5, "got {}", decoded[1]);
}
