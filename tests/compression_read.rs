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
        .chunks_exact(2)
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
        .chunks_exact(4)
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
        .chunks_exact(2)
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
        .chunks_exact(4)
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
        .chunks_exact(8)
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
