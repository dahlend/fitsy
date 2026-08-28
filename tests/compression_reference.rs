//! Decode tile-compressed images written by astropy (whose compression
//! code is derived from cfitsio) and compare pixel-for-pixel.
//!
//! These are NOT round-trip tests: fitsy's own encoders are never
//! involved. A self-consistent codec bug (encoder and decoder sharing
//! the same error) passes every round-trip test but fails here.
//! Fixtures come from `tests/data/gen_reference_fixtures.py`.

#![cfg(feature = "compression")]

use fitsy::{FitsFile, Hdu};
use std::path::PathBuf;

fn test_data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

/// Deterministic value stream shared with the Python generator: a plain
/// LCG `x_{k+1} = (x_k * 1103515245 + 12345) mod 2^31`, seeded with 1;
/// the k-th output is the state after k+1 steps.
fn lcg(n: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(n);
    let mut x: i64 = 1;
    for _ in 0..n {
        x = (x * 1103515245 + 12345) % (1 << 31);
        out.push(x);
    }
    out
}

fn decompress(name: &str) -> Vec<u8> {
    let f = FitsFile::open(test_data(name)).unwrap();
    let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
        panic!("{name}: HDU 1 is not a compressed image");
    };
    c.decompress().unwrap()
}

#[test]
fn rice_i16_matches_reference() {
    let raw = decompress("ref_rice_i16.fits");
    let expected: Vec<i16> = lcg(33 * 17)
        .iter()
        .map(|v| (v.rem_euclid(1000) - 500) as i16)
        .collect();
    assert_eq!(raw.len(), expected.len() * 2, "decoded byte length");
    let decoded: Vec<i16> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(
        decoded, expected,
        "RICE_1 i16 decode differs from the reference fixture"
    );
}

#[test]
fn rice_i32_matches_reference() {
    let raw = decompress("ref_rice_i32.fits");
    let expected: Vec<i32> = lcg(40 * 12)
        .iter()
        .map(|v| (v - (1 << 30)) as i32)
        .collect();
    assert_eq!(raw.len(), expected.len() * 4, "decoded byte length");
    let decoded: Vec<i32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(
        decoded, expected,
        "RICE_1 i32 decode differs from the reference fixture"
    );
}

#[test]
fn hcompress_i32_large_values_matches_reference() {
    // Values near +-2^30: the H-transform coefficients exceed i32, so this
    // exercises the 64-bit coefficient path (cfitsio `fits_hdecompress64`).
    let raw = decompress("ref_hcomp_i32.fits");
    let expected: Vec<i32> = lcg(64 * 64)
        .iter()
        .map(|v| (v - (1 << 30)) as i32)
        .collect();
    assert_eq!(raw.len(), expected.len() * 4, "decoded byte length");
    let decoded: Vec<i32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(
        decoded, expected,
        "HCOMPRESS_1 i32 decode differs from the reference fixture"
    );
}

#[test]
fn subtractive_dither_f32_matches_reference() {
    // Quantized float tile with SUBTRACTIVE_DITHER_1 and ZDITHER0 = 42.
    // The de-dithered floats must match astropy's own decode bit-for-bit
    // (HDU "EXPECTED" stores astropy's output); GZIP_1 payload isolates
    // the dither sequence from any entropy-codec concern.
    let f = FitsFile::open(test_data("ref_dither_f32.fits")).unwrap();
    let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
        panic!("HDU 1 is not a compressed image");
    };
    let raw = c.decompress().unwrap();
    let decoded: Vec<u32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let Hdu::Image(exp) = f.hdu(2).unwrap() else {
        panic!("HDU 2 is not the EXPECTED image");
    };
    let expected: Vec<u32> = exp
        .raw_bytes()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    assert_eq!(decoded.len(), expected.len(), "pixel count");
    // Compare raw f32 bit patterns: any dither-seed offset shows up as
    // quantization-level noise that a tolerance compare could miss.
    assert_eq!(
        decoded, expected,
        "de-dithered floats differ from the reference decode"
    );
}

/// Standard Sec.10 Table 10 lists `NOCOMPRESS` as a valid `ZCMPTYPE`
/// alongside the five real algorithms: the tile bytes are stored
/// verbatim and the HDU "remains uncompressed".
///
/// Regression: fitsy rejected the mnemonic outright, so a file any
/// astropy user can produce with one kwarg
/// (`compression_type='NOCOMPRESS'`, which astropy lists in
/// `COMPRESSION_TYPES`, writes, and reads back) failed at the first
/// pixel read -- and only there, since the header alone parses fine.
/// Fixture from `tests/data/gen_nocompress.py`.
#[test]
fn nocompress_i16_matches_reference() {
    let raw = decompress("ref_nocompress.fits");
    let expected: Vec<i16> = (0..6 * 8_i16).map(|v| v * 37 - 500).collect();
    assert_eq!(raw.len(), expected.len() * 2, "decoded byte length");
    let decoded: Vec<i16> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(
        decoded, expected,
        "NOCOMPRESS i16 decode differs from the reference fixture"
    );
}

/// The float half of the same fixture. A lossless float image is
/// normally restricted to `GZIP_1`/`GZIP_2`; `NOCOMPRESS` stores the
/// IEEE bytes untouched, so it cannot lose anything and has to pass
/// that guard too.
#[test]
fn nocompress_f32_matches_reference() {
    let f = FitsFile::open(test_data("ref_nocompress.fits")).unwrap();
    let Hdu::CompressedImage(c) = f.hdu(2).unwrap() else {
        panic!("HDU 2 is not a compressed image");
    };
    let raw = c.decompress().unwrap();
    let expected: Vec<f32> = (0..6 * 8).map(|v| (v as f32) * 0.25 - 3.5).collect();
    assert_eq!(raw.len(), expected.len() * 4, "decoded byte length");
    let decoded: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(
        decoded, expected,
        "NOCOMPRESS f32 decode differs from the reference fixture"
    );
}

/// The `RICE_1` encoder reproduces the reference streams byte for
/// byte. Measured over the i16 fixture (15 tiles), the i32 fixture
/// (4 tiles, values near +-2^30) and an astropy-written full-range
/// byte image: every tile matched. This test pins the two fixtures.
///
/// Byte identity is stronger than the format requires -- any stream
/// that decodes to the same pixels is valid -- so a failure here
/// after an encoder change means "re-verify against astropy", not
/// necessarily "wrong".
#[test]
fn rice_encoder_matches_reference_streams() {
    use fitsy::header::Value;
    use fitsy::{Header, TileOpts, compress_image_to_hdu};

    // (fixture, BITPIX, tile shape, pixel builder)
    let cases: [(&str, i64, [u64; 2], Vec<u8>); 2] = [
        (
            "ref_rice_i16.fits",
            16,
            [16, 4],
            lcg(33 * 17)
                .iter()
                .flat_map(|v| ((v.rem_euclid(1000) - 500) as i16).to_be_bytes())
                .collect(),
        ),
        (
            "ref_rice_i32.fits",
            32,
            [40, 3],
            lcg(40 * 12)
                .iter()
                .flat_map(|v| ((v - (1 << 30)) as i32).to_be_bytes())
                .collect(),
        ),
    ];
    for (name, bitpix, tile, data) in cases {
        let f = FitsFile::open(test_data(name)).unwrap();
        let Hdu::CompressedImage(c) = f.hdu(1).unwrap() else {
            panic!("{name}: HDU 1 is not a compressed image");
        };
        let mut h = Header::empty();
        h.push("XTENSION", Value::String("IMAGE".into()), None)
            .unwrap();
        h.push("BITPIX", Value::Integer(bitpix), None).unwrap();
        h.push("NAXIS", Value::Integer(2), None).unwrap();
        for (i, n) in c.axes().iter().enumerate() {
            h.push(format!("NAXIS{}", i + 1), Value::Integer(*n as i64), None)
                .unwrap();
        }
        h.push("PCOUNT", Value::Integer(0), None).unwrap();
        h.push("GCOUNT", Value::Integer(1), None).unwrap();
        let opts = TileOpts::new()
            .codec(fitsy::Codec::rice())
            .tile(tile.to_vec());
        let (bh, bytes) = compress_image_to_hdu(&h, &data, &opts).unwrap();

        // The heap holds the tile payloads in row order in both
        // files, so heap equality is stream equality.
        let heap_start = bh.optional_int("NAXIS1").unwrap() as usize
            * bh.optional_int("NAXIS2").unwrap() as usize;
        assert_eq!(
            &bytes[heap_start..],
            c.as_bintable().heap_bytes(),
            "{name}: RICE_1 stream differs from the reference"
        );
    }
}
