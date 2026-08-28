//! Tile-compressed IMAGE write -> read round-trip (`GZIP_1`).
#![cfg(feature = "compression")]

use fitsy::{FitsFile, FitsWriter, Hdu, ImageBuilder, TileOpts};

#[test]
fn gzip1_compressed_image_round_trips() {
    // 8x6 image, BITPIX=16. Default tiling: NAXIS1x1 = 8x1, so 6
    // tiles, one per row.
    let pixels: Vec<i16> = (0..48_i16).map(|i| i * 3 - 17).collect();
    let (img_h, img_data) = ImageBuilder::<i16>::new(vec![8_u64, 6], pixels.clone())
        .unwrap()
        .primary(false)
        .build()
        .unwrap();

    let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary.0, &primary.1).unwrap();
    w.write_hdu_compressed(&img_h, &img_data, &TileOpts::new())
        .unwrap();
    w.finish().unwrap();

    let parsed = FitsFile::from_bytes(buf).unwrap();
    assert_eq!(parsed.len(), 2);
    let Hdu::CompressedImage(ci) = parsed.hdu(1).unwrap() else {
        panic!("not a compressed image: {:?}", parsed.hdu(1).unwrap());
    };
    let img = ci.as_image().unwrap();
    assert_eq!(img.axes(), &[8_u64, 6]);
    let got: Vec<i16> = img
        .raw_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(got, pixels);
}

#[test]
fn gzip1_3d_with_custom_tiles() {
    // 4x4x2 i32 image. Tile shape (2,2,1) -> 2*2*2 = 8 tiles.
    let pixels: Vec<i32> = (0..32).collect();
    let (img_h, img_data) = ImageBuilder::<i32>::new(vec![4_u64, 4, 2], pixels.clone())
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary.0, &primary.1).unwrap();
    w.write_hdu_compressed(&img_h, &img_data, &TileOpts::new().tile(vec![2_u64, 2, 1]))
        .unwrap();
    w.finish().unwrap();
    let parsed = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(ci) = parsed.hdu(1).unwrap() else {
        panic!("not compressed image");
    };
    let img = ci.as_image().unwrap();
    assert_eq!(img.axes(), &[4_u64, 4, 2]);
    let got: Vec<i32> = img
        .raw_bytes()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got, pixels);
}

// -- Codec coverage --------------------------------------------------

use fitsy::Codec;
use fitsy::header::CommentaryKind;

/// Compress `(header, data)` with `opts`, parse the result, and
/// return the decompressed image from HDU 1.
fn round_trip(
    header: &fitsy::Header,
    data: &[u8],
    opts: &TileOpts,
) -> fitsy::compression::OwnedImage {
    let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary.0, &primary.1).unwrap();
    w.write_hdu_compressed(header, data, opts).unwrap();
    w.finish().unwrap();
    let parsed = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(ci) = parsed.hdu(1).unwrap() else {
        panic!("HDU 1 is not a compressed image");
    };
    ci.as_image().unwrap()
}

/// One deterministic image per `BITPIX`, 21 x 13 so the default
/// row tiling and a 2-D tile both hit partial edge tiles.
fn sample_image(bitpix: i64) -> (fitsy::Header, Vec<u8>) {
    let axes = vec![21_u64, 13];
    let n: usize = 21 * 13;
    macro_rules! build {
        ($t:ty, $f:expr) => {
            ImageBuilder::<$t>::new(axes, (0..n).map($f).collect::<Vec<$t>>())
                .unwrap()
                .primary(false)
                .build()
                .unwrap()
        };
    }
    match bitpix {
        8 => build!(u8, |i| ((i * 7) % 256) as u8),
        16 => build!(i16, |i| (i as i16) * 89 - 9000),
        32 => build!(i32, |i| (i as i32) * 1_000_003 - 100_000_000),
        64 => build!(i64, |i| (i as i64) * 4_000_000_007 - 5_000_000_000),
        -32 => build!(f32, |i| (i as f32) * 0.37 - 20.0),
        -64 => build!(f64, |i| (i as f64) * 0.37 - 20.0),
        _ => unreachable!("test covers the six legal BITPIX values"),
    }
}

#[test]
fn every_codec_against_every_bitpix() {
    let codecs = [Codec::Gzip1, Codec::Gzip2, Codec::rice()];
    for &bitpix in &[8_i64, 16, 32, 64, -32, -64] {
        let (h, data) = sample_image(bitpix);
        for &codec in &codecs {
            let opts = TileOpts::new().codec(codec);
            let rice_applies = matches!(bitpix, 8 | 16 | 32);
            if matches!(codec, Codec::Rice1 { .. }) && !rice_applies {
                // RICE_1 takes 1/2/4-byte integer pixels only.
                let r = fitsy::compress_image_to_hdu(&h, &data, &opts);
                assert!(r.is_err(), "RICE_1 must reject BITPIX={bitpix}");
                continue;
            }
            let img = round_trip(&h, &data, &opts);
            assert_eq!(
                img.raw_bytes(),
                &data[..],
                "codec {codec:?} BITPIX {bitpix} round trip"
            );
        }
    }
}

#[test]
fn rice_full_byte_range() {
    // BITPIX = 8 pixels are unsigned; the Rice coder differences them
    // with wrapping arithmetic, so the full 0..=255 range must
    // survive, not just small values.
    let pixels: Vec<u8> = (0..1024).map(|i| (i * 251 % 256) as u8).collect();
    let (h, data) = ImageBuilder::<u8>::new(vec![256_u64, 4], pixels)
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let img = round_trip(&h, &data, &TileOpts::new().codec(Codec::rice()));
    assert_eq!(img.raw_bytes(), &data[..]);
}

#[test]
fn rice_partial_edge_tiles() {
    // 21 x 13 with 8 x 8 tiles: partial tiles along both axes.
    let (h, data) = sample_image(16);
    let opts = TileOpts::new().codec(Codec::rice()).tile(vec![8_u64, 8]);
    let img = round_trip(&h, &data, &opts);
    assert_eq!(img.raw_bytes(), &data[..]);
}

#[test]
fn rice_3d_custom_tiles() {
    let pixels: Vec<i32> = (0..5 * 7 * 3).map(|i| i * 17 - 100).collect();
    let (h, data) = ImageBuilder::<i32>::new(vec![5_u64, 7, 3], pixels)
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let opts = TileOpts::new().codec(Codec::rice()).tile(vec![3_u64, 4, 2]);
    let img = round_trip(&h, &data, &opts);
    assert_eq!(img.axes(), &[5_u64, 7, 3]);
    assert_eq!(img.raw_bytes(), &data[..]);
}

// -- Header carry-through --------------------------------------------

#[test]
fn header_cards_survive_compression() {
    let (h, data) = ImageBuilder::<i16>::new(vec![8_u64, 8], vec![3_i16; 64])
        .unwrap()
        .primary(false)
        .card("OBJECT", "M31", Some("target"))
        .card("EXPTIME", 30.0, None)
        .card("CTYPE1", "RA---TAN", None)
        .card("CRVAL1", 10.68, None)
        .build()
        .unwrap();
    let img = round_trip(&h, &data, &TileOpts::new());
    let out = img.header();
    assert_eq!(out.optional_string("OBJECT"), Some("M31"));
    assert_eq!(out.optional_real("EXPTIME"), Some(30.0));
    assert_eq!(out.optional_string("CTYPE1"), Some("RA---TAN"));
    assert_eq!(out.optional_real("CRVAL1"), Some(10.68));
    // The comment survives with its card.
    let entry = out
        .entries()
        .iter()
        .find(|e| e.keyword == "OBJECT")
        .unwrap();
    assert_eq!(entry.comment.as_deref(), Some("target"));
}

#[test]
fn commentary_cards_survive_compression() {
    let (mut h, data) = ImageBuilder::<i16>::new(vec![4_u64, 4], vec![0_i16; 16])
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    h.push_commentary(CommentaryKind::History, "flat-fielded");
    h.push_commentary(CommentaryKind::Comment, "unit test");
    let img = round_trip(&h, &data, &TileOpts::new());
    let texts: Vec<(&str, &str)> = img
        .header()
        .entries()
        .iter()
        .filter_map(|e| e.commentary.as_deref().map(|t| (e.keyword.as_str(), t)))
        .collect();
    assert!(texts.contains(&("HISTORY", "flat-fielded")));
    assert!(texts.contains(&("COMMENT", "unit test")));
}

#[test]
fn blank_survives_compression() {
    let (h, data) = ImageBuilder::<i16>::new(vec![4_u64, 4], vec![-32768_i16; 16])
        .unwrap()
        .primary(false)
        .card("BLANK", -32768_i64, None)
        .build()
        .unwrap();
    let img = round_trip(&h, &data, &TileOpts::new().codec(Codec::rice()));
    assert_eq!(img.header().optional_int("BLANK"), Some(-32768));
    assert_eq!(img.raw_bytes(), &data[..]);
}

#[test]
fn primary_image_records_zsimple_but_reads_as_an_extension() {
    // A primary array compresses with ZSIMPLE = T, which `was_primary`
    // reports. The decompressed view still describes the HDU where it
    // sits -- an IMAGE extension -- so it can be written back into the
    // slot it came from.
    let (h, data) = ImageBuilder::<i16>::new(vec![6_u64, 6], (0..36_i16).collect::<Vec<_>>())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary.0, &primary.1).unwrap();
    w.write_hdu_compressed(&h, &data, &TileOpts::new()).unwrap();
    w.finish().unwrap();
    let parsed = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(ci) = parsed.hdu(1).unwrap() else {
        panic!("HDU 1 is not a compressed image");
    };
    assert!(ci.was_primary(), "ZSIMPLE = T was not recorded");

    let img = ci.as_image().unwrap();
    let out = img.header();
    assert_eq!(out.optional_string("XTENSION"), Some("IMAGE"));
    assert!(out.first("SIMPLE").is_none());
    assert_eq!(out.optional_int("PCOUNT"), Some(0));
    assert_eq!(out.optional_int("GCOUNT"), Some(1));
    assert_eq!(img.raw_bytes(), &data[..]);
}

/// The decompressed view of every HDU must be writable back into the
/// slot it came from. A header shaped for a different slot fails
/// `FitsWriter`'s mandatory-keyword check.
#[test]
fn decompressed_hdus_can_be_written_straight_back() {
    let (h, data) = ImageBuilder::<i16>::new(vec![6_u64, 6], (0..36_i16).collect::<Vec<_>>())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let stub = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&stub.0, &stub.1).unwrap();
    w.write_hdu_compressed(&h, &data, &TileOpts::new()).unwrap();
    w.finish().unwrap();

    let parsed = FitsFile::from_bytes(buf).unwrap();
    let mut out = Vec::new();
    let mut w2 = FitsWriter::new(&mut out);
    for i in 0..parsed.len() {
        match parsed.hdu(i).unwrap() {
            Hdu::CompressedImage(c) => {
                let img = c.as_image().unwrap();
                w2.write_hdu(img.header(), img.raw_bytes())
                    .unwrap_or_else(|e| panic!("HDU {i} could not be written back: {e}"));
            }
            other => {
                w2.write_hdu(other.header(), other.data_bytes()).unwrap();
            }
        }
    }
    w2.finish().unwrap();
    let reparsed = FitsFile::from_bytes(out).unwrap();
    assert_eq!(reparsed.len(), 2);
    let Hdu::Image(img) = reparsed.hdu(1).unwrap() else {
        panic!("HDU 1 is not a plain image");
    };
    assert_eq!(img.raw_bytes(), &data[..]);
}

#[test]
fn source_extname_carries_unless_overridden() {
    let (h, data) = ImageBuilder::<i16>::new(vec![4_u64, 4], vec![0_i16; 16])
        .unwrap()
        .primary(false)
        .card("EXTNAME", "SCI", None)
        .build()
        .unwrap();
    let (bh, _) = fitsy::compress_image_to_hdu(&h, &data, &TileOpts::new()).unwrap();
    assert_eq!(bh.optional_string("EXTNAME"), Some("SCI"));
    let (bh, _) =
        fitsy::compress_image_to_hdu(&h, &data, &TileOpts::new().extname("TILED")).unwrap();
    assert_eq!(bh.optional_string("EXTNAME"), Some("TILED"));
}

// -- Quantization ----------------------------------------------------

use fitsy::compression::{DitherMethod, Quantize};

/// A float image with deterministic pseudo-noise, sigma of order 1.
///
/// The noise comes from a 32-bit LCG, so a second difference is
/// essentially never exactly zero. A low-period integer pattern would
/// read as a flat tile and refuse to quantize, which would make these
/// tests pass through the lossless fallback without ever exercising
/// the quantizer.
fn noisy_f32(n: usize) -> Vec<f32> {
    let mut x: u32 = 12_345;
    (0..n)
        .map(|i| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (i as f32) * 0.01 + (x as f32 / u32::MAX as f32) * 4.0
        })
        .collect()
}

/// Assert that `opts` really quantized: a `ZSCALE` column exists, so
/// the tiles went through the quantizer rather than the lossless
/// fallback.
fn assert_quantized(header: &fitsy::Header, data: &[u8], opts: &TileOpts) {
    let (bh, _) = fitsy::compress_image_to_hdu(header, data, opts).unwrap();
    let has_zscale = (1..=bh.optional_int("TFIELDS").unwrap_or(0)).any(|i| {
        bh.optional_string(&format!("TTYPE{i}"))
            .is_some_and(|t| t.trim() == "ZSCALE")
    });
    assert!(has_zscale, "expected a ZSCALE column; the tiles fell back");
}

#[test]
fn quantized_f32_round_trips_within_step() {
    let pixels = noisy_f32(64 * 32);
    let (h, data) = ImageBuilder::<f32>::new(vec![64_u64, 32], pixels.clone())
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let opts = TileOpts::new()
        .codec(Codec::rice())
        .quantize(Quantize::level(4.0));
    assert_quantized(&h, &data, &opts);
    let img = round_trip(&h, &data, &opts);
    let decoded: Vec<f32> = img
        .raw_bytes()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // The quantization step is the tile noise over the level; with
    // sigma of order 1 and level 4 every value lands well inside 1.0.
    for (a, b) in pixels.iter().zip(&decoded) {
        assert!((a - b).abs() < 1.0, "{a} decoded to {b}");
    }
}

#[test]
fn quantize_falls_back_losslessly_on_a_flat_tile_with_outliers() {
    // No noise to hide a step in. Quantizing would let the outliers
    // set the step and wreck the zeros around them, so every tile
    // must fall back to lossless gzip and round-trip exactly.
    let mut pixels = vec![0.0_f32; 64 * 16];
    for i in (0..pixels.len()).step_by(200) {
        pixels[i] = 1000.0;
    }
    let (h, data) = ImageBuilder::<f32>::new(vec![64_u64, 16], pixels)
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let opts = TileOpts::new()
        .codec(Codec::rice())
        .quantize(Quantize::level(4.0));
    let img = round_trip(&h, &data, &opts);
    assert_eq!(img.raw_bytes(), &data[..]);
}

#[test]
fn rice_rejects_a_zero_block_size() {
    // A zero block size cannot advance through a tile, and the
    // decoder rejects the resulting stream anyway.
    let (h, data) = sample_image(16);
    let opts = TileOpts::new().codec(Codec::Rice1 { blocksize: 0 });
    assert!(fitsy::compress_image_to_hdu(&h, &data, &opts).is_err());
}

#[test]
fn quantized_nan_and_zero_survive() {
    let mut pixels = noisy_f32(48 * 16);
    pixels[5] = f32::NAN;
    pixels[100] = 0.0;
    let (h, data) = ImageBuilder::<f32>::new(vec![48_u64, 16], pixels)
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let q = Quantize {
        dither: DitherMethod::Subtractive2,
        ..Quantize::default()
    };
    let opts = TileOpts::new().codec(Codec::rice()).quantize(q);
    let img = round_trip(&h, &data, &opts);
    let decoded: Vec<f32> = img
        .raw_bytes()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(decoded[5].is_nan());
    assert_eq!(decoded[100], 0.0);
}

#[test]
fn quantized_constant_image_falls_back_losslessly() {
    // No measurable noise: every tile falls back to lossless gzip in
    // GZIP_COMPRESSED_DATA, so the round trip is exact.
    let (h, data) = ImageBuilder::<f64>::new(vec![32_u64, 8], vec![2.5_f64; 256])
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let opts = TileOpts::new()
        .codec(Codec::rice())
        .quantize(Quantize::default());
    let img = round_trip(&h, &data, &opts);
    assert_eq!(img.raw_bytes(), &data[..]);
}

#[test]
fn quantize_rejects_integer_images() {
    let (h, data) = sample_image(16);
    let opts = TileOpts::new().quantize(Quantize::default());
    assert!(fitsy::compress_image_to_hdu(&h, &data, &opts).is_err());
}
