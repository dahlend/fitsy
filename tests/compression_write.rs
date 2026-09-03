//! Tile-compressed IMAGE write -> read round-trip (`GZIP_1`).
#![cfg(feature = "compression")]

use fitsy::{FitsFile, FitsWriter, Hdu, ImageBuilder, TileOpts};

#[test]
fn gzip1_compressed_image_round_trips() {
    // 8x6 image, BITPIX=16. Default tiling: NAXIS1x1 = 8x1, so 6
    // tiles, one per row.
    let pixels: Vec<i16> = (0..48_i16).map(|i| i * 3 - 17).collect();
    let img_h = ImageBuilder::<i16>::new(vec![8_u64, 6], pixels.clone())
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
    w.write_hdu(&primary).unwrap();
    w.write_hdu_compressed(&img_h, &TileOpts::new()).unwrap();
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
    let img_h = ImageBuilder::<i32>::new(vec![4_u64, 4, 2], pixels.clone())
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
    w.write_hdu(&primary).unwrap();
    w.write_hdu_compressed(&img_h, &TileOpts::new().tile(vec![2_u64, 2, 1]))
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

/// Compress `img` with `opts`, parse the result, and return the
/// decompressed image from HDU 1.
fn round_trip(img: &fitsy::ImageHdu<'_>, opts: &TileOpts) -> fitsy::ImageHdu<'static> {
    let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary).unwrap();
    w.write_hdu_compressed(img, opts).unwrap();
    w.finish().unwrap();
    let parsed = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(ci) = parsed.hdu(1).unwrap() else {
        panic!("HDU 1 is not a compressed image");
    };
    ci.as_image().unwrap()
}

/// One deterministic image per `BITPIX`, 21 x 13 so the default
/// row tiling and a 2-D tile both hit partial edge tiles.
fn sample_image(bitpix: i64) -> fitsy::ImageHdu<'static> {
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
        let h = sample_image(bitpix);
        for &codec in &codecs {
            let opts = TileOpts::new().codec(codec);
            let rice_applies = matches!(bitpix, 8 | 16 | 32);
            if matches!(codec, Codec::Rice1 { .. }) && !rice_applies {
                // RICE_1 takes 1/2/4-byte integer pixels only.
                let r = fitsy::compress_image_to_hdu(&h, &opts);
                assert!(r.is_err(), "RICE_1 must reject BITPIX={bitpix}");
                continue;
            }
            let img = round_trip(&h, &opts);
            assert_eq!(
                img.raw_bytes(),
                h.raw_bytes(),
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
    let h = ImageBuilder::<u8>::new(vec![256_u64, 4], pixels)
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let img = round_trip(&h, &TileOpts::new().codec(Codec::rice()));
    assert_eq!(img.raw_bytes(), h.raw_bytes());
}

#[test]
fn rice_partial_edge_tiles() {
    // 21 x 13 with 8 x 8 tiles: partial tiles along both axes.
    let h = sample_image(16);
    let opts = TileOpts::new().codec(Codec::rice()).tile(vec![8_u64, 8]);
    let img = round_trip(&h, &opts);
    assert_eq!(img.raw_bytes(), h.raw_bytes());
}

#[test]
fn rice_3d_custom_tiles() {
    let pixels: Vec<i32> = (0..5 * 7 * 3).map(|i| i * 17 - 100).collect();
    let h = ImageBuilder::<i32>::new(vec![5_u64, 7, 3], pixels)
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let opts = TileOpts::new().codec(Codec::rice()).tile(vec![3_u64, 4, 2]);
    let img = round_trip(&h, &opts);
    assert_eq!(img.axes(), &[5_u64, 7, 3]);
    assert_eq!(img.raw_bytes(), h.raw_bytes());
}

// -- Header carry-through --------------------------------------------

#[test]
fn header_cards_survive_compression() {
    let h = ImageBuilder::<i16>::new(vec![8_u64, 8], vec![3_i16; 64])
        .unwrap()
        .primary(false)
        .card("OBJECT", "M31", Some("target"))
        .card("EXPTIME", 30.0, None)
        .card("CTYPE1", "RA---TAN", None)
        .card("CRVAL1", 10.68, None)
        .build()
        .unwrap();
    let img = round_trip(&h, &TileOpts::new());
    let out = img.header();
    assert_eq!(out.optional_string("OBJECT").as_deref(), Some("M31"));
    assert_eq!(out.optional_real("EXPTIME"), Some(30.0));
    assert_eq!(out.optional_string("CTYPE1").as_deref(), Some("RA---TAN"));
    assert_eq!(out.optional_real("CRVAL1"), Some(10.68));
    // The comment survives with its card.
    let card = out.first_card("OBJECT").unwrap();
    assert_eq!(card.comment().as_deref(), Some("target"));
}

#[test]
fn commentary_cards_survive_compression() {
    let (mut h, data) = ImageBuilder::<i16>::new(vec![4_u64, 4], vec![0_i16; 16])
        .unwrap()
        .primary(false)
        .build()
        .unwrap()
        .into_parts();
    h.push_commentary(CommentaryKind::History, "flat-fielded")
        .unwrap();
    h.push_commentary(CommentaryKind::Comment, "unit test")
        .unwrap();
    let hdu = fitsy::ImageHdu::new(h, data).unwrap();
    let img = round_trip(&hdu, &TileOpts::new());
    let texts: Vec<(String, String)> = img
        .header()
        .cards()
        .filter_map(|c| c.commentary().map(|t| (c.keyword(), t)))
        .collect();
    assert!(texts.contains(&("HISTORY".to_string(), "flat-fielded".to_string())));
    assert!(texts.contains(&("COMMENT".to_string(), "unit test".to_string())));
}

#[test]
fn blank_survives_compression() {
    let h = ImageBuilder::<i16>::new(vec![4_u64, 4], vec![-32768_i16; 16])
        .unwrap()
        .primary(false)
        .card("BLANK", -32768_i64, None)
        .build()
        .unwrap();
    let img = round_trip(&h, &TileOpts::new().codec(Codec::rice()));
    assert_eq!(img.header().optional_int("BLANK"), Some(-32768));
    assert_eq!(img.raw_bytes(), h.raw_bytes());
}

#[test]
fn primary_image_records_zsimple_but_reads_as_an_extension() {
    // A primary array compresses with ZSIMPLE = T, which `was_primary`
    // reports. The decompressed view still describes the HDU where it
    // sits -- an IMAGE extension -- so it can be written back into the
    // slot it came from.
    let h = ImageBuilder::<i16>::new(vec![6_u64, 6], (0..36_i16).collect::<Vec<_>>())
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
    w.write_hdu(&primary).unwrap();
    w.write_hdu_compressed(&h, &TileOpts::new()).unwrap();
    w.finish().unwrap();
    let parsed = FitsFile::from_bytes(buf).unwrap();
    let Hdu::CompressedImage(ci) = parsed.hdu(1).unwrap() else {
        panic!("HDU 1 is not a compressed image");
    };
    assert!(ci.was_primary(), "ZSIMPLE = T was not recorded");

    let img = ci.as_image().unwrap();
    let out = img.header();
    assert_eq!(out.optional_string("XTENSION").as_deref(), Some("IMAGE"));
    assert!(out.first("SIMPLE").is_none());
    assert_eq!(out.optional_int("PCOUNT"), Some(0));
    assert_eq!(out.optional_int("GCOUNT"), Some(1));
    assert_eq!(img.raw_bytes(), h.raw_bytes());
}

/// The decompressed view of every HDU must be writable back into the
/// slot it came from. A header shaped for a different slot fails
/// `FitsWriter`'s mandatory-keyword check.
#[test]
fn decompressed_hdus_can_be_written_straight_back() {
    let h = ImageBuilder::<i16>::new(vec![6_u64, 6], (0..36_i16).collect::<Vec<_>>())
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
    w.write_hdu(&stub).unwrap();
    w.write_hdu_compressed(&h, &TileOpts::new()).unwrap();
    w.finish().unwrap();

    let parsed = FitsFile::from_bytes(buf).unwrap();
    let mut out = Vec::new();
    let mut w2 = FitsWriter::new(&mut out);
    for i in 0..parsed.len() {
        match parsed.hdu(i).unwrap() {
            Hdu::CompressedImage(c) => {
                let img = c.as_image().unwrap();
                w2.write_hdu(&img)
                    .unwrap_or_else(|e| panic!("HDU {i} could not be written back: {e}"));
            }
            other => {
                w2.write_hdu(&other).unwrap();
            }
        }
    }
    w2.finish().unwrap();
    let reparsed = FitsFile::from_bytes(out).unwrap();
    assert_eq!(reparsed.len(), 2);
    let Hdu::Image(img) = reparsed.hdu(1).unwrap() else {
        panic!("HDU 1 is not a plain image");
    };
    assert_eq!(img.raw_bytes(), h.raw_bytes());
}

#[test]
fn source_extname_carries_unless_overridden() {
    let h = ImageBuilder::<i16>::new(vec![4_u64, 4], vec![0_i16; 16])
        .unwrap()
        .primary(false)
        .card("EXTNAME", "SCI", None)
        .build()
        .unwrap();
    let bh = fitsy::compress_image_to_hdu(&h, &TileOpts::new()).unwrap();
    let bh = bh.header();
    assert_eq!(bh.optional_string("EXTNAME").as_deref(), Some("SCI"));
    let bh = fitsy::compress_image_to_hdu(&h, &TileOpts::new().extname("TILED")).unwrap();
    let bh = bh.header();
    assert_eq!(bh.optional_string("EXTNAME").as_deref(), Some("TILED"));
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
fn assert_quantized(img: &fitsy::ImageHdu<'_>, opts: &TileOpts) {
    let bh = fitsy::compress_image_to_hdu(img, opts).unwrap();
    let bh = bh.header();
    let has_zscale = (1..=bh.optional_int("TFIELDS").unwrap_or(0)).any(|i| {
        bh.optional_string(&format!("TTYPE{i}"))
            .is_some_and(|t| t.trim() == "ZSCALE")
    });
    assert!(has_zscale, "expected a ZSCALE column; the tiles fell back");
}

#[test]
fn quantized_f32_round_trips_within_step() {
    let pixels = noisy_f32(64 * 32);
    let h = ImageBuilder::<f32>::new(vec![64_u64, 32], pixels.clone())
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let opts = TileOpts::new()
        .codec(Codec::rice())
        .quantize(Quantize::level(4.0));
    assert_quantized(&h, &opts);
    let img = round_trip(&h, &opts);
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
    let h = ImageBuilder::<f32>::new(vec![64_u64, 16], pixels)
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let opts = TileOpts::new()
        .codec(Codec::rice())
        .quantize(Quantize::level(4.0));
    let img = round_trip(&h, &opts);
    assert_eq!(img.raw_bytes(), h.raw_bytes());
}

#[test]
fn rice_rejects_a_zero_block_size() {
    // A zero block size cannot advance through a tile, and the
    // decoder rejects the resulting stream anyway.
    let h = sample_image(16);
    let opts = TileOpts::new().codec(Codec::Rice1 { blocksize: 0 });
    assert!(fitsy::compress_image_to_hdu(&h, &opts).is_err());
}

#[test]
fn quantized_nan_and_zero_survive() {
    let mut pixels = noisy_f32(48 * 16);
    pixels[5] = f32::NAN;
    pixels[100] = 0.0;
    let h = ImageBuilder::<f32>::new(vec![48_u64, 16], pixels)
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let q = Quantize {
        dither: DitherMethod::Subtractive2,
        ..Quantize::default()
    };
    let opts = TileOpts::new().codec(Codec::rice()).quantize(q);
    let img = round_trip(&h, &opts);
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
    let h = ImageBuilder::<f64>::new(vec![32_u64, 8], vec![2.5_f64; 256])
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let opts = TileOpts::new()
        .codec(Codec::rice())
        .quantize(Quantize::default());
    let img = round_trip(&h, &opts);
    assert_eq!(img.raw_bytes(), h.raw_bytes());
}

#[test]
fn quantize_rejects_integer_images() {
    let h = sample_image(16);
    let opts = TileOpts::new().quantize(Quantize::default());
    assert!(fitsy::compress_image_to_hdu(&h, &opts).is_err());
}

/// Sec.10.2 keeps the original header's structural cards in `Z` form,
/// and a reader rebuilds that header by walking them in the order it
/// finds them. `ZSIMPLE` after `ZBITPIX` rebuilds a header starting
/// with `BITPIX`, which is not a header at all -- cfitsio's `funpack`
/// rejects such a file outright.
#[test]
fn z_structural_cards_follow_image_header_order() {
    let (mut h, data) = ImageBuilder::<i16>::new(vec![8_u64, 8], vec![7_i16; 64])
        .unwrap()
        .primary(true)
        .build()
        .unwrap()
        .into_parts();
    h.push("OBJECT", "M31", None).unwrap();
    let h = fitsy::ImageHdu::new(h, data).unwrap();
    let zh = fitsy::compress_image_to_hdu(&h, &TileOpts::new()).unwrap();
    let zh = zh.header();

    let order: Vec<String> = zh
        .cards()
        .map(|c| c.keyword())
        .filter(|k| {
            matches!(
                k.as_str(),
                "ZSIMPLE" | "ZTENSION" | "ZBITPIX" | "ZNAXIS" | "ZNAXIS1" | "ZNAXIS2" | "ZEXTEND"
            )
        })
        .collect();
    let at = |k: &str| order.iter().position(|x| x == k);
    assert_eq!(at("ZSIMPLE"), Some(0), "ZSIMPLE must lead: {order:?}");
    assert!(at("ZSIMPLE") < at("ZBITPIX"), "{order:?}");
    assert!(at("ZBITPIX") < at("ZNAXIS"), "{order:?}");
    assert!(at("ZNAXIS") < at("ZNAXIS1"), "{order:?}");
    assert!(at("ZNAXIS2") < at("ZEXTEND"), "{order:?}");
}

/// The compressed extension carries the conventional name, which is
/// what a reader looks for.
#[test]
fn compressed_extension_is_named() {
    let h = ImageBuilder::<i16>::new(vec![4_u64, 4], vec![0_i16; 16])
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let zh = fitsy::compress_image_to_hdu(&h, &TileOpts::new()).unwrap();
    let zh = zh.header();
    assert_eq!(
        zh.first("EXTNAME"),
        Some(fitsy::Value::String("COMPRESSED_IMAGE".into()))
    );
}

/// The compressed `BINTABLE` owns the table keyword space, so an image
/// card in it cannot be carried. cfitsio refuses such a header
/// outright and astropy drops the card; fitsy drops it too, so a
/// header compressed by any of the three loses the same cards and no
/// others.
#[test]
fn reserved_table_keywords_are_dropped_like_other_tools() {
    let (mut h, data) = ImageBuilder::<i16>::new(vec![8_u64, 8], vec![1_i16; 64])
        .unwrap()
        .primary(true)
        .build()
        .unwrap()
        .into_parts();
    // Every indexed label cfitsio and astropy reserve.
    for kw in [
        "TTYPE1", "TFORM1", "TUNIT1", "TNULL1", "TSCAL1", "TZERO1", "TDISP1", "TBCOL1", "TDIM1",
        "TCTYP1", "TCUNI1", "TCRPX1", "TCRVL1", "TCDLT1", "TRPOS1",
    ] {
        h.push(kw, "reserved", None).unwrap();
    }
    h.push("THEAP", 10_i64, None).unwrap();
    h.push("OBSERVER", "me", None).unwrap();
    h.push("EXPTIME", 30.0_f64, None).unwrap();

    let h = fitsy::ImageHdu::new(h, data).unwrap();
    let img = round_trip(&h, &TileOpts::new());
    let restored = img.header();
    for kw in ["TTYPE1", "TFORM1", "TCRVL1", "TRPOS1", "THEAP"] {
        assert!(
            restored.first(kw).is_none(),
            "{kw} is reserved by the compressed table and must not come back"
        );
    }
    // Everything the table does not own is untouched.
    assert_eq!(
        restored.first("OBSERVER"),
        Some(fitsy::Value::String("me".into()))
    );
    assert_eq!(restored.first("EXPTIME"), Some(fitsy::Value::Real(30.0)));
}

/// A compressed image written as the first HDU gets the primary stub
/// from the writer.
#[test]
fn write_hdu_compressed_writes_the_primary_stub() {
    let pixels: Vec<i16> = (0..48_i16).map(|i| i * 3 - 17).collect();
    let img = ImageBuilder::<i16>::new(vec![8_u64, 6], pixels.clone())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    // No primary HDU written first. This used to fail with
    // `primary HDU header is missing SIMPLE = T`.
    w.write_hdu_compressed(&img, &TileOpts::new()).unwrap();
    w.finish().unwrap();

    let parsed = FitsFile::from_bytes(buf).unwrap();
    assert_eq!(parsed.len(), 2, "stub plus the compressed image");
    let stub = parsed.hdu(0).unwrap();
    assert!(stub.data_bytes().is_empty());
    assert_eq!(
        stub.header().first("SIMPLE"),
        Some(fitsy::Value::Logical(true))
    );

    let back = parsed.image(1).unwrap();
    assert_eq!(back.axes(), &[8, 6]);
    assert_eq!(
        back.read_raw::<i16>().unwrap().as_slice(),
        pixels.as_slice()
    );
}

/// A caller who writes their own primary HDU gets no stub.
#[test]
fn write_hdu_compressed_adds_no_stub_behind_a_primary_hdu() {
    let pixels: Vec<i16> = (0..48_i16).map(|i| i * 3 - 17).collect();
    let img = ImageBuilder::<i16>::new(vec![8_u64, 6], pixels)
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
        .unwrap()
        .primary(true)
        .card("OBJECT", fitsy::Value::from("mine"), None)
        .build()
        .unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary).unwrap();
    w.write_hdu_compressed(&img, &TileOpts::new()).unwrap();
    w.finish().unwrap();

    let parsed = FitsFile::from_bytes(buf).unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(
        parsed.hdu(0).unwrap().header().first("OBJECT"),
        Some(fitsy::Value::String("mine".into())),
        "the caller's primary HDU is the one that was written"
    );
}

/// `FitsFile::image` returns the same type for a plain and a
/// compressed image, and `into_owned` outlives the file.
#[test]
fn image_returns_one_type_and_into_owned_releases_the_file() {
    let pixels: Vec<i16> = (0..48_i16).map(|i| i * 3 - 17).collect();
    let img = ImageBuilder::<i16>::new(vec![8_u64, 6], pixels.clone())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu_compressed(&img, &TileOpts::new()).unwrap();
    w.finish().unwrap();

    let owned = {
        let parsed = FitsFile::from_bytes(buf).unwrap();
        assert_eq!(parsed.image(0).unwrap().n_elements(), 0, "the stub");
        parsed.image(1).unwrap().into_owned()
    };
    // `parsed` is gone; the image still holds its pixels.
    assert_eq!(
        owned.read_raw::<i16>().unwrap().as_slice(),
        pixels.as_slice()
    );
}

/// `iter_decompressed` yields the item type of `iter`, with a
/// compressed image already decoded.
#[test]
fn iter_decompressed_yields_plain_image_hdus() {
    let pixels: Vec<i16> = (0..48_i16).map(|i| i * 3 - 17).collect();
    let img = ImageBuilder::<i16>::new(vec![8_u64, 6], pixels.clone())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu_compressed(&img, &TileOpts::new()).unwrap();
    w.finish().unwrap();

    let parsed = FitsFile::from_bytes(buf).unwrap();
    let kinds: Vec<bool> = parsed
        .iter_decompressed()
        .map(|h| matches!(h.unwrap(), Hdu::Image(_)))
        .collect();
    assert_eq!(kinds, vec![true, true], "no CompressedImage comes through");

    let Hdu::Image(back) = parsed.iter_decompressed().nth(1).unwrap().unwrap() else {
        panic!("HDU 1 is not an image");
    };
    assert_eq!(
        back.read_raw::<i16>().unwrap().as_slice(),
        pixels.as_slice()
    );
}

/// Build a one-image `.fz` file at a temporary path.
fn write_fz(name: &str) -> std::path::PathBuf {
    let pixels: Vec<i16> = (0..48_i16).map(|i| i * 3 - 17).collect();
    let img = ImageBuilder::<i16>::new(vec![8_u64, 6], pixels)
        .unwrap()
        .primary(true)
        .card("OBJECT", fitsy::Value::from("target"), None)
        .build()
        .unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu_compressed(&img, &TileOpts::new()).unwrap();
    w.finish().unwrap();

    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, buf).unwrap();
    path
}

/// `write_decompressed` reverses the stub: the image returns to the
/// primary slot and the file holds one HDU fewer.
#[test]
fn write_decompressed_reverses_the_primary_stub() {
    let path = write_fz("fitsy_test_write_decompressed.fits.fz");
    let out = std::env::temp_dir().join("fitsy_test_write_decompressed.fits");

    let f = FitsFile::open(&path).unwrap();
    assert_eq!(f.len(), 2);
    assert_eq!(f.write_decompressed(&out, true, true).unwrap(), 1);

    let back = FitsFile::open(&out).unwrap();
    assert_eq!(back.len(), 1, "the stub is gone");
    let head = back.image_header(0).unwrap();
    assert_eq!(head.first("SIMPLE"), Some(fitsy::Value::Logical(true)));
    assert!(head.first("XTENSION").is_none());
    assert!(head.first("PCOUNT").is_none());
    assert_eq!(
        head.first("OBJECT"),
        Some(fitsy::Value::String("target".into()))
    );

    std::fs::remove_file(&path).unwrap();
    std::fs::remove_file(&out).unwrap();
}

/// A tile-compressed HDU takes no in-place patch, and says so.
#[test]
fn updater_reports_a_compressed_hdu() {
    let path = write_fz("fitsy_test_updater_compressed.fits.fz");
    let mut up = fitsy::FitsUpdater::open(&path).unwrap();

    assert!(up.image_axes(1).is_none());
    assert!(up.image_bitpix(1).is_none());

    let msg = up
        .write_image_subarray::<i16>(1, &[0, 0], &[1, 1], &[7_i16])
        .unwrap_err()
        .to_string();
    assert!(
        msg.contains("tile-compressed"),
        "the error must name the real cause, got: {msg}"
    );

    std::fs::remove_file(&path).unwrap();
}

/// `EXTNAME = COMPRESSED_IMAGE` names the table, so it does not reach
/// the recovered image header. Any other name does, as astropy does.
#[test]
fn placeholder_extname_does_not_reach_the_image_header() {
    let img = ImageBuilder::<i16>::new(vec![8_u64, 6], (0..48_i16).collect::<Vec<_>>())
        .unwrap()
        .primary(false)
        .build()
        .unwrap();

    assert!(
        round_trip(&img, &TileOpts::new())
            .header()
            .first("EXTNAME")
            .is_none(),
        "the placeholder names the compressed table, not the image"
    );
    assert_eq!(
        round_trip(&img, &TileOpts::new().extname("SCI"))
            .header()
            .first("EXTNAME"),
        Some(fitsy::Value::String("SCI".into()))
    );
}

/// The image header is available without decoding pixels, and it is
/// the header `image` reports. The WCS reads out of it.
#[test]
fn image_header_matches_the_decompressed_header() {
    let img = ImageBuilder::<i16>::new(vec![8_u64, 6], (0..48_i16).collect::<Vec<_>>())
        .unwrap()
        .primary(true)
        .card("CTYPE1", fitsy::Value::from("RA---TAN"), None)
        .card("CTYPE2", fitsy::Value::from("DEC--TAN"), None)
        .card("CRVAL1", fitsy::Value::Real(10.0), None)
        .card("CRVAL2", fitsy::Value::Real(20.0), None)
        .card("CRPIX1", fitsy::Value::Real(4.0), None)
        .card("CRPIX2", fitsy::Value::Real(3.0), None)
        .card("CDELT1", fitsy::Value::Real(-0.001), None)
        .card("CDELT2", fitsy::Value::Real(0.001), None)
        .build()
        .unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu_compressed(&img, &TileOpts::new()).unwrap();
    w.finish().unwrap();
    let f = FitsFile::from_bytes(buf).unwrap();

    let cheap: Vec<Vec<u8>> = f
        .image_header(1)
        .unwrap()
        .cards()
        .map(|c| c.raw().to_vec())
        .collect();
    let full: Vec<Vec<u8>> = f
        .image(1)
        .unwrap()
        .header()
        .cards()
        .map(|c| c.raw().to_vec())
        .collect();
    assert_eq!(cheap, full, "the cheap path and the full decode agree");

    let wcs = f.wcs(1, ' ').unwrap().expect("the header declares a WCS");
    let world = wcs.pixel_to_world(&[3.0, 2.0]).unwrap();
    assert!((world[0] - 10.0).abs() < 1e-9, "got {world:?}");
    assert!((world[1] - 20.0).abs() < 1e-9, "got {world:?}");
}

/// `kind` classifies from the header, and `Hdu::bintable` reaches the
/// table under a compressed image.
#[test]
fn kind_and_bintable_see_a_compressed_image_for_what_it_is() {
    let img = ImageBuilder::<i16>::new(vec![8_u64, 6], (0..48_i16).collect::<Vec<_>>())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu_compressed(&img, &TileOpts::new()).unwrap();
    w.finish().unwrap();

    let f = FitsFile::from_bytes(buf).unwrap();
    assert_eq!(f.kind(0).unwrap(), fitsy::HduKind::Image);
    assert_eq!(f.kind(1).unwrap(), fitsy::HduKind::CompressedImage);
    assert!(f.kind(1).unwrap().is_image());

    let hdu = f.hdu(1).unwrap();
    let bt = hdu
        .bintable()
        .expect("a compressed image is a binary table");
    assert!(bt.column_by_name("COMPRESSED_DATA").is_some());
    assert_eq!(bt.n_rows(), 6, "one row per tile, one tile per row");
    assert!(f.hdu(0).unwrap().bintable().is_none());
}

/// A primary HDU that carries a card of its own is not a bare stub,
/// so `write_decompressed` keeps it rather than dropping it.
#[test]
fn write_decompressed_keeps_a_primary_hdu_that_holds_metadata() {
    let img = ImageBuilder::<i16>::new(vec![8_u64, 6], (0..48_i16).collect::<Vec<_>>())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
        .unwrap()
        .primary(true)
        .card("TELESCOP", fitsy::Value::from("mine"), None)
        .build()
        .unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary).unwrap();
    w.write_hdu_compressed(&img, &TileOpts::new()).unwrap();
    w.finish().unwrap();

    let path = std::env::temp_dir().join("fitsy_test_keep_primary.fits.fz");
    let out = std::env::temp_dir().join("fitsy_test_keep_primary.fits");
    std::fs::write(&path, buf).unwrap();

    let f = FitsFile::open(&path).unwrap();
    f.write_decompressed(&out, true, true).unwrap();

    let back = FitsFile::open(&out).unwrap();
    assert_eq!(back.len(), 2, "the caller's primary HDU is kept");
    assert_eq!(
        back.parsed_header(0).unwrap().first("TELESCOP"),
        Some(fitsy::Value::String("mine".into()))
    );

    std::fs::remove_file(&path).unwrap();
    std::fs::remove_file(&out).unwrap();
}
