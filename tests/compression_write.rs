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
        .chunks_exact(2)
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
        .chunks_exact(4)
        .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got, pixels);
}
