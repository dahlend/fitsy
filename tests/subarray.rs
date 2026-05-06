//! Sub-array image reads and sub-row binary table reads.

use fitsy::{BinFieldKind, BinTableBuilder, FitsFile, FitsWriter, Hdu, ImageBuilder};

#[test]
fn read_subarray_2d() {
    // 4x3 image (NAXIS1=4, NAXIS2=3). Pixel value = row*10 + col.
    // FITS row-major: NAXIS1 is fastest, so flat index = y*4 + x.
    let mut pixels: Vec<i16> = Vec::with_capacity(12);
    for y in 0..3 {
        for x in 0..4 {
            pixels.push((y * 10 + x) as i16);
        }
    }
    let (h, data) = ImageBuilder::<i16>::new(vec![4_u64, 3], pixels.clone())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&h, &data).unwrap();
    w.finish().unwrap();
    let parsed = FitsFile::from_bytes(buf).unwrap();
    let Hdu::Image(img) = parsed.hdu(0).unwrap() else {
        panic!("not image");
    };

    // Read 2x2 region starting at (NAXIS1=1, NAXIS2=1) -- should be
    // [[11,12],[21,22]].
    let region = img.read_subarray::<i16>(&[1, 1], &[2, 2]).unwrap();
    assert_eq!(region.axes(), &[2_u64, 2]);
    assert_eq!(region.into_vec(), vec![11_i16, 12, 21, 22]);

    // Full read via sub-array equals full read via read_raw.
    let full = img.read_subarray::<i16>(&[0, 0], &[4, 3]).unwrap();
    assert_eq!(full.into_vec(), pixels);
}

#[test]
fn read_subarray_3d() {
    // 2x3x2: NAXIS1=2, NAXIS2=3, NAXIS3=2.
    let mut pixels: Vec<i32> = Vec::with_capacity(12);
    for z in 0..2 {
        for y in 0..3 {
            for x in 0..2 {
                pixels.push(z * 100 + y * 10 + x);
            }
        }
    }
    let (h, data) = ImageBuilder::<i32>::new(vec![2_u64, 3, 2], pixels.clone())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&h, &data).unwrap();
    w.finish().unwrap();
    let parsed = FitsFile::from_bytes(buf).unwrap();
    let Hdu::Image(img) = parsed.hdu(0).unwrap() else {
        panic!("not image");
    };
    // Slice the second z-plane fully: start=(0,0,1), shape=(2,3,1)
    let plane = img.read_subarray::<i32>(&[0, 0, 1], &[2, 3, 1]).unwrap();
    assert_eq!(plane.axes(), &[2_u64, 3, 1]);
    assert_eq!(plane.into_vec(), vec![100, 101, 110, 111, 120, 121]);
}

#[test]
fn read_subarray_out_of_bounds_errors() {
    let pixels = vec![0_i16; 6];
    let (h, data) = ImageBuilder::<i16>::new(vec![3_u64, 2], pixels)
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&h, &data).unwrap();
    w.finish().unwrap();
    let parsed = FitsFile::from_bytes(buf).unwrap();
    let Hdu::Image(img) = parsed.hdu(0).unwrap() else {
        panic!("not image");
    };
    assert!(img.read_subarray::<i16>(&[2, 0], &[2, 1]).is_err());
    assert!(img.read_subarray::<i16>(&[0, 0, 0], &[1, 1, 1]).is_err());
}

#[test]
fn bintable_row_range() {
    let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let mut bt = BinTableBuilder::new();
    bt.add_column("X", BinFieldKind::I32, 1, None, None)
        .unwrap();
    let mut row_bytes: Vec<u8> = Vec::new();
    for i in 0_i32..10 {
        row_bytes.extend_from_slice(&i.to_be_bytes());
    }
    let (h, data) = bt.build(10, row_bytes).unwrap();
    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary.0, &primary.1).unwrap();
    w.write_hdu(&h, &data).unwrap();
    w.finish().unwrap();
    let parsed = FitsFile::from_bytes(buf).unwrap();
    let Hdu::BinTable(t) = parsed.hdu(1).unwrap() else {
        panic!("not bintable");
    };
    let rows: Vec<i32> = t
        .row_range(3, 4)
        .unwrap()
        .map(|r| i32::from_be_bytes(r[..4].try_into().unwrap()))
        .collect();
    assert_eq!(rows, vec![3, 4, 5, 6]);
    assert!(t.row_range(8, 5).is_err());
}
