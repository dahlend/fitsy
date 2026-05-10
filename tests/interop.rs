//! Cross-check the optional `nalgebra` and `faer` integrations
//! against the existing `Wcs::pixel_to_world` / `world_to_pixel` and
//! `ImageData::as_slice` paths.
#![cfg(any(feature = "nalgebra", feature = "faer"))]

use fitsy::{Header, Wcs};

fn header(cards: &[&str]) -> Header {
    use std::fmt::Write as _;
    let mut s = String::new();
    for c in cards {
        write!(s, "{c:<80}").unwrap();
    }
    write!(s, "{:<80}", "END").unwrap();
    while !s.len().is_multiple_of(2880) {
        s.push(' ');
    }
    Header::parse(s.as_bytes(), 0).unwrap().0
}

fn tan_wcs() -> Wcs {
    let h = header(&[
        "NAXIS   =                    2",
        "NAXIS1  =                  200",
        "NAXIS2  =                  200",
        "WCSAXES =                    2",
        "CTYPE1  = 'RA---TAN'",
        "CTYPE2  = 'DEC--TAN'",
        "CRPIX1  =                100.0",
        "CRPIX2  =                100.0",
        "CRVAL1  =                210.0",
        "CRVAL2  =                 54.0",
        "CDELT1  =              -0.001",
        "CDELT2  =               0.001",
    ]);
    Wcs::from_header(&h, ' ').unwrap().unwrap()
}

#[cfg(feature = "nalgebra")]
mod na {
    use super::*;
    use fitsy::data::ImageData;
    use nalgebra::DMatrix;

    #[test]
    fn linear_matrix_round_trips() {
        let w = tan_wcs();
        let m = w.linear.matrix_na();
        assert_eq!(m.shape(), (2, 2));
        let row_major = w.linear.matrix_row_major();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(m[(i, j)], row_major[i * 2 + j]);
            }
        }
        let inv = w.linear.inverse_na();
        let id = &m * &inv;
        for i in 0..2 {
            for j in 0..2 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((id[(i, j)] - expect).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn image_to_dmatrix_2d() {
        // 3 fast (NAXIS1) x 2 slow (NAXIS2). Memory order: row-major
        // over (y, x) so [y0x0, y0x1, y0x2, y1x0, y1x1, y1x2].
        let img = ImageData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let m = img.to_dmatrix().unwrap();
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 2)], 6.0);
    }

    #[test]
    fn image_to_dmatrix_rejects_non_2d() {
        let img = ImageData::new(vec![0.0_f32; 8], vec![2, 2, 2]).unwrap();
        assert!(img.to_dmatrix().is_err());
    }

    #[test]
    fn image_from_dmatrix_round_trips() {
        let img = ImageData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let m = img.to_dmatrix().unwrap();
        let back = ImageData::from_dmatrix(&m).unwrap();
        assert_eq!(back.axes(), img.axes());
        assert_eq!(back.as_slice(), img.as_slice());
    }

    #[test]
    fn image_builder_from_dmatrix_writes_pixels_in_fits_order() {
        use fitsy::ImageBuilder;
        // 3 fast (NAXIS1) x 2 slow (NAXIS2): nrows=2, ncols=3.
        let m = DMatrix::<f32>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (header, data) = ImageBuilder::from_dmatrix(&m).unwrap().build().unwrap();
        assert_eq!(header.naxis().unwrap(), 2);
        assert_eq!(header.naxisn(1).unwrap(), 3);
        assert_eq!(header.naxisn(2).unwrap(), 2);
        // 6 pixels x 4 bytes; ImageBuilder::build returns the raw
        // pixel section without the 2880-byte block padding.
        assert_eq!(data.len(), 24);
        let pix: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(pix, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn batched_round_trip_matches_scalar() {
        let w = tan_wcs();
        let pts = [
            (50.0_f64, 50.0_f64),
            (100.0, 100.0),
            (150.0, 60.0),
            (200.0, 175.0),
        ];
        let mut pix = DMatrix::<f64>::zeros(2, pts.len());
        for (k, &(x, y)) in pts.iter().enumerate() {
            pix[(0, k)] = x;
            pix[(1, k)] = y;
        }
        let world = w.pixel_to_world_na(&pix).unwrap();
        for (k, &(x, y)) in pts.iter().enumerate() {
            let scalar = w.pixel_to_world(&[x, y]).unwrap();
            assert!((world[(0, k)] - scalar[0]).abs() < 1e-15);
            assert!((world[(1, k)] - scalar[1]).abs() < 1e-15);
        }
        let back = w.world_to_pixel_na(&world).unwrap();
        for (k, &(x, y)) in pts.iter().enumerate() {
            assert!((back[(0, k)] - x).abs() < 1e-7);
            assert!((back[(1, k)] - y).abs() < 1e-7);
        }
    }

    #[test]
    fn batched_rejects_wrong_shape() {
        let w = tan_wcs();
        let bad = DMatrix::<f64>::zeros(3, 4);
        assert!(w.pixel_to_world_na(&bad).is_err());
    }
}

#[cfg(feature = "faer")]
mod fa {
    use super::*;
    use faer::Mat;
    use fitsy::data::ImageData;

    #[test]
    fn linear_matrix_round_trips() {
        let w = tan_wcs();
        let m = w.linear.matrix_faer();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        let row_major = w.linear.matrix_row_major();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(m[(i, j)], row_major[i * 2 + j]);
            }
        }
    }

    #[test]
    fn image_to_faer_2d() {
        let img = ImageData::new(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let m = img.to_faer().unwrap();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 2)], 6.0);
    }

    #[test]
    fn image_from_faer_round_trips() {
        let img = ImageData::new(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let m = img.to_faer().unwrap();
        let back = ImageData::from_faer(&m).unwrap();
        assert_eq!(back.axes(), img.axes());
        assert_eq!(back.as_slice(), img.as_slice());
    }

    #[test]
    fn image_builder_from_faer_writes_pixels_in_fits_order() {
        use fitsy::ImageBuilder;
        // 3 fast (NAXIS1) x 2 slow (NAXIS2): nrows=2, ncols=3.
        let m = Mat::<f64>::from_fn(2, 3, |r, c| (r * 3 + c + 1) as f64);
        let (header, data) = ImageBuilder::from_faer(&m).unwrap().build().unwrap();
        assert_eq!(header.naxis().unwrap(), 2);
        assert_eq!(header.naxisn(1).unwrap(), 3);
        assert_eq!(header.naxisn(2).unwrap(), 2);
        // 6 pixels x 8 bytes; build() returns unpadded pixel bytes.
        assert_eq!(data.len(), 48);
        let pix: Vec<f64> = data
            .chunks_exact(8)
            .map(|c| {
                f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
            })
            .collect();
        assert_eq!(pix, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn batched_round_trip_matches_scalar() {
        let w = tan_wcs();
        let pts = [(50.0_f64, 50.0_f64), (100.0, 100.0), (150.0, 60.0)];
        let pix = Mat::<f64>::from_fn(
            2,
            pts.len(),
            |i, j| if i == 0 { pts[j].0 } else { pts[j].1 },
        );
        let world = w.pixel_to_world_faer(&pix).unwrap();
        for (k, &(x, y)) in pts.iter().enumerate() {
            let scalar = w.pixel_to_world(&[x, y]).unwrap();
            assert!((world[(0, k)] - scalar[0]).abs() < 1e-15);
            assert!((world[(1, k)] - scalar[1]).abs() < 1e-15);
        }
        let back = w.world_to_pixel_faer(&world).unwrap();
        for (k, &(x, y)) in pts.iter().enumerate() {
            assert!((back[(0, k)] - x).abs() < 1e-7);
            assert!((back[(1, k)] - y).abs() < 1e-7);
        }
    }
}
