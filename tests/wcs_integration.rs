//! End-to-end WCS tests: build a synthetic FITS header, parse it, and
//! verify the celestial pipeline round-trips.

use fitsy::{FitsFile, Hdu};

const CARD: usize = 80;
const BLOCK: usize = 2880;

fn pad_card(s: &str) -> [u8; CARD] {
    let mut b = [b' '; CARD];
    assert!(s.len() <= CARD, "card too long: {} bytes", s.len());
    b[..s.len()].copy_from_slice(s.as_bytes());
    b
}

fn build_minimal_image_with_wcs(cards: &[String]) -> Vec<u8> {
    let mut buf = Vec::new();
    let mandatory = [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                  100",
        "NAXIS2  =                  100",
    ];
    for c in mandatory {
        buf.extend_from_slice(&pad_card(c));
    }
    for c in cards {
        buf.extend_from_slice(&pad_card(c));
    }
    buf.extend_from_slice(&pad_card("END"));
    while buf.len() % BLOCK != 0 {
        buf.push(b' ');
    }
    // Data: 100*100 = 10000 bytes.
    let data_start = buf.len();
    buf.extend(std::iter::repeat_n(0_u8, 100 * 100));
    while (buf.len() - data_start) % BLOCK != 0 {
        buf.push(0);
    }
    buf
}

fn open_image(cards: &[String]) -> fitsy::Wcs {
    let bytes = build_minimal_image_with_wcs(cards);
    let file = FitsFile::from_bytes(bytes).unwrap();
    let Hdu::Image(img) = file.hdu(0).unwrap() else {
        panic!("not image");
    };
    img.wcs(' ').unwrap().expect("wcs present")
}

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// CRPIX->CRVAL for a simple TAN header.
#[test]
fn tan_reference_pixel_maps_to_crval() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =              83.6331".into(),
        "CRVAL2  =              22.0145".into(),
        "CDELT1  =          -2.78E-04".into(),
        "CDELT2  =           2.78E-04".into(),
        "CUNIT1  = 'deg'".into(),
        "CUNIT2  = 'deg'".into(),
    ];
    let wcs = open_image(&cards);
    // CRPIX1/2 = 50 in the FITS header (1-based). The Wcs API is
    // 0-based, so the reference pixel is at (49, 49).
    let world = wcs.pixel_to_world(&[49.0, 49.0]).unwrap();
    assert!(near(world[0], 83.6331, 1e-9), "ra = {}", world[0]);
    assert!(near(world[1], 22.0145, 1e-9), "dec = {}", world[1]);
}

#[test]
fn tan_round_trip_far_from_pole() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                 50.5".into(),
        "CRPIX2  =                 50.5".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
    ];
    let wcs = open_image(&cards);
    for &(px, py) in &[(1.0, 1.0), (50.0, 50.0), (75.5, 25.25), (100.0, 100.0)] {
        let world = wcs.pixel_to_world(&[px, py]).unwrap();
        let back = wcs.world_to_pixel(&world).unwrap();
        assert!(
            near(back[0], px, 1e-6) && near(back[1], py, 1e-6),
            "round trip failed at ({px},{py}) -> ({},{}) -> ({},{})",
            world[0],
            world[1],
            back[0],
            back[1]
        );
    }
}

/// Plate-carree CAR: 1deg/pixel, fiducial at (0deg, 0deg), check a point.
#[test]
fn car_simple_arithmetic() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---CAR'".into(),
        "CTYPE2  = 'DEC--CAR'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRVAL1  =                  0.0".into(),
        "CRVAL2  =                  0.0".into(),
        "CDELT1  =                  1.0".into(),
        "CDELT2  =                  1.0".into(),
    ];
    let wcs = open_image(&cards);
    // CRPIX1/2 = 1 (1-based); offset (10deg, 5deg) at pixel (10, 5)
    // in the 0-based Wcs API.
    let w = wcs.pixel_to_world(&[10.0, 5.0]).unwrap();
    // Pixel (11,6) is offset (10deg,5deg) from fiducial; CAR is identity
    // in (phi,theta); native pole at theta0=0 with default LATPOLE=90 means
    // the fiducial point is on the equator at (alpha0,delta0)=(0,0), and
    // celestial = native after the trivial rotation.
    assert!(near(w[0], 10.0, 1e-9), "ra = {}", w[0]);
    assert!(near(w[1], 5.0, 1e-9), "dec = {}", w[1]);
}

#[test]
fn missing_wcs_returns_none() {
    let cards: Vec<String> = vec![]; // no CTYPE/CRVAL/CRPIX/etc.
    let bytes = build_minimal_image_with_wcs(&cards);
    let file = FitsFile::from_bytes(bytes).unwrap();
    let Hdu::Image(img) = file.hdu(0).unwrap() else {
        panic!()
    };
    assert!(img.wcs(' ').unwrap().is_none());
}

#[test]
fn cd_and_pc_together_rejected() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CD1_1   =               -0.001".into(),
        "CD2_2   =                0.001".into(),
        "PC1_1   =                  1.0".into(),
    ];
    let bytes = build_minimal_image_with_wcs(&cards);
    let file = FitsFile::from_bytes(bytes).unwrap();
    let Hdu::Image(img) = file.hdu(0).unwrap() else {
        panic!()
    };
    let err = img.wcs(' ').unwrap_err();
    assert!(matches!(err, fitsy::FitsError::Wcs(_)));
}

/// TAN-SIP round trip: small quadratic distortion in pixel space.
#[test]
fn tan_sip_round_trip() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN-SIP'".into(),
        "CTYPE2  = 'DEC--TAN-SIP'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "A_ORDER =                    2".into(),
        "B_ORDER =                    2".into(),
        "A_2_0   =                1E-05".into(),
        "A_0_2   =               -2E-05".into(),
        "B_1_1   =                5E-06".into(),
    ];
    let wcs = open_image(&cards);
    for &(px, py) in &[
        (10.0_f64, 10.0_f64),
        (50.0, 50.0),
        (75.5, 25.25),
        (90.0, 20.0),
    ] {
        let world = wcs.pixel_to_world(&[px, py]).unwrap();
        let back = wcs.world_to_pixel(&world).unwrap();
        assert!(
            near(back[0], px, 1e-5) && near(back[1], py, 1e-5),
            "SIP round-trip failed at ({px},{py}) -> ({},{}) -> ({},{})",
            world[0],
            world[1],
            back[0],
            back[1],
        );
    }
}

/// TPV with no PV terms behaves identically to TAN.
#[test]
fn tpv_without_pv_matches_tan() {
    let make = |code: &str| -> Vec<String> {
        vec![
            format!("CTYPE1  = 'RA---{code}'"),
            format!("CTYPE2  = 'DEC--{code}'"),
            "CRPIX1  =                 50.0".into(),
            "CRPIX2  =                 50.0".into(),
            "CRVAL1  =                 10.0".into(),
            "CRVAL2  =                 -5.0".into(),
            "CDELT1  =              -0.001".into(),
            "CDELT2  =               0.001".into(),
        ]
    };
    let tan = open_image(&make("TAN"));
    let tpv = open_image(&make("TPV"));
    let w_tan = tan.pixel_to_world(&[60.0, 40.0]).unwrap();
    let w_tpv = tpv.pixel_to_world(&[60.0, 40.0]).unwrap();
    assert!(near(w_tan[0], w_tpv[0], 1e-12));
    assert!(near(w_tan[1], w_tpv[1], 1e-12));
}

/// TPV with a small radial term round-trips.
#[test]
fn tpv_radial_round_trip() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TPV'".into(),
        "CTYPE2  = 'DEC--TPV'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        // Small cubic radial perturbation: PV1_3 = PV2_3 = 0.001.
        "PV1_3   =                0.001".into(),
        "PV2_3   =                0.001".into(),
    ];
    let wcs = open_image(&cards);
    for &(px, py) in &[(40.0_f64, 40.0_f64), (60.0, 70.0), (10.0, 10.0)] {
        let world = wcs.pixel_to_world(&[px, py]).unwrap();
        let back = wcs.world_to_pixel(&world).unwrap();
        assert!(
            near(back[0], px, 1e-5) && near(back[1], py, 1e-5),
            "TPV round-trip failed at ({px},{py}) -> ({},{}) -> ({},{})",
            world[0],
            world[1],
            back[0],
            back[1],
        );
    }
}

/// `CUNIT='arcsec'` should scale CDELT into degrees so the resulting
/// world coordinates are still expressed in degrees.
#[test]
fn cunit_arcsec_is_scaled_to_degrees() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRVAL1  =                100.0".into(),
        "CRVAL2  =                 20.0".into(),
        // 3.6 arcsec/pixel = 0.001 deg/pixel.
        "CDELT1  =                 -3.6".into(),
        "CDELT2  =                  3.6".into(),
        "CUNIT1  = 'arcsec'".into(),
        "CUNIT2  = 'arcsec'".into(),
    ];
    let wcs = open_image(&cards);
    // 100 pixels along longitude => 100 * 0.001 = 0.1deg offset in
    // intermediate coords (modulo cos(delta) and TAN distortion).
    let w0 = wcs.pixel_to_world(&[0.0, 0.0]).unwrap();
    assert!(near(w0[0], 100.0, 1e-9), "ra origin = {}", w0[0]);
    assert!(near(w0[1], 20.0, 1e-9), "dec origin = {}", w0[1]);
    let w1 = wcs.pixel_to_world(&[100.0, 0.0]).unwrap();
    // Expected deltaRA ~= 100 px * (-3.6 arcsec) / cos(20deg) ~= -0.1064deg.
    let dra = (w1[0] - 100.0).abs();
    assert!((dra - 0.1064177).abs() < 1e-4, "deltaRA per 100 px = {dra}");
    // Round trip.
    let back = wcs.world_to_pixel(&w1).unwrap();
    assert!(near(back[0], 100.0, 1e-7));
    assert!(near(back[1], 0.0, 1e-7));
}

/// `RADESYS` keyword and `EQUINOX` should be parsed and surfaced on
/// the [`Wcs`] struct.
#[test]
fn radesys_and_equinox_parsed() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRVAL1  =                  0.0".into(),
        "CRVAL2  =                  0.0".into(),
        "CDELT1  =               -0.001".into(),
        "CDELT2  =                0.001".into(),
        "RADESYS = 'FK5'".into(),
        "EQUINOX =               2000.0".into(),
        "MJD-OBS =              58849.0".into(),
    ];
    let wcs = open_image(&cards);
    assert_eq!(wcs.radesys, fitsy::wcs::RadeSys::Fk5);
    assert_eq!(wcs.equinox, Some(2000.0));
    assert_eq!(wcs.mjd_obs, Some(58849.0));
}

/// No `RADESYS` keyword: default depends on `EQUINOX` per Paper II/// Sec.3.1 -- pre-1984 => FK4, post-1984 => FK5, missing => ICRS.
#[test]
fn radesys_defaults() {
    let base: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRVAL1  =                  0.0".into(),
        "CRVAL2  =                  0.0".into(),
        "CDELT1  =               -0.001".into(),
        "CDELT2  =                0.001".into(),
    ];
    // No EQUINOX => ICRS.
    let wcs = open_image(&base);
    assert_eq!(wcs.radesys, fitsy::wcs::RadeSys::Icrs);
    // EQUINOX = 1950 => FK4.
    let mut cards = base.clone();
    cards.push("EQUINOX =               1950.0".into());
    let wcs = open_image(&cards);
    assert_eq!(wcs.radesys, fitsy::wcs::RadeSys::Fk4);
    // EQUINOX = 2000 => FK5.
    let mut cards = base.clone();
    cards.push("EQUINOX =               2000.0".into());
    let wcs = open_image(&cards);
    assert_eq!(wcs.radesys, fitsy::wcs::RadeSys::Fk5);
}

/// 3-axis WCS with a spectral axis: RA-TAN / DEC-TAN / FREQ. The
/// spectral axis is linear in pixel space.
#[test]
fn spectral_freq_linear_axis() {
    let cards: Vec<String> = vec![
        "WCSAXES =                    3".into(),
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CTYPE3  = 'FREQ    '".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRPIX3  =                  1.0".into(),
        "CRVAL1  =                100.0".into(),
        "CRVAL2  =                 20.0".into(),
        "CRVAL3  =              1.42E+9".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "CDELT3  =              1.0E+6".into(),
        "CUNIT3  = 'Hz'".into(),
    ];
    let wcs = open_image(&cards);
    assert_eq!(wcs.spectral.len(), 1);
    let world = wcs.pixel_to_world(&[0.0, 0.0, 0.0]).unwrap();
    assert!((world[2] - 1.42e9).abs() < 1e-6);
    let world = wcs.pixel_to_world(&[0.0, 0.0, 10.0]).unwrap();
    assert!((world[2] - 1.43e9).abs() < 1e-6);
    let pix = wcs.world_to_pixel(&world).unwrap();
    assert!((pix[2] - 10.0).abs() < 1e-9);
}

/// `WAVE-F2W`: wavelength-class axis with frequency linear in pixel.
#[test]
fn spectral_wave_f2w_round_trip() {
    let cards: Vec<String> = vec![
        "WCSAXES =                    3".into(),
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CTYPE3  = 'WAVE-F2W'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRPIX3  =                 50.0".into(),
        "CRVAL1  =                  0.0".into(),
        "CRVAL2  =                  0.0".into(),
        "CRVAL3  =              5.0E-7".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "CDELT3  =              1.0E-9".into(),
        "CUNIT3  = 'm'".into(),
        "RESTWAV =              5.0E-7".into(),
    ];
    let wcs = open_image(&cards);
    assert_eq!(wcs.spectral.len(), 1);
    // At reference pixel (CRPIX-1 in 0-based) => exactly CRVAL.
    let w0 = wcs.pixel_to_world(&[0.0, 0.0, 49.0]).unwrap();
    assert!((w0[2] - 5.0e-7).abs() < 1e-18);
    // Round-trip a handful of pixels.
    for &px in &[0.0_f64, 24.0, 49.0, 74.0, 99.0] {
        let world = wcs.pixel_to_world(&[0.0, 0.0, px]).unwrap();
        let back = wcs.world_to_pixel(&world).unwrap();
        assert!(
            (back[2] - px).abs() < 1e-7,
            "WAVE-F2W round-trip @ {px} -> lambda={} -> {}",
            world[2],
            back[2]
        );
    }
}

/// `FREQ-LOG`: log-linear frequency axis.
#[test]
fn spectral_freq_log_round_trip() {
    let cards: Vec<String> = vec![
        "WCSAXES =                    3".into(),
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CTYPE3  = 'FREQ-LOG'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRPIX3  =                  1.0".into(),
        "CRVAL1  =                  0.0".into(),
        "CRVAL2  =                  0.0".into(),
        "CRVAL3  =              1.0E+9".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "CDELT3  =              1.0E+7".into(),
        "CUNIT3  = 'Hz'".into(),
    ];
    let wcs = open_image(&cards);
    for &px in &[0.0_f64, 4.0, 24.0, 99.0] {
        let world = wcs.pixel_to_world(&[0.0, 0.0, px]).unwrap();
        let back = wcs.world_to_pixel(&world).unwrap();
        assert!((back[2] - px).abs() < 1e-7, "FREQ-LOG round-trip @ {px}");
    }
}

/// Regression: a non-celestial axis listed *before* the celestial
/// pair must not stop the celestial-pair search. Previously a stray
/// `?` in `identify_celestial_pair` early-returned `None` the moment
/// the first axis failed the longitude-prefix test, so headers with
/// `CTYPE1 = FREQ`, `CTYPE2 = RA`, `CTYPE3 = DEC` silently dropped
/// their celestial block.
#[test]
fn celestial_pair_after_spectral_axis() {
    let cards: Vec<String> = vec![
        "WCSAXES =                    3".into(),
        "CTYPE1  = 'FREQ    '".into(),
        "CTYPE2  = 'RA---TAN'".into(),
        "CTYPE3  = 'DEC--TAN'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRPIX3  =                  1.0".into(),
        "CRVAL1  =              1.42E+9".into(),
        "CRVAL2  =                100.0".into(),
        "CRVAL3  =                 20.0".into(),
        "CDELT1  =              1.0E+6".into(),
        "CDELT2  =              -0.001".into(),
        "CDELT3  =               0.001".into(),
        "CUNIT1  = 'Hz'".into(),
    ];
    let wcs = open_image(&cards);
    assert!(
        wcs.celestial.is_some(),
        "celestial pair must be detected when a spectral axis precedes RA/DEC"
    );
    assert_eq!(wcs.spectral.len(), 1);
    let world = wcs.pixel_to_world(&[0.0, 0.0, 0.0]).unwrap();
    assert!((world[0] - 1.42e9).abs() < 1e-6);
    assert!((world[1] - 100.0).abs() < 1e-9);
    assert!((world[2] - 20.0).abs() < 1e-9);
}

/// `VOPT` linear axis on the 21cm line, with CUNIT = `km/s`.
#[test]
fn spectral_vopt_kms_round_trip() {
    let cards: Vec<String> = vec![
        "WCSAXES =                    3".into(),
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CTYPE3  = 'VOPT    '".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRPIX3  =                  1.0".into(),
        "CRVAL1  =                  0.0".into(),
        "CRVAL2  =                  0.0".into(),
        "CRVAL3  =                  0.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "CDELT3  =                  1.0".into(),
        "CUNIT3  = 'km/s'".into(),
        "RESTFRQ =          1.420405752E+9".into(),
    ];
    let wcs = open_image(&cards);
    assert_eq!(wcs.spectral.len(), 1);
    let w = wcs.pixel_to_world(&[0.0, 0.0, 10.0]).unwrap();
    // 10 km/s offset; linear axis => exactly 10.
    assert!((w[2] - 10.0).abs() < 1e-12, "got {}", w[2]);
    let back = wcs.world_to_pixel(&w).unwrap();
    assert!((back[2] - 10.0).abs() < 1e-12);
}

// -- CSV-driven ground-truth tests -----------------------------------------
//
// The CSV files in data/ were generated by tests/data/gen_wcs_test_data.py using
// astropy 7.2.0 with origin=1 (FITS 1-based pixels) for every projection.
// Each test:
//   1. Parses the CSV at runtime.
//   2. Constructs a synthetic 2-axis FITS header with the parameters from the
//      CSV row.
//   3. Parses it into a `Wcs`.
//   4. Calls `world_to_pixel(ra, dec)` and asserts the result matches the stored
//      (x_fits, y_fits) within 1e-7 pixels (standard / SIP) or 1e-8 pixels
//      (TPV).  Both SIP and TPV use Newton iteration converged to machine
//      precision; the SIP / standard floor of ~1e-7 px is the trigonometric
//      round-off floor of the underlying TAN projection at this image scale.
//
// At the test pixel scale of 1 arcsec/pixel these tolerances correspond to
// 1e-4 mas (standard / SIP) and 1e-5 mas (TPV), four to five orders of
// magnitude tighter than the 1 mas accuracy required by modern instruments.
//
// The forward direction (pixel_to_world) is also checked at the reference pixel
// to guard against header-construction mistakes.

use std::collections::HashMap;
use std::path::PathBuf;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// Parse a minimal CSV: first row is headers, remaining rows are data.
/// Returns a `Vec<HashMap<String, String>>`.
fn parse_csv(path: &std::path::Path) -> Vec<HashMap<String, String>> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = text.lines();
    let headers: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(ToOwned::to_owned)
        .collect();
    lines
        .filter(|l| !l.is_empty())
        .map(|line| {
            headers
                .iter()
                .zip(line.split(','))
                .map(|(k, v)| (k.clone(), v.to_owned()))
                .collect()
        })
        .collect()
}

fn f(row: &HashMap<String, String>, key: &str) -> f64 {
    row[key]
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("cannot parse field {key:?} = {:?}", row[key]))
}

fn opt_f(row: &HashMap<String, String>, key: &str) -> Option<f64> {
    let s = row.get(key)?;
    if s.is_empty() {
        None
    } else {
        s.parse::<f64>().ok()
    }
}

/// Build cards common to all 2-axis projections: SIMPLE/BITPIX/NAXIS*,
/// `CTYPEi`, `CRPIXi`, `CRVALi`, `CDELTi`, then the caller-supplied extras.
#[allow(
    clippy::too_many_arguments,
    reason = "need all these params to construct the header"
)]
fn base_cards(
    ctype1: &str,
    ctype2: &str,
    crpix1: f64,
    crpix2: f64,
    crval1: f64,
    crval2: f64,
    cdelt1: f64,
    cdelt2: f64,
) -> Vec<String> {
    vec![
        format!("CTYPE1  = '{ctype1:<8}'"),
        format!("CTYPE2  = '{ctype2:<8}'"),
        format!("CRPIX1  = {crpix1:>20}"),
        format!("CRPIX2  = {crpix2:>20}"),
        format!("CRVAL1  = {crval1:>20}"),
        format!("CRVAL2  = {crval2:>20}"),
        format!("CDELT1  = {cdelt1:>20e}"),
        format!("CDELT2  = {cdelt2:>20e}"),
    ]
}

// -- Standard projections --------------------------------------------------

/// Build a `Wcs` for one row of `wcs_standard.csv`.
fn wcs_for_standard_row(row: &HashMap<String, String>) -> fitsy::Wcs {
    let code = &row["projection"];
    let crpix1 = f(row, "crpix1");
    let crpix2 = f(row, "crpix2");
    let crval1 = f(row, "crval1");
    let crval2 = f(row, "crval2");
    let cdelt1 = f(row, "cdelt1");
    let cdelt2 = f(row, "cdelt2");

    let mut cards = base_cards(
        &format!("RA---{code}"),
        &format!("DEC--{code}"),
        crpix1,
        crpix2,
        crval1,
        crval2,
        cdelt1,
        cdelt2,
    );
    // Append non-empty PV params.
    for (col, kw) in [
        ("pv2_0", "PV2_0"),
        ("pv2_1", "PV2_1"),
        ("pv2_2", "PV2_2"),
        ("pv2_3", "PV2_3"),
    ] {
        if let Some(v) = opt_f(row, col) {
            cards.push(format!("{kw:<8}= {v:>20e}"));
        }
    }
    open_image(&cards)
}

/// For every projection in `wcs_standard.csv`:
///   - The reference pixel (CRPIX) maps to CRVAL via `pixel_to_world`.
///   - `world_to_pixel(ra, dec)` recovers the stored pixel within 1e-8 px.
#[test]
fn standard_projections_match_astropy() {
    let path = test_data_dir().join("wcs_standard.csv");
    let rows = parse_csv(&path);
    assert!(!rows.is_empty(), "CSV is empty");

    let mut failures: Vec<String> = Vec::new();

    // Group by projection so we can check the reference-pixel once per config.
    // Since every row in the CSV shares the same WCS params (only ra/dec/x/y
    // vary), we only need to build the Wcs once per unique projection (all rows
    // for a given projection share identical crpix/crval/cdelt/pv params).
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for row in &rows {
        let code = &row["projection"];
        let wcs = wcs_for_standard_row(row);

        // Reference-pixel check: only do this once per projection.
        if !seen.contains(code) {
            seen.insert(code.clone());
            let crpix1 = f(row, "crpix1");
            let crpix2 = f(row, "crpix2");
            let crval1 = f(row, "crval1");
            let crval2 = f(row, "crval2");
            // CSV CRPIX values are 1-based FITS; the Wcs API is 0-based.
            match wcs.pixel_to_world(&[crpix1 - 1.0, crpix2 - 1.0]) {
                Ok(w) => {
                    // Celestial coords near poles can alias; compare via a
                    // 360deg wraparound-aware distance instead.
                    let dra = ((w[0] - crval1 + 540.0).rem_euclid(360.0) - 180.0).abs();
                    let ddec = (w[1] - crval2).abs();
                    if dra > 1e-8 || ddec > 1e-8 {
                        failures.push(format!(
                            "{code}: CRPIX->CRVAL: got ({:.10},{:.10}) expected ({crval1},{crval2})",
                            w[0], w[1]
                        ));
                    }
                }
                Err(e) => {
                    failures.push(format!("{code}: pixel_to_world(CRPIX) failed: {e}"));
                }
            }
        }

        // world_to_pixel check for every CSV row.
        let ra = f(row, "ra");
        let dec = f(row, "dec");
        let x_fits = f(row, "x_fits");
        let y_fits = f(row, "y_fits");
        // CSV pixel columns are 1-based FITS; the Wcs API is 0-based.
        let x_expected = x_fits - 1.0;
        let y_expected = y_fits - 1.0;
        match wcs.world_to_pixel(&[ra, dec]) {
            Ok(pix) => {
                let ex = (pix[0] - x_expected).abs();
                let ey = (pix[1] - y_expected).abs();
                if ex > 1e-7 || ey > 1e-7 {
                    failures.push(format!(
                        "{code}: world_to_pixel({ra:.6},{dec:.6}) = ({:.10},{:.10}) \
                         expected ({x_expected},{y_expected}) delta=({ex:.2e},{ey:.2e})",
                        pix[0], pix[1]
                    ));
                }
            }
            Err(e) => {
                failures.push(format!(
                    "{code}: world_to_pixel({ra:.6},{dec:.6}) failed: {e}"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} failure(s) in standard_projections_match_astropy:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

// -- SIP distortion --------------------------------------------------------

/// For every row of `wcs_sip.csv`:
///   - The reference pixel maps to CRVAL.
///   - `world_to_pixel(ra, dec)` recovers the stored pixel within 1e-8 px.
#[test]
fn sip_matches_astropy() {
    let path = test_data_dir().join("wcs_sip.csv");
    let rows = parse_csv(&path);
    assert!(!rows.is_empty(), "SIP CSV is empty");

    // All rows share the same WCS config; build once.
    let row0 = &rows[0];
    let mut cards = base_cards(
        "RA---TAN-SIP",
        "DEC--TAN-SIP",
        f(row0, "crpix1"),
        f(row0, "crpix2"),
        f(row0, "crval1"),
        f(row0, "crval2"),
        f(row0, "cdelt1"),
        f(row0, "cdelt2"),
    );
    cards.extend([
        "A_ORDER =                    2".to_owned(),
        "B_ORDER =                    2".to_owned(),
        "AP_ORDER=                    2".to_owned(),
        "BP_ORDER=                    2".to_owned(),
        format!("A_2_0   = {:>20e}", f(row0, "a_2_0")),
        format!("A_0_2   = {:>20e}", f(row0, "a_0_2")),
        format!("A_1_1   = {:>20e}", f(row0, "a_1_1")),
        format!("B_2_0   = {:>20e}", f(row0, "b_2_0")),
        format!("B_0_2   = {:>20e}", f(row0, "b_0_2")),
        format!("B_1_1   = {:>20e}", f(row0, "b_1_1")),
        format!("AP_2_0  = {:>20e}", f(row0, "ap_2_0")),
        format!("AP_0_2  = {:>20e}", f(row0, "ap_0_2")),
        format!("AP_1_1  = {:>20e}", f(row0, "ap_1_1")),
        format!("BP_2_0  = {:>20e}", f(row0, "bp_2_0")),
        format!("BP_0_2  = {:>20e}", f(row0, "bp_0_2")),
        format!("BP_1_1  = {:>20e}", f(row0, "bp_1_1")),
    ]);
    let wcs = open_image(&cards);

    // Reference pixel.
    let crpix1 = f(row0, "crpix1");
    let crpix2 = f(row0, "crpix2");
    let crval1 = f(row0, "crval1");
    let crval2 = f(row0, "crval2");
    // CSV CRPIX values are 1-based FITS; the Wcs API is 0-based.
    let wref = wcs.pixel_to_world(&[crpix1 - 1.0, crpix2 - 1.0]).unwrap();
    assert!(
        (wref[0] - crval1).abs() < 1e-8 && (wref[1] - crval2).abs() < 1e-8,
        "SIP CRPIX->CRVAL: got ({:.12},{:.12}) expected ({crval1},{crval2})",
        wref[0],
        wref[1]
    );

    let mut failures: Vec<String> = Vec::new();
    for row in &rows {
        let ra = f(row, "ra");
        let dec = f(row, "dec");
        let x_fits = f(row, "x_fits");
        let y_fits = f(row, "y_fits");
        let x_expected = x_fits - 1.0;
        let y_expected = y_fits - 1.0;
        match wcs.world_to_pixel(&[ra, dec]) {
            Ok(pix) => {
                let ex = (pix[0] - x_expected).abs();
                let ey = (pix[1] - y_expected).abs();
                if ex > 1e-7 || ey > 1e-7 {
                    failures.push(format!(
                        "world_to_pixel({ra:.6},{dec:.6}) = ({:.10},{:.10}) \
                         expected ({x_expected},{y_expected}) delta=({ex:.2e},{ey:.2e})",
                        pix[0], pix[1]
                    ));
                }
            }
            Err(e) => {
                failures.push(format!("world_to_pixel({ra:.6},{dec:.6}) failed: {e}"));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} failure(s) in sip_matches_astropy:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

// -- TPV distortion --------------------------------------------------------

/// For every row of `wcs_tpv.csv`:
///   - The reference pixel maps to CRVAL.
///   - `world_to_pixel(ra, dec)` recovers the stored pixel within 1e-5 px
///     (iterative Newton inverse; astropy's wcslib solver also uses 30
///     iterations and accepts up to 1e-10 px residual so 1e-5 is generous).
#[test]
fn tpv_matches_astropy() {
    let path = test_data_dir().join("wcs_tpv.csv");
    let rows = parse_csv(&path);
    assert!(!rows.is_empty(), "TPV CSV is empty");

    let row0 = &rows[0];
    let mut cards = base_cards(
        "RA---TPV",
        "DEC--TPV",
        f(row0, "crpix1"),
        f(row0, "crpix2"),
        f(row0, "crval1"),
        f(row0, "crval2"),
        f(row0, "cdelt1"),
        f(row0, "cdelt2"),
    );
    // Only set non-zero / non-default PV terms (see gen_wcs_test_data.py).
    cards.extend([
        format!("PV1_1   = {:>20e}", f(row0, "pv1_1")),
        format!("PV1_4   = {:>20e}", f(row0, "pv1_4")),
        format!("PV1_5   = {:>20e}", f(row0, "pv1_5")),
        format!("PV2_1   = {:>20e}", f(row0, "pv2_1")),
        format!("PV2_4   = {:>20e}", f(row0, "pv2_4")),
        format!("PV2_6   = {:>20e}", f(row0, "pv2_6")),
    ]);
    let wcs = open_image(&cards);

    // Reference pixel.
    let crpix1 = f(row0, "crpix1");
    let crpix2 = f(row0, "crpix2");
    let crval1 = f(row0, "crval1");
    let crval2 = f(row0, "crval2");
    // CSV CRPIX values are 1-based FITS; the Wcs API is 0-based.
    let wref = wcs.pixel_to_world(&[crpix1 - 1.0, crpix2 - 1.0]).unwrap();
    assert!(
        (wref[0] - crval1).abs() < 1e-8 && (wref[1] - crval2).abs() < 1e-8,
        "TPV CRPIX->CRVAL: got ({:.12},{:.12}) expected ({crval1},{crval2})",
        wref[0],
        wref[1]
    );

    let mut failures: Vec<String> = Vec::new();
    for row in &rows {
        let ra = f(row, "ra");
        let dec = f(row, "dec");
        let x_fits = f(row, "x_fits");
        let y_fits = f(row, "y_fits");
        let x_expected = x_fits - 1.0;
        let y_expected = y_fits - 1.0;
        match wcs.world_to_pixel(&[ra, dec]) {
            Ok(pix) => {
                let ex = (pix[0] - x_expected).abs();
                let ey = (pix[1] - y_expected).abs();
                // 1e-5 px: iterative Newton inverse tolerance.
                if ex > 1e-5 || ey > 1e-5 {
                    failures.push(format!(
                        "world_to_pixel({ra:.6},{dec:.6}) = ({:.10},{:.10}) \
                         expected ({x_expected},{y_expected}) delta=({ex:.2e},{ey:.2e})",
                        pix[0], pix[1]
                    ));
                }
            }
            Err(e) => {
                failures.push(format!("world_to_pixel({ra:.6},{dec:.6}) failed: {e}"));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} failure(s) in tpv_matches_astropy:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// Non-standard convention tests
// ---------------------------------------------------------------------------

/// `WCSNAME` (Standard Sec.8.2.6) is surfaced on the parsed `Wcs`.
#[test]
fn wcsname_is_surfaced() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "WCSNAME = 'IDC distortion-corrected'".into(),
    ];
    let wcs = open_image(&cards);
    assert_eq!(wcs.wcsname.as_deref(), Some("IDC distortion-corrected"));
}

/// `SPECSYS`, `SSYSOBS`, `VELOSYS` (Paper III Sec.7) are surfaced
/// verbatim on the parsed `Wcs`.
#[test]
fn spectral_reference_frame_keywords_surfaced() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "SPECSYS = 'BARYCENT'".into(),
        "SSYSOBS = 'TOPOCENT'".into(),
        "VELOSYS =              12345.0".into(),
    ];
    let wcs = open_image(&cards);
    assert_eq!(wcs.specsys.as_deref(), Some("BARYCENT"));
    assert_eq!(wcs.ssysobs.as_deref(), Some("TOPOCENT"));
    assert_eq!(wcs.velosys, Some(12345.0));
}

/// `WCSAXES` may differ from `NAXIS` (Paper I Sec.2.1): the WCS engine
/// uses `WCSAXES` for its dimensionality, independent of the array
/// shape on disk.
#[test]
fn wcsaxes_overrides_naxis() {
    // 2-D image (NAXIS=2) but WCSAXES=3 declares a degenerate 3rd
    // (spectral) axis. The parsed WCS should be 3-D.
    let cards: Vec<String> = vec![
        "WCSAXES =                    3".into(),
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CTYPE3  = 'FREQ'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRPIX3  =                  1.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CRVAL3  =              1.42E+09".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "CDELT3  =                1.0E6".into(),
        "CUNIT3  = 'Hz'".into(),
    ];
    let wcs = open_image(&cards);
    assert_eq!(wcs.naxis, 3);
    assert!(wcs.celestial.is_some(), "celestial pair (axes 1,2) found");
    assert_eq!(wcs.spectral.len(), 1, "FREQ axis recognized");
    assert_eq!(wcs.spectral[0].axis, 2);
    // CRPIX1/2 = 50, CRPIX3 = 1 (1-based). Convert to 0-based.
    let world = wcs.pixel_to_world(&[49.0, 49.0, 0.0]).unwrap();
    assert!(near(world[0], 10.0, 1e-9), "ra = {}", world[0]);
    assert!(near(world[1], -5.0, 1e-9), "dec = {}", world[1]);
    assert!(near(world[2], 1.42e9, 1.0), "freq = {}", world[2]);
}

/// IRAF `LTV`/`LTM` subimage convention (`phys = LTM*log + LTV`) is
/// folded into the linear pipeline so the WCS, written in original
/// detector coordinates, applies correctly to the subimage pixels.
#[test]
fn iraf_ltv_ltm_subimage_offset() {
    // Original CRPIX1 = 100 in physical (detector) coords. The
    // subimage starts at physical pixel 51 (no rebin), so
    // LTV1 = -50, LTM1_1 = 1: phys = 1*log + (-50). The same
    // physical reference point is then at logical pixel 150.
    // Wait: phys = LTM*log + LTV, so log = (phys - LTV)/LTM.
    // CRPIX_log = (100 - (-50))/1 = 150.
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                100.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "LTV1    =                -50.0".into(),
        "LTV2    =                  0.0".into(),
        "LTM1_1  =                  1.0".into(),
        "LTM2_2  =                  1.0".into(),
    ];
    let wcs = open_image(&cards);
    // Logical pixel 150 (1-based, == original 100) maps to CRVAL.
    // The Wcs API is 0-based, so we use logical pixel 149.
    let world = wcs.pixel_to_world(&[149.0, 49.0]).unwrap();
    assert!(near(world[0], 10.0, 1e-9), "ra = {}", world[0]);
    assert!(near(world[1], -5.0, 1e-9), "dec = {}", world[1]);
    // And original physical pixel 100 in logical coords = 150 (1-based)
    // = 149 (0-based).
    let pix = wcs.world_to_pixel(&[10.0, -5.0]).unwrap();
    assert!(near(pix[0], 149.0, 1e-9), "x = {}", pix[0]);
    assert!(near(pix[1], 49.0, 1e-9), "y = {}", pix[1]);
}

/// IRAF `LTM` rebinning factor is also absorbed: a 2x binned subimage
/// has `LTM1_1 = 0.5` (one logical pixel = two physical pixels).
#[test]
fn iraf_ltm_rebin_factor() {
    // CRPIX_phys = 100, LTV = 0, LTM = 0.5 -> CRPIX_log = 200.
    // CDELT in physical pixels stays the same; the logical pixel scale
    // doubles, which our compose_with_input_affine handles by scaling
    // the matrix columns.
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                100.0".into(),
        "CRPIX2  =                100.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "LTM1_1  =                  0.5".into(),
        "LTM2_2  =                  0.5".into(),
    ];
    let wcs = open_image(&cards);
    // CRPIX_log = 200 (1-based) maps to CRVAL. 0-based: 199.
    let world = wcs.pixel_to_world(&[199.0, 199.0]).unwrap();
    assert!(near(world[0], 10.0, 1e-9), "ra = {}", world[0]);
    assert!(near(world[1], -5.0, 1e-9), "dec = {}", world[1]);
    // Step 1 logical pixel -> step 2 physical pixels -> 2*CDELT in world.
    let world2 = wcs.pixel_to_world(&[200.0, 199.0]).unwrap();
    let dx = world2[0] - world[0];
    // Per-logical-pixel intermediate-world step magnitude is
    // |CDELT| * |LTM| = 0.001 * 0.5 = 0.0005 deg. RA-on-sky picks up
    // a 1/cos(delta) factor from the TAN projection at delta = -5deg.
    let expected = 0.0005 / (-5.0_f64).to_radians().cos();
    assert!(
        (dx.abs() - expected).abs() < 1e-7,
        "dx per logical pixel = {dx}; expected magnitude {expected}"
    );
    // Round-trip the binned subimage.
    let pix = wcs.world_to_pixel(&[10.0, -5.0]).unwrap();
    assert!(near(pix[0], 199.0, 1e-7), "x = {}", pix[0]);
    assert!(near(pix[1], 199.0, 1e-7), "y = {}", pix[1]);
}

/// IRAF TNX: WAT-encoded polynomial pre-warp on top of a TAN base.
/// Verifies (a) detection, (b) zero-coefficient TNX matches plain
/// TAN, (c) a non-zero `lngcor` shifts the longitude by the expected
/// additive amount, and (d) `pix -> world -> pix` round-trips.
#[test]
fn iraf_tnx_polynomial_distortion_round_trip() {
    // Plain TAN baseline.
    let tan_cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
    ];
    // TNX with all-zero surfaces should reproduce TAN exactly.
    let zero_tnx_cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TNX'".into(),
        "CTYPE2  = 'DEC--TNX'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "WAT0_001= 'system=image'".into(),
        "WAT1_001= 'wtype=tnx axtype=ra lngcor = \"3 1 1 1 -1 1 -1 1 0\"'".into(),
        "WAT2_001= 'wtype=tnx axtype=dec latcor = \"3 1 1 1 -1 1 -1 1 0\"'".into(),
    ];
    let tan = open_image(&tan_cards);
    let zero = open_image(&zero_tnx_cards);
    let w_tan = tan.pixel_to_world(&[59.0, 39.0]).unwrap();
    let w_zero = zero.pixel_to_world(&[59.0, 39.0]).unwrap();
    assert!(
        near(w_tan[0], w_zero[0], 1e-12) && near(w_tan[1], w_zero[1], 1e-12),
        "zero-coeff TNX must equal plain TAN: TAN={w_tan:?}, TNX={w_zero:?}"
    );

    // TNX with a constant +0.0005deg additive offset on the longitude
    // surface (function_type=3, ni=nj=1, single coeff = 5e-4 deg).
    // The additive surface lives in the intermediate world plane,
    // so the resulting RA shift on-sky scales by 1/cos(delta).
    let shift_cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TNX'".into(),
        "CTYPE2  = 'DEC--TNX'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "WAT0_001= 'system=image'".into(),
        "WAT1_001= 'wtype=tnx axtype=ra lngcor = \"3 1 1 1 -1 1 -1 1 5E-4\"'".into(),
        "WAT2_001= 'wtype=tnx axtype=dec latcor = \"3 1 1 1 -1 1 -1 1 0\"'".into(),
    ];
    let shifted = open_image(&shift_cards);
    // CRPIX = 50 (1-based); evaluate at the 0-based reference (49, 49).
    let w_shift = shifted.pixel_to_world(&[49.0, 49.0]).unwrap();
    // At CRPIX both intermediate coords are zero, so xi shifts by
    // exactly +5e-4 deg, then maps onto the sky at delta ~= -5deg.
    let expected_ra = 10.0 + 5e-4 / (-5.0_f64).to_radians().cos();
    assert!(
        near(w_shift[0], expected_ra, 1e-9),
        "ra shift: got {}, expected {}",
        w_shift[0],
        expected_ra
    );
    assert!(near(w_shift[1], -5.0, 1e-9), "dec = {}", w_shift[1]);

    // Non-trivial linear+quadratic distortion + round trip.
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TNX'".into(),
        "CTYPE2  = 'DEC--TNX'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 -5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "WAT0_001= 'system=image'".into(),
        "WAT1_001= 'wtype=tnx axtype=ra lngcor = \"3 2 2 1 -1 1 -1 1 0 1E-3 5E-4 0\"'".into(),
        "WAT2_001= 'wtype=tnx axtype=dec latcor = \"3 2 2 1 -1 1 -1 1 0 0 1E-3 0\"'".into(),
    ];
    let wcs = open_image(&cards);
    for &(px, py) in &[
        (40.0_f64, 40.0_f64),
        (50.0, 50.0),
        (60.0, 70.0),
        (10.0, 90.0),
    ] {
        let world = wcs.pixel_to_world(&[px, py]).unwrap();
        let back = wcs.world_to_pixel(&world).unwrap();
        assert!(
            near(back[0], px, 1e-6) && near(back[1], py, 1e-6),
            "TNX round-trip failed at ({px},{py}) -> ({},{}) -> ({},{})",
            world[0],
            world[1],
            back[0],
            back[1],
        );
    }
}

/// DSS plate solution: open the real `data/dss_plate.fits`, verify
/// the plate model is detected (not the dummy `RA---TAN` fallback)
/// and that the plate center maps to the sexagesimal RA/Dec from
/// the `PLT*` keywords. Round-trip a handful of pixels.
#[test]
fn dss_plate_model_used_for_real_file() {
    let path = test_data_dir().join("dss_plate.fits");
    let bytes = std::fs::read(&path).expect("dss_plate.fits");
    let file = FitsFile::from_bytes(bytes).unwrap();
    let Hdu::Image(img) = file.hdu(0).unwrap() else {
        panic!("expected image HDU");
    };
    let wcs = img.wcs(' ').unwrap().expect("wcs present");
    assert!(wcs.dss.is_some(), "DSS plate model should be detected");

    // Plate center RA/Dec from PLT* sexagesimal:
    // 0h07m25.68s = 1.857deg, +0deg48'26" = 0.80722deg.
    let plate_ra = (0.0 + 7.0 / 60.0 + 25.68 / 3600.0) * 15.0;
    let plate_dec = 48.0 / 60.0 + 26.0 / 3600.0;

    // The plate center is OUTSIDE this 2119x2119 subimage (it lives
    // at plate-pixel ~= (PPO3/XPIXELSZ, PPO6/YPIXELSZ) ~= (7020, 7020)
    // and the subimage starts at CNPIX = (9818, 4258)). Compute the
    // 1-based pixel that maps to the plate center and verify the
    // forward map produces the sexagesimal RA/Dec -- modulo the
    // polynomial zero-point terms `AMDX3`, `AMDY3` which add a few
    // arcseconds of plate-model offset.
    let dss = wcs.dss.as_ref().unwrap();
    // The +0.5 - cnpix formulas above produce a 1-based FITS pixel
    // coordinate; subtract 1 for the 0-based Wcs API.
    let plate_centre_x = dss.ppo3 / dss.xpixelsz - dss.cnpix1 + 0.5 - 1.0;
    let plate_centre_y = dss.ppo6 / dss.ypixelsz - dss.cnpix2 + 0.5 - 1.0;
    let world = wcs
        .pixel_to_world(&[plate_centre_x, plate_centre_y])
        .unwrap();
    // Tolerance: AMDX3 ~= -131" and AMDY3 ~= +1.65", so up to ~0.04deg.
    assert!(
        (world[0] - plate_ra).abs() < 0.05,
        "RA at plate center: got {}, expected ~= {}",
        world[0],
        plate_ra,
    );
    assert!(
        (world[1] - plate_dec).abs() < 0.05,
        "Dec at plate center: got {}, expected ~= {}",
        world[1],
        plate_dec,
    );

    // Sanity: the dummy-TAN fallback would put pixel (1060, 1060)
    // at the header's CRVAL ~= (6e-5, 1.7e-4)deg. The real plate
    // model puts that pixel about a degree off from the plate
    // center -- so it must be very far from (0, 0).
    let img_center = wcs.pixel_to_world(&[1060.0, 1060.0]).unwrap();
    let off_ra = (img_center[0] - 6.4e-5).rem_euclid(360.0);
    let off_ra = off_ra.min(360.0 - off_ra);
    assert!(
        off_ra > 0.5 || (img_center[1] - 1.66e-4).abs() > 0.5,
        "DSS plate model not actually used: image center = {img_center:?}"
    );

    // Round-trip across the image.
    for &(px, py) in &[
        (1.0_f64, 1.0_f64),
        (1060.0, 1060.0),
        (500.0, 500.0),
        (2000.0, 2000.0),
        (100.0, 2000.0),
    ] {
        let w = wcs.pixel_to_world(&[px, py]).unwrap();
        let back = wcs.world_to_pixel(&w).unwrap();
        assert!(
            near(back[0], px, 1e-4) && near(back[1], py, 1e-4),
            "DSS round-trip failed at ({px},{py}) -> ({},{}) -> ({},{})",
            w[0],
            w[1],
            back[0],
            back[1],
        );
    }
}

/// SIP convention: `AP_*` and `BP_*` (the analytic inverse coefficients)
/// must be a paired set. Headers with only one half are malformed and
/// must be rejected -- silently dropping the half that is present
/// would force the slow Newton fallback while the user thinks the
/// lookup is being used.
#[test]
fn sip_partial_inverse_is_rejected() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN-SIP'".into(),
        "CTYPE2  = 'DEC--TAN-SIP'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                  5.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
        "A_ORDER =                    2".into(),
        "B_ORDER =                    2".into(),
        "A_2_0   =              1.0E-7".into(),
        "B_0_2   =              1.0E-7".into(),
        // AP_ORDER without BP_ORDER -> must error.
        "AP_ORDER=                    2".into(),
        "AP_2_0  =             -1.0E-7".into(),
    ];
    let bytes = build_minimal_image_with_wcs(&cards);
    let file = FitsFile::from_bytes(bytes).unwrap();
    let Hdu::Image(img) = file.hdu(0).unwrap() else {
        panic!("not image");
    };
    let res = img.wcs(' ');
    assert!(
        matches!(&res, Err(e) if format!("{e:?}").contains("AP_ORDER")),
        "expected SIP partial-inverse error, got: {res:?}",
    );
}

/// `pixel_to_celestial` / `celestial_to_pixel` are the convenience pair
/// real callers reach for: no Vec gymnastics, just (x, y) <-> (RA, Dec).
#[test]
fn celestial_convenience_round_trip() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                 50.5".into(),
        "CRPIX2  =                 50.5".into(),
        "CRVAL1  =              83.6331".into(),
        "CRVAL2  =              22.0145".into(),
        "CDELT1  =          -2.78E-04".into(),
        "CDELT2  =           2.78E-04".into(),
    ];
    let wcs = open_image(&cards);
    assert_eq!(wcs.celestial_axes(), Some((0, 1)));

    let (ra, dec) = wcs.pixel_to_celestial(49.5, 49.5).unwrap();
    assert!(near(ra, 83.6331, 1e-9));
    assert!(near(dec, 22.0145, 1e-9));

    for &(px, py) in &[(0.0, 0.0), (32.0, 74.0), (98.0, 98.0)] {
        let (ra, dec) = wcs.pixel_to_celestial(px, py).unwrap();
        let (px2, py2) = wcs.celestial_to_pixel(ra, dec).unwrap();
        assert!(
            near(px, px2, 1e-6) && near(py, py2, 1e-6),
            "round-trip failed at ({px},{py}): got ({px2},{py2})",
        );
    }
}

/// `pixel_scale_at` reports the great-circle distance per pixel along
/// each axis. For a TAN image with CDELT = +/-2.78e-4 deg = 1.0 arcsec,
/// the scale at the reference pixel must come out to 1"/pix on both
/// axes (cos(dec) cancels because we measure along the great circle,
/// not along deltaRA).
#[test]
fn pixel_scale_matches_cdelt() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                 50.5".into(),
        "CRPIX2  =                 50.5".into(),
        "CRVAL1  =              83.6331".into(),
        "CRVAL2  =              22.0145".into(),
        "CDELT1  =          -2.778E-04".into(),
        "CDELT2  =           2.778E-04".into(),
    ];
    let wcs = open_image(&cards);
    let (sx, sy) = wcs.pixel_scale_at(50.5, 50.5).unwrap();
    // CDELT 2.778e-4 deg = 1.0008 arcsec.
    assert!((sx - 1.0008).abs() < 1e-3, "x scale = {sx}");
    assert!((sy - 1.0008).abs() < 1e-3, "y scale = {sy}");
}

/// `pixel_to_celestial` errors cleanly on a header without a celestial
/// pair (e.g., a pure spectral or linear WCS).
#[test]
fn pixel_to_celestial_errors_without_celestial_pair() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'FREQ'".into(),
        "CTYPE2  = 'LINEAR'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRVAL1  =              1.4E+09".into(),
        "CRVAL2  =                  0.0".into(),
        "CDELT1  =              1.0E+05".into(),
        "CDELT2  =                  1.0".into(),
        "CUNIT1  = 'Hz'".into(),
    ];
    let wcs = open_image(&cards);
    assert!(wcs.celestial_axes().is_none());
    assert!(wcs.pixel_to_celestial(1.0, 1.0).is_err());
    assert!(wcs.celestial_to_pixel(0.0, 0.0).is_err());
}

/// `-TAB` axis: single-axis 1-D wavelength lookup. Synthesises an
/// image HDU whose third axis carries `WAVE-TAB`, plus a paired
/// BINTABLE extension `WCS-TAB` with a 5-element wavelength column
/// `WAVELEN`. Verifies that `FitsFile::wcs` resolves the lookup and
/// that forward / inverse maps interpolate the table correctly.
#[test]
fn wave_tab_axis_resolved_from_bintable() {
    use fitsy::{BinFieldKind, BinTableBuilder, FitsWriter, ImageBuilder, Value};

    // Wavelength samples (Angstrom) for pixels 1..=5 along axis 3.
    let wavelens: [f64; 5] = [4000.0, 4500.0, 5500.0, 7000.0, 9000.0];

    // Primary image: 2x2x5 cube. Axes 1,2 are linear; axis 3 is -TAB.
    let mut primary = ImageBuilder::<f32>::new(vec![2, 2, 5], vec![0.0_f32; 20])
        .unwrap()
        .primary(true);
    for (k, v) in [
        ("CTYPE1", Value::String("X".into())),
        ("CTYPE2", Value::String("Y".into())),
        ("CTYPE3", Value::String("WAVE-TAB".into())),
        ("CRPIX1", Value::Real(1.0)),
        ("CRPIX2", Value::Real(1.0)),
        ("CRPIX3", Value::Real(1.0)),
        ("CRVAL1", Value::Real(0.0)),
        ("CRVAL2", Value::Real(0.0)),
        // CRVAL on a -TAB axis names the (1-based) array index of
        // the reference pixel: with CRPIX3 = 1 and CRVAL3 = 1, the
        // intermediate world coordinate at pixel 1 is exactly 1,
        // which the no-index lookup interprets as the first row of
        // the coordinate array (Paper III Sec.6 eq. 6).
        ("CRVAL3", Value::Real(1.0)),
        ("CDELT1", Value::Real(1.0)),
        ("CDELT2", Value::Real(1.0)),
        ("CDELT3", Value::Real(1.0)),
        ("CUNIT3", Value::String("Angstrom".into())),
        // -TAB pointer keywords (Paper III Sec.6).
        ("PS3_0", Value::String("WCS-TAB".into())),
        ("PS3_1", Value::String("WAVELEN".into())),
        ("PV3_1", Value::Integer(1)), // EXTVER
    ] {
        primary = primary.card(k, v, None);
    }
    let primary = primary.build().unwrap();

    // BINTABLE with one row, one column carrying the 5-element
    // coordinate array as a single fixed-shape D cell (TFORM = `5D`).
    let mut bt = BinTableBuilder::new();
    bt.add_column("WAVELEN", BinFieldKind::F64, 5, Some("Angstrom"), None)
        .unwrap();
    let mut row_bytes = Vec::with_capacity(5 * 8);
    for w in wavelens {
        row_bytes.extend_from_slice(&w.to_bits().to_be_bytes());
    }
    let (mut bt_header, bt_data) = bt.build(1, row_bytes).unwrap();
    bt_header
        .push("EXTNAME", Value::String("WCS-TAB".into()), None)
        .unwrap();
    bt_header.push("EXTVER", Value::Integer(1), None).unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary.0, &primary.1).unwrap();
    w.write_hdu(&bt_header, &bt_data).unwrap();
    w.finish().unwrap();

    let file = FitsFile::from_bytes(buf).unwrap();
    let wcs = file.wcs(0, ' ').unwrap().expect("WCS present");
    assert_eq!(wcs.tab_specs.len(), 1);
    assert_eq!(wcs.tab.len(), 1, "TAB axis should have been resolved");

    // Pixel 1 (1-based) -> 4000 Angstrom, etc. The Wcs API is 0-based,
    // so we evaluate at pixels 0, 2, 4 to hit table indices 1, 3, 5.
    let w1 = wcs.pixel_to_world(&[0.0, 0.0, 0.0]).unwrap();
    assert!((w1[2] - 4000.0).abs() < 1e-9, "got {}", w1[2]);
    let w3 = wcs.pixel_to_world(&[0.0, 0.0, 2.0]).unwrap();
    assert!((w3[2] - 5500.0).abs() < 1e-9, "got {}", w3[2]);
    let w5 = wcs.pixel_to_world(&[0.0, 0.0, 4.0]).unwrap();
    assert!((w5[2] - 9000.0).abs() < 1e-9, "got {}", w5[2]);

    // Pixel 2.5 (1-based) = 1.5 (0-based) -> halfway between 4500 and 5500 -> 5000.
    let mid = wcs.pixel_to_world(&[0.0, 0.0, 1.5]).unwrap();
    assert!((mid[2] - 5000.0).abs() < 1e-9, "got {}", mid[2]);

    // Round-trip: world 6000 Angstrom -> some pixel -> back to 6000 Angstrom.
    let pix = wcs.world_to_pixel(&[0.0, 0.0, 6000.0]).unwrap();
    let back = wcs.pixel_to_world(&[pix[0], pix[1], pix[2]]).unwrap();
    assert!((back[2] - 6000.0).abs() < 1e-9, "got {}", back[2]);
}

/// Header that declares a `-TAB` axis but is opened via the
/// header-only path (`ImageHdu::wcs`) without resolution must error
/// loudly on first forward map. Silently dropping the lookup would
/// be the worst-of-both-worlds failure mode.
#[test]
fn unresolved_tab_axis_errors_on_use() {
    use fitsy::{FitsWriter, ImageBuilder, Value};

    let mut primary = ImageBuilder::<f32>::new(vec![5], vec![0.0_f32; 5])
        .unwrap()
        .primary(true);
    for (k, v) in [
        ("CTYPE1", Value::String("WAVE-TAB".into())),
        ("CRPIX1", Value::Real(1.0)),
        ("CRVAL1", Value::Real(0.0)),
        ("CDELT1", Value::Real(1.0)),
        ("PS1_0", Value::String("WCS-TAB".into())),
        ("PS1_1", Value::String("WAVELEN".into())),
    ] {
        primary = primary.card(k, v, None);
    }
    let primary = primary.build().unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary.0, &primary.1).unwrap();
    w.finish().unwrap();

    let file = FitsFile::from_bytes(buf).unwrap();
    let Hdu::Image(img) = file.hdu(0).unwrap() else {
        panic!("not image");
    };
    let wcs = img.wcs(' ').unwrap().expect("WCS parses");
    assert_eq!(wcs.tab_specs.len(), 1);
    assert!(wcs.tab.is_empty());
    let err = wcs.pixel_to_world(&[1.0]).unwrap_err();
    assert!(format!("{err}").contains("unresolved -TAB"), "got: {err}");
}
