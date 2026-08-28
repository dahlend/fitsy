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
    try_open_image(cards).expect("header should describe a valid WCS")
}

/// `open_image` for a 4x4x16 cube, so tests can exercise a third
/// (e.g. spectral) axis.
fn open_image_3d(cards: &[String]) -> fitsy::Wcs {
    wcs_3d_result(cards)
        .expect("header should describe a valid WCS")
        .expect("wcs present")
}

/// [`open_image_3d`] keeping the error, for tests that assert a
/// non-conforming header is *rejected*.
fn wcs_3d_result(cards: &[String]) -> Result<Option<fitsy::Wcs>, fitsy::FitsError> {
    let mut buf = Vec::new();
    for c in [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    3",
        "NAXIS1  =                    4",
        "NAXIS2  =                    4",
        "NAXIS3  =                   16",
    ] {
        buf.extend_from_slice(&pad_card(c));
    }
    for c in cards {
        buf.extend_from_slice(&pad_card(c));
    }
    buf.extend_from_slice(&pad_card("END"));
    while buf.len() % BLOCK != 0 {
        buf.push(b' ');
    }
    let data_start = buf.len();
    buf.extend(std::iter::repeat_n(0_u8, 4 * 4 * 16));
    while (buf.len() - data_start) % BLOCK != 0 {
        buf.push(0);
    }
    let file = FitsFile::from_bytes(buf).unwrap();
    let Hdu::Image(img) = file.hdu(0).unwrap() else {
        panic!("not image");
    };
    img.wcs(' ')
}

/// `open_image` for callers sweeping parameter combinations, some of
/// which are not valid inputs for a given projection (e.g. a LATPOLE
/// with no native-pole solution). `None` means "this fixture is not a
/// WCS", which such callers skip rather than fail on.
fn try_open_image(cards: &[String]) -> Option<fitsy::Wcs> {
    let bytes = build_minimal_image_with_wcs(cards);
    let file = FitsFile::from_bytes(bytes).unwrap();
    let Hdu::Image(img) = file.hdu(0).unwrap() else {
        panic!("not image");
    };
    img.wcs(' ').ok().flatten()
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

/// `Wcs::crval` on the celestial pair must hold the true reference
/// value after parsing a header, matching what `fit_celestial_wcs`
/// produces directly. Regression test: the parser used to zero this
/// field out (the value lived only in `celestial.rotation`), which a
/// downstream consumer reading `wcs.crval()` directly (rather than
/// reading `crval()` directly) would silently see as `(0.0, 0.0)`.
#[test]
fn crval_field_matches_header_on_celestial_pair() {
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
    assert!(
        near(wcs.crval()[0], 83.6331, 1e-9),
        "crval[0] = {} (expected 83.6331, not zeroed)",
        wcs.crval()[0]
    );
    assert!(
        near(wcs.crval()[1], 22.0145, 1e-9),
        "crval[1] = {} (expected 22.0145, not zeroed)",
        wcs.crval()[1]
    );
    // Must also agree with the rotation block it's mirrored into.
    let cb = wcs.celestial.as_ref().unwrap();
    assert!(near(wcs.crval()[0], cb.rotation.alpha0, 1e-12));
    assert!(near(wcs.crval()[1], cb.rotation.delta0, 1e-12));
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

/// `EQUINOX` is retained only where a frame defines it. Paper II
/// Sec.3.1 makes it "not applicable to ICRS or GAPPT", yet
/// `EQUINOX = 2000.0` beside `RADESYS = 'ICRS'` is ubiquitous in real
/// headers -- the interpreted `Wcs` drops the redundant value, and the
/// source `Header` keeps the card.
#[test]
fn equinox_is_gated_on_a_frame_that_defines_it() {
    let base: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRVAL1  =                  0.0".into(),
        "CRVAL2  =                  0.0".into(),
        "CDELT1  =               -0.001".into(),
        "CDELT2  =                0.001".into(),
        "EQUINOX =               2000.0".into(),
    ];
    // ICRS defines the equinox away.
    let mut cards = base.clone();
    cards.push("RADESYS = 'ICRS'".into());
    let wcs = open_image(&cards);
    assert_eq!(wcs.radesys, fitsy::wcs::RadeSys::Icrs);
    assert_eq!(wcs.equinox, None);
    // ... and to_header emits no EQUINOX card for it.
    let written = wcs.to_header(' ').unwrap();
    assert!(written.first("EQUINOX").is_none());

    // A bare EQUINOX still drives the Sec.8.3 frame default -- the
    // value is read before the gate, so FK5 resolves and keeps it.
    let wcs = open_image(&base);
    assert_eq!(wcs.radesys, fitsy::wcs::RadeSys::Fk5);
    assert_eq!(wcs.equinox, Some(2000.0));
    let mut cards = base.clone();
    cards[8] = "EQUINOX =               1950.0".into();
    let wcs = open_image(&cards);
    assert_eq!(wcs.radesys, fitsy::wcs::RadeSys::Fk4);
    assert_eq!(wcs.equinox, Some(1950.0));

    // No celestial pair: no frame for either keyword to describe.
    let cards: Vec<String> = vec![
        "CTYPE1  = 'FREQ'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRVAL1  =               1.0E+9".into(),
        "CDELT1  =               1.0E+6".into(),
        "CUNIT1  = 'Hz'".into(),
        "EQUINOX =               2000.0".into(),
        "RADESYS = 'FK5'".into(),
    ];
    let wcs = open_image(&cards);
    assert_eq!(wcs.equinox, None);
    assert_eq!(wcs.radesys, fitsy::wcs::RadeSys::default());
}

/// `Wcs::axes` is the per-axis metadata table and is always exactly
/// `naxis` long, populated or not -- the invariant that used to span
/// five parallel vectors.
#[test]
fn axes_metadata_is_always_naxis_long() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRVAL1  =                  0.0".into(),
        "CRVAL2  =                  0.0".into(),
        "CDELT1  =               -0.001".into(),
        "CDELT2  =                0.001".into(),
    ];
    let wcs = open_image(&cards);
    assert_eq!(wcs.axes().len(), wcs.naxis());
    // No CNAME/CRDER/CSYER cards, so every axis carries only the
    // CTYPE/CUNIT it was parsed from and no metadata.
    assert!(
        wcs.axes()
            .iter()
            .all(|a| a.cname.is_none() && a.crder.is_none() && a.csyer.is_none())
    );

    let fitted = fitsy::wcs::fit_celestial_wcs(
        &[
            (0.0, 0.0),
            (10.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (5.0, 3.0),
        ],
        &[
            (150.0, 2.0),
            (150.01, 2.0),
            (150.0, 2.01),
            (150.01, 2.01),
            (150.005, 2.003),
        ],
        &fitsy::wcs::WcsFitOptions::default(),
    )
    .expect("fit");
    assert_eq!(fitted.wcs.axes().len(), fitted.wcs.naxis());
}

/// No `RADESYS` keyword: the default depends on `EQUINOX` per Paper II
/// Sec.3.1 -- pre-1984 => FK4, post-1984 => FK5, missing => ICRS.
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

/// `wcs.crval()` must be usable as a "hold every other axis at its
/// reference value" filler for `world_to_pixel` -- exactly the pattern
/// the celestial inverse uses internally (`let mut world =
/// self.crval().clone(); world[lon] = ra; world[lat] = dec;`), and the
/// natural thing for any external caller to do.
///
/// Regression test: before the parser stopped zeroing `crval` on the
/// celestial/spectral axes, feeding `wcs.crval()` straight back into
/// `world_to_pixel` did not merely give a wrong spectral-axis pixel --
/// for this fixture (celestial axes zeroed to RA=0/Dec=0, nowhere near
/// the true tangent point at RA=100/Dec=20) it made the TAN projection
/// inverse hit the unprojected hemisphere and return `Err`.
#[test]
fn crval_is_a_valid_reference_point_filler_with_spectral_axis() {
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
    assert!(near(wcs.crval()[0], 1.42e9, 1.0));
    assert!(near(wcs.crval()[1], 100.0, 1e-9));
    assert!(near(wcs.crval()[2], 20.0, 1e-9));

    let pix = wcs
        .world_to_pixel(wcs.crval())
        .expect("world_to_pixel(wcs.crval()) must succeed: it's the reference point");
    // CRPIX = 1.0 (1-based) on every axis -> 0-based reference pixel = 0.
    for (i, &p) in pix.iter().enumerate() {
        assert!(near(p, 0.0, 1e-6), "axis {i}: pix = {p}, expected ~0.0");
    }
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

/// Sky position of a pixel on a two-axis celestial WCS, as
/// `(lon, lat)`.
///
/// The public API returns world values in axis order and
/// `Wcs::axis_kinds` says which is which. Every WCS in this file puts
/// longitude on axis 1, so this helper indexes directly and states
/// that assumption once.
fn sky(wcs: &fitsy::Wcs, px: f64, py: f64) -> (f64, f64) {
    let w = wcs.pixel_to_world(&[px, py]).expect("pixel_to_world");
    (w[0], w[1])
}

/// Inverse of [`sky`].
fn pix(wcs: &fitsy::Wcs, lon: f64, lat: f64) -> (f64, f64) {
    let p = wcs.world_to_pixel(&[lon, lat]).expect("world_to_pixel");
    (p[0], p[1])
}

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
fn standard_projections_match_reference() {
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
        "{} failure(s) in standard_projections_match_reference:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

// -- SIP distortion --------------------------------------------------------

/// For every row of `wcs_sip.csv`:
///   - The reference pixel maps to CRVAL.
///   - `world_to_pixel(ra, dec)` recovers the stored pixel within 1e-8 px.
#[test]
fn sip_matches_reference() {
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
        "{} failure(s) in sip_matches_reference:\n  {}",
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
fn tpv_matches_reference() {
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
        "{} failure(s) in tpv_matches_reference:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

// -- Reference edge cases ---------------------------------------------------
//
// Moved here when `tests/wcs_integration.rs` was retired: that file had
// drifted into a stale copy of this one, and this was the only coverage
// it still held alone.

/// Build a `Wcs` for one row of `wcs_edgecases.csv` (per-row CRVAL/CDELT,
/// optional `PV2_1`/`PV2_2` and `LONPOLE`).
fn wcs_for_edge_row(row: &HashMap<String, String>) -> fitsy::Wcs {
    let code = &row["code"];
    let mut cards = base_cards(
        &format!("RA---{code}"),
        &format!("DEC--{code}"),
        f(row, "crpix1"),
        f(row, "crpix2"),
        f(row, "crval1"),
        f(row, "crval2"),
        f(row, "cdelt1"),
        f(row, "cdelt2"),
    );
    for (col, kw) in [
        ("pv2_1", "PV2_1"),
        ("pv2_2", "PV2_2"),
        ("lonpole", "LONPOLE"),
    ] {
        if let Some(v) = opt_f(row, col) {
            cards.push(format!("{kw:<8}= {v:>20e}"));
        }
    }
    open_image(&cards)
}

/// Reference values (see `gen_wcs_test_data.py`) in regions the standard grid never
/// reaches: HPX polar zone (incl. defaulted H/K), XPH away from the pole,
/// quad-cube faces 2-4, slant SIN (`PV2_1`/`PV2_2` != 0), and nonstandard
/// `LONPOLE` (incl. the degenerate `delta_p` = +/-90 branch). Both directions
/// are checked; forward/inverse bugs that cancel in round-trips fail here.
#[test]
fn edge_case_projections_match_reference() {
    let path = test_data_dir().join("wcs_edgecases.csv");
    let rows = parse_csv(&path);
    assert!(!rows.is_empty(), "CSV is empty");

    let mut failures: Vec<String> = Vec::new();
    let mut wcs_cache: HashMap<String, fitsy::Wcs> = HashMap::new();

    for row in &rows {
        let label = &row["label"];
        let wcs = wcs_cache
            .entry(label.clone())
            .or_insert_with(|| wcs_for_edge_row(row));

        let ra = f(row, "ra");
        let dec = f(row, "dec");
        // CSV pixels are 1-based FITS; the Wcs API is 0-based.
        let x_expected = f(row, "x_fits") - 1.0;
        let y_expected = f(row, "y_fits") - 1.0;
        let tol_px = f(row, "tol_px").max(1e-7);
        // Pixel-space tolerance converted to degrees on the sky.
        let tol_deg = (tol_px * f(row, "cdelt2").abs()).max(1e-8);

        match wcs.pixel_to_world(&[x_expected, y_expected]) {
            Ok(w) => {
                let dra =
                    ((w[0] - ra + 540.0).rem_euclid(360.0) - 180.0).abs() * dec.to_radians().cos();
                let ddec = (w[1] - dec).abs();
                if dra > tol_deg || ddec > tol_deg {
                    failures.push(format!(
                        "{label}: pixel_to_world({x_expected},{y_expected}) = \
                         ({:.10},{:.10}) expected ({ra:.10},{dec:.10})",
                        w[0], w[1]
                    ));
                }
            }
            Err(e) => {
                failures.push(format!(
                    "{label}: pixel_to_world({x_expected},{y_expected}) failed: {e}"
                ));
            }
        }

        match wcs.world_to_pixel(&[ra, dec]) {
            Ok(pix) => {
                let ex = (pix[0] - x_expected).abs();
                let ey = (pix[1] - y_expected).abs();
                if ex > tol_px || ey > tol_px {
                    failures.push(format!(
                        "{label}: world_to_pixel({ra:.6},{dec:.6}) = ({:.10},{:.10}) \
                         expected ({x_expected},{y_expected})",
                        pix[0], pix[1]
                    ));
                }
            }
            Err(e) => {
                failures.push(format!(
                    "{label}: world_to_pixel({ra:.6},{dec:.6}) failed: {e}"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} failure(s) in edge_case_projections_match_reference:\n  {}",
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

/// `SPECSYS`, `SSYSOBS`, `VELOSYS` (Paper III Sec.7) are retained on
/// the parsed `Wcs` -- but only when there is a spectral axis for them
/// to describe. `Wcs` is the interpreted layer; a header carrying them
/// without one keeps them in its `Header` alone.
#[test]
fn spectral_reference_frame_keywords_gated_on_a_spectral_axis() {
    let celestial: Vec<String> = vec![
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
    // No spectral axis: the frame describes nothing and is dropped.
    let wcs = open_image(&celestial);
    assert!(wcs.spectral_frame.is_none());

    // A spectral axis -- even a bare linear one -- retains it.
    let mut cards = celestial;
    cards.insert(2, "CTYPE3  = 'FREQ'".into());
    cards.push("CRPIX3  =                  1.0".into());
    cards.push("CRVAL3  =               1.0E+09".into());
    cards.push("CDELT3  =               1.0E+06".into());
    cards.push("CUNIT3  = 'Hz'".into());
    let wcs = open_image_3d(&cards);
    let frame = wcs.spectral_frame.expect("frame retained");
    assert_eq!(frame.specsys.as_deref(), Some("BARYCENT"));
    assert_eq!(frame.ssysobs.as_deref(), Some("TOPOCENT"));
    assert_eq!(frame.velosys, Some(12345.0));
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
    assert_eq!(wcs.naxis(), 3);
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
    let plate_center_x = dss.ppo3() / dss.xpixelsz() - dss.cnpix1() + 0.5 - 1.0;
    let plate_center_y = dss.ppo6() / dss.ypixelsz() - dss.cnpix2() + 0.5 - 1.0;
    let world = wcs
        .pixel_to_world(&[plate_center_x, plate_center_y])
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

/// `pixel_to_world` / `world_to_pixel` are the general pair
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

    let (ra, dec) = sky(&wcs, 49.5, 49.5);
    assert!(near(ra, 83.6331, 1e-9));
    assert!(near(dec, 22.0145, 1e-9));

    for &(px, py) in &[(0.0, 0.0), (32.0, 74.0), (98.0, 98.0)] {
        let (ra, dec) = sky(&wcs, px, py);
        let (px2, py2) = pix(&wcs, ra, dec);
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

/// The celestial-only entry points error cleanly on a header without a
/// celestial pair (e.g., a pure spectral or linear WCS). The general
/// transform still works there -- that is the point of it.
#[test]
fn celestial_only_helpers_error_without_celestial_pair() {
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
    assert!(wcs.pixel_scale_at(1.0, 1.0).is_err());
    // The general transform has no such requirement.
    assert!(wcs.pixel_to_world(&[1.0, 1.0]).is_ok());
    assert!(wcs.pixel_to_world_many(&[1.0, 1.0]).is_ok());
}

/// `-TAB` axis: single-axis 1-D wavelength lookup. Synthesizes an
/// image HDU whose third axis carries `WAVE-TAB`, plus a paired
/// BINTABLE extension `WCS-TAB` with a 5-element wavelength column
/// `WAVELEN`. Verifies that `FitsFile::wcs` resolves the lookup and
/// that forward / inverse maps interpolate the table correctly.
#[test]
fn wave_tab_axis_resolved_from_bintable() {
    use fitsy::{AxisKind, BinFieldKind, BinTableBuilder, FitsWriter, ImageBuilder, Value};

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

    // `wcs_inherited` reads the same file, so it resolves the
    // lookup too.
    let inherited = file.wcs_inherited(0, ' ').unwrap().expect("WCS present");
    assert_eq!(inherited.tab.len(), 1, "wcs_inherited left -TAB unresolved");

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

    // `-TAB` is an algorithm, not a coordinate type: axis 3 is
    // spectral *and* tabular. The parser files a `WAVE-TAB` axis under
    // `tab_specs` alone, so a kind read from parsed algorithm state
    // rather than from CTYPE would call this axis linear.
    assert_eq!(
        wcs.axis_kinds(),
        vec![AxisKind::Linear, AxisKind::Linear, AxisKind::Spectral]
    );
    assert!(wcs.is_tabular(2), "axis 3 carries -TAB");
    assert!(!wcs.is_tabular(0));

    // Round-trip: world 6000 Angstrom -> some pixel -> back to 6000 Angstrom.
    let pix = wcs.world_to_pixel(&[0.0, 0.0, 6000.0]).unwrap();
    let back = wcs.pixel_to_world(&[pix[0], pix[1], pix[2]]).unwrap();
    assert!((back[2] - 6000.0).abs() < 1e-9, "got {}", back[2]);

    // `to_header` must reproduce the PSi_/PVi_ pointer cards so the
    // serialized header still finds the same lookup table. The table
    // lives in its own HDU and cannot travel inside a `Header`, so we
    // re-resolve against the original file.
    let serialized = wcs.to_header(' ').unwrap();
    let mut reparsed = fitsy::Wcs::from_header(&serialized, ' ')
        .unwrap()
        .expect("serialized -TAB header still describes a WCS");
    assert_eq!(reparsed.tab_specs.len(), 1, "PSi_/PVi_ cards were dropped");
    assert_eq!(reparsed.resolve_tab(&file).unwrap(), 1);
    for pz in [0.0, 1.5, 2.0, 4.0] {
        let a = wcs.pixel_to_world(&[0.0, 0.0, pz]).unwrap();
        let b = reparsed.pixel_to_world(&[0.0, 0.0, pz]).unwrap();
        assert!(
            (a[2] - b[2]).abs() < 1e-12,
            "-TAB round-trip drifted at pixel {pz}: {} vs {}",
            a[2],
            b[2]
        );
    }
}

/// `PVi_2` (`EXTLEVEL`) must survive `to_header`: the resolver checks
/// it against the table HDU, so dropping it breaks the round trip for
/// any file whose lookup table carries `EXTLEVEL != 1`.
#[test]
fn tab_extlevel_round_trips() {
    use fitsy::{BinFieldKind, BinTableBuilder, FitsWriter, ImageBuilder, Value};

    let mut primary = ImageBuilder::<f32>::new(vec![2, 2, 5], vec![0.0_f32; 20])
        .unwrap()
        .primary(true);
    for (k, v) in [
        ("CTYPE1", Value::String("X".into())),
        ("CTYPE2", Value::String("Y".into())),
        ("CTYPE3", Value::String("WAVE-TAB".into())),
        ("CRPIX3", Value::Real(1.0)),
        ("CRVAL3", Value::Real(1.0)),
        ("CDELT3", Value::Real(1.0)),
        ("CUNIT3", Value::String("m".into())),
        ("PS3_0", Value::String("WCS-TAB".into())),
        ("PS3_1", Value::String("WAVELEN".into())),
        ("PV3_1", Value::Integer(1)),
        ("PV3_2", Value::Integer(7)), // EXTLEVEL
    ] {
        primary = primary.card(k, v, None);
    }
    let primary = primary.build().unwrap();

    let mut bt = BinTableBuilder::new();
    bt.add_column("WAVELEN", BinFieldKind::F64, 5, Some("m"), None)
        .unwrap();
    let mut row = Vec::with_capacity(5 * 8);
    for w in [4e-7_f64, 4.5e-7, 5.5e-7, 7e-7, 9e-7] {
        row.extend_from_slice(&w.to_bits().to_be_bytes());
    }
    let (mut bh, bd) = bt.build(1, row).unwrap();
    bh.push("EXTNAME", Value::String("WCS-TAB".into()), None)
        .unwrap();
    bh.push("EXTVER", Value::Integer(1), None).unwrap();
    bh.push("EXTLEVEL", Value::Integer(7), None).unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary.0, &primary.1).unwrap();
    w.write_hdu(&bh, &bd).unwrap();
    w.finish().unwrap();
    let file = FitsFile::from_bytes(buf).unwrap();

    let wcs = file.wcs(0, ' ').unwrap().expect("WCS present");
    assert_eq!(wcs.tab_specs[0].extlevel, 7);

    let serialized = wcs.to_header(' ').unwrap();
    let mut reparsed = fitsy::Wcs::from_header(&serialized, ' ')
        .unwrap()
        .expect("serialized header still describes a WCS");
    assert_eq!(
        reparsed.tab_specs[0].extlevel, 7,
        "PV3_2 dropped by to_header"
    );
    // ... and the reparsed WCS still resolves against the same file.
    assert_eq!(reparsed.resolve_tab(&file).unwrap(), 1);
    let a = wcs.pixel_to_world(&[0.0, 0.0, 2.0]).unwrap();
    let b = reparsed.pixel_to_world(&[0.0, 0.0, 2.0]).unwrap();
    assert!((a[2] - b[2]).abs() < 1e-15, "{} vs {}", a[2], b[2]);
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

    // Every entry point, not just the forward single-point one. The
    // guard is hoisted out of the per-point body and run once per
    // call, so each entry point carries its own copy. A batch in
    // particular must fail the whole call: filling `NaN` would report
    // an unresolved lookup as a set of out-of-domain points.
    let err = wcs.pixel_to_world(&[1.0]).unwrap_err();
    assert!(format!("{err}").contains("unresolved -TAB"), "got: {err}");
    let err = wcs.world_to_pixel(&[1.0]).unwrap_err();
    assert!(format!("{err}").contains("unresolved -TAB"), "got: {err}");
    let err = wcs.pixel_to_world_many(&[1.0, 2.0]).unwrap_err();
    assert!(format!("{err}").contains("unresolved -TAB"), "got: {err}");
    let err = wcs.world_to_pixel_many(&[1.0, 2.0]).unwrap_err();
    assert!(format!("{err}").contains("unresolved -TAB"), "got: {err}");
}

// -------------------------------------------------------------------
// Edge-case coverage near the celestial poles and the alpha = pm 180
// meridian. These are the regions where projection inverses lose
// numerical precision (at the pole, alpha is undefined; near alpha =
// pm 180 the wrap convention matters). We exercise the four most
// commonly used celestial projections.
// -------------------------------------------------------------------

fn pole_round_trip_for_projection(proj: &str, dec_sign: f64) {
    // CRVAL placed 0.01 deg from the pole; image pixels span a
    // region that crosses the pole when dec_sign = +/-1.
    let crval2 = dec_sign * (90.0 - 0.01);
    let cards: Vec<String> = vec![
        format!("CTYPE1  = 'RA---{proj}'"),
        format!("CTYPE2  = 'DEC--{proj}'"),
        "CRPIX1  =                 50.5".into(),
        "CRPIX2  =                 50.5".into(),
        "CRVAL1  =                  0.0".into(),
        format!("CRVAL2  =          {crval2:>14.4}"),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
    ];
    let wcs = open_image(&cards);
    // Pixels off-axis, near the reference pixel: small enough that
    // every projection here is well-defined.
    for &(px, py) in &[
        (45.0, 45.0),
        (50.5, 50.5),
        (55.0, 55.0),
        (49.0, 51.0),
        (51.0, 49.0),
    ] {
        let world = wcs
            .pixel_to_world(&[px, py])
            .unwrap_or_else(|e| panic!("{proj} fwd failed at ({px},{py}): {e}"));
        let back = wcs
            .world_to_pixel(&world)
            .unwrap_or_else(|e| panic!("{proj} inv failed at ({px},{py}): {e}"));
        assert!(
            back[0].is_finite() && back[1].is_finite(),
            "{proj} non-finite inverse near {}-pole: world=({},{}) -> ({},{})",
            if dec_sign > 0.0 { "north" } else { "south" },
            world[0],
            world[1],
            back[0],
            back[1]
        );
        assert!(
            near(back[0], px, 1e-5) && near(back[1], py, 1e-5),
            "{proj} round trip failed near {}-pole at ({px},{py}): world=({},{}) -> ({},{})",
            if dec_sign > 0.0 { "north" } else { "south" },
            world[0],
            world[1],
            back[0],
            back[1]
        );
        // Dec must lie in [-90, 90] and must not exceed pole.
        assert!(
            world[1].abs() <= 90.0 + 1e-9,
            "{proj} produced |dec|>90: dec = {}",
            world[1]
        );
    }
}

fn dateline_round_trip_for_projection(proj: &str) {
    // CRVAL straddles the alpha = +/- 180 meridian; an off-center
    // pixel will land on the other side of the wrap.
    let cards: Vec<String> = vec![
        format!("CTYPE1  = 'RA---{proj}'"),
        format!("CTYPE2  = 'DEC--{proj}'"),
        "CRPIX1  =                 50.5".into(),
        "CRPIX2  =                 50.5".into(),
        "CRVAL1  =               179.99".into(),
        "CRVAL2  =                  0.0".into(),
        "CDELT1  =              -0.001".into(),
        "CDELT2  =               0.001".into(),
    ];
    let wcs = open_image(&cards);
    for &(px, py) in &[
        (1.0, 50.5),
        (40.0, 50.5),
        (50.5, 50.5),
        (60.0, 50.5),
        (100.0, 50.5),
    ] {
        let world = wcs
            .pixel_to_world(&[px, py])
            .unwrap_or_else(|e| panic!("{proj} fwd failed at ({px},{py}): {e}"));
        let back = wcs
            .world_to_pixel(&world)
            .unwrap_or_else(|e| panic!("{proj} inv failed at ({px},{py}): {e}"));
        assert!(
            back[0].is_finite() && back[1].is_finite(),
            "{proj} non-finite across dateline: world=({},{}) -> ({},{})",
            world[0],
            world[1],
            back[0],
            back[1]
        );
        assert!(
            near(back[0], px, 1e-5) && near(back[1], py, 1e-5),
            "{proj} dateline round trip failed at ({px},{py}): world=({},{}) -> ({},{})",
            world[0],
            world[1],
            back[0],
            back[1]
        );
        // RA returned in [0, 360) by FITS convention.
        assert!(
            world[0] >= 0.0 && world[0] < 360.0 + 1e-9,
            "{proj} returned out-of-range RA: {}",
            world[0]
        );
    }
}

#[test]
fn tan_round_trip_near_north_pole() {
    pole_round_trip_for_projection("TAN", 1.0);
}

#[test]
fn tan_round_trip_near_south_pole() {
    pole_round_trip_for_projection("TAN", -1.0);
}

#[test]
fn sin_round_trip_near_north_pole() {
    pole_round_trip_for_projection("SIN", 1.0);
}

#[test]
fn zea_round_trip_near_north_pole() {
    pole_round_trip_for_projection("ZEA", 1.0);
}

#[test]
fn ait_round_trip_near_north_pole() {
    pole_round_trip_for_projection("AIT", 1.0);
}

#[test]
fn tan_round_trip_across_dateline() {
    dateline_round_trip_for_projection("TAN");
}

#[test]
fn sin_round_trip_across_dateline() {
    dateline_round_trip_for_projection("SIN");
}

#[test]
fn zea_round_trip_across_dateline() {
    dateline_round_trip_for_projection("ZEA");
}

#[test]
fn ait_round_trip_across_dateline() {
    dateline_round_trip_for_projection("AIT");
}

// ---------------------------------------------------------------------
// pixel_shape / footprint
// ---------------------------------------------------------------------

/// The batch fast path must reproduce the general body exactly.
///
/// `pixel_to_world_many` routes a plain two-axis celestial WCS through
/// a specialized loop that skips the per-point work such a WCS does not
/// need. `pixel_to_world` always takes the general body, so comparing
/// them checks the specialization against its own reference.
///
/// Bit-for-bit, not near: the fast path performs the same operations in
/// the same order, so any difference at all means the two have drifted.
#[test]
fn fast_path_matches_general() {
    // Every projection is reached through the same specialized loop.
    // The set below therefore spans the shapes that loop has to
    // handle: swapped axis order, a rotated matrix, a non-degree
    // CUNIT, a moved fiducial point, and a bounded domain.
    let cases: Vec<(&str, Vec<String>)> = vec![
        ("plain TAN", tan_cards()),
        ("swapped axis order", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'DEC--TAN'".into();
            c[1] = "CTYPE2  = 'RA---TAN'".into();
            c
        }),
        ("rotated CD matrix", {
            let mut c = tan_cards();
            c.truncate(6);
            c.push("CD1_1   =            -0.0008".into());
            c.push("CD1_2   =             0.0003".into());
            c.push("CD2_1   =             0.0003".into());
            c.push("CD2_2   =             0.0008".into());
            c
        }),
        ("arcsec CUNIT", {
            let mut c = tan_cards();
            c.push("CUNIT1  = 'arcsec'".into());
            c.push("CUNIT2  = 'arcsec'".into());
            c
        }),
        ("moved fiducial point", {
            let mut c = tan_cards();
            c.push("PV1_1   =                 30.0".into());
            c.push("PV1_2   =                 70.0".into());
            c
        }),
        ("SIN, bounded domain", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'RA---SIN'".into();
            c[1] = "CTYPE2  = 'DEC--SIN'".into();
            c[6] = "CDELT1  =                 -1.0".into();
            c[7] = "CDELT2  =                  1.0".into();
            c
        }),
        ("AIT, non-zenithal", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'GLON-AIT'".into();
            c[1] = "CTYPE2  = 'GLAT-AIT'".into();
            c
        }),
        // The fast path carries SIP, so SIP has to agree here too. It
        // applies the distortion to the celestial pair by index. The
        // swapped case below is what pins that indexing down.
        ("SIP", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'RA---TAN-SIP'".into();
            c[1] = "CTYPE2  = 'DEC--TAN-SIP'".into();
            c.extend(sip_cards());
            c
        }),
        ("SIP, swapped axis order", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'DEC--TAN-SIP'".into();
            c[1] = "CTYPE2  = 'RA---TAN-SIP'".into();
            c.extend(sip_cards());
            c
        }),
        // The fast path carries TPV too. It applies between the linear
        // stage and the projection, so these pin that slot, and the
        // swapped case pins which intermediate coordinate is which.
        ("TPV", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'RA---TPV'".into();
            c[1] = "CTYPE2  = 'DEC--TPV'".into();
            c.extend(tpv_cards());
            c
        }),
        ("TPV, swapped axis order", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'DEC--TPV'".into();
            c[1] = "CTYPE2  = 'RA---TPV'".into();
            c.extend(tpv_cards());
            c
        }),
        // The fast path carries TNX too, in the same slot as TPV.
        // The nonzero 2x2 surface exercises the Newton inverse; the
        // swapped case pins the intermediate-coordinate indexing.
        ("TNX polynomial pre-warp", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'RA---TNX'".into();
            c[1] = "CTYPE2  = 'DEC--TNX'".into();
            c.push(
                "WAT1_001= 'wtype=tnx axtype=ra lngcor = \"3 2 2 1 -1 1 -1 1 0 1E-3 5E-4 0\"'"
                    .into(),
            );
            c.push(
                "WAT2_001= 'wtype=tnx axtype=dec latcor = \"3 2 2 1 -1 1 -1 1 0 -4E-4 8E-4 0\"'"
                    .into(),
            );
            c
        }),
        ("TNX, swapped axis order", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'DEC--TNX'".into();
            c[1] = "CTYPE2  = 'RA---TNX'".into();
            c.push(
                "WAT1_001= 'wtype=tnx axtype=dec latcor = \"3 2 2 1 -1 1 -1 1 0 -4E-4 8E-4 0\"'"
                    .into(),
            );
            c.push(
                "WAT2_001= 'wtype=tnx axtype=ra lngcor = \"3 2 2 1 -1 1 -1 1 0 1E-3 5E-4 0\"'"
                    .into(),
            );
            c
        }),
        ("TPV with a radial term", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'RA---TPV'".into();
            c[1] = "CTYPE2  = 'DEC--TPV'".into();
            c.extend(tpv_cards());
            // `PV_3` is the `r` term, whose gradient diverges at the
            // origin. The fast path must reach the same guard.
            c.push("PV1_3   =            2.0E-05".into());
            c.push("PV2_3   =           -1.0E-05".into());
            c
        }),
        ("SIP with a rotated CD matrix", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'RA---TAN-SIP'".into();
            c[1] = "CTYPE2  = 'DEC--TAN-SIP'".into();
            c.truncate(6);
            c.push("CD1_1   =            -0.0008".into());
            c.push("CD1_2   =             0.0003".into());
            c.push("CD2_1   =             0.0003".into());
            c.push("CD2_2   =             0.0008".into());
            c.extend(sip_cards());
            c
        }),
    ];

    for (name, cards) in cases {
        let wcs = open_image(&cards);
        let mut flat = Vec::new();
        for i in 0..24 {
            for j in 0..24 {
                flat.push(f64::from(i) * 11.0 - 60.0);
                flat.push(f64::from(j) * 11.0 - 60.0);
            }
        }
        let batch = wcs.pixel_to_world_many(&flat).unwrap();
        let mut compared = 0;
        for (point, got) in flat
            .as_chunks::<2>()
            .0
            .iter()
            .zip(batch.as_chunks::<2>().0.iter())
        {
            match wcs.pixel_to_world(point) {
                Ok(want) => {
                    assert_eq!(
                        got[0].to_bits(),
                        want[0].to_bits(),
                        "{name}: lon differs at {point:?}: {} vs {}",
                        got[0],
                        want[0]
                    );
                    assert_eq!(
                        got[1].to_bits(),
                        want[1].to_bits(),
                        "{name}: lat differs at {point:?}: {} vs {}",
                        got[1],
                        want[1]
                    );
                    compared += 1;
                }
                // The general body reports the reason; the batch fills
                // NaN. Both must agree that the point is unusable.
                Err(_) => assert!(
                    got[0].is_nan() && got[1].is_nan(),
                    "{name}: expected NaN at {point:?}, got {got:?}"
                ),
            }
        }
        // Guards against a vacuous pass: `SIN` rejects most of this
        // grid, so the bar is well below the 576 points offered, but
        // above zero.
        assert!(compared > 50, "{name}: only {compared} usable points");

        // The inverse takes the same fast path, checked against the
        // general body on the world values just produced.
        let world: Vec<f64> = batch.iter().copied().filter(|v| !v.is_nan()).collect();
        let world = &world[..world.len() - world.len() % 2];
        let back = wcs.world_to_pixel_many(world).unwrap();
        let mut inv_compared = 0;
        for (point, got) in world
            .as_chunks::<2>()
            .0
            .iter()
            .zip(back.as_chunks::<2>().0.iter())
        {
            match wcs.world_to_pixel(point) {
                Ok(want) => {
                    assert_eq!(
                        got[0].to_bits(),
                        want[0].to_bits(),
                        "{name}: inverse x differs at {point:?}"
                    );
                    assert_eq!(
                        got[1].to_bits(),
                        want[1].to_bits(),
                        "{name}: inverse y differs at {point:?}"
                    );
                    inv_compared += 1;
                }
                Err(_) => assert!(
                    got[0].is_nan() && got[1].is_nan(),
                    "{name}: expected NaN from the inverse at {point:?}"
                ),
            }
        }
        assert!(
            inv_compared > 50,
            "{name}: only {inv_compared} usable inverse points"
        );
    }
}

/// Every feature the fast path cannot handle must send the WCS back to
/// the general body, and the answer must not change either way.
#[test]
fn fast_path_declines_what_it_cannot_handle() {
    let disqualifying: Vec<(&str, Vec<String>)> = vec![
        // SIP and TPV are absent on purpose. The fast path carries
        // both, and `fast_path_matches_general` checks them there.
        ("TNX distortion", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'RA---TNX'".into();
            c[1] = "CTYPE2  = 'DEC--TNX'".into();
            c.push("WAT1_001= 'wtype=tnx axtype=ra lngcor = \"3 1 1 1 -1 1 -1 1 5E-4\"'".into());
            c
        }),
        ("no celestial pair", {
            let mut c = tan_cards();
            c[0] = "CTYPE1  = 'WAVE'".into();
            c[1] = "CTYPE2  = 'TIME'".into();
            c
        }),
    ];

    for (name, cards) in disqualifying {
        let wcs = open_image(&cards);
        let flat = vec![0.0, 0.0, 31.0, 23.0, 63.0, 47.0, 10.5, 40.25];
        let batch = wcs.pixel_to_world_many(&flat).unwrap();
        for (point, got) in flat
            .as_chunks::<2>()
            .0
            .iter()
            .zip(batch.as_chunks::<2>().0.iter())
        {
            let want = wcs.pixel_to_world(point).unwrap();
            assert_eq!(got[0].to_bits(), want[0].to_bits(), "{name} at {point:?}");
            assert_eq!(got[1].to_bits(), want[1].to_bits(), "{name} at {point:?}");
        }
        let back = wcs.world_to_pixel_many(&batch).unwrap();
        for (point, got) in batch
            .as_chunks::<2>()
            .0
            .iter()
            .zip(back.as_chunks::<2>().0.iter())
        {
            let want = wcs.world_to_pixel(point).unwrap();
            assert_eq!(
                got[0].to_bits(),
                want[0].to_bits(),
                "{name} inverse at {point:?}"
            );
            assert_eq!(
                got[1].to_bits(),
                want[1].to_bits(),
                "{name} inverse at {point:?}"
            );
        }
    }

    // A cube keeps every axis, including the one the fast path has no
    // slot for.
    let cube: Vec<String> = [
        "CTYPE1  = 'RA---TAN'",
        "CTYPE2  = 'DEC--TAN'",
        "CTYPE3  = 'FREQ'",
        "CRPIX1  =                 32.0",
        "CRPIX2  =                 24.0",
        "CRPIX3  =                  5.0",
        "CRVAL1  =              202.469",
        "CRVAL2  =               47.195",
        "CRVAL3  =              1.4E+09",
        "CDELT1  =               -0.001",
        "CDELT2  =                0.001",
        "CDELT3  =              1.0E+06",
        "CUNIT3  = 'Hz'",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let wcs = open_image_3d(&cube);
    let pts = vec![0.0, 0.0, 1.0, 31.0, 23.0, 4.0];
    let batch = wcs.pixel_to_world_many(&pts).unwrap();
    for (point, got) in pts
        .as_chunks::<3>()
        .0
        .iter()
        .zip(batch.as_chunks::<3>().0.iter())
    {
        let want = wcs.pixel_to_world(point).unwrap();
        for k in 0..3 {
            assert_eq!(got[k].to_bits(), want[k].to_bits(), "cube axis {k}");
        }
    }
}

/// A round trip must not lose accuracy near the reference pixel.
///
/// A zenithal projection puts the whole field near the native pole,
/// where `theta` approaches 90 degrees. Recovering it with `asin` alone
/// is ill-conditioned there, because `d(asin)/dz = 1 / cos(theta)`
/// diverges. The error therefore grew as the point approached `CRPIX`,
/// reaching 7.5e-8 pixels one pixel away. That is the opposite of what
/// a caller expects: the worst accuracy at the field center.
///
/// The residual is now flat across the field, so this asserts a bound
/// the old code missed by two orders of magnitude at short range.
///
/// The defect was in the shared native-to-celestial rotation, so this
/// checks every projection rather than the `TAN` the fix was found on.
/// The distances reach a thousandth of a pixel. The error grew as the
/// point approached the reference, so a one-pixel floor saw only the
/// tail of it.
///
/// `CAR`, `AIT` and `MOL` are the controls. Their fiducial point is not
/// a pole, so the defect never reached them. They must not move.
#[test]
fn celestial_round_trip_holds_accuracy_near_the_reference_pixel() {
    for proj in [
        "TAN", "SIN", "ARC", "ZEA", "STG", "AZP", "SZP", "AIR", "CAR", "AIT", "MOL",
    ] {
        let mut cards = tan_cards();
        cards[0] = format!("CTYPE1  = 'RA---{proj}'");
        cards[1] = format!("CTYPE2  = 'DEC--{proj}'");
        // A wide field, so "far from CRPIX" is genuinely far.
        cards[2] = "CRPIX1  =               2000.0".into();
        cards[3] = "CRPIX2  =               2000.0".into();
        let wcs = open_image(&cards);

        // CRPIX is 1-based, so the reference pixel is 1999 here.
        for d in [1e-3_f64, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1999.0] {
            for (px, py) in [
                (1999.0 + d, 1999.0),
                (1999.0, 1999.0 + d),
                (1999.0 + d, 1999.0 + d),
            ] {
                let w = wcs.pixel_to_world(&[px, py]).unwrap();
                let back = wcs.world_to_pixel(&w).unwrap();
                let err = (back[0] - px).abs().max((back[1] - py).abs());
                assert!(
                    err < 1e-9,
                    "{proj}: round trip lost {err:e} pixels at ({px}, {py}), \
                     {d} from the reference"
                );
            }
        }
    }
}

/// A third-order SIP distortion, forward and inverse.
///
/// Each coefficient is large enough to matter. Dropping any one of them
/// moves a corner pixel well past the bit-level tolerance the
/// fast-path comparison holds to.
/// A cubic TPV solution, the shape a mosaic camera actually carries.
///
/// `PV*_1` is the identity scaling. The quadratic and cubic terms are
/// large enough that dropping any one of them moves a corner pixel well
/// past the bit-level tolerance the fast-path comparison holds to.
fn tpv_cards() -> Vec<String> {
    [
        "PV1_1   =                  1.0",
        "PV1_4   =            2.0000E-04",
        "PV1_5   =            1.0000E-04",
        "PV1_6   =           -1.5000E-04",
        "PV1_7   =            3.0000E-03",
        "PV1_8   =            1.0000E-03",
        "PV1_9   =           -2.0000E-03",
        "PV1_10  =            5.0000E-04",
        "PV2_1   =                  1.0",
        "PV2_4   =           -1.0000E-04",
        "PV2_5   =            2.0000E-04",
        "PV2_6   =            1.0000E-04",
        "PV2_7   =           -2.0000E-03",
        "PV2_8   =            4.0000E-03",
        "PV2_9   =            1.0000E-03",
        "PV2_10  =           -3.0000E-03",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

fn sip_cards() -> Vec<String> {
    [
        "A_ORDER =                    3",
        "B_ORDER =                    3",
        "A_2_0   =           1.2000E-05",
        "A_1_1   =          -3.1000E-06",
        "A_0_2   =           8.0000E-06",
        "A_3_0   =           2.0000E-09",
        "A_2_1   =          -1.0000E-09",
        "A_1_2   =           3.0000E-09",
        "A_0_3   =           1.0000E-09",
        "B_2_0   =          -9.0000E-06",
        "B_1_1   =           2.2000E-05",
        "B_0_2   =          -4.0000E-06",
        "B_3_0   =           1.0000E-09",
        "B_2_1   =           2.0000E-09",
        "B_1_2   =          -1.0000E-09",
        "B_0_3   =           4.0000E-09",
        "AP_ORDER=                    3",
        "BP_ORDER=                    3",
        "AP_2_0  =          -1.2000E-05",
        "AP_1_1  =           3.1000E-06",
        "AP_0_2  =          -8.0000E-06",
        "BP_2_0  =           9.0000E-06",
        "BP_1_1  =          -2.2000E-05",
        "BP_0_2  =           4.0000E-06",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

fn tan_cards() -> Vec<String> {
    [
        "CTYPE1  = 'RA---TAN'",
        "CTYPE2  = 'DEC--TAN'",
        "CRPIX1  =                 50.0",
        "CRPIX2  =                 50.0",
        "CRVAL1  =                 10.0",
        "CRVAL2  =                 -5.0",
        "CDELT1  =              -0.0010",
        "CDELT2  =               0.0010",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

#[test]
fn pixel_shape_snapshots_naxisn() {
    let wcs = open_image(&tan_cards());
    assert_eq!(wcs.pixel_shape.as_deref(), Some([100_u64, 100].as_slice()));
}

#[test]
fn pixel_shape_is_absent_for_a_fitted_wcs() {
    use fitsy::wcs::WcsFitOptions;
    let pixels = [(10.0, 10.0), (90.0, 10.0), (10.0, 90.0), (90.0, 90.0)];
    let sky = [(10.00, -5.00), (9.92, -5.00), (10.00, -4.92), (9.92, -4.92)];
    let fit = fitsy::wcs::fit_celestial_wcs(&pixels, &sky, &WcsFitOptions::default()).unwrap();
    assert!(fit.wcs.pixel_shape.is_none());
    assert!(fit.wcs.footprint().is_err());
}

#[test]
fn footprint_returns_corner_pixel_centers() {
    let wcs = open_image(&tan_cards());
    let fp = wcs.footprint().unwrap();
    // Gray-code order walks a two-axis image counter-clockwise from
    // the origin: (0,0), (99,0), (99,99), (0,99).
    assert_eq!(fp.len(), 4 * 2);
    for (got, (px, py)) in
        fp.as_chunks::<2>()
            .0
            .iter()
            .zip([(0.0, 0.0), (99.0, 0.0), (99.0, 99.0), (0.0, 99.0)])
    {
        let want = wcs.pixel_to_world(&[px, py]).unwrap();
        assert!(near(got[0], want[0], 1e-12) && near(got[1], want[1], 1e-12));
    }
}

/// A footprint no longer needs a celestial pair. A spectral / time
/// image has corners in its own world units.
#[test]
fn footprint_works_without_a_celestial_pair() {
    let cards: Vec<String> = [
        "CTYPE1  = 'WAVE'",
        "CTYPE2  = 'TIME'",
        "CUNIT1  = 'm'",
        "CUNIT2  = 's'",
        "CRPIX1  =                  1.0",
        "CRPIX2  =                  1.0",
        "CRVAL1  =                  0.0",
        "CRVAL2  =                  0.0",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let wcs = open_image(&cards);
    assert!(!wcs.is_celestial());
    let fp = wcs.footprint().unwrap();
    assert_eq!(fp.len(), 4 * 2);
    for (got, (px, py)) in
        fp.as_chunks::<2>()
            .0
            .iter()
            .zip([(0.0, 0.0), (99.0, 0.0), (99.0, 99.0), (0.0, 99.0)])
    {
        let want = wcs.pixel_to_world(&[px, py]).unwrap();
        assert!(near(got[0], want[0], 1e-12) && near(got[1], want[1], 1e-12));
    }
}

/// A three-axis WCS yields 2^3 corners, still one axis apart between
/// consecutive entries.
#[test]
fn footprint_covers_every_axis_of_a_cube() {
    let cards: Vec<String> = [
        "CTYPE1  = 'RA---TAN'",
        "CTYPE2  = 'DEC--TAN'",
        "CTYPE3  = 'FREQ'",
        "CRPIX1  =                 32.0",
        "CRPIX2  =                 24.0",
        "CRPIX3  =                  5.0",
        "CRVAL1  =              202.469",
        "CRVAL2  =               47.195",
        "CRVAL3  =              1.4E+09",
        "CDELT1  =               -0.001",
        "CDELT2  =                0.001",
        "CDELT3  =              1.0E+06",
        "CUNIT3  = 'Hz'",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let wcs = open_image_3d(&cards);
    let fp = wcs.footprint().unwrap();
    assert_eq!(fp.len(), 8 * 3, "2^3 corners of 3 values each");
    // The spectral axis must actually vary: a celestial-only footprint
    // would hold it at the reference plane for every corner.
    let freqs: Vec<f64> = fp.as_chunks::<3>().0.iter().map(|c| c[2]).collect();
    let lo = freqs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = freqs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(hi - lo > 0.0, "spectral axis is constant across corners");
}

/// The corner count doubles per axis, and `WCSAXES` comes from the
/// header, so a file claiming many axes must be refused rather than
/// allowed to drive an unbounded allocation.
#[test]
fn footprint_refuses_an_absurd_axis_count() {
    let over = fitsy::Wcs::MAX_FOOTPRINT_AXES + 1;
    let mut cards = tan_cards();
    cards.push(format!("WCSAXES = {over:20}"));
    let wcs = open_image(&cards);
    assert_eq!(wcs.naxis(), over, "header should describe {over} axes");
    let err = wcs.footprint().unwrap_err().to_string();
    assert!(err.contains("corners"), "unexpected message: {err}");

    // One axis below the limit still works, so the guard is not simply
    // rejecting every wide WCS. The image is 2-D, so the other 14 axes
    // are degenerate. The corner count follows the image rather than
    // `WCSAXES`: 4 corners, each a full 16-value world vector.
    let mut ok_cards = tan_cards();
    ok_cards.push(format!("WCSAXES = {:20}", fitsy::Wcs::MAX_FOOTPRINT_AXES));
    let wide = open_image(&ok_cards);
    let fp = wide.footprint().unwrap();
    assert_eq!(fp.len(), 4 * fitsy::Wcs::MAX_FOOTPRINT_AXES);
}

/// `WCSAXESa` may exceed `NAXIS` (Sec.8.2). A coordinate axis past the
/// end of the image shape then has no length. `footprint` holds that
/// axis at its reference pixel instead of returning an error.
#[test]
fn footprint_holds_a_degenerate_axis_at_its_reference_pixel() {
    let mut cards = tan_cards();
    cards.push("WCSAXES =                    3".into());
    cards.push("CTYPE3  = 'FREQ'".into());
    cards.push("CRPIX3  =                  5.0".into());
    cards.push("CRVAL3  =              1.4E+09".into());
    cards.push("CDELT3  =              1.0E+06".into());
    cards.push("CUNIT3  = 'Hz'".into());
    // A 2-D image describing 3 WCS axes. NAXIS3 is absent, so the
    // spectral axis has no length.
    let wcs = open_image(&cards);
    assert_eq!(wcs.naxis(), 3);
    assert_eq!(wcs.pixel_shape.as_deref(), Some([100_u64, 100].as_slice()));

    let fp = wcs.footprint().unwrap();
    // 2^2 corners -- the image, not `WCSAXES` -- of 3 values each.
    assert_eq!(fp.len(), 4 * 3);
    // The degenerate axis sits at CRVAL3 for every corner, which is
    // where its reference pixel lands.
    for corner in fp.as_chunks::<3>().0 {
        assert!(
            near(corner[2], 1.4e9, 1e-6),
            "degenerate axis moved: {}",
            corner[2]
        );
    }
    // And the two covered axes still walk the image corners.
    for (got, (px, py)) in
        fp.as_chunks::<3>()
            .0
            .iter()
            .zip([(0.0, 0.0), (99.0, 0.0), (99.0, 99.0), (0.0, 99.0)])
    {
        let want = wcs.pixel_to_world(&[px, py, 4.0]).unwrap();
        assert!(near(got[0], want[0], 1e-12) && near(got[1], want[1], 1e-12));
    }
}

/// A corner outside the domain of the projection comes back `NaN`.
/// `footprint` runs the batch transform, so it follows the batch rule.
#[test]
fn footprint_marks_out_of_domain_corners_nan() {
    // 100 px at 1 deg/px puts every corner outside the SIN domain.
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---SIN'".into(),
        "CTYPE2  = 'DEC--SIN'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                  0.0".into(),
        "CRVAL2  =                  0.0".into(),
        "CDELT1  =                 -1.0".into(),
        "CDELT2  =                  1.0".into(),
    ];
    let wcs = open_image(&cards);
    assert!(
        wcs.pixel_to_world(&[0.0, 0.0]).is_err(),
        "test needs a corner outside the SIN domain"
    );
    let fp = wcs.footprint().unwrap();
    assert_eq!(fp.len(), 4 * 2);
    assert!(fp.iter().all(|v| v.is_nan()), "expected all NaN: {fp:?}");
}

#[test]
fn wcs_is_cloneable_and_the_clone_is_independent() {
    // Downstream crates need to hold a `Wcs` by value. The projection
    // is shared via `Arc` (it is immutable after parse), so a clone
    // costs a refcount bump rather than a re-parse.
    let wcs = open_image(&tan_cards());
    let mut copy = wcs.clone();
    let a = sky(&wcs, 10.0, 20.0);
    let b = sky(&copy, 10.0, 20.0);
    assert!(near(a.0, b.0, 1e-12) && near(a.1, b.1, 1e-12));

    // Mutating the clone must not disturb the original.
    copy.pixel_shape = Some(vec![7, 9]);
    copy.wcsname = Some("copy".into());
    assert_eq!(wcs.pixel_shape.as_deref(), Some([100_u64, 100].as_slice()));
    assert!(wcs.wcsname.is_none());
    // Flat layout: 2^2 corners of 2 values each.
    assert_eq!(copy.footprint().unwrap().len(), 4 * 2);
}

#[test]
fn cloning_a_sip_wcs_preserves_distortion() {
    let mut cards = tan_cards();
    cards[0] = "CTYPE1  = 'RA---TAN-SIP'".to_string();
    cards[1] = "CTYPE2  = 'DEC--TAN-SIP'".to_string();
    for c in [
        "A_ORDER =                    2",
        "B_ORDER =                    2",
        "A_2_0   =              1.0E-05",
        "B_0_2   =              2.0E-05",
    ] {
        cards.push(c.to_string());
    }
    let wcs = open_image(&cards);
    assert!(wcs.celestial.as_ref().unwrap().sip.is_some());
    let copy = wcs.clone();
    let a = sky(&wcs, 80.0, 15.0);
    let b = sky(&copy, 80.0, 15.0);
    assert!(near(a.0, b.0, 1e-12) && near(a.1, b.1, 1e-12));
}

/// A single out-of-domain point must not discard the rest of a batch.
///
/// SIN projects only the hemisphere facing the observer, so a
/// wide-CDELT frame mixes valid and invalid pixels. The established
/// convention is to mark the invalid ones NaN and return the rest;
/// fitsy used to propagate the first error and throw the batch away.
#[test]
fn batch_transforms_nan_fill_out_of_domain_points() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---SIN'".into(),
        "CTYPE2  = 'DEC--SIN'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                 45.0".into(),
        "CRVAL2  =                 30.0".into(),
        // 2 deg/pixel over 100 pixels: the corners sit ~140 deg from
        // the reference, well outside the hemisphere SIN can project.
        "CDELT1  =                 -2.0".into(),
        "CDELT2  =                  2.0".into(),
        "CUNIT1  = 'deg'".into(),
        "CUNIT2  = 'deg'".into(),
    ];
    let wcs = open_image(&cards);

    // Reference pixel is valid; (0,0) is far outside the domain.
    assert!(wcs.pixel_to_world(&[49.0, 49.0]).is_ok());
    assert!(
        wcs.pixel_to_world(&[0.0, 0.0]).is_err(),
        "single-point calls keep reporting the diagnostic"
    );

    let pts = [49.0, 49.0, 0.0, 0.0, 50.0, 49.0];
    let out = wcs
        .pixel_to_world_many(&pts)
        .expect("one bad point must not fail the batch");
    assert_eq!(out.len(), 6);
    assert!(out[0].is_finite() && out[1].is_finite());
    assert!(out[2].is_nan() && out[3].is_nan());
    assert!(out[4].is_finite() && out[5].is_finite());

    // Same on the inverse: a sky position in the far hemisphere.
    let far = [45.0, 30.0, 225.0, -30.0, 45.5, 30.0];
    let back = wcs
        .world_to_pixel_many(&far)
        .expect("one bad point must not fail the batch");
    assert_eq!(back.len(), 6);
    assert!(back[0].is_finite() && back[1].is_finite());
    assert!(back[2].is_nan() && back[3].is_nan());
    assert!(back[4].is_finite() && back[5].is_finite());
}

/// A WCS with no celestial pair transforms fine: the general batch
/// path has no celestial requirement. Only `pixel_scale_at`, which
/// measures on the sphere, still needs the pair.
#[test]
fn batch_transforms_do_not_require_a_celestial_pair() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'WAVE'".into(),
        "CTYPE2  = 'TIME'".into(),
        "CRPIX1  =                  1.0".into(),
        "CRPIX2  =                  1.0".into(),
        "CRVAL1  =                  0.0".into(),
        "CRVAL2  =                  0.0".into(),
        "CDELT1  =                  1.0".into(),
        "CDELT2  =                  1.0".into(),
    ];
    let wcs = open_image(&cards);
    assert!(wcs.pixel_to_world_many(&[0.0, 0.0]).is_ok());
    assert!(wcs.world_to_pixel_many(&[0.0, 0.0]).is_ok());
    assert!(wcs.pixel_scale_at(0.0, 0.0).is_err());
}

/// The descriptive keywords fitsy retains must survive `to_header` ->
/// re-parse. They are stored rather than applied, so writing them back
/// *is* their whole contract -- a dropped `SPECSYS` silently changes
/// which frame the coordinates claim to be in. The round-trip contract
/// is `from_header(to_header(w)) == w`, so the header here carries a
/// spectral axis for the frame to describe.
#[test]
fn to_header_round_trips_frame_and_axis_name_keywords() {
    let cards: Vec<String> = [
        "CTYPE1  = 'RA---TAN'",
        "CTYPE2  = 'DEC--TAN'",
        "CTYPE3  = 'VOPT-F2W'",
        "CRVAL1  =                150.0",
        "CRVAL2  =                  2.5",
        "CRVAL3  =               1.0E+05",
        "CRPIX1  =                 10.0",
        "CRPIX2  =                 20.0",
        "CRPIX3  =                  1.0",
        "CDELT1  =                -0.01",
        "CDELT2  =                 0.01",
        "CDELT3  =               1.0E+03",
        "CUNIT3  = 'm/s'",
        "RESTFRQ =        1.42040575E+09",
        "CNAME1  = 'Right ascension'",
        "CNAME2  = 'Declination'",
        "CNAME3  = 'Optical velocity'",
        "SPECSYS = 'BARYCENT'",
        "SSYSOBS = 'TOPOCENT'",
        "SSYSSRC = 'LSRK    '",
        "VELOSYS =              -12345.0",
        "ZSOURCE =                0.0123",
        "VELANGL =                  75.0",
    ]
    .iter()
    .map(ToString::to_string)
    .collect();

    let wcs = open_image_3d(&cards);
    let frame = wcs.spectral_frame.clone().expect("frame retained");
    assert_eq!(frame.specsys.as_deref(), Some("BARYCENT"));
    assert_eq!(frame.ssysobs.as_deref(), Some("TOPOCENT"));
    assert_eq!(frame.velosys, Some(-12345.0));
    let source = frame.source.expect("ZSOURCE group retained");
    assert_eq!(source.zsource, 0.0123);
    assert_eq!(source.ssyssrc.as_deref(), Some("LSRK"));
    assert_eq!(source.velangl, Some(75.0));
    assert_eq!(wcs.axes()[0].cname.as_deref(), Some("Right ascension"));
    assert_eq!(wcs.axes()[1].cname.as_deref(), Some("Declination"));
    assert_eq!(wcs.axes()[2].cname.as_deref(), Some("Optical velocity"));

    let written = wcs.to_header(' ').unwrap();
    let reparsed = fitsy::Wcs::from_header(&written, ' ').unwrap().unwrap();
    assert_eq!(reparsed.spectral_frame, wcs.spectral_frame);
    assert_eq!(reparsed.axes(), wcs.axes());
}

/// `SSYSSRC` names the frame `ZSOURCE` is expressed in and `VELANGL`
/// orients its velocity vector; without `ZSOURCE` neither describes
/// anything and neither is retained (Sec.8.4.3).
#[test]
fn source_frame_satellites_require_zsource() {
    let cards: Vec<String> = [
        "CTYPE3  = 'FREQ'",
        "CRVAL3  =               1.0E+09",
        "CRPIX3  =                  1.0",
        "CDELT3  =               1.0E+06",
        "CUNIT3  = 'Hz'",
        "SSYSSRC = 'LSRK    '",
        "VELANGL =                  75.0",
        "SPECSYS = 'BARYCENT'",
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    let wcs = open_image_3d(&cards);
    let frame = wcs.spectral_frame.expect("SPECSYS keeps the frame");
    assert_eq!(frame.specsys.as_deref(), Some("BARYCENT"));
    assert!(frame.source.is_none(), "no ZSOURCE, no source group");
}

/// The same keywords in their table-resident spellings (Table 22):
/// `SPECna`, `SOBSna`, `SSRCna`, `VSYSna`, `ZSOUna`, `VANGna`,
/// `TCNAna`.
#[test]
fn pixel_list_carries_the_source_frame_keywords() {
    use fitsy::header::Header;
    use fitsy::wcs::TableWcs;

    let mut buf = Vec::new();
    for c in [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                    8",
        "NAXIS2  =                    1",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    1",
        "TCTYP1  = 'VOPT-F2W'",
        "TCRVL1  =              1.0E+05",
        "TCRPX1  =                  1.0",
        "TCDLT1  =              1.0E+03",
        "TCUNI1  = 'm/s     '",
        "TCNA1   = 'Optical velocity'",
        "RFRQ1   =        1.42040575E+09",
        "SPEC1   = 'SOURCE  '",
        "SOBS1   = 'TOPOCENT'",
        "SSRC1   = 'LSRK    '",
        "VSYS1   =              -12345.0",
        "ZSOU1   =                0.0123",
        "VANG1   =                  75.0",
        "END",
    ] {
        buf.extend_from_slice(&pad_card(c));
    }
    while !buf.len().is_multiple_of(BLOCK) {
        buf.push(b' ');
    }
    let h = Header::parse(&buf, 0).unwrap().0;
    let tw = TableWcs::from_pixel_list(&h, ' ').unwrap().unwrap();
    let frame = tw
        .wcs
        .spectral_frame
        .clone()
        .expect("VOPT axis keeps the frame");
    assert_eq!(frame.specsys.as_deref(), Some("SOURCE"));
    assert_eq!(frame.ssysobs.as_deref(), Some("TOPOCENT"));
    assert_eq!(frame.velosys, Some(-12345.0));
    let source = frame.source.expect("ZSOURCE group");
    assert_eq!(source.zsource, 0.0123);
    assert_eq!(source.ssyssrc.as_deref(), Some("LSRK"));
    assert_eq!(source.velangl, Some(75.0));
    assert_eq!(tw.wcs.axes()[0].cname.as_deref(), Some("Optical velocity"));
}

/// Every projection, with its parameters and both pole conventions,
/// must survive `to_header` -> re-parse unchanged.
///
/// Regression: `to_header` used to drop the projection's `PVi_m`
/// table and LONPOLE/LATPOLE entirely. Parameterless projections
/// looked fine, which is why it went unnoticed; the parameterized
/// ones either drifted by degrees or failed to re-parse at all.
#[test]
fn to_header_round_trips_every_projection() {
    /// `(code, extra PV2 cards)` -- the parameters each projection
    /// needs to be well-defined.
    const CASES: &[(&str, &[(&str, f64)])] = &[
        ("AZP", &[("PV2_1", 2.0), ("PV2_2", 30.0)]),
        ("SZP", &[("PV2_1", 2.0), ("PV2_2", 180.0), ("PV2_3", 60.0)]),
        ("TAN", &[]),
        ("STG", &[]),
        ("SIN", &[("PV2_1", 0.3), ("PV2_2", -0.2)]),
        ("ARC", &[]),
        ("ZPN", &[("PV2_1", 1.0), ("PV2_3", 220.0)]),
        ("ZEA", &[]),
        ("AIR", &[("PV2_1", 45.0)]),
        // CYP with mu = 1, lambda = 1/sqrt(2) is the Gall
        // stereographic case from Paper II Sec.5.2.1.
        (
            "CYP",
            &[("PV2_1", 1.0), ("PV2_2", std::f64::consts::FRAC_1_SQRT_2)],
        ),
        ("CEA", &[("PV2_1", 0.75)]),
        ("CAR", &[]),
        ("MER", &[]),
        ("SFL", &[]),
        ("PAR", &[]),
        ("MOL", &[]),
        ("AIT", &[]),
        ("COP", &[("PV2_1", 45.0), ("PV2_2", 25.0)]),
        ("COE", &[("PV2_1", 45.0), ("PV2_2", 25.0)]),
        ("COD", &[("PV2_1", 45.0), ("PV2_2", 25.0)]),
        ("COO", &[("PV2_1", 45.0), ("PV2_2", 25.0)]),
        ("BON", &[("PV2_1", 45.0)]),
        ("PCO", &[]),
        ("TSC", &[]),
        ("CSC", &[]),
        ("QSC", &[]),
        ("HPX", &[("PV2_1", 4.0), ("PV2_2", 3.0)]),
        ("XPH", &[]),
    ];
    // `None` exercises the Paper II defaults; the explicit values
    // exercise the branch-selection LATPOLE controls, which matters
    // for every theta0 = 0 projection.
    const POLES: &[(Option<f64>, Option<f64>)] = &[
        (None, None),
        (Some(150.0), None),
        (None, Some(-30.0)),
        (Some(150.0), Some(-30.0)),
    ];

    let probes = [
        (0.0, 0.0),
        (25.0, 60.0),
        (99.0, 99.0),
        (150.0, 30.0),
        (199.0, 199.0),
    ];
    let mut checked = 0;
    for (code, pvs) in CASES {
        for (lonpole, latpole) in POLES {
            let mut cards: Vec<String> = vec![
                format!("CTYPE1  = 'RA---{code}'"),
                format!("CTYPE2  = 'DEC--{code}'"),
                "CRPIX1  =                 50.5".into(),
                "CRPIX2  =                 40.25".into(),
                "CRVAL1  =                 45.0".into(),
                "CRVAL2  =                 30.0".into(),
                "CDELT1  =                -0.05".into(),
                "CDELT2  =                 0.05".into(),
                "CUNIT1  = 'deg'".into(),
                "CUNIT2  = 'deg'".into(),
            ];
            for (k, v) in *pvs {
                cards.push(format!("{k:<8}= {v:>20}"));
            }
            if let Some(lp) = lonpole {
                cards.push(format!("LONPOLE = {lp:>20}"));
            }
            if let Some(tp) = latpole {
                cards.push(format!("LATPOLE = {tp:>20}"));
            }

            // Not every (projection, pole) pair is a legal WCS --
            // some have no native-pole solution at all. Those are
            // invalid *inputs*, not serialization failures.
            let Some(truth) = try_open_image(&cards) else {
                continue;
            };
            let header = truth
                .to_header(' ')
                .unwrap_or_else(|e| panic!("{code} lonpole={lonpole:?} latpole={latpole:?}: {e}"));
            let round = fitsy::Wcs::from_header(&header, ' ')
                .unwrap_or_else(|e| {
                    panic!("{code} lonpole={lonpole:?} latpole={latpole:?} re-parse: {e}")
                })
                .expect("serialized header still describes a WCS");

            let mut compared = 0;
            for (px, py) in probes {
                let (Ok(w1), Ok(w2)) = (
                    truth.pixel_to_world(&[px, py]),
                    round.pixel_to_world(&[px, py]),
                ) else {
                    // Both must agree on being out of domain.
                    assert_eq!(
                        truth.pixel_to_world(&[px, py]).is_ok(),
                        round.pixel_to_world(&[px, py]).is_ok(),
                        "{code} lonpole={lonpole:?} latpole={latpole:?}: domain differs at ({px},{py})"
                    );
                    continue;
                };
                let ((ra1, de1), (ra2, de2)) = ((w1[0], w1[1]), (w2[0], w2[1]));
                assert!(
                    (ra1 - ra2).abs() < 1e-11 && (de1 - de2).abs() < 1e-11,
                    "{code} lonpole={lonpole:?} latpole={latpole:?} drifted at ({px},{py}): \
                     ({ra1},{de1}) vs ({ra2},{de2})"
                );
                compared += 1;
            }
            assert!(
                compared > 0,
                "{code} lonpole={lonpole:?} latpole={latpole:?}: every probe was out of \
                 domain, so the comparison proved nothing"
            );
            checked += 1;
        }
    }
    // Guard against the skip above quietly hollowing the test out.
    assert!(
        checked >= CASES.len() * 2,
        "only {checked} configurations were actually round-tripped"
    );
}

/// Paper II eq. (9) has two `delta_p` roots and `LATPOLE` picks
/// between them. Getting that choice wrong mirrors the sky by several
/// degrees while still round-tripping perfectly through our own
/// serializer, so it needs pinning explicitly.
///
/// Two regressions are pinned here:
///
/// 1. The candidates were not wrapped into `(-pi, pi]`, so with
///    `LONPOLE = 180` one root came out near `345deg` and was
///    discarded as "outside [-90, 90]" -- it is the `-15deg`
///    solution. The surviving root mirrors the sky: 21000 arcsec of
///    error on the `CYP` header this was found with.
/// 2. An exact tie (`LATPOLE` equidistant from both roots, e.g. 0
///    with roots at `+/-60`) went to `arg + acos`; the established
///    convention takes `arg - acos`.
#[test]
fn native_pole_root_selection_resolves_the_documented_root() {
    // (crval2, lonpole, latpole, expected LATPOLE that `to_header`
    // should emit). Each is the root Paper II eq. (9) selects for
    // that LATPOLE, cross-checked against a reference implementation.
    const CASES: &[(f64, f64, Option<f64>, f64)] = &[
        // Wrapping: LONPOLE = 180 puts `arg + acos` past pi.
        (-75.0, 180.0, Some(-90.0), -15.0),
        (-75.0, 180.0, Some(-30.0), -15.0),
        // Exact tie, roots at +/-60. Only ties with a stable answer
        // are pinned: the standard leaves the equidistant case to
        // Paper II, and reference implementations decide it on their
        // own rounding (nudging CRVAL2 by 1e-13 deg flips the answer
        // for `crval2 = -30, LONPOLE = 180`), so there is no rule to
        // match there.
        (30.0, 0.0, Some(0.0), -60.0),
        (-45.0, 180.0, Some(0.0), 45.0),
        // Just off the tie, either side.
        (30.0, 0.0, Some(1.0), 60.0),
        (30.0, 0.0, Some(-1.0), -60.0),
        // Absent LATPOLE defaults to +90, so the northerly root wins.
        (30.0, 0.0, None, 60.0),
        (-30.0, 180.0, None, 60.0),
    ];
    for &(crval2, lonpole, latpole, want) in CASES {
        let mut cards: Vec<String> = vec![
            "CTYPE1  = 'RA---CAR'".into(),
            "CTYPE2  = 'DEC--CAR'".into(),
            "CRPIX1  =                  4.0".into(),
            "CRPIX2  =                  4.0".into(),
            "CRVAL1  =                 45.0".into(),
            format!("CRVAL2  = {crval2:>20}"),
            "CDELT1  =                -0.05".into(),
            "CDELT2  =                 0.05".into(),
            "CUNIT1  = 'deg'".into(),
            "CUNIT2  = 'deg'".into(),
            format!("LONPOLE = {lonpole:>20}"),
        ];
        if let Some(lp) = latpole {
            cards.push(format!("LATPOLE = {lp:>20}"));
        }
        let wcs = open_image(&cards);
        let got = wcs.to_header(' ').unwrap();
        let Some(fitsy::Value::Real(got)) = got.first("LATPOLE") else {
            panic!("to_header emitted no LATPOLE");
        };
        assert!(
            (got - want).abs() < 1e-9,
            "crval2={crval2} lonpole={lonpole} latpole={latpole:?}: \
             resolved LATPOLE {got}, expected {want}"
        );
    }
}

/// A spectral axis may specify its rest quantity as `RESTWAV`
/// *or* `RESTFRQ`; `SpectralAxis::new` accepts either. The transform
/// code then demanded `RESTFRQ` specifically in two places.
///
/// Regression: `WAVE-V2W` with only `RESTWAV` **panicked** (an
/// `.expect("validated in new()")` on `restfrq`), and the inverse
/// returned a spurious "RESTFRQ required" error. Both now derive the
/// rest frequency from whichever quantity was supplied.
#[test]
fn spectral_accepts_restwav_alone() {
    for ctype in ["WAVE-V2W", "FREQ-V2F"] {
        let cards: Vec<String> = vec![
            "CTYPE1  = 'RA---TAN'".into(),
            "CTYPE2  = 'DEC--TAN'".into(),
            format!("CTYPE3  = '{ctype}'"),
            "CRPIX1  =                  2.0".into(),
            "CRPIX2  =                  2.0".into(),
            "CRPIX3  =                  8.0".into(),
            "CRVAL1  =                 45.0".into(),
            "CRVAL2  =                 30.0".into(),
            if ctype.starts_with("WAVE") {
                "CRVAL3  =                 0.21".into()
            } else {
                "CRVAL3  =              1.4E+09".into()
            },
            "CDELT1  =              -1.0E-03".into(),
            "CDELT2  =               1.0E-03".into(),
            if ctype.starts_with("WAVE") {
                "CDELT3  =               1.0E-06".into()
            } else {
                "CDELT3  =               1.0E+04".into()
            },
            "CUNIT1  = 'deg'".into(),
            "CUNIT2  = 'deg'".into(),
            if ctype.starts_with("WAVE") {
                "CUNIT3  = 'm'".into()
            } else {
                "CUNIT3  = 'Hz'".into()
            },
            // RESTWAV only -- no RESTFRQ anywhere in the header.
            "RESTWAV =    0.211061140542".into(),
        ];
        let wcs = open_image_3d(&cards);
        let forward = wcs
            .pixel_to_world(&[1.0, 1.0, 3.0])
            .unwrap_or_else(|e| panic!("{ctype} forward with RESTWAV only: {e}"));
        assert!(
            forward[2].is_finite(),
            "{ctype}: spectral axis came back non-finite"
        );
        // And the inverse, which had its own RESTFRQ-only guard.
        let back = wcs
            .world_to_pixel(&[forward[0], forward[1], forward[2]])
            .unwrap_or_else(|e| panic!("{ctype} inverse with RESTWAV only: {e}"));
        assert!(
            (back[2] - 3.0).abs() < 1e-6,
            "{ctype}: spectral round-trip gave pixel {}, want 3",
            back[2]
        );
    }
}

/// A spectral axis must survive `to_header` -> re-parse.
///
/// `to_header` used to refuse outright for anything spectral; now it
/// writes the rest quantity and leans on CTYPE/CUNIT/CRVAL for the
/// rest. This pins that the four together really do reconstruct the
/// axis, for both the RESTFRQ and the RESTWAV spellings.
#[test]
fn to_header_round_trips_spectral_axes() {
    for (ctype, cunit, crval, cdelt, rest) in [
        (
            "VOPT-F2W",
            "m/s",
            "1.0E+04",
            "1.0E+03",
            "RESTFRQ =        1.420405752E9",
        ),
        (
            "WAVE-V2W",
            "m",
            "0.21",
            "1.0E-06",
            "RESTWAV =    0.211061140542",
        ),
        (
            "FREQ-V2F",
            "Hz",
            "1.4E+09",
            "1.0E+04",
            "RESTWAV =    0.211061140542",
        ),
    ] {
        let cards: Vec<String> = vec![
            "CTYPE1  = 'RA---TAN'".into(),
            "CTYPE2  = 'DEC--TAN'".into(),
            format!("CTYPE3  = '{ctype}'"),
            "CRPIX1  =                  2.0".into(),
            "CRPIX2  =                  2.0".into(),
            "CRPIX3  =                  8.0".into(),
            "CRVAL1  =                 45.0".into(),
            "CRVAL2  =                 30.0".into(),
            format!("CRVAL3  = {crval:>20}"),
            "CDELT1  =              -1.0E-03".into(),
            "CDELT2  =               1.0E-03".into(),
            format!("CDELT3  = {cdelt:>20}"),
            "CUNIT1  = 'deg'".into(),
            "CUNIT2  = 'deg'".into(),
            format!("CUNIT3  = '{cunit}'"),
            rest.into(),
        ];
        let truth = open_image_3d(&cards);
        let header = truth
            .to_header(' ')
            .unwrap_or_else(|e| panic!("{ctype}: to_header: {e}"));
        let round = fitsy::Wcs::from_header(&header, ' ')
            .unwrap_or_else(|e| panic!("{ctype}: re-parse: {e}"))
            .unwrap_or_else(|| panic!("{ctype}: re-parse found no WCS"));
        for p in [1.0_f64, 4.0, 8.0, 15.0] {
            let a = truth.pixel_to_world(&[1.0, 1.0, p]).unwrap();
            let b = round.pixel_to_world(&[1.0, 1.0, p]).unwrap();
            assert!(
                a[2].is_finite() && (a[2] - b[2]).abs() <= 1e-9 * a[2].abs().max(1.0),
                "{ctype}: spectral value drifted at pixel {p}: {} vs {}",
                a[2],
                b[2]
            );
        }
    }
}

/// A DSS plate solution must survive `to_header` -> re-parse.
///
/// The plate model bypasses the standard pipeline entirely, so the
/// serialized header carries both a placeholder TAN *and* the
/// `PLT*`/`AMD*` family; re-parsing has to pick the plate model back
/// up, and the sexagesimal `PLTRAH/M/S` round trip has to be exact
/// enough not to move the plate center.
#[test]
fn to_header_round_trips_dss_plate_solution() {
    let path = test_data_dir().join("dss_plate.fits");
    let bytes = std::fs::read(&path).expect("dss_plate.fits");
    let file = FitsFile::from_bytes(bytes).unwrap();
    let truth = file.wcs(0, ' ').unwrap().expect("DSS header has a WCS");
    assert!(truth.dss.is_some(), "fixture is not a DSS plate");

    let header = truth.to_header(' ').expect("to_header");
    let round = fitsy::Wcs::from_header(&header, ' ')
        .expect("re-parse")
        .expect("re-parse found no WCS");
    assert!(
        round.dss.is_some(),
        "plate solution dropped; the re-parsed WCS fell back to the placeholder TAN"
    );
    for &(px, py) in &[
        (1.0_f64, 1.0_f64),
        (500.0, 500.0),
        (1060.0, 1060.0),
        (2000.0, 2000.0),
        (100.0, 2000.0),
    ] {
        let a = truth.pixel_to_world(&[px, py]).unwrap();
        let b = round.pixel_to_world(&[px, py]).unwrap();
        assert!(
            near(a[0], b[0], 1e-9) && near(a[1], b[1], 1e-9),
            "DSS round-trip drifted at ({px},{py}): {a:?} vs {b:?}"
        );
    }
}

/// `-TAB` lookups must stop extrapolating half a sample step past the
/// table, per Paper III Sec.6.1.2 ("the value of `Upsilon_m` derived
/// from `psi_m` must lie in the range `0.5 <= Upsilon_m <= K + 0.5`").
///
/// Regression: extrapolation was unbounded, so a pixel well outside
/// the table received a confidently wrong coordinate instead of an
/// error.
#[test]
fn tab_axis_extrapolation_is_bounded() {
    use fitsy::{BinFieldKind, BinTableBuilder, FitsWriter, ImageBuilder, Value};

    let wavelens: [f64; 5] = [4000.0, 4500.0, 5500.0, 7000.0, 9000.0];
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
        ("CRVAL3", Value::Real(1.0)),
        ("CDELT1", Value::Real(1.0)),
        ("CDELT2", Value::Real(1.0)),
        ("CDELT3", Value::Real(1.0)),
        ("PS3_0", Value::String("WCS-TAB".into())),
        ("PS3_1", Value::String("WAVELEN".into())),
        ("PV3_1", Value::Integer(1)),
    ] {
        primary = primary.card(k, v, None);
    }
    let primary = primary.build().unwrap();

    let mut bt = BinTableBuilder::new();
    bt.add_column("WAVELEN", BinFieldKind::F64, 5, Some("Angstrom"), None)
        .unwrap();
    let mut row = Vec::new();
    for w in wavelens {
        row.extend_from_slice(&w.to_bits().to_be_bytes());
    }
    let (mut th, td) = bt.build(1, row).unwrap();
    th.push("EXTNAME", Value::String("WCS-TAB".into()), None)
        .unwrap();
    th.push("EXTVER", Value::Integer(1), None).unwrap();

    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu(&primary.0, &primary.1).unwrap();
    w.write_hdu(&th, &td).unwrap();
    w.finish().unwrap();
    let file = FitsFile::from_bytes(buf).unwrap();
    let wcs = file.wcs(0, ' ').unwrap().expect("WCS present");

    // 0-based pixel p maps to psi = 1 + p, so the table spans p in
    // 0..=4 and the permitted half-step margin reaches p = -0.5 and
    // p = 4.5.
    for p in [0.0, 2.0, 4.0, -0.5, 4.5, -0.4999, 4.4999] {
        assert!(
            wcs.pixel_to_world(&[0.0, 0.0, p]).is_ok(),
            "pixel {p} is inside the table or its half-step margin"
        );
    }
    for p in [-0.5001, 4.5001, -3.0, 9.0] {
        assert!(
            wcs.pixel_to_world(&[0.0, 0.0, p]).is_err(),
            "pixel {p} is outside the table and must not be extrapolated"
        );
    }
    // The margin endpoints extrapolate linearly from the end segment.
    let lo = wcs.pixel_to_world(&[0.0, 0.0, -0.5]).unwrap()[2];
    assert!(
        (lo - 3750.0).abs() < 1e-9,
        "lower margin should extrapolate to 3750, got {lo}"
    );
}

// -- Spectral axis conformance (Paper III Sec.3.3, Standard Sec.8.4) --

fn spectral_cube_cards(ctype3: &str, extra: &[&str]) -> Vec<String> {
    spectral_cube_cards_at(ctype3, 1000.0, 100.0, extra)
}

/// [`spectral_cube_cards`] with an explicit reference value and
/// increment on axis 3, for the codes where the numbers have to be
/// physically sensible (a wavelength axis cannot sit at 1000 m).
fn spectral_cube_cards_at(ctype3: &str, crval3: f64, cdelt3: f64, extra: &[&str]) -> Vec<String> {
    let mut cards = vec![
        "CTYPE1  = 'RA---TAN'".to_string(),
        "CTYPE2  = 'DEC--TAN'".to_string(),
        format!("CTYPE3  = '{ctype3}'"),
        "CRVAL1  =                  0.0".to_string(),
        "CRVAL2  =                  0.0".to_string(),
        format!("CRVAL3  = {crval3:>20E}"),
        "CRPIX1  =                  1.0".to_string(),
        "CRPIX2  =                  1.0".to_string(),
        "CRPIX3  =                  1.0".to_string(),
        "CDELT1  =                -0.01".to_string(),
        "CDELT2  =                 0.01".to_string(),
        format!("CDELT3  = {cdelt3:>20E}"),
    ];
    cards.extend(extra.iter().map(|s| (*s).to_string()));
    cards
}

/// Paper III Sec.3.3.4 requires RESTFRQ/RESTWAV only for the
/// `F2V`/`V2F`/`W2V`/`V2W`/`A2V`/`V2A` codes, and Standard Sec.8.4
/// makes writing one a `should`. A *linear* velocity axis is
/// `S = CRVAL + x` and needs neither.
///
/// Regression: the whole WCS parse failed for a plain `CTYPE3='VRAD'`
/// cube with no rest quantity, so such files had no WCS at all --
/// including their perfectly ordinary celestial axes.
#[test]
fn linear_velocity_axis_parses_without_rest_frequency() {
    // `ZOPT` and `BETA` are ratios, so Table 25 gives them no unit;
    // declaring `m/s` there is a header error the unit check now
    // catches.
    for (ctype, cunit) in [
        ("VRAD", "CUNIT3  = 'm/s'"),
        ("VOPT", "CUNIT3  = 'm/s'"),
        ("ZOPT", "CUNIT3  = ' '"),
        ("VELO", "CUNIT3  = 'm/s'"),
        ("BETA", "CUNIT3  = ' '"),
    ] {
        let cards = spectral_cube_cards(ctype, &[cunit]);
        let wcs = wcs_3d_result(&cards)
            .unwrap_or_else(|e| panic!("{ctype} without RESTFRQ was rejected: {e}"))
            .expect("wcs present");
        assert!(wcs.is_celestial(), "{ctype}: celestial pair lost");
        // Linear: CRVAL3 + 3 * CDELT3 at the fourth plane.
        let world = wcs.pixel_to_world(&[0.0, 0.0, 3.0]).unwrap();
        assert!(
            near(world[2], 1300.0, 1e-9),
            "{ctype}: expected 1300, got {}",
            world[2]
        );
    }
}

/// The `F2V` family still needs a rest quantity, and the failure must
/// arrive at parse time rather than on the first transform.
#[test]
fn velocity_algorithm_without_rest_quantity_is_rejected() {
    let cards = spectral_cube_cards("VOPT-F2W", &["CUNIT3  = 'm/s'"]);
    let err = wcs_3d_result(&cards).expect_err("VOPT-F2W needs RESTWAV");
    assert!(
        err.to_string().contains("RESTFRQ or RESTWAV"),
        "unexpected error: {err}"
    );
    // With RESTWAV it parses.
    let cards = spectral_cube_cards(
        "VOPT-F2W",
        &["CUNIT3  = 'm/s'", "RESTWAV =    0.2110611410"],
    );
    assert!(wcs_3d_result(&cards).unwrap().is_some());
}

/// A code that is not in Table 26 at all must be rejected.
///
/// Regression: an unrecognized algorithm code fell through to the
/// linear pipeline, so the axis silently reported a coordinate that
/// was wrong by orders of magnitude instead of failing.
#[test]
fn unregistered_spectral_algorithm_is_rejected_not_linearized() {
    for ctype in ["FREQ-XYZ", "WAVE-ABC", "AWAV-Q2Q"] {
        let cards = spectral_cube_cards(ctype, &["CUNIT3  = 'm'"]);
        let err = wcs_3d_result(&cards)
            .err()
            .unwrap_or_else(|| panic!("{ctype} should be rejected, not treated as linear"));
        assert!(
            err.to_string().contains("not a spectral algorithm code"),
            "{ctype}: unexpected error: {err}"
        );
    }
}

/// A `-GRI`/`-GRA` axis carries its disperser in `PVk_m`; without it
/// there is no coordinate function, so the header must be refused
/// rather than silently given the Table 7 defaults (which describe no
/// disperser at all).
#[test]
fn grism_without_disperser_parameters_is_rejected() {
    let cards = spectral_cube_cards("WAVE-GRI", &["CUNIT3  = 'm'"]);
    let err = wcs_3d_result(&cards).expect_err("no PV3_m present");
    assert!(
        err.to_string().contains("denominator") || err.to_string().contains("disperser"),
        "unexpected error: {err}"
    );
}

/// `Wcs::to_header` must carry the grism parameters back out, or a
/// read/write round trip silently degrades the axis.
#[test]
fn to_header_round_trips_grism_parameters() {
    let cards = spectral_cube_cards_at(
        "AWAV-GRA",
        5.0e-7,
        1.0e-10,
        &[
            "CUNIT3  = 'm'",
            "PV3_0   =              316000.",
            "PV3_1   =                   1.",
            "PV3_2   =                 13.9",
            "PV3_3   =                1.765",
            "PV3_4   =            -1077000.",
        ],
    );
    let wcs = open_image_3d(&cards);
    let header = wcs.to_header(' ').expect("to_header");
    let again = fitsy::Wcs::from_header(&header, ' ')
        .expect("reparse")
        .expect("wcs present");

    let a = wcs.spectral.first().expect("spectral axis").grism.unwrap();
    let b = again
        .spectral
        .first()
        .expect("spectral axis")
        .grism
        .unwrap();
    assert_eq!(a, b, "grism parameters did not survive to_header");

    for px in [0.0_f64, 7.0, 15.0] {
        let want = wcs.pixel_to_world(&[0.0, 0.0, px]).unwrap()[2];
        let got = again.pixel_to_world(&[0.0, 0.0, px]).unwrap()[2];
        assert!(
            near(got, want, want.abs() * 1e-12),
            "pixel {px}: {got} != {want}"
        );
    }
}

/// The recognized codes, a bare type code, and `-TAB` (handled by the
/// table machinery) must all still be accepted.
#[test]
fn recognized_spectral_codes_still_parse() {
    for (ctype, extra) in [
        ("FREQ", vec!["CUNIT3  = 'Hz'"]),
        ("WAVE", vec!["CUNIT3  = 'm'"]),
        ("AWAV", vec!["CUNIT3  = 'm'"]),
        ("WAVE-F2W", vec!["CUNIT3  = 'm'"]),
        ("FREQ-W2F", vec!["CUNIT3  = 'Hz'"]),
        ("FREQ-LOG", vec!["CUNIT3  = 'Hz'"]),
        (
            "VELO-F2V",
            vec!["CUNIT3  = 'm/s'", "RESTFRQ =        1.420e9"],
        ),
        (
            "WAVE-V2W",
            vec!["CUNIT3  = 'm'", "RESTWAV =    0.2110611410"],
        ),
        // Air-wavelength codes (Paper III Sec.4).
        ("AWAV", vec!["CUNIT3  = 'm'"]),
        ("AWAV-F2A", vec!["CUNIT3  = 'm'"]),
        ("WAVE-A2W", vec!["CUNIT3  = 'm'"]),
        ("FREQ-A2F", vec!["CUNIT3  = 'Hz'"]),
        // Grisms (Paper III Sec.5.1); PV3_m is the disperser.
        (
            "WAVE-GRI",
            vec![
                "CUNIT3  = 'm'",
                "PV3_0   =              316000.",
                "PV3_1   =                   1.",
            ],
        ),
        (
            "AWAV-GRA",
            vec![
                "CUNIT3  = 'm'",
                "PV3_0   =              316000.",
                "PV3_1   =                   1.",
            ],
        ),
    ] {
        // Reference value and increment matched to the class, so the
        // grism cases describe a real disperser rather than a 1000 m
        // wavelength.
        let (crval, cdelt) = match &ctype[..4] {
            "FREQ" => (6.0e14, -1.0e11),
            "WAVE" | "AWAV" => (5.0e-7, 1.0e-10),
            _ => (0.0, 1.0e3),
        };
        let cards = spectral_cube_cards_at(ctype, crval, cdelt, &extra);
        let wcs = wcs_3d_result(&cards)
            .unwrap_or_else(|e| panic!("{ctype} should parse: {e}"))
            .expect("wcs present");
        // and evaluate: parsing alone would not catch a broken chain.
        let world = wcs
            .pixel_to_world(&[0.0, 0.0, 8.0])
            .unwrap_or_else(|e| panic!("{ctype} should evaluate: {e}"));
        assert!(world[2].is_finite(), "{ctype}: non-finite world value");
    }
}

/// Sec.8.2 allows `PVi_m` up to `m = 99`; ZPN's polynomial runs
/// `PV2_0..PV2_20`.
///
/// Regression: only `m = 0..19` was collected, so a ZPN header's
/// highest-order term vanished from the transform with no diagnostic.
#[test]
fn zpn_uses_its_full_pv2_20_polynomial() {
    let cards = |extra: &str| {
        vec![
            "CTYPE1  = 'RA---ZPN'".to_string(),
            "CTYPE2  = 'DEC--ZPN'".to_string(),
            "CRVAL1  =                  0.0".to_string(),
            "CRVAL2  =                  0.0".to_string(),
            "CRPIX1  =                 50.0".to_string(),
            "CRPIX2  =                 50.0".to_string(),
            // A wide field on purpose: zeta^20 is ~1e-43 over an
            // arcminute-scale image, so a 20th-order term could only
            // be seen out at tens of degrees.
            "CDELT1  =                 -0.5".to_string(),
            "CDELT2  =                  0.5".to_string(),
            "PV2_1   =                   1.".to_string(),
            extra.to_string(),
        ]
    };
    let with = open_image(&cards("PV2_20  =                 500."));
    let without = open_image(&cards("PV2_19  =                   0."));

    let pv = with.celestial.as_ref().unwrap().projection.pv2();
    assert!(
        pv.iter()
            .any(|&(m, v)| m == 20 && (v - 500.0).abs() < 1e-12),
        "PV2_20 missing from the parsed projection: {pv:?}"
    );
    // And it must actually move the sky, not just be stored.
    let a = sky(&with, 0.0, 0.0);
    let b = sky(&without, 0.0, 0.0);
    assert!(
        (a.1 - b.1).abs() > 1e-6,
        "PV2_20 = 500 changed nothing: {a:?} vs {b:?}"
    );
}

/// Table 22 (and its footnote 4) give `RESTFRQ`/`RESTWAV` the alternate
/// version code.
///
/// Regression: only the bare spelling was read, so an alternate
/// description could not carry its own rest quantity -- and the writer
/// emitted an unsuffixed card that wcslib reads as the primary's.
#[test]
fn rest_quantity_honors_the_alternate_code() {
    let cards = vec![
        "CTYPE1  = 'RA---TAN'".to_string(),
        "CTYPE2  = 'DEC--TAN'".to_string(),
        "CTYPE3  = 'VOPT-F2W'".to_string(),
        "CTYPE1A = 'RA---TAN'".to_string(),
        "CTYPE2A = 'DEC--TAN'".to_string(),
        "CTYPE3A = 'VOPT-F2W'".to_string(),
        "CRVAL1  =                  0.0".to_string(),
        "CRVAL2  =                  0.0".to_string(),
        "CRVAL3  =                  0.0".to_string(),
        "CRVAL1A =                  0.0".to_string(),
        "CRVAL2A =                  0.0".to_string(),
        "CRVAL3A =                  0.0".to_string(),
        "CRPIX1  =                  1.0".to_string(),
        "CRPIX2  =                  1.0".to_string(),
        "CRPIX3  =                  1.0".to_string(),
        "CRPIX1A =                  1.0".to_string(),
        "CRPIX2A =                  1.0".to_string(),
        "CRPIX3A =                  1.0".to_string(),
        "CDELT1  =                -0.01".to_string(),
        "CDELT2  =                 0.01".to_string(),
        "CDELT3  =               1000.0".to_string(),
        "CDELT1A =                -0.01".to_string(),
        "CDELT2A =                 0.01".to_string(),
        "CDELT3A =               1000.0".to_string(),
        "CUNIT3  = 'm/s'".to_string(),
        "CUNIT3A = 'm/s'".to_string(),
        // Different lines for the two descriptions.
        "RESTWAV =         0.2110611410".to_string(),
        "RESTWAVA=        0.00260174200".to_string(),
    ];
    let primary = open_image_3d(&cards);
    let alt = {
        let bytes = {
            // `open_image_3d` only reads the primary; build and read alt 'A'.
            let mut buf = Vec::new();
            for c in [
                "SIMPLE  =                    T",
                "BITPIX  =                    8",
                "NAXIS   =                    3",
                "NAXIS1  =                    4",
                "NAXIS2  =                    4",
                "NAXIS3  =                   16",
            ] {
                buf.extend_from_slice(&pad_card(c));
            }
            for c in &cards {
                buf.extend_from_slice(&pad_card(c));
            }
            buf.extend_from_slice(&pad_card("END"));
            while buf.len() % BLOCK != 0 {
                buf.push(b' ');
            }
            let start = buf.len();
            buf.extend(std::iter::repeat_n(0_u8, 4 * 4 * 16));
            while (buf.len() - start) % BLOCK != 0 {
                buf.push(0);
            }
            buf
        };
        let file = FitsFile::from_bytes(bytes).unwrap();
        let Hdu::Image(img) = file.hdu(0).unwrap() else {
            panic!("not image");
        };
        img.wcs('A').unwrap().expect("alt A present")
    };

    let p = primary.spectral.first().expect("primary spectral axis");
    let a = alt.spectral.first().expect("alt spectral axis");
    assert!(
        (p.restwav.unwrap() - 0.211_061_141_0).abs() < 1e-15,
        "primary took the wrong RESTWAV: {:?}",
        p.restwav
    );
    assert!(
        (a.restwav.unwrap() - 0.002_601_742_00).abs() < 1e-15,
        "alternate ignored RESTWAVA: {:?}",
        a.restwav
    );

    // And the writer must emit the suffixed card for the alternate.
    let hdr = alt.to_header('A').expect("to_header");
    assert!(hdr.first("RESTWAVA").is_some(), "RESTWAVA not written");
    assert!(
        hdr.first("RESTWAV").is_none(),
        "alternate wrote an unsuffixed RESTWAV, which reads as the primary's"
    );
}

/// An alternate that gives no rest quantity of its own falls back to
/// the primary's, rather than failing. Sec.8.2.1 asks writers to repeat
/// every keyword, but they routinely do not.
#[test]
fn alternate_without_its_own_rest_quantity_falls_back() {
    let mut cards = spectral_cube_cards("VOPT-F2W", &["CUNIT3  = 'm/s'"]);
    cards.push("RESTWAV =         0.2110611410".to_string());
    cards.push("CTYPE3A = 'VOPT-F2W'".to_string());
    cards.push("CRVAL3A =                  0.0".to_string());
    cards.push("CRPIX3A =                  1.0".to_string());
    cards.push("CDELT3A =               1000.0".to_string());
    cards.push("CUNIT3A = 'm/s'".to_string());
    assert!(wcs_3d_result(&cards).is_ok(), "primary should parse");
}

/// Sec.4.3 compound unit syntax on a real axis.
///
/// Regression: unrecognized `CUNIT` values fell through with a factor
/// of 1.0, so `'km s-1'` -- a legal spelling of km/s, since a space is
/// multiplication and `s-1` is `s**-1` -- was read as m/s. Every
/// spelling below must give the same world value.
#[test]
fn compound_cunit_spellings_agree() {
    let world = |cunit: &str| {
        let cards = spectral_cube_cards(
            "VOPT-F2W",
            &[
                &format!("CUNIT3  = '{cunit}'"),
                "RESTWAV =         0.2110611410",
            ],
        );
        let wcs = wcs_3d_result(&cards)
            .unwrap_or_else(|e| panic!("CUNIT3='{cunit}': {e}"))
            .expect("wcs present");
        wcs.pixel_to_world(&[0.0, 0.0, 8.0]).unwrap()[2]
    };
    let reference = world("km/s");
    for spelling in ["km s-1", "km s**-1", "km.s**(-1)", "km*s^-1", "1000 m/s"] {
        let got = world(spelling);
        assert!(
            (got - reference).abs() <= reference.abs() * 1e-12,
            "CUNIT3='{spelling}' gave {got}, but 'km/s' gives {reference}"
        );
    }
    // ... and the factor is genuinely applied, not defaulted to 1:
    // describe the *same* physical axis in m/s and the answers must
    // agree once converted. (`world` returns values in the header's own
    // CUNIT, so this needs matching CRVAL/CDELT, not the same numbers.)
    let in_ms = {
        let cards = spectral_cube_cards_at(
            "VOPT-F2W",
            1.0e6,
            1.0e5,
            &["CUNIT3  = 'm/s'", "RESTWAV =         0.2110611410"],
        );
        wcs_3d_result(&cards)
            .unwrap()
            .unwrap()
            .pixel_to_world(&[0.0, 0.0, 8.0])
            .unwrap()[2]
    };
    let in_kms = {
        let cards = spectral_cube_cards_at(
            "VOPT-F2W",
            1.0e3,
            1.0e2,
            &["CUNIT3  = 'km/s'", "RESTWAV =         0.2110611410"],
        );
        wcs_3d_result(&cards)
            .unwrap()
            .unwrap()
            .pixel_to_world(&[0.0, 0.0, 8.0])
            .unwrap()[2]
    };
    assert!(
        (in_kms * 1000.0 - in_ms).abs() < in_ms.abs() * 1e-12,
        "the same axis in km/s and m/s disagree: {in_kms} km/s vs {in_ms} m/s"
    );
}

/// A `CUNIT` of the wrong dimension is a broken header, not something
/// to rescale. `FREQ`, `ENER` and `WAVN` all linearize through
/// frequency but are `s^-1`, `J` and `m^-1`, and used to share one
/// lookup table.
#[test]
fn cunit_of_the_wrong_dimension_is_rejected() {
    for (ctype, bad) in [
        ("WAVE", "Hz"),
        ("FREQ", "m"),
        ("ENER", "Hz"),
        ("WAVN", "Hz"),
        ("VELO", "m"),
    ] {
        let cards = spectral_cube_cards(ctype, &[&format!("CUNIT3  = '{bad}'")]);
        let err = wcs_3d_result(&cards)
            .err()
            .unwrap_or_else(|| panic!("{ctype} with CUNIT3='{bad}' should be rejected"));
        assert!(
            err.to_string().contains("required"),
            "{ctype}/{bad}: unexpected error: {err}"
        );
    }
    // ... and the matching ones still work.
    for (ctype, good) in [
        ("WAVE", "nm"),
        ("FREQ", "GHz"),
        ("ENER", "keV"),
        ("WAVN", "/m"),
        ("VELO", "km/s"),
    ] {
        let cards = spectral_cube_cards(ctype, &[&format!("CUNIT3  = '{good}'")]);
        assert!(
            wcs_3d_result(&cards).is_ok(),
            "{ctype} with CUNIT3='{good}' should parse"
        );
    }
}

/// A celestial axis in arcsec must be honored, and one declaring a
/// non-angle must be refused rather than silently treated as degrees.
#[test]
fn celestial_cunit_is_checked_and_applied() {
    let cards = |cunit: &str| {
        vec![
            "CTYPE1  = 'RA---TAN'".to_string(),
            "CTYPE2  = 'DEC--TAN'".to_string(),
            "CRVAL1  =                 10.0".to_string(),
            "CRVAL2  =                 20.0".to_string(),
            "CRPIX1  =                  1.0".to_string(),
            "CRPIX2  =                  1.0".to_string(),
            "CDELT1  =                 -1.0".to_string(),
            "CDELT2  =                  1.0".to_string(),
            format!("CUNIT1  = '{cunit}'"),
            format!("CUNIT2  = '{cunit}'"),
        ]
    };
    // 1 arcsec/px against 1 deg/px: the same pixel must land 3600x closer.
    let deg = open_image(&cards("deg"));
    let asec = open_image(&cards("arcsec"));
    let (_, d_deg) = sky(&deg, 0.0, 10.0);
    let (_, d_asec) = sky(&asec, 0.0, 10.0);
    let off_deg = d_deg - 20.0;
    let off_asec = d_asec - 20.0;
    // 10 px at 1 arcsec/px is 10/3600 deg from the reference. Compared
    // against that directly rather than as a ratio to the degree case:
    // TAN curvature is negligible over 3 arcsec but is 1% over 10 deg.
    assert!(
        (off_asec - 10.0 / 3600.0).abs() < 1e-9,
        "arcsec CUNIT ignored: latitude offset {off_asec}"
    );
    assert!(
        off_deg > 9.0,
        "degree CUNIT should move ~10 deg, moved {off_deg}"
    );
    assert!(
        try_open_image(&cards("Hz")).is_none(),
        "a frequency is not an angle"
    );
    assert!(
        try_open_image(&cards("DEG")).is_none(),
        "case is significant"
    );
}

/// Paper III Sec.3.3.1 fixes one associate variable per spectral type
/// and names `ZOPT-F2V` as unrecognized, since z goes with lambda, not
/// v. Accepting it silently reinterpreted the header as `ZOPT-F2W`.
#[test]
fn spectral_code_with_the_wrong_associate_is_rejected() {
    for (ctype, cunit) in [
        ("ZOPT-F2V", "' '"),
        ("VRAD-F2W", "'m/s'"),
        ("WAVE-F2V", "'m'"),
        ("VELO-F2W", "'m/s'"),
    ] {
        let cards = spectral_cube_cards_at(
            ctype,
            0.0,
            1.0,
            &[
                &format!("CUNIT3  = {cunit}"),
                "RESTWAV =         0.2110611410",
            ],
        );
        let err = wcs_3d_result(&cards)
            .err()
            .unwrap_or_else(|| panic!("{ctype} should be rejected"));
        assert!(
            err.to_string().contains("associate"),
            "{ctype}: unexpected error: {err}"
        );
    }
    // The matching combinations still parse.
    for (ctype, crval, cunit) in [
        ("ZOPT-F2W", 0.0, "' '"),
        ("VRAD-W2F", 0.0, "'m/s'"),
        ("WAVE-F2W", 5.0e-7, "'m'"),
        ("VELO-F2V", 0.0, "'m/s'"),
    ] {
        let cards = spectral_cube_cards_at(
            ctype,
            crval,
            1.0e-10,
            &[
                &format!("CUNIT3  = {cunit}"),
                "RESTWAV =         0.2110611410",
            ],
        );
        assert!(wcs_3d_result(&cards).is_ok(), "{ctype} should parse");
    }
}

/// A parser must refuse malformed input, not abort on it.
///
/// Regression: `CTYPE` handling sliced at byte 4, which panics when
/// that byte falls inside a multi-byte character. Unreachable from a
/// file (the card reader keeps non-ASCII out) but reachable through a
/// programmatically built `Header`.
#[test]
fn non_ascii_ctype_errors_instead_of_panicking() {
    let mut h = fitsy::Header::empty();
    h.push("NAXIS", 2_i64, None).unwrap();
    h.push("NAXIS1", 10_i64, None).unwrap();
    h.push("NAXIS2", 10_i64, None).unwrap();
    // 'e' with an acute accent spans bytes 3..5, so byte 4 splits it.
    h.push("CTYPE1", "RAB\u{e9}-TAN".to_string(), None).unwrap();
    h.push("CTYPE2", "DEC--TAN".to_string(), None).unwrap();
    // Must return, either way -- the point is that it does not panic.
    let _ = fitsy::Wcs::from_header(&h, ' ');

    // And the same CTYPE on the latitude axis, which reaches
    // `projection_code` rather than `first4`.
    let mut h = fitsy::Header::empty();
    h.push("NAXIS", 2_i64, None).unwrap();
    h.push("NAXIS1", 10_i64, None).unwrap();
    h.push("NAXIS2", 10_i64, None).unwrap();
    h.push("CTYPE1", "RA---TAN".to_string(), None).unwrap();
    h.push("CTYPE2", "DEC\u{e9}TAN".to_string(), None).unwrap();
    let _ = fitsy::Wcs::from_header(&h, ' ');
}

/// Paper III Sec.6.1.1's non-separable case: a celestial pair whose
/// longitude and latitude share one `(M, K_1, K_2)` coordinate array
/// and interpolate together.
///
/// Checked against wcslib rather than round-tripped -- the fixture
/// carries wcslib's own decode in a `REFERENCE` HDU. See
/// `tests/data/gen_tab_reference.py`.
#[test]
fn multi_dimensional_tab_matches_wcslib() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ref_tab_2d.fits");
    let file = FitsFile::open(&path).unwrap();
    let wcs = file
        .wcs(0, ' ')
        .expect("a celestial -TAB pair must parse")
        .expect("wcs present");

    // One group drives both axes; neither is a projection.
    assert_eq!(wcs.tab.len(), 1, "the two axes share one table");
    assert_eq!(wcs.tab[0].rank(), 2, "M = 2");
    assert_eq!(wcs.tab[0].axes, vec![0, 1], "PVi_3 places the axes");
    assert_eq!(wcs.tab[0].dims, vec![5, 4], "K_1, K_2 from TDIM");
    assert!(wcs.celestial.is_none(), "a -TAB pair carries no projection");
    assert!(wcs.is_celestial(), "but it is still a celestial pair");
    assert_eq!(wcs.celestial_axes(), Some((0, 1)));

    // Reference values recorded by wcslib.
    let Hdu::Image(refhdu) = file.hdu_by_name("REFERENCE", None).unwrap() else {
        panic!("no REFERENCE hdu");
    };
    let h = refhdu.header();
    let get = |k: &str| -> Vec<f64> {
        h.entries()
            .iter()
            .filter(|e| e.keyword == k)
            .filter_map(|e| match e.value.as_ref()? {
                fitsy::header::value::Value::Real(r) => Some(*r),
                fitsy::header::value::Value::Integer(i) => Some(*i as f64),
                _ => None,
            })
            .collect()
    };
    let (px, py, wlon, wlat) = (get("PIXX"), get("PIXY"), get("WLON"), get("WLAT"));
    assert!(px.len() >= 7, "reference points missing");

    for i in 0..px.len() {
        let got = wcs.pixel_to_world(&[px[i], py[i]]).unwrap();
        assert!(
            (got[0] - wlon[i]).abs() < 1e-10 && (got[1] - wlat[i]).abs() < 1e-10,
            "pixel ({}, {}): got {got:?}, wcslib [{}, {}]",
            px[i],
            py[i],
            wlon[i],
            wlat[i],
        );
        // And the inverse recovers the pixel wcslib started from. The
        // map is non-separable, so this is a genuine 2-D solve.
        let back = wcs.world_to_pixel(&[wlon[i], wlat[i]]).unwrap();
        assert!(
            (back[0] - px[i]).abs() < 1e-8 && (back[1] - py[i]).abs() < 1e-8,
            "inverse of ({}, {}) gave {back:?}",
            wlon[i],
            wlat[i],
        );
        // The celestial convenience wrappers go through the same path.
        let (ra, dec) = sky(&wcs, px[i], py[i]);
        assert!((ra - wlon[i]).abs() < 1e-10 && (dec - wlat[i]).abs() < 1e-10);
    }
}

/// A celestial pair with only one `-TAB` member has no defined
/// transform: the tabular side needs the shared coordinate array, the
/// projected side needs its partner in the spherical rotation, and
/// Paper III Sec.6.1.1 declares unmet `-TAB` group conditions
/// undefined. wcslib rejects the same header as "unmatched celestial
/// axes".
///
/// `RADESYS` must survive `to_header` for a fully tabular celestial
/// pair. The frame is real for `RA---TAB`/`DEC--TAB`. Dropping it
/// breaks `from_header(to_header(w)) == w`.
///
/// The header carries no `EQUINOX`. With one present, re-parsing
/// derives the same frame from the Sec.8.3 equinox default. That
/// would mask the dropped card.
///
/// Regression: the serializer gated `RADESYS` on the `CelestialBlock`.
/// A tabular pair does not carry one. The parser keeps the keyword
/// whenever the *pair* exists. `FK5` parsed, serialized to nothing,
/// and re-parsed as the ICRS default.
#[test]
fn radesys_round_trips_for_a_tabular_celestial_pair() {
    use fitsy::wcs::RadeSys;

    let mut h = fitsy::Header::empty();
    h.push("NAXIS", 2_i64, None).unwrap();
    h.push("NAXIS1", 5_i64, None).unwrap();
    h.push("NAXIS2", 4_i64, None).unwrap();
    h.push("CTYPE1", "RA---TAB".to_string(), None).unwrap();
    h.push("CTYPE2", "DEC--TAB".to_string(), None).unwrap();
    // Both axes point at one shared coordinate array (Sec.6.1.1).
    for axis in [1, 2] {
        h.push(format!("PS{axis}_0"), "WCS-TAB".to_string(), None)
            .unwrap();
        h.push(format!("PS{axis}_1"), "COORDS".to_string(), None)
            .unwrap();
    }
    h.push("RADESYS", "FK5".to_string(), None).unwrap();

    let wcs = fitsy::Wcs::from_header(&h, ' ')
        .unwrap()
        .expect("a tabular celestial pair is a WCS");
    assert!(wcs.celestial.is_none(), "a -TAB pair carries no projection");
    assert!(wcs.is_celestial(), "but it is still a celestial pair");
    assert_eq!(wcs.radesys, RadeSys::Fk5);

    let serialized = wcs.to_header(' ').unwrap();
    assert!(
        serialized.contains("RADESYS"),
        "to_header dropped the RADESYS card"
    );
    let reparsed = fitsy::Wcs::from_header(&serialized, ' ')
        .unwrap()
        .expect("serialized header still describes a WCS");
    assert_eq!(
        reparsed.radesys,
        RadeSys::Fk5,
        "RADESYS did not survive the round trip"
    );
}

/// Regression: the pair parsed with no `CelestialBlock`, so the
/// projected axis ran the bare linear pipeline -- coordinates with no
/// projection applied, silently.
#[test]
fn mixed_tab_celestial_pair_is_rejected() {
    for (lon, lat) in [("RA---TAB", "DEC--TAN"), ("RA---TAN", "DEC--TAB")] {
        let mut h = fitsy::Header::empty();
        h.push("NAXIS", 2_i64, None).unwrap();
        h.push("NAXIS1", 5_i64, None).unwrap();
        h.push("NAXIS2", 5_i64, None).unwrap();
        h.push("CTYPE1", lon.to_string(), None).unwrap();
        h.push("CTYPE2", lat.to_string(), None).unwrap();
        h.push("CRVAL1", 1.0_f64, None).unwrap();
        h.push("CRVAL2", 30.0_f64, None).unwrap();
        // Pointer cards for whichever axis is tabular, so the header
        // is complete apart from the mismatch itself.
        let tab_axis = if lon.ends_with("TAB") { 1 } else { 2 };
        h.push(format!("PS{tab_axis}_0"), "WCS-TAB".to_string(), None)
            .unwrap();
        h.push(format!("PS{tab_axis}_1"), "COORDS".to_string(), None)
            .unwrap();
        let err = fitsy::Wcs::from_header(&h, ' ')
            .err()
            .unwrap_or_else(|| panic!("{lon}/{lat} must be rejected"));
        assert!(
            err.to_string().contains("both axes or neither"),
            "{lon}/{lat}: unexpected error: {err}"
        );
    }
}

/// The coordinates are genuinely coupled: moving one pixel axis must
/// change *both* world coordinates. A pair of independent 1-D tables
/// could not reproduce this, so it pins the multilinear blend.
#[test]
fn multi_dimensional_tab_is_non_separable() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ref_tab_2d.fits");
    let file = FitsFile::open(&path).unwrap();
    let wcs = file.wcs(0, ' ').unwrap().unwrap();

    let a = wcs.pixel_to_world(&[2.0, 0.0]).unwrap();
    let b = wcs.pixel_to_world(&[2.0, 3.0]).unwrap();
    assert!(
        (a[0] - b[0]).abs() > 1e-6,
        "longitude ignores the second axis"
    );
    let c = wcs.pixel_to_world(&[0.0, 2.0]).unwrap();
    let d = wcs.pixel_to_world(&[4.0, 2.0]).unwrap();
    assert!(
        (c[1] - d[1]).abs() > 1e-6,
        "latitude ignores the first axis"
    );
}

/// Standard Sec.9.5.3: an image time axis *is* the linear transform,
/// so the numbers must not change -- what the parser adds is the
/// axis's identity. wcslib draws the same line, classifying the axis
/// (`axis_types` 4000) while still returning elapsed time.
#[test]
fn time_axis_is_recognized_without_changing_the_transform() {
    let cards = |ctype3: &str, extra: &[&str]| {
        let mut v = spectral_cube_cards_at(ctype3, 100.0, 30.0, &["CUNIT3  = 's'"]);
        v.extend(extra.iter().map(|s| (*s).to_string()));
        v
    };
    let wcs = open_image_3d(&cards(
        "TIME",
        &["TIMESYS = 'TT'", "MJDREF  =              57754."],
    ));
    let time = wcs.time.as_ref().expect("CTYPE3 = 'TIME' is a time axis");
    assert_eq!(time.axis, 2);
    assert_eq!(time.scale, "TT", "CTYPE 'TIME' defers to TIMESYS");
    assert!(wcs.is_celestial(), "the celestial pair is untouched");

    // Elapsed time, exactly as before: CRVAL3 + n * CDELT3. This is
    // also what wcslib returns -- it does not fold in MJDREF either.
    for (px, want) in [(0.0, 100.0), (1.0, 130.0), (5.0, 250.0)] {
        let got = wcs.pixel_to_world(&[0.0, 0.0, px]).unwrap()[2];
        assert!((got - want).abs() < 1e-9, "pixel {px}: {got} != {want}");
    }

    // Sec.9.2.1: the axis may name its own scale, overriding TIMESYS.
    let wcs = open_image_3d(&cards("TDB", &["TIMESYS = 'UTC'"]));
    assert_eq!(wcs.time.as_ref().unwrap().scale, "TDB");

    // A realization is kept but does not change the scale.
    let wcs = open_image_3d(&cards("TT(TAI)", &[]));
    let t = wcs.time.as_ref().unwrap();
    assert_eq!(
        (t.scale.as_str(), t.realization.as_deref()),
        ("TT", Some("TAI"))
    );

    // TIMESYS itself defaults to UTC (Sec.9.2.1).
    let wcs = open_image_3d(&cards("TIME", &[]));
    assert_eq!(wcs.time.as_ref().unwrap().scale, "UTC");

    // And a non-time axis is not mistaken for one.
    let wcs = open_image_3d(&spectral_cube_cards_at(
        "FREQ",
        6.0e14,
        -1.0e11,
        &["CUNIT3  = 'Hz'"],
    ));
    assert!(wcs.time.is_none());
}

/// The Sec.9 metadata wcslib keeps on `wcsprm`, surfaced and
/// round-tripped through `to_header`.
#[test]
fn time_reference_frame_keywords_are_surfaced_and_round_trip() {
    let mut cards = spectral_cube_cards_at("TIME", 0.0, 1.0, &["CUNIT3  = 's'"]);
    cards.extend(
        [
            "TIMESYS = 'TT'",
            "TREFPOS = 'BARYCENTER'",
            "TREFDIR = 'RA_NOM,DEC_NOM'",
            "PLEPHEM = 'DE430'",
            "CZPHS3  =                 12.5",
            "CPERI3  =                 0.25",
            "CRDER3  =                1.E-3",
            "CSYER3  =                5.E-3",
        ]
        .iter()
        .map(|s| (*s).to_string()),
    );
    let wcs = open_image_3d(&cards);
    let t = wcs.time.clone().expect("time axis");
    assert_eq!(t.trefpos.as_deref(), Some("BARYCENTER"));
    assert_eq!(t.trefdir.as_deref(), Some("RA_NOM,DEC_NOM"));
    assert_eq!(t.plephem.as_deref(), Some("DE430"));
    // CZPHS/CPERI describe a phase axis (Sec.9.6); on this TIME axis
    // they describe nothing and are dropped.
    assert!(wcs.phase.is_empty());
    // The error pair is per-axis metadata legal on any axis (Sec.8.2).
    assert_eq!(wcs.axes()[2].crder, Some(1e-3));
    assert_eq!(wcs.axes()[2].csyer, Some(5e-3));
    // Absent on the other axes rather than defaulted to zero.
    assert_eq!(wcs.axes()[0].crder, None);

    let header = wcs.to_header(' ').expect("to_header");
    let again = fitsy::Wcs::from_header(&header, ' ').unwrap().unwrap();
    assert_eq!(again.time, wcs.time);
    assert_eq!(again.axes(), wcs.axes());
    assert_eq!(again.time.as_ref().map(|t| t.scale.as_str()), Some("TT"));
}

/// Sec.9.2.1: a CTYPE naming a Table 29 scale overrides `TIMESYS` and
/// never involves it. `to_header` must not copy that scale into a
/// `TIMESYS` card the source never had -- `TIMESYS` governs the scale
/// of the header's *other* time keywords (`DATE-OBS`, `MJD-OBS`),
/// which would silently be reinterpreted on the axis's scale.
#[test]
fn ctype_scale_override_does_not_invent_a_timesys_card() {
    let cards = spectral_cube_cards_at("TDB", 0.0, 1.0, &["CUNIT3  = 's'"]);
    let wcs = open_image_3d(&cards);
    assert_eq!(wcs.time.as_ref().map(|t| t.scale.as_str()), Some("TDB"));

    let header = wcs.to_header(' ').expect("to_header");
    assert!(
        header.first("TIMESYS").is_none(),
        "no TIMESYS in the source header, none may be invented"
    );
    // The scale still round-trips, carried by CTYPE itself.
    let again = fitsy::Wcs::from_header(&header, ' ').unwrap().unwrap();
    assert_eq!(again.time, wcs.time);
}

/// A `'PHASE'` axis (Sec.9.6) carries its zero point and period; the
/// pair round-trips, and a time axis alongside keeps its own trio.
#[test]
fn phase_axis_carries_czphs_and_cperi() {
    let mut cards = spectral_cube_cards_at("PHASE", 0.0, 0.01, &["CUNIT3  = 's'"]);
    cards.extend(
        [
            "CZPHS3  =                 12.5",
            "CPERI3  =                 0.25",
        ]
        .iter()
        .map(|s| (*s).to_string()),
    );
    let wcs = open_image_3d(&cards);
    assert!(wcs.time.is_none(), "PHASE is not a time axis");
    assert_eq!(wcs.phase.len(), 1);
    let p = &wcs.phase[0];
    assert_eq!((p.axis, p.czphs, p.cperi), (2, Some(12.5), Some(0.25)));

    let header = wcs.to_header(' ').expect("to_header");
    let again = fitsy::Wcs::from_header(&header, ' ').unwrap().unwrap();
    assert_eq!(again.phase, wcs.phase);
}

/// Which `CTYPE` values count as a time axis, checked against wcslib.
///
/// The expectations below are wcslib's own (`WCS.wcs.axis_types[0] ==
/// 4000`), sampled across Table 29 plus the neighboring axis types.
/// Two entries deliberately disagree, both places where Sec.9.2.1 is
/// more permissive than wcslib -- see the second half of the test.
#[test]
fn time_axis_classification_matches_wcslib() {
    let agreed: &[(&str, bool)] = &[
        ("TIME", true),
        ("TAI", true),
        ("TT", true),
        ("TDT", true),
        ("ET", true),
        ("IAT", true),
        ("UT1", true),
        ("UTC", true),
        ("GMT", true),
        ("GPS", true),
        ("TCG", true),
        ("TCB", true),
        ("TDB", true),
        ("LOCAL", true),
        ("PHASE", false),
        ("TIMELAG", false),
        ("FREQUENCY", false),
        ("FREQ", false),
        ("STOKES", false),
        ("DETX", false),
        ("RA---TAN", false),
    ];
    for &(ctype, want) in agreed {
        let got = fitsy::wcs::TimeAxis::recognize(0, ctype, "UTC").is_some();
        assert_eq!(got, want, "CTYPE = {ctype:?}");
    }

    // Sec.9.2.1 says a CTYPE "may also assume the value TIME
    // (case-insensitive)". wcslib matches only upper case; fitsy
    // follows the standard.
    assert!(fitsy::wcs::TimeAxis::recognize(0, "time", "TT").is_some());

    // Sec.9.2.1 also permits a realization in parentheses on the
    // Table 29 values -- "TT(TAI), TT(BIPM08), UTC(NIST)". wcslib does
    // not classify those as time axes; fitsy does, and keeps the
    // realization. Harmless either way for the coordinates, since
    // Sec.9.5.3 makes the transform linear regardless.
    for ctype in ["TT(TAI)", "UTC(NIST)", "UT(NIST)"] {
        assert!(
            fitsy::wcs::TimeAxis::recognize(0, ctype, "UTC").is_some(),
            "{ctype} is a Table 29 value with a realization"
        );
    }
}

/// The per-axis fields of a `Wcs` used to be public parallel vectors,
/// and truncating any of them desynchronized the description: four of
/// five mutations panicked the transform or the serializer, and `naxis`
/// was a second source of truth alongside `linear.naxis()`.
///
/// They are private now, so this pins the replacement surface instead:
/// the count is *derived*, and the indexed readers are total.
#[test]
fn per_axis_reads_are_total_and_the_axis_count_is_derived() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CUNIT1  = 'deg'".into(),
        "CRVAL1  =                 10.0".into(),
        "CRVAL2  =                 20.0".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CDELT1  =               -0.001".into(),
        "CDELT2  =                0.001".into(),
    ];
    let wcs = open_image(&cards);

    // Derived, so it cannot disagree with the per-axis data or with
    // the linear transform.
    assert_eq!(wcs.naxis(), 2);
    assert_eq!(wcs.naxis(), wcs.axes().len());
    assert_eq!(wcs.naxis(), wcs.linear().naxis());
    assert_eq!(wcs.crval().len(), wcs.naxis());

    assert_eq!(wcs.ctype(0), "RA---TAN");
    assert_eq!(wcs.ctype(1), "DEC--TAN");
    assert_eq!(wcs.cunit(0), "deg");
    // An absent CUNIT reads the same as an out-of-range axis: blank.
    assert_eq!(wcs.cunit(1), "");
    assert_eq!(wcs.crval(), &[10.0, 20.0]);

    // Past the end is empty, not a panic.
    assert_eq!(wcs.ctype(99), "");
    assert_eq!(wcs.cunit(99), "");
    assert!(wcs.axis(99).is_none());
    assert!(wcs.axis(1).is_some());
}

/// `Wcs::new` range-checks the axis indices the per-axis-type structs
/// carry, because each one is used to index a `naxis`-long slice.
///
/// Reachable from a header: `WCSAXES` sets the axis count, so a
/// `-TAB` pointer on a higher axis number names an axis that does not
/// exist. It used to be accepted and panic in the transform.
#[test]
fn out_of_range_axis_indices_are_refused() {
    let mut h = fitsy::Header::empty();
    h.push("NAXIS", 3_i64, None).unwrap();
    for i in 1..=3 {
        h.push(format!("NAXIS{i}"), 4_i64, None).unwrap();
    }
    // Two WCS axes declared, but a -TAB pointer on axis 3.
    h.push("WCSAXES", 2_i64, None).unwrap();
    h.push("CTYPE1", "RA---TAN".to_string(), None).unwrap();
    h.push("CTYPE2", "DEC--TAN".to_string(), None).unwrap();
    h.push("CTYPE3", "WAVE-TAB".to_string(), None).unwrap();
    h.push("PS3_0", "WCS-TAB".to_string(), None).unwrap();
    h.push("PS3_1", "COORDS".to_string(), None).unwrap();
    // Whatever the parser makes of it, it must not build a `Wcs` whose
    // -TAB spec points past the axis list.
    if let Ok(Some(w)) = fitsy::Wcs::from_header(&h, ' ') {
        for s in &w.tab_specs {
            assert!(
                s.axis < w.naxis(),
                "-TAB spec names axis {} of a {}-axis WCS",
                s.axis + 1,
                w.naxis()
            );
        }
    }
}

/// The batch transform must agree with the single-point transform on
/// every point, including a `-SIN` field where part of the plane lies
/// outside the projection.
#[test]
fn batch_matches_single_point_transform() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---SIN'".into(),
        "CTYPE2  = 'DEC--SIN'".into(),
        "CRPIX1  =                 50.0".into(),
        "CRPIX2  =                 50.0".into(),
        "CRVAL1  =                  0.0".into(),
        "CRVAL2  =                  0.0".into(),
        "CDELT1  =                 -1.0".into(),
        "CDELT2  =                  1.0".into(),
    ];
    let wcs = open_image(&cards);
    let mut flat = Vec::new();
    for i in 0..40 {
        for j in 0..40 {
            flat.push(f64::from(i) * 5.0 - 20.0);
            flat.push(f64::from(j) * 5.0 - 20.0);
        }
    }
    let batch = wcs.pixel_to_world_many(&flat).unwrap();
    assert_eq!(batch.len(), flat.len());
    let mut outside = 0;
    for (point, got) in flat
        .as_chunks::<2>()
        .0
        .iter()
        .zip(batch.as_chunks::<2>().0.iter())
    {
        if let Ok(want) = wcs.pixel_to_world(point) {
            assert!(near(got[0], want[0], 1e-12), "lon {} {}", got[0], want[0]);
            assert!(near(got[1], want[1], 1e-12), "lat {} {}", got[1], want[1]);
        } else {
            // A point the single call rejects becomes NaN in the batch.
            outside += 1;
            assert!(
                got[0].is_nan() && got[1].is_nan(),
                "expected NaN, got {got:?}"
            );
        }
    }
    assert!(outside > 0, "test needs points outside the SIN domain");
}

/// A batch round trip must land back on the pixel it started from.
#[test]
fn batch_round_trip_returns_original_pixels() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                 32.0".into(),
        "CRPIX2  =                 24.0".into(),
        "CRVAL1  =              202.469".into(),
        "CRVAL2  =               47.195".into(),
        "CDELT1  =               -0.001".into(),
        "CDELT2  =                0.001".into(),
    ];
    let wcs = open_image(&cards);
    let pixels: Vec<f64> = vec![0.0, 0.0, 31.0, 23.0, 63.0, 47.0, 10.5, 40.25];
    let world = wcs.pixel_to_world_many(&pixels).unwrap();
    let back = wcs.world_to_pixel_many(&world).unwrap();
    // The TAN inverse leaves a residual of a few times 1e-9 pixels.
    // The scalar path leaves the same one, bit for bit.
    for (want, got) in pixels.iter().zip(&back) {
        assert!(near(*want, *got, 1e-6), "want {want}, got {got}");
    }
}

/// A length that is not a whole number of points is a whole-batch
/// error, not a per-point NaN.
#[test]
fn batch_rejects_a_partial_point() {
    let cards: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                 32.0".into(),
        "CRPIX2  =                 24.0".into(),
        "CRVAL1  =              202.469".into(),
        "CRVAL2  =               47.195".into(),
        "CDELT1  =               -0.001".into(),
        "CDELT2  =                0.001".into(),
    ];
    let wcs = open_image(&cards);
    assert!(wcs.pixel_to_world_many(&[1.0, 2.0, 3.0]).is_err());
    assert!(wcs.world_to_pixel_many(&[1.0, 2.0, 3.0]).is_err());
    // An empty batch is a valid batch of zero points.
    assert_eq!(wcs.pixel_to_world_many(&[]).unwrap().len(), 0);
}

/// `pixel_scale_at` on a cube: the non-celestial axes evaluate at their
/// reference pixel, so the scale matches the same WCS with the third
/// axis dropped. This is the surviving public user of that fill.
#[test]
fn pixel_scale_at_fills_non_celestial_axes_on_a_cube() {
    let mut cube: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CTYPE3  = 'FREQ'".into(),
        "CRPIX1  =                 32.0".into(),
        "CRPIX2  =                 24.0".into(),
        "CRPIX3  =                  5.0".into(),
        "CRVAL1  =              202.469".into(),
        "CRVAL2  =               47.195".into(),
        "CRVAL3  =              1.4E+09".into(),
        "CDELT1  =               -0.001".into(),
        "CDELT2  =                0.001".into(),
        "CDELT3  =              1.0E+06".into(),
        "CUNIT3  = 'Hz'".into(),
    ];
    let wcs3 = open_image_3d(&cube);
    cube.truncate(2);
    let flat: Vec<String> = vec![
        "CTYPE1  = 'RA---TAN'".into(),
        "CTYPE2  = 'DEC--TAN'".into(),
        "CRPIX1  =                 32.0".into(),
        "CRPIX2  =                 24.0".into(),
        "CRVAL1  =              202.469".into(),
        "CRVAL2  =               47.195".into(),
        "CDELT1  =               -0.001".into(),
        "CDELT2  =                0.001".into(),
    ];
    let wcs2 = open_image(&flat);
    for (px, py) in [(0.0, 0.0), (31.0, 23.0), (63.0, 47.0)] {
        let (ax, ay) = wcs3.pixel_scale_at(px, py).unwrap();
        let (bx, by) = wcs2.pixel_scale_at(px, py).unwrap();
        assert!(near(ax, bx, 1e-9), "x scale {ax} vs {bx}");
        assert!(near(ay, by, 1e-9), "y scale {ay} vs {by}");
    }
}

/// Every axis reports a kind, so a caller can find an axis by meaning
/// rather than by position.
#[test]
fn axis_kinds_name_every_axis_of_a_cube() {
    use fitsy::AxisKind;
    let cards: Vec<String> = [
        "CTYPE1  = 'RA---TAN'",
        "CTYPE2  = 'DEC--TAN'",
        "CTYPE3  = 'FREQ'",
        "CRPIX1  =                 32.0",
        "CRPIX2  =                 24.0",
        "CRPIX3  =                  5.0",
        "CRVAL1  =              202.469",
        "CRVAL2  =               47.195",
        "CRVAL3  =              1.4E+09",
        "CDELT1  =               -0.001",
        "CDELT2  =                0.001",
        "CDELT3  =              1.0E+06",
        "CUNIT3  = 'Hz'",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let wcs = open_image_3d(&cards);
    assert_eq!(
        wcs.axis_kinds(),
        vec![AxisKind::Longitude, AxisKind::Latitude, AxisKind::Spectral]
    );
    assert_eq!(wcs.axis_kind(0), Some(AxisKind::Longitude));
    assert_eq!(wcs.axis_kind(3), None, "past the last axis");
    // The kinds line up with the values `pixel_to_world` returns.
    let world = wcs.pixel_to_world(&[31.0, 23.0, 4.0]).unwrap();
    assert_eq!(world.len(), wcs.axis_kinds().len());
    let spectral = wcs
        .axis_kinds()
        .iter()
        .position(|k| *k == AxisKind::Spectral)
        .unwrap();
    assert!((world[spectral] - 1.4e9).abs() < 1.0);
}

/// A swapped `CTYPE` order must be reported honestly: the kind follows
/// the axis, not its position.
#[test]
fn axis_kinds_follow_a_swapped_ctype_order() {
    use fitsy::AxisKind;
    let cards: Vec<String> = [
        "CTYPE1  = 'DEC--TAN'",
        "CTYPE2  = 'RA---TAN'",
        "CRPIX1  =                 32.0",
        "CRPIX2  =                 24.0",
        "CRVAL1  =               47.195",
        "CRVAL2  =              202.469",
        "CDELT1  =                0.001",
        "CDELT2  =               -0.001",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let wcs = open_image(&cards);
    assert_eq!(
        wcs.axis_kinds(),
        vec![AxisKind::Latitude, AxisKind::Longitude]
    );
}

/// A non-celestial, non-spectral axis falls back to `Linear`, and
/// `STOKES` is named rather than lumped in with it.
#[test]
fn axis_kinds_cover_stokes_and_plain_linear() {
    use fitsy::AxisKind;
    let cards: Vec<String> = [
        "CTYPE1  = 'STOKES'",
        "CTYPE2  = 'DETX'",
        "CRPIX1  =                  1.0",
        "CRPIX2  =                  1.0",
        "CRVAL1  =                  1.0",
        "CRVAL2  =                  0.0",
        "CDELT1  =                  1.0",
        "CDELT2  =                  1.0",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let wcs = open_image(&cards);
    assert_eq!(wcs.axis_kinds(), vec![AxisKind::Stokes, AxisKind::Linear]);
    assert!(!wcs.is_tabular(0));
    assert!(!wcs.is_tabular(99), "an absent axis is not tabular");
}

/// A linear transform of the wrong rank is rejected at `set_linear`,
/// not carried into the per-point bodies.
///
/// The per-point bodies index `CRPIX` unchecked and zip the matrix
/// against `naxis` rows, so a mismatched transform must never be
/// installed. The field is private; `Wcs::new` and `set_linear` are
/// the two ways a transform gets in, and both validate the rank.
#[test]
fn a_mismatched_linear_transform_is_rejected() {
    use fitsy::wcs::LinearTransform;
    let mut wcs = open_image(&tan_cards());
    assert_eq!(wcs.naxis(), 2);
    for rank in [1_usize, 3] {
        let wrong = LinearTransform::from_cd(
            vec![1.0; rank],
            vec![0.0; rank],
            (0..rank * rank)
                .map(|i| if i % (rank + 1) == 0 { 1.0 } else { 0.0 })
                .collect(),
        )
        .expect("a well-formed transform of the wrong rank");
        let msg = wcs.set_linear(wrong).unwrap_err().to_string();
        assert!(
            msg.contains("linear transform"),
            "rank {rank}: expected a shape complaint, got {msg}"
        );
    }
    // A rank-matched replacement is accepted, and the transforms
    // keep working afterwards.
    let same_rank =
        LinearTransform::from_cd(vec![1.0, 1.0], vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 1.0])
            .expect("a rank-2 transform");
    wcs.set_linear(same_rank).expect("rank matches");
    wcs.pixel_to_world(&[0.0, 0.0]).expect("still transforms");
}

/// `Time` and `Phase`, the two kinds that come from a parsed side
/// table rather than from the `CTYPE` text.
///
/// `axis_kind` probes both after the spectral check, which reads the
/// first four characters of `CTYPE`. This test therefore also pins that
/// no time or phase spelling collides with a spectral type code.
#[test]
fn axis_kinds_name_time_and_phase() {
    use fitsy::AxisKind;
    // Sec.9.5.3 lets the time axis carry the scale as its `CTYPE`,
    // so the kind cannot be keyed off a single literal.
    for ctype in ["TIME", "UTC", "TAI", "TT"] {
        let cards: Vec<String> = [
            "CTYPE1  = 'RA---TAN'".to_string(),
            "CTYPE2  = 'DEC--TAN'".to_string(),
            format!("CTYPE3  = '{ctype}'"),
            "CRPIX1  =                  1.0".into(),
            "CRPIX2  =                  1.0".into(),
            "CRPIX3  =                  1.0".into(),
            "CRVAL1  =                 10.0".into(),
            "CRVAL2  =                 -5.0".into(),
            "CRVAL3  =                  0.0".into(),
            "CDELT1  =              -0.0010".into(),
            "CDELT2  =               0.0010".into(),
            "CDELT3  =                  1.0".into(),
        ]
        .into_iter()
        .collect();
        let wcs = open_image_3d(&cards);
        assert_eq!(
            wcs.axis_kind(2),
            Some(AxisKind::Time),
            "CTYPE3 = '{ctype}' should name a time axis"
        );
    }

    // Sec.9.6: a phase axis is recognized by `CTYPE` plus its
    // `CPERIia` period.
    let cards: Vec<String> = [
        "CTYPE1  = 'RA---TAN'",
        "CTYPE2  = 'DEC--TAN'",
        "CTYPE3  = 'PHASE'",
        "CRPIX1  =                  1.0",
        "CRPIX2  =                  1.0",
        "CRPIX3  =                  1.0",
        "CRVAL1  =                 10.0",
        "CRVAL2  =                 -5.0",
        "CRVAL3  =                  0.0",
        "CDELT1  =              -0.0010",
        "CDELT2  =               0.0010",
        "CDELT3  =                  0.1",
        "CPERI3  =                  1.5",
        "CZPHS3  =                  0.0",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let wcs = open_image_3d(&cards);
    assert_eq!(
        wcs.axis_kinds(),
        vec![AxisKind::Longitude, AxisKind::Latitude, AxisKind::Phase]
    );
}
