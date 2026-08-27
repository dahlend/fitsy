//! Regression tests for conformance gaps found by checking the code
//! against FITS Standard 4.0. Each names the section it enforces.

use fitsy::header::card::CARD_SIZE;
use fitsy::{FitsAppender, FitsFile, Hdu, Header, ImageBuilder, Wcs};

/// Sky position of a celestial pixel, as `(lon, lat)`.
///
/// The public API takes one pixel value per axis and returns one world
/// value per axis, both in axis order. This helper supplies the
/// celestial pair and holds every other axis at its reference pixel,
/// so a cube compares against the equivalent 2-D image. Every WCS in
/// this file puts longitude on axis 1.
fn sky(wcs: &Wcs, px: f64, py: f64) -> (f64, f64) {
    // `CRPIX` is 1-based and the API is 0-based, hence the shift.
    let mut point: Vec<f64> = wcs.linear().crpix().iter().map(|c| c - 1.0).collect();
    point[0] = px;
    point[1] = py;
    let w = wcs.pixel_to_world(&point).expect("pixel_to_world");
    (w[0], w[1])
}

const BLOCK: usize = 2880;

fn block(cards: &[&str]) -> Vec<u8> {
    let mut buf = Vec::new();
    for c in cards {
        let mut card = [b' '; CARD_SIZE];
        assert!(c.len() <= CARD_SIZE, "card too long: {c}");
        card[..c.len()].copy_from_slice(c.as_bytes());
        buf.extend_from_slice(&card);
    }
    let mut end = [b' '; CARD_SIZE];
    end[..3].copy_from_slice(b"END");
    buf.extend_from_slice(&end);
    while buf.len() % BLOCK != 0 {
        buf.push(b' ');
    }
    buf
}

fn header_of(cards: &[&str]) -> Header {
    Header::parse(&block(cards), 0).unwrap().0
}

fn wcs_of(cards: &[&str]) -> Wcs {
    Wcs::from_header(&header_of(cards), ' ')
        .expect("WCS parse failed")
        .expect("no WCS in header")
}

const TAN_2D: &[&str] = &[
    "SIMPLE  =                    T",
    "BITPIX  =                  -32",
    "NAXIS   =                    2",
    "NAXIS1  =                  100",
    "NAXIS2  =                  100",
    "CTYPE1  = 'RA---TAN'",
    "CTYPE2  = 'DEC--TAN'",
    "CRVAL1  =                 10.0",
    "CRVAL2  =                 20.0",
    "CRPIX1  =                 50.0",
    "CRPIX2  =                 50.0",
    "CDELT1  =                -0.01",
    "CDELT2  =                 0.01",
];

/// Sec.8.2 Table 22 defines `CROTAi` as indexed and attached to the
/// latitude axis. It used to be read only when `NAXIS == 2`, so a
/// cube came back unrotated -- a third of a degree off.
#[test]
fn crota_applies_to_a_cube_not_just_a_2d_image() {
    let mut cube: Vec<&str> = TAN_2D.to_vec();
    cube[2] = "NAXIS   =                    3";
    let extra = [
        "NAXIS3  =                    2",
        "CTYPE3  = 'FREQ'",
        "CRVAL3  =                  1.0",
        "CRPIX3  =                  1.0",
        "CDELT3  =                  1.0",
    ];
    cube.extend_from_slice(&extra);

    let mut flat: Vec<&str> = TAN_2D.to_vec();
    flat.push("CROTA2  =                 30.0");
    cube.push("CROTA2  =                 30.0");

    let (fx, fy) = sky(&wcs_of(&flat), 0.0, 0.0);
    let (cx, cy) = sky(&wcs_of(&cube), 0.0, 0.0);
    assert!(
        (fx - cx).abs() < 1e-12 && (fy - cy).abs() < 1e-12,
        "CROTA2 dropped for NAXIS=3: 2-D gives ({fx}, {fy}), cube gives ({cx}, {cy})"
    );

    // And the rotation is really applied, rather than both agreeing
    // on the unrotated answer.
    let (ux, uy) = sky(&wcs_of(TAN_2D), 0.0, 0.0);
    assert!(
        (fx - ux).abs() > 1e-3 || (fy - uy).abs() > 1e-3,
        "CROTA2 had no effect at all"
    );
}

/// `CROTAi` follows the latitude axis, so a cube whose celestial pair
/// is axes 2 and 3 carries `CROTA3`.
#[test]
fn crota_follows_the_latitude_axis() {
    let cube = [
        "SIMPLE  =                    T",
        "BITPIX  =                  -32",
        "NAXIS   =                    3",
        "NAXIS1  =                   16",
        "NAXIS2  =                  100",
        "NAXIS3  =                  100",
        "CTYPE1  = 'FREQ'",
        "CTYPE2  = 'RA---TAN'",
        "CTYPE3  = 'DEC--TAN'",
        "CRVAL1  =                  1.0",
        "CRVAL2  =                 10.0",
        "CRVAL3  =                 20.0",
        "CRPIX1  =                  1.0",
        "CRPIX2  =                 50.0",
        "CRPIX3  =                 50.0",
        "CDELT1  =                  1.0",
        "CDELT2  =                -0.01",
        "CDELT3  =                 0.01",
        "CROTA3  =                 30.0",
    ];
    let w = wcs_of(&cube);
    let m = w.linear().matrix_row_major();
    // The FREQ axis keeps its own CDELT and picks up no rotation...
    assert!((m[0] - 1.0).abs() < 1e-12, "FREQ axis scale changed");
    assert!(
        m[1].abs() < 1e-12 && m[2].abs() < 1e-12,
        "FREQ axis rotated"
    );
    // ...while the celestial 2x2 block carries the 30 degree rotation.
    let c30 = 30_f64.to_radians().cos();
    assert!(
        (m[4] / -0.01 - c30).abs() < 1e-12,
        "no rotation on the sky pair"
    );
    assert!(m[5].abs() > 1e-12, "sky pair has no off-diagonal term");
}

/// Sec.8.2: "`CDELTi` ... The value *must not* be zero", and the
/// matrices "*must not* be singular". A zero used to produce a NaN
/// matrix that inverted silently, so every transform returned
/// `Ok(NaN)`.
#[test]
fn zero_cdelt_with_crota_is_rejected() {
    for (i, bad) in [
        "CDELT1  =                  0.0",
        "CDELT2  =                  0.0",
    ]
    .into_iter()
    .enumerate()
    {
        let mut cards: Vec<&str> = TAN_2D.to_vec();
        cards[11 + i] = bad;
        cards.push("CROTA2  =                 30.0");
        let err = Wcs::from_header(&header_of(&cards), ' ');
        assert!(err.is_err(), "CDELT{} = 0 accepted", i + 1);
    }
}

/// Sec.8.4: "pairs of the form `yzLN` and `yzLT` *may* be used as
/// well" for planetary, lunar and solar systems. These used to parse
/// as non-celestial, so `pixel_to_world` skipped the projection and
/// returned the linear values with no error.
#[test]
fn planetary_lnlt_ctype_is_celestial() {
    let mut cards: Vec<&str> = TAN_2D.to_vec();
    cards[5] = "CTYPE1  = 'MELN-TAN'";
    cards[6] = "CTYPE2  = 'MELT-TAN'";
    let w = wcs_of(&cards);
    assert!(
        w.is_celestial(),
        "yzLN/yzLT not recognized as a celestial pair"
    );
    assert_eq!(w.celestial_axes(), Some((0, 1)));

    // The projection is really applied: a TAN plate differs from the
    // linear values away from the reference pixel.
    let world = w.pixel_to_world(&[0.0, 0.0]).unwrap();
    let linear_lon = 10.0 + -0.01 * (0.0 + 1.0 - 50.0);
    assert!(
        (world[0] - linear_lon).abs() > 1e-6,
        "no projection applied: got {world:?}"
    );
}

/// Sec.8.4 also admits the generic `xLON`/`xLAT` spelling, which is
/// what `CelestialFrame::Other` serializes to -- so failing to parse
/// it back lost the celestial block on a round trip.
#[test]
fn generic_xlon_xlat_round_trips() {
    let mut cards: Vec<&str> = TAN_2D.to_vec();
    cards[5] = "CTYPE1  = 'XLON-TAN'";
    cards[6] = "CTYPE2  = 'XLAT-TAN'";
    let w = wcs_of(&cards);
    assert!(w.is_celestial(), "XLON/XLAT not recognized");

    let reparsed = Wcs::from_header(&w.to_header(' ').unwrap(), ' ')
        .unwrap()
        .unwrap();
    assert!(reparsed.is_celestial(), "XLON/XLAT lost on round trip");
    let a = sky(&w, 10.0, 20.0);
    let b = sky(&reparsed, 10.0, 20.0);
    assert!((a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
}

/// A named frame must keep its identity rather than falling into the
/// generic `xLON` rule that also matches `GLON`.
#[test]
fn named_frames_win_over_the_generic_rule() {
    let mut cards: Vec<&str> = TAN_2D.to_vec();
    cards[5] = "CTYPE1  = 'GLON-TAN'";
    cards[6] = "CTYPE2  = 'GLAT-TAN'";
    let w = wcs_of(&cards);
    let frame = w.celestial.as_ref().unwrap().pair.frame;
    assert_eq!(frame, fitsy::wcs::CelestialFrame::Galactic);
}

/// Sec.8.2, LONPOLE: "`phi_0` is zero unless a non-zero value has
/// been set for `PVi_1a`, which is associated with the *longitude*
/// axis". Only the latitude axis used to be read, so moving the
/// fiducial point changed nothing.
#[test]
fn longitude_axis_pv_relocates_the_fiducial_point() {
    let mut cards: Vec<&str> = TAN_2D.to_vec();
    cards[5] = "CTYPE1  = 'RA---SFL'";
    cards[6] = "CTYPE2  = 'DEC--SFL'";
    let plain = wcs_of(&cards);

    let mut moved_cards = cards.clone();
    moved_cards.push("PV1_1   =                 90.0");
    moved_cards.push("PV1_2   =                 20.0");
    let moved = wcs_of(&moved_cards);

    assert!((moved.celestial.as_ref().unwrap().rotation.phi0 - 90.0).abs() < 1e-12);
    assert!((moved.celestial.as_ref().unwrap().rotation.theta0 - 20.0).abs() < 1e-12);

    let a = sky(&plain, 0.0, 0.0);
    let b = sky(&moved, 0.0, 0.0);
    assert!(
        (a.0 - b.0).abs() > 1e-6 || (a.1 - b.1).abs() > 1e-6,
        "PV1_1/PV1_2 ignored: both give {a:?}"
    );

    // The reference pixel still lands on CRVAL: moving the fiducial
    // point moves the projection origin, not the reference point.
    // (CRPIX is 1-based, this API 0-based.)
    let (ra, dec) = sky(&moved, 49.0, 49.0);
    assert!(
        (ra - 10.0).abs() < 1e-9 && (dec - 20.0).abs() < 1e-9,
        "reference pixel no longer maps to CRVAL: ({ra}, {dec})"
    );

    // ... and it survives serialization.
    let reparsed = Wcs::from_header(&moved.to_header(' ').unwrap(), ' ')
        .unwrap()
        .unwrap();
    let c = sky(&reparsed, 0.0, 0.0);
    assert!(
        (b.0 - c.0).abs() < 1e-9 && (b.1 - c.1).abs() < 1e-9,
        "relocated fiducial point lost on round trip"
    );
}

/// Sec.4.4.1.1: a zero `NAXIS` "signifies that no data follow the
/// header". Decoding one used to fail, because the empty axis
/// product is 1 rather than 0.
#[test]
fn empty_primary_hdu_decodes_to_an_empty_array() {
    let bytes = block(&[
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
    ]);
    let f = FitsFile::from_bytes(bytes).unwrap();
    let Hdu::Image(img) = f.hdu(0).unwrap() else {
        panic!("expected an image");
    };
    assert!(img.read_physical().unwrap().as_slice().is_empty());
    assert!(img.read_physical_f32().unwrap().as_slice().is_empty());
    assert!(img.read_raw::<u8>().unwrap().as_slice().is_empty());
    assert!(img.read_raw_dyn().unwrap().axes().is_empty());
}

/// Opening an appender must not touch the file. It used to truncate
/// bytes after the last HDU that `FitsFile::open` accepts, as a side
/// effect of merely being constructed.
#[test]
fn appender_open_does_not_truncate() {
    let dir = std::env::temp_dir().join("fitsy_standards_regressions");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("appender_open_does_not_truncate.fits");
    let mut bytes = block(&[
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
    ]);
    bytes.extend_from_slice(b"trailing vendor metadata");
    std::fs::write(&path, &bytes).unwrap();
    let before = std::fs::metadata(&path).unwrap().len();

    let app = FitsAppender::open(&path).unwrap();
    drop(app);
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        before,
        "opening an appender changed the file"
    );

    // Appending overwrites the tail and trims what it did not cover.
    let (header, data) = ImageBuilder::new(vec![4_u64, 4], vec![0_u8; 16])
        .unwrap()
        .build()
        .unwrap();
    let mut app = FitsAppender::open(&path).unwrap();
    app.append_hdu(&header, &data).unwrap();
    let n = app.finish().unwrap();
    assert_eq!(n, 2);
    let len = std::fs::metadata(&path).unwrap().len();
    assert_eq!(len % BLOCK as u64, 0, "appended file is not block-aligned");
    let f = FitsFile::open(&path).unwrap();
    assert_eq!(f.len(), 2);
    std::fs::remove_file(&path).ok();
}

/// A DSS plate replaces the celestial pipeline, but the other axes
/// still come from the linear one. `world_to_pixel` used to return 0
/// for every one of them.
#[test]
fn dss_inverse_keeps_non_celestial_axes() {
    let f = FitsFile::open("tests/data/dss_plate.fits").unwrap();
    let wcs = f.wcs(0, ' ').unwrap().expect("dss_plate.fits has a WCS");
    assert!(wcs.dss.is_some(), "fixture is not a DSS plate");
    let (lon, lat) = wcs.celestial_axes().unwrap();
    let world = wcs.pixel_to_world(&[100.0, 120.0]).unwrap();
    let pix = wcs.world_to_pixel(&world).unwrap();
    assert!((pix[lon] - 100.0).abs() < 1e-6, "lon axis: {pix:?}");
    assert!((pix[lat] - 120.0).abs() < 1e-6, "lat axis: {pix:?}");
}

/// Sec.4.1.2.1 makes `-` and `_` distinct legal keyword characters.
/// The fallback that lets `MJD-OBS` find a misspelled `MJD_OBS` must
/// not run in reverse, or `CD1_1` answers with a `CD1-1` card.
#[test]
fn keyword_fallback_is_one_directional() {
    let h = header_of(&[
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "MJD_OBS =              57754.0",
        "CD1-1   =                  1.0",
    ]);
    assert!(h.first("MJD-OBS").is_some(), "MJD-OBS should find MJD_OBS");
    assert!(h.first("CD1_1").is_none(), "CD1_1 must not find CD1-1");
}

/// Standard Sec. 4.1.2.2: `= ` in bytes 9 and 10 marks a value field
/// "unless it is one of the commentary keywords ... which by
/// definition have no value". Sec. 4.4.2 repeats it: a commentary
/// keyword "shall have no associated value even if the value
/// indicator characters appear in bytes 9 and 10".
///
/// Regression: fitsy applied the value-indicator test to commentary
/// keywords, so `HISTORY = text` parsed as a value card. A strict
/// parse then failed outright on a file astropy reads without
/// complaint, and a lenient parse hid the card from
/// [`Header::history`], silently dropping provenance a caller asked
/// for by name.
#[test]
fn commentary_keywords_ignore_a_value_indicator() {
    let h = header_of(&[
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "HISTORY = looks like a value",
        "HISTORY plain entry",
        "COMMENT = also value-shaped",
    ]);
    // Bytes 9 through 80 are the text, so the `= ` belongs to it.
    let history: Vec<&str> = h.history().collect();
    assert_eq!(history, vec!["= looks like a value", "plain entry"]);
    let comments: Vec<&str> = h.comments().collect();
    assert_eq!(comments, vec!["= also value-shaped"]);
    // No commentary card leaks into the value namespace.
    assert!(h.first("HISTORY").is_none());
    assert!(h.first("COMMENT").is_none());
}

/// Commentary text longer than one card is split at 72 characters,
/// which is bytes 9 through 80, and rejoins on read.
#[test]
fn long_commentary_splits_on_the_card_boundary() {
    use fitsy::header::CommentaryKind;
    let text = "z".repeat(200);
    let mut h = header_of(&[
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
    ]);
    h.push_commentary(CommentaryKind::History, &text);
    let bytes = h.to_bytes().unwrap();
    let (re, _) = Header::parse(&bytes, 0).unwrap();
    let parts: Vec<&str> = re.history().collect();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].len(), 72);
    assert_eq!(parts[1].len(), 72);
    assert_eq!(parts.concat(), text);
}

/// Standard Sec. 4.4.2: a commentary keyword has no associated value.
/// Writing `COMMENT = 'text'` emits a card the Standard forbids, and
/// one that reads back as the commentary text `= 'text'`.
///
/// Regression: `Header::push` and `Header::set` built a value card for
/// `COMMENT`, `HISTORY` and the blank keyword. The card did not
/// round-trip, and in Python `header["COMMENT"] = "x"` then
/// `header["COMMENT"]` raised `KeyError`.
#[test]
fn commentary_keywords_write_commentary_cards() {
    use fitsy::header::Value;
    let mut h = Header::empty();
    h.push("SIMPLE", Value::Logical(true), None).unwrap();
    h.push("BITPIX", Value::Integer(8), None).unwrap();
    h.push("NAXIS", Value::Integer(0), None).unwrap();
    h.push("COMMENT", Value::String("note".into()), None)
        .unwrap();
    h.set("HISTORY", Value::String("step".into()), None)
        .unwrap();

    // The emitted cards carry no value indicator.
    let bytes = h.to_bytes().unwrap();
    let cards: Vec<String> = bytes
        .chunks(80)
        .map(|c| String::from_utf8_lossy(c).trim_end().to_string())
        .collect();
    assert!(cards.contains(&"COMMENT note".to_string()), "{cards:?}");
    assert!(cards.contains(&"HISTORY step".to_string()), "{cards:?}");

    // And they survive a round trip as commentary.
    let (re, _) = Header::parse(&bytes, 0).unwrap();
    assert_eq!(re.comments().collect::<Vec<_>>(), vec!["note"]);
    assert_eq!(re.history().collect::<Vec<_>>(), vec!["step"]);
    // A commentary card holds no value, so it stays out of the value
    // namespace.
    assert!(re.first("COMMENT").is_none());
    assert!(re.first("HISTORY").is_none());
}
