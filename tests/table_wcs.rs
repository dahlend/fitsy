//! WCS keywords carried in a binary table (Standard Sec.8.2, Table 22):
//! the pixel-list (`TCTYPn`) and BINTABLE-vector (`iCTYPn`) forms.
//!
//! Each test builds the table-form header and the equivalent image-form
//! header and asserts the two transform identically -- Table 22 defines
//! the forms as spellings of one description, so any divergence is a
//! translation bug.

use fitsy::header::Header;
use fitsy::wcs::{TableWcs, Wcs};

const CARD: usize = 80;
const BLOCK: usize = 2880;

fn header(cards: &[&str]) -> Header {
    let mut buf = Vec::new();
    for c in cards {
        assert!(c.len() <= CARD, "card too long: {c}");
        let mut b = [b' '; CARD];
        b[..c.len()].copy_from_slice(c.as_bytes());
        buf.extend_from_slice(&b);
    }
    let mut end = [b' '; CARD];
    end[..3].copy_from_slice(b"END");
    buf.extend_from_slice(&end);
    while !buf.len().is_multiple_of(BLOCK) {
        buf.push(b' ');
    }
    Header::parse(&buf, 0).expect("header parses").0
}

/// A BINTABLE shell with three columns: TIME, X, Y.
fn event_table(extra: &[&str]) -> Header {
    let mut cards: Vec<&str> = vec![
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                   16",
        "NAXIS2  =                    4",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    3",
        "TTYPE1  = 'TIME    '",
        "TFORM1  = '1D      '",
        "TTYPE2  = 'X       '",
        "TFORM2  = '1E      '",
        "TTYPE3  = 'Y       '",
        "TFORM3  = '1E      '",
    ];
    cards.extend_from_slice(extra);
    header(&cards)
}

fn image(extra: &[&str]) -> Header {
    image_naxis(2, extra)
}

fn image_naxis(naxis: usize, extra: &[&str]) -> Header {
    let naxis_card = format!("NAXIS   = {naxis:>20}");
    let mut cards: Vec<&str> = vec![
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        &naxis_card,
    ];
    cards.extend_from_slice(extra);
    header(&cards)
}

/// Round-trip a handful of pixel positions through both descriptions
/// and require agreement to better than a micro-arcsecond.
fn assert_same_transform(a: &Wcs, b: &Wcs, probes: &[[f64; 2]]) {
    for p in probes {
        let wa = a.pixel_to_world(p).expect("table WCS transforms");
        let wb = b.pixel_to_world(p).expect("image WCS transforms");
        assert_eq!(wa.len(), wb.len(), "axis count at {p:?}");
        for (k, (x, y)) in wa.iter().zip(&wb).enumerate() {
            assert!(
                (x - y).abs() < 1e-10,
                "axis {k} at pixel {p:?}: table {x} vs image {y}"
            );
        }
    }
}

const PROBES: &[[f64; 2]] = &[
    [0.0, 0.0],
    [4095.5, 4095.5],
    [100.0, 8000.0],
    [-50.0, 250.5],
];

/// The primary pixel-list form, laid out as a Chandra/XMM event list:
/// the sky columns carry `TCTYPn`/`TCRVLn`/`TCRPXn`/`TCDLTn`/`TCUNIn`
/// and the frame keywords use the column-indexed global spellings.
#[test]
fn pixel_list_primary_matches_image_form() {
    let t = event_table(&[
        "TCTYP2  = 'RA---TAN'",
        "TCRVL2  =                150.0",
        "TCRPX2  =               4096.5",
        "TCDLT2  =      -1.36667E-04",
        "TCUNI2  = 'deg     '",
        "TCTYP3  = 'DEC--TAN'",
        "TCRVL3  =                  2.5",
        "TCRPX3  =               4096.5",
        "TCDLT3  =       1.36667E-04",
        "TCUNI3  = 'deg     '",
        // Sec.8.2: the column index on a global keyword is not
        // meaningful. Point these at the TIME column to prove it.
        "EQUI1   =               2000.0",
        "RADE1   = 'FK5     '",
        "MJDOB1  =              52000.0",
    ]);
    let tw = TableWcs::from_pixel_list(&t, ' ')
        .expect("pixel list parses")
        .expect("pixel list present");

    assert_eq!(tw.colax, vec![2, 3], "axis 1 is the lowest column");
    assert_eq!(tw.column, None);
    assert_eq!(tw.wcs.naxis(), 2);
    assert_eq!(
        tw.wcs
            .axes()
            .iter()
            .map(|a| a.ctype.clone())
            .collect::<Vec<_>>(),
        vec!["RA---TAN", "DEC--TAN"]
    );
    assert_eq!(tw.wcs.equinox, Some(2000.0));
    assert_eq!(tw.wcs.mjd_obs, Some(52000.0));
    // No pixel array behind a pixel list, so no extent to report.
    assert!(tw.wcs.pixel_shape.is_none());

    let img = Wcs::from_header(
        &image(&[
            "CTYPE1  = 'RA---TAN'",
            "CRVAL1  =                150.0",
            "CRPIX1  =               4096.5",
            "CDELT1  =      -1.36667E-04",
            "CUNIT1  = 'deg     '",
            "CTYPE2  = 'DEC--TAN'",
            "CRVAL2  =                  2.5",
            "CRPIX2  =               4096.5",
            "CDELT2  =       1.36667E-04",
            "CUNIT2  = 'deg     '",
            "EQUINOX =               2000.0",
            "RADESYS = 'FK5     '",
        ]),
        ' ',
    )
    .unwrap()
    .unwrap();
    assert_same_transform(&tw.wcs, &img, PROBES);
}

/// Parity against `wcslib`, via `astropy.wcs.WCS(header,
/// keysel=['pixel'], colsel=[2, 3])` on the header above and
/// `keysel=['binary'], colsel=[5]` on the vector-column one. These are
/// the numbers the reference implementation returns for 0-based pixels;
/// they pin the column-to-axis mapping and the `12PC5` index order,
/// neither of which the standard spells out.
#[test]
fn table_wcs_matches_wcslib() {
    let t = event_table(&[
        "TCTYP2  = 'RA---TAN'",
        "TCRVL2  =                150.0",
        "TCRPX2  =               4096.5",
        "TCDLT2  =      -1.36667E-04",
        "TCUNI2  = 'deg     '",
        "TCTYP3  = 'DEC--TAN'",
        "TCRVL3  =                  2.5",
        "TCRPX3  =               4096.5",
        "TCDLT3  =       1.36667E-04",
        "TCUNI3  = 'deg     '",
    ]);
    let tw = TableWcs::from_pixel_list(&t, ' ').unwrap().unwrap();
    let reference: &[([f64; 2], [f64; 2])] = &[
        (
            [0.0, 0.0],
            [150.559_996_244_809_98, 1.940_205_502_006_725_5],
        ),
        ([4095.5, 4095.5], [150.0, 2.5]),
        (
            [100.0, 8000.0],
            [150.546_778_959_539_44, 3.033_462_996_985_149_7],
        ),
        (
            [-50.0, 250.5],
            [150.566_847_308_321_8, 1.974_433_563_350_117_5],
        ),
    ];
    for (pixel, expected) in reference {
        let got = tw.wcs.pixel_to_world(pixel).unwrap();
        for (k, (g, e)) in got.iter().zip(expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-11,
                "pixel-list axis {k} at {pixel:?}: {g} vs wcslib {e}"
            );
        }
    }

    let v = header(&[
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                 4096",
        "NAXIS2  =                    3",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    5",
        "TFORM5  = '1024E   '",
        "TDIM5   = '(32,32)'",
        "1CTYP5  = 'RA---SIN'",
        "1CRVL5  =                 83.6",
        "1CRPX5  =                 16.5",
        "1CDLT5  =               -0.001",
        "2CTYP5  = 'DEC--SIN'",
        "2CRVL5  =                 22.0",
        "2CRPX5  =                 16.5",
        "2CDLT5  =                0.001",
        "12PC5   =                  1.0",
        "21PC5   =                  0.0",
    ]);
    let vw = TableWcs::from_table_column(&v, 5, ' ').unwrap().unwrap();
    let reference: &[([f64; 2], [f64; 2])] = &[
        ([0.0, 0.0], [83.633_430_925_358_11, 21.984_496_611_892_79]),
        (
            [16.0, 16.0],
            [83.598_921_461_454_52, 22.000_499_996_474_197],
        ),
        ([31.0, 5.0], [83.594_607_725_444_48, 21.989_499_911_802_9]),
    ];
    for (pixel, expected) in reference {
        let got = vw.wcs.pixel_to_world(pixel).unwrap();
        for (k, (g, e)) in got.iter().zip(expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-11,
                "vector-column axis {k} at {pixel:?}: {g} vs wcslib {e}"
            );
        }
    }
}

/// Table 22's alternate spellings shorten the root to keep the keyword
/// within 8 characters: `TCTYna`, `TCRVna`, `TCRPna`, `TCDEna`,
/// `TCUNna`. The alternate must be selectable independently.
#[test]
fn pixel_list_alternate_uses_shortened_roots() {
    let t = event_table(&[
        "TCTYP2  = 'RA---TAN'",
        "TCRVL2  =                150.0",
        "TCRPX2  =               4096.5",
        "TCDLT2  =      -1.36667E-04",
        "TCTYP3  = 'DEC--TAN'",
        "TCRVL3  =                  2.5",
        "TCRPX3  =               4096.5",
        "TCDLT3  =       1.36667E-04",
        // Alternate A: galactic, on the same two columns.
        "TCTY2A  = 'GLON-AIT'",
        "TCRV2A  =                 45.0",
        "TCRP2A  =                100.0",
        "TCDE2A  =                 -0.01",
        "TCUN2A  = 'deg     '",
        "TCTY3A  = 'GLAT-AIT'",
        "TCRV3A  =                -30.0",
        "TCRP3A  =                200.0",
        "TCDE3A  =                  0.01",
        "TCUN3A  = 'deg     '",
    ]);

    assert_eq!(TableWcs::pixel_list_alternates(&t), vec![' ', 'A']);

    let primary = TableWcs::from_pixel_list(&t, ' ').unwrap().unwrap();
    assert_eq!(
        primary
            .wcs
            .axes()
            .iter()
            .map(|a| a.ctype.clone())
            .collect::<Vec<_>>(),
        vec!["RA---TAN", "DEC--TAN"]
    );

    let alt = TableWcs::from_pixel_list(&t, 'A').unwrap().unwrap();
    assert_eq!(alt.colax, vec![2, 3]);
    assert_eq!(
        alt.wcs
            .axes()
            .iter()
            .map(|a| a.ctype.clone())
            .collect::<Vec<_>>(),
        vec!["GLON-AIT", "GLAT-AIT"]
    );

    let img = Wcs::from_header(
        &image(&[
            "CTYPE1A = 'GLON-AIT'",
            "CRVAL1A =                 45.0",
            "CRPIX1A =                100.0",
            "CDELT1A =                 -0.01",
            "CUNIT1A = 'deg     '",
            "CTYPE2A = 'GLAT-AIT'",
            "CRVAL2A =                -30.0",
            "CRPIX2A =                200.0",
            "CDELT2A =                  0.01",
            "CUNIT2A = 'deg     '",
        ]),
        'A',
    )
    .unwrap()
    .unwrap();
    assert_same_transform(&alt.wcs, &img, PROBES);

    // An alternate with no keywords at all is absent, not an error.
    assert!(TableWcs::from_pixel_list(&t, 'B').unwrap().is_none());
}

/// `TPn_ka` indexes *columns* on both sides of the underscore, so the
/// matrix must be permuted into axis order. Columns 2 and 3 here map to
/// axes 1 and 2, and a non-symmetric matrix proves the mapping is not
/// accidentally transposed.
#[test]
fn pixel_list_matrix_is_indexed_by_column() {
    let common = [
        "TCTYP2  = 'RA---TAN'",
        "TCRVL2  =                150.0",
        "TCRPX2  =                 10.0",
        "TCDLT2  =                 -0.01",
        "TCTYP3  = 'DEC--TAN'",
        "TCRVL3  =                  2.5",
        "TCRPX3  =                 20.0",
        "TCDLT3  =                  0.01",
    ];
    let mut cards = common.to_vec();
    cards.extend_from_slice(&[
        "TP2_2   =                  0.60",
        "TP2_3   =                 -0.80",
        "TP3_2   =                  0.80",
        "TP3_3   =                  0.60",
    ]);
    let tw = TableWcs::from_pixel_list(&event_table(&cards), ' ')
        .unwrap()
        .unwrap();

    let img = Wcs::from_header(
        &image(&[
            "CTYPE1  = 'RA---TAN'",
            "CRVAL1  =                150.0",
            "CRPIX1  =                 10.0",
            "CDELT1  =                 -0.01",
            "CTYPE2  = 'DEC--TAN'",
            "CRVAL2  =                  2.5",
            "CRPIX2  =                 20.0",
            "CDELT2  =                  0.01",
            "PC1_1   =                  0.60",
            "PC1_2   =                 -0.80",
            "PC2_1   =                  0.80",
            "PC2_2   =                  0.60",
        ]),
        ' ',
    )
    .unwrap()
    .unwrap();
    assert_same_transform(&tw.wcs, &img, PROBES);

    // The long form TPCn_ka, legitimised by the WCS papers' errata, is
    // the same matrix.
    let mut long = common.to_vec();
    long.extend_from_slice(&[
        "TPC2_2  =                  0.60",
        "TPC2_3  =                 -0.80",
        "TPC3_2  =                  0.80",
        "TPC3_3  =                  0.60",
    ]);
    let tw_long = TableWcs::from_pixel_list(&event_table(&long), ' ')
        .unwrap()
        .unwrap();
    assert_same_transform(&tw_long.wcs, &img, PROBES);
}

/// Non-adjacent, out-of-order coordinate columns still number their
/// axes 1..N by ascending column, and `colax` reports the mapping.
#[test]
fn pixel_list_axis_order_follows_ascending_column() {
    let t = header(&[
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                   16",
        "NAXIS2  =                    4",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                   12",
        // Declared latitude-first in the header; column order decides.
        "TCTYP11 = 'DEC--TAN'",
        "TCRVL11 =                  2.5",
        "TCRPX11 =                 20.0",
        "TCDLT11 =                  0.01",
        "TCTYP7  = 'RA---TAN'",
        "TCRVL7  =                150.0",
        "TCRPX7  =                 10.0",
        "TCDLT7  =                 -0.01",
    ]);
    let tw = TableWcs::from_pixel_list(&t, ' ').unwrap().unwrap();
    assert_eq!(tw.colax, vec![7, 11]);
    assert_eq!(
        tw.wcs
            .axes()
            .iter()
            .map(|a| a.ctype.clone())
            .collect::<Vec<_>>(),
        vec!["RA---TAN", "DEC--TAN"]
    );

    let img = Wcs::from_header(
        &image(&[
            "CTYPE1  = 'RA---TAN'",
            "CRVAL1  =                150.0",
            "CRPIX1  =                 10.0",
            "CDELT1  =                 -0.01",
            "CTYPE2  = 'DEC--TAN'",
            "CRVAL2  =                  2.5",
            "CRPIX2  =                 20.0",
            "CDELT2  =                  0.01",
        ]),
        ' ',
    )
    .unwrap()
    .unwrap();
    assert_same_transform(&tw.wcs, &img, PROBES);
}

/// `TVn_ma` carries projection parameters; the column index becomes the
/// axis index. `LONPna`'s column index is ignored per Sec.8.2.
#[test]
fn pixel_list_projection_parameters_and_pole() {
    let t = event_table(&[
        "TCTYP2  = 'RA---ZPN'",
        "TCRVL2  =                150.0",
        "TCRPX2  =                 10.0",
        "TCDLT2  =                 -0.01",
        "TCTYP3  = 'DEC--ZPN'",
        "TCRVL3  =                  2.5",
        "TCRPX3  =                 20.0",
        "TCDLT3  =                  0.01",
        "TV3_1   =                  1.00",
        "TV3_3   =                220.00",
        "LONP1   =                180.0",
    ]);
    let tw = TableWcs::from_pixel_list(&t, ' ').unwrap().unwrap();

    let img = Wcs::from_header(
        &image(&[
            "CTYPE1  = 'RA---ZPN'",
            "CRVAL1  =                150.0",
            "CRPIX1  =                 10.0",
            "CDELT1  =                 -0.01",
            "CTYPE2  = 'DEC--ZPN'",
            "CRVAL2  =                  2.5",
            "CRPIX2  =                 20.0",
            "CDELT2  =                  0.01",
            "PV2_1   =                  1.00",
            "PV2_3   =                220.00",
            "LONPOLE =                180.0",
        ]),
        ' ',
    )
    .unwrap()
    .unwrap();
    // ZPN's radial polynomial is only invertible near the reference
    // point for these coefficients, so probe close to it.
    assert_same_transform(
        &tw.wcs,
        &img,
        &[[9.0, 19.0], [10.0, 20.0], [30.0, 45.0], [-5.0, 5.0]],
    );
}

/// The BINTABLE-vector form: an image in a vector cell of column 5,
/// with axis number as a keyword *prefix*. `TDIM5` supplies the extent.
#[test]
fn bintable_vector_column_matches_image_form() {
    let t = header(&[
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                 4096",
        "NAXIS2  =                    3",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    5",
        "TFORM5  = '1024E   '",
        "TDIM5   = '(32,32)'",
        "1CTYP5  = 'RA---SIN'",
        "1CRVL5  =                 83.6",
        "1CRPX5  =                 16.5",
        "1CDLT5  =                -0.001",
        "1CUNI5  = 'deg     '",
        "2CTYP5  = 'DEC--SIN'",
        "2CRVL5  =                 22.0",
        "2CRPX5  =                 16.5",
        "2CDLT5  =                 0.001",
        "2CUNI5  = 'deg     '",
        "12PC5   =                  1.00",
        "21PC5   =                  0.00",
        "EQUI5   =               2000.0",
    ]);
    let tw = TableWcs::from_table_column(&t, 5, ' ')
        .expect("column WCS parses")
        .expect("column WCS present");

    assert_eq!(tw.column, Some(5));
    assert!(tw.colax.is_empty());
    assert_eq!(
        tw.wcs
            .axes()
            .iter()
            .map(|a| a.ctype.clone())
            .collect::<Vec<_>>(),
        vec!["RA---SIN", "DEC--SIN"]
    );
    assert_eq!(tw.wcs.equinox, Some(2000.0));
    // TDIMn is the cell shape in FITS axis order, i.e. NAXISn.
    assert_eq!(tw.wcs.pixel_shape, Some(vec![32, 32]));

    let img = Wcs::from_header(
        &image(&[
            "CTYPE1  = 'RA---SIN'",
            "CRVAL1  =                 83.6",
            "CRPIX1  =                 16.5",
            "CDELT1  =                -0.001",
            "CUNIT1  = 'deg     '",
            "CTYPE2  = 'DEC--SIN'",
            "CRVAL2  =                 22.0",
            "CRPIX2  =                 16.5",
            "CDELT2  =                 0.001",
            "CUNIT2  = 'deg     '",
            "PC1_2   =                  1.00",
            "PC2_1   =                  0.00",
            "EQUINOX =               2000.0",
        ]),
        ' ',
    )
    .unwrap()
    .unwrap();
    assert_same_transform(&tw.wcs, &img, &[[0.0, 0.0], [16.0, 16.0], [31.0, 5.0]]);

    // Columns without WCS keywords are absent, not an error.
    assert!(TableWcs::from_table_column(&t, 4, ' ').unwrap().is_none());
    assert_eq!(TableWcs::image_columns(&t), vec![(5, vec![' '])]);
}

/// A spectral pixel-list axis: the table form must reach the Paper III
/// machinery, including the rest frequency from `RFRQna`.
#[test]
fn pixel_list_spectral_axis_uses_table_rest_frequency() {
    let t = event_table(&[
        "TCTYP2  = 'VOPT-F2W'",
        "TCRVL2  =              1.0E+05",
        "TCRPX2  =                  1.0",
        "TCDLT2  =              1.0E+03",
        "TCUNI2  = 'm/s     '",
        "RFRQ2   =           1.42040575E+09",
        "SPEC2   = 'BARYCENT'",
    ]);
    let tw = TableWcs::from_pixel_list(&t, ' ').unwrap().unwrap();
    assert_eq!(tw.colax, vec![2]);
    assert_eq!(
        tw.wcs
            .spectral_frame
            .as_ref()
            .and_then(|f| f.specsys.as_deref()),
        Some("BARYCENT")
    );
    assert_eq!(tw.wcs.spectral.len(), 1, "spectral axis recognised");

    let img = Wcs::from_header(
        &image_naxis(
            1,
            &[
                "CTYPE1  = 'VOPT-F2W'",
                "CRVAL1  =              1.0E+05",
                "CRPIX1  =                  1.0",
                "CDELT1  =              1.0E+03",
                "CUNIT1  = 'm/s     '",
                "RESTFRQ =           1.42040575E+09",
                "SPECSYS = 'BARYCENT'",
            ],
        ),
        ' ',
    )
    .unwrap()
    .unwrap();
    for p in [0.0, 1.0, 17.5, -4.0] {
        let a = tw.wcs.pixel_to_world(&[p]).unwrap();
        let b = img.pixel_to_world(&[p]).unwrap();
        assert!(
            (a[0] - b[0]).abs() <= 1e-6 * b[0].abs().max(1.0),
            "pixel {p}: table {} vs image {}",
            a[0],
            b[0]
        );
    }
}

/// Image-form keywords in the table header supply defaults for the
/// representation-wide values, and the column-indexed form overrides
/// them. Per-axis image keywords must *not* leak in: their axis numbers
/// describe a different image.
#[test]
fn global_defaults_inherit_but_axis_keywords_do_not() {
    let t = event_table(&[
        "TCTYP2  = 'RA---TAN'",
        "TCRVL2  =                150.0",
        "TCRPX2  =                 10.0",
        "TCDLT2  =                 -0.01",
        "TCTYP3  = 'DEC--TAN'",
        "TCRVL3  =                  2.5",
        "TCRPX3  =                 20.0",
        "TCDLT3  =                  0.01",
        // Image-form default, no table-form override.
        "RADESYS = 'GALACTIC'",
        // Image-form default that IS overridden by the table form.
        "EQUINOX =               1950.0",
        "EQUI9   =               2000.0",
        // Stale per-axis image keywords describing something else.
        "CTYPE1  = 'WAVE    '",
        "CRVAL1  =              9.9E+09",
        "CDELT1  =                999.0",
        "CRPIX1  =                999.0",
    ]);
    let tw = TableWcs::from_pixel_list(&t, ' ').unwrap().unwrap();
    assert_eq!(
        tw.wcs
            .axes()
            .iter()
            .map(|a| a.ctype.clone())
            .collect::<Vec<_>>(),
        vec!["RA---TAN", "DEC--TAN"]
    );
    assert_eq!(tw.wcs.crval(), vec![150.0, 2.5]);
    assert_eq!(tw.wcs.equinox, Some(2000.0), "EQUI9 overrides EQUINOX");
}

/// Table 22's forms only apply to a `BINTABLE`; asking an image HDU for
/// one should say so rather than silently returning nothing.
#[test]
fn file_level_entry_points() {
    use fitsy::FitsFile;

    let mut buf = Vec::new();
    for c in [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
        "END",
    ] {
        let mut b = [b' '; CARD];
        b[..c.len()].copy_from_slice(c.as_bytes());
        buf.extend_from_slice(&b);
    }
    while !buf.len().is_multiple_of(BLOCK) {
        buf.push(b' ');
    }
    for c in [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        "NAXIS1  =                    8",
        "NAXIS2  =                    2",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    2",
        "TTYPE1  = 'X       '",
        "TFORM1  = '1E      '",
        "TTYPE2  = 'Y       '",
        "TFORM2  = '1E      '",
        "TCTYP1  = 'RA---TAN'",
        "TCRVL1  =                150.0",
        "TCRPX1  =                 10.0",
        "TCDLT1  =                 -0.01",
        "TCTYP2  = 'DEC--TAN'",
        "TCRVL2  =                  2.5",
        "TCRPX2  =                 20.0",
        "TCDLT2  =                  0.01",
        "END",
    ] {
        let mut b = [b' '; CARD];
        b[..c.len()].copy_from_slice(c.as_bytes());
        buf.extend_from_slice(&b);
    }
    while !buf.len().is_multiple_of(BLOCK) {
        buf.push(b' ');
    }
    let data_start = buf.len();
    buf.extend(std::iter::repeat_n(0_u8, 16));
    while !(buf.len() - data_start).is_multiple_of(BLOCK) {
        buf.push(0);
    }

    let f = FitsFile::from_bytes(buf).unwrap();
    let tw = f.pixel_list_wcs(1, ' ').unwrap().unwrap();
    assert_eq!(tw.colax, vec![1, 2]);
    let sky = tw.wcs.pixel_to_world(&[9.0, 19.0]).unwrap();
    assert!((sky[0] - 150.0).abs() < 1e-9, "{sky:?}");
    assert!((sky[1] - 2.5).abs() < 1e-9, "{sky:?}");

    // No pixel-list keywords for alternate B.
    assert!(f.pixel_list_wcs(1, 'B').unwrap().is_none());
    // The primary HDU is not a binary table.
    assert!(f.pixel_list_wcs(0, ' ').is_err());
    // `wcs()` still rejects the table: Table 22's forms are separate
    // entry points, not a silent fallback.
    assert!(f.wcs(1, ' ').is_err());
}
