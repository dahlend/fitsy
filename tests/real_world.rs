//! Smoke-test: walk every HDU of every FITS file in `data/` and
//! report any failures. This exercises the *real-world* feature
//! coverage rather than synthesised fixtures.
#![cfg(feature = "compression")]

use std::path::PathBuf;

use fitsy::{FitsFile, Hdu};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn list_files() -> Vec<PathBuf> {
    let mut out: Vec<_> = std::fs::read_dir(data_dir())
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            // Restrict to FITS files; the directory may also hold
            // companion fixtures (e.g. CSV ground truth for WCS tests).
            matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("fits" | "fit" | "fts" | "fz" | "gz")
            )
        })
        .collect();
    out.sort();
    out
}

#[test]
fn open_all_real_files() {
    let mut errors = Vec::new();
    for path in list_files() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let result = (|| -> Result<usize, String> {
            let f = FitsFile::open(&path).map_err(|e| format!("open: {e}"))?;
            let n = f.len();
            for i in 0..n {
                let hdu = f.hdu(i).map_err(|e| format!("hdu({i}): {e}"))?;
                // Touch header + data classification to exercise readers.
                let _ = hdu.header();
                match &hdu {
                    Hdu::Image(img) => {
                        let _ = img.bitpix();
                        let _ = img.axes();
                    }
                    Hdu::BinTable(b) => {
                        let _ = b.n_rows();
                        let _ = b.columns();
                    }
                    Hdu::AsciiTable(a) => {
                        let _ = a.n_rows();
                        let _ = a.columns();
                    }
                    Hdu::CompressedImage(c) => {
                        // Try to decompress the first tile only.
                        let _ = c.as_image().map_err(|e| format!("decompress: {e}"))?;
                    }
                    Hdu::RandomGroups(rg) => {
                        let _ = rg.n_groups();
                    }
                    _ => {}
                }
            }
            Ok(n)
        })();
        match result {
            Ok(n) => println!("{name}: {n} HDU(s) ok"),
            Err(e) => errors.push(format!("{name}: {e}")),
        }
    }
    assert!(
        errors.is_empty(),
        "real-world failures:\n{}",
        errors.join("\n")
    );
}

/// Spot-check: read full + sub-array of a real image HDU and verify
/// they agree on the overlapping window. Exercises BSCALE/BZERO and
/// `read_subarray` against real-world data rather than fixtures.
#[test]
fn subarray_matches_full_read_on_real_image() {
    let path = data_dir().join("SPITZER_I1_34767104_0019_0000_2_bcd.fits");
    let f = FitsFile::open(&path).unwrap();
    let Hdu::Image(img) = f.hdu(0).unwrap() else {
        panic!("expected primary IMAGE");
    };
    let axes = img.axes().to_vec();
    assert_eq!(axes.len(), 2, "expected a 2D image");
    let nx = axes[0] as usize;
    let ny = axes[1] as usize;
    // Read full image as f32 (Spitzer BCDs are BITPIX=-32).
    let full = img.read_raw::<f32>().unwrap();
    let full_data = full.as_slice();
    // Sub-array: 4x3 starting at (1, 2).
    let sub = img.read_subarray::<f32>(&[1, 2], &[4, 3]).unwrap();
    let sub_data = sub.as_slice();
    for j in 0..3 {
        for i in 0..4 {
            let got = sub_data[j * 4 + i];
            let want = full_data[(2 + j) * nx + (1 + i)];
            assert!(
                (got.is_nan() && want.is_nan()) || got == want,
                "({i},{j}) mismatch: sub={got} full={want}"
            );
        }
    }
    // Width/height sanity.
    assert!(
        nx > 4 && ny > 3,
        "image too small for this test ({nx}x{ny})"
    );
}

/// Verify `Header::date_obs` / `mjd_obs` / timesys + COMMENT iteration on
/// a real-world LCOGT observation that carries all three keywords
/// per FITS Standard Sec.9.
#[test]
fn date_obs_and_commentary_on_real_image() {
    let path = data_dir().join("coj0m416-sq36-20240423-0201-e91.fits");
    let f = FitsFile::open(&path).unwrap();
    let h = f.hdu(0).unwrap();
    let hdr = h.header();

    let dt = hdr.date_obs().expect("DATE-OBS should parse");
    assert_eq!((dt.year, dt.month, dt.day), (2024, 4, 23));
    assert_eq!((dt.hour, dt.minute, dt.second), (14, 42, 4));

    // Header carries an explicit MJD-OBS; cross-check with our
    // calendar conversion. Many observatories round MJD-OBS to a
    // reduced-precision representation (here LCOGT writes only 7
    // decimal places, ~9 ms granularity) so we tolerate up to 1 s.
    let mjd_keyword = hdr.mjd_obs().unwrap();
    let mjd_calc = dt.mjd();
    assert!(
        (mjd_keyword - mjd_calc).abs() < 1.0 / 86_400.0,
        "DATE-OBS-derived MJD {mjd_calc} disagrees with header MJD-OBS {mjd_keyword}"
    );
    assert_eq!(hdr.timesys(), "UTC");
}

/// Verify `Header::comments()` / `history()` against a Spitzer BCD,
/// which carries dozens of HISTORY cards and a few COMMENTs in its
/// primary HDU.
#[test]
fn commentary_iteration_on_real_image() {
    let path = data_dir().join("SPITZER_I1_34767104_0019_0000_2_bcd.fits");
    let f = FitsFile::open(&path).unwrap();
    let h = f.hdu(0).unwrap();
    let hdr = h.header();
    let comments: Vec<&str> = hdr.comments().collect();
    let history: Vec<&str> = hdr.history().collect();
    assert!(
        comments.len() >= 3,
        "expected >=3 COMMENTs, got {}",
        comments.len()
    );
    assert!(
        history.len() >= 30,
        "expected >=30 HISTORYs, got {}",
        history.len()
    );
}

/// Verify CHECKSUM/DATASUM against a real LCOGT file with multi-MB
/// HDUs. This regression-tests the 64-bit accumulator fix: with a
/// naive `u32` accumulator the inner loop wraps for any HDU larger
/// than ~256 KiB and the verifier silently returns `false`. astropy
/// confirms every HDU in this file has valid CHECKSUM and DATASUM.
#[test]
fn checksums_validate_on_real_multi_hdu_file() {
    let path = data_dir().join("coj0m416-sq36-20240423-0201-e91.fits");
    let f = FitsFile::open(&path).unwrap();
    let reports = f.verify_checksums().unwrap();
    assert_eq!(reports.len(), 4, "file has 4 HDUs");
    for r in &reports {
        assert_eq!(
            r.datasum_ok,
            Some(true),
            "HDU {}: DATASUM verification failed",
            r.hdu
        );
        assert_eq!(
            r.checksum_ok,
            Some(true),
            "HDU {}: CHECKSUM verification failed",
            r.hdu
        );
    }
}
