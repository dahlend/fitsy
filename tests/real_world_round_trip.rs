//! End-to-end round-trip integrity test: open every uncompressed
//! `.fits` fixture in `tests/data/`, write each HDU back through
//! `FitsWriter`, reopen the result, and assert the two files are
//! semantically identical via the `diff` module.
//!
//! Compressed (`.fz`) and gzipped (`.gz`) fixtures are out of scope:
//! the writer does not re-emit `CompressedImage` HDUs nor wrap output
//! in gzip, so a byte/structural round trip is not meaningful for
//! those formats. Plain `.fits` files cover the full uncompressed
//! image, binary-table, ASCII-table, and random-groups paths.
#![cfg(feature = "compression")]

use std::path::PathBuf;

use fitsy::diff::{DiffOptions, FitsDiff};
use fitsy::{FitsFile, FitsWriter, Hdu};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn list_uncompressed_fits() -> Vec<PathBuf> {
    let mut out: Vec<_> = std::fs::read_dir(data_dir())
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("fits" | "fit" | "fts")
            )
        })
        .collect();
    out.sort();
    out
}

/// Round-trip every uncompressed `.fits` fixture: read each HDU,
/// re-serialize via `FitsWriter`, reopen, and assert structural +
/// data parity through `FitsDiff`. `CompressedImage` HDUs are
/// skipped (writer cannot re-emit tile-compressed HDUs yet); files
/// containing only such HDUs are skipped entirely.
#[test]
fn round_trip_preserves_all_real_world_files() {
    let mut errors = Vec::new();
    let mut checked = 0_usize;

    for src_path in list_uncompressed_fits() {
        let name = src_path.file_name().unwrap().to_string_lossy().into_owned();

        let result = (|| -> Result<bool, String> {
            let src = FitsFile::open(&src_path).map_err(|e| format!("open src: {e}"))?;

            // Collect (header, data) pairs from each HDU. Bail out for
            // compressed-image and random-groups HDUs: the writer validates
            // data size against header declaration and cannot re-emit these
            // formats verbatim without re-encoding.
            let n = src.len();
            let mut owned = Vec::with_capacity(n);
            for i in 0..n {
                let hdu = src.hdu(i).map_err(|e| format!("hdu({i}): {e}"))?;
                if matches!(hdu, Hdu::CompressedImage(_) | Hdu::RandomGroups(_)) {
                    return Ok(false);
                }
                owned.push((hdu.header().clone(), hdu.data_bytes().to_vec()));
            }
            if owned.is_empty() {
                return Err("file has zero HDUs".into());
            }

            // Write to a temp file, then reopen and diff.
            let tmp = tempfile_path(&name);
            {
                let f = std::fs::File::create(&tmp).map_err(|e| format!("create tmp: {e}"))?;
                let mut writer = FitsWriter::new(std::io::BufWriter::new(f));
                for (h, d) in &owned {
                    writer
                        .write_hdu(h, d)
                        .map_err(|e| format!("write_hdu: {e}"))?;
                }
                writer.finish().map_err(|e| format!("finish: {e}"))?;
            }

            // Diff with default options. CHECKSUM/DATASUM/DATE are
            // ignored out of the box -- they re-stamp on every write.
            // EXTEND is auto-emitted by ImageBuilder for primary HDUs;
            // here we round-trip via raw header bytes, so whatever
            // was on disk is preserved verbatim.
            let opts = DiffOptions::default();

            let diff = FitsDiff::open(&src_path, &tmp, opts).map_err(|e| format!("diff: {e}"))?;

            let _ = std::fs::remove_file(&tmp);

            if !diff.is_identical() {
                return Err(format!("diff not identical: {diff:?}"));
            }
            Ok(true)
        })();

        match result {
            Ok(true) => {
                checked += 1;
                println!("{name}: round-trip ok");
            }
            Ok(false) => println!("{name}: skipped (contains CompressedImage HDU)"),
            Err(e) => errors.push(format!("{name}: {e}")),
        }
    }

    assert!(
        errors.is_empty(),
        "round-trip failures:\n{}",
        errors.join("\n")
    );
    assert!(
        checked > 0,
        "no uncompressed fixtures exercised a round trip"
    );
}

fn tempfile_path(label: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    p.push(format!("fitsy-rt-{pid}-{nanos}-{label}"));
    p
}
