//! End-to-end checks of the `fpack` and `funpack` subcommands.
#![cfg(feature = "compression")]

use std::path::PathBuf;
use std::process::Command;

use fitsy::{FitsFile, FitsWriter, Hdu, ImageBuilder, write};

fn tempfile_path(label: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("fitsy_cli_{label}_{}", std::process::id()));
    p
}

/// The hidden sibling path `fpack` and `funpack` build into before
/// renaming, mirroring `write_via_temp` in `src/main.rs`.
fn temp_sibling_of(output: &std::path::Path) -> PathBuf {
    let name = output.file_name().unwrap().to_str().unwrap();
    // The binary runs as its own process, so its pid is not this
    // process's pid. Match on the stable prefix instead.
    let prefix = format!(".{name}.tmp");
    let dir = output.parent().unwrap();
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&prefix))
        })
        .unwrap_or_else(|| dir.join(format!("{prefix}-none")))
}

fn run(args: &[&str]) {
    let status = Command::new(env!("CARGO_BIN_EXE_fitsy"))
        .args(args)
        .status()
        .expect("failed to launch the fitsy binary");
    assert!(status.success(), "fitsy {args:?} exited with {status}");
}

fn run_failing(args: &[&str]) {
    let status = Command::new(env!("CARGO_BIN_EXE_fitsy"))
        .args(args)
        .status()
        .expect("failed to launch the fitsy binary");
    assert!(!status.success(), "fitsy {args:?} was expected to fail");
}

/// A failure part-way through `fpack` must leave the destination
/// untouched, rather than replacing it with a truncated file.
#[test]
fn fpack_failure_does_not_clobber_the_output() {
    let src = tempfile_path("mixed.fits");
    let dest = tempfile_path("existing.fz");

    // A 2-D primary and a 3-D extension: one `-t 5,5` cannot fit both,
    // so the second HDU fails after the first is already written.
    let two_d = ImageBuilder::<i16>::new(vec![8_u64, 8], (0..64_i16).collect::<Vec<_>>())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let three_d = ImageBuilder::<i16>::new(vec![5_u64, 4, 3], (0..60_i16).collect::<Vec<_>>())
        .unwrap()
        .primary(false)
        .build()
        .unwrap();
    write(&src, &[two_d, three_d], true).unwrap();

    std::fs::write(&dest, b"pre-existing contents").unwrap();
    run_failing(&[
        "fpack",
        src.to_str().unwrap(),
        "-t",
        "5,5",
        "-o",
        dest.to_str().unwrap(),
    ]);
    assert_eq!(
        std::fs::read(&dest).unwrap(),
        b"pre-existing contents",
        "a failed fpack overwrote the destination"
    );

    // The temporary file this command would have used is gone. Test
    // only that exact path: the temp directory is shared, and another
    // test running in parallel has temp files of its own in flight.
    assert!(
        !temp_sibling_of(&dest).exists(),
        "the temporary file was left behind"
    );

    for p in [&src, &dest] {
        let _ = std::fs::remove_file(p);
    }
}

/// `funpack` drops the empty primary that `fpack` inserts, but only
/// when that HDU carries nothing of its own. A stub holding metadata
/// is the caller's data, not bookkeeping.
#[test]
fn funpack_keeps_a_stub_that_carries_metadata() {
    let src = tempfile_path("stubsrc.fits");
    let packed = tempfile_path("stub.fz");
    let restored = tempfile_path("stubback.fits");

    let pixels: Vec<i16> = (0..64).collect();
    let hdu = ImageBuilder::<i16>::new(vec![8_u64, 8], pixels)
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    write(&src, std::slice::from_ref(&hdu), true).unwrap();
    run(&[
        "fpack",
        src.to_str().unwrap(),
        "-o",
        packed.to_str().unwrap(),
    ]);

    // Without the added card the stub is dropped and one HDU returns.
    run(&[
        "funpack",
        packed.to_str().unwrap(),
        "-o",
        restored.to_str().unwrap(),
    ]);
    assert_eq!(FitsFile::open(&restored).unwrap().len(), 1);

    // Add a card to the stub, and it must survive as its own HDU.
    let f = FitsFile::open(&packed).unwrap();
    let mut stub = f.hdu(0).unwrap().header().clone();
    stub.push("OBJECT", fitsy::header::Value::String("kept".into()), None)
        .unwrap();
    let mut buf = Vec::new();
    let mut w = FitsWriter::new(&mut buf);
    w.write_hdu_parts(&stub, &[]).unwrap();
    for i in 1..f.len() {
        let hdu = f.hdu(i).unwrap();
        w.write_hdu(&hdu).unwrap();
    }
    w.finish().unwrap();
    std::fs::write(&packed, &buf).unwrap();

    run(&[
        "funpack",
        packed.to_str().unwrap(),
        "-o",
        restored.to_str().unwrap(),
    ]);
    let out = FitsFile::open(&restored).unwrap();
    assert_eq!(out.len(), 2, "the stub carried metadata and must be kept");
    assert_eq!(
        out.parsed_header(0)
            .unwrap()
            .optional_string("OBJECT")
            .as_deref(),
        Some("kept")
    );

    for p in [&src, &packed, &restored] {
        let _ = std::fs::remove_file(p);
    }
}

/// The same guarantee for `funpack`: a compressed HDU that cannot be
/// decoded must not cost the user whatever was at the output path.
#[test]
fn funpack_failure_does_not_clobber_the_output() {
    let src = tempfile_path("twoext.fits");
    let packed = tempfile_path("twoext.fz");
    let broken = tempfile_path("broken.fz");
    let dest = tempfile_path("existing.fits");

    let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
        .unwrap()
        .primary(true)
        .build()
        .unwrap();
    let ext = |scale: i16| {
        ImageBuilder::<i16>::new(
            vec![8_u64, 8],
            (0..64_i16).map(|v| v * scale).collect::<Vec<_>>(),
        )
        .unwrap()
        .primary(false)
        .build()
        .unwrap()
    };
    write(&src, &[primary, ext(1), ext(3)], true).unwrap();
    run(&[
        "fpack",
        src.to_str().unwrap(),
        "-c",
        "rice",
        "-o",
        packed.to_str().unwrap(),
    ]);

    // Break only the second compressed HDU, by renaming its codec to
    // one no reader supports. The replacement is the same length, so
    // every byte offset in the file is unchanged and the first
    // compressed HDU still decodes.
    let bytes = std::fs::read(&packed).unwrap();
    let card = b"ZCMPTYPE= 'RICE_1  '";
    let at = bytes
        .windows(card.len())
        .rposition(|w| w == card)
        .expect("no ZCMPTYPE card found");
    let mut broken_bytes = bytes.clone();
    broken_bytes[at..at + card.len()].copy_from_slice(b"ZCMPTYPE= 'BOGUS1  '");
    std::fs::write(&broken, &broken_bytes).unwrap();

    std::fs::write(&dest, b"pre-existing contents").unwrap();
    run_failing(&[
        "funpack",
        broken.to_str().unwrap(),
        "-o",
        dest.to_str().unwrap(),
    ]);
    assert_eq!(
        std::fs::read(&dest).unwrap(),
        b"pre-existing contents",
        "a failed funpack overwrote the destination"
    );

    for p in [&src, &packed, &broken, &dest] {
        let _ = std::fs::remove_file(p);
    }
}

/// `fpack` then `funpack` restores the original pixels, header cards
/// and HDU layout, including a data-bearing primary array.
#[test]
fn fpack_funpack_round_trip() {
    let src = tempfile_path("src.fits");
    let packed = tempfile_path("packed.fits.fz");
    let restored = tempfile_path("restored.fits");

    let pixels: Vec<i16> = (0..48 * 32).map(|i| (i % 3000) as i16 - 1500).collect();
    let hdu = ImageBuilder::<i16>::new(vec![48_u64, 32], pixels)
        .unwrap()
        .primary(true)
        .card("OBJECT", "cli test", None)
        .card("EXPTIME", 12.5, None)
        .build()
        .unwrap();
    write(&src, std::slice::from_ref(&hdu), true).unwrap();

    run(&[
        "fpack",
        src.to_str().unwrap(),
        "-o",
        packed.to_str().unwrap(),
    ]);
    run(&[
        "funpack",
        packed.to_str().unwrap(),
        "-o",
        restored.to_str().unwrap(),
    ]);

    // The stub primary that fpack inserts is gone again.
    let f = FitsFile::open(&restored).unwrap();
    assert_eq!(f.len(), 1, "funpack must restore the original layout");
    let Hdu::Image(img) = f.hdu(0).unwrap() else {
        panic!("HDU 0 is not an image");
    };
    assert_eq!(img.raw_bytes(), hdu.raw_bytes());
    let h = img.header();
    assert!(matches!(
        h.first("SIMPLE"),
        Some(fitsy::header::Value::Logical(true))
    ));
    assert_eq!(h.optional_string("OBJECT").as_deref(), Some("cli test"));
    assert_eq!(h.optional_real("EXPTIME"), Some(12.5));

    for p in [&src, &packed, &restored] {
        let _ = std::fs::remove_file(p);
    }
}
