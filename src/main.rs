//! `fitsy` command-line tool.
//!
//! A small Swiss-army knife around the `fitsy` library:
//!
//! * `fitsy info <file>`                      -- one-line summary per HDU (with WCS details).
//! * `fitsy header <file> [--hdu N] [filter]` -- dump parsed header cards, optionally filtered.
//! * `fitsy checksum <file>`                  -- verify CHECKSUM / DATASUM keywords.
//! * `fitsy stats <file> [--hdu N]`           -- pixel statistics for image HDUs.
//! * `fitsy funpack <input> [-o out]`         -- write a tile-decompressed copy
//!   (the inverse of `fpack`).
//!
//! Designed to require no external dependencies: argument parsing is
//! manual and intentionally simple.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use fitsy::header::{HeaderEntry, Value};
use fitsy::wcs::celestial::CelestialFrame;
use fitsy::{FitsFile, FitsWriter, Hdu, Header};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        print_top_usage();
        return ExitCode::from(2);
    };
    let rest: Vec<String> = args.collect();
    let result = match cmd.as_str() {
        "info" => cmd_info(&rest),
        "header" => cmd_header(&rest),
        "checksum" => cmd_checksum(&rest),
        "stats" => cmd_stats(&rest),
        "funpack" => cmd_funpack(&rest),
        "-h" | "--help" | "help" => {
            print_top_usage();
            return ExitCode::SUCCESS;
        }
        "-V" | "--version" => {
            println!("fitsy {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        other => {
            eprintln!("fitsy: unknown subcommand `{other}`\n");
            print_top_usage();
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fitsy: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_top_usage() {
    eprintln!(
        "fitsy {}\n\
         A FITS command-line utility.\n\n\
         USAGE:\n    \
             fitsy <SUBCOMMAND> [ARGS...]\n\n\
         SUBCOMMANDS:\n    \
             info      Summarize the HDUs of a FITS file\n    \
             header    Print parsed header cards\n    \
             checksum  Verify CHECKSUM / DATASUM keywords\n    \
             stats     Pixel statistics for image HDUs\n    \
             funpack   Decompress a tile-compressed (.fz) file\n    \
             help      Show this message\n\n\
         Run `fitsy <SUBCOMMAND> --help` for subcommand details.",
        env!("CARGO_PKG_VERSION")
    );
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

fn cmd_info(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "fitsy info <file>\n\n\
             List every HDU with its kind, dimensions, and a brief\n\
             data summary (BITPIX for images, row/column counts for\n\
             tables, tile shape for compressed images).\n\
             WCS information (projection, CRVAL, CRPIX, pixel scale,\n\
             distortion) is shown for HDUs that carry a WCS."
        );
        return Ok(());
    }
    let path = single_path(args, "info")?;
    let file = open_fits(&path)?;
    println!("File: {}", path.display());
    println!("HDUs: {}", file.len());
    println!("{:>3}  {:<10}  {:<24}  SHAPE", "#", "KIND", "EXTNAME/VER");
    for i in 0..file.len() {
        let hdu = file.hdu(i)?;
        let header = hdu.header();
        let extname = string_card(header, "EXTNAME").unwrap_or_default();
        let extver = header
            .first("EXTVER")
            .and_then(|v| match v {
                Value::Integer(n) => Some(*n),
                _ => None,
            })
            .map(|n| format!(" v{n}"))
            .unwrap_or_default();
        let label = if extname.is_empty() {
            String::new()
        } else {
            format!("{extname}{extver}")
        };
        let (kind, shape) = describe_hdu(&hdu);
        println!("{i:>3}  {kind:<10}  {label:<24}  {shape}");

        // WCS info -- try primary (alt=' ') then alternates A..Z.
        let alts: Vec<char> = std::iter::once(' ').chain('A'..='Z').collect();
        for alt in alts {
            if let Ok(Some(wcs)) = file.wcs(i, alt) {
                let suffix = if alt == ' ' {
                    String::new()
                } else {
                    format!(" [{alt}]")
                };
                let wcs_line = format_wcs_summary(&wcs, &suffix);
                for line in wcs_line {
                    println!("       {line}");
                }
            }
        }
    }
    Ok(())
}

fn format_wcs_summary(wcs: &fitsy::Wcs, suffix: &str) -> Vec<String> {
    let mut lines = Vec::new();

    // Header line: "WCS[A]: CTYPE1 / CTYPE2 / ..."
    let ctypes = wcs.ctype.join(" / ");
    let name = wcs
        .wcsname
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| format!(" \"{s}\""))
        .unwrap_or_default();
    lines.push(format!("WCS{suffix}:{name}  CTYPE = {ctypes}"));

    if let Some(cb) = &wcs.celestial {
        let ra = cb.rotation.alpha0;
        let dec = cb.rotation.delta0;
        let frame = match cb.pair.frame {
            CelestialFrame::Equatorial => "Equatorial",
            CelestialFrame::Galactic => "Galactic",
            CelestialFrame::Ecliptic => "Ecliptic",
            CelestialFrame::Supergalactic => "Supergalactic",
            CelestialFrame::HelioEcliptic => "HelioEcliptic",
            CelestialFrame::Other => "Other",
        };
        // CRPIX
        let crpix = wcs.linear.crpix();
        let lon_idx = cb.pair.lon;
        let lat_idx = cb.pair.lat;
        let crpix1 = crpix.get(lon_idx).copied().unwrap_or(0.0);
        let crpix2 = crpix.get(lat_idx).copied().unwrap_or(0.0);

        lines.push(format!(
            "       frame={frame}  CRVAL=({ra:.6}, {dec:+.6})  CRPIX=({crpix1:.1}, {crpix2:.1})"
        ));

        // Pixel scale at CRPIX (pixel_scale_at returns arcseconds).
        // CRPIX is 1-based per the FITS standard; pixel_scale_at takes
        // 0-based pixels.
        if let Ok((sx, sy)) = wcs.pixel_scale_at(crpix1 - 1.0, crpix2 - 1.0) {
            lines.push(format!(
                "       pixel scale ~ {sx:.4}\"/px (lon) x {sy:.4}\"/px (lat)"
            ));
        }

        // Projection name comes from the CTYPE string (chars 5..8).
        let proj = wcs
            .ctype
            .get(lon_idx)
            .and_then(|ct| ct.get(5..8))
            .unwrap_or("?");
        let mut extras = Vec::new();
        if cb.sip.is_some() {
            extras.push("SIP");
        }
        if cb.tpv.is_some() {
            extras.push("TPV");
        }
        if cb.tnx.is_some() {
            extras.push("TNX/ZPX");
        }
        let distortion = if extras.is_empty() {
            String::new()
        } else {
            format!("  distortion={}", extras.join("+"))
        };
        lines.push(format!("       projection={proj}{distortion}"));
    }

    for sa in &wcs.spectral {
        let ct = wcs.ctype.get(sa.axis).map_or("?", String::as_str);
        lines.push(format!("       spectral axis {} = {ct}", sa.axis + 1));
    }

    lines
}

fn describe_hdu(hdu: &Hdu<'_>) -> (&'static str, String) {
    match hdu {
        Hdu::Image(img) => {
            let axes = img.axes();
            let shape = if axes.is_empty() {
                "(no data)".into()
            } else {
                let dims = axes
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(" x ");
                format!("{dims}, BITPIX={}", img.bitpix().as_i64())
            };
            ("Image", shape)
        }
        Hdu::RandomGroups(rg) => (
            "RandomGrp",
            format!(
                "{} groups, PCOUNT={}, BITPIX={}",
                rg.n_groups(),
                rg.pcount(),
                rg.bitpix().as_i64(),
            ),
        ),
        Hdu::AsciiTable(t) => (
            "AsciiTab",
            format!("{} rows x {} cols", t.n_rows(), t.columns().len()),
        ),
        Hdu::BinTable(t) => (
            "BinTable",
            format!("{} rows x {} cols", t.n_rows(), t.columns().len()),
        ),
        #[cfg(feature = "compression")]
        Hdu::CompressedImage(c) => {
            let axes = c.axes();
            let dims = axes
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(" x ");
            let tile = c
                .tile_shape()
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(" x ");
            (
                "CompImage",
                format!("{dims}, BITPIX={}, tiles {tile}", c.bitpix().as_i64()),
            )
        }
        Hdu::Conforming(h) => ("Other", format!("XTENSION={}", h.xtension())),
        #[allow(
            unreachable_patterns,
            reason = "Hdu is #[non_exhaustive]; needed for forward compatibility"
        )]
        _ => ("Other", String::new()),
    }
}

// ---------------------------------------------------------------------------
// header
// ---------------------------------------------------------------------------

fn cmd_header(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "fitsy header <file> [--hdu N] [filter]\n\n\
             Dump parsed header cards. Without --hdu, every HDU's\n\
             header is printed in turn, separated by a banner line.\n\
             An optional filter string restricts output to cards\n\
             whose keyword contains the string (case-insensitive)."
        );
        return Ok(());
    }
    let mut path: Option<PathBuf> = None;
    let mut hdu_idx: Option<usize> = None;
    let mut filter: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hdu" => {
                let n = it.next().ok_or("`--hdu` requires an integer argument")?;
                hdu_idx = Some(n.parse().map_err(|_| {
                    format!("invalid --hdu value `{n}` (expected non-negative integer)")
                })?);
            }
            s if s.starts_with("--hdu=") => {
                let n = &s["--hdu=".len()..];
                hdu_idx = Some(
                    n.parse()
                        .map_err(|_| format!("invalid --hdu value `{n}`"))?,
                );
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag `{other}`").into());
            }
            other => {
                if path.is_none() {
                    path = Some(PathBuf::from(other));
                } else if filter.is_none() {
                    filter = Some(other.to_ascii_lowercase());
                } else {
                    return Err(format!("unexpected extra argument `{other}`").into());
                }
            }
        }
    }
    let path = path.ok_or("`header` requires a path argument")?;
    let file = open_fits(&path)?;
    let range: Box<dyn Iterator<Item = usize>> = match hdu_idx {
        Some(i) => Box::new(std::iter::once(i)),
        None => Box::new(0..file.len()),
    };
    for i in range {
        let hdu = file.hdu(i)?;
        if hdu_idx.is_none() {
            println!("==== HDU {i} ====");
        }
        print_header(hdu.header(), filter.as_deref());
    }
    Ok(())
}

fn print_header(h: &Header, filter: Option<&str>) {
    for entry in h.entries() {
        if let Some(f) = filter
            && !entry.keyword.to_ascii_lowercase().contains(f)
        {
            continue;
        }
        println!("{}", format_entry(entry));
    }
}

fn format_entry(e: &HeaderEntry) -> String {
    if let Some(text) = e.commentary.as_deref() {
        // COMMENT, HISTORY, blank-keyword commentary cards.
        let kw = if e.keyword.is_empty() {
            "       "
        } else {
            &e.keyword
        };
        return format!("{kw:<8} {text}");
    }
    let value = match &e.value {
        None => String::from("(no value)"),
        Some(v) => display_value(v),
    };
    let comment = e
        .comment
        .as_deref()
        .filter(|c| !c.is_empty())
        .map(|c| format!(" / {c}"))
        .unwrap_or_default();
    format!("{:<8}= {}{}", e.keyword, value, comment)
}

fn display_value(v: &Value) -> String {
    match v {
        Value::Logical(b) => {
            if *b {
                "T".into()
            } else {
                "F".into()
            }
        }
        Value::Integer(n) => n.to_string(),
        Value::Real(x) => format!("{x:.17e}"),
        Value::ComplexInteger(re, im) => format!("({re}, {im})"),
        Value::ComplexReal(re, im) => format!("({re:.17e}, {im:.17e})"),
        Value::String(s) => format!("'{s}'"),
        Value::Undefined => "(undefined)".into(),
    }
}

fn string_card(h: &Header, key: &str) -> Option<String> {
    match h.first(key)? {
        Value::String(s) => Some(s.trim().to_string()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// funpack
// ---------------------------------------------------------------------------

#[cfg(feature = "compression")]
fn cmd_funpack(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "fitsy funpack <input> [-o <output>]\n\n\
             Decompress every tile-compressed image HDU in `input`\n\
             and write the result to `output`. If `-o` is omitted,\n\
             `.fz` is stripped from the input name (or `.funpacked\n\
             .fits` is appended).\n\n\
             Non-compressed HDUs are copied through unchanged."
        );
        return Ok(());
    }
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" | "--output" => {
                let p = it.next().ok_or("`-o` requires a path argument")?;
                output = Some(PathBuf::from(p));
            }
            s if s.starts_with("--output=") => {
                output = Some(PathBuf::from(&s["--output=".len()..]));
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag `{other}`").into());
            }
            other => {
                if input.is_some() {
                    return Err(format!("unexpected extra argument `{other}`").into());
                }
                input = Some(PathBuf::from(other));
            }
        }
    }
    let input = input.ok_or("`funpack` requires an input path")?;
    let output = output.unwrap_or_else(|| default_funpack_output(&input));
    if output == input {
        return Err("refusing to write output on top of input; pass -o explicitly".into());
    }
    let file = open_fits(&input)?;
    let mut sink = File::create(&output)?;
    let mut writer = FitsWriter::new(&mut sink);
    let mut decompressed = 0_usize;
    for i in 0..file.len() {
        let hdu = file.hdu(i)?;
        match hdu {
            Hdu::CompressedImage(c) => {
                let img = c.as_image()?;
                writer.write_hdu(img.header(), img.raw_bytes())?;
                decompressed += 1;
            }
            other => {
                writer.write_hdu(other.header(), other.data_bytes())?;
            }
        }
    }
    eprintln!(
        "wrote {} (decompressed {decompressed} HDU{})",
        output.display(),
        if decompressed == 1 { "" } else { "s" }
    );
    Ok(())
}

#[cfg(not(feature = "compression"))]
fn cmd_funpack(_args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    Err("`funpack` requires the `compression` feature (enabled by default)".into())
}

#[cfg(feature = "compression")]
fn default_funpack_output(input: &Path) -> PathBuf {
    if let Some(stem) = input.file_name().and_then(|s| s.to_str())
        && let Some(stripped) = stem.strip_suffix(".fz")
    {
        return input.with_file_name(stripped);
    }
    let mut out = input.to_path_buf();
    let new_name = match input.file_name().and_then(|s| s.to_str()) {
        Some(name) => format!("{name}.funpacked.fits"),
        None => "funpacked.fits".to_string(),
    };
    out.set_file_name(new_name);
    out
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

fn single_path(args: &[String], name: &'static str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut path: Option<PathBuf> = None;
    for a in args {
        if a.starts_with('-') {
            return Err(format!("unknown flag `{a}` for `{name}`").into());
        }
        if path.is_some() {
            return Err(format!("`{name}` takes a single path argument").into());
        }
        path = Some(PathBuf::from(a));
    }
    path.ok_or_else(|| format!("`{name}` requires a path argument").into())
}

fn open_fits(path: &Path) -> Result<FitsFile, Box<dyn std::error::Error>> {
    Ok(FitsFile::open(path)?)
}

// ---------------------------------------------------------------------------
// checksum
// ---------------------------------------------------------------------------

fn cmd_checksum(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "fitsy checksum <file>\n\n\
             Verify the CHECKSUM and DATASUM keywords in every HDU.\n\
             HDUs that lack both keywords are reported as skipped.\n\
             Exits with status 1 if any present checksum fails."
        );
        return Ok(());
    }
    let path = single_path(args, "checksum")?;
    let file = open_fits(&path)?;
    let reports = file.verify_checksums()?;

    let mut any_fail = false;
    println!("{:>3}  {:<9}  {:<9}  EXTNAME", "#", "CHECKSUM", "DATASUM");
    for r in &reports {
        let hdu = file.hdu(r.hdu)?;
        let extname = string_card(hdu.header(), "EXTNAME").unwrap_or_default();
        let fmt = |v: Option<bool>| match v {
            None => "absent   ",
            Some(true) => "OK       ",
            Some(false) => "FAIL     ",
        };
        println!(
            "{:>3}  {}  {}  {}",
            r.hdu,
            fmt(r.checksum_ok),
            fmt(r.datasum_ok),
            extname,
        );
        if r.checksum_ok == Some(false) || r.datasum_ok == Some(false) {
            any_fail = true;
        }
    }

    let n_checked = reports
        .iter()
        .filter(|r| r.checksum_ok.is_some() || r.datasum_ok.is_some())
        .count();
    if n_checked == 0 {
        eprintln!("note: no CHECKSUM/DATASUM keywords found");
    }
    if any_fail {
        return Err("one or more checksums failed".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------

fn cmd_stats(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "fitsy stats <file> [--hdu N]\n\n\
             Print pixel statistics (N, min, max, mean, std) for\n\
             every image HDU. NaN/BLANK pixels are excluded.\n\
             Without --hdu, every image/compressed-image HDU is shown."
        );
        return Ok(());
    }

    let mut path: Option<PathBuf> = None;
    let mut hdu_idx: Option<usize> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hdu" => {
                let n = it.next().ok_or("`--hdu` requires an integer argument")?;
                hdu_idx = Some(
                    n.parse()
                        .map_err(|_| format!("invalid --hdu value `{n}`"))?,
                );
            }
            s if s.starts_with("--hdu=") => {
                let n = &s["--hdu=".len()..];
                hdu_idx = Some(
                    n.parse()
                        .map_err(|_| format!("invalid --hdu value `{n}`"))?,
                );
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag `{other}`").into());
            }
            other => {
                if path.is_some() {
                    return Err(format!("unexpected extra argument `{other}`").into());
                }
                path = Some(PathBuf::from(other));
            }
        }
    }
    let path = path.ok_or("`stats` requires a path argument")?;
    let file = open_fits(&path)?;

    let range: Box<dyn Iterator<Item = usize>> = match hdu_idx {
        Some(i) => Box::new(std::iter::once(i)),
        None => Box::new(0..file.len()),
    };

    println!(
        "{:>3}  {:>12}  {:>14}  {:>14}  {:>14}  {:>14}  EXTNAME",
        "#", "N_VALID", "MIN", "MAX", "MEAN", "STD"
    );

    for i in range {
        let hdu = file.hdu(i)?;
        let extname = string_card(hdu.header(), "EXTNAME").unwrap_or_default();
        let pixels: Option<Vec<f64>> = match hdu {
            Hdu::Image(ref img) if !img.axes().is_empty() => Some(img.read_physical()?.into_vec()),
            #[cfg(feature = "compression")]
            Hdu::CompressedImage(ref c) => {
                let owned = c.as_image()?;
                Some(decode_owned_physical(&owned)?)
            }
            _ => None,
        };

        let Some(pixels) = pixels else {
            // Skip non-image or empty HDUs silently.
            continue;
        };

        let stats = pixel_stats(&pixels);
        println!(
            "{i:>3}  {:>12}  {:>14}  {:>14}  {:>14}  {:>14}  {extname}",
            stats.n,
            compact(stats.min),
            compact(stats.max),
            compact(stats.mean),
            compact(stats.std),
        );
    }
    Ok(())
}

/// Format a float compactly: use scientific notation for very large/small
/// values, fixed otherwise. Width is always &lt;= 14 chars.
fn compact(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 {
            "+Inf".to_string()
        } else {
            "-Inf".to_string()
        };
    }
    let abs = v.abs();
    if abs == 0.0 || (1e-4..1e10_f64).contains(&abs) {
        // Fixed, up to 6 significant digits.
        let s = format!("{v:.6}");
        // Trim trailing zeros after the decimal point.
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    } else {
        format!("{v:.6e}")
    }
}

struct Stats {
    n: usize,
    min: f64,
    max: f64,
    mean: f64,
    std: f64,
}

fn pixel_stats(pixels: &[f64]) -> Stats {
    let valid: Vec<f64> = pixels.iter().copied().filter(|x| x.is_finite()).collect();
    if valid.is_empty() {
        return Stats {
            n: 0,
            min: f64::NAN,
            max: f64::NAN,
            mean: f64::NAN,
            std: f64::NAN,
        };
    }
    let n = valid.len();
    let mut min = valid[0];
    let mut max = valid[0];
    let mut sum = 0.0_f64;
    for &v in &valid {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v;
    }
    let mean = sum / n as f64;
    // Two-pass variance for numerical stability.
    let var = valid.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    Stats {
        n,
        min,
        max,
        mean,
        std: var.sqrt(),
    }
}

/// Decode an [`OwnedImage`]'s raw bytes to physical (BZERO/BSCALE applied) `f64` pixels.
#[cfg(feature = "compression")]
#[allow(
    clippy::unnecessary_wraps,
    reason = "matches the error-returning style of the surrounding decode helpers"
)]
fn decode_owned_physical(
    img: &fitsy::compression::OwnedImage,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    use fitsy::data::Scaling;
    use fitsy::data::encoding::Bitpix;
    let h = img.header();
    let scaling = Scaling {
        bzero: h.bzero(),
        bscale: h.bscale(),
        blank: h.blank(),
    };
    let bytes = img.raw_bytes();
    let bp = img.bitpix();
    let bsize = bp.byte_size();
    let mut out = Vec::with_capacity(bytes.len() / bsize.max(1));
    for chunk in bytes.chunks_exact(bsize) {
        let v: f64 = match bp {
            Bitpix::U8 => scaling.apply_int(i64::from(chunk[0])),
            Bitpix::I16 => scaling.apply_int(i64::from(i16::from_be_bytes([chunk[0], chunk[1]]))),
            Bitpix::I32 => scaling.apply_int(i64::from(i32::from_be_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
            ]))),
            Bitpix::I64 => scaling.apply_int(i64::from_be_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])),
            Bitpix::F32 => scaling.apply_real(f64::from(f32::from_bits(u32::from_be_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
            ])))),
            Bitpix::F64 => scaling.apply_real(f64::from_bits(u64::from_be_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]))),
        };
        out.push(v);
    }
    Ok(out)
}
