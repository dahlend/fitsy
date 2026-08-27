//! The `fitsy` command-line tool.
//!
//! This binary wraps the `fitsy` library. It offers six subcommands:
//!
//! * `fitsy info <file>` -- one summary line per HDU, with WCS
//!   details.
//! * `fitsy header <file> [--hdu N] [filter]` -- print the parsed
//!   header cards, filtered by `filter` when one is given.
//! * `fitsy checksum <file>` -- verify the `CHECKSUM` and `DATASUM`
//!   keywords.
//! * `fitsy stats <file> [--hdu N]` -- pixel statistics for each image
//!   HDU.
//! * `fitsy fpack <input> [-o out]` -- write a tile-compressed copy
//!   of the input.
//! * `fitsy funpack <input> [-o out]` -- write a tile-decompressed
//!   copy of the input.
//!
//! The binary takes no dependency outside the library. Argument
//! parsing is manual and stays simple for that reason.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[cfg(feature = "compression")]
use fitsy::header::{CardKind, CommentaryKind};
use fitsy::header::{HeaderEntry, Value};
use fitsy::wcs::celestial::CelestialFrame;
#[cfg(feature = "compression")]
use fitsy::{Codec, FitsWriter, TileOpts, compression::Quantize};
use fitsy::{FitsFile, Hdu, Header};
#[cfg(feature = "compression")]
use std::fs::File;

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
        "fpack" => cmd_fpack(&rest),
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
             fpack     Tile-compress a FITS file (.fz)\n    \
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
        // Header-only: nothing `info` prints needs the data.
        let header = &file.parsed_header(i)?;
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
        let (kind, shape) = describe_header(header, i);
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
    let ctypes = wcs
        .axes()
        .iter()
        .map(|a| a.ctype.as_str())
        .collect::<Vec<_>>()
        .join(" / ");
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
        let crpix = wcs.linear().crpix();
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
            .axis(lon_idx)
            .and_then(|a| a.ctype.get(5..8))
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
        let ct = wcs.ctype(sa.axis);
        lines.push(format!("       spectral axis {} = {ct}", sa.axis + 1));
    }

    lines
}

/// Summarize an HDU. Every figure printed is a header keyword, so
/// the data section is never read.
fn describe_header(h: &Header, index: usize) -> (&'static str, String) {
    let int = |key: &str| match h.first(key) {
        Some(Value::Integer(n)) => Some(*n),
        _ => None,
    };
    let axes = |naxis_key: &str, axis_key: &dyn Fn(usize) -> String| -> Vec<i64> {
        let n = match h.first(naxis_key) {
            Some(Value::Integer(n)) if *n > 0 => *n as usize,
            _ => 0,
        };
        (1..=n).filter_map(|i| int(&axis_key(i))).collect()
    };
    let join = |dims: &[i64]| {
        dims.iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(" x ")
    };
    let image_shape = || {
        let dims = axes("NAXIS", &|i| format!("NAXIS{i}"));
        if dims.is_empty() {
            "(no data)".to_string()
        } else {
            format!("{}, BITPIX={}", join(&dims), int("BITPIX").unwrap_or(0))
        }
    };

    let xtension = string_card(h, "XTENSION").unwrap_or_default();
    if index == 0 {
        // The same predicate the reader dispatches on (Sec.6). The
        // summary therefore matches the HDU kind the reader parses.
        if h.is_random_groups() {
            return (
                "RandomGrp",
                format!(
                    "{} groups, PCOUNT={}, BITPIX={}",
                    int("GCOUNT").unwrap_or(1),
                    int("PCOUNT").unwrap_or(0),
                    int("BITPIX").unwrap_or(0),
                ),
            );
        }
        return ("Image", image_shape());
    }
    match xtension.as_str() {
        "IMAGE" => ("Image", image_shape()),
        "TABLE" => (
            "AsciiTab",
            format!(
                "{} rows x {} cols",
                int("NAXIS2").unwrap_or(0),
                int("TFIELDS").unwrap_or(0)
            ),
        ),
        "BINTABLE" if matches!(h.first("ZIMAGE"), Some(Value::Logical(true))) => {
            let dims = axes("ZNAXIS", &|i| format!("ZNAXIS{i}"));
            let tiles = axes("ZNAXIS", &|i| format!("ZTILE{i}"));
            let tile = if tiles.is_empty() {
                "?".to_string()
            } else {
                join(&tiles)
            };
            (
                "CompImage",
                format!(
                    "{}, BITPIX={}, tiles {tile}",
                    join(&dims),
                    int("ZBITPIX").unwrap_or(0)
                ),
            )
        }
        "BINTABLE" => (
            "BinTable",
            format!(
                "{} rows x {} cols",
                int("NAXIS2").unwrap_or(0),
                int("TFIELDS").unwrap_or(0)
            ),
        ),
        "" => ("Other", String::new()),
        other => ("Other", format!("XTENSION={other}")),
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
        // `{:?}` is the shortest decimal that round-trips and keeps a
        // decimal point. `{:.17e}` printed 30.0 as `3.0000...e1`.
        Value::Real(x) => format!("{x:?}"),
        Value::ComplexInteger(re, im) => format!("({re}, {im})"),
        Value::ComplexReal(re, im) => format!("({re:?}, {im:?})"),
        Value::String(s) => format!("'{s}'"),
        Value::Undefined => "(undefined)".into(),
        Value::Unparsed(s) => s.clone(),
    }
}

fn string_card(h: &Header, key: &str) -> Option<String> {
    match h.first(key)? {
        Value::String(s) => Some(s.trim().to_string()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// fpack
// ---------------------------------------------------------------------------

/// Codec requested on the `fpack` command line.
#[cfg(feature = "compression")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum CodecArg {
    /// No `-c` flag: `RICE_1` where it applies, `GZIP_1` elsewhere.
    Auto,
    Rice,
    Gzip,
    Gzip2,
}

/// Resolve the codec for one image HDU.
///
/// `RICE_1` takes integer pixels of 1, 2 or 4 bytes only. Both the
/// automatic choice and an explicit `rice` request fall back to
/// `GZIP_1` on any other `BITPIX`. An explicit request also notes the
/// fallback on stderr.
#[cfg(feature = "compression")]
fn resolve_codec(arg: CodecArg, bitpix: i64, hdu_index: usize) -> Codec {
    let rice_ok = matches!(bitpix, 8 | 16 | 32);
    match arg {
        CodecArg::Auto => {
            if rice_ok {
                Codec::rice()
            } else {
                Codec::Gzip1
            }
        }
        CodecArg::Rice => {
            if rice_ok {
                Codec::rice()
            } else {
                eprintln!(
                    "HDU {hdu_index}: RICE_1 does not apply to BITPIX {bitpix}; using GZIP_1"
                );
                Codec::Gzip1
            }
        }
        CodecArg::Gzip => Codec::Gzip1,
        CodecArg::Gzip2 => Codec::Gzip2,
    }
}

#[cfg(feature = "compression")]
fn cmd_fpack(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "fitsy fpack <input> [-o <output>] [-c <codec>] [-t <tile>]\n\
             [-q <level>] [-C]\n\n\
             Tile-compress every image HDU in `input` and write the\n\
             result to `output`. If `-o` is omitted, `.fz` is appended\n\
             to the input name.\n\n\
             -c, --codec    rice | gzip | gzip2. Without the flag,\n\
                            RICE_1 is used for integer images of 1, 2\n\
                            or 4 bytes per pixel and GZIP_1 for\n\
                            everything else. Float images compress\n\
                            losslessly unless -q is given.\n\
             -t, --tile     Tile shape, comma separated, FITS axis\n\
                            order (e.g. 100,100). Default: one row\n\
                            per tile.\n\
             -q, --quantize <level>  Quantize float images before\n\
                            compression. LOSSY: the quantization step\n\
                            is the per-tile noise divided by <level>\n\
                            (4 matches the fpack default). Integer\n\
                            images are not affected.\n\
             -C, --no-checksum  Skip CHECKSUM / DATASUM.\n\n\
             A primary array moves behind an empty primary HDU,\n\
             because a compressed image is a BINTABLE extension.\n\
             `fitsy funpack` restores the original layout.\n\n\
             The output is written in full before it replaces anything\n\
             at that path, so a failure leaves an existing file as it\n\
             was.\n\n\
             Non-image HDUs and already-compressed HDUs are copied\n\
             through unchanged."
        );
        return Ok(());
    }
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut codec = CodecArg::Auto;
    let mut tile: Option<Vec<u64>> = None;
    let mut quantize: Option<Quantize> = None;
    let mut checksums = true;
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
            "-c" | "--codec" => {
                let c = it.next().ok_or("`-c` requires a codec argument")?;
                codec = parse_codec_arg(c)?;
            }
            s if s.starts_with("--codec=") => {
                codec = parse_codec_arg(&s["--codec=".len()..])?;
            }
            "-t" | "--tile" => {
                let t = it.next().ok_or("`-t` requires a tile argument")?;
                tile = Some(parse_tile_arg(t)?);
            }
            s if s.starts_with("--tile=") => {
                tile = Some(parse_tile_arg(&s["--tile=".len()..])?);
            }
            "-q" | "--quantize" => {
                let q = it.next().ok_or("`-q` requires a level argument")?;
                quantize = Some(parse_quantize_arg(q)?);
            }
            s if s.starts_with("--quantize=") => {
                quantize = Some(parse_quantize_arg(&s["--quantize=".len()..])?);
            }
            "-C" | "--no-checksum" => checksums = false,
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
    let input = input.ok_or("`fpack` requires an input path")?;
    let output = output.unwrap_or_else(|| default_fpack_output(&input));
    if output == input {
        return Err("refusing to write output on top of input; pass -o explicitly".into());
    }
    let file = open_fits(&input)?;
    let compressed = write_via_temp(&output, |tmp| {
        fpack_into(&file, tmp, codec, tile.as_deref(), quantize, checksums)
    })?;
    eprintln!(
        "wrote {} (compressed {compressed} HDU{})",
        output.display(),
        if compressed == 1 { "" } else { "s" }
    );
    Ok(())
}

/// Run `build` against a hidden sibling of `output`, then rename that
/// file into place.
///
/// The `build` argument receives the temporary path and writes the
/// whole file. A failure part-way through leaves any existing
/// `output` untouched. It also removes the temporary file. Both
/// `fpack` and `funpack` write this way, because each can fail on a
/// later HDU after it has written earlier ones.
#[cfg(feature = "compression")]
fn write_via_temp<T>(
    output: &Path,
    build: impl FnOnce(&Path) -> Result<T, Box<dyn std::error::Error>>,
) -> Result<T, Box<dyn std::error::Error>> {
    let name = output
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("fitsy.out");
    let tmp = output.with_file_name(format!(".{name}.tmp{}", std::process::id()));
    match build(&tmp) {
        Ok(v) => {
            std::fs::rename(&tmp, output).inspect_err(|_| {
                let _ = std::fs::remove_file(&tmp);
            })?;
            Ok(v)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Compress every image HDU of `file` into a new FITS file at `dest`,
/// and report how many HDUs were compressed.
#[cfg(feature = "compression")]
fn fpack_into(
    file: &FitsFile,
    dest: &Path,
    codec: CodecArg,
    tile: Option<&[u64]>,
    quantize: Option<Quantize>,
    checksums: bool,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut sink = File::create(dest)?;
    let mut writer = FitsWriter::new(&mut sink);
    if checksums {
        writer = writer.with_checksums();
    }
    let mut compressed = 0_usize;
    for i in 0..file.len() {
        let hdu = file.hdu(i)?;
        let is_plain_image = matches!(hdu, Hdu::Image(_));
        if !is_plain_image {
            writer.write_hdu(hdu.header(), hdu.data_bytes())?;
            continue;
        }
        let header = hdu.header();
        let axes = header.axes()?;
        if axes.is_empty() || axes.iter().product::<u64>() == 0 {
            // Nothing to compress; keep the HDU as it is.
            writer.write_hdu(header, hdu.data_bytes())?;
            continue;
        }
        if i == 0 {
            // A compressed image is a BINTABLE extension, so the
            // primary array moves to extension 1 behind this stub.
            // The compressed header records the move as ZSIMPLE = T.
            let mut stub = Header::empty();
            stub.push("SIMPLE", Value::Logical(true), Some("conforming FITS file"))?;
            stub.push("BITPIX", Value::Integer(8), None)?;
            stub.push("NAXIS", Value::Integer(0), None)?;
            stub.push(
                "EXTEND",
                Value::Logical(true),
                Some("FITS dataset may contain extensions"),
            )?;
            writer.write_hdu(&stub, &[])?;
        }
        let bitpix = header.bitpix()?;
        let hdu_quantize = if bitpix < 0 { quantize } else { None };
        // Quantized tiles hold i32 samples, so RICE_1 applies.
        let codec_bitpix = if hdu_quantize.is_some() { 32 } else { bitpix };
        let mut opts = TileOpts::new().codec(resolve_codec(codec, codec_bitpix, i));
        opts.tile = tile.map(<[u64]>::to_vec);
        opts.quantize = hdu_quantize;
        writer.write_hdu_compressed(header, hdu.data_bytes(), &opts)?;
        compressed += 1;
    }
    writer.finish()?;
    Ok(compressed)
}

#[cfg(not(feature = "compression"))]
fn cmd_fpack(_args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    Err("`fpack` requires the `compression` feature (enabled by default)".into())
}

#[cfg(feature = "compression")]
fn parse_codec_arg(s: &str) -> Result<CodecArg, Box<dyn std::error::Error>> {
    match s {
        "rice" => Ok(CodecArg::Rice),
        "gzip" => Ok(CodecArg::Gzip),
        "gzip2" => Ok(CodecArg::Gzip2),
        other => Err(format!("unknown codec `{other}` (expected rice, gzip or gzip2)").into()),
    }
}

#[cfg(feature = "compression")]
fn parse_quantize_arg(s: &str) -> Result<Quantize, Box<dyn std::error::Error>> {
    let level: f64 = s
        .parse()
        .map_err(|_| format!("quantize level `{s}` is not a number"))?;
    if !level.is_finite() || level <= 0.0 {
        return Err(format!("quantize level `{s}` must be a positive number").into());
    }
    Ok(Quantize::level(level))
}

#[cfg(feature = "compression")]
fn parse_tile_arg(s: &str) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let dims: Vec<u64> = s
        .split(',')
        .map(|d| d.trim().parse::<u64>())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("tile `{s}` is not a comma-separated list of integers"))?;
    if dims.is_empty() || dims.contains(&0) {
        return Err(format!("tile `{s}` must hold dimensions of 1 or more").into());
    }
    Ok(dims)
}

#[cfg(feature = "compression")]
fn default_fpack_output(input: &Path) -> PathBuf {
    let mut out = input.to_path_buf();
    let new_name = match input.file_name().and_then(|s| s.to_str()) {
        Some(name) => format!("{name}.fz"),
        None => "packed.fits.fz".to_string(),
    };
    out.set_file_name(new_name);
    out
}

// ---------------------------------------------------------------------------
// funpack
// ---------------------------------------------------------------------------

#[cfg(feature = "compression")]
fn cmd_funpack(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "fitsy funpack <input> [-o <output>] [-C]\n\n\
             Decompress every tile-compressed image HDU in `input`\n\
             and write the result to `output`. If `-o` is omitted,\n\
             `.fz` is stripped from the input name (or `.funpacked\n\
             .fits` is appended).\n\n\
             Non-compressed HDUs are copied through unchanged.\n\n\
             CHECKSUM and DATASUM are recomputed per HDU, as cfitsio's\n\
             funpack does; the `.fz` sums are not carried over because\n\
             tile compression may be lossy. Pass -C to skip.\n\n\
             The output is written in full before it replaces anything\n\
             at that path, so a failure leaves an existing file as it\n\
             was."
        );
        return Ok(());
    }
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut checksums = true;
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
            "-C" | "--no-checksum" => checksums = false,
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
    let decompressed = write_via_temp(&output, |tmp| funpack_into(&file, tmp, checksums))?;
    eprintln!(
        "wrote {} (decompressed {decompressed} HDU{})",
        output.display(),
        if decompressed == 1 { "" } else { "s" }
    );
    Ok(())
}

/// Decompress every tile-compressed image HDU of `file` into a new
/// FITS file at `dest`, and report how many HDUs were decompressed.
#[cfg(feature = "compression")]
fn funpack_into(
    file: &FitsFile,
    dest: &Path,
    checksums: bool,
) -> Result<usize, Box<dyn std::error::Error>> {
    // fpack moves a primary array behind an empty stub primary,
    // because a compressed image is a BINTABLE extension. When the
    // first extension restores to a primary array (ZSIMPLE = T), drop
    // the stub so the output matches the original layout.
    let skip_stub = file.len() >= 2
        && {
            let first = file.hdu(0)?;
            matches!(first, Hdu::Image(_))
                && first.data_bytes().is_empty()
                && is_bare_stub(first.header())
        }
        && matches!(file.hdu(1)?, Hdu::CompressedImage(ref c) if c.was_primary());
    let mut sink = File::create(dest)?;
    let mut writer = FitsWriter::new(&mut sink);
    if checksums {
        writer = writer.with_checksums();
    }
    let mut decompressed = 0_usize;
    for i in 0..file.len() {
        if i == 0 && skip_stub {
            continue;
        }
        let hdu = file.hdu(i)?;
        match hdu {
            Hdu::CompressedImage(c) => {
                let img = c.as_image()?;
                // `as_image` yields an IMAGE extension header, which
                // is what this HDU is. Only an image that fpack moved
                // out of the primary slot, and that is landing back in
                // that slot, becomes a primary header again.
                if c.was_primary() && writer.hdu_count() == 0 {
                    let promoted = promote_to_primary(img.header())?;
                    writer.write_hdu(&promoted, img.raw_bytes())?;
                } else {
                    writer.write_hdu(img.header(), img.raw_bytes())?;
                }
                decompressed += 1;
            }
            other => {
                writer.write_hdu(other.header(), other.data_bytes())?;
            }
        }
    }
    writer.finish()?;
    Ok(decompressed)
}

#[cfg(not(feature = "compression"))]
fn cmd_funpack(_args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    Err("`funpack` requires the `compression` feature (enabled by default)".into())
}

/// Whether `h` is the empty primary header that `fpack` inserts, and
/// nothing more.
///
/// `fpack` writes `SIMPLE`, `BITPIX`, `NAXIS` and `EXTEND`, plus
/// `CHECKSUM` and `DATASUM` unless the caller passed `-C`. A stub
/// that carries any other card holds metadata a caller added, so
/// `funpack` keeps the HDU rather than dropping it.
#[cfg(feature = "compression")]
fn is_bare_stub(h: &Header) -> bool {
    h.entries().iter().all(|e| {
        matches!(
            e.keyword.as_str(),
            "SIMPLE" | "BITPIX" | "NAXIS" | "EXTEND" | "CHECKSUM" | "DATASUM"
        )
    })
}

/// Rebuild an IMAGE extension header as a primary header, for a
/// decompressed array that `fpack` moved out of the primary slot.
#[cfg(feature = "compression")]
fn promote_to_primary(h: &Header) -> Result<Header, Box<dyn std::error::Error>> {
    let mut out = Header::empty();
    out.push("SIMPLE", Value::Logical(true), Some("conforming FITS file"))?;
    out.push("BITPIX", Value::Integer(h.bitpix()?), None)?;
    let axes = h.axes()?;
    out.push("NAXIS", Value::Integer(axes.len() as i64), None)?;
    for (i, n) in axes.iter().enumerate() {
        out.push(format!("NAXIS{}", i + 1), Value::Integer(*n as i64), None)?;
    }
    // Standard Sec.4.4.1.1: EXTEND follows the last NAXISn card.
    out.push(
        "EXTEND",
        Value::Logical(true),
        Some("FITS dataset may contain extensions"),
    )?;
    for entry in h.entries() {
        let kw = entry.keyword.as_str();
        if matches!(entry.kind, CardKind::Commentary) {
            if let Some(text) = entry.commentary.as_deref() {
                let kind = match kw {
                    "COMMENT" => CommentaryKind::Comment,
                    "HISTORY" => CommentaryKind::History,
                    _ => CommentaryKind::Blank,
                };
                out.push_commentary(kind, text);
            }
            continue;
        }
        let structural = matches!(
            kw,
            "XTENSION" | "SIMPLE" | "EXTEND" | "BITPIX" | "NAXIS" | "PCOUNT" | "GCOUNT"
        ) || (kw.starts_with("NAXIS")
            && kw[5..].chars().all(|c| c.is_ascii_digit()));
        if structural {
            continue;
        }
        if let Some(v) = entry.value.clone() {
            out.push(kw.to_string(), v, entry.comment.as_deref())?;
        }
    }
    Ok(out)
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
        // `verify_checksums` streams without caching; loading the HDU
        // for EXTNAME would pull the whole file back into memory.
        let extname = string_card(&file.parsed_header(r.hdu)?, "EXTNAME").unwrap_or_default();
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

/// Decode the raw bytes of an [`OwnedImage`] into `f64` pixels in
/// physical units, with `BZERO` and `BSCALE` applied.
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
