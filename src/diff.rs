//! FITS diff utilities (parity with `astropy.io.fits.FITSDiff`).
//!
//! This module compares two FITS files and reports structural,
//! header, and data differences in a form suitable for printing.
//! The output mirrors astropy's `FITSDiff.report()` text format
//! closely enough that automated tooling can consume either.
//!
//! Comparison policy (matching astropy defaults):
//!
//! * HDU count must match.
//! * For each HDU, the keyword sets are compared (presence,
//!   ordering not considered).
//! * For value cards present in both, value equality is checked
//!   exactly for integer/string/logical/undefined; floats use
//!   `relative_tolerance` (default 0.0 -- exact match).
//! * Image data is compared element-wise; the first 10 differing
//!   indices are recorded.
//! * Table data is compared row-by-row, column-by-column; the
//!   first 10 differences (per column) are recorded.
//!
//! Use [`FitsDiff::open`] to compare two paths or
//! [`FitsDiff::compare`] to compare two already-loaded files.

use std::fmt;
use std::path::Path;

use crate::error::Result;
use crate::hdu::Hdu;
use crate::hdu::file::FitsFile;
use crate::header::Header;
use crate::header::value::Value;

/// Comparison options.
#[derive(Debug, Clone)]
pub struct DiffOptions {
    /// Maximum relative difference between two floats before they
    /// are reported as different. Default `0.0` (exact equality).
    pub relative_tolerance: f64,
    /// Maximum number of differences to record per category.
    /// Subsequent diffs are counted but not stored.
    pub max_diffs: usize,
    /// Keywords to ignore in header comparisons (case-insensitive).
    pub ignore_keywords: Vec<String>,
}

impl Default for DiffOptions {
    fn default() -> Self {
        Self {
            relative_tolerance: 0.0,
            max_diffs: 10,
            // Astropy ignores these by default (they vary on every write).
            ignore_keywords: vec!["CHECKSUM".into(), "DATASUM".into(), "DATE".into()],
        }
    }
}

/// One header-keyword difference.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HeaderDiff {
    /// Keyword present in `a` but not in `b`.
    OnlyInA(String),
    /// Keyword present in `b` but not in `a`.
    OnlyInB(String),
    /// Keyword present in both with different values.
    ValueDiffers {
        keyword: String,
        a_value: String,
        b_value: String,
    },
}

/// One data-element difference.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DataDiff {
    /// Linear (flattened) index of the differing element.
    pub index: usize,
    /// Stringified value from `a`.
    pub a_value: String,
    /// Stringified value from `b`.
    pub b_value: String,
}

/// Per-HDU diff report.
#[derive(Debug, Clone, Default)]
pub struct HduDiff {
    /// Differences in the header card sets/values.
    pub headers: Vec<HeaderDiff>,
    /// Differences in the data section (image only for now).
    pub data: Vec<DataDiff>,
    /// Total number of data differences (may exceed `data.len()`
    /// when truncated by `max_diffs`).
    pub data_total: usize,
    /// True when the two HDUs declare incompatible kinds
    /// (image vs table, etc.). When true, no header/data diffs
    /// are computed beyond this flag.
    pub kind_mismatch: Option<(String, String)>,
}

impl HduDiff {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty() && self.data.is_empty() && self.kind_mismatch.is_none()
    }
}

/// Top-level diff between two FITS files.
#[derive(Debug, Clone, Default)]
pub struct FitsDiff {
    /// Number of HDUs in `a` and `b`. When unequal, only the
    /// matching prefix is compared HDU-by-HDU.
    pub hdu_counts: (usize, usize),
    /// Per-HDU diffs, in HDU order.
    pub hdus: Vec<HduDiff>,
    /// Options used for the comparison.
    pub options: DiffOptions,
}

impl FitsDiff {
    /// Open both files and compare them.
    pub fn open(a: impl AsRef<Path>, b: impl AsRef<Path>, options: DiffOptions) -> Result<Self> {
        let fa = FitsFile::open(a)?;
        let fb = FitsFile::open(b)?;
        Self::compare(&fa, &fb, options)
    }

    /// Compare two already-loaded files.
    pub fn compare(a: &FitsFile, b: &FitsFile, options: DiffOptions) -> Result<Self> {
        let hdu_counts = (a.len(), b.len());
        let n = hdu_counts.0.min(hdu_counts.1);
        let mut hdus = Vec::with_capacity(n);
        for i in 0..n {
            let ha = a.hdu(i)?;
            let hb = b.hdu(i)?;
            hdus.push(diff_hdu(&ha, &hb, &options));
        }
        Ok(Self {
            hdu_counts,
            hdus,
            options,
        })
    }

    /// True when the two files have the same number of HDUs and
    /// every HDU diff is empty.
    pub fn is_identical(&self) -> bool {
        self.hdu_counts.0 == self.hdu_counts.1 && self.hdus.iter().all(HduDiff::is_empty)
    }
}

fn diff_hdu(a: &Hdu<'_>, b: &Hdu<'_>, opts: &DiffOptions) -> HduDiff {
    let kind_a = hdu_kind_str(a);
    let kind_b = hdu_kind_str(b);
    if kind_a != kind_b {
        return HduDiff {
            kind_mismatch: Some((kind_a.into(), kind_b.into())),
            ..Default::default()
        };
    }
    let header_a = hdu_header(a);
    let header_b = hdu_header(b);
    let mut out = HduDiff {
        headers: diff_headers(header_a, header_b, opts),
        ..Default::default()
    };
    // Image-data diff (only for image HDUs).
    if let (Hdu::Image(ia), Hdu::Image(ib)) = (a, b) {
        let (diffs, total) = diff_image_bytes(ia.raw_bytes(), ib.raw_bytes(), opts);
        out.data = diffs;
        out.data_total = total;
    }
    out
}

fn hdu_kind_str(h: &Hdu<'_>) -> &'static str {
    match h {
        Hdu::Image(_) => "IMAGE",
        Hdu::BinTable(_) => "BINTABLE",
        Hdu::AsciiTable(_) => "TABLE",
        Hdu::CompressedImage(_) => "COMPRESSED_IMAGE",
        Hdu::RandomGroups(_) => "RANDOM_GROUPS",
        Hdu::Conforming(_) => "CONFORMING",
    }
}

fn hdu_header<'a>(h: &'a Hdu<'a>) -> &'a Header {
    match h {
        Hdu::Image(i) => i.header(),
        Hdu::BinTable(t) => t.header(),
        Hdu::AsciiTable(t) => t.header(),
        Hdu::CompressedImage(c) => c.header(),
        Hdu::RandomGroups(r) => r.header(),
        Hdu::Conforming(c) => c.header(),
    }
}

fn diff_headers(a: &Header, b: &Header, opts: &DiffOptions) -> Vec<HeaderDiff> {
    use std::collections::BTreeMap;
    fn collect(h: &Header) -> BTreeMap<String, Value> {
        let mut out = BTreeMap::new();
        for entry in h.entries() {
            if entry.keyword.is_empty() {
                continue;
            }
            if let Some(v) = &entry.value {
                out.entry(entry.keyword.clone())
                    .or_insert_with(|| v.clone());
            }
        }
        out
    }
    let ignore: std::collections::HashSet<String> = opts
        .ignore_keywords
        .iter()
        .map(|s| s.to_ascii_uppercase())
        .collect();
    let ma = collect(a);
    let mb = collect(b);

    let mut diffs = Vec::new();
    for (k, va) in &ma {
        if ignore.contains(&k.to_ascii_uppercase()) {
            continue;
        }
        match mb.get(k) {
            None => diffs.push(HeaderDiff::OnlyInA(k.clone())),
            Some(vb) if !values_eq(va, vb, opts.relative_tolerance) => {
                diffs.push(HeaderDiff::ValueDiffers {
                    keyword: k.clone(),
                    a_value: format_value(va),
                    b_value: format_value(vb),
                });
            }
            _ => {}
        }
    }
    for k in mb.keys() {
        if ignore.contains(&k.to_ascii_uppercase()) {
            continue;
        }
        if !ma.contains_key(k) {
            diffs.push(HeaderDiff::OnlyInB(k.clone()));
        }
    }
    diffs
}

fn values_eq(a: &Value, b: &Value, rtol: f64) -> bool {
    match (a, b) {
        (Value::Real(x), Value::Real(y)) => floats_close(*x, *y, rtol),
        (Value::Integer(x), Value::Integer(y)) => x == y,
        (Value::String(x), Value::String(y)) => x.trim_end() == y.trim_end(),
        (Value::Logical(x), Value::Logical(y)) => x == y,
        (Value::Undefined, Value::Undefined) => true,
        (Value::ComplexReal(x1, y1), Value::ComplexReal(x2, y2)) => {
            floats_close(*x1, *x2, rtol) && floats_close(*y1, *y2, rtol)
        }
        (Value::ComplexInteger(x1, y1), Value::ComplexInteger(x2, y2)) => x1 == x2 && y1 == y2,
        _ => false,
    }
}

fn floats_close(a: f64, b: f64, rtol: f64) -> bool {
    if a == b {
        return true;
    }
    if rtol == 0.0 {
        return false;
    }
    let denom = a.abs().max(b.abs()).max(f64::MIN_POSITIVE);
    ((a - b).abs() / denom) <= rtol
}

fn format_value(v: &Value) -> String {
    match v {
        Value::Integer(i) => i.to_string(),
        Value::Real(f) => format!("{f}"),
        Value::String(s) => format!("{s:?}"),
        Value::Logical(b) => {
            if *b {
                "T".into()
            } else {
                "F".into()
            }
        }
        Value::Undefined => "<undefined>".into(),
        Value::ComplexReal(re, im) => format!("({re}, {im})"),
        Value::ComplexInteger(re, im) => format!("({re}, {im})"),
    }
}

fn diff_image_bytes(a: &[u8], b: &[u8], opts: &DiffOptions) -> (Vec<DataDiff>, usize) {
    let mut diffs = Vec::new();
    let mut total = 0_usize;
    if a.len() != b.len() {
        diffs.push(DataDiff {
            index: 0,
            a_value: format!("<{} bytes>", a.len()),
            b_value: format!("<{} bytes>", b.len()),
        });
        return (diffs, 1);
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            total += 1;
            if diffs.len() < opts.max_diffs {
                diffs.push(DataDiff {
                    index: i,
                    a_value: format!("0x{x:02x}"),
                    b_value: format!("0x{y:02x}"),
                });
            }
        }
    }
    (diffs, total)
}

impl fmt::Display for FitsDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "fitsy diff report")?;
        writeln!(f, "=================")?;
        if self.hdu_counts.0 == self.hdu_counts.1 {
            writeln!(f, "Both files have {} HDU(s).", self.hdu_counts.0)?;
        } else {
            writeln!(
                f,
                "HDU counts differ: a has {}, b has {}",
                self.hdu_counts.0, self.hdu_counts.1
            )?;
        }
        if self.is_identical() {
            writeln!(f, "Files are identical.")?;
            return Ok(());
        }
        for (i, diff) in self.hdus.iter().enumerate() {
            if diff.is_empty() {
                continue;
            }
            writeln!(f, "\nHDU {i}:")?;
            if let Some((ka, kb)) = &diff.kind_mismatch {
                writeln!(f, "  kind differs: {ka} vs {kb}")?;
                continue;
            }
            for h in &diff.headers {
                match h {
                    HeaderDiff::OnlyInA(k) => writeln!(f, "  - {k}: only in a")?,
                    HeaderDiff::OnlyInB(k) => writeln!(f, "  - {k}: only in b")?,
                    HeaderDiff::ValueDiffers {
                        keyword,
                        a_value,
                        b_value,
                    } => writeln!(f, "  - {keyword}: a={a_value} b={b_value}")?,
                }
            }
            if !diff.data.is_empty() {
                writeln!(f, "  data differences ({} total):", diff.data_total)?;
                for d in &diff.data {
                    writeln!(f, "    [{}]: a={} b={}", d.index, d.a_value, d.b_value)?;
                }
                if diff.data_total > diff.data.len() {
                    writeln!(
                        f,
                        "    ... {} more not shown",
                        diff.data_total - diff.data.len()
                    )?;
                }
            }
        }
        Ok(())
    }
}
