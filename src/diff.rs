//! FITS file comparison.
//!
//! # Purpose
//!
//! [`FitsDiff`] compares two FITS files and reports their structural,
//! header and data differences. Call [`FitsDiff::open`] to compare two
//! paths, or [`FitsDiff::compare`] to compare two loaded files.
//!
//! # Comparison policy
//!
//! * The HDU counts must match.
//! * Keyword sets compare by presence, not by order.
//! * A shared value card compares exactly, except for a float, which
//!   uses `relative_tolerance` and `absolute_tolerance`. Both default
//!   to `0.0`.
//! * Image pixels compare numerically, in physical units, under the
//!   same tolerances. That means `BZERO` and `BSCALE` applied, and
//!   `BLANK` mapped to `NaN`. The report names the pixel number and
//!   the decoded values. A tile-compressed image decompresses first,
//!   so two tiles that differ byte-wise but decode alike compare
//!   equal.
//! * Table data compares cell by cell, in decoded values.
//! * A random-groups HDU, and an extension that this crate does not
//!   recognize, have no decoded form. Each gets one byte-level
//!   verdict.
//!
//! # Design constraints
//!
//! Byte-identical data with matching decoding cards short-circuits.
//! That is the common case. Otherwise both sides decode to `f64`, so
//! comparing two `BITPIX = 16` images holds four times the on-disk
//! size per side.

use std::fmt;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use crate::error::Result;
use crate::hdu::Hdu;
use crate::hdu::file::FitsFile;
use crate::header::Header;
use crate::header::value::Value;

/// The options that control a comparison.
///
/// The two tolerance fields set how closely two floats must agree. The
/// ignore lists exclude a keyword or an HDU from the comparison. The
/// default compares every float exactly and ignores nothing.
#[derive(Debug, Clone)]
pub struct DiffOptions {
    /// Maximum relative difference between two float values -- header
    /// card, image pixel, or table cell -- before they are reported as
    /// different. Default `0.0` (exact equality).
    pub relative_tolerance: f64,
    /// Maximum absolute difference between two float values before
    /// they are reported as different. Default `0.0`.
    ///
    /// Combined with [`Self::relative_tolerance`] as
    /// `|a - b| <= absolute_tolerance + relative_tolerance * |b|`
    /// A relative tolerance
    /// alone can never reconcile values straddling zero, which is
    /// what this exists for.
    pub absolute_tolerance: f64,
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
            absolute_tolerance: 0.0,
            max_diffs: 10,
            // Astropy ignores these by default (they vary on every write).
            ignore_keywords: vec!["CHECKSUM".into(), "DATASUM".into(), "DATE".into()],
        }
    }
}

/// One difference between two headers.
///
/// The variant records why the two disagree: a keyword present on one
/// side alone, or a keyword present on both with different values.
#[derive(Debug, Clone)]
pub enum HeaderDiff {
    /// Keyword present in `a` but not in `b`.
    OnlyInA(String),
    /// Keyword present in `b` but not in `a`.
    OnlyInB(String),
    /// Keyword present in both with different values.
    ValueDiffers {
        /// The keyword that differs.
        keyword: String,
        /// Its value in `a`, rendered for display.
        a_value: String,
        /// Its value in `b`, rendered for display.
        b_value: String,
    },
}

/// One difference between two data sections.
///
/// This records the position of the element that differs and the value
/// each side holds there. An image reports a pixel number, and a table
/// reports a row and a column.
#[derive(Debug, Clone)]
pub struct DataDiff {
    /// Column name for a table HDU; `None` for an image.
    pub column: Option<String>,
    /// Flattened element index: pixel number for an image (row-major
    /// over `NAXISn`, first axis fastest), row number for a table.
    pub index: usize,
    /// Stringified value from `a`, in physical units.
    pub a_value: String,
    /// Stringified value from `b`, in physical units.
    pub b_value: String,
}

/// The comparison result for one pair of HDUs.
///
/// This holds the header differences and the data differences of that
/// pair. Call [`HduDiff::is_empty`] to test whether the two matched in
/// every respect compared.
#[derive(Debug, Clone, Default)]
pub struct HduDiff {
    /// Differences in the header card sets/values.
    pub headers: Vec<HeaderDiff>,
    /// Differences in the data section: pixels for images, cells for
    /// tables.
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
    /// True when the two inputs matched in every respect compared.
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty() && self.data.is_empty() && self.kind_mismatch.is_none()
    }
}

/// The comparison result for two whole files.
///
/// This holds the HDU count of each side, one [`HduDiff`] per compared
/// pair, and the [`DiffOptions`] that produced the result. Build one
/// with [`FitsDiff::open`] or [`FitsDiff::compare`].
///
/// # Examples
///
/// ```
/// use fitsy::diff::{DiffOptions, FitsDiff};
/// use fitsy::{FitsFile, FitsWriter, ImageBuilder};
///
/// // Two files that differ in one pixel.
/// let mut bytes = Vec::new();
/// for last in [4.0_f32, 9.0] {
///     let px = vec![1.0, 2.0, 3.0, last];
///     let (h, d) = ImageBuilder::new(vec![2_u64, 2], px)?
///         .primary(true)
///         .build()?;
///     let mut buf: Vec<u8> = Vec::new();
///     FitsWriter::new(&mut buf).write_hdu(&h, &d)?;
///     bytes.push(buf);
/// }
///
/// let a = FitsFile::from_bytes(bytes.remove(0))?;
/// let b = FitsFile::from_bytes(bytes.remove(0))?;
/// let report = FitsDiff::compare(&a, &b, DiffOptions::default())?;
///
/// assert_eq!(report.hdus.len(), 1);
/// assert!(!report.hdus[0].is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
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
    ///
    /// The `a` and `b` arguments name the two paths to compare. The
    /// `options` argument sets the float tolerances and the ignore
    /// lists. Pass `DiffOptions::default()` for an exact comparison.
    ///
    /// # Errors
    ///
    /// The conditions of [`FitsFile::open`](crate::FitsFile::open) for
    /// either path, and the conditions of [`FitsDiff::compare`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(a: impl AsRef<Path>, b: impl AsRef<Path>, options: DiffOptions) -> Result<Self> {
        let fa = FitsFile::open(a)?;
        let fb = FitsFile::open(b)?;
        Self::compare(&fa, &fb, options)
    }

    /// Compare two loaded files.
    ///
    /// The `a` and `b` arguments are the two files to compare. The
    /// `options` argument sets the float tolerances and the ignore
    /// lists, as it does for [`FitsDiff::open`].
    ///
    /// # Errors
    ///
    /// The conditions of [`FitsFile::hdu`](crate::FitsFile::hdu) for
    /// either file, because this reads every HDU. A tile-compressed
    /// HDU adds [`crate::FitsError::Data`] when a tile fails to
    /// decompress.
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
    let (diffs, total) = match (a, b) {
        (Hdu::Image(ia), Hdu::Image(ib)) => diff_image_data(ia, ib, opts),
        (Hdu::BinTable(ta), Hdu::BinTable(tb)) => diff_bintable_data(ta, tb, opts),
        (Hdu::AsciiTable(ta), Hdu::AsciiTable(tb)) => diff_ascii_table_data(ta, tb, opts),
        // Tile-compressed images are compared on their decompressed
        // pixels: two files can hold byte-different tiles that decode
        // to identical (or within-tolerance) images.
        #[cfg(feature = "compression")]
        (Hdu::CompressedImage(ca), Hdu::CompressedImage(cb)) => {
            match (ca.as_image(), cb.as_image()) {
                (Ok(oa), Ok(ob)) => {
                    match (
                        crate::hdu::image::ImageHdu::new(oa.header().clone(), oa.raw_bytes()),
                        crate::hdu::image::ImageHdu::new(ob.header().clone(), ob.raw_bytes()),
                    ) {
                        (Ok(ia), Ok(ib)) => diff_image_data(&ia, &ib, opts),
                        _ => diff_bintable_data(ca.as_bintable(), cb.as_bintable(), opts),
                    }
                }
                // Undecompressable tiles: compare the compressed
                // BINTABLE itself rather than return no verdict.
                _ => diff_bintable_data(ca.as_bintable(), cb.as_bintable(), opts),
            }
        }
        // Random groups have no decoded comparison yet, so fall back
        // to the raw data section: coarse, but it never reports two
        // differing HDUs as identical.
        (Hdu::RandomGroups(ra), Hdu::RandomGroups(rb)) => {
            diff_raw_bytes(ra.raw_bytes(), rb.raw_bytes(), opts)
        }
        // An unrecognized XTENSION has no decoded form *by
        // definition*, so it gets the same byte fallback. Leaving it
        // out is not a smaller gap than random groups, it is a
        // larger one: any extension type this crate does not know
        // lands here, and returning "no differences" would report
        // two wholly different files as identical.
        (Hdu::Conforming(ca), Hdu::Conforming(cb)) => {
            diff_raw_bytes(ca.data_bytes(), cb.data_bytes(), opts)
        }
        // Same-kind pairs are exhaustive above; a mixed pair was
        // already reported as `kind_mismatch` and returned early.
        _ => (Vec::new(), 0),
    };
    out.data = diffs;
    out.data_total = total;
    out
}

/// Last-resort byte comparison for HDU kinds with no decoded form.
/// Reports at most one difference: byte offsets are not useful to a
/// caller, so the point is only to avoid claiming equality.
fn diff_raw_bytes(a: &[u8], b: &[u8], opts: &DiffOptions) -> (Vec<DataDiff>, usize) {
    let mut sink = DiffSink::new(opts);
    if a != b {
        sink.push(
            None,
            0,
            format_args!("<{} bytes, differing>", a.len()),
            format_args!("<{} bytes, differing>", b.len()),
        );
    }
    sink.finish()
}

/// Collects differences while honoring `max_diffs`, so each
/// comparator can stay a plain loop.
struct DiffSink<'a> {
    diffs: Vec<DataDiff>,
    total: usize,
    opts: &'a DiffOptions,
}

impl<'a> DiffSink<'a> {
    fn new(opts: &'a DiffOptions) -> Self {
        Self {
            diffs: Vec::new(),
            total: 0,
            opts,
        }
    }

    fn push(
        &mut self,
        column: Option<&str>,
        index: usize,
        a: impl fmt::Display,
        b: impl fmt::Display,
    ) {
        self.total += 1;
        if self.diffs.len() < self.opts.max_diffs {
            self.diffs.push(DataDiff {
                column: column.map(str::to_string),
                index,
                a_value: a.to_string(),
                b_value: b.to_string(),
            });
        }
    }

    /// Hand back the recorded differences and the running total. The
    /// two are separate because the total keeps counting past
    /// `max_diffs`: callers are told how many differences exist, not
    /// just how many fit in the report.
    fn finish(self) -> (Vec<DataDiff>, usize) {
        (self.diffs, self.total)
    }
}

/// Compare image pixels in physical units (`BZERO` and `BSCALE`
/// applied, `BLANK` mapped to NaN), honoring the float tolerances.
///
/// Byte-wise comparison was the previous behavior and is wrong twice
/// over: it ignores `rtol`/`atol` entirely, and it reports byte
/// offsets rather than pixel numbers. Two files whose pixels agree to
/// within tolerance but differ in the last mantissa bit compared as
/// different; a single differing `f64` pixel produced eight entries.
fn diff_image_data(
    a: &crate::hdu::image::ImageHdu<'_>,
    b: &crate::hdu::image::ImageHdu<'_>,
    opts: &DiffOptions,
) -> (Vec<DataDiff>, usize) {
    let mut sink = DiffSink::new(opts);
    if a.axes() != b.axes() {
        sink.push(
            None,
            0,
            format_args!("shape {:?}", a.axes()),
            format_args!("shape {:?}", b.axes()),
        );
        return sink.finish();
    }
    // Fast path for the common "files match" case: skip decoding two
    // whole images. Only valid when the *scaling* matches as well --
    // identical raw bytes under different BZERO/BSCALE/BLANK decode
    // to different physical pixels.
    let scaling = |h: &Header| (h.bzero(), h.bscale(), h.blank());
    if scaling(a.header()) == scaling(b.header()) && a.raw_bytes() == b.raw_bytes() {
        return sink.finish();
    }
    let (Ok(pa), Ok(pb)) = (a.read_physical(), b.read_physical()) else {
        // Undecodable data still deserves a verdict rather than a
        // silent "identical" -- but a *byte* verdict, not an
        // unconditional one: two HDUs that fail to decode for the
        // same reason and hold the same bytes are not a difference,
        // and reporting one would make a file differ from itself.
        return diff_raw_bytes(a.raw_bytes(), b.raw_bytes(), opts);
    };
    for (i, (&x, &y)) in pa.as_slice().iter().zip(pb.as_slice().iter()).enumerate() {
        if !floats_close(x, y, opts.relative_tolerance, opts.absolute_tolerance) {
            sink.push(None, i, x, y);
        }
    }
    sink.finish()
}

fn hdu_kind_str(h: &Hdu<'_>) -> &'static str {
    match h {
        Hdu::Image(_) => "IMAGE",
        Hdu::BinTable(_) => "BINTABLE",
        Hdu::AsciiTable(_) => "TABLE",
        #[cfg(feature = "compression")]
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
        #[cfg(feature = "compression")]
        Hdu::CompressedImage(c) => c.header(),
        Hdu::RandomGroups(r) => r.header(),
        Hdu::Conforming(c) => c.header(),
    }
}

fn diff_headers(a: &Header, b: &Header, opts: &DiffOptions) -> Vec<HeaderDiff> {
    use std::collections::BTreeMap;
    fn collect(h: &Header) -> BTreeMap<String, Value> {
        let mut out = BTreeMap::new();
        for entry in h.cards() {
            let keyword = entry.keyword();
            if keyword.is_empty() {
                continue;
            }
            if let Some(v) = entry.value() {
                out.entry(keyword).or_insert(v);
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
            Some(vb) if !values_eq(va, vb, opts.relative_tolerance, opts.absolute_tolerance) => {
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

fn values_eq(a: &Value, b: &Value, rtol: f64, atol: f64) -> bool {
    match (a, b) {
        (Value::Real(x), Value::Real(y)) => floats_close(*x, *y, rtol, atol),
        (Value::Integer(x), Value::Integer(y)) => x == y,
        (Value::String(x), Value::String(y)) => x.trim_end() == y.trim_end(),
        (Value::Logical(x), Value::Logical(y)) => x == y,
        (Value::Undefined, Value::Undefined) => true,
        (Value::ComplexReal(x1, y1), Value::ComplexReal(x2, y2)) => {
            floats_close(*x1, *x2, rtol, atol) && floats_close(*y1, *y2, rtol, atol)
        }
        (Value::ComplexInteger(x1, y1), Value::ComplexInteger(x2, y2)) => x1 == x2 && y1 == y2,
        _ => false,
    }
}

fn floats_close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    if a == b {
        return true;
    }
    // NaN != NaN under IEEE, but two headers that both record NaN
    // are not a difference worth reporting.
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if rtol == 0.0 && atol == 0.0 {
        return false;
    }
    // `numpy.isclose` / astropy form, asymmetric in `b` by
    // convention. Guard the infinities: `inf - inf` is NaN, and the
    // `a == b` check above has already accepted matching infinities.
    let delta = (a - b).abs();
    if !delta.is_finite() {
        return false;
    }
    delta <= atol + rtol * b.abs()
}

fn format_value(v: &Value) -> String {
    match v {
        Value::Integer(i) => i.to_string(),
        Value::Real(f) => format!("{f}"),
        Value::String(s) | Value::Unparsed(s) => format!("{s:?}"),
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

/// Report the structural mismatches that make a cell-by-cell table
/// comparison meaningless. Returns `true` when one was recorded.
fn diff_table_structure(
    sink: &mut DiffSink<'_>,
    rows_a: usize,
    rows_b: usize,
    cols_a: &[String],
    cols_b: &[String],
) -> bool {
    if rows_a != rows_b {
        sink.push(
            None,
            0,
            format_args!("{rows_a} rows"),
            format_args!("{rows_b} rows"),
        );
        return true;
    }
    if cols_a != cols_b {
        sink.push(
            None,
            0,
            format_args!("columns {cols_a:?}"),
            format_args!("columns {cols_b:?}"),
        );
        return true;
    }
    false
}

/// The per-column cards that decide how raw table bytes decode into
/// values, sorted so two headers can be compared regardless of card
/// order.
fn table_decoding_cards(h: &Header) -> Vec<(String, Option<Value>)> {
    const PREFIXES: [&str; 5] = ["TSCAL", "TZERO", "TNULL", "TFORM", "TDIM"];
    let mut out: Vec<(String, Option<Value>)> = h
        .cards()
        .map(|c| (c.keyword(), c.value()))
        .filter(|(k, _)| PREFIXES.iter().any(|p| k.starts_with(p)))
        .collect();
    out.sort_by(|x, y| x.0.cmp(&y.0));
    out
}

/// Byte-level short-circuit for the common "files match" case, and
/// the guard that makes an undecodable cell reportable: past this
/// point the two data sections are known to differ somewhere.
///
/// Byte equality only implies value equality when the *decoding* is
/// the same on both sides -- identical bytes under different
/// `TSCAL`/`TZERO` are different physical values. Same reasoning as
/// the image fast path, which compares `BZERO`/`BSCALE`/`BLANK`.
fn table_bytes_identical(a: &[u8], b: &[u8], ha: &Header, hb: &Header) -> bool {
    a == b && table_decoding_cards(ha) == table_decoding_cards(hb)
}

/// Compare a BINTABLE cell by cell, in decoded (post-`TSCAL`/`TZERO`)
/// values, honoring the float tolerances.
fn diff_bintable_data(
    a: &crate::hdu::bintable::BinTableHdu<'_>,
    b: &crate::hdu::bintable::BinTableHdu<'_>,
    opts: &DiffOptions,
) -> (Vec<DataDiff>, usize) {
    let mut sink = DiffSink::new(opts);
    let names_a: Vec<String> = a.columns().iter().map(|c| c.name.clone()).collect();
    let names_b: Vec<String> = b.columns().iter().map(|c| c.name.clone()).collect();
    if diff_table_structure(&mut sink, a.n_rows(), b.n_rows(), &names_a, &names_b) {
        return sink.finish();
    }
    if table_bytes_identical(a.data_bytes(), b.data_bytes(), a.header(), b.header()) {
        return sink.finish();
    }
    // Row-major, not column-major: `max_diffs` truncates the report,
    // and iterating column-outer would spend the whole budget inside
    // the first column. A table whose every column shifted would then
    // read as a problem confined to column 1. `data_total` is right
    // either way; this is about what the truncated sample shows.
    for row in 0..a.n_rows() {
        for (ca, cb) in a.columns().iter().zip(b.columns().iter()) {
            let (Ok(va), Ok(vb)) = (a.cell_value(row, ca), b.cell_value(row, cb)) else {
                sink.push(Some(&ca.name), row, "<undecodable>", "<undecodable>");
                continue;
            };
            if !bin_values_close(&va, &vb, opts) {
                sink.push(
                    Some(&ca.name),
                    row,
                    format_bin_value(&va),
                    format_bin_value(&vb),
                );
            }
        }
    }
    sink.finish()
}

/// Render a BINTABLE cell for the report. Single-element cells --
/// the overwhelming majority -- print as bare scalars rather than
/// `F64([30.0])`; longer vectors keep brackets and elide the tail.
fn format_bin_value(v: &crate::hdu::bintable::BinValue) -> String {
    use crate::hdu::bintable::BinValue as V;
    fn list<T: fmt::Display>(items: &[T]) -> String {
        const SHOWN: usize = 4;
        if items.len() == 1 {
            return items[0].to_string();
        }
        let head: Vec<String> = items.iter().take(SHOWN).map(ToString::to_string).collect();
        if items.len() > SHOWN {
            format!("[{}, ... {} total]", head.join(", "), items.len())
        } else {
            format!("[{}]", head.join(", "))
        }
    }
    fn opts<T: fmt::Display>(items: &[Option<T>]) -> String {
        let rendered: Vec<String> = items
            .iter()
            .map(|o| {
                o.as_ref()
                    .map_or_else(|| "null".to_string(), ToString::to_string)
            })
            .collect();
        list(&rendered)
    }
    match v {
        V::Float(x) | V::F64(x) => list(x),
        V::F32(x) => list(x),
        V::Int(x) => opts(x),
        V::Uint(x) => opts(x),
        V::Logical(x) => opts(x),
        V::Str(s) => format!("{:?}", s.trim_end()),
        V::StrArray(v) => list(
            &v.iter()
                .map(|s| format!("{:?}", s.trim_end()))
                .collect::<Vec<_>>(),
        ),
        V::Bits(bytes, n) => format!("{n} bits {bytes:02x?}"),
        V::C64(x) => list(
            &x.iter()
                .map(|(re, im)| format!("({re}, {im})"))
                .collect::<Vec<_>>(),
        ),
        V::C128(x) => list(
            &x.iter()
                .map(|(re, im)| format!("({re}, {im})"))
                .collect::<Vec<_>>(),
        ),
        V::Vla(inner) => format_bin_value(inner),
    }
}

/// Numeric-aware equality for one BINTABLE cell. Float payloads go
/// through the tolerances; everything else is exact.
fn bin_values_close(
    a: &crate::hdu::bintable::BinValue,
    b: &crate::hdu::bintable::BinValue,
    opts: &DiffOptions,
) -> bool {
    use crate::hdu::bintable::BinValue as V;
    let (rtol, atol) = (opts.relative_tolerance, opts.absolute_tolerance);
    let floats = |x: &[f64], y: &[f64]| {
        x.len() == y.len()
            && x.iter()
                .zip(y)
                .all(|(p, q)| floats_close(*p, *q, rtol, atol))
    };
    match (a, b) {
        (V::Float(x), V::Float(y)) | (V::F64(x), V::F64(y)) => floats(x, y),
        (V::F32(x), V::F32(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|(p, q)| floats_close(f64::from(*p), f64::from(*q), rtol, atol))
        }
        (V::C64(x), V::C64(y)) => {
            x.len() == y.len()
                && x.iter().zip(y).all(|(p, q)| {
                    floats_close(f64::from(p.0), f64::from(q.0), rtol, atol)
                        && floats_close(f64::from(p.1), f64::from(q.1), rtol, atol)
                })
        }
        (V::C128(x), V::C128(y)) => {
            x.len() == y.len()
                && x.iter().zip(y).all(|(p, q)| {
                    floats_close(p.0, q.0, rtol, atol) && floats_close(p.1, q.1, rtol, atol)
                })
        }
        (V::Vla(x), V::Vla(y)) => bin_values_close(x, y, opts),
        (V::Str(x), V::Str(y)) => x.trim_end() == y.trim_end(),
        (V::Logical(x), V::Logical(y)) => x == y,
        (V::Bits(x, nx), V::Bits(y, ny)) => x == y && nx == ny,
        (V::Int(x), V::Int(y)) => x == y,
        (V::Uint(x), V::Uint(y)) => x == y,
        // Differing variants mean differing TFORM, which the header
        // comparison already reports; treat as unequal here too.
        _ => false,
    }
}

/// Compare an ASCII `TABLE` cell by cell.
fn diff_ascii_table_data(
    a: &crate::hdu::ascii_table::AsciiTableHdu<'_>,
    b: &crate::hdu::ascii_table::AsciiTableHdu<'_>,
    opts: &DiffOptions,
) -> (Vec<DataDiff>, usize) {
    use crate::hdu::ascii_table::AsciiCell;
    let mut sink = DiffSink::new(opts);
    let names_a: Vec<String> = a.columns().iter().map(|c| c.name.clone()).collect();
    let names_b: Vec<String> = b.columns().iter().map(|c| c.name.clone()).collect();
    if diff_table_structure(&mut sink, a.n_rows(), b.n_rows(), &names_a, &names_b) {
        return sink.finish();
    }
    if table_bytes_identical(a.data_bytes(), b.data_bytes(), a.header(), b.header()) {
        return sink.finish();
    }
    // Row-major for the same reason as `diff_bintable_data`.
    for row in 0..a.n_rows() {
        for (ca, cb) in a.columns().iter().zip(b.columns().iter()) {
            let (Ok(va), Ok(vb)) = (a.cell_value(row, ca), b.cell_value(row, cb)) else {
                sink.push(Some(&ca.name), row, "<undecodable>", "<undecodable>");
                continue;
            };
            let same = match (&va, &vb) {
                (None, None) => true,
                (Some(AsciiCell::Float(x)), Some(AsciiCell::Float(y))) => {
                    floats_close(*x, *y, opts.relative_tolerance, opts.absolute_tolerance)
                }
                (Some(AsciiCell::Int(x)), Some(AsciiCell::Int(y))) => x == y,
                (Some(AsciiCell::Str(x)), Some(AsciiCell::Str(y))) => x.trim_end() == y.trim_end(),
                _ => false,
            };
            if !same {
                let render = |c: &Option<AsciiCell>| match c {
                    None => "null".to_string(),
                    Some(AsciiCell::Int(i)) => i.to_string(),
                    Some(AsciiCell::Float(x)) => x.to_string(),
                    Some(AsciiCell::Str(s)) => format!("{:?}", s.trim_end()),
                };
                sink.push(Some(&ca.name), row, render(&va), render(&vb));
            }
        }
    }
    sink.finish()
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
                    let loc = match &d.column {
                        Some(col) => format!("{col}[{}]", d.index),
                        None => format!("[{}]", d.index),
                    };
                    writeln!(f, "    {loc}: a={} b={}", d.a_value, d.b_value)?;
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

#[cfg(test)]
mod tests {
    use super::{DiffOptions, FitsDiff, floats_close, values_eq};
    use crate::FitsFile;
    use crate::header::value::Value;

    #[test]
    fn defaults_are_exact_equality() {
        let o = DiffOptions::default();
        assert_eq!(o.relative_tolerance, 0.0);
        assert_eq!(o.absolute_tolerance, 0.0);
        assert!(floats_close(1.0, 1.0, 0.0, 0.0));
        assert!(!floats_close(1.0, 1.0 + f64::EPSILON, 0.0, 0.0));
    }

    /// The reason `atol` exists: no relative tolerance can reconcile
    /// values straddling zero, because the relative term collapses
    /// with `|b|`.
    #[test]
    fn atol_reconciles_values_near_zero() {
        assert!(
            !floats_close(1e-14, 0.0, 1.0, 0.0),
            "rtol is powerless here"
        );
        assert!(floats_close(1e-14, 0.0, 0.0, 1e-12));
        assert!(!floats_close(1e-11, 0.0, 0.0, 1e-12));
    }

    #[test]
    fn rtol_scales_with_magnitude() {
        assert!(floats_close(1_000.1, 1_000.0, 1e-3, 0.0));
        assert!(!floats_close(1_000.1, 1_000.0, 1e-6, 0.0));
    }

    /// The two terms add (`numpy.isclose` form): a gap neither
    /// tolerance covers alone is covered by the pair. Values chosen
    /// to be exactly representable so the assertion tests the
    /// formula rather than rounding.
    #[test]
    fn tolerances_are_additive() {
        // gap = 0.5; rtol term = 0.1; atol term = 0.45.
        assert!(!floats_close(1_000.5, 1_000.0, 1e-4, 0.0));
        assert!(!floats_close(1_000.5, 1_000.0, 0.0, 0.45));
        assert!(floats_close(1_000.5, 1_000.0, 1e-4, 0.45));
    }

    #[test]
    fn non_finite_values_are_handled() {
        assert!(
            floats_close(f64::NAN, f64::NAN, 0.0, 0.0),
            "NaN == NaN here"
        );
        assert!(floats_close(f64::INFINITY, f64::INFINITY, 0.0, 0.0));
        assert!(!floats_close(f64::INFINITY, f64::NEG_INFINITY, 1.0, 1e30));
        assert!(!floats_close(f64::INFINITY, 1.0, 1.0, 1e30));
        assert!(!floats_close(f64::NAN, 1.0, 1.0, 1e30));
    }

    #[test]
    fn tolerances_reach_card_values() {
        let a = Value::Real(0.0);
        let b = Value::Real(1e-14);
        assert!(!values_eq(&a, &b, 0.0, 0.0));
        assert!(values_eq(&a, &b, 0.0, 1e-12));
        // Non-float kinds stay exact regardless of tolerance.
        assert!(!values_eq(
            &Value::Integer(1),
            &Value::Integer(2),
            1.0,
            1e30
        ));
        assert!(values_eq(
            &Value::ComplexReal(0.0, 0.0),
            &Value::ComplexReal(1e-14, -1e-14),
            0.0,
            1e-12
        ));
    }

    // --- data comparison -------------------------------------------------

    use crate::hdu::builder::ImageBuilder;
    use crate::io::FitsWriter;

    /// Build a one-image FITS file in memory.
    fn image_file(pixels: Vec<f32>, axes: Vec<u64>) -> FitsFile {
        let (header, data) = ImageBuilder::new(axes, pixels)
            .unwrap()
            .primary(true)
            .build()
            .unwrap();
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu(&header, &data).unwrap();
        w.finish().unwrap();
        FitsFile::from_bytes(buf).unwrap()
    }

    fn diff_of(a: &FitsFile, b: &FitsFile, opts: DiffOptions) -> FitsDiff {
        FitsDiff::compare(a, b, opts).unwrap()
    }

    /// The tolerances must reach *pixels*, not just header cards.
    /// Regression: image data was compared byte-for-byte, so `rtol`
    /// and `atol` were silently ignored on the actual data.
    #[test]
    fn tolerances_reach_image_pixels() {
        let a = image_file(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = image_file(vec![1.0, 2.0, 3.0, 4.000_001], vec![2, 2]);

        let exact = diff_of(&a, &b, DiffOptions::default());
        assert!(!exact.is_identical(), "a 1e-6 pixel change must be seen");

        let loose = diff_of(
            &a,
            &b,
            DiffOptions {
                relative_tolerance: 1e-3,
                ..Default::default()
            },
        );
        assert!(
            loose.is_identical(),
            "rtol=1e-3 must absorb a 2.5e-7 relative pixel change, got {loose}"
        );
    }

    /// Reported indices are pixel numbers with decoded values, not
    /// byte offsets with hex bytes. Regression: one differing f64
    /// pixel used to produce up to eight entries.
    #[test]
    fn image_diffs_are_pixels_not_bytes() {
        let a = image_file(vec![0.0; 6], vec![3, 2]);
        let mut px = vec![0.0_f32; 6];
        px[4] = 9.0;
        let b = image_file(px, vec![3, 2]);

        let d = diff_of(&a, &b, DiffOptions::default());
        let hdu = &d.hdus[0];
        assert_eq!(hdu.data_total, 1, "one pixel changed, one diff expected");
        assert_eq!(hdu.data[0].index, 4, "index should be the pixel number");
        assert_eq!(hdu.data[0].column, None);
        assert_eq!(hdu.data[0].a_value, "0");
        assert_eq!(hdu.data[0].b_value, "9");
    }

    /// Scaled integers are compared in physical units, so two files
    /// whose raw bytes differ but whose physical pixels agree are
    /// identical -- and vice versa.
    #[test]
    fn image_comparison_uses_physical_units() {
        // Same physical values (2, 4, 6, 8) via different BZERO/BSCALE.
        let build = |raw: Vec<i16>, bscale: f64, bzero: f64| {
            let (mut header, data) = ImageBuilder::new(vec![2_u64, 2], raw)
                .unwrap()
                .primary(true)
                .build()
                .unwrap();
            header.set("BSCALE", Value::Real(bscale), None).unwrap();
            header.set("BZERO", Value::Real(bzero), None).unwrap();
            let mut buf = Vec::new();
            let mut w = FitsWriter::new(&mut buf);
            w.write_hdu(&header, &data).unwrap();
            w.finish().unwrap();
            FitsFile::from_bytes(buf).unwrap()
        };
        // BSCALE itself is a header card, and differing header cards
        // are a real difference -- so compare the *data* verdict.
        let a = build(vec![1, 2, 3, 4], 2.0, 0.0);
        let b = build(vec![2, 4, 6, 8], 1.0, 0.0);
        let d = diff_of(&a, &b, DiffOptions::default());
        assert_eq!(
            d.hdus[0].data_total, 0,
            "different raw bytes, identical physical pixels: {d}"
        );

        let c = build(vec![1, 2, 3, 4], 2.0, 1.0);
        let d = diff_of(&a, &c, DiffOptions::default());
        assert_eq!(
            d.hdus[0].data_total, 4,
            "identical raw bytes, every physical pixel shifted by BZERO: {d}"
        );
    }

    /// Table data used to be ignored entirely: two tables whose
    /// headers agreed compared identical whatever their rows held.
    #[test]
    fn table_data_is_compared() {
        use crate::hdu::bintable::BinFieldKind;
        use crate::hdu::builder::BinTableBuilder;

        fn table_file(ra: [f64; 3]) -> FitsFile {
            let (primary_h, primary_d) = ImageBuilder::<f32>::new(vec![0_u64; 0], Vec::new())
                .unwrap()
                .primary(true)
                .build()
                .unwrap();
            let mut bt = BinTableBuilder::new();
            bt.add_column("RA", BinFieldKind::F64, 1, None, None)
                .unwrap();
            let mut rows = Vec::new();
            for v in ra {
                rows.extend_from_slice(&v.to_bits().to_be_bytes());
            }
            let (th, td) = bt.build(3, rows).unwrap();
            let mut buf = Vec::new();
            let mut w = FitsWriter::new(&mut buf);
            w.write_hdu(&primary_h, &primary_d).unwrap();
            w.write_hdu(&th, &td).unwrap();
            w.finish().unwrap();
            FitsFile::from_bytes(buf).unwrap()
        }

        let a = table_file([10.0, 20.0, 30.0]);
        let b = table_file([10.0, 20.0, 31.0]);
        let d = diff_of(&a, &b, DiffOptions::default());
        assert!(!d.is_identical(), "a changed table cell must be reported");
        let t = &d.hdus[1];
        assert_eq!(t.data_total, 1, "exactly one cell differs: {d}");
        assert_eq!(t.data[0].column.as_deref(), Some("RA"));
        assert_eq!(t.data[0].index, 2, "row index");

        // And the tolerances reach table cells too.
        let c = table_file([10.0, 20.0, 30.000_001]);
        assert!(
            diff_of(
                &a,
                &c,
                DiffOptions {
                    relative_tolerance: 1e-3,
                    ..Default::default()
                }
            )
            .is_identical(),
            "rtol must absorb a tiny cell change"
        );
        assert!(
            !diff_of(&a, &c, DiffOptions::default()).is_identical(),
            "and must still catch it at the exact default"
        );
    }

    /// `max_diffs` truncates the report, so the order cells are
    /// visited decides what a truncated report *shows*. Row-major
    /// means the sample spans columns; column-major would spend the
    /// whole budget inside column 1 and make a table-wide shift look
    /// like a single-column problem.
    #[test]
    fn truncated_table_report_samples_every_column() {
        use crate::hdu::bintable::BinFieldKind;
        use crate::hdu::builder::BinTableBuilder;

        const ROWS: usize = 4;

        // Two f64 columns, every cell offset by `shift`, so *both*
        // columns differ in *every* row and only the ordering can
        // decide which differences land in the report.
        fn two_column_file(shift: f64) -> FitsFile {
            let (ph, pd) = ImageBuilder::<f32>::new(vec![0_u64; 0], Vec::new())
                .unwrap()
                .primary(true)
                .build()
                .unwrap();
            let mut bt = BinTableBuilder::new();
            for name in ["X", "Y"] {
                bt.add_column(name, BinFieldKind::F64, 1, None, None)
                    .unwrap();
            }
            let mut rows = Vec::new();
            for r in 0..ROWS {
                for col in 0..2 {
                    let v = (r * 2 + col) as f64 + shift;
                    rows.extend_from_slice(&v.to_bits().to_be_bytes());
                }
            }
            let (th, td) = bt.build(ROWS, rows).unwrap();
            let mut buf = Vec::new();
            let mut w = FitsWriter::new(&mut buf);
            w.write_hdu(&ph, &pd).unwrap();
            w.write_hdu(&th, &td).unwrap();
            w.finish().unwrap();
            FitsFile::from_bytes(buf).unwrap()
        }

        let d = diff_of(
            &two_column_file(0.0),
            &two_column_file(1.0),
            DiffOptions {
                max_diffs: 2,
                ..Default::default()
            },
        );
        let t = &d.hdus[1];
        assert_eq!(
            t.data_total,
            ROWS * 2,
            "the total must count past the cap: {d}"
        );
        assert_eq!(t.data.len(), 2, "the report itself is capped at 2");
        let columns: Vec<Option<&str>> = t.data.iter().map(|x| x.column.as_deref()).collect();
        assert_eq!(
            columns,
            vec![Some("X"), Some("Y")],
            "a two-entry report of a table-wide shift must show both \
             columns, not row 0 and row 1 of column X: {d}"
        );
        assert!(
            t.data.iter().all(|x| x.index == 0),
            "both entries should come from row 0: {d}"
        );
    }

    /// Identical table bytes short-circuit before any cell is
    /// decoded -- but only when the cards that decide how those bytes
    /// decode agree, since the same bytes under a different `TSCAL`
    /// are different values.
    #[test]
    fn table_fast_path_respects_scaling_cards() {
        use crate::hdu::bintable::BinFieldKind;
        use crate::hdu::builder::BinTableBuilder;

        fn scaled_table(tscal: Option<f64>) -> FitsFile {
            let (ph, pd) = ImageBuilder::<f32>::new(vec![0_u64; 0], Vec::new())
                .unwrap()
                .primary(true)
                .build()
                .unwrap();
            let mut bt = BinTableBuilder::new();
            bt.add_column("V", BinFieldKind::I32, 1, None, None)
                .unwrap();
            let mut rows = Vec::new();
            for v in [1_i32, 2, 3] {
                rows.extend_from_slice(&v.to_be_bytes());
            }
            let (mut th, td) = bt.build(3, rows).unwrap();
            if let Some(s) = tscal {
                th.set("TSCAL1", Value::Real(s), None).unwrap();
            }
            let mut buf = Vec::new();
            let mut w = FitsWriter::new(&mut buf);
            w.write_hdu(&ph, &pd).unwrap();
            w.write_hdu(&th, &td).unwrap();
            w.finish().unwrap();
            FitsFile::from_bytes(buf).unwrap()
        }

        // Same bytes, same scaling: nothing to report.
        let a = scaled_table(None);
        assert_eq!(
            diff_of(&a, &scaled_table(None), DiffOptions::default()).hdus[1].data_total,
            0
        );

        // Same bytes, different scaling: every physical value differs,
        // and the byte short-circuit must not hide that.
        let d = diff_of(&a, &scaled_table(Some(10.0)), DiffOptions::default());
        assert_eq!(
            d.hdus[1].data_total, 3,
            "identical bytes under TSCAL=10 are different values: {d}"
        );
    }

    /// An unrecognized `XTENSION` still has to get a verdict.
    ///
    /// Regression: unknown extensions fell through to "no
    /// differences", so two files with matching headers and wholly
    /// different bytes reported "Files are identical." Random groups
    /// already had a byte fallback for the same reason.
    #[test]
    fn unknown_extension_data_is_not_assumed_identical() {
        fn card(s: &str) -> Vec<u8> {
            let mut b = vec![b' '; 80];
            b[..s.len()].copy_from_slice(s.as_bytes());
            b
        }
        fn file_with(fill: u8) -> FitsFile {
            let mut buf: Vec<u8> = Vec::new();
            for c in [
                "SIMPLE  =                    T",
                "BITPIX  =                    8",
                "NAXIS   =                    0",
                "EXTEND  =                    T",
                "END",
            ] {
                buf.extend_from_slice(&card(c));
            }
            while !buf.len().is_multiple_of(2880) {
                buf.push(b' ');
            }
            for c in [
                "XTENSION= 'WEIRDEXT'",
                "BITPIX  =                    8",
                "NAXIS   =                    1",
                "NAXIS1  =                 2880",
                "PCOUNT  =                    0",
                "GCOUNT  =                    1",
                "END",
            ] {
                buf.extend_from_slice(&card(c));
            }
            while !buf.len().is_multiple_of(2880) {
                buf.push(b' ');
            }
            buf.extend(std::iter::repeat_n(fill, 2880));
            FitsFile::from_bytes(buf).unwrap()
        }

        let a = file_with(0x11);
        assert!(
            diff_of(&a, &file_with(0x11), DiffOptions::default()).is_identical(),
            "a file must not differ from its own twin"
        );
        let d = diff_of(&a, &file_with(0x99), DiffOptions::default());
        assert!(
            !d.is_identical(),
            "2880 differing data bytes reported as identical: {d}"
        );
    }
}
