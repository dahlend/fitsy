//! WCS keywords carried in a binary table (Standard Sec.8.2, Table 22).
//!
//! Table 22 gives every WCS keyword three spellings. The image form
//! (`CTYPEia`, `CRVALia`, ...) applies to a `NAXIS`-dimensional array in
//! the data section and is handled by [`Wcs::from_header`]. The other
//! two live in a `BINTABLE` header:
//!
//! * BINTABLE vector. An image stored in a vector cell of column `n`,
//!   described by `iCTYPn`, `jCRPXn`, `ijPCna` and their siblings.
//!   Here `i` and `j` are axis numbers within the cell, and `n` is the
//!   column number.
//!
//! * Pixel list. One image whose pixel coordinates are spread across
//!   scalar columns, one per axis, described by `TCTYPn`, `TCRPXn`,
//!   `TPn_ka` and their siblings. Here `n` is a column number and also
//!   the axis number. An event list from a high-energy instrument
//!   georeferences its sky columns this way.
//!
//! Both are translated into a synthetic image-form header and handed to
//! the existing parser, so projections, `SIP`/`TPV`, spectral axes and
//! `-TAB` all work unchanged.
//!
//! The standard leaves two points open. This module resolves them as
//! follows:
//!
//! 1. Sec.8.2 does not say how pixel-list columns map to axis numbers.
//!    This module sorts the participating columns in ascending order
//!    and numbers them from 1.
//! 2. For the table-scoped global keywords (`LONPna`, `EQUIna`,
//!    `MJDOBn`, `OBSGXn`, ...) the column index `n` is ignored; they are
//!    matched on the alternate letter alone. An image-form keyword in
//!    the same header supplies the default when the table form is
//!    absent.

use std::collections::BTreeMap;

use crate::error::{FitsError, Result};
use crate::header::Header;
use crate::header::value::Value;
use crate::wcs::{Wcs, alt_suffix};

/// A coordinate description read out of a binary table header.
#[derive(Debug, Clone)]
pub struct TableWcs {
    /// The coordinate description itself, identical in kind to one
    /// parsed from an image header.
    pub wcs: Wcs,
    /// Pixel-list form: the 1-based table column supplying each WCS
    /// axis, in axis order, so `colax[0]` is the column holding axis 1.
    /// Empty for the vector-column form.
    pub colax: Vec<usize>,
    /// Vector-column form: the 1-based column whose cells hold the
    /// image. `None` for a pixel list.
    pub column: Option<usize>,
}

/// Per-axis keywords, as `(image root, pixel-list roots, BINTABLE
/// roots)`. Where the primary and alternate spellings use different
/// roots, both are listed and tried in order; a trailing alternate
/// letter is optional on each.
///
/// The four auxiliary rows (`CRDER`, `CSYER`, `CZPHS`, `CPERI`) are
/// spelled inconsistently between Table 22's main block and its time
/// block, so every spelling either block gives is accepted.
const AXIS_KEYWORDS: &[(&str, &[&str], &[&str])] = &[
    ("CTYPE", &["TCTYP", "TCTY"], &["CTYP", "CTY"]),
    ("CUNIT", &["TCUNI", "TCUN"], &["CUNI", "CUN"]),
    ("CRVAL", &["TCRVL", "TCRV"], &["CRVL", "CRV"]),
    ("CDELT", &["TCDLT", "TCDE"], &["CDLT", "CDE"]),
    ("CRPIX", &["TCRPX", "TCRP"], &["CRPX", "CRP"]),
    ("CROTA", &["TCROT"], &["CROT"]),
    ("CNAME", &["TCNAM", "TCNA"], &["CNAM", "CNA"]),
    ("CRDER", &["TCRDE", "TCRD"], &["CRDE", "CRD"]),
    ("CSYER", &["TCSYE", "TCSY"], &["CSYE", "CSY"]),
    ("CZPHS", &["TCZPH", "TCZP"], &["CZPH", "CZP"]),
    ("CPERI", &["TCPER", "TCPR"], &["CPER", "CPR"]),
];

/// Keywords that describe the representation as a whole. Table 22 gives
/// them a column index which, per `wcsbth()`, carries no meaning; only
/// the alternate letter selects. `has_alt` is false for the rows Table
/// 22 leaves without an alternate code.
const GLOBAL_KEYWORDS: &[(&str, &[&str], bool)] = &[
    ("WCSAXES", &["WCAX"], true),
    ("WCSNAME", &["WCSN", "TWCS"], true),
    ("LONPOLE", &["LONP"], true),
    ("LATPOLE", &["LATP"], true),
    ("EQUINOX", &["EQUI"], true),
    ("RADESYS", &["RADE"], true),
    ("RESTFRQ", &["RFRQ"], true),
    ("RESTWAV", &["RWAV"], true),
    ("SPECSYS", &["SPEC"], true),
    ("SSYSOBS", &["SOBS"], true),
    ("SSYSSRC", &["SSRC"], true),
    ("VELOSYS", &["VSYS"], true),
    ("ZSOURCE", &["ZSOU"], true),
    ("VELANGL", &["VANG"], true),
    ("DATE-OBS", &["DOBS"], false),
    ("MJD-OBS", &["MJDOB"], false),
    ("DATE-AVG", &["DAVG"], false),
    ("MJD-AVG", &["MJDA"], false),
    ("TREFPOS", &["TRPOS"], false),
    ("TREFDIR", &["TRDIR"], false),
    ("OBSGEO-X", &["OBSGX"], false),
    ("OBSGEO-Y", &["OBSGY"], false),
    ("OBSGEO-Z", &["OBSGZ"], false),
];

/// Image-form keywords that are per-axis and must therefore never be
/// inherited into a synthetic header: their axis numbers refer to a
/// different image than the one being described.
const AXIS_ROOTS_TO_STRIP: &[&str] = &[
    "CTYPE", "CUNIT", "CRVAL", "CDELT", "CRPIX", "CROTA", "CRDER", "CSYER", "CZPHS", "CPERI",
    "CNAME", "PC", "CD", "PV", "PS", "WCSAXES", "NAXIS",
];

impl TableWcs {
    /// Parse the pixel-list representation for descriptor `alt`.
    ///
    /// Pass a space for `alt` to select the primary description, or a
    /// letter from `A` to `Z` for an alternate.
    ///
    /// The result is `Ok(None)` when the header carries no pixel-list
    /// axis keyword for that descriptor.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `alt` is neither a space nor a letter
    /// from `A` to `Z`, or when the keywords describe a WCS that the
    /// parser rejects.
    pub fn from_pixel_list(header: &Header, alt: char) -> Result<Option<Self>> {
        let suffix = alt_suffix(alt)?;
        let mut columns: Vec<usize> = Vec::new();
        for entry in header.entries() {
            if let Some(col) = pixel_list_axis_column(&entry.keyword, alt)
                && !columns.contains(&col)
            {
                columns.push(col);
            }
        }
        if columns.is_empty() {
            return Ok(None);
        }
        // Sec.8.2 does not define the column-to-axis mapping; wcsbth()
        // walks columns in ascending order filling colax[], so axis 1
        // is the lowest-numbered participating column.
        columns.sort_unstable();
        let axis_of = |col: usize| columns.iter().position(|&c| c == col).map(|i| i + 1);

        let mut out = inherited_cards(header);
        for (image_root, pix_roots, _) in AXIS_KEYWORDS {
            for (axis, &col) in columns.iter().enumerate() {
                for root in *pix_roots {
                    if let Some(v) = header.first(&format!("{root}{col}{suffix}")) {
                        out.insert(image_key(image_root, axis + 1, &suffix), v.clone());
                        break;
                    }
                }
            }
        }
        // Matrices and coordinate parameters index columns, not axes,
        // on both sides of the underscore.
        for entry in header.entries() {
            let kw = &entry.keyword;
            let Some(value) = entry.value.as_ref() else {
                continue;
            };
            for (image_root, roots) in [("PC", ["TPC", "TP"]), ("CD", ["TCD", "TC"])] {
                for root in roots {
                    if let Some((n, k, a)) = split_pair(kw, root)
                        && a == alt
                        && let (Some(i), Some(j)) = (axis_of(n), axis_of(k))
                    {
                        out.insert(format!("{image_root}{i}_{j}{suffix}"), value.clone());
                    }
                }
            }
            for (image_root, roots) in [("PV", ["TPV", "TV"]), ("PS", ["TPS", "TS"])] {
                for root in roots {
                    if let Some((n, m, a)) = split_pair(kw, root)
                        && a == alt
                        && let Some(i) = axis_of(n)
                    {
                        out.insert(format!("{image_root}{i}_{m}{suffix}"), value.clone());
                    }
                }
            }
        }
        insert_globals(header, &mut out, alt, &suffix);
        // A pixel list has no pixel array, so leave NAXISn unset:
        // `pixel_shape` stays `None` rather than reporting the table's
        // own row/byte counts as an image extent.
        out.insert("NAXIS".into(), Value::Integer(columns.len() as i64));

        let synthetic = build_header(out);
        Ok(Wcs::from_header(&synthetic, alt)?.map(|wcs| Self {
            wcs,
            colax: columns,
            column: None,
        }))
    }

    /// Parse the image stored in the vector cells of column `column`
    /// for descriptor `alt`. The `column` argument is 1-based.
    ///
    /// The result is `Ok(None)` when the header carries no `iCTYPn`
    /// axis keyword for that column and that descriptor.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in three cases:
    ///
    /// - `alt` is neither a space nor a letter from `A` to `Z`.
    /// - `column` falls outside the range 1 to 999.
    /// - The keywords describe a WCS that the parser rejects.
    pub fn from_table_column(header: &Header, column: usize, alt: char) -> Result<Option<Self>> {
        let suffix = alt_suffix(alt)?;
        if column == 0 || column > 999 {
            return Err(FitsError::Wcs(format!(
                "BINTABLE column {column} out of range 1..=999"
            )));
        }
        let mut max_axis = 0_usize;
        for entry in header.entries() {
            if let Some(axis) = bintable_axis(&entry.keyword, column, alt) {
                max_axis = max_axis.max(axis);
            }
        }
        if max_axis == 0 {
            return Ok(None);
        }
        // WCAXna may declare more axes than any keyword mentions.
        let declared = header
            .first(&format!("WCAX{column}{suffix}"))
            .and_then(|v| match v {
                Value::Integer(n) if *n > 0 && *n <= 999 => Some(*n as usize),
                _ => None,
            });
        let naxes = declared.unwrap_or(max_axis).max(max_axis);

        let mut out = inherited_cards(header);
        for (image_root, _, bin_roots) in AXIS_KEYWORDS {
            for axis in 1..=naxes {
                for root in *bin_roots {
                    if let Some(v) = header.first(&format!("{axis}{root}{column}{suffix}")) {
                        out.insert(image_key(image_root, axis, &suffix), v.clone());
                        break;
                    }
                }
            }
        }
        for entry in header.entries() {
            let kw = &entry.keyword;
            let Some(value) = entry.value.as_ref() else {
                continue;
            };
            for (image_root, root) in [("PC", "PC"), ("CD", "CD")] {
                if let Some((i, j, n, a)) = split_matrix_prefixed(kw, root)
                    && a == alt
                    && n == column
                {
                    out.insert(format!("{image_root}{i}_{j}{suffix}"), value.clone());
                }
            }
            for (image_root, roots) in [("PV", ["PV", "V"]), ("PS", ["PS", "S"])] {
                for root in roots {
                    if let Some((i, n, m, a)) = split_param_prefixed(kw, root)
                        && a == alt
                        && n == column
                    {
                        out.insert(format!("{image_root}{i}_{m}{suffix}"), value.clone());
                    }
                }
            }
        }
        insert_globals(header, &mut out, alt, &suffix);
        out.insert("NAXIS".into(), Value::Integer(naxes as i64));
        // `TDIMn` gives the cell's shape in FITS axis order, which is
        // exactly what `NAXISn` means, so a vector-column image can
        // report a real `pixel_shape`.
        if let Some(dims) = tdim(header, column)
            && dims.len() == naxes
        {
            for (axis, d) in dims.iter().enumerate() {
                out.insert(format!("NAXIS{}", axis + 1), Value::Integer(*d as i64));
            }
        }

        let synthetic = build_header(out);
        Ok(Wcs::from_header(&synthetic, alt)?.map(|wcs| Self {
            wcs,
            colax: Vec::new(),
            column: Some(column),
        }))
    }

    /// Alternate codes for which the header carries a pixel-list
    /// description, in order, with `' '` for the primary one.
    #[must_use]
    pub fn pixel_list_alternates(header: &Header) -> Vec<char> {
        let mut out = Vec::new();
        for entry in header.entries() {
            if let Some((_, alt)) = split_pixel_list_axis(&entry.keyword)
                && !out.contains(&alt)
            {
                out.push(alt);
            }
        }
        // `' '` is 0x20, so ascending order puts the primary first.
        out.sort_unstable();
        out
    }

    /// Columns holding a vector-cell image, in ascending order, paired
    /// with the alternate codes each one defines.
    #[must_use]
    pub fn image_columns(header: &Header) -> Vec<(usize, Vec<char>)> {
        let mut by_col: BTreeMap<usize, Vec<char>> = BTreeMap::new();
        for entry in header.entries() {
            let Some((_, col, alt)) = split_bintable_axis(&entry.keyword) else {
                continue;
            };
            let alts = by_col.entry(col).or_default();
            if !alts.contains(&alt) {
                alts.push(alt);
            }
        }
        by_col.into_iter().collect()
    }
}

fn image_key(root: &str, axis: usize, suffix: &str) -> String {
    format!("{root}{axis}{suffix}")
}

/// Every card that is safe to carry into the synthetic header: the
/// per-axis WCS keywords are dropped because their axis numbers belong
/// to a different image, and so is the table's own geometry. What
/// remains -- `TIMESYS`, `MJDREF*`, `OBSGEO-*`, `DATE-OBS`, ... -- is
/// what `Header`'s own accessors read, and Table 22 makes those apply
/// to the table's coordinate descriptions too.
fn inherited_cards(header: &Header) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for entry in header.entries() {
        let Some(value) = entry.value.as_ref() else {
            continue;
        };
        if is_image_axis_keyword(&entry.keyword) {
            continue;
        }
        out.entry(entry.keyword.clone())
            .or_insert_with(|| value.clone());
    }
    out
}

/// True for an image-form keyword parameterized by axis number, e.g.
/// `CTYPE2`, `PC1_2A`, `NAXIS3`. `CD` is matched only in its `CDi_j`
/// form so `CDELT`-like names are not caught by the `CD` prefix.
fn is_image_axis_keyword(kw: &str) -> bool {
    // `WCSAXES`, bare or with an alternate letter (`WCSAXESA`): it
    // dictates the image description's axis count, which must never
    // stand in for the table's.
    if let Some(rest) = kw.strip_prefix("WCSAXES")
        && (rest.is_empty() || (rest.len() == 1 && rest.as_bytes()[0].is_ascii_uppercase()))
    {
        return true;
    }
    if split_indexed(kw, "WCSAXES").is_some() {
        return true;
    }
    for root in AXIS_ROOTS_TO_STRIP {
        if matches!(*root, "PC" | "CD" | "PV" | "PS") {
            if split_pair(kw, root).is_some() {
                return true;
            }
        } else if split_indexed(kw, root).is_some() {
            return true;
        }
    }
    false
}

fn insert_globals(header: &Header, out: &mut BTreeMap<String, Value>, alt: char, suffix: &str) {
    for (image_key, roots, has_alt) in GLOBAL_KEYWORDS {
        let mut found: Option<Value> = None;
        'roots: for root in *roots {
            for entry in header.entries() {
                let Some(value) = entry.value.as_ref() else {
                    continue;
                };
                let matched = if *has_alt {
                    split_indexed(&entry.keyword, root).is_some_and(|(_, a)| a == alt)
                } else {
                    // No alternate code defined for this row, so the
                    // whole tail must be the (ignored) column index.
                    split_indexed(&entry.keyword, root).is_some_and(|(_, a)| a == ' ')
                };
                if matched {
                    found = Some(value.clone());
                    break 'roots;
                }
            }
        }
        if let Some(v) = found {
            // The table form overrides the image-form default that
            // `inherited_cards` may already have placed here.
            out.insert(format!("{image_key}{suffix}"), v);
        }
    }
}

/// `(column, alternate)` if `kw` is a pixel-list keyword that defines
/// an axis. Only the keywords that establish an axis count -- not the
/// matrix or parameter forms, which index columns that must already be
/// known.
fn split_pixel_list_axis(kw: &str) -> Option<(usize, char)> {
    for (_, pix_roots, _) in AXIS_KEYWORDS {
        for root in *pix_roots {
            if let Some((col, alt)) = split_indexed(kw, root)
                && (1..=999).contains(&col)
            {
                return Some((col, alt));
            }
        }
    }
    None
}

/// The column number if `kw` is a pixel-list axis keyword for `alt`.
fn pixel_list_axis_column(kw: &str, alt: char) -> Option<usize> {
    split_pixel_list_axis(kw).and_then(|(col, a)| (a == alt).then_some(col))
}

/// `(axis, column, alternate)` if `kw` is a BINTABLE-vector axis
/// keyword, e.g. `2CTYP3A` -> `(2, 3, 'A')`.
fn split_bintable_axis(kw: &str) -> Option<(usize, usize, char)> {
    for (_, _, bin_roots) in AXIS_KEYWORDS {
        for root in *bin_roots {
            if let Some((axis, col, alt)) = split_prefixed(kw, root)
                && (1..=999).contains(&axis)
                && (1..=999).contains(&col)
            {
                return Some((axis, col, alt));
            }
        }
    }
    None
}

/// The axis number if `kw` is a BINTABLE-vector keyword for `column`
/// and `alt`.
fn bintable_axis(kw: &str, column: usize, alt: char) -> Option<usize> {
    split_bintable_axis(kw).and_then(|(axis, col, a)| (col == column && a == alt).then_some(axis))
}

/// Split `root` + decimal index + optional alternate letter, e.g.
/// `TCTYP3` -> `(3, ' ')`, `TCTY3A` -> `(3, 'A')`. Leading zeros are
/// rejected: Sec.4.1.2.1 forbids them in an indexed keyword.
fn split_indexed(kw: &str, root: &str) -> Option<(usize, char)> {
    let rest = kw.strip_prefix(root)?;
    let (digits, alt) = split_trailing_alt(rest);
    parse_index(digits).map(|n| (n, alt))
}

/// Split `root` + `n` + `_` + `k` + optional alternate letter, e.g.
/// `TP1_2A` -> `(1, 2, 'A')`.
fn split_pair(kw: &str, root: &str) -> Option<(usize, usize, char)> {
    let rest = kw.strip_prefix(root)?;
    let (body, alt) = split_trailing_alt(rest);
    let (a, b) = body.split_once('_')?;
    Some((parse_index(a)?, parse_index(b)?, alt))
}

/// Split `i` + `j` + `root` + `n` + optional alternate letter, e.g.
/// `12PC3A` -> `(1, 2, 3, 'A')`. Table 22 gives `i` and `j` one digit
/// each here; two would overrun the 8-character keyword field.
fn split_matrix_prefixed(kw: &str, root: &str) -> Option<(usize, usize, usize, char)> {
    let mut chars = kw.chars();
    let i = chars.next()?.to_digit(10)? as usize;
    let j = chars.next()?.to_digit(10)? as usize;
    let rest: String = chars.collect();
    let (col, alt) = split_indexed(&rest, root)?;
    if i == 0 || j == 0 {
        return None;
    }
    Some((i, j, col, alt))
}

/// Split `i` + `root` + `n` + `_` + `m` + optional alternate letter,
/// e.g. `1V3_2A` -> `(1, 3, 2, 'A')`.
fn split_param_prefixed(kw: &str, root: &str) -> Option<(usize, usize, usize, char)> {
    let digits: String = kw.chars().take_while(char::is_ascii_digit).collect();
    let axis = parse_index(&digits)?;
    let (n, m, alt) = split_pair(&kw[digits.len()..], root)?;
    Some((axis, n, m, alt))
}

/// Split `i` + `root` + `n` + optional alternate letter, e.g.
/// `2CTYP3A` -> `(2, 3, 'A')`.
fn split_prefixed(kw: &str, root: &str) -> Option<(usize, usize, char)> {
    let digits: String = kw.chars().take_while(char::is_ascii_digit).collect();
    let axis = parse_index(&digits)?;
    let (col, alt) = split_indexed(&kw[digits.len()..], root)?;
    Some((axis, col, alt))
}

fn split_trailing_alt(s: &str) -> (&str, char) {
    match s.chars().next_back() {
        Some(c) if c.is_ascii_uppercase() => (&s[..s.len() - c.len_utf8()], c),
        _ => (s, ' '),
    }
}

fn parse_index(s: &str) -> Option<usize> {
    if s.is_empty() || s.len() > 3 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if s.len() > 1 && s.starts_with('0') {
        return None;
    }
    let n: usize = s.parse().ok()?;
    (n >= 1).then_some(n)
}

/// `TDIMn` as a list of extents, fastest axis first.
fn tdim(header: &Header, column: usize) -> Option<Vec<usize>> {
    let raw = header.optional_string(&format!("TDIM{column}"))?;
    let inner = raw.trim().strip_prefix('(')?.strip_suffix(')')?;
    inner
        .split(',')
        .map(|p| p.trim().parse::<usize>().ok().filter(|&d| d > 0))
        .collect()
}

fn build_header(cards: BTreeMap<String, Value>) -> Header {
    let mut h = Header::empty();
    for (k, v) in cards {
        // A keyword copied from the source header could be a HIERARCH
        // or otherwise unrepresentable name; those carry no WCS meaning,
        // so drop them rather than failing the whole parse.
        let _ = h.push(k, v, None);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_split_rejects_leading_zero_and_overlong() {
        assert_eq!(split_indexed("TCTYP3", "TCTYP"), Some((3, ' ')));
        assert_eq!(split_indexed("TCTY12A", "TCTY"), Some((12, 'A')));
        assert_eq!(split_indexed("TCTYP03", "TCTYP"), None);
        assert_eq!(split_indexed("TCTYP1234", "TCTYP"), None);
        assert_eq!(split_indexed("TCTYP", "TCTYP"), None);
    }

    #[test]
    fn pair_and_prefixed_splits() {
        assert_eq!(split_pair("TP1_2", "TP"), Some((1, 2, ' ')));
        assert_eq!(split_pair("TPC10_2A", "TPC"), Some((10, 2, 'A')));
        assert_eq!(split_matrix_prefixed("12PC3A", "PC"), Some((1, 2, 3, 'A')));
        assert_eq!(split_param_prefixed("1V3_2A", "V"), Some((1, 3, 2, 'A')));
        assert_eq!(split_prefixed("2CTYP3", "CTYP"), Some((2, 3, ' ')));
    }

    #[test]
    fn image_axis_keywords_are_recognized_for_stripping() {
        for kw in [
            "CTYPE1", "CRVAL2A", "PC1_2", "CD2_1B", "PV2_1", "NAXIS3", "WCSAXES", "WCSAXESA",
        ] {
            assert!(is_image_axis_keyword(kw), "{kw}");
        }
        for kw in ["TIMESYS", "MJDREFI", "OBSGEO-X", "EQUINOX", "TCTYP1"] {
            assert!(!is_image_axis_keyword(kw), "{kw}");
        }
    }
}
