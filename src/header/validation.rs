//! Header diagnostics for deprecated and non-standard keywords.

use crate::header::Header;
use crate::header::value::Value;

/// Severity of a [`Diagnostic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Level {
    /// Non-standard or outdated, but not a strict standard violation.
    Warning,
    /// Standard violation or type error.
    Error,
}

/// An automated fix suggested by a [`Diagnostic`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Fix {
    /// Rename every occurrence of a keyword, preserving value and comment.
    RenameKeyword { from: String, to: String },
    /// Remove all value cards with the given keyword.
    RemoveKeyword { keyword: String },
    /// Replace the string value of an existing keyword.
    SetStringValue { keyword: String, value: String },
}

/// A single header diagnostic produced by [`Header::validate`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Diagnostic {
    /// The keyword that triggered this diagnostic.
    pub keyword: String,
    /// Severity.
    pub level: Level,
    /// Human-readable description of the issue.
    pub message: String,
    /// Suggested automated fix, if one is available.
    pub fix: Option<Fix>,
}

impl Header {
    /// Validate the header against the FITS standard.
    ///
    /// Returns diagnostics and a new `Header`. When `fix` is `true`, every
    /// diagnostic that carries a [`Fix`] is applied to the returned header;
    /// when `false` the returned header is an unmodified clone of `self`.
    #[must_use]
    pub fn validate(&self, fix: bool) -> (Vec<Diagnostic>, Self) {
        let mut diags = Vec::new();
        // Structural / ordering checks (FITS Standard Sec.4.4.1).
        check_mandatory_order(self, &mut diags);
        check_naxisn_consistency(self, &mut diags);
        check_blank_float_bitpix(self, &mut diags);
        check_pcount_gcount(self, &mut diags);
        // Deprecated keywords and semantic checks.
        check_deprecated_keyword(self, &mut diags, "EPOCH", "EQUINOX", "WCS Paper I, Sec.2.4");
        check_deprecated_keyword(
            self,
            &mut diags,
            "RADECSYS",
            "RADESYS",
            "WCS Paper I, Sec.3.1",
        );
        check_date_obs(self, &mut diags);
        check_exptime_type(self, &mut diags);
        check_crota(self, &mut diags);
        // Content recommendations.
        check_recommended_obs_keywords(self, &mut diags);

        let mut header = self.clone();
        if fix {
            for diag in &diags {
                if let Some(f) = &diag.fix {
                    apply_fix(&mut header, f);
                }
            }
        }
        (diags, header)
    }
}

/// Apply a single fix to `header` in place.
fn apply_fix(header: &mut Header, fix: &Fix) {
    match fix {
        Fix::RenameKeyword { from, to } => {
            for entry in header.cards_mut() {
                if entry.keyword == *from {
                    entry.keyword.clone_from(to);
                }
            }
            header.rebuild_index();
        }
        Fix::RemoveKeyword { keyword } => {
            header.remove(keyword);
        }
        Fix::SetStringValue { keyword, value } => {
            if let Some(entry) = header.first_value_entry_mut(keyword) {
                entry.value = Some(Value::String(value.clone()));
            }
        }
    }
}

/// Shared logic for deprecated-keyword pairs: if only `old` is present, suggest
/// renaming it to `new`; if both are present, suggest removing `old`.
fn check_deprecated_keyword(
    hdr: &Header,
    diags: &mut Vec<Diagnostic>,
    old: &str,
    new: &str,
    citation: &str,
) {
    if hdr.first(old).is_none() {
        return;
    }
    let (message, fix) = if hdr.first(new).is_some() {
        (
            format!("both {old} and {new} are present; {new} takes precedence  -  remove {old}"),
            Fix::RemoveKeyword {
                keyword: old.into(),
            },
        )
    } else {
        (
            format!("{old} is deprecated; use {new} ({citation})"),
            Fix::RenameKeyword {
                from: old.into(),
                to: new.into(),
            },
        )
    };
    diags.push(Diagnostic {
        keyword: old.into(),
        level: Level::Warning,
        message,
        fix: Some(fix),
    });
}

/// `DATE-OBS` in the pre-standard `DD/MM/YY` format must be converted to
/// ISO 8601 (FITS Standard Sec.4.4.2.1, originally NOST 100-2.0).
fn check_date_obs(hdr: &Header, diags: &mut Vec<Diagnostic>) {
    let Some(s) = hdr.optional_string("DATE-OBS") else {
        return;
    };
    let b = s.as_bytes();
    // Old format is exactly "DD/MM/YY" (8 bytes, '/' at [2] and [5]).
    if b.len() != 8
        || b[2] != b'/'
        || b[5] != b'/'
        || !b[0..2].iter().all(u8::is_ascii_digit)
        || !b[3..5].iter().all(u8::is_ascii_digit)
        || !b[6..8].iter().all(u8::is_ascii_digit)
    {
        return;
    }
    let day: u32 = s[0..2].parse().unwrap_or(0);
    let month: u32 = s[3..5].parse().unwrap_or(0);
    let yy: u32 = s[6..8].parse().unwrap_or(0);
    // Per NOST 100-2.0 the old format covered 1900-1999. Use pivot 70 to
    // handle post-2000 writers that erroneously continued to emit DD/MM/YY.
    let year = if yy >= 70 { 1900 + yy } else { 2000 + yy };
    let iso = format!("{year:04}-{month:02}-{day:02}");
    diags.push(Diagnostic {
        keyword: "DATE-OBS".into(),
        level: Level::Warning,
        message: format!(
            "DATE-OBS '{s}' uses the deprecated DD/MM/YY format; \
             ISO 8601 YYYY-MM-DD is required (FITS Standard Sec.4.4.2.1)"
        ),
        fix: Some(Fix::SetStringValue {
            keyword: "DATE-OBS".into(),
            value: iso,
        }),
    });
}

/// `EXPTIME` must be a numeric value (seconds); a string value is a type error.
fn check_exptime_type(hdr: &Header, diags: &mut Vec<Diagnostic>) {
    if let Some(s) = hdr.optional_string("EXPTIME") {
        diags.push(Diagnostic {
            keyword: "EXPTIME".into(),
            level: Level::Error,
            message: format!("EXPTIME has string value '{s}'; expected a numeric value (seconds)"),
            fix: None,
        });
    }
}

/// `CROTAn` is a deprecated WCS rotation keyword (WCS Paper I, Sec.6.1).
/// The canonical representation uses the `PCi_j` matrix.
fn check_crota(hdr: &Header, diags: &mut Vec<Diagnostic>) {
    for entry in hdr.entries() {
        if let Some(suffix) = entry.keyword.strip_prefix("CROTA")
            && !suffix.is_empty()
            && suffix.bytes().all(|b| b.is_ascii_digit())
        {
            let keyword = entry.keyword.clone();
            let message = format!(
                "{keyword} is a deprecated rotation keyword; \
                 migrate to the PCi_j matrix (WCS Paper I, Sec.6.1)"
            );
            diags.push(Diagnostic {
                keyword,
                level: Level::Warning,
                message,
                fix: None,
            });
        }
    }
}

/// First three value keywords must be SIMPLE/XTENSION, BITPIX, NAXIS in that
/// order (FITS Standard Sec.4.4.1.1). Commentary cards may appear anywhere.
fn check_mandatory_order(hdr: &Header, diags: &mut Vec<Diagnostic>) {
    let mut value_kws = hdr
        .entries()
        .iter()
        .filter(|e| e.value.is_some())
        .map(|e| e.keyword.as_str());
    let pos1 = value_kws.next();
    let pos2 = value_kws.next();
    let pos3 = value_kws.next();

    if !matches!(pos1, Some("SIMPLE" | "XTENSION") | None) {
        diags.push(Diagnostic {
            keyword: pos1.unwrap_or("").into(),
            level: Level::Error,
            message: format!(
                "first value keyword must be SIMPLE or XTENSION, found '{}' \
                 (FITS Standard Sec.4.4.1.1)",
                pos1.unwrap_or("<none>")
            ),
            fix: None,
        });
    }
    if !matches!(pos2, Some("BITPIX") | None) {
        diags.push(Diagnostic {
            keyword: "BITPIX".into(),
            level: Level::Error,
            message: format!(
                "second value keyword must be BITPIX, found '{}' \
                 (FITS Standard Sec.4.4.1.1)",
                pos2.unwrap_or("<none>")
            ),
            fix: None,
        });
    }
    if !matches!(pos3, Some("NAXIS") | None) {
        diags.push(Diagnostic {
            keyword: "NAXIS".into(),
            level: Level::Error,
            message: format!(
                "third value keyword must be NAXIS, found '{}' \
                 (FITS Standard Sec.4.4.1.1)",
                pos3.unwrap_or("<none>")
            ),
            fix: None,
        });
    }
}

/// If `NAXIS=n`, all of `NAXIS1`...`NAXISn` must be present (Sec.4.4.1.1).
fn check_naxisn_consistency(hdr: &Header, diags: &mut Vec<Diagnostic>) {
    let Ok(n) = hdr.naxis() else { return };
    for axis in 1..=n {
        let kw = format!("NAXIS{axis}");
        if hdr.first(&kw).is_none() {
            let message = format!("{kw} is absent but NAXIS={n}");
            diags.push(Diagnostic {
                keyword: kw,
                level: Level::Error,
                message,
                fix: None,
            });
        }
    }
}

/// `BLANK` is undefined for floating-point images; IEEE NaN fills that role
/// (FITS Standard Sec.4.4.2.4).
fn check_blank_float_bitpix(hdr: &Header, diags: &mut Vec<Diagnostic>) {
    if hdr.first("BLANK").is_none() {
        return;
    }
    let Ok(bitpix) = hdr.bitpix() else { return };
    if bitpix < 0 {
        diags.push(Diagnostic {
            keyword: "BLANK".into(),
            level: Level::Warning,
            message: format!(
                "BLANK is undefined for floating-point images (BITPIX={bitpix}); \
                 use IEEE NaN for undefined pixels (FITS Standard Sec.4.4.2.4)"
            ),
            fix: Some(Fix::RemoveKeyword {
                keyword: "BLANK".into(),
            }),
        });
    }
}

/// Check `PCOUNT` and `GCOUNT` for standard extension HDUs (Sec.7.1).
///
/// All standard extensions require `GCOUNT=1`. `IMAGE` extensions additionally
/// require `PCOUNT=0`. `BINTABLE`/`TABLE` allow non-zero `PCOUNT` (heap data).
fn check_pcount_gcount(hdr: &Header, diags: &mut Vec<Diagnostic>) {
    let Some(xtension) = hdr.optional_string("XTENSION") else {
        return;
    };
    let xtension = xtension.trim();
    if !matches!(xtension, "IMAGE" | "BINTABLE" | "TABLE") {
        return;
    }
    check_required_int_value(hdr, diags, "GCOUNT", 1, xtension);
    if xtension == "IMAGE" {
        check_required_int_value(hdr, diags, "PCOUNT", 0, xtension);
    }
}

fn check_required_int_value(
    hdr: &Header,
    diags: &mut Vec<Diagnostic>,
    kw: &str,
    expected: i64,
    context: &str,
) {
    match hdr.optional_int(kw) {
        Some(v) if v == expected => {}
        Some(v) => diags.push(Diagnostic {
            keyword: kw.into(),
            level: Level::Error,
            message: format!(
                "{kw} must be {expected} for {context} extensions, found {v} \
                 (FITS Standard Sec.7.1)"
            ),
            fix: None,
        }),
        None => diags.push(Diagnostic {
            keyword: kw.into(),
            level: Level::Error,
            message: format!(
                "{kw} is absent; must be {expected} for {context} extensions \
                 (FITS Standard Sec.7.1)"
            ),
            fix: None,
        }),
    }
}

/// Warn about absent commonly-expected observation keywords in image HDUs.
fn check_recommended_obs_keywords(hdr: &Header, diags: &mut Vec<Diagnostic>) {
    // Only applies to image HDUs with actual data (NAXIS > 0).
    let Ok(n) = hdr.naxis() else { return };
    if n == 0 {
        return;
    }
    for kw in ["OBJECT", "TELESCOP", "INSTRUME", "DATE-OBS", "EXPTIME"] {
        if hdr.first(kw).is_none() {
            diags.push(Diagnostic {
                keyword: kw.into(),
                level: Level::Warning,
                message: format!("{kw} is absent; recommended for image HDUs"),
                fix: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::card::CARD_SIZE;
    use crate::io::block::BLOCK_SIZE;

    fn hdr(cards: &[&str]) -> Header {
        let mut bytes: Vec<u8> = Vec::new();
        for c in cards {
            let mut card = c.as_bytes().to_vec();
            card.resize(CARD_SIZE, b' ');
            bytes.extend_from_slice(&card);
        }
        let mut end = b"END".to_vec();
        end.resize(CARD_SIZE, b' ');
        bytes.extend_from_slice(&end);
        while !bytes.len().is_multiple_of(BLOCK_SIZE) {
            bytes.push(b' ');
        }
        Header::parse(&bytes, 0).expect("parse").0
    }

    #[test]
    fn epoch_without_equinox_suggests_rename() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EPOCH   =              2000.0",
        ]);
        let (diags, _) = h.validate(false);
        let d = diags.iter().find(|d| d.keyword == "EPOCH").unwrap();
        assert_eq!(d.level, Level::Warning);
        assert!(
            matches!(&d.fix, Some(Fix::RenameKeyword { from, to }) if from == "EPOCH" && to == "EQUINOX")
        );
    }

    #[test]
    fn both_epoch_and_equinox_suggests_remove() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EPOCH   =              2000.0",
            "EQUINOX =              2000.0",
        ]);
        let (diags, _) = h.validate(false);
        let d = diags.iter().find(|d| d.keyword == "EPOCH").unwrap();
        assert!(matches!(&d.fix, Some(Fix::RemoveKeyword { keyword }) if keyword == "EPOCH"));
    }

    #[test]
    fn validate_fix_renames_epoch_to_equinox() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EPOCH   =              2000.0",
        ]);
        let (_, fixed) = h.validate(true);
        assert!(!fixed.contains("EPOCH"));
        assert!(fixed.contains("EQUINOX"));
    }

    #[test]
    fn radecsys_renamed_to_radesys() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "RADECSYS= 'FK5     '",
        ]);
        let (diags, _) = h.validate(false);
        assert!(diags.iter().any(|d| d.keyword == "RADECSYS"
            && matches!(&d.fix, Some(Fix::RenameKeyword { from, to }) if from == "RADECSYS" && to == "RADESYS")));
    }

    #[test]
    fn old_date_obs_format_detected_and_fixed() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "DATE-OBS= '15/06/98'",
        ]);
        let (diags, _) = h.validate(false);
        let d = diags.iter().find(|d| d.keyword == "DATE-OBS").unwrap();
        assert_eq!(d.level, Level::Warning);
        // DD=15, MM=06, YY=98 -> 1998-06-15
        assert!(matches!(&d.fix,
            Some(Fix::SetStringValue { keyword, value })
            if keyword == "DATE-OBS" && value == "1998-06-15"
        ));
    }

    #[test]
    fn validate_fix_converts_date_obs_format() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "DATE-OBS= '15/06/98'",
        ]);
        let (_, fixed) = h.validate(true);
        assert_eq!(fixed.optional_string("DATE-OBS"), Some("1998-06-15"));
    }

    #[test]
    fn modern_date_obs_is_clean() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "DATE-OBS= '2003-08-22T10:30:00'",
        ]);
        let (diags, _) = h.validate(false);
        assert!(!diags.iter().any(|d| d.keyword == "DATE-OBS"));
    }

    #[test]
    fn exptime_as_string_is_error() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EXPTIME = '300s    '",
        ]);
        let (diags, _) = h.validate(false);
        let d = diags.iter().find(|d| d.keyword == "EXPTIME").unwrap();
        assert_eq!(d.level, Level::Error);
        assert!(d.fix.is_none());
    }

    #[test]
    fn crota_deprecated() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                  100",
            "NAXIS2  =                  100",
            "CROTA2  =                  0.0",
        ]);
        let (diags, _) = h.validate(false);
        assert!(
            diags
                .iter()
                .any(|d| d.keyword == "CROTA2" && d.level == Level::Warning)
        );
    }

    #[test]
    fn missing_obs_keywords_warned_for_image() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                  100",
            "NAXIS2  =                  100",
        ]);
        let (diags, _) = h.validate(false);
        let missing: Vec<&str> = diags.iter().map(|d| d.keyword.as_str()).collect();
        assert!(missing.contains(&"OBJECT"));
        assert!(missing.contains(&"TELESCOP"));
        assert!(missing.contains(&"INSTRUME"));
        assert!(missing.contains(&"DATE-OBS"));
        assert!(missing.contains(&"EXPTIME"));
    }

    #[test]
    fn no_missing_keywords_warning_for_empty_image() {
        // NAXIS=0 means no data; recommended keywords don't apply.
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
        ]);
        let (diags, _) = h.validate(false);
        assert!(!diags.iter().any(|d| d.keyword == "OBJECT"));
    }

    // -- check_mandatory_order ------------------------------------------------

    #[test]
    fn mandatory_order_correct_is_clean() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
        ]);
        let (diags, _) = h.validate(false);
        assert!(!diags.iter().any(|d| {
            d.message.contains("first value keyword")
                || d.message.contains("second value keyword")
                || d.message.contains("third value keyword")
        }));
    }

    #[test]
    fn mandatory_order_wrong_first_card() {
        // BITPIX before SIMPLE  -  first position violated.
        let h = hdr(&[
            "BITPIX  =                    8",
            "SIMPLE  =                    T",
            "NAXIS   =                    0",
        ]);
        let (diags, _) = h.validate(false);
        assert!(
            diags
                .iter()
                .any(|d| d.level == Level::Error && d.message.contains("first value keyword"))
        );
    }

    #[test]
    fn mandatory_order_naxis_before_bitpix() {
        // NAXIS in position 2 means BITPIX is displaced.
        let h = hdr(&[
            "SIMPLE  =                    T",
            "NAXIS   =                    0",
            "BITPIX  =                    8",
        ]);
        let (diags, _) = h.validate(false);
        assert!(
            diags
                .iter()
                .any(|d| d.keyword == "BITPIX" && d.message.contains("second value keyword"))
        );
    }

    // -- check_naxisn_consistency ---------------------------------------------

    #[test]
    fn naxisn_all_present_is_clean() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                  512",
            "NAXIS2  =                  256",
        ]);
        let (diags, _) = h.validate(false);
        assert!(
            !diags
                .iter()
                .any(|d| d.keyword.starts_with("NAXIS") && d.level == Level::Error)
        );
    }

    #[test]
    fn naxisn_missing_is_error() {
        // NAXIS=2 but only NAXIS1 present.
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                  512",
        ]);
        let (diags, _) = h.validate(false);
        assert!(
            diags
                .iter()
                .any(|d| d.keyword == "NAXIS2" && d.level == Level::Error)
        );
    }

    // -- check_blank_float_bitpix ---------------------------------------------

    #[test]
    fn blank_with_integer_bitpix_is_clean() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    0",
            "BLANK   =               -32768",
        ]);
        let (diags, _) = h.validate(false);
        assert!(!diags.iter().any(|d| d.keyword == "BLANK"));
    }

    #[test]
    fn blank_with_float_bitpix_warns_and_offers_remove() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    0",
            "BLANK   =                   -1",
        ]);
        let (diags, _) = h.validate(false);
        let d = diags.iter().find(|d| d.keyword == "BLANK").unwrap();
        assert_eq!(d.level, Level::Warning);
        assert!(matches!(&d.fix, Some(Fix::RemoveKeyword { keyword }) if keyword == "BLANK"));
    }

    #[test]
    fn validate_fix_removes_blank_for_float_bitpix() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    0",
            "BLANK   =                   -1",
        ]);
        let (_, fixed) = h.validate(true);
        assert!(!fixed.contains("BLANK"));
    }

    // -- check_pcount_gcount --------------------------------------------------

    #[test]
    fn image_extension_correct_pcount_gcount_is_clean() {
        let h = hdr(&[
            "XTENSION= 'IMAGE   '",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "PCOUNT  =                    0",
            "GCOUNT  =                    1",
        ]);
        let (diags, _) = h.validate(false);
        assert!(
            !diags
                .iter()
                .any(|d| matches!(d.keyword.as_str(), "PCOUNT" | "GCOUNT"))
        );
    }

    #[test]
    fn image_extension_wrong_pcount_is_error() {
        let h = hdr(&[
            "XTENSION= 'IMAGE   '",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "PCOUNT  =                    4",
            "GCOUNT  =                    1",
        ]);
        let (diags, _) = h.validate(false);
        assert!(
            diags
                .iter()
                .any(|d| d.keyword == "PCOUNT" && d.level == Level::Error)
        );
    }

    #[test]
    fn image_extension_wrong_gcount_is_error() {
        let h = hdr(&[
            "XTENSION= 'IMAGE   '",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "PCOUNT  =                    0",
            "GCOUNT  =                    2",
        ]);
        let (diags, _) = h.validate(false);
        assert!(
            diags
                .iter()
                .any(|d| d.keyword == "GCOUNT" && d.level == Level::Error)
        );
    }

    #[test]
    fn image_extension_missing_gcount_is_error() {
        let h = hdr(&[
            "XTENSION= 'IMAGE   '",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "PCOUNT  =                    0",
        ]);
        let (diags, _) = h.validate(false);
        assert!(
            diags
                .iter()
                .any(|d| d.keyword == "GCOUNT" && d.level == Level::Error)
        );
    }

    #[test]
    fn bintable_pcount_not_required_to_be_zero() {
        // BINTABLE with non-zero PCOUNT (heap) is valid; only GCOUNT is checked.
        let h = hdr(&[
            "XTENSION= 'BINTABLE'",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                   10",
            "NAXIS2  =                  100",
            "PCOUNT  =                 1024",
            "GCOUNT  =                    1",
        ]);
        let (diags, _) = h.validate(false);
        assert!(!diags.iter().any(|d| d.keyword == "PCOUNT"));
        assert!(!diags.iter().any(|d| d.keyword == "GCOUNT"));
    }

    #[test]
    fn primary_hdu_no_pcount_gcount_check() {
        // Primary HDU (SIMPLE=T) does not require PCOUNT/GCOUNT.
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
        ]);
        let (diags, _) = h.validate(false);
        assert!(
            !diags
                .iter()
                .any(|d| matches!(d.keyword.as_str(), "PCOUNT" | "GCOUNT"))
        );
    }
}
