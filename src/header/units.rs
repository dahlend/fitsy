//! Units strings (Standard Sec.4.3) and basic SI conversion.
//!
//! Formal unit keywords (`BUNIT`, `CUNIT*`, `TUNIT*`) are stored verbatim.
//! This module also supports the widely-used `[unit]` comment convention,
//! where the unit appears in brackets anywhere in a card's inline comment.

use crate::header::Header;

// -- Comment-based unit extraction -------------------------------------------

/// Extract the first `[unit]` token from a FITS inline comment string.
///
/// Matches the first `[...]` pair regardless of position, returning the
/// content between the brackets.
#[must_use]
pub fn parse_comment_unit(comment: &str) -> Option<&str> {
    let start = comment.find('[')? + 1;
    let end = comment[start..].find(']')? + start;
    let unit = comment[start..end].trim();
    if unit.is_empty() { None } else { Some(unit) }
}

// -- SI conversion table -----------------------------------------------------

/// Multiplier that converts one `unit` into the SI base for its dimension.
///
/// Dimensions and their SI bases:
/// - Length -> meters (m)
/// - Angle  -> Degrees (deg)
/// - Time   -> seconds (s)
/// - Velocity -> meters per second (m/s)
/// - Frequency -> hertz (Hz)
/// - Flux density -> Jansky (Jy)
///
/// Returns `None` for unrecognized or compound units.
#[must_use]
#[allow(
    clippy::match_same_arms,
    reason = "matches are grouped for readability, not to avoid repetition"
)]
pub fn si_factor(unit: &str) -> Option<f64> {
    let s = unit.trim().to_ascii_lowercase();
    Some(match s.as_str() {
        // --- Length (SI base: m) ---
        "m" | "meter" | "meters" | "metre" | "metres" => 1.0,
        "km" | "kilometer" | "kilometers" | "kilometre" | "kilometres" => 1e3,
        "cm" | "centimeter" | "centimeters" | "centimetre" | "centimetres" => 1e-2,
        "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => 1e-3,
        "um" | "micrometer" | "micrometers" | "micrometre" | "micrometres" => 1e-6,
        "nm" | "nanometer" | "nanometers" | "nanometre" | "nanometres" => 1e-9,
        "au" => 1.495_978_707e11, // IAU 2012

        // --- Angle (Deg) ---
        "rad" | "radian" | "radians" => 180.0 / std::f64::consts::PI,
        "deg" | "degree" | "degrees" => 1.0,
        "arcmin" | "arcmins" | "amin" | "'" => 1.0 / 60.0,
        "arcsec" | "arcsecs" | "asec" | "\"" | "as" => 1.0 / 3600.0,
        "mas" => 1.0 / 3_600_000.0,
        "uas" => 1.0 / 3_600_000_000.0,

        // --- Time (SI base: s) ---
        "s" | "sec" | "second" | "seconds" => 1.0,
        "min" | "minute" | "minutes" => 60.0,
        "h" | "hr" | "hour" | "hours" => 3_600.0,
        "d" | "day" | "days" => 86_400.0,
        "yr" | "a" | "year" | "years" => 31_557_600.0, // Julian year (365.25 d)

        // --- Velocity (SI base: m/s) ---
        "m/s" => 1.0,
        "km/s" => 1e3,
        "cm/s" | "cm/sec" => 1e-2,
        "km/h" => 1_000.0 / 3_600.0,
        "au/d" | "au/day" => 1.495_978_707e11 / 86_400.0,

        // --- Frequency (SI base: Hz) ---
        "hz" => 1.0,
        "khz" => 1e3,
        "mhz" => 1e6,
        "ghz" => 1e9,
        "thz" => 1e12,

        // --- Flux density (Base: 1 Jy) ---
        "jy" => 1.0,
        "mjy" => 1e-3,
        "ujy" => 1e-6,

        _ => return None,
    })
}

// -- Header accessors --------------------------------------------------------

impl Header {
    /// Unit string for `key`, taken from the keyword's inline comment via
    /// the `[unit]` convention. Returns `None` if no unit is found.
    #[must_use]
    pub fn keyword_unit(&self, key: &str) -> Option<String> {
        let k = key.trim().to_ascii_uppercase();
        let entry = self.entries().iter().find(|e| e.keyword == k).or_else(|| {
            let alt = Self::alt_key(&k)?;
            self.entries().iter().find(|e| e.keyword == alt)
        })?;
        let comment = entry.comment.as_deref()?;
        parse_comment_unit(comment).map(str::to_owned)
    }

    /// Value of `key` expressed in `target_unit`.
    ///
    /// Uses the `[unit]` comment annotation as the source unit when present;
    /// otherwise assumes the stored value is already in `target_unit` and
    /// returns it unchanged. Returns `None` if the keyword is absent,
    /// non-numeric, or either the annotation or `target_unit` is unrecognized.
    #[must_use]
    pub fn real_in_unit(&self, key: &str, target_unit: &str) -> Option<f64> {
        let v = self.optional_real(key)?;
        let source = self.keyword_unit(key);
        let source = source.as_deref().unwrap_or(target_unit);
        let src = si_factor(source)?;
        let tgt = si_factor(target_unit)?;
        Some(v * src / tgt)
    }
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_unit_at_start() {
        assert_eq!(parse_comment_unit("[AU] Distance to target"), Some("AU"));
    }

    #[test]
    fn parse_unit_mid_comment() {
        assert_eq!(parse_comment_unit("exposure time [s]"), Some("s"));
    }

    #[test]
    fn parse_unit_none_when_absent() {
        assert!(parse_comment_unit("no brackets here").is_none());
        assert!(parse_comment_unit("[] empty brackets").is_none());
    }

    #[test]
    fn si_factor_length() {
        assert_eq!(si_factor("m").unwrap(), 1.0);
        assert!((si_factor("km").unwrap() - 1e3).abs() < 1e-6);
        assert!((si_factor("AU").unwrap() - 1.495_978_707e11).abs() < 1e3);
    }

    #[test]
    fn si_factor_angle() {
        assert_eq!(si_factor("deg").unwrap(), 1.0);
        assert!((si_factor("rad").unwrap() - 180.0 / std::f64::consts::PI).abs() < 1e-12);
        assert!((si_factor("arcsec").unwrap() - 1.0 / 3600.0).abs() < 1e-15);
    }

    #[test]
    fn si_factor_velocity() {
        assert_eq!(si_factor("m/s").unwrap(), 1.0);
        assert!((si_factor("km/s").unwrap() - 1e3).abs() < 1e-6);
        assert!((si_factor("km/h").unwrap() - 1000.0 / 3600.0).abs() < 1e-10);
        let au_day = 1.495_978_707e11 / 86_400.0;
        assert!((si_factor("AU/day").unwrap() - au_day).abs() < 1.0);
    }

    #[test]
    fn si_factor_unknown_returns_none() {
        assert!(si_factor("parsec").is_none());
        assert!(si_factor("erg/cm2/s").is_none());
    }

    #[test]
    fn real_in_unit_converts_annotation_to_target() {
        // Annotated [AU], asked for metres -> convert AU -> m.
        let mut h = Header::empty();
        h.push("DIST", 1.0_f64, Some("[AU] Distance")).unwrap();
        let m = h.real_in_unit("DIST", "m").unwrap();
        assert!((m - 1.495_978_707e11).abs() < 1e3);
    }

    #[test]
    fn real_in_unit_matching_units_passes_through() {
        // Annotated [AU], asked for AU -> return raw.
        let mut h = Header::empty();
        h.push("DIST", 1.5_f64, Some("[AU] Distance")).unwrap();
        assert!((h.real_in_unit("DIST", "AU").unwrap() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn real_in_unit_no_annotation_assumes_target() {
        // No annotation -> assume value already in target unit.
        let mut h = Header::empty();
        h.push("ALT", 500.0_f64, None).unwrap();
        assert_eq!(h.real_in_unit("ALT", "m"), Some(500.0));
    }

    #[test]
    fn header_keyword_unit_missing() {
        let mut h = Header::empty();
        h.push("FOO", 1.0_f64, Some("no unit here")).unwrap();
        assert!(h.keyword_unit("FOO").is_none());
    }

    #[test]
    fn underscore_and_hyphen_are_interchangeable() {
        let mut h = Header::empty();
        h.push("MJD-OBS", 57754.0_f64, None).unwrap();
        assert!(h.optional_real("MJD_OBS").is_some());

        let mut h2 = Header::empty();
        h2.push("MJD_OBS", 57754.0_f64, None).unwrap();
        assert!(h2.optional_real("MJD-OBS").is_some());
    }
}
