//! Unit conversions to the canonical WCS units (Paper I Sec.3.1,
//! Standard Sec.8.1). Celestial axes are reported in *degrees*; the
//! parser converts non-degree CUNIT values (`arcsec`, `arcmin`,
//! `mas`, `rad`) to degrees so downstream code can assume one unit.
//!
//! Non-celestial axes pass through with no conversion (the linear
//! pipeline already preserves whatever unit the header declared).

/// Multiplier that converts a value in `cunit` to degrees. Unknown
/// or empty units fall through with factor 1.0 (treated as already
/// in the canonical unit). Recognized aliases follow the OGIP /
/// FITS unit conventions and are case-insensitive.
#[must_use]
pub(crate) fn to_degrees_factor(cunit: &str) -> f64 {
    let s = cunit.trim().to_ascii_lowercase();
    match s.as_str() {
        // Arcseconds and aliases.
        "arcsec" | "arcsecs" | "asec" | "\"" | "as" => 1.0 / 3600.0,
        // Milliarcseconds.
        "mas" => 1.0 / 3_600_000.0,
        // Microarcseconds.
        "uas" => 1.0 / 3_600_000_000.0,
        // Arcminutes.
        "arcmin" | "arcmins" | "amin" | "'" => 1.0 / 60.0,
        // Radians.
        "rad" | "radian" | "radians" => 180.0 / std::f64::consts::PI,
        // Degrees (including unspecified/empty) and any other recognized but
        // unity-factor unit fall through to the wildcard.
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::to_degrees_factor;

    #[test]
    fn known_units() {
        assert!((to_degrees_factor("deg") - 1.0).abs() < 1e-15);
        assert!((to_degrees_factor("arcsec") - 1.0 / 3600.0).abs() < 1e-15);
        assert!((to_degrees_factor("ARCMIN") - 1.0 / 60.0).abs() < 1e-15);
        assert!((to_degrees_factor("rad") - 180.0 / std::f64::consts::PI).abs() < 1e-12);
        assert!((to_degrees_factor("mas") - 1.0 / 3_600_000.0).abs() < 1e-18);
    }

    #[test]
    fn empty_or_unknown_passes_through() {
        assert_eq!(to_degrees_factor(""), 1.0);
        assert_eq!(to_degrees_factor("hz"), 1.0);
    }
}
