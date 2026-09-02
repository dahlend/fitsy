//! `Header` accessors for keyword units.
//!
//! [`crate::units`] holds the unit strings themselves: the Sec.4.3
//! syntax, the base tables, and the `[unit]` comment convention. This
//! module holds only the `Header` methods that apply them. It sits
//! here with the other `impl Header` blocks grouped by topic.

use crate::header::Header;
use crate::units::{Unit, parse_comment_unit, parse_unit_lenient};

impl Header {
    /// Unit string for `key`, taken from the keyword's inline comment
    /// via the Sec.4.3.2 `[unit]` convention. `None` if there is no
    /// such card or its comment carries no annotation.
    #[must_use]
    pub fn keyword_unit(&self, key: &str) -> Option<String> {
        let k = key.trim().to_ascii_uppercase();
        let card = self.first_card(&k)?;
        let comment = card.comment()?;
        parse_comment_unit(&comment).map(str::to_owned)
    }

    /// Value of `key` expressed in `target_unit`.
    ///
    /// The `[unit]` comment annotation gives the source unit; without
    /// one the value is taken to be in `target_unit` already. Both are
    /// read leniently, so `[degrees]` and `[sec]` resolve alongside the
    /// strict Sec.4.3 spellings.
    ///
    /// The result is `None` in three cases:
    ///
    /// - The keyword is absent, or its value is not numeric.
    /// - Either unit string fails to parse.
    /// - The two units carry different dimensions. A value annotated
    ///   `[s]` cannot be reported in meters.
    #[must_use]
    pub fn real_in_unit(&self, key: &str, target_unit: &str) -> Option<f64> {
        let v = self.optional_real(key)?;
        let source = self.keyword_unit(key);
        let source = source.as_deref().unwrap_or(target_unit);
        let src = parse_unit_lenient(source).ok()?;
        let tgt = parse_unit_lenient(target_unit).ok()?;
        // The converter is affine, so a magnitude annotation converts by
        // its shift rather than being silently multiplied.
        Some(src.converter_to(tgt).ok()?.apply(v))
    }

    /// Value of `key` converted to the canonical unit for its dimension
    /// (meter, kilogram, second, degree, ...).
    ///
    /// The source unit comes from the `[unit]` comment annotation,
    /// read leniently.
    ///
    /// An unannotated keyword gives `None` here, unlike in
    /// [`Self::real_in_unit`]. With no unit on record there is nothing
    /// to convert from, and reporting the raw number as canonical
    /// would be a guess.
    ///
    /// The result is `None` in four cases:
    ///
    /// - The keyword is absent, or its value is not numeric.
    /// - The keyword carries no `[unit]` annotation.
    /// - The annotation fails to parse.
    /// - The unit is a bare level such as `mag`, with no linear
    ///   reading.
    #[must_use]
    pub fn real_in_canonical(&self, key: &str) -> Option<f64> {
        let v = self.optional_real(key)?;
        let src = parse_unit_lenient(&self.keyword_unit(key)?).ok()?;
        let canonical = Unit::new(1.0, src.dimension);
        Some(src.converter_to(canonical).ok()?.apply(v))
    }
}

#[cfg(test)]
mod tests {
    use super::Header;

    #[test]
    fn real_in_unit_converts_annotation_to_target() {
        // Annotated [AU], asked for meters -> convert AU -> m.
        let mut h = Header::empty();
        h.push("DIST", 1.0_f64, Some("[AU] Distance")).unwrap();
        let m = h.real_in_unit("DIST", "m").unwrap();
        assert!((m - 1.495_978_707e11).abs() < 1e3);
    }

    #[test]
    fn real_in_unit_matching_units_passes_through() {
        let mut h = Header::empty();
        h.push("DIST", 1.5_f64, Some("[AU] Distance")).unwrap();
        assert!((h.real_in_unit("DIST", "AU").unwrap() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn real_in_unit_no_annotation_assumes_target() {
        let mut h = Header::empty();
        h.push("ALT", 500.0_f64, None).unwrap();
        assert_eq!(h.real_in_unit("ALT", "m"), Some(500.0));
    }

    /// Converting between different quantities is not a scaling
    /// problem, it is a broken header. The old lookup returned a
    /// number here.
    #[test]
    fn real_in_unit_refuses_a_dimension_mismatch() {
        let mut h = Header::empty();
        h.push("DIST", 1.0_f64, Some("[s] mislabeled")).unwrap();
        assert!(h.real_in_unit("DIST", "m").is_none());
        // ... and an unparseable unit on either side.
        let mut h = Header::empty();
        h.push("DIST", 1.0_f64, Some("[furlong] nope")).unwrap();
        assert!(h.real_in_unit("DIST", "m").is_none());
    }

    /// Annotations are written by people, not by the Sec.4.3 grammar.
    ///
    /// Regression: moving to the strict parser silently dropped these
    /// spellings, and `obs_geodetic` lost the whole observatory
    /// location to a `[degrees]`.
    #[test]
    fn real_in_unit_accepts_informal_annotation_spellings() {
        let mut h = Header::empty();
        h.push("PA", 2.0_f64, Some("[degrees] position angle"))
            .unwrap();
        assert!((h.real_in_unit("PA", "deg").unwrap() - 2.0).abs() < 1e-12);
        let mut h = Header::empty();
        h.push("RATE", 1.0_f64, Some("[AU/day] sky motion"))
            .unwrap();
        let want = 1.495_978_707e11 / 86_400.0;
        assert!((h.real_in_unit("RATE", "m/s").unwrap() - want).abs() < 1.0);
    }

    /// A surface brightness rescales *additively*: 1 deg^2 is 3600^2
    /// arcsec^2, so the same sky is 2.5 log10(3600^2) = 17.78 magnitudes
    /// brighter per square degree.
    ///
    /// Regression: this multiplied by 1.296e7, turning 20 mag/arcsec2
    /// into 2.592e8 mag/deg2.
    #[test]
    fn real_in_unit_shifts_a_magnitude_rather_than_scaling_it() {
        let mut h = Header::empty();
        h.push("SKYMAG", 20.0_f64, Some("[mag/arcsec2] sky brightness"))
            .unwrap();
        let want = 20.0 - 2.5 * (3600.0_f64 * 3600.0).log10();
        let got = h.real_in_unit("SKYMAG", "mag/deg2").unwrap();
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        // Against itself it is the identity, not a factor of 1.296e7.
        assert!((h.real_in_unit("SKYMAG", "mag/arcsec2").unwrap() - 20.0).abs() < 1e-12);
        // A magnitude still refuses a linear unit.
        assert!(h.real_in_unit("SKYMAG", "Jy").is_none());
    }

    #[test]
    fn header_keyword_unit_missing() {
        let mut h = Header::empty();
        h.push("FOO", 1.0_f64, Some("no unit here")).unwrap();
        assert!(h.keyword_unit("FOO").is_none());
    }

    /// The Standard spells these with `-`, so the standard name also
    /// finds the `_` misspelling. See `Header::alt_key` for why the
    /// reverse is not allowed.
    #[test]
    fn hyphenated_keyword_also_matches_the_underscore_misspelling() {
        let mut h = Header::empty();
        h.push("MJD_OBS", 57754.0_f64, None).unwrap();
        assert!(h.optional_real("MJD-OBS").is_some());
        assert!(h.contains("MJD-OBS"));
    }

    #[test]
    fn underscore_lookup_does_not_reach_a_hyphenated_card() {
        let mut h = Header::empty();
        h.push("CD1-1", 1.0_f64, None).unwrap();
        assert!(h.optional_real("CD1_1").is_none());
        assert!(!h.contains("CD1_1"));
    }

    /// `real_in_canonical` converts an annotated value to the
    /// canonical unit for its dimension, and refuses to guess when
    /// there is no annotation to convert from -- passing the raw
    /// number off as canonical is exactly the silent wrong answer the
    /// method exists to avoid.
    #[test]
    fn real_in_canonical_requires_an_annotation() {
        let mut h = Header::empty();
        h.push("DIST", 1.5_f64, Some("[km] distance")).unwrap();
        assert!((h.real_in_canonical("DIST").unwrap() - 1500.0).abs() < 1e-9);

        let mut plain = Header::empty();
        plain.push("PLAIN", 42.0_f64, None).unwrap();
        assert_eq!(plain.real_in_canonical("PLAIN"), None);
    }
}
