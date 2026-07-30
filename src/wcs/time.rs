//! Time coordinate axes (Standard Sec.9).
//!
//! Sec.9.5.3 defines an image time axis as the ordinary *linear*
//! transform: `CRVALia` holds elapsed time in `TIMEUNIT` or `CUNITia`,
//! `CDELTia` the interval, and the `PCi_j` diagonal element "would take
//! the exact value 1, the default". So there is no algorithm here, only
//! recognition -- the pipeline in [`crate::Wcs`] transforms a time axis
//! exactly as it does any other linear one, which is what `wcslib` does
//! too.
//!
//! What this module adds is the *identity* of the axis: which one is
//! time, and on which scale. Without it the elapsed value has no
//! meaning, since the zero point lives in `MJDREF`/`JDREF`/`DATEREF`
//! and the scale in `TIMESYS`.
//!
//! # Absolute times
//!
//! [`TimeAxis`] deliberately reports elapsed time, not an epoch, since
//! that is what the axis stores and what `wcslib` returns. Combine it
//! with [`Header::mjd_ref`](crate::Header::mjd_ref) -- which already
//! folds in `TIMEOFFS` -- to get an MJD.

use crate::header::time::base_time_scale;

/// Time scales of Standard Sec.9.2.1 Table 29.
///
/// Recognized as a `CTYPE` value as well as a `TIMESYS` one: Sec.9.2.1
/// lets an axis name its own scale, overriding the global keyword.
const TIME_SCALES: &[&str] = &[
    "TAI", "TT", "TDT", "ET", "IAT", "UT1", "UTC", "GMT", "GPS", "TCG", "TCB", "TDB", "LOCAL",
];

/// True if `code` names a Table 29 time scale, ignoring any
/// parenthesised realization (`TT(TAI)`, `UTC(NIST)`).
#[must_use]
pub fn is_time_scale(code: &str) -> bool {
    let base = base_time_scale(code.trim()).to_ascii_uppercase();
    TIME_SCALES.contains(&base.as_str())
        // Sec.9.2.1 Table 29 also lists `UT()` with a qualifier, for
        // radio time signals between 1955 and 1972.
        || (base == "UT" && code.contains('('))
}

/// A recognized time axis (Standard Sec.9.5.3).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TimeAxis {
    /// Zero-based axis index.
    pub axis: usize,
    /// Time scale governing this axis, upper-cased, without any
    /// realization: `TT`, `UTC`, `TDB`, ...
    ///
    /// Resolved per Sec.9.2.1: a `CTYPE` naming a Table 29 scale wins;
    /// `CTYPE = 'TIME'` defers to `TIMESYS`, which itself defaults to
    /// `UTC`.
    pub scale: String,
    /// The realization in parentheses, if the header gave one --
    /// `TAI` from `TT(TAI)`, `NIST` from `UTC(NIST)`. Records
    /// provenance; it does not change the reduction.
    pub realization: Option<String>,
    /// `TREFPOS` (Sec.9.2.3) -- where the time is valid
    /// (`TOPOCENTER`, `BARYCENTER`, ...). Stored; not applied -- no
    /// pathlength correction is performed.
    pub trefpos: Option<String>,
    /// `TREFDIR` (Sec.9.2.4) -- the keywords holding the direction
    /// used for the pathlength correction. Stored; not applied.
    pub trefdir: Option<String>,
    /// `PLEPHEM` (Sec.9.2.5) -- the solar-system ephemeris used for
    /// that correction. Stored; not applied.
    pub plephem: Option<String>,
}

impl TimeAxis {
    /// Recognize a time axis from its `CTYPE` (Sec.9.2.1, Sec.9.5.3).
    ///
    /// `timesys` is the header's `TIMESYS` value, already upper-cased,
    /// which Sec.9.2.1 defaults to `UTC`.
    ///
    /// Returns `None` for any other axis type.
    #[must_use]
    pub fn recognize(axis: usize, ctype: &str, timesys: &str) -> Option<Self> {
        let ct = ctype.trim();
        // Sec.9.2.1: "for backward compatibility, all except TIMESYS
        // and PTYPEi may also assume the value TIME (case-insensitive),
        // whereupon the time scale shall be that recorded in TIMESYS
        // or, in its absence, its default value, UTC".
        let source = if ct.eq_ignore_ascii_case("TIME") {
            timesys
        } else if is_time_scale(ct) {
            ct
        } else {
            return None;
        };
        let scale = base_time_scale(source).to_ascii_uppercase();
        let realization = source
            .split_once('(')
            .and_then(|(_, rest)| rest.strip_suffix(')'))
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty());
        Some(Self {
            axis,
            scale,
            realization,
            trefpos: None,
            trefdir: None,
            plephem: None,
        })
    }
}

/// A phase axis (Standard Sec.9.6, `CTYPE = 'PHASE'`).
///
/// Like a time axis this is recognition, not an algorithm: a phase
/// axis stays on the linear pipeline. What this records is the pair of
/// keywords only a phase axis may carry -- the time at its zero point
/// and its period.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PhaseAxis {
    /// Zero-based axis index.
    pub axis: usize,
    /// `CZPHSia` -- time at the zero point of the phase axis, in the
    /// same unit as the time keywords.
    pub czphs: Option<f64>,
    /// `CPERIia` -- period of the phase axis, for the constant-period
    /// case the keyword can express.
    pub cperi: Option<f64>,
}

/// True if `ctype` names a phase axis. Matched case-insensitively,
/// like `'TIME'` in [`TimeAxis::recognize`].
#[must_use]
pub fn is_phase_ctype(ctype: &str) -> bool {
    ctype.trim().eq_ignore_ascii_case("PHASE")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctype_time_defers_to_timesys() {
        let t = TimeAxis::recognize(2, "TIME", "TT").unwrap();
        assert_eq!(t.axis, 2);
        assert_eq!(t.scale, "TT");
        assert!(t.realization.is_none());
        // Case-insensitive, per Sec.9.2.1.
        assert_eq!(TimeAxis::recognize(0, "time", "TAI").unwrap().scale, "TAI");
        // TIMESYS itself defaults to UTC, which the caller supplies.
        assert_eq!(TimeAxis::recognize(0, "TIME", "UTC").unwrap().scale, "UTC");
    }

    /// Sec.9.2.1 lets the axis name its own scale, overriding TIMESYS.
    #[test]
    fn ctype_may_name_the_scale_itself() {
        for s in [
            "TAI", "TT", "TDT", "ET", "IAT", "UT1", "UTC", "GMT", "GPS", "TCG", "TCB", "TDB",
            "LOCAL",
        ] {
            let t = TimeAxis::recognize(0, s, "UTC")
                .unwrap_or_else(|| panic!("{s} should be a time scale"));
            assert_eq!(t.scale, s, "{s}");
        }
        // The axis wins over a disagreeing TIMESYS.
        assert_eq!(TimeAxis::recognize(0, "TDB", "UTC").unwrap().scale, "TDB");
    }

    #[test]
    fn realization_is_kept_but_does_not_change_the_scale() {
        let t = TimeAxis::recognize(0, "TT(BIPM08)", "UTC").unwrap();
        assert_eq!(t.scale, "TT");
        assert_eq!(t.realization.as_deref(), Some("BIPM08"));
        // Reached through TIMESYS too.
        let t = TimeAxis::recognize(0, "TIME", "UTC(NIST)").unwrap();
        assert_eq!(t.scale, "UTC");
        assert_eq!(t.realization.as_deref(), Some("NIST"));
        // Table 29's `UT()` form needs its qualifier to be recognized.
        assert!(TimeAxis::recognize(0, "UT(NIST)", "UTC").is_some());
    }

    #[test]
    fn other_axis_types_are_not_time() {
        for ct in [
            "RA---TAN", "DEC--TAN", "FREQ", "WAVE-F2W", "STOKES", "DETX", "PHASE", "",
        ] {
            assert!(
                TimeAxis::recognize(0, ct, "UTC").is_none(),
                "{ct} should not be a time axis"
            );
        }
    }
}
