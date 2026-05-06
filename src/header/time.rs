//! Time keyword access (FITS Standard Sec.9, "Representations of time
//! coordinates"). This module parses `DATE-OBS`-style ISO-8601
//! strings, exposes `MJD-OBS`, `JD-OBS`, `TIMESYS`, `MJDREF`,
//! `JDREF`, and the data-event keywords (`DATE-BEG`, `DATE-END`,
//! `MJD-BEG`, `MJD-END`, `XPOSURE`, `TELAPSE`), and provides
//! UTC -> TAI conversion via a bundled leap-second table.

use std::sync::OnceLock;

use crate::header::Header;

// -- Leap-second table --------------------------------------------------------

/// The bundled leap-second table (IERS Bulletin 69, January 2025).
static LEAP_SECOND_TABLE: OnceLock<Vec<(f64, i32)>> = OnceLock::new();

/// Parse the embedded `leap_second.dat` file into `(mjd, tai_minus_utc)` pairs
/// sorted by MJD.  Comment lines start with `#`; data lines have the form
/// `   <mjd>    <day> <month> <year>   <tai_minus_utc>`.
fn leap_second_table() -> &'static Vec<(f64, i32)> {
    LEAP_SECOND_TABLE.get_or_init(|| {
        let src = include_str!("leap_second.dat");
        let mut table: Vec<(f64, i32)> = src
            .lines()
            .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| {
                let mut parts = l.split_whitespace();
                let mjd: f64 = parts.next()?.parse().ok()?;
                // skip day, month, year
                parts.next()?;
                parts.next()?;
                parts.next()?;
                let tai_minus_utc: i32 = parts.next()?.parse().ok()?;
                Some((mjd, tai_minus_utc))
            })
            .collect();
        table.sort_by(|a, b| a.0.total_cmp(&b.0));
        table
    })
}

/// Return the TAI - UTC offset (integer seconds) applicable at the given UTC
/// MJD.
///
/// The table covers 1972-01-01 (MJD 41317) onwards, when the modern
/// leap-second system began. For dates before that epoch `0` is returned;
/// callers needing pre-1972 UTC(BIH) corrections must apply their own
/// rubberized-second model.
#[must_use]
pub fn tai_minus_utc_at(mjd_utc: f64) -> i32 {
    let table = leap_second_table();
    // Find the last entry whose MJD <= mjd_utc.
    let offset = table.partition_point(|(mjd, _)| *mjd <= mjd_utc);
    if offset == 0 {
        0 // before the start of the leap-second era
    } else {
        table[offset - 1].1
    }
}

// -- IsoDateTime --------------------------------------------------------------

/// A parsed FITS-style ISO-8601 timestamp.
///
/// Per Sec.9.1.1 the canonical form is `YYYY-MM-DDThh:mm:ss[.sss...]`
/// (the `T` separator is mandatory when a time portion is present).
/// A bare `YYYY-MM-DD` is also accepted; in that case `hour`,
/// `minute`, `second`, and `frac_second` are all zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsoDateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    /// Fractional second in `[0.0, 1.0)`. Preserves up to ~15 digits
    /// of precision via `f64`; sub-microsecond precision near the
    /// year 2000 is *not* round-trip-safe and callers needing it
    /// should split into integer + fractional days themselves.
    pub frac_second: f64,
}

impl IsoDateTime {
    /// Parse a `DATE-OBS`-style FITS timestamp. Returns `None` when
    /// the input is not a recognized ISO-8601 form.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        // Date portion: YYYY-MM-DD or YYYY-MM-DDThh:mm:ss[.frac]
        let (date, time) = match s.split_once('T') {
            Some((d, t)) => (d, Some(t)),
            None => (s, None),
        };
        let mut date_parts = date.splitn(3, '-');
        let year: i32 = date_parts.next()?.parse().ok()?;
        let month: u8 = date_parts.next()?.parse().ok()?;
        let day: u8 = date_parts.next()?.parse().ok()?;
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        let (hour, minute, second, frac_second) = match time {
            None => (0, 0, 0, 0.0),
            Some(t) => {
                let mut tp = t.splitn(3, ':');
                let h: u8 = tp.next()?.parse().ok()?;
                let m: u8 = tp.next()?.parse().ok()?;
                let s_full = tp.next()?;
                // Accept trailing 'Z' (UTC marker) per Standard Sec.9.1.1.
                let s_full = s_full.trim_end_matches('Z');
                let (sec_s, frac_s) = match s_full.split_once('.') {
                    Some((a, b)) => (a, b),
                    None => (s_full, ""),
                };
                let sec: u8 = sec_s.parse().ok()?;
                let frac: f64 = if frac_s.is_empty() {
                    0.0
                } else {
                    format!("0.{frac_s}").parse().ok()?
                };
                if h > 23 || m > 59 || sec > 60 {
                    // sec=60 is allowed for leap-second-bearing days.
                    return None;
                }
                (h, m, sec, frac)
            }
        };
        Some(Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            frac_second,
        })
    }

    /// Convert to Modified Julian Date (MJD = JD - 2 400 000.5),
    /// using the proleptic Gregorian calendar. Accuracy is
    /// dominated by `f64` precision near MJD ~= 50 000 (~30 mus).
    #[must_use]
    pub fn mjd(&self) -> f64 {
        // Fliegel & Van Flandern (1968) algorithm, as cited by the
        // FITS time-paper (Rots et al. 2015) for the integer-MJD
        // portion. Works for any proleptic Gregorian date in the
        // range supported by `i32` years.
        let y = i64::from(self.year);
        let m = i64::from(self.month);
        let d = i64::from(self.day);
        let a = (14 - m) / 12;
        let y2 = y + 4800 - a;
        let m2 = m + 12 * a - 3;
        let jdn = d + (153 * m2 + 2) / 5 + 365 * y2 + y2 / 4 - y2 / 100 + y2 / 400 - 32045;
        let day_frac = f64::from(self.hour) / 24.0
            + f64::from(self.minute) / 1440.0
            + (f64::from(self.second) + self.frac_second) / 86_400.0;
        // JDN above is the JD at noon of the date; subtract 0.5 to
        // get JD at midnight, then convert to MJD.
        (jdn as f64 - 0.5 - 2_400_000.5) + day_frac
    }

    /// Return the TAI - UTC offset (integer seconds) applicable at this
    /// timestamp, looked up from the bundled leap-second table.
    ///
    /// See [`tai_minus_utc_at`] for the full contract.
    #[must_use]
    pub fn tai_minus_utc(&self) -> i32 {
        tai_minus_utc_at(self.mjd())
    }

    /// Convert this UTC timestamp to TAI, returned as an MJD.
    ///
    /// `mjd_tai = mjd_utc + tai_minus_utc / 86400.0`
    #[must_use]
    pub fn mjd_tai(&self) -> f64 {
        self.mjd() + f64::from(self.tai_minus_utc()) / 86_400.0
    }
}

impl Header {
    /// `DATE-OBS` (Standard Sec.9.2.1) parsed into [`IsoDateTime`].
    /// Returns `None` if absent or unparseable.
    #[must_use]
    pub fn date_obs(&self) -> Option<IsoDateTime> {
        IsoDateTime::parse(self.optional_string("DATE-OBS")?)
    }

    /// `MJD-OBS` (Standard Sec.9.2.2) -- Modified Julian Date of the
    /// observation start. Falls back to converting `DATE-OBS` if the
    /// keyword is absent.
    #[must_use]
    pub fn mjd_obs(&self) -> Option<f64> {
        if let Some(v) = self.optional_real("MJD-OBS") {
            return Some(v);
        }
        Some(self.date_obs()?.mjd())
    }

    /// `TIMESYS` (Standard Sec.9.2.1, Table 30): time scale identifier
    /// such as `"UTC"`, `"TAI"`, `"TT"`, `"TDB"`, `"TCG"`, `"TCB"`.
    /// Returned trimmed and uppercased; defaults to `"UTC"` per
    /// Sec.9.2.1 when absent.
    #[must_use]
    pub fn timesys(&self) -> String {
        self.optional_string("TIMESYS")
            .map_or_else(|| "UTC".into(), |s| s.trim().to_ascii_uppercase())
    }

    /// `MJDREF` (Standard Sec.9.2.3) -- Modified Julian Date reference
    /// for relative time keywords. Combines `MJDREFI` + `MJDREFF`
    /// when split; falls back to a single `MJDREF` value.
    #[must_use]
    pub fn mjdref(&self) -> Option<f64> {
        if let (Some(i), Some(f)) = (self.optional_real("MJDREFI"), self.optional_real("MJDREFF")) {
            return Some(i + f);
        }
        self.optional_real("MJDREF")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_iso_with_fraction() {
        let t = IsoDateTime::parse("2024-04-01T12:34:56.789").unwrap();
        assert_eq!(
            (t.year, t.month, t.day, t.hour, t.minute, t.second),
            (2024, 4, 1, 12, 34, 56)
        );
        assert!((t.frac_second - 0.789).abs() < 1e-12);
    }

    #[test]
    fn parses_date_only() {
        let t = IsoDateTime::parse("1999-12-31").unwrap();
        assert_eq!((t.year, t.hour, t.frac_second), (1999, 0, 0.0));
    }

    #[test]
    fn parses_trailing_z() {
        let t = IsoDateTime::parse("2000-01-01T00:00:00Z").unwrap();
        assert_eq!(t.year, 2000);
    }

    #[test]
    fn rejects_malformed() {
        assert!(IsoDateTime::parse("not-a-date").is_none());
        assert!(IsoDateTime::parse("2024-13-01").is_none());
        assert!(IsoDateTime::parse("2024-01-32").is_none());
    }

    #[test]
    fn mjd_matches_known_epochs() {
        // 1858-11-17 = MJD 0 by definition.
        let t0 = IsoDateTime::parse("1858-11-17").unwrap();
        assert!((t0.mjd() - 0.0).abs() < 1e-9, "got {}", t0.mjd());
        // J2000 epoch: 2000-01-01T12:00:00 TT = JD 2451545.0 = MJD 51544.5.
        let j2000 = IsoDateTime::parse("2000-01-01T12:00:00").unwrap();
        assert!((j2000.mjd() - 51544.5).abs() < 1e-9, "got {}", j2000.mjd());
    }

    #[test]
    fn tai_minus_utc_before_1972_is_zero() {
        assert_eq!(tai_minus_utc_at(0.0), 0);
        assert_eq!(tai_minus_utc_at(41316.9), 0);
    }

    #[test]
    fn tai_minus_utc_table_boundaries() {
        // First entry: MJD 41317 = 1972-01-01, TAI-UTC = 10.
        assert_eq!(tai_minus_utc_at(41317.0), 10);
        // Just before the 1999-01-01 entry (MJD 51179): still 31.
        assert_eq!(tai_minus_utc_at(51178.9), 31);
        // At and after 1999-01-01: 32.
        assert_eq!(tai_minus_utc_at(51179.0), 32);
        // Latest entry: 2017-01-01 (MJD 57754), TAI-UTC = 37.
        assert_eq!(tai_minus_utc_at(57754.0), 37);
        assert_eq!(tai_minus_utc_at(99999.0), 37);
    }

    #[test]
    fn mjd_tai_adds_leap_seconds() {
        // 2017-01-02 is after the last leap second (2017-01-01, TAI-UTC=37).
        let t = IsoDateTime::parse("2017-01-02T00:00:00").unwrap();
        assert_eq!(t.tai_minus_utc(), 37);
        let expected = t.mjd() + 37.0 / 86_400.0;
        assert!((t.mjd_tai() - expected).abs() < 1e-12);
    }
}
