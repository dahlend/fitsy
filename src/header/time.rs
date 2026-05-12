//! Time keyword access per WCS Paper IV (Rots et al. 2015) and FITS Standard Sec.9.

use std::sync::OnceLock;

use crate::header::Header;

// -- Barycentric time-scale constants (WCS Paper IV Sec.3.1.2) ------------------

// WCS Paper IV Sec.3.1.2 constants.
const L_G: f64 = 6.969_290_134e-10;
const L_B: f64 = 1.550_519_768e-8;
const MJD_0: f64 = 2_443_144.500_372_5 - 2_400_000.5;
const TDB_0_DAYS: f64 = -6.55e-5 / 86_400.0;

// -- Leap-second table --------------------------------------------------------

// IERS Bulletin 69, January 2025.
static LEAP_SECOND_TABLE: OnceLock<Vec<(f64, i32)>> = OnceLock::new();

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

/// TAI - UTC offset (integer seconds) at the given UTC MJD.
///
/// Returns `0` before MJD 41317 (1972-01-01); the modern leap-second system
/// did not exist prior to that date.
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

/// A parsed FITS ISO-8601 timestamp (FITS Standard Sec.9.1.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsoDateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    /// Fractional second in `[0.0, 1.0)`. `f64` precision limits this to ~30 us
    /// near J2000; split into integer + fractional days for sub-microsecond work.
    pub frac_second: f64,
}

impl IsoDateTime {
    /// Parse a FITS ISO-8601 timestamp. Returns `None` if the input is not a
    /// recognized form.
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

    /// Proleptic Gregorian MJD. Accuracy is limited by `f64` to ~30 us near
    /// MJD 50 000.
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

    /// TAI - UTC offset at this timestamp. See [`tai_minus_utc_at`].
    #[must_use]
    pub fn tai_minus_utc(&self) -> i32 {
        tai_minus_utc_at(self.mjd())
    }

    /// This UTC timestamp as TAI MJD.
    #[must_use]
    pub fn mjd_tai(&self) -> f64 {
        self.mjd() + f64::from(self.tai_minus_utc()) / 86_400.0
    }
}

/// Convert an MJD in the given `TIMESYS` scale to UTC MJD.
/// Returns `None` for scales that have no closed-form UTC reduction (`LOCAL`, `UT1`).
fn mjd_to_utc(mjd: f64, timesys: &str) -> Option<f64> {
    let mjd_tai = |m: f64| m - f64::from(tai_minus_utc_at(m)) / 86_400.0;
    Some(match timesys {
        "UTC" | "GMT" => mjd,
        "TAI" | "IAT" => mjd_tai(mjd),
        "TDB" | "TT" | "TDT" | "ET" => mjd_tai(mjd - 32.184 / 86_400.0),
        "GPS" => mjd_tai(mjd + 19.0 / 86_400.0),
        "TCG" => mjd_tai(mjd - L_G * (mjd - MJD_0) - 32.184 / 86_400.0),
        "TCB" => mjd_tai(mjd - L_B * (mjd - MJD_0) + TDB_0_DAYS - 32.184 / 86_400.0),
        _ => return None,
    })
}

/// Parse a `HH:MM:SS[.sss]` time string and return elapsed seconds.
fn parse_hms(s: &str) -> Option<f64> {
    let s = s.trim();
    let mut parts = s.splitn(3, ':');
    let h: u8 = parts.next()?.parse().ok()?;
    let m: u8 = parts.next()?.parse().ok()?;
    let sec_str = parts.next()?;
    let (int_s, frac_s) = sec_str.split_once('.').unwrap_or((sec_str, ""));
    let sec: u8 = int_s.parse().ok()?;
    let frac: f64 = if frac_s.is_empty() {
        0.0
    } else {
        format!("0.{frac_s}").parse().ok()?
    };
    if h > 23 || m > 59 || sec > 60 {
        return None;
    }
    Some(f64::from(h) * 3_600.0 + f64::from(m) * 60.0 + f64::from(sec) + frac)
}

impl Header {
    /// HDU creation date (`DATE`), always UTC.
    #[must_use]
    pub fn date(&self) -> Option<IsoDateTime> {
        IsoDateTime::parse(self.optional_string("DATE")?)
    }

    /// Parse a `UT*` keyword (`UTSTART`, `UTSTOP`) as UTC MJD.
    ///
    /// The value may be a full ISO-8601 string or a bare `HH:MM:SS[.sss]`. In
    /// the time-only case, the date is taken from the first present keyword in
    /// `date_keys`.
    fn ut_keyword_to_mjd(&self, key: &str, date_keys: &[&str]) -> Option<f64> {
        let s = self.optional_string(key)?.to_owned();
        // Full datetime: parse directly (already UTC).
        if let Some(dt) = IsoDateTime::parse(&s) {
            return Some(dt.mjd());
        }
        // Time-only: combine with the observation date.
        let secs = parse_hms(&s)?;
        let date_str = date_keys
            .iter()
            .find_map(|k| self.optional_string(k))?
            .to_owned();
        // Strip any time component so we get midnight of that date.
        let date_only = date_str.split('T').next()?;
        let date_mjd = IsoDateTime::parse(date_only)?.mjd();
        Some(date_mjd + secs / 86_400.0)
    }

    /// Time scale (`TIMESYS`), upper-cased. Defaults to `"UTC"` per WCS Paper IV Sec.3.1.2.
    #[must_use]
    pub fn time_sys(&self) -> String {
        self.optional_string("TIMESYS")
            .map_or_else(|| "UTC".into(), |s| s.trim().to_ascii_uppercase())
    }

    /// Observation time as UTC MJD: `MJD-OBS` -> `DATE-OBS` -> `JEPOCH` (TDB)
    /// -> `BEPOCH` (ET), converted via `TIMESYS` (WCS Paper IV Sec.3.1.2).
    #[must_use]
    pub fn mjd_obs_utc(&self) -> Option<f64> {
        let ts = self.time_sys();
        if let Some(v) = self.optional_real("MJD-OBS") {
            return mjd_to_utc(v, &ts);
        }
        if let Some(dt) = self
            .optional_string("DATE-OBS")
            .and_then(IsoDateTime::parse)
        {
            return mjd_to_utc(dt.mjd(), &ts);
        }
        // JEPOCH: J = 2000.0 + (JD - 2451545.0) / 365.25 -> MJD = 51544.5 + (J - 2000.0) * 365.25
        if let Some(j) = self.optional_real("JEPOCH") {
            return mjd_to_utc(51_544.5 + (j - 2000.0) * 365.25, "TDB");
        }
        // BEPOCH: Lieske (1979) B = 1900.0 + (JD - 2415020.31352) / 365.242198781
        if let Some(b) = self.optional_real("BEPOCH") {
            return mjd_to_utc(15_019.813_52 + (b - 1900.0) * 365.242_198_781, "ET");
        }
        None
    }

    /// Observation start as UTC MJD: `MJD-BEG` -> `DATE-BEG` -> `TSTART`+`MJDREF`
    /// (converted via `TIMESYS`) -> `UTSTART`+`DATE-OBS`.
    #[must_use]
    pub fn mjd_begin_utc(&self) -> Option<f64> {
        let ts = self.time_sys();
        let mjd = self
            .optional_real("MJD-BEG")
            .or_else(|| {
                self.optional_string("DATE-BEG")
                    .and_then(IsoDateTime::parse)
                    .map(|dt| dt.mjd())
            })
            .or_else(|| {
                let seconds = self.read_time_in_seconds("TSTART")?;
                Some(self.mjd_ref()? + seconds / 86_400.0)
            });
        if let Some(utc) = mjd.and_then(|m| mjd_to_utc(m, &ts)) {
            return Some(utc);
        }
        self.ut_keyword_to_mjd("UTSTART", &["DATE-OBS"])
    }

    /// Observation end as UTC MJD: `MJD-END` -> `DATE-END` -> `TSTOP`+`MJDREF`
    /// (converted via `TIMESYS`) -> `UTSTOP`+`DATE-OBS`.
    #[must_use]
    pub fn mjd_end_utc(&self) -> Option<f64> {
        let ts = self.time_sys();
        let mjd = self
            .optional_real("MJD-END")
            .or_else(|| {
                self.optional_string("DATE-END")
                    .and_then(IsoDateTime::parse)
                    .map(|dt| dt.mjd())
            })
            .or_else(|| {
                let seconds = self.read_time_in_seconds("TSTOP")?;
                Some(self.mjd_ref()? + seconds / 86_400.0)
            });
        if let Some(utc) = mjd.and_then(|m| mjd_to_utc(m, &ts)) {
            return Some(utc);
        }
        self.ut_keyword_to_mjd("UTSTOP", &["DATE-OBS"])
    }

    /// Average time as UTC MJD: `MJD-AVG` -> `DATE-AVG` (converted via `TIMESYS`)
    /// -> mean of [`Self::mjd_begin_utc`] and [`Self::mjd_end_utc`].
    #[must_use]
    pub fn mjd_avg_utc(&self) -> Option<f64> {
        if let Some(mjd) = self.optional_real("MJD-AVG").or_else(|| {
            self.optional_string("DATE-AVG")
                .and_then(IsoDateTime::parse)
                .map(|dt| dt.mjd())
        }) {
            return mjd_to_utc(mjd, &self.time_sys());
        }
        Some(f64::midpoint(self.mjd_begin_utc()?, self.mjd_end_utc()?))
    }

    /// Time unit (`TIMEUNIT`), lower-cased. Defaults to `"s"` per WCS Paper IV Sec.3.2.
    #[must_use]
    pub fn time_unit(&self) -> String {
        self.optional_string("TIMEUNIT")
            .map_or_else(|| "s".into(), |s| s.trim().to_ascii_lowercase())
    }

    /// Effective exposure time in seconds: `XPOSURE` (in `TIMEUNIT`, with
    /// per-card `[unit]` annotation as override) -> `EXPTIME` (pre-standard,
    /// always seconds).
    #[must_use]
    pub fn time_exposure(&self) -> Option<f64> {
        if let Some(v) = self.read_time_in_seconds("XPOSURE") {
            return Some(v);
        }
        // EXPTIME is a pre-standard keyword always expressed in seconds.
        self.optional_real("EXPTIME")
    }

    /// Wall-clock elapsed time in seconds (`TELAPSE` in `TIMEUNIT`, with
    /// per-card `[unit]` annotation as override).
    #[must_use]
    pub fn time_elapsed(&self) -> Option<f64> {
        self.read_time_in_seconds("TELAPSE")
    }

    /// Read a TIMEUNIT-scaled keyword and return its value in seconds.
    /// Per-card `[unit]` annotation, if present, overrides the global TIMEUNIT.
    fn read_time_in_seconds(&self, key: &str) -> Option<f64> {
        let v = self.optional_real(key)?;
        let unit = self.keyword_unit(key).unwrap_or_else(|| self.time_unit());
        Some(v * super::units::si_factor(&unit)?)
    }

    /// Reference epoch as MJD: `MJDREFI`+`MJDREFF` -> `MJDREF` -> `JDREFI`+`JDREFF`
    /// -> `JDREF` -> `DATEREF`. Zero point for relative time values in the HDU.
    #[must_use]
    pub fn mjd_ref(&self) -> Option<f64> {
        // MJDREF family (highest precedence).
        if let (Some(i), Some(f)) = (self.optional_int("MJDREFI"), self.optional_real("MJDREFF")) {
            return Some(i as f64 + f);
        }
        if let Some(v) = self.optional_real("MJDREF") {
            return Some(v);
        }
        // JDREF family.
        if let (Some(i), Some(f)) = (self.optional_int("JDREFI"), self.optional_real("JDREFF")) {
            return Some(i as f64 + f - 2_400_000.5);
        }
        if let Some(jd) = self.optional_real("JDREF") {
            return Some(jd - 2_400_000.5);
        }
        // DATEREF (lowest precedence).
        IsoDateTime::parse(self.optional_string("DATEREF")?).map(|dt| dt.mjd())
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

    fn make_obs_header(mjd_obs: f64, timesys: &str) -> Header {
        let mut h = Header::empty();
        h.push("MJD-OBS", mjd_obs, None).unwrap();
        h.push("TIMESYS", timesys.to_string(), None).unwrap();
        h
    }

    #[test]
    fn mjd_obs_utc_passthrough_for_utc() {
        let h = make_obs_header(57754.0, "UTC");
        assert!((h.mjd_obs_utc().unwrap() - 57754.0).abs() < 1e-9);
    }

    #[test]
    fn mjd_obs_utc_from_tai() {
        // TAI-UTC = 37 s after 2017-01-01. TAI MJD 57754.0 -> UTC = 57754.0 - 37/86400.
        let h = make_obs_header(57754.0, "TAI");
        let expected = 57754.0 - 37.0 / 86_400.0;
        assert!((h.mjd_obs_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_obs_utc_from_tt() {
        // Use 57755.0 (2017-01-02 TT)  -  well past the 2017-01-01 leap second in every scale.
        // TT = TAI + 32.184 s; TAI-UTC = 37 s after 2017-01-01. UTC = TT - (32.184+37)/86400.
        // (57754.0 TT would convert to TAI that is still before the transition, giving TAI-UTC=36.)
        let h = make_obs_header(57755.0, "TT");
        let expected = 57755.0 - (32.184 + 37.0) / 86_400.0;
        assert!((h.mjd_obs_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_obs_utc_from_gps() {
        // GPS = TAI - 19 s; TAI-UTC = 37 s. GPS MJD 57754.0 -> UTC = 57754.0 - (37-19)/86400.
        let h = make_obs_header(57754.0, "GPS");
        let expected = 57754.0 - (37.0 - 19.0) / 86_400.0;
        assert!((h.mjd_obs_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_obs_utc_from_tcg() {
        // TCG -> TT: MJD(TT) = MJD(TCG) - L_G * (MJD(TCG) - MJD_0)
        // Then TT -> UTC: MJD(UTC) = MJD(TT) - (32.184 + 37) / 86400
        let mjd_tcg = 57755.0_f64;
        let mjd_tt = mjd_tcg - L_G * (mjd_tcg - MJD_0);
        let expected = mjd_tt - (32.184 + 37.0) / 86_400.0;
        let h = make_obs_header(mjd_tcg, "TCG");
        assert!((h.mjd_obs_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_obs_utc_from_tdb() {
        // TDB ~ TT within <= 2 ms; 57755.0 is the same safe date as the TT test.
        // Expected UTC = TDB - (32.184 + 37) / 86400, same formula as TT.
        let h = make_obs_header(57755.0, "TDB");
        let expected = 57755.0 - (32.184 + 37.0) / 86_400.0;
        assert!((h.mjd_obs_utc().unwrap() - expected).abs() < 1e-6); // <= 2 ms tolerance
    }

    #[test]
    fn mjd_obs_utc_from_tcb() {
        // TCB -> TDB: MJD(TDB) = MJD(TCB) - L_B * (MJD(TCB) - MJD_0) + TDB_0/86400
        // Then TDB -> UTC as TDB ~ TT: UTC = TDB - (32.184 + 37) / 86400
        // Use 57755.0 (2017-01-02), safely past the last leap second in all scales.
        let mjd_tcb = 57755.0_f64;
        let mjd_tdb = mjd_tcb - L_B * (mjd_tcb - MJD_0) + TDB_0_DAYS;
        let expected = mjd_tdb - (32.184 + 37.0) / 86_400.0;
        let h = make_obs_header(mjd_tcb, "TCB");
        assert!((h.mjd_obs_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_obs_utc_none_for_unsupported_scale() {
        assert!(make_obs_header(57754.0, "LOCAL").mjd_obs_utc().is_none());
        assert!(make_obs_header(57754.0, "UT1").mjd_obs_utc().is_none());
    }

    #[test]
    fn mjd_beg_from_mjd_keyword() {
        let mut h = Header::empty();
        h.push("MJD-BEG", 57754.0_f64, None).unwrap();
        assert!((h.mjd_begin_utc().unwrap() - 57754.0).abs() < 1e-9);
    }

    #[test]
    fn mjd_beg_falls_back_to_date_beg() {
        let mut h = Header::empty();
        h.push("DATE-BEG", "2017-01-01T00:00:00".to_string(), None)
            .unwrap();
        let expected = IsoDateTime::parse("2017-01-01T00:00:00").unwrap().mjd();
        assert!((h.mjd_begin_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_beg_falls_back_to_tstart() {
        // TSTART=100 s relative to MJDREF=57754.0, default TIMEUNIT='s'
        let mut h = Header::empty();
        h.push("MJDREF", 57754.0_f64, None).unwrap();
        h.push("TSTART", 100.0_f64, None).unwrap();
        let expected = 57754.0 + 100.0 / 86_400.0;
        assert!((h.mjd_begin_utc().unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn mjd_end_falls_back_to_tstop_with_timeunit_days() {
        let mut h = Header::empty();
        h.push("MJDREF", 57754.0_f64, None).unwrap();
        h.push("TSTOP", 1.5_f64, None).unwrap();
        h.push("TIMEUNIT", "d".to_string(), None).unwrap();
        let expected = 57754.0 + 1.5;
        assert!((h.mjd_end_utc().unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn mjd_avg_from_date_avg() {
        let mut h = Header::empty();
        h.push("DATE-AVG", "2017-01-01T12:00:00".to_string(), None)
            .unwrap();
        let expected = IsoDateTime::parse("2017-01-01T12:00:00").unwrap().mjd();
        assert!((h.mjd_avg_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_beg_utc_converts_tai() {
        let mut h = Header::empty();
        h.push("MJD-BEG", 57755.0_f64, None).unwrap();
        h.push("TIMESYS", "TAI".to_string(), None).unwrap();
        let expected = 57755.0 - 37.0 / 86_400.0;
        assert!((h.mjd_begin_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_end_utc_converts_tai() {
        let mut h = Header::empty();
        h.push("MJD-END", 57755.5_f64, None).unwrap();
        h.push("TIMESYS", "TAI".to_string(), None).unwrap();
        let expected = 57755.5 - 37.0 / 86_400.0;
        assert!((h.mjd_end_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_avg_utc_converts_tai() {
        let mut h = Header::empty();
        h.push("MJD-AVG", 57755.25_f64, None).unwrap();
        h.push("TIMESYS", "TAI".to_string(), None).unwrap();
        let expected = 57755.25 - 37.0 / 86_400.0;
        assert!((h.mjd_avg_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_obs_falls_back_to_jepoch() {
        // J2000.0 = MJD 51544.5 TDB. TAI-UTC at that date = 32 s (since 1999-01-01).
        // UTC = 51544.5 - (32.184 + 32) / 86400.
        let expected = 51_544.5 - (32.184 + 32.0) / 86_400.0;
        let mut h = Header::empty();
        h.push("JEPOCH", 2000.0_f64, None).unwrap();
        assert!((h.mjd_obs_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_obs_falls_back_to_bepoch() {
        // B1900.0 = MJD 15019.81352 ET. Pre-1972, TAI-UTC = 0.
        // UTC = 15019.81352 - 32.184 / 86400.
        let expected = 15_019.813_52 - 32.184 / 86_400.0;
        let mut h = Header::empty();
        h.push("BEPOCH", 1900.0_f64, None).unwrap();
        assert!((h.mjd_obs_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_obs_prefers_mjd_obs_over_jepoch() {
        let mut h = Header::empty();
        h.push("MJD-OBS", 57754.0_f64, None).unwrap();
        h.push("JEPOCH", 2000.0_f64, None).unwrap();
        assert!((h.mjd_obs_utc().unwrap() - 57754.0).abs() < 1e-9);
    }

    #[test]
    fn xposure_default_timeunit_seconds() {
        let mut h = Header::empty();
        h.push("XPOSURE", 120.0_f64, None).unwrap();
        assert!((h.time_exposure().unwrap() - 120.0).abs() < 1e-9);
    }

    #[test]
    fn xposure_explicit_timeunit_minutes() {
        let mut h = Header::empty();
        h.push("XPOSURE", 2.0_f64, None).unwrap();
        h.push("TIMEUNIT", "min".to_string(), None).unwrap();
        assert!((h.time_exposure().unwrap() - 120.0).abs() < 1e-9);
    }

    #[test]
    fn telapse_explicit_timeunit_days() {
        let mut h = Header::empty();
        h.push("TELAPSE", 0.5_f64, None).unwrap();
        h.push("TIMEUNIT", "d".to_string(), None).unwrap();
        assert!((h.time_elapsed().unwrap() - 43_200.0).abs() < 1e-9);
    }

    #[test]
    fn xposure_falls_back_to_exptime() {
        let mut h = Header::empty();
        h.push("EXPTIME", 300.0_f64, None).unwrap();
        assert!((h.time_exposure().unwrap() - 300.0).abs() < 1e-9);
    }

    #[test]
    fn xposure_prefers_xposure_over_exptime() {
        let mut h = Header::empty();
        h.push("XPOSURE", 120.0_f64, None).unwrap();
        h.push("EXPTIME", 300.0_f64, None).unwrap();
        assert!((h.time_exposure().unwrap() - 120.0).abs() < 1e-9);
    }

    #[test]
    fn mjd_beg_utc_falls_back_to_utstart_with_date_obs() {
        // UTSTART = "12:00:00", DATE-OBS = "2017-01-02" -> MJD 57755.5 UTC.
        let mut h = Header::empty();
        h.push("UTSTART", "12:00:00".to_string(), None).unwrap();
        h.push("DATE-OBS", "2017-01-02".to_string(), None).unwrap();
        let expected = IsoDateTime::parse("2017-01-02").unwrap().mjd() + 0.5;
        assert!((h.mjd_begin_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_end_utc_falls_back_to_utstop_with_date_obs() {
        // No MJD-END or DATE-END present, so mjd_end() returns None and we
        // fall through to UTSTOP + DATE-OBS.
        let mut h = Header::empty();
        h.push("UTSTOP", "13:30:00".to_string(), None).unwrap();
        h.push("DATE-OBS", "2017-01-02".to_string(), None).unwrap();
        let expected = IsoDateTime::parse("2017-01-02").unwrap().mjd()
            + (13.0 * 3_600.0 + 30.0 * 60.0) / 86_400.0;
        assert!((h.mjd_end_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn utstart_full_iso_parsed_directly() {
        // If UTSTART happens to be a full datetime, use it directly.
        let mut h = Header::empty();
        h.push("UTSTART", "2017-01-02T06:00:00".to_string(), None)
            .unwrap();
        let expected = IsoDateTime::parse("2017-01-02T06:00:00").unwrap().mjd();
        assert!((h.mjd_begin_utc().unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn mjd_avg_utc_falls_back_to_midpoint() {
        let mut h = Header::empty();
        h.push("MJD-BEG", 57755.0_f64, None).unwrap();
        h.push("MJD-END", 57755.5_f64, None).unwrap();
        // Default TIMESYS = UTC, so no conversion needed.
        assert!((h.mjd_avg_utc().unwrap() - 57755.25).abs() < 1e-9);
    }
}
