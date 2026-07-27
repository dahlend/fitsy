//! Time-scale reductions (FITS Standard Sec.9, WCS Paper IV Sec.3.1).
//!
//! Expected values are *derived* here from the defining relations
//! rather than quoted from another implementation:
//!
//! * `TAI = TT - 32.184 s`, exactly (Paper IV Sec.3.1.1).
//! * `GPS = TAI - 19 s`, exactly.
//! * `TAI - UTC` comes from the IERS leap-second table shipped in
//!   `src/header/leap_second.dat`; each offset used below cites the
//!   row it comes from.
//! * `T(TCG) = T(TT) + L_G * 86400 * (JD(TT) - JD_0)` and
//!   `T(TDB) = T(TCB) - L_B * 86400 * (JD(TCB) - JD_0) + TDB_0`
//!   (Paper IV Sec.3.1.2).
//!
//! The two coordinate scales are checked by their defining *rate*
//! rather than by recomputing the implementation's own expression,
//! so the assertion stays independent of how the reduction is
//! written. `TDB - TT` has no closed form -- a rigorous value needs a
//! time ephemeris -- so it is pinned by its physical signature:
//! bounded amplitude, annual period, and not identically zero.

use fitsy::header::time::{IsoDateTime, tai_minus_utc_at};
use fitsy::{Header, Value};

/// `TT - TAI`, in days. Paper IV Sec.3.1.1: exactly 32.184 s.
const TT_MINUS_TAI: f64 = 32.184 / 86_400.0;
/// `TAI - GPS`, in days: exactly 19 s.
const TAI_MINUS_GPS: f64 = 19.0 / 86_400.0;
/// Paper IV Sec.3.1.2 rate constants.
const L_G: f64 = 6.969_290_134e-10;
const L_B: f64 = 1.550_519_768e-8;

/// One f64 ulp at MJD ~60000 is ~7e-12 d ~ 6e-7 s: the floor for any
/// comparison that should be exact.
const ULP_SECONDS: f64 = 2e-6;

fn seconds_between(a_mjd: f64, b_mjd: f64) -> f64 {
    (a_mjd - b_mjd).abs() * 86_400.0
}

fn utc_from(timesys: &str, mjd: f64) -> f64 {
    let mut h = Header::empty();
    h.push("MJD-OBS", Value::Real(mjd), None).unwrap();
    h.push("TIMESYS", Value::String(timesys.into()), None)
        .unwrap();
    h.mjd_obs_utc()
        .unwrap_or_else(|| panic!("{timesys} has no UTC reduction"))
}

/// Calendar -> MJD, against epochs fixed outside this crate: the MJD
/// origin, the J2000.0 definition, and dates whose MJD the shipped
/// IERS table states directly.
#[test]
fn mjd_matches_defining_epochs() {
    const CASES: &[(&str, f64)] = &[
        // MJD is defined to be zero at 1858-11-17 00:00 UT.
        ("1858-11-17T00:00:00", 0.0),
        // J2000.0 = 2000-01-01 12:00 TT = JD 2451545.0.
        ("2000-01-01T12:00:00", 51_544.5),
        // These three are the MJDs the leap-second table gives for
        // its own rows, so the calendar arithmetic is checked against
        // a table the crate must agree with anyway.
        ("1972-01-01T00:00:00", 41_317.0),
        ("2015-07-01T00:00:00", 57_204.0),
        ("2017-01-01T00:00:00", 57_754.0),
        // A leap day, to exercise the Gregorian rule.
        ("2024-02-29T00:00:00", 60_369.0),
        // Century non-leap-year (1900 is not a leap year).
        ("1900-01-01T00:00:00", 15_020.0),
    ];
    for &(iso, want) in CASES {
        let got = IsoDateTime::parse(iso)
            .unwrap_or_else(|| panic!("failed to parse {iso}"))
            .mjd();
        assert!(
            (got - want).abs() < 1e-9,
            "{iso}: MJD {got}, expected {want}"
        );
    }
}

/// `TAI - UTC` must step exactly where the shipped IERS table says,
/// and hold its value in between.
#[test]
fn tai_offsets_follow_the_iers_table() {
    // (MJD, TAI-UTC) read off `leap_second.dat`.
    const ROWS: &[(f64, i32)] = &[
        (41_317.0, 10),
        (49_534.0, 29),
        (50_083.0, 30),
        (51_179.0, 32),
        (57_204.0, 36),
        (57_754.0, 37),
    ];
    for &(mjd, offset) in ROWS {
        assert_eq!(tai_minus_utc_at(mjd), offset, "at the step, MJD {mjd}");
        assert_eq!(
            tai_minus_utc_at(mjd + 0.5),
            offset,
            "the day after the step, MJD {mjd}"
        );
        assert!(
            tai_minus_utc_at(mjd - 0.5) < offset,
            "the day before the step should still hold the older offset, MJD {mjd}"
        );
    }
    // Before the modern system began there is no offset to apply.
    assert_eq!(tai_minus_utc_at(41_316.0), 0);
    assert_eq!(tai_minus_utc_at(30_000.0), 0);
}

/// A `23:59:60` stamp lies *inside* the inserted second, so it takes
/// the offset in force before the step -- the step happens at the
/// following midnight.
///
/// Regression: a real-valued UTC MJD has no 86401st second, so
/// `IsoDateTime::mjd` reports the leap second as that midnight; the
/// offset was then looked up there, picked the post-step value, and
/// placed the instant one second late. The three stamps around a leap
/// second collapsed to two.
#[test]
fn leap_second_stamp_uses_the_pre_step_offset() {
    for (day, leap_mjd, before, after) in [
        ("2016-12-31", 57_754.0, 36, 37),
        ("2015-06-30", 57_204.0, 35, 36),
    ] {
        let leap = IsoDateTime::parse(&format!("{day}T23:59:60")).unwrap();
        assert_eq!(
            leap.tai_minus_utc(),
            before,
            "{day}T23:59:60 is inside the inserted second"
        );
        // The step itself lands on the following midnight.
        assert_eq!(tai_minus_utc_at(leap_mjd), after);

        // TAI is uniform, so the three stamps must be one second
        // apart and strictly increasing.
        let t0 = IsoDateTime::parse(&format!("{day}T23:59:59"))
            .unwrap()
            .mjd_tai();
        let t1 = leap.mjd_tai();
        let midnight = leap_mjd + f64::from(after) / 86_400.0;
        for (a, b, label) in [
            (t0, t1, "23:59:59 -> 23:59:60"),
            (t1, midnight, "23:59:60 -> 00:00:00"),
        ] {
            let gap = (b - a) * 86_400.0;
            assert!(
                (gap - 1.0).abs() < 1e-5,
                "{day} {label}: TAI gap {gap} s, expected exactly 1"
            );
        }
    }
}

/// The scales that differ from TAI by a fixed, defined offset.
/// Expected values are written out from those offsets plus the IERS
/// table entry, so nothing here depends on how the reduction is
/// implemented.
#[test]
fn fixed_offset_scales_reduce_by_their_definitions() {
    // (MJD, TAI-UTC at that epoch, and the table row it comes from).
    const EPOCHS: &[(f64, f64)] = &[
        // 1995-10-10: between the 49534 and 50083 rows.
        (50_000.0, 29.0),
        // 2009-06-17: between the 54832 and 56109 rows.
        (55_000.25, 34.0),
        // 2023-02-25: after the final 57754 row.
        (60_000.5, 37.0),
    ];
    for &(mjd, leap) in EPOCHS {
        let leap_days = leap / 86_400.0;
        for (sys, want) in [
            ("TAI", mjd - leap_days),
            ("TT", mjd - TT_MINUS_TAI - leap_days),
            ("GPS", mjd + TAI_MINUS_GPS - leap_days),
            // UTC is the identity.
            ("UTC", mjd),
        ] {
            let got = utc_from(sys, mjd);
            let off = seconds_between(got, want);
            assert!(
                off < ULP_SECONDS,
                "{sys} MJD {mjd}: UTC {got}, defining relation gives {want} ({off} s apart)"
            );
        }
    }
}

/// TCG runs fast relative to TT at the constant rate `L_G`
/// (Paper IV Sec.3.1.2). Checking the *rate* over a long baseline
/// tests the relation without restating the implementation's formula.
#[test]
fn tcg_runs_fast_relative_to_tt_at_l_g() {
    // Same nominal reading on both scales; the UTC each reduces to
    // differs by the accumulated TCG-TT drift at that epoch.
    let drift_at = |mjd: f64| (utc_from("TT", mjd) - utc_from("TCG", mjd)) * 86_400.0;
    let (a, b) = (45_000.0_f64, 60_000.0_f64);
    let observed_rate = (drift_at(b) - drift_at(a)) / ((b - a) * 86_400.0);
    assert!(
        (observed_rate - L_G).abs() < L_G * 1e-6,
        "TCG-TT drift rate {observed_rate}, expected L_G = {L_G}"
    );
}

/// TCB runs fast relative to TDB at the constant rate `L_B`
/// (Paper IV Sec.3.1.2), by the same argument.
#[test]
fn tcb_runs_fast_relative_to_tdb_at_l_b() {
    let drift_at = |mjd: f64| (utc_from("TDB", mjd) - utc_from("TCB", mjd)) * 86_400.0;
    let (a, b) = (45_000.0_f64, 60_000.0_f64);
    let observed_rate = (drift_at(b) - drift_at(a)) / ((b - a) * 86_400.0);
    assert!(
        (observed_rate - L_B).abs() < L_B * 1e-4,
        "TCB-TDB drift rate {observed_rate}, expected L_B = {L_B}"
    );
}

/// `TDB - TT` is a periodic relativistic term, dominated by the
/// Earth's orbital eccentricity: amplitude ~1.7 ms, period one year,
/// mean ~zero. A rigorous value needs a time ephemeris (Paper IV
/// Sec.3.1.2), so the signature is what gets pinned here.
///
/// Regression: TDB was reduced in the same branch as TT, dropping the
/// term outright. That leaves the difference identically zero, which
/// every assertion below rejects.
#[test]
fn tdb_departs_from_tt_by_the_annual_relativistic_term() {
    let diff_ms = |mjd: f64| (utc_from("TDB", mjd) - utc_from("TT", mjd)) * 86_400.0 * 1e3;

    // Sample a full year at ~5-day spacing.
    let year: Vec<f64> = (0..73).map(|i| 55_000.0 + f64::from(i) * 5.0).collect();
    let samples: Vec<f64> = year.iter().map(|&m| diff_ms(m)).collect();
    let peak = samples.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));

    assert!(
        peak > 1.0,
        "TDB-TT peaks at only {peak} ms over a year; the term looks absent \
         (treating TDB as TT would give exactly 0)"
    );
    assert!(
        peak < 2.0,
        "TDB-TT peaks at {peak} ms; the term should not exceed ~1.7 ms"
    );
    // Mean over a whole year is ~zero for a sinusoid.
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    assert!(
        mean.abs() < 0.1,
        "TDB-TT averages {mean} ms over a year; expected ~0 for a periodic term"
    );
    // Annual periodicity: one year later the term repeats.
    for &mjd in &[55_000.0, 55_120.0, 55_240.0] {
        let now = diff_ms(mjd);
        let next = diff_ms(mjd + 365.25);
        assert!(
            (now - next).abs() < 0.1,
            "TDB-TT at MJD {mjd} is {now} ms but {next} ms a year later; \
             the term should be annual"
        );
    }
    // And it must actually move -- a constant offset is not this term.
    let spread = samples.iter().fold(f64::MIN, |a, &b| a.max(b))
        - samples.iter().fold(f64::MAX, |a, &b| a.min(b));
    assert!(
        spread > 2.0,
        "TDB-TT varies by only {spread} ms over a year"
    );
}

/// Scales with no closed-form UTC reduction must say so rather than
/// silently returning something plausible.
#[test]
fn unreducible_timesys_returns_none() {
    for sys in ["LOCAL", "UT1", "NONSENSE"] {
        let mut h = Header::empty();
        h.push("MJD-OBS", Value::Real(50_000.0), None).unwrap();
        h.push("TIMESYS", Value::String(sys.into()), None).unwrap();
        assert!(
            h.mjd_obs_utc().is_none(),
            "TIMESYS = {sys} should have no UTC reduction"
        );
    }
}
