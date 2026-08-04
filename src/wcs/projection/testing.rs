//! The round-trip harness that the family test modules share.
//!
//! Every family asserts one property. Whatever `s2x` accepts, `x2s`
//! must invert. One definition of the sweep holds every family to the
//! same grid density and the same tolerance.

use crate::wcs::D2R;
use crate::wcs::projection::Projection;

/// Round-trip a native grid through `s2x`/`x2s` at machine tolerance.
/// Points the forward refuses are skipped -- the contract is that
/// whatever it *accepts* must come back.
pub(crate) fn round_trip(p: &Projection, label: &str) {
    round_trip_tol(p, label, 1e-9);
}

/// [`round_trip`] with an explicit tolerance in degrees, for the
/// projections whose own formulation costs digits. Each caller states
/// why its projection cannot reach `1e-9`.
pub(crate) fn round_trip_tol(p: &Projection, label: &str, tol: f64) {
    let mut checked = 0_usize;
    let mut theta = -88.0_f64;
    while theta <= 88.0 {
        let mut phi = -179.0_f64;
        while phi <= 179.0 {
            if let Ok((x, y)) = p.s2x(phi, theta)
                && x.is_finite()
                && y.is_finite()
            {
                let (p2, t2) = p.x2s(x, y).unwrap_or_else(|e| {
                    panic!("{label}: x2s failed after s2x accepted ({phi}, {theta}): {e}")
                });
                let sep = separation(phi, theta, p2, t2);
                assert!(
                    sep < tol,
                    "{label}: ({phi}, {theta}) -> ({x}, {y}) -> ({p2}, {t2}), \
                     off by {sep:.3e} deg (tol {tol:.0e})"
                );
                checked += 1;
            }
            phi += 7.0;
        }
        theta += 4.0;
    }
    assert!(checked > 100, "{label}: only {checked} points accepted");
}

/// Angle in degrees between two native directions.
///
/// Compared as unit vectors because `phi` is degenerate at the poles
/// and wraps at +-180, so differencing the angles would flag both as
/// errors. `atan2(|u x w|, u.w)` is stable near zero, where
/// `acos(u.w)` would lose half the mantissa.
fn separation(phi1: f64, theta1: f64, phi2: f64, theta2: f64) -> f64 {
    let v = |a: f64, b: f64| {
        let (c, s) = ((b * D2R).cos(), (b * D2R).sin());
        [c * (a * D2R).cos(), c * (a * D2R).sin(), s]
    };
    let (u, w) = (v(phi1, theta1), v(phi2, theta2));
    let dot = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    let cr = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];
    (cr[0].powi(2) + cr[1].powi(2) + cr[2].powi(2))
        .sqrt()
        .atan2(dot)
        / D2R
}
