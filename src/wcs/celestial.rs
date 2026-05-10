//! Native <-> celestial spherical rotation (Paper II Sec.2.4, Standard
//! Sec.8.3.1).
//!
//! Given the native pole `(LONPOLE, LATPOLE)` and the fiducial point
//! `(alpha0, delta0) = (CRVAL1, CRVAL2)`, this module rotates the native
//! spherical coordinates `(phi, theta)` produced by a projection into
//! celestial spherical coordinates `(alpha, delta)`. All angles are degrees.
//!
//! Equations (Paper II Sec.2.4):
//!
//! $$ \sin\delta = \`sin\theta\sin\delta_p` + \cos\theta\cos\delta_p\cos(\phi-\phi_p) $$
//! $$ \alpha - \`alpha_p` = \mathrm{atan2}(-\cos\theta\sin(\phi-\phi_p), \;
//!     \`sin\theta\cos\delta_p` - \cos\theta\sin\delta_p\cos(\phi-\phi_p)) $$
//!
//! where `(alpha_p, delta_p)` is the celestial pole's position and `phi_p` is
//! the native longitude of the celestial pole (`LONPOLE`, with the
//! defaults from Paper II Sec.2.4).

#![allow(
    clippy::doc_markdown,
    reason = "math formulae use backtick notation for subscripts within KaTeX blocks"
)]

use crate::error::{FitsError, Result};
use crate::wcs::{D2R, R2D};

/// Frame of reference attached to the celestial axis pair (Paper II
/// Sec.3.1, Standard Sec.8.4 Table 26).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CelestialFrame {
    Equatorial,
    Galactic,
    Ecliptic,
    Supergalactic,
    HelioEcliptic,
    Other,
}

/// Mapping between named celestial frames and their CTYPE axis-prefix
/// pair `(lon, lat)`. `Other` is excluded: it is the catch-all frame
/// for unrecognized prefixes and is not part of the parsing table
/// (its serialization-time prefix is hard-coded as `XLON`/`XLAT`).
const NAMED_FRAME_PREFIXES: &[(CelestialFrame, &str, &str)] = &[
    (CelestialFrame::Equatorial, "RA--", "DEC-"),
    (CelestialFrame::Galactic, "GLON", "GLAT"),
    (CelestialFrame::Ecliptic, "ELON", "ELAT"),
    (CelestialFrame::Supergalactic, "SLON", "SLAT"),
    (CelestialFrame::HelioEcliptic, "HLON", "HLAT"),
];

impl CelestialFrame {
    /// Recognize the frame from the first 4 characters of the
    /// longitude axis `CTYPE` value.
    #[must_use]
    pub fn from_ctype_prefix(prefix: &str) -> Self {
        match prefix {
            "RA--" => Self::Equatorial,
            "GLON" => Self::Galactic,
            "ELON" => Self::Ecliptic,
            "SLON" => Self::Supergalactic,
            "HLON" => Self::HelioEcliptic,
            _ => Self::Other,
        }
    }

    /// Canonical CTYPE axis-prefix pair `(lon, lat)` for this frame.
    /// `Other` is encoded as `("XLON", "XLAT")` per Paper II Sec.3.1.
    #[must_use]
    pub fn axis_prefixes(self) -> (&'static str, &'static str) {
        match self {
            Self::Equatorial => ("RA--", "DEC-"),
            Self::Galactic => ("GLON", "GLAT"),
            Self::Ecliptic => ("ELON", "ELAT"),
            Self::Supergalactic => ("SLON", "SLAT"),
            Self::HelioEcliptic => ("HLON", "HLAT"),
            Self::Other => ("XLON", "XLAT"),
        }
    }

    /// Iterate the five named frames (excludes `Other`). Useful for
    /// scanning a header's CTYPE values to pick a celestial axis pair.
    pub(crate) fn named_with_prefixes() -> impl Iterator<Item = (Self, &'static str, &'static str)>
    {
        NAMED_FRAME_PREFIXES.iter().copied()
    }
}

/// Equatorial reference system identifier (Standard Sec.8.4, Paper II
/// Sec.3.1, RADESYS keyword). Only meaningful for [`CelestialFrame::Equatorial`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RadeSys {
    /// International Celestial Reference System (default for
    /// EQUINOX absent or >= 1984.0).
    #[default]
    Icrs,
    /// FK5 (default if EQUINOX >= 1984.0 with no RADESYS).
    Fk5,
    /// FK4, mean place at the EQUINOX epoch.
    Fk4,
    /// FK4 without applied E-terms of aberration.
    Fk4NoE,
    /// Geocentric Apparent Place at MJD-OBS.
    Gappt,
    /// Other / non-equatorial / unknown.
    Other,
}

impl RadeSys {
    /// Parse a `RADESYS`/`RADECSYS` keyword value (case-insensitive).
    #[must_use]
    pub fn from_keyword(s: &str) -> Self {
        match s.trim().to_ascii_uppercase().as_str() {
            "ICRS" => Self::Icrs,
            "FK5" => Self::Fk5,
            "FK4" => Self::Fk4,
            "FK4-NO-E" => Self::Fk4NoE,
            "GAPPT" => Self::Gappt,
            _ => Self::Other,
        }
    }

    /// Resolve the default per Paper II Sec.3.1: ICRS unless EQUINOX
    /// indicates an FK4 epoch (< 1984.0) or FK5 epoch (>= 1984.0).
    #[must_use]
    pub fn default_for_equinox(equinox: Option<f64>) -> Self {
        match equinox {
            None => Self::Icrs,
            Some(e) if e < 1984.0 => Self::Fk4,
            Some(_) => Self::Fk5,
        }
    }
}

/// Rotation parameters (Paper II Sec.2.4) cached for repeated use.
#[derive(Debug, Clone)]
pub struct CelestialRotation {
    /// Celestial longitude of the fiducial point (CRVAL1), degrees.
    pub alpha0: f64,
    /// Celestial latitude of the fiducial point (CRVAL2), degrees.
    pub delta0: f64,
    /// Native longitude of the celestial pole, degrees (LONPOLE).
    pub phi_p: f64,
    /// Native latitude of the celestial pole, degrees (LATPOLE,
    /// resolved per Sec.2.4).
    pub theta_p: f64,
    /// Celestial longitude of the native pole.
    alpha_p: f64,
    /// Celestial latitude of the native pole.
    delta_p: f64,
}

impl CelestialRotation {
    /// Construct from the fiducial point and pole conventions. The
    /// rules for resolving `LONPOLE`/`LATPOLE` defaults given the
    /// projection's reference native latitude `theta0` are in
    /// Paper II Sec.2.4 / Sec.7.
    ///
    /// `lonpole`/`latpole` may be `None` to apply the defaults; pass
    /// the raw header values otherwise. `theta0` is the projection's
    /// native latitude of the fiducial point (`90deg` for zenithal,
    /// `0deg` for cylindrical/conventional, the conic reference latitude
    /// for conics; obtained from the projection).
    pub fn new(
        alpha0: f64,
        delta0: f64,
        lonpole: Option<f64>,
        latpole: Option<f64>,
        theta0_deg: f64,
    ) -> Result<Self> {
        // Default LONPOLE per Paper II Sec.2.4: 0deg if delta0 >= theta0,
        // 180deg otherwise.
        let phi_p = lonpole.unwrap_or(if delta0 >= theta0_deg { 0.0 } else { 180.0 });

        // Compute native pole position. We follow Paper II eqs. (8)-(10).
        // For theta0 = 90deg (zenithal), the native pole IS the fiducial
        // point, so (alpha_p, delta_p) = (alpha0, delta0).
        let (alpha_p, delta_p) = if (theta0_deg - 90.0).abs() < 1e-12 {
            (alpha0, delta0)
        } else {
            compute_native_pole(alpha0, delta0, phi_p, latpole, theta0_deg)?
        };

        Ok(Self {
            alpha0,
            delta0,
            phi_p,
            theta_p: 90.0,
            alpha_p,
            delta_p,
        })
    }

    /// Native (phi, theta) -> celestial (alpha, delta). All in degrees.
    #[must_use]
    pub fn native_to_celestial(&self, phi_deg: f64, theta_deg: f64) -> (f64, f64) {
        let phi = phi_deg * D2R;
        let theta = theta_deg * D2R;
        let phi_p = self.phi_p * D2R;
        let dp = self.delta_p * D2R;

        let dphi = phi - phi_p;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let sin_delta = sin_theta * dp.sin() + cos_theta * dp.cos() * dphi.cos();
        let delta = sin_delta.clamp(-1.0, 1.0).asin();

        let y = -cos_theta * dphi.sin();
        let x = sin_theta * dp.cos() - cos_theta * dp.sin() * dphi.cos();
        let alpha = self.alpha_p * D2R + y.atan2(x);

        let mut alpha_deg = alpha * R2D;
        // Normalize to [0, 360).
        alpha_deg = alpha_deg.rem_euclid(360.0);
        (alpha_deg, delta * R2D)
    }

    /// Celestial (alpha, delta) -> native (phi, theta). All in degrees.
    #[must_use]
    pub fn celestial_to_native(&self, alpha_deg: f64, delta_deg: f64) -> (f64, f64) {
        let alpha = alpha_deg * D2R;
        let delta = delta_deg * D2R;
        let phi_p = self.phi_p * D2R;
        let ap = self.alpha_p * D2R;
        let dp = self.delta_p * D2R;

        let dalpha = alpha - ap;
        let cos_delta = delta.cos();
        let sin_delta = delta.sin();

        let sin_theta = sin_delta * dp.sin() + cos_delta * dp.cos() * dalpha.cos();
        let theta = sin_theta.clamp(-1.0, 1.0).asin();

        let y = -cos_delta * dalpha.sin();
        let x = sin_delta * dp.cos() - cos_delta * dp.sin() * dalpha.cos();
        let phi = phi_p + y.atan2(x);

        let mut phi_deg = phi * R2D;
        phi_deg = ((phi_deg + 180.0).rem_euclid(360.0)) - 180.0;
        (phi_deg, theta * R2D)
    }
}

/// Resolve the native pole's celestial coordinates for non-zenithal
/// projections (Paper II eqs. 8-10). For the typical cylindrical /
/// conic case where `theta0 = 0deg`, the native pole is offset from the
/// fiducial point by `LATPOLE` along the meridian.
fn compute_native_pole(
    alpha0: f64,
    delta0: f64,
    phi_p_deg: f64,
    latpole: Option<f64>,
    theta0_deg: f64,
) -> Result<(f64, f64)> {
    // Paper II eq. (8): for theta0 != 90deg, given the fiducial point
    // (alpha0, delta0) and native pole offset phi_p, solve for delta_p.
    let phi_p = phi_p_deg * D2R;
    let d0 = delta0 * D2R;
    let t0 = theta0_deg * D2R;

    // sin(delta_p - asin(sin(theta0)/cos(d0)*sec(...))) = ... -- use
    // the explicit form from WCSLIB:
    //   delta_p = atan2(sin(theta0), cos(theta0)*cos(phi_p - phi0))  +/- term
    // Equivalent compact derivation (Calabretta & Greisen 2002 eq. 9):
    //
    //   delta_p = arg +/- acos( sin(d0)/sqrt(1 - cos^2t0*sin^2phi_p) )
    //   where arg = atan2(sin(t0), cos(t0)*cos(phi_p))   .. with phi_p
    //   measured from the fiducial native longitude (which is 0 by
    //   construction here).
    //
    // The ambiguity is resolved by LATPOLE.
    let sin_t0 = t0.sin();
    let cos_t0 = t0.cos();
    let cos_pp = phi_p.cos();

    let arg = sin_t0.atan2(cos_t0 * cos_pp);
    let denom = (1.0 - cos_t0 * cos_t0 * phi_p.sin().powi(2)).sqrt();
    if denom < 1e-15 {
        return Err(FitsError::Wcs(
            "LATPOLE indeterminate (denominator vanishes)".into(),
        ));
    }
    let ratio = (d0.sin() / denom).clamp(-1.0, 1.0);
    let acos = ratio.acos();
    let cand1 = arg + acos;
    let cand2 = arg - acos;

    // Per Paper II Sec.2.4 only candidates in [-pi/2, pi/2] are valid
    // delta_p; LATPOLE (default 90deg) selects between valid candidates.
    let half_pi = std::f64::consts::FRAC_PI_2;
    let in_range = |c: f64| c >= -half_pi - 1e-12 && c <= half_pi + 1e-12;
    let target = latpole.map_or(half_pi, |lp| lp * D2R);
    let chosen = match (in_range(cand1), in_range(cand2)) {
        (true, true) => {
            if (cand1 - target).abs() <= (cand2 - target).abs() {
                cand1
            } else {
                cand2
            }
        }
        (true, false) => cand1,
        (false, true) => cand2,
        (false, false) => {
            return Err(FitsError::Wcs(
                "LATPOLE: no valid native pole solution in [-90deg, 90deg]".into(),
            ));
        }
    };
    let delta_p = chosen.clamp(-half_pi, half_pi);

    // Pole-degenerate alpha_p (Paper II Sec.2.4 limiting form, matches
    // WCSLIB celset()): when delta_p ~= +/-90deg the standard formula for
    // alpha_p is indeterminate and the limit must be used.
    let dp_deg = delta_p * R2D;
    if (dp_deg - 90.0).abs() < 1e-6 {
        let alpha_p = (alpha0 - phi_p_deg - 180.0).rem_euclid(360.0);
        return Ok((alpha_p, dp_deg));
    }
    if (dp_deg + 90.0).abs() < 1e-6 {
        let alpha_p = (alpha0 + phi_p_deg).rem_euclid(360.0);
        return Ok((alpha_p, dp_deg));
    }

    // Paper II eqs. 5-7 give, at (alpha_0, delta_0, phi=0, theta=theta_0):
    //   sin(alpha_0 - alpha_p) cos delta_0 =  cos theta_0 sin phi_p
    //   cos(alpha_0 - alpha_p) cos delta_0 = (sin theta_0 - sin delta_p sin delta_0)/cos delta_p
    // => alpha_p = alpha_0 - atan2(cos theta_0 sin phi_p,
    //                    (sin theta_0 - sin delta_p sin delta_0)/cos delta_p).
    let dp = delta_p;
    let cos_dp = dp.cos();
    let y = cos_t0 * phi_p.sin();
    let x = (sin_t0 - dp.sin() * d0.sin()) / cos_dp;
    let alpha_p_rad = alpha0 * D2R - y.atan2(x);
    let alpha_p = (alpha_p_rad * R2D).rem_euclid(360.0);

    Ok((alpha_p, dp * R2D))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zenithal: native pole at (alpha0, delta0); the fiducial point
    /// must round-trip.
    #[test]
    fn zenithal_fiducial_round_trip() {
        let rot = CelestialRotation::new(83.633, 22.0145, None, None, 90.0).unwrap();
        // Fiducial point in native coordinates: (phi=0, theta=90).
        // (Crab nebula coordinates picked arbitrarily.)
        let (a, d) = rot.native_to_celestial(0.0, 90.0);
        assert!((a - 83.633).abs() < 1e-9 || (a - 83.633 + 360.0).abs() < 1e-9);
        assert!((d - 22.0145).abs() < 1e-9);
        let (phi, theta) = rot.celestial_to_native(83.633, 22.0145);
        assert!(theta > 89.999_999);
        // phi is undefined at the pole; we only check theta.
        let _ = phi;
    }

    #[test]
    fn round_trip_off_pole() {
        let rot = CelestialRotation::new(83.633, 22.0145, None, None, 90.0).unwrap();
        for &phi in &[0.0_f64, 45.0, 90.0, 200.0, 350.0] {
            for &theta in &[10.0_f64, 45.0, 80.0] {
                let (a, d) = rot.native_to_celestial(phi, theta);
                let (phi2, theta2) = rot.celestial_to_native(a, d);
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                assert!(
                    dphi.abs() < 1e-9,
                    "phi mismatch: {phi} vs {phi2} (a={a},d={d})"
                );
                assert!((theta - theta2).abs() < 1e-9);
            }
        }
    }
}
