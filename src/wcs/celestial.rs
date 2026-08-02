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
//! ```text
//! sin(delta) = sin(theta) sin(delta_p)
//!            + cos(theta) cos(delta_p) cos(phi - phi_p)
//!
//! alpha - alpha_p = atan2(-cos(theta) sin(phi - phi_p),
//!                          sin(theta) cos(delta_p)
//!                        - cos(theta) sin(delta_p) cos(phi - phi_p))
//! ```
//!
//! where `(alpha_p, delta_p)` is the celestial pole's position and `phi_p` is
//! the native longitude of the celestial pole (`LONPOLE`, with the
//! defaults from Paper II Sec.2.4).

use crate::error::{FitsError, Result};
use crate::wcs::{D2R, R2D};

/// Frame of reference attached to the celestial axis pair (Paper II
/// Sec.3.1, Standard Sec.8.4 Table 26).
// `non_exhaustive`: a frame now falling into `Other` may later get its
// own variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CelestialFrame {
    /// `RA`/`DEC` -- equatorial, the frame `RADESYSa` qualifies.
    Equatorial,
    /// `GLON`/`GLAT` -- galactic.
    Galactic,
    /// `ELON`/`ELAT` -- ecliptic.
    Ecliptic,
    /// `SLON`/`SLAT` -- supergalactic.
    Supergalactic,
    /// `HLON`/`HLAT` -- helioecliptic.
    HelioEcliptic,
    /// Any other pair, including the generic `xLON`/`xLAT` and
    /// `yzLN`/`yzLT` forms of Sec.8.2 that name a frame this enum has
    /// no variant for.
    Other,
}

/// Mapping between named celestial frames and their CTYPE axis-prefix
/// pair `(lon, lat)`. `Other` is excluded: it is the catch-all frame
/// for the generic `xLON`/`xLAT` and `yzLN`/`yzLT` forms, which
/// [`CelestialFrame::lat_prefix_for`] derives instead.
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
    ///
    /// `Other` means "not one of the five named frames", which covers
    /// both the generic `xLON`/`yzLN` forms and prefixes that are not
    /// celestial at all. Use [`Self::lat_prefix_for`] to tell those
    /// apart.
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

    /// For a 4-character CTYPE prefix naming a celestial *longitude*
    /// axis, the frame and the prefix its latitude partner must carry.
    /// `None` if the prefix is not a longitude form.
    ///
    /// Sec.8.4 allows three shapes: `RA--`/`DEC-`, `xLON`/`xLAT` where
    /// `x` names the frame, and `yzLN`/`yzLT` for planetary, lunar and
    /// solar systems. Only the five registered `x` values map to a
    /// named frame; the rest are [`CelestialFrame::Other`] and still a
    /// celestial pair.
    #[must_use]
    pub fn lat_prefix_for(prefix: &str) -> Option<(Self, String)> {
        let p = prefix.to_ascii_uppercase();
        if let Some((frame, _, lat)) = NAMED_FRAME_PREFIXES
            .iter()
            .find(|(_, lon, _)| *lon == p.as_str())
        {
            return Some((*frame, (*lat).to_string()));
        }
        // Generic forms; both are 4 characters, so the suffix
        // length is what tells them apart.
        if let Some(x) = p.strip_suffix("LON") {
            return Some((Self::Other, format!("{x}LAT")));
        }
        if let Some(yz) = p.strip_suffix("LN")
            && yz.len() == 2
        {
            return Some((Self::Other, format!("{yz}LT")));
        }
        None
    }
}

/// Equatorial reference system identifier (Standard Sec.8.4, Paper II
/// Sec.3.1, `RADESYS` keyword). This carries meaning only for
/// [`CelestialFrame::Equatorial`].
// `non_exhaustive`: new realizations of the equatorial system keep
// appearing, each one promoted out of `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum RadeSys {
    /// International Celestial Reference System. The default when
    /// `EQUINOX` is absent; with `EQUINOX` present the default is
    /// FK4 below 1984.0 and FK5 at or above it (Sec.8.3).
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
    /// `LATPOLE`, degrees, with the Sec.8.2 branch resolved.
    ///
    /// Sec.8.2 gives the keyword two definitions and calls them
    /// equivalent -- both are 90 degrees minus the angle between the
    /// two poles -- so this one field is `theta_p` and `delta_p` at
    /// once.
    ///
    /// A header's `LATPOLE` is only a hint: it picks between the two
    /// roots of Paper II eq. (9) when both are valid. This holds the
    /// root chosen, so writing it back re-selects the same one.
    pub theta_p: f64,
    /// Native longitude of the fiducial point, degrees (`PVi_1` on the
    /// longitude axis). Zero unless the header moves it.
    pub phi0: f64,
    /// Native latitude of the fiducial point, degrees (`PVi_2` on the
    /// longitude axis; defaults to the projection's `theta0`).
    pub theta0: f64,
    /// Celestial longitude of the native pole.
    alpha_p: f64,
}

impl CelestialRotation {
    /// Construct from the fiducial point and the pole conventions
    /// (Paper II Sec.2.4, Sec.7).
    ///
    /// The `lonpole` and `latpole` arguments take the raw header
    /// values, or `None` for the defaults.
    ///
    /// The `theta0_deg` argument is the native latitude of the
    /// fiducial point. It is 90 degrees for a zenithal projection, 0
    /// for a cylindrical one, and the reference latitude for a conic
    /// one. The projection supplies it unless `PVi_2` overrides it.
    ///
    /// The `phi0_deg` argument is the native longitude of that point,
    /// from `PVi_1`. It is normally 0.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the fiducial point and the pole
    /// convention admit no pole, or when `LATPOLE` selects neither of
    /// the two candidate poles.
    pub fn new(
        alpha0: f64,
        delta0: f64,
        lonpole: Option<f64>,
        latpole: Option<f64>,
        phi0_deg: f64,
        theta0_deg: f64,
    ) -> Result<Self> {
        // Default LONPOLE per Sec.8.2: `phi0` if delta0 >= theta0,
        // `phi0 + 180deg` otherwise.
        let phi_p = lonpole.unwrap_or(if delta0 >= theta0_deg {
            phi0_deg
        } else {
            phi0_deg + 180.0
        });

        // Compute native pole position. We follow Paper II eqs. (8)-(10).
        // For theta0 = 90deg (zenithal), the native pole IS the fiducial
        // point, so (alpha_p, delta_p) = (alpha0, delta0) whatever phi0 is.
        let (alpha_p, delta_p) = if (theta0_deg - 90.0).abs() < 1e-12 {
            (alpha0, delta0)
        } else {
            compute_native_pole(alpha0, delta0, phi_p, latpole, phi0_deg, theta0_deg)?
        };

        Ok(Self {
            alpha0,
            delta0,
            phi_p,
            theta_p: delta_p,
            phi0: phi0_deg,
            theta0: theta0_deg,
            alpha_p,
        })
    }

    /// Celestial longitude of the native pole, degrees.
    #[must_use]
    pub fn alpha_p(&self) -> f64 {
        self.alpha_p
    }

    /// Native (phi, theta) -> celestial (alpha, delta). All in degrees.
    #[must_use]
    pub fn native_to_celestial(&self, phi_deg: f64, theta_deg: f64) -> (f64, f64) {
        let phi = phi_deg * D2R;
        let theta = theta_deg * D2R;
        let phi_p = self.phi_p * D2R;
        let dp = self.theta_p * D2R;

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
        let dp = self.theta_p * D2R;

        let dalpha = alpha - ap;
        let cos_delta = delta.cos();
        let sin_delta = delta.sin();

        let sin_theta = sin_delta * dp.sin() + cos_delta * dp.cos() * dalpha.cos();
        let theta = sin_theta.clamp(-1.0, 1.0).asin();

        let y = -cos_delta * dalpha.sin();
        let x = sin_delta * dp.cos() - cos_delta * dp.sin() * dalpha.cos();
        let phi = phi_p + y.atan2(x);

        // Paper II Sec.2.1 computes phi with `arg`, whose range is
        // **(-180, 180]** -- closed at the top, open at the bottom. The
        // obvious `(phi + 180) mod 360 - 180` gives [-180, 180) instead
        // and so sends a point exactly on the antipodal meridian to
        // -180, mirroring it to the far edge of an all-sky map;
        // `wcslib` puts it at +180.
        let phi_deg = 180.0 - (180.0 - phi * R2D).rem_euclid(360.0);
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
    phi0_deg: f64,
    theta0_deg: f64,
) -> Result<(f64, f64)> {
    // Paper II eq. (8): for theta0 != 90deg, given the fiducial point
    // (alpha0, delta0) and native pole offset phi_p, solve for delta_p.
    //
    // Everything below uses the native longitudes only through
    // `phi_p - phi0`, so moving the fiducial point is exactly that
    // substitution -- and a no-op at the usual `phi0 = 0`.
    let phi_p = (phi_p_deg - phi0_deg) * D2R;
    let d0 = delta0 * D2R;
    let t0 = theta0_deg * D2R;

    // Calabretta & Greisen 2002 eq. (9), with phi_p measured from the
    // fiducial native longitude:
    //
    //   delta_p = arg +/- acos( sin(d0)/sqrt(1 - cos^2t0*sin^2phi_p) )
    //   where arg = atan2(sin(t0), cos(t0)*cos(phi_p))
    //
    // LATPOLE resolves the +/- ambiguity.
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

    // `arg + acos` can exceed pi -- at LONPOLE = 180deg it lands near
    // 2pi. Wrap before the range test below, or 345deg is rejected
    // instead of being read as the -15deg root, leaving the wrong one
    // and mirroring the sky. One +/-2pi step suffices, and it must be
    // an exact no-op in range: `rem_euclid` shifts values by an ulp,
    // breaking the symmetry the tie-break relies on.
    let wrap = |c: f64| {
        if c > std::f64::consts::PI {
            c - std::f64::consts::TAU
        } else if c <= -std::f64::consts::PI {
            c + std::f64::consts::TAU
        } else {
            c
        }
    };
    let cand1 = wrap(arg + acos);
    let cand2 = wrap(arg - acos);

    // Per Paper II Sec.2.4 only candidates in [-pi/2, pi/2] are valid
    // delta_p; LATPOLE (default 90deg) selects between valid candidates.
    let half_pi = std::f64::consts::FRAC_PI_2;
    let in_range = |c: f64| c >= -half_pi - 1e-12 && c <= half_pi + 1e-12;
    let target = latpole.map_or(half_pi, |lp| lp * D2R);
    let chosen = match (in_range(cand1), in_range(cand2)) {
        (true, true) => {
            // Closest to LATPOLE wins; an exact tie falls to
            // `arg - acos`.
            //
            // The tie needs a tolerance, not a bare `<`: a candidate
            // that required the 2pi wrap is an ulp off, which would
            // decide a mathematical tie on rounding noise. 1e-12 rad
            // is far below any meaningful LATPOLE.
            //
            // Which root a tie should take is unspecified, and
            // `wcslib` is not self-consistent: nudging CRVAL2 by
            // 1e-13 degrees flips its answer. We match it wherever
            // the comparison actually decides, and pick
            // deterministically here. The two roots are mirrored
            // skies, so a tie means the header itself is ambiguous --
            // hence `to_header` always writes the resolved LATPOLE.
            let d1 = (cand1 - target).abs();
            let d2 = (cand2 - target).abs();
            let tied = (d1 - d2).abs() <= 1e-12 * (1.0 + d1 + d2);
            if !tied && d1 < d2 { cand1 } else { cand2 }
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
    let dphi_p_deg = phi_p_deg - phi0_deg;
    if (dp_deg - 90.0).abs() < 1e-6 {
        let alpha_p = (alpha0 + dphi_p_deg - 180.0).rem_euclid(360.0);
        return Ok((alpha_p, dp_deg));
    }
    if (dp_deg + 90.0).abs() < 1e-6 {
        let alpha_p = (alpha0 - dphi_p_deg).rem_euclid(360.0);
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

    /// Sec.8.2 defines `LATPOLE` two ways and calls them equivalent.
    /// `theta_p` stores the second (what the rotation math needs);
    /// this checks the first agrees, by asking where the celestial
    /// north pole lands in native coordinates. The field used to be a
    /// hardcoded `90.0`, which satisfies neither.
    #[test]
    fn theta_p_is_latpole_under_both_readings() {
        let mut worst: f64 = 0.0;
        let mut checked = 0;
        for &t0 in &[90.0_f64, 0.0, 30.0, -45.0] {
            for &a0 in &[0.0_f64, 45.0, 200.0, 359.0] {
                for &d0 in &[0.0_f64, 30.0, -60.0, 89.0, -89.0] {
                    for lonpole in [None, Some(0.0), Some(90.0), Some(150.0), Some(180.0)] {
                        for latpole in [None, Some(-30.0), Some(90.0)] {
                            let Ok(r) = CelestialRotation::new(a0, d0, lonpole, latpole, 0.0, t0)
                            else {
                                // Not a legal pole configuration.
                                continue;
                            };
                            // Native latitude of the celestial north
                            // pole. Longitude is irrelevant at a pole.
                            let (_, theta_at_celestial_pole) = r.celestial_to_native(a0, 90.0);
                            worst = worst.max((theta_at_celestial_pole - r.theta_p).abs());
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 500, "only {checked} configurations exercised");
        assert!(
            worst < 1e-9,
            "the two readings of LATPOLE disagree by up to {worst} deg"
        );
    }

    /// Zenithal: native pole at (alpha0, delta0); the fiducial point
    /// must round-trip.
    #[test]
    fn zenithal_fiducial_round_trip() {
        let rot = CelestialRotation::new(83.633, 22.0145, None, None, 0.0, 90.0).unwrap();
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
        let rot = CelestialRotation::new(83.633, 22.0145, None, None, 0.0, 90.0).unwrap();
        // 180 exactly: `arg` is defined over `(-180, 180]`, and the
        // antipodal meridian used to come back mirrored to -180.
        for &phi in &[0.0_f64, 45.0, 90.0, 180.0, 200.0, 350.0] {
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
