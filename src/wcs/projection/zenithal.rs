// Several projections never fail in one direction. The unit structs
// have no state for `&self` to read. Every method still keeps one
// uniform signature, a fallible `Result` behind a `&self` receiver,
// so the `Projection` enum can dispatch all 28 variants through one
// match arm shape.
#![allow(
    clippy::unnecessary_wraps,
    clippy::unused_self,
    clippy::trivially_copy_pass_by_ref,
    reason = "uniform method signature across all projections for enum dispatch"
)]

//! Zenithal (azimuthal) projections -- Paper II Sec.5.1.
//!
//! All nine members of this family have `theta_0 = 90deg` except for the
//! auxiliary formula helpers that are used privately below.

use std::f64::consts::PI;

use crate::error::{FitsError, Result};
use crate::wcs::{D2R, R2D};

// -- shared zenithal helpers ------------------------------------------

/// Plane `(x, y)` from a zenithal radius and native longitude, both in
/// degrees. Every zenithal projection differs only in how it computes
/// the radius, so they all finish here.
#[inline]
pub(super) fn zenithal_xy(r_deg: f64, phi_deg: f64) -> (f64, f64) {
    // Paper II eq. (12)-(13): x = R sin(phi), y = -R cos(phi).
    let phi = phi_deg * D2R;
    (r_deg * phi.sin(), -r_deg * phi.cos())
}

/// Native longitude and zenithal radius from plane `(x, y)`, all in
/// degrees. This is the inverse of [`zenithal_xy`] and the first step
/// of every zenithal `x2s`.
#[inline]
pub(super) fn zenithal_phi_r(x_deg: f64, y_deg: f64) -> (f64, f64) {
    // Paper II eq. (14)-(15): phi = atan2(x, -y); R = sqrt(x^2+y^2).
    let phi = x_deg.atan2(-y_deg) * R2D;
    let r = (x_deg * x_deg + y_deg * y_deg).sqrt();
    (phi, r)
}

/// `sin((90 - theta) / 2)`, whose square is `(1 - sin theta) / 2`.
///
/// A zenithal projection is expanded about `theta = 90`. Its whole
/// field therefore sits where `1 - sin(theta)` subtracts two nearly
/// equal numbers. One arcsecond from the pole the two agree to eleven
/// digits, so the subtraction keeps five.
///
/// The identity `1 - sin(theta) = 2 sin^2((90 - theta) / 2)` is exact.
/// It evaluates a sine near zero, which is well conditioned.
///
/// The result is negative for `theta > 90`. Callers clamp it. A native
/// latitude past the pole is round-off, not a real point.
#[inline]
fn sin_half_colat(theta_deg: f64) -> f64 {
    ((90.0 - theta_deg) * D2R / 2.0).sin()
}

/// `theta` in degrees from `u = 1 - sin(theta)`.
///
/// This inverts [`sin_half_colat`] through the square. The literal
/// reading is `asin(1 - u)`, which loses the pole: the derivative of
/// `asin` diverges as its argument approaches 1. Returning through the
/// half angle keeps the `asin` argument near zero.
#[inline]
fn theta_from_one_minus_sin(u: f64) -> f64 {
    90.0 - 2.0 * (0.5 * u).max(0.0).sqrt().min(1.0).asin() * R2D
}

/// `ln(cos xi)` without the cancellation, for `xi` in `[0, pi/2]`.
///
/// `xi.cos().ln()` reads a cosine that has already rounded to 1. Below
/// `xi` of about 1.5e-8 it returns exactly zero. That term carries the
/// leading behavior of the `AIR` radius near the pole, so the radius
/// loses a factor of two there, not a few digits.
///
/// `ln(1 - sin^2 xi) / 2` is the same quantity. It keeps the small part
/// small, which is the range `ln_1p` is accurate over.
///
/// The identity fails at the other end of the domain. `sin(xi)` rounds
/// to 1 once `cos(xi)` falls below about 2e-8, and `1 - sin^2` is then
/// zero for a cosine that is still positive, which reports `-inf`. The
/// cancellation runs the other way there: the cosine is the small,
/// well-resolved quantity, so read the logarithm straight off it. That
/// range is `theta_b` within 2.4e-6 degrees of the south pole, where
/// `AIR` is degenerate anyway -- this keeps the value finite rather
/// than making it accurate.
#[inline]
fn ln_cos(xi: f64) -> f64 {
    let s = xi.sin();
    let s2 = s * s;
    if s2 < 1.0 {
        0.5 * (-s2).ln_1p()
    } else {
        xi.cos().ln()
    }
}

// -- TAN --------------------------------------------------------------

/// Gnomonic / tangent-plane projection (Paper II Sec.5.1.4).
#[derive(Debug, Clone, Copy)]
pub struct Tan;
impl Tan {
    /// No parameters, so the table is empty.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        Vec::new()
    }

    /// Reference native latitude, 90 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        90.0
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `theta` is 0 or less. The tangent plane
    /// only reaches the hemisphere around the reference point.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let t = theta * D2R;
        if t.sin() <= 0.0 {
            return Err(FitsError::Wcs(
                "TAN: theta <= 0 lies in the unprojected hemisphere".into(),
            ));
        }
        // R = (180/pi)*cot(theta)
        let r = R2D / t.tan();
        Ok(zenithal_xy(r, phi))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The plane covers the whole hemisphere,
    /// and a radius of 0 is the reference point.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (phi, r) = zenithal_phi_r(x, y);
        let theta = if r == 0.0 {
            90.0
        } else {
            (R2D / r).atan() * R2D
        };
        Ok((phi, theta))
    }
}

// -- STG --------------------------------------------------------------

/// Stereographic projection (Paper II Sec.5.1.6).
#[derive(Debug, Clone, Copy)]
pub struct Stg;
impl Stg {
    /// No parameters, so the table is empty.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        Vec::new()
    }

    /// Reference native latitude, 90 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        90.0
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] at `theta = -90`, the point diametrically
    /// opposite the reference point. It has no image on the plane.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let t = theta * D2R;
        let denom = 1.0 + t.sin();
        if denom.abs() < 1e-15 {
            return Err(FitsError::Wcs(
                "STG: theta = -90deg is the singular point".into(),
            ));
        }
        let r = 2.0 * R2D * t.cos() / denom;
        Ok(zenithal_xy(r, phi))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The stereographic plane maps to the
    /// sphere minus one point, and every radius names a latitude.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (phi, r) = zenithal_phi_r(x, y);
        let theta = 90.0 - 2.0 * (r / (2.0 * R2D)).atan() * R2D;
        Ok((phi, theta))
    }
}

// -- SIN --------------------------------------------------------------

/// Orthographic / synthesis / NCP projection (Paper II Sec.5.1.5,
/// eq. 17). Parameters `xi = PV2_1`, `eta = PV2_2` describe the slant.
/// `xi = eta = 0` is the orthogonal (simple) case.
#[derive(Debug, Clone, Copy)]
pub struct Sin {
    /// `PV2_1` -- the `xi` slant parameter. Zero for the orthographic
    /// case.
    pub xi: f64,
    /// `PV2_2` -- the `eta` slant parameter. Zero for the orthographic
    /// case.
    pub eta: f64,
}
impl Sin {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. This form takes its parameters
    /// without a validity constraint.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let xi = pv2.get(1).copied().unwrap_or(0.0);
        let eta = pv2.get(2).copied().unwrap_or(0.0);
        Ok(Self { xi, eta })
    }
}
impl Sin {
    /// Reference native latitude, 90 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        90.0
    }
    /// `PV2_1` is `xi` and `PV2_2` is `eta`.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.xi), (2, self.eta)]
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the point lies in the hemisphere facing
    /// away from the projection direction `(xi, eta, 1)`. That
    /// hemisphere shares the plane with the visible one, and `x2s`
    /// resolves every plane point to the visible branch.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        // Paper II eq. (30): x = R_0[costheta*sinphi + xi(1-sintheta)],
        //                    y = -R_0[costheta*cosphi - eta(1-sintheta)].
        let t = theta * D2R;
        let p = phi * D2R;
        let s = t.sin();
        let c = t.cos();
        let (sin_p, cos_p) = (p.sin(), p.cos());
        // eq. (30) is orthographic projection along `v = (xi, eta, 1)`:
        // with direction cosines `l = costheta sinphi`, `m = -costheta
        // cosphi`, `n = sintheta`, it reads `(x, y)/R_0 = (l, m) +
        // (1 - n)(xi, eta)`, which is where the line `P + s*v` meets
        // the plane `n = 1`.
        //
        // That map is two-to-one: the two preimages of a plane point
        // are reflections across `P.v = 0`, and only the hemisphere
        // facing the projection direction is the one the projection
        // represents. `x2s` resolves the pair by taking the smaller
        // root of its quadratic, which is the `P.v >= 0` branch, so a
        // forward that accepted the far side would report a plane
        // coordinate whose inverse names a different point.
        //
        // For `xi = eta = 0` this is `sintheta >= 0`, the orthographic
        // condition, reproduced here exactly -- the two zero products
        // cancel to a signed zero that leaves `s` unchanged.
        if self.xi * c * sin_p - self.eta * c * cos_p + s < 0.0 {
            return Err(FitsError::Wcs(
                "SIN: the point lies in the unprojected hemisphere".into(),
            ));
        }
        let one_minus_s = 1.0 - s;
        let x = R2D * (c * sin_p + self.xi * one_minus_s);
        let y = -R2D * (c * cos_p - self.eta * one_minus_s);
        Ok((x, y))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the point lies outside the disc that the
    /// sphere projects onto. The orthographic case tests the radius
    /// against `R_0`; the slant case tests the quadratic discriminant
    /// and requires a root within `[0, 2]`.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        // Solve the quadratic in u = 1 - sintheta derived from
        //   (X - xiu)^2 + (-Y + etau)^2 = cos^2theta = u(2-u)
        // => (1 + xi^2 + eta^2)*u^2 - 2(Xxi + Yeta + 1)*u + (X^2 + Y^2) = 0.
        let big_x = x / R2D;
        let big_y = y / R2D;
        if self.xi == 0.0 && self.eta == 0.0 {
            let (phi, r) = zenithal_phi_r(x, y);
            if r > R2D + 1e-9 {
                return Err(FitsError::Wcs("SIN: x^2+y^2 > R_0^2".into()));
            }
            let ratio = (r / R2D).clamp(-1.0, 1.0);
            let theta = ratio.acos() * R2D;
            return Ok((phi, theta));
        }
        let a = 1.0 + self.xi * self.xi + self.eta * self.eta;
        let b = -2.0 * (big_x * self.xi + big_y * self.eta + 1.0);
        let c = big_x * big_x + big_y * big_y;
        let disc = b * b - 4.0 * a * c;
        if disc < -1e-12 {
            return Err(FitsError::Wcs("SIN: outside the projection disc".into()));
        }
        let disc = disc.max(0.0);
        let u1 = (-b - disc.sqrt()) / (2.0 * a);
        let u2 = (-b + disc.sqrt()) / (2.0 * a);
        let u = if (-1e-12..=2.0 + 1e-12).contains(&u1) {
            u1
        } else if (-1e-12..=2.0 + 1e-12).contains(&u2) {
            u2
        } else {
            return Err(FitsError::Wcs("SIN: no admissible root".into()));
        };
        let u = u.clamp(0.0, 2.0);
        let sin_t = 1.0 - u;
        let theta = sin_t.clamp(-1.0, 1.0).asin() * R2D;
        let phi = if (u * (2.0 - u)).max(0.0).sqrt() < 1e-15 {
            0.0
        } else {
            (big_x - self.xi * u).atan2(-(big_y) + self.eta * u) * R2D
        };
        Ok((phi, theta))
    }
}

// -- ZPN --------------------------------------------------------------

/// Zenithal polynomial (Paper II Sec.5.1.4 ext, eq. 26). Parameters are
/// `P_m = PV2_m` for `m = 0..N`. The forward map evaluates a
/// polynomial in the zenith angle `zeta = (pi/2 - theta)` (radians); the
/// inverse uses Newton iteration on the same polynomial.
#[derive(Debug, Clone)]
pub struct Zpn {
    /// `P_m = PV2_m`, lowest order first, so `coeffs[m]` is `PV2_m`.
    ///
    /// Private; [`Self::from_pv`] is the only way in. It trims
    /// trailing zeros, and the evaluators rely on the last
    /// coefficient being non-zero.
    coeffs: Vec<f64>,
}
impl Zpn {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when every polynomial coefficient in the
    /// `PV2_m` table is 0.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let mut coeffs: Vec<f64> = pv2.to_vec();
        while coeffs.len() > 2 && coeffs.last().copied() == Some(0.0) {
            coeffs.pop();
        }
        if coeffs.iter().all(|&c| c == 0.0) {
            return Err(FitsError::Wcs(
                "ZPN: all polynomial coefficients are zero".into(),
            ));
        }
        Ok(Self { coeffs })
    }
    fn eval(&self, zeta: f64) -> f64 {
        let mut acc = 0.0_f64;
        for &c in self.coeffs.iter().rev() {
            acc = acc * zeta + c;
        }
        acc
    }
    fn deriv(&self, zeta: f64) -> f64 {
        let mut acc = 0.0_f64;
        for (m, &c) in self.coeffs.iter().enumerate().rev() {
            if m == 0 {
                break;
            }
            acc = acc * zeta + c * m as f64;
        }
        acc
    }
}
impl Zpn {
    /// Reference native latitude, 90 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        90.0
    }
    /// `PV2_m` is the coefficient of the degree-`m` term.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        // `coeffs[m]` is PV2_m directly. Trailing zeros are dropped;
        // the parser zero-fills, and `from_pv` rejects an all-zero
        // table, so at least one term always survives.
        let last = self.coeffs.iter().rposition(|c| *c != 0.0);
        match last {
            Some(n) => (0..=n).map(|m| (m as u32, self.coeffs[m])).collect(),
            None => Vec::new(),
        }
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The polynomial evaluates at every
    /// colatitude.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let zeta = (90.0 - theta) * D2R;
        let r = R2D * self.eval(zeta);
        Ok(zenithal_xy(r, phi))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// The polynomial has no closed-form inverse, so Newton iteration
    /// solves it.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the iteration does not converge in 64
    /// steps, which a non-monotonic polynomial causes, or when the
    /// solved colatitude falls outside 0 to `pi`.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (phi, r) = zenithal_phi_r(x, y);
        let target = r / R2D;
        let mut zeta = if self.coeffs.len() >= 2 && self.coeffs[1] != 0.0 {
            target / self.coeffs[1]
        } else {
            1.0
        };
        let mut converged = false;
        for _ in 0..64 {
            let f = self.eval(zeta) - target;
            let fp = self.deriv(zeta);
            if fp.abs() < 1e-15 {
                break;
            }
            let dz = f / fp;
            zeta -= dz;
            if dz.abs() < 1e-13 {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(FitsError::Wcs(
                "ZPN: Newton iteration failed to converge \
                 (polynomial may be non-monotonic at this point)"
                    .into(),
            ));
        }
        if !zeta.is_finite() || !(-1e-9..=PI + 1e-9).contains(&zeta) {
            return Err(FitsError::Wcs(
                "ZPN: solved zeta out of [0, pi] -- input is outside \
                 the projection's valid range"
                    .into(),
            ));
        }
        Ok((phi, 90.0 - zeta * R2D))
    }
}

// -- AZP --------------------------------------------------------------

/// Slant zenithal perspective (Paper II Sec.5.1.1, eqs. 16-22).
/// Parameters `mu = PV2_1` (default 0, != -1) and `gamma = PV2_2`
/// (default 0, degrees, |gamma| < 90deg).
#[derive(Debug, Clone, Copy)]
pub struct Azp {
    /// `PV2_1` -- `mu`, the distance of the projection point from the
    /// sphere center in spherical radii.
    pub mu: f64,
    /// `PV2_2` -- `gamma`, the tilt of the projection plane, degrees.
    pub gamma: f64,
}
impl Azp {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `PV2_1` (`mu`) is -1, which makes the
    /// projection singular, or when `PV2_2` (`gamma`) reaches 90
    /// degrees in magnitude.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let mu = pv2.get(1).copied().unwrap_or(0.0);
        let gamma = pv2.get(2).copied().unwrap_or(0.0);
        if (mu + 1.0).abs() < 1e-12 {
            return Err(FitsError::Wcs("AZP: PV2_1 (mu) = -1 is singular".into()));
        }
        if gamma.abs() >= 90.0 {
            return Err(FitsError::Wcs(format!(
                "AZP: |PV2_2 (gamma)| = {} >= 90deg",
                gamma.abs()
            )));
        }
        Ok(Self { mu, gamma })
    }
}
impl Azp {
    /// Reference native latitude, 90 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        90.0
    }
    /// `PV2_1` is `mu` and `PV2_2` is `gamma`.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.mu), (2, self.gamma)]
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in three cases:
    ///
    /// - `|mu| > 1` and the point lies past the fold at
    ///   `asin(-1 / mu)`, where two latitudes share one radius.
    /// - The perspective denominator vanishes.
    /// - The radius comes out negative, which puts the point behind
    ///   the projection point.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let p = phi * D2R;
        let t = theta * D2R;
        let g = self.gamma * D2R;
        // Where the projection folds (Paper II Sec.5.1.1). With
        // `D = mu + sin t + cos t cos p tan g`, the radius is
        // `R = (mu+1) cos t / D`, and
        //
        //     dR/dt = (mu+1) (-mu sin t - 1) / D^2
        //
        // -- the `tan g` terms cancel identically, so the stationary
        // point is at `sin t = -1/mu` whatever the tilt. Past it two
        // latitudes share one radius and the inverse cannot tell them
        // apart: it returns the branch holding the reference point
        // (`theta = 90`), which satisfies `sin t > -1/mu` for every
        // `|mu| > 1`. So the far branch has no representation here and
        // must be refused rather than silently mapped onto the near
        // one. For `|mu| <= 1` the stationary point falls outside
        // `[-90, 90]` and the whole sphere is single-valued.
        //
        // Without this an antipodal point mapped to the *reference
        // pixel itself*: at `mu = 2`, both `theta = 90` and
        // `theta = -90` give `R = 0`. `wcslib` reports these points as
        // invalid.
        if self.mu.abs() > 1.0 && t.sin() < -1.0 / self.mu {
            return Err(FitsError::Wcs(format!(
                "AZP: theta = {theta} is beyond the fold at asin(-1/mu) = {:.6} deg \
                 for mu = {}; the projection is two-valued there",
                (-1.0 / self.mu).clamp(-1.0, 1.0).asin() * R2D,
                self.mu,
            )));
        }
        let denom = self.mu + t.sin() + t.cos() * p.cos() * g.tan();
        if denom.abs() < 1e-15 {
            return Err(FitsError::Wcs("AZP: denominator vanishes".into()));
        }
        let r = R2D * (self.mu + 1.0) * t.cos() / denom;
        if r < 0.0 {
            return Err(FitsError::Wcs(format!(
                "AZP: theta = {theta}, phi = {phi} projects to a negative radius, \
                 i.e. behind the projection point"
            )));
        }
        Ok((r * p.sin(), -r * p.cos() / g.cos()))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the point is degenerate at the origin of
    /// the slanted frame, or when the `asin` argument exceeds 1. The
    /// second case is a plane point with no counterpart on the sphere.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let g = self.gamma * D2R;
        let cg = g.cos();
        let phi_rad = x.atan2(-y * cg);
        let rho = (x * x + (y * cg) * (y * cg)).sqrt() / R2D;
        let alpha = (self.mu + 1.0) - rho * phi_rad.cos() * g.tan();
        let k = (rho * rho + alpha * alpha).sqrt();
        if k < 1e-15 {
            return Err(FitsError::Wcs("AZP: degenerate (rho = alpha = 0)".into()));
        }
        let beta = alpha.atan2(rho);
        let arg = -rho * self.mu / k;
        if arg.abs() > 1.0 + 1e-9 {
            return Err(FitsError::Wcs(
                "AZP: argument out of range -- point not on the sphere".into(),
            ));
        }
        let theta_rad = beta + arg.clamp(-1.0, 1.0).asin();
        Ok((phi_rad * R2D, theta_rad * R2D))
    }
}

// -- ARC --------------------------------------------------------------

/// Zenithal equidistant (Paper II Sec.5.1.7).
#[derive(Debug, Clone, Copy)]
pub struct Arc;
impl Arc {
    /// No parameters, so the table is empty.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        Vec::new()
    }

    /// Reference native latitude, 90 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        90.0
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The radius is the colatitude itself,
    /// so the whole sphere projects.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        Ok(zenithal_xy(90.0 - theta, phi))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. A radius past 180 degrees wraps rather
    /// than failing, which matches the equidistant definition.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (phi, r) = zenithal_phi_r(x, y);
        Ok((phi, 90.0 - r))
    }
}

// -- ZEA --------------------------------------------------------------

/// Zenithal equal-area (Paper II Sec.5.1.8).
#[derive(Debug, Clone, Copy)]
pub struct Zea;
impl Zea {
    /// No parameters, so the table is empty.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        Vec::new()
    }

    /// Reference native latitude, 90 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        90.0
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The whole sphere projects onto a disc
    /// of radius `2 R_0`.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        // Paper II eq. (24) is `R = sqrt(2 (1 - sin theta))`, whose
        // radicand cancels at the pole. `sqrt(2 * 2 sin^2 h)` is
        // `2 sin h` for the half-colatitude `h`; see `sin_half_colat`.
        let r = 2.0 * R2D * sin_half_colat(theta).max(0.0);
        Ok(zenithal_xy(r, phi))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the radius exceeds `2 R_0`, which is
    /// outside the disc the sphere fills.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (phi, r) = zenithal_phi_r(x, y);
        let arg = r / (2.0 * R2D);
        if arg.abs() > 1.0 + 1e-9 {
            return Err(FitsError::Wcs("ZEA: outside disk".into()));
        }
        // Already the half-angle form, so this inverts `s2x` exactly.
        Ok((phi, 90.0 - 2.0 * arg.clamp(-1.0, 1.0).asin() * R2D))
    }
}

// -- SZP --------------------------------------------------------------

/// Slant zenithal perspective `SZP` (Paper II Sec.5.1.2, eqs. 9-11).
/// Parameters `mu = PV2_1` (default 0), `phi_c = PV2_2` (default 0deg),
/// `theta_c = PV2_3` (default 90deg).
#[derive(Debug, Clone, Copy)]
pub struct Szp {
    /// `PV2_1` -- `mu`, the projection point's distance from the
    /// sphere center in spherical radii.
    pub mu: f64,
    /// `PV2_2` -- native longitude of the projection point, degrees.
    pub phi_c: f64,
    /// `PV2_3` -- native latitude of the projection point, degrees.
    pub theta_c: f64,
    xp: f64,
    yp: f64,
    zp: f64,
}
impl Szp {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `PV2_3` (`theta_c`) exceeds 90 degrees
    /// in magnitude.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let mu = pv2.get(1).copied().unwrap_or(0.0);
        let phi_c = pv2.get(2).copied().unwrap_or(0.0);
        let theta_c = pv2.get(3).copied().unwrap_or(90.0);
        if theta_c.abs() > 90.0 {
            return Err(FitsError::Wcs(format!(
                "SZP: |PV2_3 (theta_c)| = {} > 90deg",
                theta_c.abs()
            )));
        }
        let pc = phi_c * D2R;
        let tc = theta_c * D2R;
        Ok(Self {
            mu,
            phi_c,
            theta_c,
            xp: -mu * tc.cos() * pc.sin(),
            yp: mu * tc.cos() * pc.cos(),
            zp: mu * tc.sin() + 1.0,
        })
    }

    /// The ray through plane point `(big_x, big_y)` (radians) meets the
    /// sphere twice; return the intersection the inverse resolves to,
    /// as `u = 1 - sin(theta)`.
    // Shared with `s2x`, which uses it to reject a latitude on the
    // hidden branch. The oblique fold has no tidy closed form, so the
    // forward asks the inverse and the two agree by construction.
    fn select_u(&self, big_x: f64, big_y: f64) -> Result<f64> {
        let (zp, xp, yp) = (self.zp, self.xp, self.yp);
        let b_sum = big_x * big_x + big_y * big_y;
        let d_sum = xp * xp + yp * yp;
        let a_dot = big_x * xp + big_y * yp;
        let qa = b_sum + d_sum - 2.0 * a_dot + zp * zp;
        let qb = -2.0 * zp * (b_sum - a_dot + zp);
        let qc = zp * zp * b_sum;
        let disc = qb * qb - 4.0 * qa * qc;
        if disc < -1e-12 {
            return Err(FitsError::Wcs(
                "SZP: ray misses the unit sphere -- point outside the projection".into(),
            ));
        }
        let disc = disc.max(0.0).sqrt();
        // Stable quadratic roots. Reading the root nearer zero from
        // `(-qb - disc) / (2 qa)` subtracts two nearly equal numbers.
        // That happens whenever `4 qa qc` is small against `qb^2`,
        // which is the near-pole case. There `qc` carries the whole
        // answer and the subtraction discards it. Build the root of
        // larger magnitude first, then reach the other through
        // `u1 u2 = qc / qa`.
        let q = -0.5 * (qb + if qb < 0.0 { -disc } else { disc });
        let (u1, u2) = if q == 0.0 {
            // Both roots are zero, or there is nothing to divide by.
            (0.0, 0.0)
        } else if qa == 0.0 {
            // The quadratic degenerated to a linear equation, whose
            // single root `-qc / qb` is what `qc / q` reduces to.
            (qc / q, qc / q)
        } else {
            (q / qa, qc / q)
        };
        let pick = |u: f64| (-1e-9..=2.0 + 1e-9).contains(&u);
        match (pick(u1), pick(u2)) {
            (true, true) => Ok(u1.min(u2)),
            (true, false) => Ok(u1),
            (false, true) => Ok(u2),
            _ => Err(FitsError::Wcs("SZP: no admissible root".into())),
        }
    }
}
impl Szp {
    /// Reference native latitude, 90 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        90.0
    }
    /// `PV2_1` is `mu`, `PV2_2` is `phi_c` and `PV2_3` is `theta_c`.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.mu), (2, self.phi_c), (3, self.theta_c)]
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the perspective denominator vanishes, or
    /// when the point lies on the hidden branch. The hidden branch has
    /// no closed form here, so the check asks the same root selection
    /// that `x2s` uses and compares the answer.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let p = phi * D2R;
        let t = theta * D2R;
        let cos_t = t.cos();
        // `1 - sin(theta)` through the half angle, which does not
        // cancel at the pole. See `sin_half_colat`.
        let h = sin_half_colat(theta);
        let one_ms = 2.0 * h * h;
        let denom = self.zp - one_ms;
        if denom.abs() < 1e-15 {
            return Err(FitsError::Wcs("SZP: denominator vanishes".into()));
        }
        let x = R2D * (self.zp * cos_t * p.sin() - self.xp * one_ms) / denom;
        let y = -R2D * (self.zp * cos_t * p.cos() + self.yp * one_ms) / denom;
        // Both sphere points on this ray land here; only the one the
        // inverse resolves to is representable. `one_ms` is exactly the
        // `u` that `select_u` returns, so this compares like with like.
        let selected = self.select_u(x / R2D, y / R2D)?;
        if (selected - one_ms).abs() > 1e-9 {
            return Err(FitsError::Wcs(format!(
                "SZP: theta = {theta} lies on the hidden branch -- the plane point it \
                 maps to belongs to theta = {:.6} deg instead",
                theta_from_one_minus_sin(selected),
            )));
        }
        Ok((x, y))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when no root of the perspective quadratic
    /// falls in `[0, 2]`. The point then lies off the region the
    /// sphere projects onto.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let big_x = x / R2D;
        let big_y = y / R2D;
        let (zp, xp, yp) = (self.zp, self.xp, self.yp);
        let u = self.select_u(big_x, big_y)?.clamp(0.0, 2.0);
        // `u` is `1 - sin(theta)`; recover the angle through the half
        // angle rather than `asin(1 - u)`, which loses the pole.
        let theta = theta_from_one_minus_sin(u);
        let phi = if (u * (2.0 - u)).max(0.0).sqrt() < 1e-15 {
            0.0
        } else {
            let yn = big_x * (zp - u) + xp * u;
            let xn = -(big_y * (zp - u) + yp * u);
            yn.atan2(xn) * R2D
        };
        Ok((phi, theta))
    }
}

// -- AIR --------------------------------------------------------------

/// Airy projection (Paper II Sec.5.1.9, eq. 25). Parameter `theta_b = PV2_1`
/// (default 90deg) is the latitude where azimuthal scale equals radial
/// scale. The inverse uses Newton iteration on `R(theta)`.
#[derive(Debug, Clone, Copy)]
pub struct Air {
    /// `PV2_1` -- `theta_b`, the latitude at which the error is
    /// minimized, degrees.
    ///
    /// Private; [`Self::from_pv`] is the only way to set it. It
    /// resolves `cb`, `xi_max` and `r_max` at construction. A later
    /// assignment would leave a projection whose serialized `PV2_1`
    /// disagrees with its own arithmetic.
    theta_b: f64,
    cb: f64,
    /// Half-colatitude where the invertible branch of `R` ends.
    ///
    /// `R` normally rises without bound as `theta` falls. For `theta_b`
    /// below about -76.5 degrees it turns over first and then rises
    /// again. One radius then answers to two latitudes, and the inverse
    /// cannot say which. `AZP` and `SZP` fold in the same way.
    ///
    /// [`Self::branch_end`] resolves this once, when the projection is
    /// built.
    xi_max: f64,
    /// `R` at [`Self::xi_max`].
    ///
    /// This is the largest radius the branch reaches. [`Air::x2s`]
    /// rejects a radius beyond it.
    r_max: f64,
}
impl Air {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `PV2_1` (`theta_b`) lies outside the
    /// range -90 to 90 degrees.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let theta_b = pv2.get(1).copied().unwrap_or(90.0);
        if !(-90.0..=90.0).contains(&theta_b) {
            return Err(FitsError::Wcs(format!(
                "AIR: PV2_1 (theta_b) = {theta_b} outside [-90deg, 90deg]"
            )));
        }
        let xi_b = (90.0 - theta_b) * D2R / 2.0;
        // The ratio tends to -1/2 as `xi_b` goes to zero. `ln_cos` is
        // accurate all the way down, so the guard is only about the
        // division, not about where the formula stops being usable.
        let cb = if xi_b == 0.0 {
            -0.5
        } else {
            ln_cos(xi_b) / xi_b.tan().powi(2)
        };
        let xi_max = Self::branch_end(cb);
        let mut air = Self {
            theta_b,
            cb,
            xi_max,
            r_max: 0.0,
        };
        air.r_max = air.r_of_xi(xi_max);
        Ok(air)
    }

    /// Half-colatitude of `theta`, the variable eq. (25) is written in.
    #[inline]
    fn xi_of(theta_deg: f64) -> f64 {
        (90.0 - theta_deg) * D2R / 2.0
    }

    /// A `cb` at or below this cannot fold, so [`Self::branch_end`]
    /// returns the domain edge without searching.
    ///
    /// `cb` is negative, so this is a test on its magnitude. Split
    /// [`Self::branch_end`]'s `h` into the part that depends on `cb`
    /// and the part that does not:
    ///
    /// ```text
    /// h(xi) = g(xi) + |cb| tan^2(xi),   g(xi) = sin^2(xi) + ln(cos xi)
    /// ```
    ///
    /// `h(xi) < 0` is then `|cb| < -g(xi) / tan^2(xi)`, so a fold
    /// exists exactly when `|cb|` falls below the largest value that
    /// ratio reaches. `g` is negative only past `xi = 63.2` degrees,
    /// and the ratio peaks at `xi = 74.66` degrees, where it is
    /// `0.0300797`. That is the whole criterion -- not a heuristic.
    ///
    /// The constant rounds that peak up, so the margin errs toward
    /// searching. A `cb` between the true peak and this value still
    /// takes the scan and still finds no fold. The test
    /// `air_skips_the_scan_only_when_r_is_monotonic` holds the bound
    /// against the behavior it stands for.
    ///
    /// In `theta_b` the threshold is about -76.4 degrees. Every
    /// projection north of that, which is every one in practice,
    /// skips the search.
    const NO_FOLD_CB: f64 = -0.0302;

    /// Largest `xi` this projection admits.
    ///
    /// This is the first turning point of `R`, or the edge of the
    /// domain when `R` has none.
    ///
    /// `dR/dxi` shares its sign with
    ///
    /// ```text
    /// h(xi) = sin^2(xi) + ln(cos xi) - cb tan^2(xi)
    /// ```
    ///
    /// which is `dR/dxi` scaled by the positive factor
    /// `tan^2(xi) / sec^2(xi)`. Near zero `h` tends to
    /// `(1/2 - cb) xi^2`, which is positive. As `xi` approaches `pi/2`
    /// the `-cb tan^2` term diverges upward, so `h` ends positive as
    /// well. Between the two ends `h` can dip below zero. It does so
    /// when `cb` is small enough, which means when `theta_b` is far
    /// enough south. The first zero of that dip is where `R` stops
    /// being injective.
    ///
    /// Most projections need no search at all, and [`Self::NO_FOLD_CB`]
    /// says which. The rest scan for the dip and then bisect. That runs
    /// once per projection rather than per point, so the scan can be
    /// dense. `STEPS` steps across the domain resolve 0.022 degrees of
    /// `xi`, which is 0.044 degrees of `theta`. The dip spans degrees
    /// where it exists, so the scan finds it.
    ///
    /// The exception is a `theta_b` at the threshold itself, near
    /// -76.5 degrees. The dip is born there as a tangency: `dR/dxi`
    /// touches zero without crossing it, and the dip is narrower than
    /// any step. A scan that misses it costs little, because the
    /// non-monotonicity it missed is itself infinitesimal. Sweeping
    /// `theta_b` across that threshold, the worst round trip is 2e-8
    /// degrees.
    fn branch_end(cb: f64) -> f64 {
        const STEPS: usize = 4096;
        // The domain edge: `R` is already ~6.6e6 degrees here.
        let limit = Self::xi_of(-89.999);
        if cb <= Self::NO_FOLD_CB {
            return limit;
        }
        let h = |xi: f64| {
            let s = xi.sin();
            s * s + ln_cos(xi) - cb * xi.tan().powi(2)
        };
        let mut prev = limit / (STEPS as f64);
        for k in 2..=STEPS {
            let xi = limit * (k as f64) / (STEPS as f64);
            if h(xi) < 0.0 {
                // Sign change in `(prev, xi)`: bisect for the zero.
                let (mut lo, mut hi) = (prev, xi);
                for _ in 0..80 {
                    let mid = lo.midpoint(hi);
                    if h(mid) > 0.0 { lo = mid } else { hi = mid }
                }
                return lo;
            }
            prev = xi;
        }
        limit
    }

    /// `R` at half-colatitude `xi`, for `xi` in `[0, pi/2)`.
    #[inline]
    fn r_of_xi(&self, xi: f64) -> f64 {
        if xi == 0.0 {
            return 0.0;
        }
        let tan_xi = xi.tan();
        -2.0 * R2D * (ln_cos(xi) / tan_xi + tan_xi * self.cb)
    }

    /// `R` and `dR/dxi` together, in degrees of radius and degrees per
    /// radian.
    ///
    /// The two share every transcendental they need. This is the
    /// per-iteration cost of the inverse, so the sharing matters.
    ///
    /// Differentiating `R` in closed form, with `L = ln(cos xi)` and
    /// `T = tan(xi)`:
    ///
    /// ```text
    /// dR/dxi = 2 R2D (1 + L sec^2(xi) / T^2 - cb sec^2(xi))
    /// ```
    #[inline]
    fn r_and_slope(&self, xi: f64) -> (f64, f64) {
        let tan_xi = xi.tan();
        let l = ln_cos(xi);
        let t2 = tan_xi * tan_xi;
        let sec2 = 1.0 + t2;
        (
            -2.0 * R2D * (l / tan_xi + tan_xi * self.cb),
            2.0 * R2D * (1.0 + l * sec2 / t2 - self.cb * sec2),
        )
    }
}
impl Air {
    /// Reference native latitude, 90 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        90.0
    }
    /// `PV2_1` is `theta_b`, the second point of minimum distortion.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.theta_b)]
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] at `theta = -90`, where the radius diverges,
    /// and past the fold. Below a `theta_b` of about -76.5 degrees the
    /// radius turns over, and two latitudes then share one radius.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        if theta <= -90.0 + 1e-12 {
            return Err(FitsError::Wcs("AIR: south pole maps to infinity".into()));
        }
        let xi = Self::xi_of(theta);
        // Past the fold two latitudes share one radius. The inverse
        // cannot tell them apart, so the forward refuses the far one.
        // Accepting it would map the far latitude onto a plane point
        // that belongs to the near one. `AZP` and `SZP` refuse their
        // own folds in the same way.
        if xi > self.xi_max {
            return Err(FitsError::Wcs(format!(
                "AIR: theta = {theta} lies past the fold at {:.6} deg, where R turns \
                 over and stops being invertible",
                90.0 - 2.0 * self.xi_max * R2D,
            )));
        }
        Ok(zenithal_xy(self.r_of_xi(xi), phi))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the radius exceeds the largest one this
    /// projection reaches before it folds. A radius of 0 is the
    /// reference point.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (phi, r) = zenithal_phi_r(x, y);
        if r < 1e-12 {
            return Ok((phi, 90.0));
        }
        if r > self.r_max {
            return Err(FitsError::Wcs(format!(
                "AIR: R = {r} exceeds {}, the largest radius this projection reaches \
                 before it folds",
                self.r_max
            )));
        }
        // Solve `R(xi) = r` on `[0, xi_max]`. `R` is strictly
        // increasing there, by construction of `xi_max`.
        //
        // Newton, safeguarded by a bracket. Each evaluation tells which
        // side of the root `xi` fell on and narrows the bracket. A step
        // that leaves the bracket is replaced by a bisection of it.
        // This keeps the guarantee of bisection and pays its price only
        // on the steps that need it. Bisecting the whole way costs one
        // evaluation per bit of the answer, which is about forty. Each
        // of those evaluations is three transcendentals.
        //
        // The seed is the small-angle limit
        // `R -> 2 R2D (1/2 - cb) xi`, which is exact to first order. A
        // zenithal projection keeps `xi` small over a real field. The
        // seed is then good to several digits, and Newton needs two or
        // three steps.
        let (mut lo, mut hi) = (0.0_f64, self.xi_max);
        let mut xi = (r / (2.0 * R2D * (0.5 - self.cb))).clamp(lo, hi);
        for _ in 0..64 {
            let (r_xi, slope) = self.r_and_slope(xi);
            let f = r_xi - r;
            // Narrow the bracket. `xi` becomes one of its ends, so the
            // acceptance test below is inclusive. A converged Newton
            // step lands on that end. A strict test would reject it and
            // bisect instead, discarding the answer.
            if f > 0.0 {
                hi = xi;
            } else {
                lo = xi;
            }
            // `slope > 0` across the branch; the comparison also
            // rejects the NaN a degenerate `xi` would produce.
            let next = if slope > 0.0 {
                let cand = xi - f / slope;
                if cand >= lo && cand <= hi {
                    cand
                } else {
                    lo.midpoint(hi)
                }
            } else {
                lo.midpoint(hi)
            };
            let delta = next - xi;
            xi = next;
            // The tolerance is relative to `xi`, not absolute. Near the
            // pole `xi` is itself of order 1e-10. An absolute floor
            // there stops while the leading digits are still moving.
            if delta.abs() <= 4.0 * f64::EPSILON * xi || hi - lo <= f64::EPSILON * hi {
                break;
            }
        }
        Ok((phi, 90.0 - 2.0 * xi * R2D))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wcs::projection::Projection;
    use crate::wcs::projection::testing::{round_trip, round_trip_tol};

    #[test]
    fn zenithal_projections_round_trip() {
        round_trip(&Tan.into(), "TAN");
        round_trip(&Stg.into(), "STG");
        round_trip(&Arc.into(), "ARC");
        round_trip(&Zea.into(), "ZEA");
        // SIN's inverse is `theta = acos(R/R_0)`, whose derivative is
        // infinite at the limb (`theta = 0`, `R = R_0`). One ulp in
        // `R/R_0` there is ~sqrt(2e-16) rad ~ 9e-7 deg, so the limb --
        // which this grid lands on exactly -- cannot do better. Away
        // from it the error is 4e-10 deg by `theta = 0.001` and 3e-13
        // by `theta = 0.1`.
        round_trip_tol(&Sin::from_pv(&[0.0, 0.0, 0.0]).unwrap().into(), "SIN", 1e-6);
        // `theta_b` spans its whole range: the southern half was
        // untested, and it is where `R` develops a fold.
        for theta_b in [90.0_f64, 60.0, 45.0, 0.0, -45.0, -70.0, -80.0, -89.0] {
            round_trip(
                &Air::from_pv(&[0.0, theta_b]).unwrap().into(),
                &format!("AIR theta_b={theta_b}"),
            );
        }
        round_trip(&Zpn::from_pv(&[0.0, 1.0]).unwrap().into(), "ZPN");
        for mu in [0.0_f64, 0.5, 1.0, 2.0, 5.0] {
            for gamma in [0.0_f64, 30.0, -20.0] {
                round_trip(
                    &Azp::from_pv(&[0.0, mu, gamma]).unwrap().into(),
                    &format!("AZP mu={mu} gamma={gamma}"),
                );
            }
        }
        for mu in [0.0_f64, 1.0, 2.0] {
            round_trip(
                &Szp::from_pv(&[0.0, mu, 180.0, 60.0]).unwrap().into(),
                &format!("SZP mu={mu}"),
            );
        }
    }

    /// A zenithal projection is expanded about `theta = 90`. An image
    /// using one puts its whole field within a fraction of a degree of
    /// the pole, and the field center lands closest of all. Accuracy
    /// there is what a caller sees.
    ///
    /// The grid above cannot check this. It steps `theta` by four
    /// degrees and never comes within two degrees of the pole. An
    /// absolute bound at that range hides a total relative loss at
    /// short range. This test asserts that the colatitude survives with
    /// relative accuracy. The colatitude is the quantity the projection
    /// formula is parameterized by.
    ///
    /// Regression: `ZEA` built its radius from `1 - sin(theta)`. `SZP`
    /// took the quadratic root nearer zero by direct subtraction. `AIR`
    /// read `ln(cos xi)` from a cosine that had rounded to one. Each is
    /// a cancellation that appears only at the pole, and each cost the
    /// colatitude most of its digits.
    #[test]
    fn zenithal_projections_keep_the_colatitude_near_the_pole() {
        let cases: Vec<(String, Projection)> = vec![
            ("TAN".into(), Projection::from(Tan)),
            ("STG".into(), Projection::from(Stg)),
            ("ARC".into(), Projection::from(Arc)),
            ("ZEA".into(), Projection::from(Zea)),
            (
                "SIN".into(),
                Projection::from(Sin::from_pv(&[0.0, 0.0, 0.0]).unwrap()),
            ),
            (
                "AIR tb=45".into(),
                Projection::from(Air::from_pv(&[0.0, 45.0]).unwrap()),
            ),
            (
                "AIR tb=90".into(),
                Projection::from(Air::from_pv(&[0.0, 90.0]).unwrap()),
            ),
            (
                "ZPN".into(),
                Projection::from(Zpn::from_pv(&[0.0, 1.0]).unwrap()),
            ),
            (
                "AZP mu=0".into(),
                Projection::from(Azp::from_pv(&[0.0, 0.0, 0.0]).unwrap()),
            ),
            (
                "AZP mu=2".into(),
                Projection::from(Azp::from_pv(&[0.0, 2.0, 30.0]).unwrap()),
            ),
            (
                "SZP mu=0".into(),
                Projection::from(Szp::from_pv(&[0.0, 0.0, 180.0, 60.0]).unwrap()),
            ),
            (
                "SZP mu=2".into(),
                Projection::from(Szp::from_pv(&[0.0, 2.0, 180.0, 60.0]).unwrap()),
            ),
        ];
        // One part in a million of the colatitude. The fixed code sits
        // at or below 1e-7 of it; the unfixed code lost the whole
        // quantity (relative error 1) by a milliarcsecond off the pole.
        let tol = 1e-6;
        for (label, p) in &cases {
            for k in 2..=8 {
                let colat = 10.0_f64.powi(-k);
                let theta = 90.0 - colat;
                for phi in [0.0_f64, 37.0, 123.0, -95.0] {
                    let Ok((x, y)) = p.s2x(phi, theta) else {
                        continue;
                    };
                    let (_, t2) = p.x2s(x, y).unwrap_or_else(|e| {
                        panic!("{label}: x2s failed after s2x accepted ({phi}, {theta}): {e}")
                    });
                    let rel = (t2 - theta).abs() / colat;
                    assert!(
                        rel < tol,
                        "{label}: colatitude {colat:e} deg at phi={phi} came back \
                         {:e}, a relative error of {rel:.3e}",
                        90.0 - t2
                    );
                }
            }
        }
    }

    /// `AZP` folds at `sin(theta) = -1/mu` for `|mu| > 1`: past it two
    /// latitudes share one radius and the inverse resolves to the
    /// branch holding the reference point.
    ///
    /// Regression: the forward step accepted the hidden branch, so
    /// the antipode of the reference point landed on the reference
    /// pixel itself. At `mu = 2`, both `theta = 90` and `theta = -90`
    /// give `R = 0`.
    #[test]
    fn azp_refuses_the_hidden_branch() {
        let azp = Azp::from_pv(&[0.0, 2.0, 0.0]).unwrap();
        // asin(-1/2) = -30 deg is the fold.
        assert!(azp.s2x(0.0, 90.0).is_ok(), "the reference point");
        assert!(azp.s2x(0.0, 0.0).is_ok(), "well inside");
        assert!(azp.s2x(0.0, -29.0).is_ok(), "just inside the fold");
        assert!(azp.s2x(0.0, -31.0).is_err(), "just past the fold");
        assert!(azp.s2x(0.0, -90.0).is_err(), "the antipode");
        // The antipode used to land exactly on the origin, colliding
        // with the reference point.
        let (rx, ry) = azp.s2x(0.0, 90.0).unwrap();
        assert!(rx.abs() < 1e-12 && ry.abs() < 1e-12);
        // `|mu| <= 1` has no fold inside [-90, 90]; STG (mu = 1) keeps
        // the whole sphere bar the south pole.
        let stg_like = Azp::from_pv(&[0.0, 1.0, 0.0]).unwrap();
        assert!(stg_like.s2x(0.0, -85.0).is_ok());
    }

    /// `SZP`'s hidden branch has no closed form, so the forward asks
    /// the same root-selection the inverse uses.
    ///
    /// Unlike `AZP`, the boundary depends on `phi` too -- the
    /// projection point is off-axis, so the horizon is not a parallel.
    #[test]
    fn szp_refuses_the_hidden_branch() {
        let szp = Szp::from_pv(&[0.0, 2.0, 180.0, 60.0]).unwrap();
        assert!(szp.s2x(0.0, 90.0).is_ok(), "the reference point");
        assert!(szp.s2x(0.0, -88.0).is_err(), "deep on the hidden branch");
        // The cut is genuinely phi-dependent: theta = -30 is hidden
        // along phi = 0 but visible along phi = 90.
        assert!(szp.s2x(0.0, -30.0).is_err());
        assert!(szp.s2x(90.0, -30.0).is_ok());
        // Whatever it accepts must invert -- the property that makes
        // the check worth having, pinned across the grid by
        // `zenithal_projections_round_trip`.
        let (x, y) = szp.s2x(90.0, -30.0).unwrap();
        let (_, back) = szp.x2s(x, y).unwrap();
        assert!((back + 30.0).abs() < 1e-9, "got {back}");
    }

    /// `AIR` folds too, for a far-southern `theta_b`.
    ///
    /// `R` normally rises without bound as `theta` falls. Below
    /// `theta_b` of about -76.5 degrees it turns over and rises again.
    /// One radius then answers to two latitudes, and the inverse cannot
    /// say which.
    ///
    /// Regression: the inverse bisected on `R`, which assumes the
    /// monotonicity the fold removes. The forward accepted the far
    /// branch. A `theta_b = -80` projection round-tripped `theta = -77`
    /// to `-18`, an error of 59 degrees, and reported nothing.
    #[test]
    fn air_refuses_the_fold() {
        // No fold: the whole domain stays invertible.
        for theta_b in [90.0_f64, 45.0, 0.0, -45.0, -70.0] {
            let air = Air::from_pv(&[0.0, theta_b]).unwrap();
            assert!(
                air.s2x(0.0, -89.0).is_ok(),
                "theta_b = {theta_b} should reach the far south"
            );
        }

        let air = Air::from_pv(&[0.0, -80.0]).unwrap();
        assert!(air.s2x(0.0, 90.0).is_ok(), "the reference point");
        assert!(air.s2x(0.0, 0.0).is_ok(), "well inside the branch");
        assert!(
            air.s2x(0.0, -77.0).is_err(),
            "theta = -77 is past the fold and used to round-trip to -18"
        );
        assert!(air.s2x(0.0, -89.0).is_err(), "deep past the fold");

        // The boundary is a real turning point, not an arbitrary cut.
        // Inside it the radius still grows, and the inverse recovers
        // the latitude it was given.
        let mut last = f64::NEG_INFINITY;
        let mut theta = 89.0_f64;
        while theta > -90.0 {
            match air.s2x(0.0, theta) {
                Ok((x, y)) => {
                    let r = (x * x + y * y).sqrt();
                    assert!(
                        r > last,
                        "R turned over at theta = {theta} while still accepted"
                    );
                    last = r;
                    let (_, back) = air.x2s(x, y).unwrap();
                    assert!(
                        (back - theta).abs() < 1e-7,
                        "accepted theta = {theta} came back {back}"
                    );
                }
                // Past the fold; everything below is refused too.
                Err(_) => break,
            }
            theta -= 0.05;
        }
        assert!(
            theta < 0.0,
            "the fold should sit south, found it at {theta}"
        );

        // A plane point beyond the largest radius the branch reaches
        // has no latitude on it.
        let (x, y) = air.s2x(0.0, theta + 0.05).unwrap();
        let r_max = (x * x + y * y).sqrt();
        assert!(
            air.x2s(r_max * 1.01, 0.0).is_err(),
            "past the branch maximum"
        );
    }

    /// The no-fold shortcut must never skip a real fold.
    ///
    /// `branch_end` returns the domain edge without searching when
    /// `cb <= NO_FOLD_CB`. If that bound were too close to zero the
    /// shortcut would hand back a domain `R` turns over inside, the
    /// forward would accept the far branch, and the inverse would
    /// answer with the near one -- silently, which is the failure
    /// `air_refuses_the_fold` describes.
    ///
    /// This sweeps `theta_b` across the threshold at a step far finer
    /// than the margin and asserts the property the bound stands for:
    /// over the domain the projection accepts, `R` rises the whole way.
    #[test]
    fn air_skips_the_scan_only_when_r_is_monotonic() {
        // Dense either side of the -76.4 degree threshold, plus the
        // range a real header uses.
        let mut theta_bs: Vec<f64> = vec![90.0, 60.0, 30.0, 0.0, -30.0, -60.0];
        for k in 0..400 {
            theta_bs.push(-70.0 - 0.025 * f64::from(k));
        }
        for theta_b in theta_bs {
            let air = Air::from_pv(&[0.0, theta_b]).unwrap();
            let mut last = f64::NEG_INFINITY;
            let mut theta = 89.9_f64;
            while theta > -89.9 {
                if let Ok((x, y)) = air.s2x(0.0, theta) {
                    let r = (x * x + y * y).sqrt();
                    assert!(
                        r > last,
                        "theta_b = {theta_b}: R turned over at theta = {theta} \
                         while still accepted (cb = {}, xi_max = {})",
                        air.cb,
                        air.xi_max
                    );
                    last = r;
                }
                theta -= 0.05;
            }
        }
    }

    /// `AIR` stays finite for a `theta_b` at the south pole.
    ///
    /// `from_pv` admits the whole closed range, so `PV2_1 = -90` is a
    /// value a header can carry. The projection is degenerate there --
    /// the point of minimum distortion is where `R` diverges -- but it
    /// has to report numbers rather than infinities.
    ///
    /// Regression: `ln_cos` read `ln(1 - sin^2 xi)`, and `sin(xi)`
    /// rounds to 1 once `cos(xi)` falls below about 2e-8. The identity
    /// then gave `-inf` for a positive cosine, `cb` became `-inf`, and
    /// `s2x` returned `Ok((NaN, -inf))` for every point on the sphere.
    #[test]
    fn air_stays_finite_at_the_southern_limit() {
        for theta_b in [-90.0_f64, -89.9999999, -89.999999, -89.99999] {
            let air = Air::from_pv(&[0.0, theta_b]).unwrap();
            assert!(
                air.cb.is_finite(),
                "theta_b = {theta_b}: cb = {} is not finite",
                air.cb
            );
            for theta in [89.0_f64, 45.0, 0.0] {
                let Ok((x, y)) = air.s2x(0.0, theta) else {
                    continue;
                };
                assert!(
                    x.is_finite() && y.is_finite(),
                    "theta_b = {theta_b}: s2x(0, {theta}) = ({x}, {y})"
                );
                let (_, back) = air.x2s(x, y).unwrap();
                assert!(
                    (back - theta).abs() < 1e-7,
                    "theta_b = {theta_b}: theta = {theta} came back {back}"
                );
            }
        }
    }

    /// `TAN` puts the reference point at the origin of the plane.
    #[test]
    fn tan_pole_is_origin() {
        let (x, y) = Tan.s2x(0.0, 90.0).unwrap();
        assert!(x.abs() < 1e-12 && y.abs() < 1e-12);
    }

    /// The slant `SIN` -- `xi`/`eta` non-zero -- is the radio
    /// interferometry case, and takes a different code path from the
    /// orthographic one the grid above walks.
    ///
    /// `eta = cot(delta_0)` is the NCP convention, so the slant runs
    /// from a fraction of a unit up to the large values a field near
    /// the equator produces.
    #[test]
    fn sin_slant_round_trip() {
        for &(xi, eta) in &[
            (0.05_f64, -0.03_f64),
            (0.0, 1.0),   // NCP at delta_0 = 45 deg
            (0.0, 1.732), // NCP at delta_0 = 30 deg
            (-0.4, 0.25),
            (2.0, -1.5),
        ] {
            round_trip_tol(
                &Sin { xi, eta }.into(),
                &format!("SIN xi={xi} eta={eta}"),
                1e-6,
            );
        }
    }

    /// `SIN` projects along `v = (xi, eta, 1)`, and that map is
    /// two-to-one: the hemisphere facing away from `v` lands on the
    /// same disc as the one facing it. `x2s` resolves the pair to the
    /// near branch, so `s2x` must refuse the far one.
    ///
    /// Regression: only the orthographic case (`xi = eta = 0`) was
    /// refused. A slant `SIN` accepted the whole sphere, and a
    /// far-side point round-tripped to its reflection -- at
    /// `xi = 0.05, eta = -0.03`, the point `(-179, -88)` came back as
    /// `(75.9, 84.1)`, an error of 174 degrees, reported as success.
    ///
    /// This asserts the domain *is* the visible hemisphere: the
    /// accepted set matches `P.v >= 0` point for point, so the check
    /// can neither admit a hidden point nor reject a visible one.
    #[test]
    fn sin_refuses_the_hidden_hemisphere() {
        for &(xi, eta) in &[
            (0.0_f64, 0.0_f64),
            (0.05, -0.03),
            (0.0, 1.732),
            (-0.4, 0.25),
            (2.0, -1.5),
        ] {
            let p = Sin { xi, eta };
            let (mut visible, mut hidden) = (0_usize, 0_usize);
            let mut theta = -89.5_f64;
            while theta <= 89.5 {
                let mut phi = -179.0_f64;
                while phi <= 180.0 {
                    let (c, s) = ((theta * D2R).cos(), (theta * D2R).sin());
                    let (sp, cp) = ((phi * D2R).sin(), (phi * D2R).cos());
                    // P.v with l = costheta sinphi, m = -costheta cosphi,
                    // n = sintheta.
                    let dot = xi * c * sp - eta * c * cp + s;
                    let accepted = p.s2x(phi, theta).is_ok();
                    // The limb (P.v == 0) is the boundary both branches
                    // share; a point within rounding of it may fall
                    // either way, and either answer is the same point.
                    if dot.abs() > 1e-12 {
                        assert_eq!(
                            accepted,
                            dot > 0.0,
                            "xi={xi} eta={eta}: ({phi}, {theta}) has P.v = {dot:e} \
                             but s2x {} it",
                            if accepted { "accepted" } else { "refused" }
                        );
                        if dot > 0.0 {
                            visible += 1;
                        } else {
                            hidden += 1;
                        }
                    }
                    phi += 3.0;
                }
                theta += 2.0;
            }
            // Both halves have to be populated, or the assertion above
            // is vacuous on one of them.
            assert!(
                visible > 100 && hidden > 100,
                "xi={xi} eta={eta}: {visible} visible, {hidden} hidden"
            );
        }
    }

    /// The orthographic case must be untouched by the general check:
    /// the two zero products cancel exactly, leaving `sintheta >= 0`.
    #[test]
    fn sin_orthographic_domain_is_the_northern_hemisphere() {
        let p = Sin { xi: 0.0, eta: 0.0 };
        for &phi in &[-179.0_f64, -90.0, 0.0, 90.0, 180.0] {
            assert!(p.s2x(phi, 0.0).is_ok(), "the limb itself");
            assert!(p.s2x(phi, 1e-12).is_ok(), "just inside");
            assert!(p.s2x(phi, -1e-12).is_err(), "just outside");
            assert!(p.s2x(phi, -45.0).is_err(), "well outside");
        }
    }

    /// With `xi = eta = 0` the slant formulation must reduce to the
    /// plain orthographic `R = cos(theta)`.
    #[test]
    fn sin_slant_zero_matches_simple() {
        let slant = Sin { xi: 0.0, eta: 0.0 };
        let (x, y) = slant.s2x(30.0, 50.0).unwrap();
        let t = 50.0_f64.to_radians();
        let p = 30.0_f64.to_radians();
        let r = R2D * t.cos();
        assert!((x - r * p.sin()).abs() < 1e-10 && (y - (-r * p.cos())).abs() < 1e-10);
    }

    /// `ZPN` with `PV2_1 = 1` and nothing else is `ARC` by definition
    /// (Paper II Sec.5.1.7), so the two must agree numerically.
    #[test]
    fn zpn_matches_arc_with_p1_only() {
        let p = Zpn::from_pv(&[0.0, 1.0]).unwrap();
        for &theta in &[-50.0_f64, 0.0, 30.0, 75.0] {
            let (x, y) = p.s2x(45.0, theta).unwrap();
            let (xr, yr) = Arc.s2x(45.0, theta).unwrap();
            assert!((x - xr).abs() < 1e-9 && (y - yr).abs() < 1e-9);
        }
    }

    /// A genuinely higher-order `ZPN`. The grid above only walks the
    /// degree-one polynomial, which never exercises the numeric root
    /// the inverse falls back on.
    #[test]
    fn zpn_round_trip_monotonic_polynomial() {
        round_trip_tol(
            &Zpn::from_pv(&[0.0, 1.0, 0.0, 0.05]).unwrap().into(),
            "ZPN cubic",
            1e-7,
        );
    }

    /// `AZP` with `mu = gamma = 0` is `TAN` (Paper II Sec.5.1.1).
    #[test]
    fn azp_zero_params_matches_tan() {
        let p = Azp::from_pv(&[0.0, 0.0, 0.0]).unwrap();
        let (x1, y1) = p.s2x(40.0, 60.0).unwrap();
        let (x2, y2) = Tan.s2x(40.0, 60.0).unwrap();
        assert!((x1 - x2).abs() < 1e-10 && (y1 - y2).abs() < 1e-10);
    }

    /// `SZP` with `mu = 0` is `TAN` too, for any projection point.
    #[test]
    fn szp_zero_params_matches_tan() {
        let szp = Szp::from_pv(&[0.0, 0.0, 0.0, 90.0]).unwrap();
        for &(phi, theta) in &[
            (0.0_f64, 90.0_f64),
            (45.0, 60.0),
            (-90.0, 30.0),
            (170.0, 5.0),
        ] {
            let (xs, ys) = szp.s2x(phi, theta).unwrap();
            let (xn, yn) = Tan.s2x(phi, theta).unwrap();
            assert!(
                (xs - xn).abs() < 1e-9 && (ys - yn).abs() < 1e-9,
                "SZP(mu=0) != TAN at ({phi},{theta})"
            );
        }
    }

    /// An off-axis projection point: the grid above holds `phi_c` at
    /// 180, where the horizon stays symmetric about the meridian.
    #[test]
    fn szp_round_trip_oblique_projection_point() {
        round_trip(
            &Szp::from_pv(&[0.0, 2.0, 30.0, 60.0]).unwrap().into(),
            "SZP phi_c=30",
        );
    }
}
