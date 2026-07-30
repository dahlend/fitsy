//! Zenithal (azimuthal) projections -- Paper II Sec.5.1.
//!
//! All nine members of this family have `theta_0 = 90deg` except for the
//! auxiliary formula helpers that are used privately below.

use std::f64::consts::PI;

use crate::error::{FitsError, Result};
use crate::wcs::projection::Projection;
use crate::wcs::{D2R, R2D};

// -- shared zenithal helpers ------------------------------------------

#[inline]
pub(super) fn zenithal_xy(r_deg: f64, phi_deg: f64) -> (f64, f64) {
    // Paper II eq. (12)-(13): x = R sin(phi), y = -R cos(phi).
    let phi = phi_deg * D2R;
    (r_deg * phi.sin(), -r_deg * phi.cos())
}

#[inline]
pub(super) fn zenithal_phi_r(x_deg: f64, y_deg: f64) -> (f64, f64) {
    // Paper II eq. (14)-(15): phi = atan2(x, -y); R = sqrt(x^2+y^2).
    let phi = x_deg.atan2(-y_deg) * R2D;
    let r = (x_deg * x_deg + y_deg * y_deg).sqrt();
    (phi, r)
}

// -- TAN --------------------------------------------------------------

/// Gnomonic / tangent-plane projection (Paper II Sec.5.1.4).
#[derive(Debug, Clone, Copy)]
pub struct Tan;
impl Projection for Tan {
    fn pv2(&self) -> Vec<(u32, f64)> {
        Vec::new()
    }

    fn theta0(&self) -> f64 {
        90.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
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
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
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
impl Projection for Stg {
    fn pv2(&self) -> Vec<(u32, f64)> {
        Vec::new()
    }

    fn theta0(&self) -> f64 {
        90.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
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
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
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
    /// Build from the latitude axis's `PV2_m` table, indexed by `m` and
    /// zero-filled where a card is absent.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let xi = pv2.get(1).copied().unwrap_or(0.0);
        let eta = pv2.get(2).copied().unwrap_or(0.0);
        Ok(Self { xi, eta })
    }
}
impl Projection for Sin {
    fn theta0(&self) -> f64 {
        90.0
    }
    fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.xi), (2, self.eta)]
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        // Paper II eq. (30): x = R_0[costheta*sinphi + xi(1-sintheta)],
        //                    y = -R_0[costheta*cosphi - eta(1-sintheta)].
        let t = theta * D2R;
        let p = phi * D2R;
        let s = t.sin();
        let c = t.cos();
        if self.xi == 0.0 && self.eta == 0.0 && s < 0.0 {
            return Err(FitsError::Wcs(
                "SIN: theta < 0 lies in the unprojected hemisphere".into(),
            ));
        }
        let one_minus_s = 1.0 - s;
        let x = R2D * (c * p.sin() + self.xi * one_minus_s);
        let y = -R2D * (c * p.cos() - self.eta * one_minus_s);
        Ok((x, y))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
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
    /// Trailing zeros are trimmed at construction.
    pub coeffs: Vec<f64>,
}
impl Zpn {
    /// Build from the latitude axis's `PV2_m` table, indexed by `m` and
    /// zero-filled where a card is absent.
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
impl Projection for Zpn {
    fn theta0(&self) -> f64 {
        90.0
    }
    fn pv2(&self) -> Vec<(u32, f64)> {
        // `coeffs[m]` is PV2_m directly. Trailing zeros are dropped;
        // the parser zero-fills, and `from_pv` rejects an all-zero
        // table, so at least one term always survives.
        let last = self.coeffs.iter().rposition(|c| *c != 0.0);
        match last {
            Some(n) => (0..=n).map(|m| (m as u32, self.coeffs[m])).collect(),
            None => Vec::new(),
        }
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let zeta = (90.0 - theta) * D2R;
        let r = R2D * self.eval(zeta);
        Ok(zenithal_xy(r, phi))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
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
    /// sphere centre in spherical radii.
    pub mu: f64,
    /// `PV2_2` -- `gamma`, the tilt of the projection plane, degrees.
    pub gamma: f64,
}
impl Azp {
    /// Build from the latitude axis's `PV2_m` table, indexed by `m` and
    /// zero-filled where a card is absent.
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
impl Projection for Azp {
    fn theta0(&self) -> f64 {
        90.0
    }
    fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.mu), (2, self.gamma)]
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
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
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
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
impl Projection for Arc {
    fn pv2(&self) -> Vec<(u32, f64)> {
        Vec::new()
    }

    fn theta0(&self) -> f64 {
        90.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        Ok(zenithal_xy(90.0 - theta, phi))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (phi, r) = zenithal_phi_r(x, y);
        Ok((phi, 90.0 - r))
    }
}

// -- ZEA --------------------------------------------------------------

/// Zenithal equal-area (Paper II Sec.5.1.8).
#[derive(Debug, Clone, Copy)]
pub struct Zea;
impl Projection for Zea {
    fn pv2(&self) -> Vec<(u32, f64)> {
        Vec::new()
    }

    fn theta0(&self) -> f64 {
        90.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let t = theta * D2R;
        let factor = (1.0 - t.sin()).max(0.0);
        let r = R2D * (2.0 * factor).sqrt();
        Ok(zenithal_xy(r, phi))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (phi, r) = zenithal_phi_r(x, y);
        let arg = r / (2.0 * R2D);
        if arg.abs() > 1.0 + 1e-9 {
            return Err(FitsError::Wcs("ZEA: outside disk".into()));
        }
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
    /// sphere centre in spherical radii.
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
    /// Build from the latitude axis's `PV2_m` table, indexed by `m` and
    /// zero-filled where a card is absent.
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
        let u1 = (-qb - disc) / (2.0 * qa);
        let u2 = (-qb + disc) / (2.0 * qa);
        let pick = |u: f64| (-1e-9..=2.0 + 1e-9).contains(&u);
        match (pick(u1), pick(u2)) {
            (true, true) => Ok(u1.min(u2)),
            (true, false) => Ok(u1),
            (false, true) => Ok(u2),
            _ => Err(FitsError::Wcs("SZP: no admissible root".into())),
        }
    }
}
impl Projection for Szp {
    fn theta0(&self) -> f64 {
        90.0
    }
    fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.mu), (2, self.phi_c), (3, self.theta_c)]
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let p = phi * D2R;
        let t = theta * D2R;
        let cos_t = t.cos();
        let one_ms = 1.0 - t.sin();
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
                (1.0 - selected).clamp(-1.0, 1.0).asin() * R2D,
            )));
        }
        Ok((x, y))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let big_x = x / R2D;
        let big_y = y / R2D;
        let (zp, xp, yp) = (self.zp, self.xp, self.yp);
        let u = self.select_u(big_x, big_y)?.clamp(0.0, 2.0);
        let s = 1.0 - u;
        let theta = s.clamp(-1.0, 1.0).asin() * R2D;
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
    /// minimised, degrees.
    pub theta_b: f64,
    cb: f64,
}
impl Air {
    /// Build from the latitude axis's `PV2_m` table, indexed by `m` and
    /// zero-filled where a card is absent.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let theta_b = pv2.get(1).copied().unwrap_or(90.0);
        if !(-90.0..=90.0).contains(&theta_b) {
            return Err(FitsError::Wcs(format!(
                "AIR: PV2_1 (theta_b) = {theta_b} outside [-90deg, 90deg]"
            )));
        }
        let xi_b = (90.0 - theta_b) * D2R / 2.0;
        let cb = if xi_b.abs() < 1e-8 {
            -0.5
        } else {
            xi_b.cos().ln() / xi_b.tan().powi(2)
        };
        Ok(Self { theta_b, cb })
    }
    fn r_of_theta(&self, theta_deg: f64) -> Option<f64> {
        let xi = (90.0 - theta_deg) * D2R / 2.0;
        if xi.abs() < 1e-12 {
            return Some(0.0);
        }
        let tan_xi = xi.tan();
        let cos_xi = xi.cos();
        if cos_xi <= 0.0 {
            return None;
        }
        Some(-2.0 * R2D * (cos_xi.ln() / tan_xi + tan_xi * self.cb))
    }
}
impl Projection for Air {
    fn theta0(&self) -> f64 {
        90.0
    }
    fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.theta_b)]
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        if theta <= -90.0 + 1e-12 {
            return Err(FitsError::Wcs("AIR: south pole maps to infinity".into()));
        }
        let r = self
            .r_of_theta(theta)
            .ok_or_else(|| FitsError::Wcs("AIR: invalid theta".into()))?;
        Ok(zenithal_xy(r, phi))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (phi, r) = zenithal_phi_r(x, y);
        if r < 1e-12 {
            return Ok((phi, 90.0));
        }
        // Bisect to get a robust bracket; then Newton-polish.
        let mut lo = -89.999;
        let mut hi = 89.999;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let r_m = self
                .r_of_theta(mid)
                .ok_or_else(|| FitsError::Wcs("AIR: invalid theta in bisection".into()))?;
            if r_m > r {
                lo = mid;
            } else {
                hi = mid;
            }
            if (hi - lo) < 1e-9 {
                break;
            }
        }
        let mut theta = 0.5 * (lo + hi);
        let h = 1e-6;
        for _ in 0..32 {
            let Some(r_t) = self.r_of_theta(theta) else {
                break;
            };
            let f = r_t - r;
            let r_tp = self
                .r_of_theta((theta + h).min(89.999_999))
                .ok_or_else(|| FitsError::Wcs("AIR: derivative failed".into()))?;
            let r_tm = self
                .r_of_theta((theta - h).max(-89.999_999))
                .ok_or_else(|| FitsError::Wcs("AIR: derivative failed".into()))?;
            let fp = (r_tp - r_tm) / (2.0 * h);
            if fp.abs() < 1e-15 {
                break;
            }
            let next = (theta - f / fp).clamp(-89.999_999, 89.999_999);
            let dt = theta - next;
            theta = next;
            if dt.abs() < 1e-12 {
                return Ok((phi, theta));
            }
        }
        Ok((phi, theta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a native grid through `s2x`/`x2s`. Points the
    /// forward refuses are skipped -- the contract is that whatever it
    /// *accepts* must come back.
    fn round_trip(p: &dyn Projection, label: &str) {
        round_trip_tol(p, label, 1e-9);
    }

    fn round_trip_tol(p: &dyn Projection, label: &str, tol: f64) {
        let (mut checked, mut worst) = (0_usize, 0.0_f64);
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
                    // Compare as unit vectors: phi is degenerate at the
                    // poles and wraps at +-180.
                    let v = |a: f64, b: f64| {
                        let (c, s) = ((b * D2R).cos(), (b * D2R).sin());
                        [c * (a * D2R).cos(), c * (a * D2R).sin(), s]
                    };
                    let (u, w) = (v(phi, theta), v(p2, t2));
                    let dot = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
                    let cr = [
                        u[1] * w[2] - u[2] * w[1],
                        u[2] * w[0] - u[0] * w[2],
                        u[0] * w[1] - u[1] * w[0],
                    ];
                    // atan2(|u x w|, u.w): stable near zero, where
                    // acos(u.w) would lose half the mantissa.
                    let sep = (cr[0].powi(2) + cr[1].powi(2) + cr[2].powi(2))
                        .sqrt()
                        .atan2(dot)
                        / D2R;
                    assert!(
                        sep < tol,
                        "{label}: ({phi}, {theta}) -> ({x}, {y}) -> ({p2}, {t2}), off by {sep:.3e} deg"
                    );
                    worst = worst.max(sep);
                    checked += 1;
                }
                phi += 7.0;
            }
            theta += 4.0;
        }
        assert!(checked > 100, "{label}: only {checked} points accepted");
    }

    #[test]
    fn zenithal_projections_round_trip() {
        round_trip(&Tan, "TAN");
        round_trip(&Stg, "STG");
        round_trip(&Arc, "ARC");
        round_trip(&Zea, "ZEA");
        // SIN's inverse is `theta = acos(R/R_0)`, whose derivative is
        // infinite at the limb (`theta = 0`, `R = R_0`). One ulp in
        // `R/R_0` there is ~sqrt(2e-16) rad ~ 9e-7 deg, so the limb --
        // which this grid lands on exactly -- cannot do better. Away
        // from it the error is 4e-10 deg by `theta = 0.001` and 3e-13
        // by `theta = 0.1`.
        round_trip_tol(&Sin::from_pv(&[0.0, 0.0, 0.0]).unwrap(), "SIN", 1e-6);
        round_trip(&Air::from_pv(&[0.0, 45.0]).unwrap(), "AIR");
        round_trip(&Zpn::from_pv(&[0.0, 1.0]).unwrap(), "ZPN");
        for mu in [0.0_f64, 0.5, 1.0, 2.0, 5.0] {
            for gamma in [0.0_f64, 30.0, -20.0] {
                round_trip(
                    &Azp::from_pv(&[0.0, mu, gamma]).unwrap(),
                    &format!("AZP mu={mu} gamma={gamma}"),
                );
            }
        }
        for mu in [0.0_f64, 1.0, 2.0] {
            round_trip(
                &Szp::from_pv(&[0.0, mu, 180.0, 60.0]).unwrap(),
                &format!("SZP mu={mu}"),
            );
        }
    }

    /// `AZP` folds at `sin(theta) = -1/mu` for `|mu| > 1`: past it two
    /// latitudes share one radius and the inverse resolves to the
    /// branch holding the reference point.
    ///
    /// Regression: the forward accepted the hidden branch, so the
    /// antipode of the reference point landed on the *reference pixel
    /// itself* -- at `mu = 2`, both `theta = 90` and `theta = -90` give
    /// `R = 0`. `wcslib` reports those points as invalid.
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
}
