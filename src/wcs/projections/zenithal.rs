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
    pub xi: f64,
    pub eta: f64,
}
impl Sin {
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
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        // Paper II eq. (17): x = R_0[costheta*sinphi - xi(1-sintheta)],
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
        let x = R2D * (c * p.sin() - self.xi * one_minus_s);
        let y = -R2D * (c * p.cos() - self.eta * one_minus_s);
        Ok((x, y))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        // Solve the quadratic in u = 1 - sintheta derived from
        //   (X + xiu)^2 + (-Y + etau)^2 = cos^2theta = u(2-u)
        // => (1 + xi^2 + eta^2)*u^2 + 2(Xxi - Yeta - 1)*u + (X^2 + Y^2) = 0.
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
        let b = 2.0 * (big_x * self.xi - big_y * self.eta - 1.0);
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
            (big_x + self.xi * u).atan2(-(big_y) + self.eta * u) * R2D
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
    pub coeffs: Vec<f64>,
}
impl Zpn {
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
    pub mu: f64,
    pub gamma: f64,
}
impl Azp {
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
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let p = phi * D2R;
        let t = theta * D2R;
        let g = self.gamma * D2R;
        let denom = self.mu + t.sin() + t.cos() * p.cos() * g.tan();
        if denom.abs() < 1e-15 {
            return Err(FitsError::Wcs("AZP: denominator vanishes".into()));
        }
        let r = R2D * (self.mu + 1.0) * t.cos() / denom;
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
    pub mu: f64,
    pub phi_c: f64,
    pub theta_c: f64,
    xp: f64,
    yp: f64,
    zp: f64,
}
impl Szp {
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
}
impl Projection for Szp {
    fn theta0(&self) -> f64 {
        90.0
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
        Ok((x, y))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let big_x = x / R2D;
        let big_y = y / R2D;
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
        let u = match (pick(u1), pick(u2)) {
            (true, true) => {
                if u1 <= u2 {
                    u1
                } else {
                    u2
                }
            }
            (true, false) => u1,
            (false, true) => u2,
            _ => return Err(FitsError::Wcs("SZP: no admissible root".into())),
        };
        let u = u.clamp(0.0, 2.0);
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
    pub theta_b: f64,
    cb: f64,
}
impl Air {
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
