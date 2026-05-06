//! Polyconic projections -- Paper II Sec.5.5: BON, PCO.

use std::f64::consts::FRAC_PI_2;

use crate::error::{FitsError, Result};
use crate::wcs::projection::Projection;
use crate::wcs::{D2R, R2D};

// -- BON --------------------------------------------------------------

/// Bonne's projection (Paper II Sec.5.5.1, eq. 31). Parameter `theta_1 =
/// PV2_1` is required and non-zero (`theta_1 = 0` degenerates to SFL).
#[derive(Debug, Clone, Copy)]
pub struct Bon {
    pub theta_1: f64,
}
impl Bon {
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let theta_1 = pv2
            .get(1)
            .copied()
            .ok_or_else(|| FitsError::Wcs("BON: PV2_1 (theta_1) is required".into()))?;
        if theta_1.abs() < 1e-12 {
            return Err(FitsError::Wcs("BON: theta_1 = 0 -- use SFL instead".into()));
        }
        if theta_1.abs() > 90.0 {
            return Err(FitsError::Wcs(format!(
                "BON: |theta_1| = {} > 90deg",
                theta_1.abs()
            )));
        }
        Ok(Self { theta_1 })
    }
    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "aligns with Projection trait's &self convention"
    )]
    fn y0(&self) -> f64 {
        R2D / (self.theta_1 * D2R).tan() + self.theta_1
    }
}
impl Projection for Bon {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let r = self.y0() - theta;
        let cos_t = (theta * D2R).cos();
        if r.abs() < 1e-15 {
            return Err(FitsError::Wcs("BON: R_theta = 0".into()));
        }
        let a_rad = phi * cos_t / r;
        Ok((r * a_rad.sin(), -r * a_rad.cos() + self.y0()))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let s = if self.theta_1 >= 0.0 { 1.0 } else { -1.0 };
        let dy = self.y0() - y;
        let r = s * (x * x + dy * dy).sqrt();
        let theta = self.y0() - r;
        let cos_t = (theta * D2R).cos();
        if cos_t.abs() < 1e-15 {
            return Ok((0.0, theta));
        }
        let a_rad = (s * x).atan2(s * dy);
        Ok((a_rad * r / cos_t, theta))
    }
}

// -- PCO --------------------------------------------------------------

/// American polyconic `PCO` (Paper II Sec.5.5.2, eqs. 32-34).
#[derive(Debug, Clone, Copy)]
pub struct Pco;
impl Projection for Pco {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let p = phi * D2R;
        let t = theta * D2R;
        if t.abs() < 1e-12 {
            return Ok((phi, 0.0));
        }
        let cot_t = t.cos() / t.sin();
        let e = p * t.sin();
        Ok((R2D * cot_t * e.sin(), theta + R2D * cot_t * (1.0 - e.cos())))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        // Newton on G(theta) = tantheta*(X^2+(Y-theta)^2) - 2(Y-theta) = 0 with
        // X = x*pi/180, Y = y*pi/180.
        let big_x = x * D2R;
        let big_y = y * D2R;
        if big_x.abs() < 1e-15 && big_y.abs() < 1e-15 {
            return Ok((0.0, 0.0));
        }
        let mut theta = big_y.clamp(-FRAC_PI_2 + 1e-3, FRAC_PI_2 - 1e-3);
        let mut converged = false;
        for _ in 0..128 {
            let dy = big_y - theta;
            let r2 = big_x * big_x + dy * dy;
            let tan_t = theta.tan();
            let g = tan_t * r2 - 2.0 * dy;
            let cos_t = theta.cos();
            let gp = r2 / (cos_t * cos_t) - 2.0 * tan_t * dy + 2.0;
            if gp.abs() < 1e-15 {
                break;
            }
            let dt = g / gp;
            theta -= dt;
            if theta >= FRAC_PI_2 {
                theta = FRAC_PI_2 - 1e-9;
            } else if theta <= -FRAC_PI_2 {
                theta = -FRAC_PI_2 + 1e-9;
            }
            if dt.abs() < 1e-13 {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(FitsError::Wcs(
                "PCO: Newton iteration failed to converge".into(),
            ));
        }
        let theta_deg = theta * R2D;
        let phi = if theta.abs() < 1e-12 {
            x
        } else {
            let tan_t = theta.tan();
            let sin_e = big_x * tan_t;
            let cos_e = 1.0 - (big_y - theta) * tan_t;
            let e = sin_e.atan2(cos_e);
            (e / theta.sin()) * R2D
        };
        Ok((phi, theta_deg))
    }
}
