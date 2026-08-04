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

//! Polyconic projections -- Paper II Sec.5.5: BON, PCO.

use std::f64::consts::FRAC_PI_2;

use crate::error::{FitsError, Result};
use crate::wcs::{D2R, R2D};

// -- BON --------------------------------------------------------------

/// Bonne's projection (Paper II Sec.5.5.1, eq. 31). Parameter `theta_1 =
/// PV2_1` is required and non-zero (`theta_1 = 0` degenerates to SFL).
#[derive(Debug, Clone, Copy)]
pub struct Bon {
    /// `PV2_1` -- `theta_1`, the reference latitude in degrees.
    pub theta_1: f64,
}
impl Bon {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `PV2_1` (`theta_1`) is 0, where the
    /// `SFL` projection applies instead, or when it exceeds 90
    /// degrees in magnitude.
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
        reason = "keeps the uniform &self receiver the enum dispatch expects"
    )]
    fn y0(&self) -> f64 {
        R2D / (self.theta_1 * D2R).tan() + self.theta_1
    }
}
impl Bon {
    /// `PV2_1` is `theta_1`, the standard parallel.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.theta_1)]
    }

    /// Reference native latitude, 0 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        0.0
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the parallel radius is 0. The parallel
    /// collapses to the cone apex there.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let r = self.y0() - theta;
        let cos_t = (theta * D2R).cos();
        if r.abs() < 1e-15 {
            return Err(FitsError::Wcs("BON: R_theta = 0".into()));
        }
        let a_rad = phi * cos_t / r;
        Ok((r * a_rad.sin(), -r * a_rad.cos() + self.y0()))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. A point at a pole returns a longitude
    /// of 0, because every longitude meets there.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
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
impl Pco {
    /// No parameters, so the table is empty.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        Vec::new()
    }

    /// Reference native latitude, 0 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        0.0
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The equator takes a closed form and
    /// every other parallel takes the cotangent form.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let p = phi * D2R;
        let t = theta * D2R;
        if t.abs() < 1e-12 {
            return Ok((phi, 0.0));
        }
        let cot_t = t.cos() / t.sin();
        let e = p * t.sin();
        Ok((R2D * cot_t * e.sin(), theta + R2D * cot_t * (1.0 - e.cos())))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// The latitude has no closed form. Newton iteration solves
    /// `G(theta) = tan(theta)(X^2 + (Y - theta)^2) - 2(Y - theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the iteration does not converge in 128
    /// steps. That is a point off the region the projection fills.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wcs::projection::testing::{round_trip, round_trip_tol};

    /// `BON` at `theta_1 = 45`, the parameter Paper II Sec.5.5.1 uses
    /// as its worked example.
    #[test]
    fn bon_round_trip() {
        round_trip(&Bon::from_pv(&[0.0, 45.0]).unwrap().into(), "BON");
    }

    /// `PCO`'s inverse solves a transcendental equation by iteration,
    /// so it lands near but not at machine precision.
    #[test]
    fn pco_round_trip() {
        round_trip_tol(&Pco.into(), "PCO", 1e-6);
    }

    /// The equator is the one parallel `PCO` maps to a straight line,
    /// and it maps to it exactly (Paper II Sec.5.5.2).
    #[test]
    fn pco_equator_is_straight() {
        for &phi in &[-170.0_f64, -50.0, 0.0, 50.0, 170.0] {
            let (x, y) = Pco.s2x(phi, 0.0).unwrap();
            assert!((x - phi).abs() < 1e-12 && y.abs() < 1e-12);
        }
    }
}
