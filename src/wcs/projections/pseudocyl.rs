//! Pseudo-cylindrical and conventional projections -- Paper II Sec.5.3:
//! SFL, PAR, MOL, AIT.

use std::f64::consts::PI;

use crate::error::{FitsError, Result};
use crate::wcs::projection::Projection;
use crate::wcs::{D2R, R2D};

// -- SFL --------------------------------------------------------------

/// Sanson-Flamsteed (Paper II Sec.5.3.1).
#[derive(Debug, Clone, Copy)]
pub struct Sfl;
impl Projection for Sfl {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        Ok((phi * (theta * D2R).cos(), theta))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let c = (y * D2R).cos();
        if c == 0.0 {
            return Err(FitsError::Wcs("SFL: cos theta = 0 at the pole".into()));
        }
        Ok((x / c, y))
    }
}

// -- PAR --------------------------------------------------------------

/// Parabolic (Craster) projection (Paper II Sec.5.3.3).
#[derive(Debug, Clone, Copy)]
pub struct Par;
impl Projection for Par {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let t = theta * D2R;
        Ok((
            phi * (2.0 * (2.0 * t / 3.0).cos() - 1.0),
            180.0 * (t / 3.0).sin(),
        ))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        if y.abs() > 90.0 + 1e-9 {
            return Err(FitsError::Wcs("PAR: |y| > 90deg".into()));
        }
        let theta = 3.0 * (y / 180.0).clamp(-1.0, 1.0).asin() * R2D;
        let denom = 2.0 * (2.0 * theta * D2R / 3.0).cos() - 1.0;
        if denom == 0.0 {
            return Err(FitsError::Wcs("PAR: denominator = 0".into()));
        }
        Ok((x / denom, theta))
    }
}

// -- MOL --------------------------------------------------------------

/// Mollweide (Paper II Sec.5.3.2).
#[derive(Debug, Clone, Copy)]
pub struct Mol;
impl Projection for Mol {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        // Solve 2gamma + sin(2gamma) = pi*sin(theta) by Newton iteration.
        let target = PI * (theta * D2R).sin();
        let mut gamma = theta * D2R * 0.5;
        for _ in 0..32 {
            let f = 2.0 * gamma + (2.0 * gamma).sin() - target;
            let fp = 2.0 + 2.0 * (2.0 * gamma).cos();
            if fp.abs() < 1e-15 {
                break;
            }
            let dg = f / fp;
            gamma -= dg;
            if dg.abs() < 1e-13 {
                break;
            }
        }
        let sqrt2 = std::f64::consts::SQRT_2;
        let x = (2.0 * sqrt2 / PI) * phi * gamma.cos();
        let y = sqrt2 * R2D * gamma.sin();
        Ok((x, y))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let sqrt2 = std::f64::consts::SQRT_2;
        let s = y / (sqrt2 * R2D);
        if s.abs() > 1.0 + 1e-9 {
            return Err(FitsError::Wcs("MOL: |y| out of range".into()));
        }
        let gamma = s.clamp(-1.0, 1.0).asin();
        let theta = ((2.0 * gamma + (2.0 * gamma).sin()) / PI)
            .clamp(-1.0, 1.0)
            .asin()
            * R2D;
        let cos_g = gamma.cos();
        if cos_g.abs() < 1e-15 {
            return Ok((0.0, theta));
        }
        Ok((x * PI / (2.0 * sqrt2 * cos_g), theta))
    }
}

// -- AIT --------------------------------------------------------------

/// Hammer-Aitoff (Paper II Sec.5.3.4).
#[derive(Debug, Clone, Copy)]
pub struct Ait;
impl Projection for Ait {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let p = phi * D2R;
        let t = theta * D2R;
        let cos_t = t.cos();
        let denom = (0.5 * (1.0 + cos_t * (p / 2.0).cos())).sqrt();
        if denom == 0.0 {
            return Err(FitsError::Wcs("AIT: denominator = 0".into()));
        }
        let gamma = R2D / denom;
        Ok((2.0 * gamma * cos_t * (p / 2.0).sin(), gamma * t.sin()))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let z2 = 1.0 - (x / (4.0 * R2D)).powi(2) - (y / (2.0 * R2D)).powi(2);
        if z2 < 0.0 {
            return Err(FitsError::Wcs("AIT: outside the projection ellipse".into()));
        }
        let z = z2.sqrt();
        let theta = (y * z / R2D).clamp(-1.0, 1.0).asin() * R2D;
        let denom = 2.0 * z * z - 1.0;
        if denom == 0.0 {
            return Err(FitsError::Wcs("AIT: degenerate inversion".into()));
        }
        Ok((2.0 * (z * x / (2.0 * R2D)).atan2(denom) * R2D, theta))
    }
}
