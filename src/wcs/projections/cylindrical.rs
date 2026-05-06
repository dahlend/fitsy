//! Cylindrical projections -- Paper II Sec.5.2: CAR, CEA, MER, CYP.

use std::f64::consts::FRAC_PI_2;

use crate::error::{FitsError, Result};
use crate::wcs::projection::Projection;
use crate::wcs::{D2R, R2D};

// -- CAR --------------------------------------------------------------

/// Plate carree -- equirectangular (Paper II Sec.5.2.3).
#[derive(Debug, Clone, Copy)]
pub struct Car;
impl Projection for Car {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        Ok((phi, theta))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        Ok((x, y))
    }
}

// -- CEA --------------------------------------------------------------

/// Cylindrical equal-area (Paper II Sec.5.2.2). Parameter `lambda = PV2_1`
/// (default 1) is the squash factor.
#[derive(Debug, Clone, Copy)]
pub struct Cea {
    pub lambda: f64,
}
impl Cea {
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let lambda = pv2.get(1).copied().unwrap_or(1.0);
        if lambda <= 0.0 || lambda > 1.0 {
            return Err(FitsError::Wcs(format!(
                "CEA: PV2_1 (lambda) = {lambda} out of (0,1]"
            )));
        }
        Ok(Self { lambda })
    }
}
impl Projection for Cea {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        Ok((phi, R2D * (theta * D2R).sin() / self.lambda))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let s = self.lambda * y / R2D;
        if s.abs() > 1.0 + 1e-9 {
            return Err(FitsError::Wcs("CEA: |lambday| > R_0".into()));
        }
        Ok((x, s.clamp(-1.0, 1.0).asin() * R2D))
    }
}

// -- MER --------------------------------------------------------------

/// Mercator (Paper II Sec.5.2.4).
#[derive(Debug, Clone, Copy)]
pub struct Mer;
impl Projection for Mer {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        if theta.abs() >= 90.0 - 1e-12 {
            return Err(FitsError::Wcs("MER: |theta| -> 90deg diverges".into()));
        }
        Ok((phi, R2D * (FRAC_PI_2 / 2.0 + theta * D2R / 2.0).tan().ln()))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        Ok((x, (2.0 * (y * D2R).exp().atan() - FRAC_PI_2) * R2D))
    }
}

// -- CYP --------------------------------------------------------------

/// Cylindrical perspective (Paper II Sec.5.2.1). Parameters `mu = PV2_1`
/// (default 1) and `lambda = PV2_2` (default sqrt2/2).
#[derive(Debug, Clone, Copy)]
pub struct Cyp {
    pub mu: f64,
    pub lambda: f64,
}
impl Cyp {
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let mu = pv2.get(1).copied().unwrap_or(1.0);
        let lambda = pv2
            .get(2)
            .copied()
            .unwrap_or(std::f64::consts::FRAC_1_SQRT_2);
        if mu + lambda == 0.0 {
            return Err(FitsError::Wcs("CYP: mu + lambda = 0 is singular".into()));
        }
        Ok(Self { mu, lambda })
    }
}
impl Projection for Cyp {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let t = theta * D2R;
        let denom = self.mu + t.cos();
        if denom == 0.0 {
            return Err(FitsError::Wcs("CYP: mu + cos theta = 0 (diverges)".into()));
        }
        Ok((
            self.lambda * phi,
            R2D * (self.mu + self.lambda) * t.sin() / denom,
        ))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let phi = x / self.lambda;
        let eta = y / (R2D * (self.mu + self.lambda));
        let arg = eta * self.mu / (eta * eta + 1.0).sqrt();
        if arg.abs() > 1.0 + 1e-9 {
            return Err(FitsError::Wcs("CYP: argument out of range".into()));
        }
        Ok((phi, (eta.atan2(1.0) + arg.clamp(-1.0, 1.0).asin()) * R2D))
    }
}
