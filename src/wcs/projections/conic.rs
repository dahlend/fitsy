//! Conic projections -- Paper II Sec.5.4: COP, COE, COD, COO.

use crate::error::{FitsError, Result};
use crate::wcs::projection::Projection;
use crate::wcs::{D2R, R2D};
use std::f64::consts::FRAC_PI_2;

// -- shared conic helpers ---------------------------------------------

/// Common state for a conic projection: `theta_a`, eta, and the derived
/// standard parallels `theta_1` = `theta_a` - eta, `theta_2` = `theta_a` + eta.
#[derive(Debug, Clone, Copy)]
pub(super) struct ConicBase {
    pub theta_a: f64,
    pub eta: f64,
    pub theta_1: f64,
    pub theta_2: f64,
}
impl ConicBase {
    pub(super) fn from_pv(pv2: &[f64]) -> Result<Self> {
        let theta_a = pv2
            .get(1)
            .copied()
            .ok_or_else(|| FitsError::Wcs("conic projection requires PV2_1 (theta_a)".into()))?;
        let eta = pv2.get(2).copied().unwrap_or(0.0);
        if theta_a.abs() > 90.0 {
            return Err(FitsError::Wcs(format!(
                "conic: |PV2_1 (theta_a)| = {} > 90deg",
                theta_a.abs()
            )));
        }
        let t1 = theta_a - eta;
        let t2 = theta_a + eta;
        if t1 < -90.0 || t2 > 90.0 {
            return Err(FitsError::Wcs(
                "conic: theta_a +/- eta falls outside [-90deg, 90deg]".into(),
            ));
        }
        Ok(Self {
            theta_a,
            eta,
            theta_1: t1,
            theta_2: t2,
        })
    }
}

/// Returns `(R_theta, phi)` from projection-plane `(x, y)` -- shared by all
/// four conic impls.
#[inline]
fn conic_inverse_xy(x: f64, y: f64, y0: f64, c: f64, theta_a: f64) -> (f64, f64) {
    let dy = y0 - y;
    let s = if theta_a >= 0.0 { 1.0 } else { -1.0 };
    let r = s * (x * x + dy * dy).sqrt();
    let phi = (s * x).atan2(s * dy) / c * R2D;
    (r, phi)
}

// -- COP --------------------------------------------------------------

/// Conic perspective (Paper II Sec.5.4.1, eq. 27).
#[derive(Debug, Clone, Copy)]
pub struct Cop(ConicBase);
impl Cop {
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let base = ConicBase::from_pv(pv2)?;
        if base.theta_a.abs() < 1e-12 {
            return Err(FitsError::Wcs("COP: theta_a = 0 is singular".into()));
        }
        Ok(Self(base))
    }
    fn c(&self) -> f64 {
        (self.0.theta_a * D2R).sin()
    }
    fn y0(&self) -> f64 {
        R2D * (self.0.eta * D2R).cos() / (self.0.theta_a * D2R).tan()
    }
}
impl Projection for Cop {
    fn theta0(&self) -> f64 {
        self.0.theta_a
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let ta = self.0.theta_a * D2R;
        let er = self.0.eta * D2R;
        let r = R2D * er.cos() * (1.0 / ta.tan() - ((theta - self.0.theta_a) * D2R).tan());
        let a = self.c() * phi * D2R;
        Ok((r * a.sin(), -r * a.cos() + self.y0()))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (r, phi) = conic_inverse_xy(x, y, self.y0(), self.c(), self.0.theta_a);
        let arg = 1.0 / (self.0.theta_a * D2R).tan() - r / (R2D * (self.0.eta * D2R).cos());
        Ok((phi, self.0.theta_a + arg.atan() * R2D))
    }
}

// -- COE --------------------------------------------------------------

/// Conic equal-area (Paper II Sec.5.4.2, eq. 28).
#[derive(Debug, Clone, Copy)]
pub struct Coe(ConicBase);
impl Coe {
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let base = ConicBase::from_pv(pv2)?;
        if base.theta_a.abs() < 1e-12 {
            return Err(FitsError::Wcs("COE: theta_a = 0 is singular".into()));
        }
        Ok(Self(base))
    }
    fn gamma(&self) -> f64 {
        (self.0.theta_1 * D2R).sin() + (self.0.theta_2 * D2R).sin()
    }
    fn c(&self) -> f64 {
        self.gamma() / 2.0
    }
    fn r_of(&self, theta: f64) -> f64 {
        let s1 = (self.0.theta_1 * D2R).sin();
        let s2 = (self.0.theta_2 * D2R).sin();
        let inside = (1.0 + s1 * s2 - self.gamma() * (theta * D2R).sin()).max(0.0);
        R2D * inside.sqrt() / self.c()
    }
    fn y0(&self) -> f64 {
        self.r_of(self.0.theta_a)
    }
}
impl Projection for Coe {
    fn theta0(&self) -> f64 {
        self.0.theta_a
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let r = self.r_of(theta);
        let a = self.c() * phi * D2R;
        Ok((r * a.sin(), -r * a.cos() + self.y0()))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (r, phi) = conic_inverse_xy(x, y, self.y0(), self.c(), self.0.theta_a);
        let s1 = (self.0.theta_1 * D2R).sin();
        let s2 = (self.0.theta_2 * D2R).sin();
        let g = self.gamma();
        if g.abs() < 1e-15 {
            return Err(FitsError::Wcs(
                "COE: gamma = 0 (theta_1 + theta_2 = 0)".into(),
            ));
        }
        let s_theta = (1.0 + s1 * s2 - (r * self.c() / R2D).powi(2)) / g;
        if s_theta.abs() > 1.0 + 1e-9 {
            return Err(FitsError::Wcs("COE: |sintheta| > 1 (outside range)".into()));
        }
        Ok((phi, s_theta.clamp(-1.0, 1.0).asin() * R2D))
    }
}

// -- COD --------------------------------------------------------------

/// Conic equidistant (Paper II Sec.5.4.3, eq. 29).
#[derive(Debug, Clone, Copy)]
pub struct Cod(ConicBase);
impl Cod {
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let base = ConicBase::from_pv(pv2)?;
        if base.theta_a.abs() < 1e-12 {
            return Err(FitsError::Wcs("COD: theta_a = 0 is singular".into()));
        }
        Ok(Self(base))
    }
    fn c(&self) -> f64 {
        let ta = self.0.theta_a * D2R;
        let er = self.0.eta * D2R;
        if er.abs() < 1e-12 {
            ta.sin()
        } else {
            ta.sin() * er.sin() / er
        }
    }
    fn y0(&self) -> f64 {
        let ta = self.0.theta_a * D2R;
        let er = self.0.eta * D2R;
        if er.abs() < 1e-12 {
            R2D / ta.tan()
        } else {
            self.0.eta * (er.cos() / er.sin()) / ta.tan()
        }
    }
}
impl Projection for Cod {
    fn theta0(&self) -> f64 {
        self.0.theta_a
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let r = self.y0() - (theta - self.0.theta_a);
        let a = self.c() * phi * D2R;
        Ok((r * a.sin(), -r * a.cos() + self.y0()))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (r, phi) = conic_inverse_xy(x, y, self.y0(), self.c(), self.0.theta_a);
        Ok((phi, self.0.theta_a + (self.y0() - r)))
    }
}

// -- COO --------------------------------------------------------------

/// Conic orthomorphic / Lambert conformal conic (Paper II Sec.5.4.4,
/// eq. 30).
#[derive(Debug, Clone, Copy)]
pub struct Coo(ConicBase);
impl Coo {
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let base = ConicBase::from_pv(pv2)?;
        if base.theta_a.abs() >= 90.0 - 1e-12 {
            return Err(FitsError::Wcs(
                "COO: |theta_a| = 90deg (cone degenerates to a plane)".into(),
            ));
        }
        Ok(Self(base))
    }
    fn c(&self) -> f64 {
        let t1 = self.0.theta_1 * D2R;
        let t2 = self.0.theta_2 * D2R;
        if self.0.eta.abs() < 1e-12 {
            (self.0.theta_a * D2R).sin()
        } else {
            (t2.cos() / t1.cos()).ln()
                / (((FRAC_PI_2 - t2) / 2.0).tan() / ((FRAC_PI_2 - t1) / 2.0).tan()).ln()
        }
    }
    fn psi(&self) -> f64 {
        let t1 = self.0.theta_1 * D2R;
        let c = self.c();
        R2D * t1.cos() / (c * ((FRAC_PI_2 - t1) / 2.0).tan().powf(c))
    }
    fn r_of(&self, theta_deg: f64) -> f64 {
        self.psi() * ((FRAC_PI_2 - theta_deg * D2R) / 2.0).tan().powf(self.c())
    }
    fn y0(&self) -> f64 {
        self.r_of(self.0.theta_a)
    }
}
impl Projection for Coo {
    fn theta0(&self) -> f64 {
        self.0.theta_a
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let r = self.r_of(theta);
        let a = self.c() * phi * D2R;
        Ok((r * a.sin(), -r * a.cos() + self.y0()))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (r, phi) = conic_inverse_xy(x, y, self.y0(), self.c(), self.0.theta_a);
        let psi = self.psi();
        if r == 0.0 {
            let theta = if self.c() > 0.0 { 90.0 } else { -90.0 };
            return Ok((phi, theta));
        }
        let ratio = r / psi;
        if ratio < 0.0 {
            return Err(FitsError::Wcs("COO: R/psi < 0".into()));
        }
        Ok((phi, 90.0 - 2.0 * ratio.powf(1.0 / self.c()).atan() * R2D))
    }
}
