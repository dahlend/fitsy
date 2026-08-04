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

//! Cylindrical projections -- Paper II Sec.5.2: CAR, CEA, MER, CYP.

use std::f64::consts::FRAC_PI_2;

use crate::error::{FitsError, Result};
use crate::wcs::{D2R, R2D};

// -- CAR --------------------------------------------------------------

/// Plate carree -- equirectangular (Paper II Sec.5.2.3).
#[derive(Debug, Clone, Copy)]
pub struct Car;
impl Car {
    /// No parameters, so the table is empty.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        Vec::new()
    }

    /// Reference native latitude, 0 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        0.0
    }
    /// Forward step. `CAR` is the identity in degrees.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The whole sphere projects.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        Ok((phi, theta))
    }
    /// Inverse step. `CAR` is the identity in degrees.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The whole plane inverts.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        Ok((x, y))
    }
}

// -- CEA --------------------------------------------------------------

/// Cylindrical equal-area (Paper II Sec.5.2.2). Parameter `lambda = PV2_1`
/// (default 1) is the squash factor.
#[derive(Debug, Clone, Copy)]
pub struct Cea {
    /// `PV2_1` -- `lambda`, the scaling of the latitude axis.
    ///
    /// Private; [`Self::from_pv`] is the only way in. It resolves
    /// `inv_lambda` at construction.
    lambda: f64,
    /// `1 / lambda`, so the forward step multiplies rather than
    /// divides per point.
    inv_lambda: f64,
}
impl Cea {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `PV2_1` (`lambda`) lies outside the
    /// half-open range 0 to 1.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let lambda = pv2.get(1).copied().unwrap_or(1.0);
        if lambda <= 0.0 || lambda > 1.0 {
            return Err(FitsError::Wcs(format!(
                "CEA: PV2_1 (lambda) = {lambda} out of (0,1]"
            )));
        }
        Ok(Self {
            lambda,
            inv_lambda: 1.0 / lambda,
        })
    }
}
impl Cea {
    /// `PV2_1` is `lambda`.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.lambda)]
    }

    /// Reference native latitude, 0 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        0.0
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The whole sphere projects, because
    /// `sin(theta)` is bounded.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        Ok((phi, R2D * (theta * D2R).sin() * self.inv_lambda))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `lambda * y` exceeds `R_0`. Such a
    /// point lies off the projected band and names no latitude.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let s = self.lambda * y * D2R;
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
impl Mer {
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
    /// [`FitsError::Wcs`] when `|theta|` reaches 90 degrees. The
    /// Mercator ordinate diverges at both poles.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        if theta.abs() >= 90.0 - 1e-12 {
            return Err(FitsError::Wcs("MER: |theta| -> 90deg diverges".into()));
        }
        Ok((phi, R2D * (FRAC_PI_2 / 2.0 + theta * D2R / 2.0).tan().ln()))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. Every ordinate names a latitude,
    /// because the Mercator inverse is a bounded arctangent.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        Ok((x, (2.0 * (y * D2R).exp().atan() - FRAC_PI_2) * R2D))
    }
}

// -- CYP --------------------------------------------------------------

/// Cylindrical perspective (Paper II Sec.5.2.1). Parameters `mu = PV2_1`
/// (default 1) and `lambda = PV2_2` (default sqrt2/2).
#[derive(Debug, Clone, Copy)]
pub struct Cyp {
    /// `PV2_1` -- `mu`, the projection point's distance from the
    /// cylinder axis in spherical radii.
    pub mu: f64,
    /// `PV2_2` -- `lambda`, the cylinder's radius in spherical radii.
    pub lambda: f64,
}
impl Cyp {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `PV2_1` (`mu`) plus `PV2_2`
    /// (`lambda`) is 0, which makes the projection singular.
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
impl Cyp {
    /// `PV2_1` is `mu` and `PV2_2` is `lambda`.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.mu), (2, self.lambda)]
    }

    /// Reference native latitude, 0 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        0.0
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `mu + cos(theta)` is 0. The projection
    /// point lies on the parallel there and the ordinate diverges.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
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
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the `asin` argument exceeds 1. The
    /// point then lies off the band that this `mu` and `lambda` fill.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let phi = x / self.lambda;
        let eta = y / (R2D * (self.mu + self.lambda));
        let arg = eta * self.mu / (eta * eta + 1.0).sqrt();
        if arg.abs() > 1.0 + 1e-9 {
            return Err(FitsError::Wcs("CYP: argument out of range".into()));
        }
        Ok((phi, (eta.atan2(1.0) + arg.clamp(-1.0, 1.0).asin()) * R2D))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wcs::projection::Projection;
    use crate::wcs::projection::testing::round_trip;

    /// Every cylindrical must invert whatever its forward accepts.
    /// `CYP` is walked at the `mu`/`lambda` pair Paper II Sec.5.2.1
    /// names as the gnomonic-like case.
    #[test]
    fn cylindrical_projections_round_trip() {
        let cases: Vec<(&str, Projection)> = vec![
            ("CAR", Projection::from(Car)),
            ("MER", Projection::from(Mer)),
            (
                "CEA",
                Projection::from(Cea::from_pv(&[0.0, 1.0]).unwrap()),
            ),
            (
                "CYP",
                Projection::from(Cyp {
                    mu: 1.0,
                    lambda: std::f64::consts::FRAC_1_SQRT_2,
                }),
            ),
        ];
        for (name, p) in &cases {
            round_trip(p, name);
        }
    }

    /// `CAR` is the identity on the native sphere (Paper II Sec.5.2.3):
    /// `x = phi`, `y = theta`, in degrees, exactly.
    #[test]
    fn car_identity() {
        let (x, y) = Car.s2x(42.0, -17.5).unwrap();
        assert_eq!((x, y), (42.0, -17.5));
    }
}
