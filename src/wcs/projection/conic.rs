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

//! Conic projections -- Paper II Sec.5.4: COP, COE, COD, COO.

use crate::error::{FitsError, Result};
use crate::wcs::{D2R, R2D};
use std::f64::consts::FRAC_PI_2;

// -- shared conic helpers ---------------------------------------------

/// Common state for a conic projection: `theta_a`, eta, and the derived
/// standard parallels `theta_1` = `theta_a` - eta, `theta_2` = `theta_a` + eta.
///
/// Each wrapper caches its own cone constants next to this at
/// construction, so the per-point bodies read fields instead of
/// re-deriving trigonometry. Nothing mutates a built projection.
#[derive(Debug, Clone, Copy)]
pub(super) struct ConicBase {
    /// `PV2_1`, the mid-latitude between the two standard parallels.
    pub theta_a: f64,
    /// `PV2_2`, the half-separation of the two standard parallels.
    pub eta: f64,
    /// The first standard parallel, `theta_a - eta`.
    pub theta_1: f64,
    /// The second standard parallel, `theta_a + eta`.
    pub theta_2: f64,
}
impl ConicBase {
    /// Build the shared conic state from the `PV2_m` table of the
    /// latitude axis. The table is indexed by `m` and holds 0 where a
    /// card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `PV2_1` (`theta_a`) exceeds 90 degrees
    /// in magnitude, or when `theta_a` plus or minus `PV2_2` (`eta`)
    /// leaves the range -90 to 90 degrees.
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
pub struct Cop {
    base: ConicBase,
    /// Cone constant `sin(theta_a)`.
    c: f64,
    /// Radius offset of the fiducial point.
    y0: f64,
    /// `1 / tan(theta_a)`.
    cot_ta: f64,
    /// `R_0 cos(eta)` in degrees.
    r2d_cos_eta: f64,
}
impl Cop {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in three cases:
    ///
    /// - `PV2_1` (`theta_a`) exceeds 90 degrees in magnitude.
    /// - `theta_a` plus or minus `PV2_2` (`eta`) leaves the range -90
    ///   to 90 degrees.
    /// - `theta_a` is 0, which makes the projection singular.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let base = ConicBase::from_pv(pv2)?;
        if base.theta_a.abs() < 1e-12 {
            return Err(FitsError::Wcs("COP: theta_a = 0 is singular".into()));
        }
        let ta = base.theta_a * D2R;
        let er = base.eta * D2R;
        Ok(Self {
            base,
            c: ta.sin(),
            y0: R2D * er.cos() / ta.tan(),
            cot_ta: 1.0 / ta.tan(),
            r2d_cos_eta: R2D * er.cos(),
        })
    }
}
impl Cop {
    /// `PV2_1` is `theta_a` and `PV2_2` is `eta`.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.base.theta_a), (2, self.base.eta)]
    }

    /// Reference native latitude, the standard parallel `theta_a`.
    pub(crate) fn theta0(&self) -> f64 {
        self.base.theta_a
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `|theta - theta_a|` reaches 90 degrees.
    /// The radius diverges there and folds back past it.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        // The `tan(theta - theta_a)` of Paper II eq. (27) diverges at
        // `|theta - theta_a| = 90`, and past it the projection folds
        // back over itself: two latitudes share one radius, and the
        // inverse cannot tell them apart. Returning a plane coordinate
        // there is a confidently wrong answer, so refuse -- `wcslib`
        // reports the same points as invalid.
        let dtheta = theta - self.base.theta_a;
        if dtheta.abs() >= 90.0 {
            return Err(FitsError::Wcs(format!(
                "COP: theta = {theta} is {dtheta} deg from theta_a = {}; the \
                 projection is defined only for |theta - theta_a| < 90",
                self.base.theta_a,
            )));
        }
        let r = self.r2d_cos_eta * (self.cot_ta - (dtheta * D2R).tan());
        let a = self.c * phi * D2R;
        Ok((r * a.sin(), -r * a.cos() + self.y0))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The arctangent accepts every radius,
    /// and `s2x` refuses the latitudes it cannot name.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (r, phi) = conic_inverse_xy(x, y, self.y0, self.c, self.base.theta_a);
        let arg = self.cot_ta - r / self.r2d_cos_eta;
        Ok((phi, self.base.theta_a + arg.atan() * R2D))
    }
}

// -- COE --------------------------------------------------------------

/// Conic equal-area (Paper II Sec.5.4.2, eq. 28).
#[derive(Debug, Clone, Copy)]
pub struct Coe {
    base: ConicBase,
    /// `sin(theta_1) + sin(theta_2)`.
    gamma: f64,
    /// Cone constant `gamma / 2`.
    c: f64,
    /// `1 + sin(theta_1) sin(theta_2)`.
    one_plus_s1s2: f64,
    /// Radius offset of the fiducial point.
    y0: f64,
}
impl Coe {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in three cases:
    ///
    /// - `PV2_1` (`theta_a`) exceeds 90 degrees in magnitude.
    /// - `theta_a` plus or minus `PV2_2` (`eta`) leaves the range -90
    ///   to 90 degrees.
    /// - `theta_a` is 0, which makes the projection singular.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let base = ConicBase::from_pv(pv2)?;
        if base.theta_a.abs() < 1e-12 {
            return Err(FitsError::Wcs("COE: theta_a = 0 is singular".into()));
        }
        let s1 = (base.theta_1 * D2R).sin();
        let s2 = (base.theta_2 * D2R).sin();
        let gamma = s1 + s2;
        let c = gamma / 2.0;
        let one_plus_s1s2 = 1.0 + s1 * s2;
        let mut coe = Self {
            base,
            gamma,
            c,
            one_plus_s1s2,
            y0: 0.0,
        };
        coe.y0 = coe.r_of(base.theta_a);
        Ok(coe)
    }
    fn r_of(&self, theta: f64) -> f64 {
        let inside = (self.one_plus_s1s2 - self.gamma * (theta * D2R).sin()).max(0.0);
        R2D * inside.sqrt() / self.c
    }
}
impl Coe {
    /// `PV2_1` is `theta_a` and `PV2_2` is `eta`.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.base.theta_a), (2, self.base.eta)]
    }

    /// Reference native latitude, the standard parallel `theta_a`.
    pub(crate) fn theta0(&self) -> f64 {
        self.base.theta_a
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The radius is a bounded square root.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let r = self.r_of(theta);
        let a = self.c * phi * D2R;
        Ok((r * a.sin(), -r * a.cos() + self.y0))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the recovered `|sin(theta)|` exceeds 1,
    /// which is a point outside the projected region. `gamma = 0` is
    /// also refused, though `from_pv` rejects that case first.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (r, phi) = conic_inverse_xy(x, y, self.y0, self.c, self.base.theta_a);
        // `gamma = 0` needs `theta_1 = -theta_2`, which `from_pv`
        // already rejects as `theta_a = 0`. Kept as a defense.
        if self.gamma.abs() < 1e-15 {
            return Err(FitsError::Wcs(
                "COE: gamma = 0 (theta_1 + theta_2 = 0)".into(),
            ));
        }
        let s_theta = (self.one_plus_s1s2 - (r * self.c / R2D).powi(2)) / self.gamma;
        if s_theta.abs() > 1.0 + 1e-9 {
            return Err(FitsError::Wcs("COE: |sintheta| > 1 (outside range)".into()));
        }
        Ok((phi, s_theta.clamp(-1.0, 1.0).asin() * R2D))
    }
}

// -- COD --------------------------------------------------------------

/// Conic equidistant (Paper II Sec.5.4.3, eq. 29).
#[derive(Debug, Clone, Copy)]
pub struct Cod {
    base: ConicBase,
    /// Cone constant.
    c: f64,
    /// Radius offset of the fiducial point.
    y0: f64,
}
impl Cod {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in three cases:
    ///
    /// - `PV2_1` (`theta_a`) exceeds 90 degrees in magnitude.
    /// - `theta_a` plus or minus `PV2_2` (`eta`) leaves the range -90
    ///   to 90 degrees.
    /// - `theta_a` is 0, which makes the projection singular.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let base = ConicBase::from_pv(pv2)?;
        if base.theta_a.abs() < 1e-12 {
            return Err(FitsError::Wcs("COD: theta_a = 0 is singular".into()));
        }
        let ta = base.theta_a * D2R;
        let er = base.eta * D2R;
        let (c, y0) = if er.abs() < 1e-12 {
            (ta.sin(), R2D / ta.tan())
        } else {
            (
                ta.sin() * er.sin() / er,
                base.eta * (er.cos() / er.sin()) / ta.tan(),
            )
        };
        Ok(Self { base, c, y0 })
    }
}
impl Cod {
    /// `PV2_1` is `theta_a` and `PV2_2` is `eta`.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.base.theta_a), (2, self.base.eta)]
    }

    /// Reference native latitude, the standard parallel `theta_a`.
    pub(crate) fn theta0(&self) -> f64 {
        self.base.theta_a
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The radius is linear in the latitude.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let r = self.y0 - (theta - self.base.theta_a);
        let a = self.c * phi * D2R;
        Ok((r * a.sin(), -r * a.cos() + self.y0))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The linear radius inverts everywhere.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (r, phi) = conic_inverse_xy(x, y, self.y0, self.c, self.base.theta_a);
        Ok((phi, self.base.theta_a + (self.y0 - r)))
    }
}

// -- COO --------------------------------------------------------------

/// Conic orthomorphic / Lambert conformal conic (Paper II Sec.5.4.4,
/// eq. 30).
#[derive(Debug, Clone, Copy)]
pub struct Coo {
    base: ConicBase,
    /// Cone constant.
    c: f64,
    /// `1 / c`, for the inverse `powf`.
    inv_c: f64,
    /// Radius scale `psi`.
    psi: f64,
    /// Radius offset of the fiducial point.
    y0: f64,
}
impl Coo {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in three cases:
    ///
    /// - `PV2_1` (`theta_a`) exceeds 90 degrees in magnitude.
    /// - `theta_a` plus or minus `PV2_2` (`eta`) leaves the range -90
    ///   to 90 degrees.
    /// - `|theta_a|` is exactly 90 degrees, where the cone becomes a
    ///   plane.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        let base = ConicBase::from_pv(pv2)?;
        if base.theta_a.abs() >= 90.0 - 1e-12 {
            return Err(FitsError::Wcs(
                "COO: |theta_a| = 90deg (cone degenerates to a plane)".into(),
            ));
        }
        let t1 = base.theta_1 * D2R;
        let t2 = base.theta_2 * D2R;
        let c = if base.eta.abs() < 1e-12 {
            (base.theta_a * D2R).sin()
        } else {
            (t2.cos() / t1.cos()).ln()
                / (((FRAC_PI_2 - t2) / 2.0).tan() / ((FRAC_PI_2 - t1) / 2.0).tan()).ln()
        };
        let psi = R2D * t1.cos() / (c * ((FRAC_PI_2 - t1) / 2.0).tan().powf(c));
        let mut coo = Self {
            base,
            c,
            inv_c: 1.0 / c,
            psi,
            y0: 0.0,
        };
        coo.y0 = coo.r_of(base.theta_a);
        Ok(coo)
    }
    fn r_of(&self, theta_deg: f64) -> f64 {
        self.psi * ((FRAC_PI_2 - theta_deg * D2R) / 2.0).tan().powf(self.c)
    }
}
impl Coo {
    /// `PV2_1` is `theta_a` and `PV2_2` is `eta`.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.base.theta_a), (2, self.base.eta)]
    }

    /// Reference native latitude, the standard parallel `theta_a`.
    pub(crate) fn theta0(&self) -> f64 {
        self.base.theta_a
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The radius is a bounded power.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let r = self.r_of(theta);
        let a = self.c * phi * D2R;
        Ok((r * a.sin(), -r * a.cos() + self.y0))
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `R / psi` is negative, which no real
    /// radius produces. A radius of 0 returns the pole the cone opens
    /// toward.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (r, phi) = conic_inverse_xy(x, y, self.y0, self.c, self.base.theta_a);
        if r == 0.0 {
            let theta = if self.c > 0.0 { 90.0 } else { -90.0 };
            return Ok((phi, theta));
        }
        let ratio = r / self.psi;
        if ratio < 0.0 {
            return Err(FitsError::Wcs("COO: R/psi < 0".into()));
        }
        Ok((phi, 90.0 - 2.0 * ratio.powf(self.inv_c).atan() * R2D))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wcs::projection::Projection;
    use crate::wcs::projection::testing::round_trip;

    /// `COP` is a perspective projection onto a cone, so its radius
    /// carries a `tan(theta - theta_a)` that diverges at 90 deg from
    /// the standard parallel and folds back beyond it.
    ///
    /// Regression: the forward step evaluated the tangent regardless,
    /// so a latitude past the divergence produced a plane coordinate
    /// whose inverse named a different latitude. That is a wrong
    /// answer reported as a right one.
    #[test]
    fn cop_refuses_beyond_the_divergence() {
        let cop = Cop::from_pv(&[0.0, 45.0, 25.0]).unwrap();
        assert!(cop.s2x(0.0, 45.0).is_ok(), "the standard parallel");
        assert!(cop.s2x(0.0, 90.0).is_ok(), "the pole is 45 deg away");
        assert!(cop.s2x(0.0, -44.0).is_ok(), "just inside");
        assert!(cop.s2x(0.0, -45.0).is_err(), "exactly at the divergence");
        assert!(cop.s2x(0.0, -88.0).is_err(), "well past it");
        // A negative standard parallel moves the window with it.
        let south = Cop::from_pv(&[0.0, -45.0, 25.0]).unwrap();
        assert!(south.s2x(0.0, -88.0).is_ok());
        assert!(south.s2x(0.0, 46.0).is_err());
    }

    /// Every conic must invert whatever its forward accepts.
    ///
    /// `eta = 0` collapses the two standard parallels onto one, which
    /// is a separate branch of each formulation, so each code is walked
    /// with and without it. A southern `theta_a` moves the cone's apex
    /// to the other hemisphere.
    #[test]
    fn conic_projections_round_trip() {
        let cases: Vec<(&str, Projection)> = vec![
            (
                "COP",
                Projection::from(Cop::from_pv(&[0.0, 45.0, 25.0]).unwrap()),
            ),
            (
                "COE",
                Projection::from(Coe::from_pv(&[0.0, 45.0, 25.0]).unwrap()),
            ),
            (
                "COD",
                Projection::from(Cod::from_pv(&[0.0, 45.0, 25.0]).unwrap()),
            ),
            (
                "COO",
                Projection::from(Coo::from_pv(&[0.0, 45.0, 25.0]).unwrap()),
            ),
            (
                "COP-",
                Projection::from(Cop::from_pv(&[0.0, -60.0, 15.0]).unwrap()),
            ),
            (
                "COE-",
                Projection::from(Coe::from_pv(&[0.0, -60.0, 15.0]).unwrap()),
            ),
            (
                "COP-S",
                Projection::from(Cop::from_pv(&[0.0, -30.0, 10.0]).unwrap()),
            ),
            (
                "COD eta=0",
                Projection::from(Cod::from_pv(&[0.0, 60.0, 0.0]).unwrap()),
            ),
            (
                "COO eta=0",
                Projection::from(Coo::from_pv(&[0.0, 30.0, 0.0]).unwrap()),
            ),
        ];
        for (name, p) in &cases {
            round_trip(p, name);
        }
    }
}
