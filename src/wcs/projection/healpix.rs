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

//! `HEALPix` projections -- Calabretta & Roukema 2007: HPX, XPH.

use crate::error::{FitsError, Result};
use crate::wcs::{D2R, R2D};

// -- HPX --------------------------------------------------------------

/// `HEALPix` grid projection `HPX` (Calabretta & Roukema 2007).
/// Parameters: `H = PV2_1` (default 4, integer >= 1) -- equatorial
/// facets; `K = PV2_2` (default 3, odd integer >= 1) -- polar facets.
#[derive(Debug, Clone, Copy)]
pub struct Hpx {
    /// `PV2_1` -- `H`, the number of equatorial facets. Default 4.
    pub h: f64,
    /// `PV2_2` -- `K`, the number of polar facets. Default 3.
    pub k: f64,
}
impl Hpx {
    /// Build from the `PV2_m` table of the latitude axis. The
    /// table is indexed by `m` and holds 0 where a card is absent.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `PV2_1` (`H`) or `PV2_2` (`K`) is not
    /// positive.
    pub fn from_pv(pv2: &[f64]) -> Result<Self> {
        // The parsed PV2 slice is zero-filled, so an absent card reads as
        // 0.0; H = 0 or K = 0 is meaningless, so treat it as "use default".
        let h = match pv2.get(1) {
            Some(&v) if v != 0.0 => v,
            _ => 4.0,
        };
        let k = match pv2.get(2) {
            Some(&v) if v != 0.0 => v,
            _ => 3.0,
        };
        if h < 0.0 || k < 0.0 {
            return Err(FitsError::Wcs("HPX: H and K must be positive".into()));
        }
        Ok(Self { h, k })
    }
    fn sin_theta_x(&self) -> f64 {
        (self.k - 1.0) / self.k
    }

    /// Center longitude of the polar facet holding `phi`.
    // The index `floor((phi + 180) H / 360)` runs `0..H-1` over
    // `(-180, 180]` except at the closed endpoint, where `phi = 180`
    // gives `H` and walks off the ring -- hence the clamp.
    fn facet_center(&self, phi: f64) -> f64 {
        let width = 360.0 / self.h;
        let idx = ((phi + 180.0) / width).floor().clamp(0.0, self.h - 1.0);
        idx * width - 180.0 + width / 2.0
    }
}
impl Hpx {
    /// `PV2_1` is `H` and `PV2_2` is `K`.
    pub(crate) fn pv2(&self) -> Vec<(u32, f64)> {
        vec![(1, self.h), (2, self.k)]
    }

    /// Reference native latitude, 0 degrees.
    pub(crate) fn theta0(&self) -> f64 {
        0.0
    }
    /// Forward step, native `(phi, theta)` to plane `(x, y)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. The equatorial zone and the polar zone
    /// each cover their own latitudes, and together they cover the
    /// sphere.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let s = (theta * D2R).sin();
        let stx = self.sin_theta_x();
        if s.abs() <= stx {
            Ok((phi, 90.0 * self.k * s / self.h))
        } else {
            // Polar zone, Calabretta & Roukema (2007) eqs. (35)-(37):
            //   sigma = sqrt(K(1 - |sin theta|)),
            //   x = phi_c + (phi - phi_c) sigma,
            //   |y| = (90/H)(K + 1 - 2 sigma).
            let abs_s = s.abs();
            let sigma = (self.k * (1.0 - abs_s)).sqrt();
            let h = self.h;
            let phi_c = self.facet_center(phi);
            let x = phi_c + (phi - phi_c) * sigma;
            let y_mag = 90.0 * (self.k + 1.0 - 2.0 * sigma) / h;
            Ok((x, if s >= 0.0 { y_mag } else { -y_mag }))
        }
    }
    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the polar-zone `sigma` is negative. Such
    /// a point lies beyond the tip of a polar facet.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let yt = self.h * y / 90.0;
        if yt.abs() <= self.k - 1.0 {
            let s = yt / self.k;
            let theta = s.clamp(-1.0, 1.0).asin() * R2D;
            Ok((x, theta))
        } else {
            // Polar zone inverse: sigma = (K + 1 - H|y|/90) / 2,
            // |sin theta| = 1 - sigma^2/K, phi = phi_c + (x - phi_c)/sigma.
            let abs_yt = yt.abs();
            let sigma = (self.k + 1.0 - abs_yt) / 2.0;
            if sigma < 0.0 {
                return Err(FitsError::Wcs("HPX: outside the projection".into()));
            }
            let sin_abs = (1.0 - sigma * sigma / self.k).clamp(-1.0, 1.0);
            let theta = if y >= 0.0 {
                sin_abs.asin() * R2D
            } else {
                -sin_abs.asin() * R2D
            };
            let half = 360.0 / self.h / 2.0;
            let phi_c = self.facet_center(x);
            // The polar facets are diamonds: at parameter sigma the facet
            // half-width is (180/H) sigma, so points outside that band are
            // not part of the projection (WCSLIB returns an error there too).
            if (x - phi_c).abs() > sigma * half + 1e-9 {
                return Err(FitsError::Wcs("HPX: outside the projection".into()));
            }
            let phi = if sigma < 1e-12 {
                phi_c
            } else {
                phi_c + (x - phi_c) / sigma
            };
            Ok((phi, theta))
        }
    }
}

// -- XPH --------------------------------------------------------------

/// Polar `HEALPix` `XPH` (Calabretta & Roukema 2007 Sec.6), also
/// called the butterfly projection.
///
/// This projection takes no `PV` parameter. Its output `(x, y)` is in
/// degrees, scaled by `1/sqrt(2)` relative to the underlying HPX facet
/// layout.
#[derive(Debug, Clone, Copy)]
pub struct Xph;
impl Xph {
    // Boundary: |sin theta| <= 2/3 => equatorial regime.
    const SINTHE_X: f64 = 2.0 / 3.0;
    // Pole-side tolerance for switching to the linearized sigma near theta=+/-90.
    const POLE_TOL: f64 = 1.0e-4;
}
impl Xph {
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
    /// [`FitsError::Wcs`] never. Every native point falls in one of the
    /// four quadrants of the butterfly layout.
    pub(crate) fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        // Normalize phi to [-180, 180), then build chi in [0, 360) and
        // psi = chi mod 90 in [0, 90) (local longitude within a facet).
        let mut chi = phi;
        if chi.abs() >= 180.0 {
            chi = chi.rem_euclid(360.0);
            if chi >= 180.0 {
                chi -= 360.0;
            }
        }
        chi += 180.0;
        let psi = chi.rem_euclid(90.0);
        // phi rounded back into [-180, 180)
        let phi_n = chi - 180.0;

        let sinthe = (theta * D2R).sin();
        let abssin = sinthe.abs();

        let (mut xi, mut eta) = if abssin <= Self::SINTHE_X {
            // Equatorial regime.
            (psi, 67.5 * sinthe)
        } else {
            // Polar regime. Use the linearized sigma very close to the pole
            // to avoid catastrophic cancellation in 1 - |sin theta|.
            let pole_lim = 90.0 - Self::POLE_TOL * (Self::SINTHE_X.sqrt() * R2D);
            let sigma = if theta.abs() < pole_lim {
                (3.0 * (1.0 - abssin)).sqrt()
            } else {
                (90.0 - theta.abs()) * (1.5_f64).sqrt() * D2R
            };
            let xi = 45.0 + (psi - 45.0) * sigma;
            let mut eta = 45.0 * (2.0 - sigma);
            if theta < 0.0 {
                eta = -eta;
            }
            (xi, eta)
        };

        xi -= 45.0;
        eta -= 90.0;

        // Pick the (x, y) quadrant from the rounded phi. Final scale
        // factor 1/sqrt2 matches WCSLIB's xphs2x.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let (x, y) = if phi_n < -90.0 {
            (s * (-xi + eta), s * (-xi - eta))
        } else if phi_n < 0.0 {
            (s * (xi + eta), s * (-xi + eta))
        } else if phi_n < 90.0 {
            (s * (xi - eta), s * (xi + eta))
        } else {
            (s * (-xi - eta), s * (xi - eta))
        };
        Ok((x, y))
    }

    /// Inverse step, plane `(x, y)` to native `(phi, theta)`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] never. Face disambiguation has a known limit
    /// near a facet edge, which an `#[ignore]`d test records.
    pub(crate) fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        // WCSLIB stores (x, y) scaled by 1/sqrt2; undo that here.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        // = x * sqrt2
        let xr = x / s;
        let yr = y / s;

        // Quadrant detection picks the facet base phi. The forward map is
        // x = s(xi - eta), y = s(xi + eta) (rotated per quadrant), so the
        // inverse carries a factor 1/2: xi = (xr + yr)/2, eta = (yr - xr)/2.
        #[allow(
            clippy::manual_midpoint,
            reason = "the +-(a +- b)/2 rotation pairs are clearer kept symmetric"
        )]
        let (xi1, eta1, mut phi) = if xr <= 0.0 && yr > 0.0 {
            ((-xr - yr) / 2.0, (xr - yr) / 2.0, -180.0_f64)
        } else if xr < 0.0 && yr <= 0.0 {
            ((xr - yr) / 2.0, (xr + yr) / 2.0, -90.0)
        } else if xr >= 0.0 && yr < 0.0 {
            ((xr + yr) / 2.0, (-xr + yr) / 2.0, 0.0)
        } else {
            ((-xr + yr) / 2.0, (-xr - yr) / 2.0, 90.0)
        };

        let xi = xi1 + 45.0;
        let eta = eta1 + 90.0;
        let abseta = eta.abs();
        if abseta > 90.0 {
            return Err(FitsError::Wcs("XPH: outside the projection".into()));
        }

        let theta = if abseta <= 45.0 {
            // Equatorial regime.
            phi += xi;
            (eta / 67.5).clamp(-1.0, 1.0).asin() * R2D
        } else {
            // Polar regime.
            let sigma = (90.0 - abseta) / 45.0;

            // Snap phi exactly on facet boundaries to avoid the 1/sigma blow-up.
            if xr == 0.0 {
                phi = if yr <= 0.0 { 0.0 } else { 180.0 };
            } else if yr == 0.0 {
                phi = if xr < 0.0 { -90.0 } else { 90.0 };
            } else {
                phi += 45.0 + xi1 / sigma;
            }

            let mut th = if sigma < Self::POLE_TOL {
                90.0 - sigma * (Self::SINTHE_X.sqrt() * R2D)
            } else {
                (1.0 - sigma * sigma / 3.0).clamp(-1.0, 1.0).asin() * R2D
            };
            if eta < 0.0 {
                th = -th;
            }
            th
        };

        // Wrap phi back into [-180, 180).
        let phi = ((phi + 180.0).rem_euclid(360.0)) - 180.0;
        Ok((phi, theta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wcs::projection::testing::{round_trip, round_trip_tol};

    /// `HPX` at the `H = 4`, `K = 3` grid of Calabretta & Roukema
    /// (2007), which is the sky-survey layout.
    #[test]
    fn hpx_round_trip() {
        round_trip(&Hpx::from_pv(&[0.0, 4.0, 3.0]).unwrap().into(), "HPX");
    }

    /// `XPH`'s inverse resolves the butterfly's face by comparison, so
    /// it does not reach the exact-arithmetic tolerance the equatorial
    /// `HPX` layout does.
    #[test]
    fn xph_round_trip() {
        round_trip_tol(&Xph.into(), "XPH", 1e-7);
    }

    /// `phi = +180` is the closed end of Paper II's `arg` range
    /// `(-180, 180]`: the facet index `floor((phi + 180) H / 360)`
    /// evaluates to `H` there and walked off the ring before the
    /// facet-center clamp -- at `H = 4`, `theta = 88` the projected
    /// `x` came out 223.08 instead of 136.92 (the CHANGELOG's worked
    /// example).
    ///
    /// The grid sweep steps `phi` by 7 from -179 and so never lands on
    /// the closed end; only this test reaches it.
    #[test]
    fn hpx_polar_facet_holds_at_phi_180() {
        let p = Hpx::from_pv(&[0.0, 4.0, 3.0]).unwrap();
        for &theta in &[60.0_f64, 88.0, -60.0, -88.0] {
            let (x, y) = p.s2x(180.0, theta).unwrap();
            assert!(
                (90.0..=180.0).contains(&x),
                "theta {theta}: x = {x} left the last facet"
            );
            let (phi2, theta2) = p.x2s(x, y).unwrap();
            let dphi = ((180.0 - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
            assert!(dphi.abs() < 1e-9, "HPX phi 180 -> {phi2} (theta {theta})");
            assert!((theta - theta2).abs() < 1e-9, "theta {theta} -> {theta2}");
        }
        let (x, _) = p.s2x(180.0, 88.0).unwrap();
        assert!((x - 136.92).abs() < 0.01, "x = {x}");
    }
}
