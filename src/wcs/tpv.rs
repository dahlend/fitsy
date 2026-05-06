//! TPV -- TAN with polynomial distortion.
//!
//! TPV is a non-standard but widely-used WCS convention (used by
//! SCAMP, SDSS, `DECam`, Pan-STARRS, ...). It augments the gnomonic
//! `TAN` projection with a polynomial in the intermediate world
//! coordinates `(xi, eta)` (in degrees):
//!
//! ```text
//! xi' = Sigma PV1_k * u_k(xi, eta)
//! eta' = Sigma PV2_k * v_k(eta, xi)
//! ```
//!
//! where `u_k`, `v_k` are 40 fixed monomials including radial terms
//! `r, r^3, r^5, r^7` with `r = sqrt(xi^2 + eta^2)`. The polynomial maps the
//! "raw" intermediate coordinates emitted by the linear pipeline to
//! the corrected coordinates that should be fed into the standard
//! TAN inverse projection.
//!
//! Note that PV2 swaps `xi <-> eta` in its linear terms (`PV2_1` multiplies
//! eta, `PV2_2` multiplies xi). Defaults are `PV1_1 = PV2_1 = 1`,
//! everything else 0.
//!
//! Reference: <https://fits.gsfc.nasa.gov/registry/tpvwcs/tpv.html>

use crate::error::{FitsError, Result};

/// Number of TPV polynomial coefficients per axis. PV0 through PV39
/// (PV38 corresponds to `r^7` and PV39 to `xi*eta^7` per the registry's
/// table; PV3, PV11, PV23, PV39 are the radial terms).
pub const TPV_NCOEFFS: usize = 40;

/// Per-axis TPV coefficient table. Defaults: PV*_1 = 1, others 0.
#[derive(Debug, Clone, Copy)]
pub struct TpvAxis {
    /// 0 -> axis 1 (xi -> xi'), 1 -> axis 2 (eta -> eta').
    pub axis: u8,
    /// 40 polynomial coefficients indexed by `m`.
    pub coeffs: [f64; TPV_NCOEFFS],
}

impl TpvAxis {
    /// Construct from a slice of `(m, value)` pairs (e.g. as parsed
    /// from `PVi_m` cards). Validates `m` is in `0..40`.
    pub fn from_pv_pairs(axis: u8, pairs: &[(u32, f64)]) -> Result<Self> {
        if axis != 1 && axis != 2 {
            return Err(FitsError::Wcs(format!(
                "TPV: axis must be 1 or 2 (got {axis})"
            )));
        }
        let mut coeffs = [0.0_f64; TPV_NCOEFFS];
        // PV*_1 default = 1 (identity scaling). Explicitly setting
        // PV*_1 in the header simply overwrites the default below.
        coeffs[1] = 1.0;
        for &(m, v) in pairs {
            if (m as usize) >= TPV_NCOEFFS {
                return Err(FitsError::Wcs(format!(
                    "TPV: PV{axis}_{m} exceeds the 40-coefficient table"
                )));
            }
            coeffs[m as usize] = v;
        }
        Ok(Self { axis, coeffs })
    }

    /// Evaluate the per-axis polynomial. For axis 1 the linear
    /// arguments are `(xi, eta)`; for axis 2 they are swapped to
    /// `(eta, xi)` per the TPV specification.
    #[must_use]
    pub fn eval(&self, xi: f64, eta: f64) -> f64 {
        // For axis 2, internal "x" becomes eta and "y" becomes xi.
        let (x, y) = if self.axis == 1 { (xi, eta) } else { (eta, xi) };
        let r = (x * x + y * y).sqrt();
        let c = &self.coeffs;
        // The 40 monomials per the TPV registry table.
        // (Indices correspond to PVi_<m>.)
        let r3 = r * r * r;
        let r5 = r3 * r * r;
        let r7 = r5 * r * r;
        let mut s = c[0]
            + c[1] * x
            + c[2] * y
            + c[3] * r
            + c[4] * x * x
            + c[5] * x * y
            + c[6] * y * y
            + c[7] * x * x * x
            + c[8] * x * x * y
            + c[9] * x * y * y
            + c[10] * y * y * y
            + c[11] * r3;
        s += c[12] * x.powi(4)
            + c[13] * x.powi(3) * y
            + c[14] * x.powi(2) * y.powi(2)
            + c[15] * x * y.powi(3)
            + c[16] * y.powi(4);
        s += c[17] * x.powi(5)
            + c[18] * x.powi(4) * y
            + c[19] * x.powi(3) * y.powi(2)
            + c[20] * x.powi(2) * y.powi(3)
            + c[21] * x * y.powi(4)
            + c[22] * y.powi(5)
            + c[23] * r5;
        s += c[24] * x.powi(6)
            + c[25] * x.powi(5) * y
            + c[26] * x.powi(4) * y.powi(2)
            + c[27] * x.powi(3) * y.powi(3)
            + c[28] * x.powi(2) * y.powi(4)
            + c[29] * x * y.powi(5)
            + c[30] * y.powi(6);
        s += c[31] * x.powi(7)
            + c[32] * x.powi(6) * y
            + c[33] * x.powi(5) * y.powi(2)
            + c[34] * x.powi(4) * y.powi(3)
            + c[35] * x.powi(3) * y.powi(4)
            + c[36] * x.powi(2) * y.powi(5)
            + c[37] * x * y.powi(6)
            + c[38] * y.powi(7)
            + c[39] * r7;
        s
    }
}

/// Pair of TPV polynomials (axis 1 + axis 2). Applied to the raw
/// intermediate world coordinates produced by the linear stage,
/// before the standard TAN projection is inverted.
#[derive(Debug, Clone, Copy)]
pub struct Tpv {
    pub pv1: TpvAxis,
    pub pv2: TpvAxis,
}

impl Tpv {
    /// Apply the forward distortion: `(xi, eta) -> (xi', eta')`.
    #[must_use]
    pub fn forward(&self, xi: f64, eta: f64) -> (f64, f64) {
        (self.pv1.eval(xi, eta), self.pv2.eval(xi, eta))
    }

    /// Apply the inverse distortion `(xi', eta') -> (xi, eta)` by Newton
    /// iteration. Converges quickly for the small-distortion regime
    /// typical of real instruments (<~ 1 pixel).
    pub fn inverse(&self, xi_p: f64, eta_p: f64) -> Result<(f64, f64)> {
        // Initial guess: undistorted = distorted (good when the
        // polynomial is close to identity).
        let mut xi = xi_p;
        let mut eta = eta_p;
        // Numerical Jacobian via central differences in degrees.
        let h = 1e-6_f64.max(1e-10 * (xi_p.abs() + eta_p.abs() + 1.0));
        for _ in 0..32 {
            let (fx, fy) = self.forward(xi, eta);
            let rx = fx - xi_p;
            let ry = fy - eta_p;
            if rx.abs() < 1e-13 && ry.abs() < 1e-13 {
                return Ok((xi, eta));
            }
            let (fxp, fyp) = self.forward(xi + h, eta);
            let (fxm, fym) = self.forward(xi - h, eta);
            let (fxpe, fype) = self.forward(xi, eta + h);
            let (fxme, fyme) = self.forward(xi, eta - h);
            let j11 = (fxp - fxm) / (2.0 * h);
            let j21 = (fyp - fym) / (2.0 * h);
            let j12 = (fxpe - fxme) / (2.0 * h);
            let j22 = (fype - fyme) / (2.0 * h);
            let det = j11 * j22 - j12 * j21;
            if det.abs() < 1e-15 {
                return Err(FitsError::Wcs(
                    "TPV: Jacobian singular during inverse iteration".into(),
                ));
            }
            // Solve J * delta = r, then xi -= delta.
            let dx = (j22 * rx - j12 * ry) / det;
            let dy = (-j21 * rx + j11 * ry) / det;
            xi -= dx;
            eta -= dy;
            if dx.abs() < 1e-13 && dy.abs() < 1e-13 {
                return Ok((xi, eta));
            }
        }
        Err(FitsError::Wcs(
            "TPV: inverse iteration did not converge".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_default() {
        let pv1 = TpvAxis::from_pv_pairs(1, &[]).unwrap();
        let pv2 = TpvAxis::from_pv_pairs(2, &[]).unwrap();
        let tpv = Tpv { pv1, pv2 };
        let (a, b) = tpv.forward(0.5, -0.3);
        // Default identity: xi' = xi, eta' = eta.
        assert!((a - 0.5).abs() < 1e-12);
        assert!((b - (-0.3)).abs() < 1e-12);
    }

    #[test]
    fn radial_distortion_round_trip() {
        // Small radial term: xi' = xi + 0.001*r, eta' = eta + 0.001*r.
        let pv1 = TpvAxis::from_pv_pairs(1, &[(3, 0.001)]).unwrap();
        let pv2 = TpvAxis::from_pv_pairs(2, &[(3, 0.001)]).unwrap();
        let tpv = Tpv { pv1, pv2 };
        for &(xi, eta) in &[(0.0_f64, 0.0_f64), (0.1, 0.05), (-0.2, 0.3), (0.4, -0.4)] {
            let (xp, yp) = tpv.forward(xi, eta);
            let (xb, yb) = tpv.inverse(xp, yp).unwrap();
            assert!((xb - xi).abs() < 1e-10, "xi {xi} -> {xb}");
            assert!((yb - eta).abs() < 1e-10, "eta {eta} -> {yb}");
        }
    }

    #[test]
    fn axis2_swaps_xi_eta() {
        // PV2_1 multiplies eta (not xi) per spec.
        let pv1 = TpvAxis::from_pv_pairs(1, &[]).unwrap();
        let pv2 = TpvAxis::from_pv_pairs(2, &[(0, 0.0), (1, 2.0)]).unwrap();
        let tpv = Tpv { pv1, pv2 };
        let (_, yp) = tpv.forward(0.0, 0.5);
        // eta' = PV2_0 + PV2_1 * eta = 0 + 2 * 0.5 = 1.0.
        assert!((yp - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_out_of_range_index() {
        assert!(TpvAxis::from_pv_pairs(1, &[(40, 0.5)]).is_err());
    }
}
