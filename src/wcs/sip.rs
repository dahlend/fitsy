// Polynomial evaluation reads more naturally with explicit (p, q)
// degree indices than with .iter().enumerate().
#![allow(
    clippy::needless_range_loop,
    reason = "polynomial evaluation is clearer with explicit (p, q) degree indices"
)]

//! SIP -- Simple Imaging Polynomial distortion.
//!
//! SIP is a FITS WCS convention that adds polynomial distortion in
//! pixel space, applied between subtracting `CRPIX` and applying the
//! linear (CD/PC) matrix:
//!
//! ```text
//! u = pix1 - CRPIX1
//! v = pix2 - CRPIX2
//! u' = u + Sigma A_{p,q} * u^p * v^q   (p + q <= A_ORDER, p + q >= 2)
//! v' = v + Sigma B_{p,q} * u^p * v^q   (p + q <= B_ORDER, p + q >= 2)
//! (xi, eta) = CD * (u', v')
//! ```
//!
//! The inverse uses the optional `AP_p_q`/`BP_p_q` polynomials when
//! present, or falls back to a Newton iteration on the forward
//! polynomial when they are not.
//!
//! Reference: Shupe et al., "The SIP Convention for Representing
//! Distortion in FITS Image Headers", ADASS XIV (2005).
//! <https://fits.gsfc.nasa.gov/registry/sip.html>
//!
//! ## Lenient handling of order-0 and order-1 terms (WISE quirk)
//!
//! Shupe (2005) Sec.3 reserves the polynomial sum to `p + q >= 2`: the
//! constant (`A_0_0`, `B_0_0`) and linear (`A_1_0`, `A_0_1`,
//! `B_1_0`, `B_0_1`) terms are forbidden because they are
//! algebraically absorbed into `CRPIX` and the `CD`/`PC` matrix.
//! Real-world WISE / NEOWISE Level-1b archive files (e.g.
//! `74721b067-w2-int-1b.fits`, where `A_0_0 ~= 0.81`) emit them
//! anyway as additional offsets, and astropy silently honours them.
//! This implementation matches that lenient behaviour: any
//! `A_p_q`/`B_p_q` with `p + q <= order` is accepted and added to
//! the polynomial without warning.

use crate::error::{FitsError, Result};

/// Maximum polynomial order accepted (matches WCSLIB / astropy).
pub const SIP_MAX_ORDER: u32 = 9;

/// One SIP polynomial: a triangular table of coefficients indexed
/// by `(p, q)` with `p + q <= order`. The coefficient layout is the
/// dense lower-triangle stored row-major by `p`.
#[derive(Debug, Clone)]
pub struct SipPoly {
    pub order: u32,
    /// `coeffs[p * (order+1) + q]`. Entries with `p + q > order`
    /// must remain zero.
    pub coeffs: Vec<f64>,
}

impl SipPoly {
    /// Build from a list of `(p, q, value)` tuples. The order is the
    /// maximum `p + q` seen; entries not provided are zero.
    pub fn from_terms(order: u32, terms: &[(u32, u32, f64)]) -> Result<Self> {
        if order > SIP_MAX_ORDER {
            return Err(FitsError::Wcs(format!(
                "SIP: order {order} exceeds maximum {SIP_MAX_ORDER}"
            )));
        }
        let n = (order as usize) + 1;
        let mut coeffs = vec![0.0_f64; n * n];
        for &(p, q, v) in terms {
            if p + q > order {
                return Err(FitsError::Wcs(format!(
                    "SIP: term ({p},{q}) exceeds declared order {order}"
                )));
            }
            coeffs[(p as usize) * n + (q as usize)] = v;
        }
        Ok(Self { order, coeffs })
    }

    /// Evaluate `Sigma c_{p,q} * u^p * v^q`.
    #[must_use]
    pub fn eval(&self, u: f64, v: f64) -> f64 {
        let n = (self.order as usize) + 1;
        // Pre-compute powers.
        let mut up = vec![1.0_f64; n];
        let mut vp = vec![1.0_f64; n];
        for i in 1..n {
            up[i] = up[i - 1] * u;
            vp[i] = vp[i - 1] * v;
        }
        let mut s = 0.0_f64;
        for p in 0..n {
            let row = p * n;
            let pmax = n - 1 - p;
            for q in 0..=pmax {
                let c = self.coeffs[row + q];
                if c != 0.0 {
                    s += c * up[p] * vp[q];
                }
            }
        }
        s
    }
}

/// Full SIP distortion table: forward `A`/`B` polynomials, plus
/// optional inverse `AP`/`BP` polynomials.
#[derive(Debug, Clone)]
pub struct Sip {
    pub a: SipPoly,
    pub b: SipPoly,
    pub ap: Option<SipPoly>,
    pub bp: Option<SipPoly>,
}

impl Sip {
    /// Forward distortion `(u, v) -> (u + f(u,v), v + g(u,v))`. The
    /// SIP convention adds the polynomial to the pixel offset; the
    /// constant + linear terms (`A_0_0`, `A_1_0`, `A_0_1` etc.) are
    /// part of the polynomial and applied additively per the spec.
    #[must_use]
    pub fn forward(&self, u: f64, v: f64) -> (f64, f64) {
        (u + self.a.eval(u, v), v + self.b.eval(u, v))
    }

    /// Inverse distortion `(u', v') -> (u, v)`. Uses `AP`/`BP` for an
    /// initial guess when available (the spec only requires AP/BP to
    /// be a "best-fit" inverse, typically accurate to ~10^-6 px), and
    /// always refines via Newton iteration on the exact forward map
    /// to converge to machine precision.
    pub fn inverse(&self, up: f64, vp: f64) -> Result<(f64, f64)> {
        // Initial guess: AP/BP if available, else identity.
        let (mut u, mut v) = if let (Some(ap), Some(bp)) = (&self.ap, &self.bp) {
            (up + ap.eval(up, vp), vp + bp.eval(up, vp))
        } else {
            (up, vp)
        };
        // Newton iteration on F(u, v) = (u + A(u,v), v + B(u,v)) - (u', v') = 0.
        let h = 1e-4_f64.max(1e-8 * (up.abs() + vp.abs() + 1.0));
        for _ in 0..32 {
            let (fu, fv) = self.forward(u, v);
            let rx = fu - up;
            let ry = fv - vp;
            if rx.abs() < 1e-13 && ry.abs() < 1e-13 {
                return Ok((u, v));
            }
            let (fup, fvp) = self.forward(u + h, v);
            let (fum, fvm) = self.forward(u - h, v);
            let (fupe, fvpe) = self.forward(u, v + h);
            let (fume, fvme) = self.forward(u, v - h);
            let j11 = (fup - fum) / (2.0 * h);
            let j21 = (fvp - fvm) / (2.0 * h);
            let j12 = (fupe - fume) / (2.0 * h);
            let j22 = (fvpe - fvme) / (2.0 * h);
            let det = j11 * j22 - j12 * j21;
            if det.abs() < 1e-15 {
                return Err(FitsError::Wcs(
                    "SIP: Jacobian singular during inverse iteration".into(),
                ));
            }
            let du = (j22 * rx - j12 * ry) / det;
            let dv = (-j21 * rx + j11 * ry) / det;
            u -= du;
            v -= dv;
            if du.abs() < 1e-13 && dv.abs() < 1e-13 {
                return Ok((u, v));
            }
        }
        Err(FitsError::Wcs(
            "SIP: inverse iteration did not converge".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly_eval_simple() {
        // p(u, v) = 0.1 + 0.2*u*v.
        let p = SipPoly::from_terms(2, &[(0, 0, 0.1), (1, 1, 0.2)]).unwrap();
        let r = p.eval(3.0, 5.0);
        assert!((r - (0.1 + 0.2 * 15.0)).abs() < 1e-12);
    }

    #[test]
    fn poly_rejects_overflow_term() {
        assert!(SipPoly::from_terms(2, &[(2, 1, 0.5)]).is_err());
    }

    #[test]
    fn forward_and_newton_inverse_round_trip() {
        // Small quadratic distortion in u; identity in v.
        let a = SipPoly::from_terms(2, &[(2, 0, 1e-4), (0, 2, -2e-4)]).unwrap();
        let b = SipPoly::from_terms(2, &[(1, 1, 5e-5)]).unwrap();
        let sip = Sip {
            a,
            b,
            ap: None,
            bp: None,
        };
        for &(u, v) in &[
            (0.0_f64, 0.0_f64),
            (10.0, -5.0),
            (-50.0, 30.0),
            (200.0, 150.0),
        ] {
            let (up, vp) = sip.forward(u, v);
            let (ub, vb) = sip.inverse(up, vp).unwrap();
            assert!((ub - u).abs() < 1e-7, "u {u} -> {ub}");
            assert!((vb - v).abs() < 1e-7, "v {v} -> {vb}");
        }
    }

    #[test]
    fn ap_bp_used_when_present() {
        // Linear shift: A(u, v) = 0.5 (constant). AP should give
        // -0.5 to invert.
        let a = SipPoly::from_terms(0, &[(0, 0, 0.5)]).unwrap();
        let b = SipPoly::from_terms(0, &[(0, 0, 0.0)]).unwrap();
        let ap = SipPoly::from_terms(0, &[(0, 0, -0.5)]).unwrap();
        let bp = SipPoly::from_terms(0, &[(0, 0, 0.0)]).unwrap();
        let sip = Sip {
            a,
            b,
            ap: Some(ap),
            bp: Some(bp),
        };
        let (up, vp) = sip.forward(10.0, 20.0);
        assert!((up - 10.5).abs() < 1e-12 && (vp - 20.0).abs() < 1e-12);
        let (u, v) = sip.inverse(up, vp).unwrap();
        assert!((u - 10.0).abs() < 1e-12 && (v - 20.0).abs() < 1e-12);
    }

    #[test]
    fn identity_when_all_zero() {
        let a = SipPoly::from_terms(0, &[]).unwrap();
        let b = SipPoly::from_terms(0, &[]).unwrap();
        let sip = Sip {
            a,
            b,
            ap: None,
            bp: None,
        };
        let (up, vp) = sip.forward(7.5, -3.5);
        assert!(up == 7.5 && vp == -3.5);
        let (u, v) = sip.inverse(up, vp).unwrap();
        assert!(u == 7.5 && v == -3.5);
    }
}
