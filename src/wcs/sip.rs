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

    /// Number of coefficient rows, capped at [`SIP_MAX_ORDER`].
    ///
    /// [`Self::from_terms`] rejects a higher order, but the fields are
    /// public, so a hand-built polynomial could exceed it. Clamping
    /// truncates such a polynomial instead of indexing past the power
    /// tables below.
    #[inline]
    fn dim(&self) -> usize {
        debug_assert!(
            self.order <= SIP_MAX_ORDER,
            "SIP order {} exceeds the maximum {SIP_MAX_ORDER}",
            self.order
        );
        (self.order.min(SIP_MAX_ORDER) as usize) + 1
    }

    /// Powers `1, x, x^2, ... x^SIP_MAX_ORDER`.
    ///
    /// Fixed-size because the order is capped: a `Vec` here would
    /// allocate twice per Newton iteration in [`Sip::inverse`].
    #[inline]
    fn powers(x: f64) -> [f64; SIP_MAX_ORDER as usize + 1] {
        let mut out = [1.0_f64; SIP_MAX_ORDER as usize + 1];
        for i in 1..out.len() {
            out[i] = out[i - 1] * x;
        }
        out
    }

    /// Evaluate `Sigma c_{p,q} * u^p * v^q`.
    #[must_use]
    pub fn eval(&self, u: f64, v: f64) -> f64 {
        let n = self.dim();
        let up = Self::powers(u);
        let vp = Self::powers(v);
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

    /// Evaluate the polynomial and both partial derivatives, as
    /// `(value, d/du, d/dv)`.
    ///
    /// Differentiating term by term is exact and costs one pass,
    /// where a central difference costs four extra evaluations and
    /// carries a step-size error.
    #[must_use]
    pub fn eval_with_derivatives(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let n = self.dim();
        let up = Self::powers(u);
        let vp = Self::powers(v);
        let (mut s, mut du, mut dv) = (0.0_f64, 0.0_f64, 0.0_f64);
        for p in 0..n {
            let row = p * n;
            let pmax = n - 1 - p;
            for q in 0..=pmax {
                let c = self.coeffs[row + q];
                if c == 0.0 {
                    continue;
                }
                s += c * up[p] * vp[q];
                if p > 0 {
                    du += c * (p as f64) * up[p - 1] * vp[q];
                }
                if q > 0 {
                    dv += c * (q as f64) * up[p] * vp[q - 1];
                }
            }
        }
        (s, du, dv)
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
        //
        // Tolerances scale with the coordinate magnitude. The residual
        // `r = F(u,v) - (u',v')` is formed by subtracting two numbers of
        // size ~|u'|,|v'|, so its smallest representable magnitude is the
        // rounding floor ~eps*|coord|. A fixed absolute tolerance (the old
        // 1e-13) is therefore unreachable once |coord| exceeds a few
        // hundred pixels -- e.g. WISE frames evaluated past the array edge,
        // where |u'|,|v'| ~ thousands give a floor ~5e-13. Newton finds the
        // correct root but then spends all 32 iterations bouncing on
        // rounding noise and spuriously reports non-convergence. A relative
        // tolerance tracks that floor while staying far below any
        // sub-pixel accuracy that matters (~1e-8 px even out at |coord|~1e3).
        let scale = 1.0 + up.abs() + vp.abs();
        let tol = 1e-11 * scale;
        for _ in 0..32 {
            // Residual and exact Jacobian of
            // F(u,v) = (u + A(u,v), v + B(u,v)) in one pass; the
            // identity part contributes the 1 on each diagonal.
            let (a, a_du, a_dv) = self.a.eval_with_derivatives(u, v);
            let (b, b_du, b_dv) = self.b.eval_with_derivatives(u, v);
            let rx = (u + a) - up;
            let ry = (v + b) - vp;
            if rx.abs() < tol && ry.abs() < tol {
                return Ok((u, v));
            }
            let j11 = 1.0 + a_du;
            let j12 = a_dv;
            let j21 = b_du;
            let j22 = 1.0 + b_dv;
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
            if du.abs() < tol && dv.abs() < tol {
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
    fn inverse_converges_at_large_coordinates() {
        // Regression: a fixed 1e-13 residual tolerance is below the
        // rounding floor (~eps*|coord|) once |u'|,|v'| reach thousands,
        // so the Newton inverse used to spuriously fail to converge far
        // from the reference pixel (e.g. WISE frames past the array edge).
        // The tolerance now scales with coordinate magnitude.
        let a = SipPoly::from_terms(2, &[(2, 0, 1e-7), (0, 2, -2e-7)]).unwrap();
        let b = SipPoly::from_terms(2, &[(1, 1, 5e-8)]).unwrap();
        let sip = Sip {
            a,
            b,
            ap: None,
            bp: None,
        };
        for &(u, v) in &[(3000.0_f64, -3500.0_f64), (-4000.0, 2500.0)] {
            let (up, vp) = sip.forward(u, v);
            let (ub, vb) = sip.inverse(up, vp).unwrap();
            // Sub-pixel accuracy at this scale (relative ~1e-11).
            assert!((ub - u).abs() < 1e-4, "u {u} -> {ub}");
            assert!((vb - v).abs() < 1e-4, "v {v} -> {vb}");
        }
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
