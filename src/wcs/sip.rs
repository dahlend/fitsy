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
//! Some archive files emit them anyway, as additional offsets. A WISE
//! or NEOWISE Level-1b file does so, and one such file carries
//! `A_0_0` near 0.81. This module accepts any `A_p_q` or `B_p_q` with
//! `p + q <= order` and adds it to the polynomial.

use crate::error::{FitsError, Result};
use crate::wcs::newton;
use crate::wcs::poly;

/// Highest polynomial order that this module accepts.
pub const SIP_MAX_ORDER: u32 = 9;

/// One SIP polynomial: a triangular table of coefficients indexed
/// by `(p, q)` with `p + q <= order`. The coefficient layout is the
/// dense lower-triangle stored row-major by `p`.
#[derive(Debug, Clone)]
pub struct SipPoly {
    /// Highest total degree `p + q` the expansion carries.
    pub order: u32,
    /// `coeffs[p * (order+1) + q]`. Entries with `p + q > order`
    /// must remain zero.
    pub coeffs: Vec<f64>,
}

impl SipPoly {
    /// Build from a list of `(p, q, value)` tuples. A term that is
    /// absent from `terms` holds 0.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `order` exceeds the supported maximum,
    /// or when a term has `p + q` greater than `order`.
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

    /// Number of coefficient rows, and the row stride of `coeffs`.
    ///
    /// [`Self::from_terms`] sizes `coeffs` as `(order + 1)^2` and
    /// refuses an order above [`SIP_MAX_ORDER`]. The fields are public,
    /// so a hand-built polynomial can hold any size. This takes the
    /// largest square the coefficients hold. A stride other than the
    /// true `order + 1` reads the wrong terms.
    #[inline]
    fn dim(&self) -> usize {
        debug_assert!(
            self.order <= SIP_MAX_ORDER,
            "SIP order {} exceeds the maximum {SIP_MAX_ORDER}",
            self.order
        );
        ((self.order as usize) + 1).min(self.coeffs.len().isqrt())
    }

    /// Evaluate `Sigma c_{p,q} * u^p * v^q`.
    ///
    /// This uses Horner's scheme in both variables. Each row `p`
    /// collapses over `v`, then the rows collapse over `u`. The cost
    /// is one multiply-add per coefficient. No power of `u` or `v` is
    /// formed.
    #[must_use]
    pub fn eval(&self, u: f64, v: f64) -> f64 {
        let n = self.dim();
        poly::triangle(n, |p, q| self.coeffs[p * n + q], u, v)
    }

    /// Evaluate the polynomial and both partial derivatives, as
    /// `(value, d/du, d/dv)`.
    ///
    /// The derivatives follow the Horner recurrence of
    /// [`Self::eval`]. For a step `s <- s * z + c`, the derivative
    /// satisfies `d <- d * z + s`, with `s` taken from before the
    /// step.
    ///
    /// Differentiating term by term is exact and costs one pass. A
    /// central difference costs four extra evaluations and carries a
    /// step-size error. [`Sip::inverse`] needs the Jacobian once per
    /// Newton step.
    #[must_use]
    pub fn eval_with_derivatives(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let n = self.dim();
        poly::triangle_with_derivatives(n, |p, q| self.coeffs[p * n + q], u, v)
    }
}

/// Full SIP distortion table: forward `A`/`B` polynomials, plus
/// optional inverse `AP`/`BP` polynomials.
#[derive(Debug, Clone)]
pub struct Sip {
    /// `A_p_q` -- forward correction to the first pixel axis.
    pub a: SipPoly,
    /// `B_p_q` -- forward correction to the second pixel axis.
    pub b: SipPoly,
    /// `AP_p_q` -- the published inverse of `a`, when the header
    /// supplies one. Absent means the inverse is solved numerically.
    pub ap: Option<SipPoly>,
    /// `BP_p_q` -- the published inverse of `b`, when supplied.
    pub bp: Option<SipPoly>,
}

impl Sip {
    /// Forward distortion, `(u, v) -> (u + f(u,v), v + g(u,v))`.
    ///
    /// The SIP convention adds the polynomial to the pixel offset. The
    /// constant and linear terms, such as `A_0_0`, `A_1_0` and
    /// `A_0_1`, belong to the polynomial and add in the same way.
    #[must_use]
    pub fn forward(&self, u: f64, v: f64) -> (f64, f64) {
        // `A` and `B` are independent Horner recurrences over the same
        // `(u, v)`. Evaluated one after the other, they form two serial
        // dependency chains run back to back. This evaluation is bound
        // by that chain, not by the number of multiply-adds in it.
        // Walking both tables in one loop lets the two chains overlap.
        // That costs nothing extra and hides most of the second chain.
        //
        // The fused loop applies only when the two agree on shape.
        // `A_ORDER` and `B_ORDER` are separate keywords, and a header
        // may differ on them.
        let n = self.a.dim();
        if n != self.b.dim() {
            return (u + self.a.eval(u, v), v + self.b.eval(u, v));
        }
        // Bound once so the indexing below needs no per-term check.
        let (ac, bc) = (&self.a.coeffs[..n * n], &self.b.coeffs[..n * n]);
        let (mut sa, mut sb) = (0.0_f64, 0.0_f64);
        for p in (0..n).rev() {
            let row = p * n;
            let qmax = n - 1 - p;
            let (mut ra, mut rb) = (ac[row + qmax], bc[row + qmax]);
            for q in (0..qmax).rev() {
                ra = ra * v + ac[row + q];
                rb = rb * v + bc[row + q];
            }
            sa = sa * u + ra;
            sb = sb * u + rb;
        }
        (u + sa, v + sb)
    }

    /// `A` and `B` with their partial derivatives, in one pass.
    ///
    /// This is [`SipPoly::eval_with_derivatives`] on both polynomials,
    /// fused the way [`Self::forward`] fuses [`SipPoly::eval`], and for
    /// the same reason. Each polynomial carries three accumulators that
    /// depend on the step before, so a `(value, d/du, d/dv)` pass is
    /// three serial chains. Running `A` and `B` back to back makes six
    /// chains into two groups of three. Interleaving them leaves one
    /// group of six, which is wider than the dependency and fills the
    /// stalls the narrower version leaves.
    ///
    /// [`Self::inverse`] calls this once per Newton step, so it is the
    /// whole inner loop of the inverse.
    ///
    /// Each accumulator sees the same operations in the same order as
    /// the unfused version, so the results are identical bit for bit.
    /// `fused_derivatives_match_separate` holds that.
    #[inline]
    fn eval_both_with_derivatives(&self, u: f64, v: f64) -> ((f64, f64, f64), (f64, f64, f64)) {
        // As in `forward`: fuse only when `A_ORDER` and `B_ORDER` agree.
        let n = self.a.dim();
        if n != self.b.dim() {
            return (
                self.a.eval_with_derivatives(u, v),
                self.b.eval_with_derivatives(u, v),
            );
        }
        let (ac, bc) = (&self.a.coeffs[..n * n], &self.b.coeffs[..n * n]);
        let (mut sa, mut dua, mut dva) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (mut sb, mut dub, mut dvb) = (0.0_f64, 0.0_f64, 0.0_f64);
        for p in (0..n).rev() {
            let row = p * n;
            let qmax = n - 1 - p;
            let (mut ra, mut rb) = (ac[row + qmax], bc[row + qmax]);
            let (mut rda, mut rdb) = (0.0_f64, 0.0_f64);
            for q in (0..qmax).rev() {
                rda = rda * v + ra;
                ra = ra * v + ac[row + q];
                rdb = rdb * v + rb;
                rb = rb * v + bc[row + q];
            }
            // `du` consumes the previous `s`, so it updates first.
            dua = dua * u + sa;
            sa = sa * u + ra;
            dva = dva * u + rda;
            dub = dub * u + sb;
            sb = sb * u + rb;
            dvb = dvb * u + rdb;
        }
        ((sa, dua, dva), (sb, dub, dvb))
    }

    /// Inverse distortion, `(u', v') -> (u, v)`.
    ///
    /// The `AP` and `BP` polynomials give the initial guess when the
    /// header carries them. The convention requires those two to be a
    /// best-fit inverse alone, so this always refines the guess by
    /// Newton iteration on the exact forward map. The iteration
    /// converges to machine precision.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the Jacobian is singular, or when the
    /// iteration does not converge within its step limit.
    pub fn inverse(&self, up: f64, vp: f64) -> Result<(f64, f64)> {
        // Initial guess: AP/BP if available, else identity.
        let guess = if let (Some(ap), Some(bp)) = (&self.ap, &self.bp) {
            (up + ap.eval(up, vp), vp + bp.eval(up, vp))
        } else {
            (up, vp)
        };
        // Newton on F(u, v) = (u + A(u,v), v + B(u,v)) - (u', v') = 0.
        newton::solve("SIP", guess, newton::residual_scale(up, vp), |u, v| {
            // Residual and exact Jacobian in one pass; the identity
            // part of F contributes the 1 on each diagonal.
            let ((a, a_du, a_dv), (b, b_du, b_dv)) = self.eval_both_with_derivatives(u, v);
            newton::Residual2 {
                rx: (u + a) - up,
                ry: (v + b) - vp,
                j11: 1.0 + a_du,
                j12: a_dv,
                j21: b_du,
                j22: 1.0 + b_dv,
            }
        })
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

    /// The fused `A`/`B` derivative pass must equal the two separate
    /// passes bit for bit.
    ///
    /// `Sip::eval_both_with_derivatives` interleaves two Horner
    /// recurrences to let their dependency chains overlap. It is a
    /// scheduling change alone: each accumulator sees the same
    /// operations in the same order. Anything less than bit equality
    /// means an operation moved, and `Sip::inverse` would then converge
    /// to a different last digit than the polynomials say it should.
    ///
    /// The unequal-order case is covered too, since it takes the
    /// fallback rather than the fused loop.
    #[test]
    fn fused_derivatives_match_separate() {
        let full = |order: u32| {
            let terms: Vec<(u32, u32, f64)> = (0..=order)
                .flat_map(|p| (0..=(order - p)).map(move |q| (p, q)))
                .enumerate()
                .map(|(i, (p, q))| (p, q, ((i % 7) as f64 - 3.0) * 1e-4))
                .collect();
            SipPoly::from_terms(order, &terms).unwrap()
        };
        // Equal orders take the fused loop; unequal takes the fallback.
        for (oa, ob) in [(2_u32, 2_u32), (3, 3), (5, 5), (2, 4), (4, 2), (1, 1)] {
            let sip = Sip {
                a: full(oa),
                b: full(ob),
                ap: None,
                bp: None,
            };
            for &(u, v) in &[
                (0.0_f64, 0.0_f64),
                (1.0, -1.0),
                (37.5, -12.25),
                (-2048.0, 2047.0),
                (1e-9, 1e-9),
            ] {
                let want_a = sip.a.eval_with_derivatives(u, v);
                let want_b = sip.b.eval_with_derivatives(u, v);
                let (got_a, got_b) = sip.eval_both_with_derivatives(u, v);
                for (got, want, name) in [
                    (got_a.0, want_a.0, "A"),
                    (got_a.1, want_a.1, "dA/du"),
                    (got_a.2, want_a.2, "dA/dv"),
                    (got_b.0, want_b.0, "B"),
                    (got_b.1, want_b.1, "dB/du"),
                    (got_b.2, want_b.2, "dB/dv"),
                ] {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "orders ({oa}, {ob}) at ({u}, {v}): {name} fused {got} vs separate {want}"
                    );
                }
            }
        }
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
