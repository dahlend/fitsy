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

/// Registry index of the coefficient multiplying `x^p y^q`, as
/// `XY_INDEX[p][q]`.
///
/// Setting the radial terms aside, the 40 monomials are every `x^p y^q`
/// with `p + q <= 7`. That is a triangle, and a triangle evaluates by
/// Horner's scheme in both variables, exactly as `SipPoly` does. This
/// maps the triangle onto the flat order the registry publishes.
///
/// Entries with `p + q > 7` are never read. The loops below bound `q`
/// by `deg - p`, and `deg` never exceeds 7.
const XY_INDEX: [[usize; 8]; 8] = [
    [0, 2, 6, 10, 16, 22, 30, 38],
    [1, 5, 9, 15, 21, 29, 37, 0],
    [4, 8, 14, 20, 28, 36, 0, 0],
    [7, 13, 19, 27, 35, 0, 0, 0],
    [12, 18, 26, 34, 0, 0, 0, 0],
    [17, 25, 33, 0, 0, 0, 0, 0],
    [24, 32, 0, 0, 0, 0, 0, 0],
    [31, 0, 0, 0, 0, 0, 0, 0],
];

/// Last registry index of each total-degree group.
///
/// The registry orders its monomials by total degree and closes each
/// odd group with that group's radial term, so a degree group is a
/// contiguous run. [`TpvAxis::top_degree`] walks these runs.
const DEGREE_END: [usize; 8] = [0, 3, 6, 11, 16, 23, 30, 39];

/// Registry indices of the radial terms `r`, `r^3`, `r^5` and `r^7`.
const RADIAL: [usize; 4] = [3, 11, 23, 39];

/// Per-axis TPV coefficient table. Defaults: PV*_1 = 1, others 0.
#[derive(Debug, Clone, Copy)]
pub struct TpvAxis {
    /// 0 -> axis 1 (xi -> xi'), 1 -> axis 2 (eta -> eta').
    pub axis: u8,
    /// 40 polynomial coefficients indexed by `m`.
    pub coeffs: [f64; TPV_NCOEFFS],
}

impl TpvAxis {
    /// Construct from a slice of `(m, value)` pairs, such as the
    /// values parsed from the `PVi_m` cards.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `axis` is neither 1 nor 2, or when an
    /// `m` index falls outside the range 0 to 39.
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

    /// The polynomial arguments, in the order this axis reads them.
    ///
    /// For axis 2 the internal `x` is `eta` and `y` is `xi`, per the
    /// registry.
    #[inline]
    fn xy(&self, xi: f64, eta: f64) -> (f64, f64) {
        if self.axis == 1 {
            (xi, eta)
        } else {
            (eta, xi)
        }
    }

    /// Highest total degree carrying a non-zero coefficient.
    ///
    /// The table holds 40 slots whatever the solution used. A real TPV
    /// is cubic or quintic, so the top groups are usually all zero, and
    /// evaluating them multiplies by zero and adds nothing. This finds
    /// where the polynomial actually stops.
    ///
    /// This reads the coefficients rather than a cached bound because
    /// [`Self::coeffs`] is public. A bound resolved at construction
    /// would go stale the moment a caller assigned to that field.
    #[inline]
    fn top_degree(&self) -> usize {
        for d in (1..8).rev() {
            if self.coeffs[DEGREE_END[d - 1] + 1..=DEGREE_END[d]]
                .iter()
                .any(|v| *v != 0.0)
            {
                return d;
            }
        }
        0
    }

    /// The radial terms `c3 r + c11 r^3 + c23 r^5 + c39 r^7`.
    ///
    /// Factoring out `r` leaves a cubic in `t = r^2`, so this needs one
    /// `sqrt` and three multiply-adds. Most solutions leave all four
    /// coefficients at zero, and then it needs neither.
    #[inline]
    fn radial(&self, x: f64, y: f64) -> f64 {
        let c = &self.coeffs;
        if RADIAL.iter().all(|&i| c[i] == 0.0) {
            return 0.0;
        }
        let t = x * x + y * y;
        t.sqrt() * (c[3] + t * (c[11] + t * (c[23] + t * c[39])))
    }

    /// The radial terms with their gradient, `(value, d/dx, d/dy)`.
    ///
    /// `d(r^k)/dx = k r^(k-2) x`, which stays finite for `r^3` and
    /// above. The linear `r` term does not: its gradient is `x / r`,
    /// and `r` is not differentiable at the origin.
    ///
    /// Real solutions leave `PV_3` at zero. When one does not, this
    /// reports a zero gradient exactly at the origin. That is the
    /// symmetric choice, and it is also what the central difference
    /// this replaced produced there, since `r` is even in `x`.
    #[inline]
    fn radial_with_gradient(&self, x: f64, y: f64) -> (f64, f64, f64) {
        let c = &self.coeffs;
        if RADIAL.iter().all(|&i| c[i] == 0.0) {
            return (0.0, 0.0, 0.0);
        }
        let t = x * x + y * y;
        let r = t.sqrt();
        let value = r * (c[3] + t * (c[11] + t * (c[23] + t * c[39])));
        // `c3 / r` is the only term that can diverge; see above.
        let linear = if r == 0.0 { 0.0 } else { c[3] / r };
        let g = linear + r * (3.0 * c[11] + t * (5.0 * c[23] + t * (7.0 * c[39])));
        (value, x * g, y * g)
    }

    /// Evaluate the per-axis polynomial. For axis 1 the linear
    /// arguments are `(xi, eta)`; for axis 2 they are swapped to
    /// `(eta, xi)` per the TPV specification.
    ///
    /// Setting the radial terms aside, the 40 monomials are every
    /// `x^p y^q` with `p + q <= 7`. That is a triangle, and a triangle
    /// evaluates by Horner's scheme in both variables. This does that,
    /// then adds the radial terms. No power of `x` or `y` is formed.
    /// The previous code raised `x` and `y` to each power the term
    /// needed, which both cost more and lost accuracy at the high
    /// degrees.
    #[must_use]
    pub fn eval(&self, xi: f64, eta: f64) -> f64 {
        let (x, y) = self.xy(xi, eta);
        let c = &self.coeffs;
        let deg = self.top_degree();
        let mut s = 0.0_f64;
        for p in (0..=deg).rev() {
            // Row `p` runs to degree `deg - p`: the triangle bound.
            let qmax = deg - p;
            let mut r = c[XY_INDEX[p][qmax]];
            for q in (0..qmax).rev() {
                r = r * y + c[XY_INDEX[p][q]];
            }
            s = s * x + r;
        }
        s + self.radial(x, y)
    }

    /// Evaluate the polynomial and both partial derivatives, as
    /// `(value, d/d xi, d/d eta)`.
    ///
    /// The derivatives ride the Horner recurrence of [`Self::eval`]:
    /// for a step `s <- s * z + c`, the derivative satisfies
    /// `d <- d * z + s` with `s` taken from before the step. This is
    /// the scheme `SipPoly::eval_with_derivatives` uses.
    ///
    /// [`Tpv::inverse`] needs the Jacobian once per Newton step. Taking
    /// it in closed form is exact and costs this one pass, where the
    /// central difference it replaced cost four extra evaluations and
    /// carried a step-size error.
    #[must_use]
    pub fn eval_with_derivatives(&self, xi: f64, eta: f64) -> (f64, f64, f64) {
        let (x, y) = self.xy(xi, eta);
        let c = &self.coeffs;
        let deg = self.top_degree();
        let (mut s, mut dx, mut dy) = (0.0_f64, 0.0_f64, 0.0_f64);
        for p in (0..=deg).rev() {
            let qmax = deg - p;
            let mut r = c[XY_INDEX[p][qmax]];
            let mut rd = 0.0_f64;
            for q in (0..qmax).rev() {
                rd = rd * y + r;
                r = r * y + c[XY_INDEX[p][q]];
            }
            // `dx` consumes the previous `s`, so it updates first.
            dx = dx * x + s;
            s = s * x + r;
            dy = dy * x + rd;
        }
        let (rv, rdx, rdy) = self.radial_with_gradient(x, y);
        let (value, dx, dy) = (s + rv, dx + rdx, dy + rdy);
        // Undo the axis-2 swap: `x` was `eta` there, so `d/dx` is the
        // derivative with respect to `eta`.
        if self.axis == 1 {
            (value, dx, dy)
        } else {
            (value, dy, dx)
        }
    }
}

/// Pair of TPV polynomials (axis 1 + axis 2). Applied to the raw
/// intermediate world coordinates produced by the linear stage,
/// before the standard TAN projection is inverted.
#[derive(Debug, Clone, Copy)]
pub struct Tpv {
    /// `PV1_m` -- polynomial for the longitude axis.
    pub pv1: TpvAxis,
    /// `PV2_m` -- polynomial for the latitude axis.
    pub pv2: TpvAxis,
}

impl Tpv {
    /// Apply the forward distortion: `(xi, eta) -> (xi', eta')`.
    #[must_use]
    pub fn forward(&self, xi: f64, eta: f64) -> (f64, f64) {
        (self.pv1.eval(xi, eta), self.pv2.eval(xi, eta))
    }

    /// Apply the inverse distortion, `(xi', eta') -> (xi, eta)`, by
    /// Newton iteration.
    ///
    /// The iteration converges quickly for the small distortions that
    /// real instruments carry, which stay near one pixel.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the Jacobian is singular, or when the
    /// iteration does not converge within its step limit.
    pub fn inverse(&self, xi_p: f64, eta_p: f64) -> Result<(f64, f64)> {
        // Initial guess: undistorted = distorted (good when the
        // polynomial is close to identity).
        let mut xi = xi_p;
        let mut eta = eta_p;
        // Tolerance scales with coordinate magnitude: the residual's
        // rounding floor is ~eps*|coord|, so a fixed absolute tolerance
        // becomes unreachable for large |xi'|,|eta'| (see SIP inverse).
        // In degree-scale intermediate coords this is rarely hit, but the
        // scaled form removes the latent failure and keeps accuracy far
        // below any sub-pixel relevance.
        let scale = 1.0 + xi_p.abs() + eta_p.abs();
        let tol = 1e-11 * scale;
        for _ in 0..32 {
            // Residual and exact Jacobian in one pass per axis. The
            // rows of the Jacobian are the two gradients.
            let (fx, j11, j12) = self.pv1.eval_with_derivatives(xi, eta);
            let (fy, j21, j22) = self.pv2.eval_with_derivatives(xi, eta);
            let rx = fx - xi_p;
            let ry = fy - eta_p;
            if rx.abs() < tol && ry.abs() < tol {
                return Ok((xi, eta));
            }
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
            if dx.abs() < tol && dy.abs() < tol {
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

    /// Every monomial of the registry table, as `(index, p, q)` for
    /// `x^p y^q`, written out from the published table rather than
    /// derived from [`XY_INDEX`].
    const MONOMIALS: [(usize, i32, i32); 36] = [
        (0, 0, 0),
        (1, 1, 0),
        (2, 0, 1),
        (4, 2, 0),
        (5, 1, 1),
        (6, 0, 2),
        (7, 3, 0),
        (8, 2, 1),
        (9, 1, 2),
        (10, 0, 3),
        (12, 4, 0),
        (13, 3, 1),
        (14, 2, 2),
        (15, 1, 3),
        (16, 0, 4),
        (17, 5, 0),
        (18, 4, 1),
        (19, 3, 2),
        (20, 2, 3),
        (21, 1, 4),
        (22, 0, 5),
        (24, 6, 0),
        (25, 5, 1),
        (26, 4, 2),
        (27, 3, 3),
        (28, 2, 4),
        (29, 1, 5),
        (30, 0, 6),
        (31, 7, 0),
        (32, 6, 1),
        (33, 5, 2),
        (34, 4, 3),
        (35, 3, 4),
        (36, 2, 5),
        (37, 1, 6),
        (38, 0, 7),
    ];

    /// The radial terms, as `(index, power of r)`.
    const RADIAL_TERMS: [(usize, i32); 4] = [(3, 1), (11, 3), (23, 5), (39, 7)];

    /// Sum the polynomial one monomial at a time, the obvious way.
    fn literal_eval(a: &TpvAxis, xi: f64, eta: f64) -> f64 {
        let (x, y) = if a.axis == 1 { (xi, eta) } else { (eta, xi) };
        let r = (x * x + y * y).sqrt();
        let c = &a.coeffs;
        MONOMIALS
            .iter()
            .map(|&(i, p, q)| c[i] * x.powi(p) * y.powi(q))
            .sum::<f64>()
            + RADIAL_TERMS
                .iter()
                .map(|&(i, k)| c[i] * r.powi(k))
                .sum::<f64>()
    }

    /// A table with every coefficient distinct and non-zero, so no
    /// term can hide behind another.
    fn dense_axis(axis: u8) -> TpvAxis {
        let pairs: Vec<(u32, f64)> = (0..TPV_NCOEFFS)
            .map(|m| (m as u32, ((m % 11) as f64 - 5.0) * 1e-3 / ((m + 1) as f64)))
            .collect();
        TpvAxis::from_pv_pairs(axis, &pairs).unwrap()
    }

    /// `eval` must agree with the registry's monomial list read
    /// literally.
    ///
    /// Horner reaches each coefficient through [`XY_INDEX`]. One wrong
    /// entry there silently swaps two monomials, and no round trip
    /// would notice: the forward and the inverse would agree with each
    /// other on the same wrong polynomial. This pins the table against
    /// the registry instead.
    ///
    /// The comparison is relative, not bit-exact. Horner associates the
    /// additions differently from a term-by-term sum, so the two agree
    /// to rounding rather than exactly.
    #[test]
    fn eval_matches_the_registry_monomials() {
        for axis in [1_u8, 2] {
            let a = dense_axis(axis);
            for &(xi, eta) in &[
                (0.0_f64, 0.0_f64),
                (0.5, -0.3),
                (-1.25, 0.75),
                (2.0, 2.0),
                (1e-6, -1e-6),
            ] {
                let got = a.eval(xi, eta);
                let want = literal_eval(&a, xi, eta);
                let scale = want.abs().max(1e-12);
                assert!(
                    (got - want).abs() / scale < 1e-12,
                    "axis {axis} at ({xi}, {eta}): Horner {got} vs literal {want}"
                );
            }
        }
    }

    /// Truncating at the top non-zero degree must not change the value.
    ///
    /// `top_degree` skips whole degree groups. Skipping a group that
    /// holds a non-zero coefficient would drop real terms.
    #[test]
    fn truncation_keeps_every_non_zero_term() {
        for top in 0..TPV_NCOEFFS {
            // One coefficient set, at each position in the table.
            let a = TpvAxis::from_pv_pairs(1, &[(1, 0.0), (top as u32, 0.25)]).unwrap();
            for &(xi, eta) in &[(0.7_f64, -0.4_f64), (-1.5, 2.5)] {
                let got = a.eval(xi, eta);
                let want = literal_eval(&a, xi, eta);
                let scale = want.abs().max(1e-12);
                assert!(
                    (got - want).abs() / scale < 1e-12,
                    "coefficient {top} at ({xi}, {eta}): {got} vs {want}"
                );
            }
        }
    }

    /// The closed-form Jacobian must match a numerical one.
    ///
    /// This is the check the central difference it replaced could not
    /// make of itself. A central difference at a well-chosen step is
    /// accurate to about eight digits, which is far tighter than the
    /// tolerance the Newton iteration needs.
    #[test]
    fn analytic_derivatives_match_a_central_difference() {
        for axis in [1_u8, 2] {
            let a = dense_axis(axis);
            for &(xi, eta) in &[(0.5_f64, -0.3_f64), (-1.25, 0.75), (2.0, 2.0), (0.01, 0.02)] {
                let (value, d_xi, d_eta) = a.eval_with_derivatives(xi, eta);
                assert!(
                    (value - a.eval(xi, eta)).abs() < 1e-15,
                    "axis {axis}: value disagrees with `eval`"
                );
                let h = 1e-5;
                let num_xi = (a.eval(xi + h, eta) - a.eval(xi - h, eta)) / (2.0 * h);
                let num_eta = (a.eval(xi, eta + h) - a.eval(xi, eta - h)) / (2.0 * h);
                for (got, want, name) in [(d_xi, num_xi, "d/dxi"), (d_eta, num_eta, "d/deta")] {
                    let scale = want.abs().max(1e-6);
                    assert!(
                        (got - want).abs() / scale < 1e-6,
                        "axis {axis} at ({xi}, {eta}): {name} analytic {got} vs numeric {want}"
                    );
                }
            }
        }
    }

    /// `PV_3` makes the gradient diverge at the origin. The evaluation
    /// must stay finite there rather than returning an infinity that
    /// would poison the Newton step.
    #[test]
    fn the_linear_radial_term_stays_finite_at_the_origin() {
        // `PV_1` defaults to 1, so clear it: the radial term must be
        // the only thing the gradient can pick up.
        let a = TpvAxis::from_pv_pairs(1, &[(1, 0.0), (3, 0.001)]).unwrap();
        let (value, d_xi, d_eta) = a.eval_with_derivatives(0.0, 0.0);
        assert!(value.is_finite() && d_xi.is_finite() && d_eta.is_finite());
        assert_eq!((d_xi, d_eta), (0.0, 0.0), "expected the symmetric choice");

        // Just off the origin the gradient is the real one: `r` grows
        // at unit rate along a ray, scaled by the coefficient.
        let (_, d_xi, _) = a.eval_with_derivatives(1e-3, 0.0);
        assert!((d_xi - 0.001).abs() < 1e-12, "got {d_xi}");
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
