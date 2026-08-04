//! Bivariate polynomial evaluation on a triangular coefficient set.
//!
//! Three distortion conventions reduce to one sum,
//! `S = sum c_{p,q} x^p y^q` over the triangle `p + q <= deg`.
//! [`Sip`](super::sip::Sip) stores that triangle directly.
//! [`Tpv`](super::tpv::Tpv) reaches it through the registry index
//! table. [`Dss`](super::dss::Dss) reaches it by expanding the named
//! plate terms. Only the addressing differs. The evaluation therefore
//! lives here once, and each caller supplies a coefficient accessor.
//!
//! # Why Horner in both variables
//!
//! Each row `p` collapses over `y`. The rows then collapse over `x`.
//! The cost is one multiply-add per coefficient. No power of `x` or
//! `y` is formed.
//!
//! The alternative tabulates the powers once and multiplies out. That
//! table takes the size of the largest order the convention allows, so
//! an order-2 polynomial pays for an order-9 one. Skipping the zero
//! coefficients does not recover the difference. The work tracks the
//! declared order either way, and the branch costs more than the
//! multiply-add it avoids. Forming each power directly costs more
//! again, and loses accuracy at the high degrees.

/// Evaluate `sum_{p+q < dim} c(p, q) x^p y^q`.
///
/// The `dim` argument is the number of degree levels, one more than
/// the highest total degree. A `dim` of 0 is the empty polynomial and
/// evaluates to zero. The `c` argument reads one coefficient. This
/// calls it once for every `(p, q)` with `p + q < dim`.
#[inline]
pub(crate) fn triangle(dim: usize, c: impl Fn(usize, usize) -> f64, x: f64, y: f64) -> f64 {
    let mut s = 0.0_f64;
    // Rows from the top, so each step multiplies the accumulated
    // higher-order rows by `x` before adding this one.
    for p in (0..dim).rev() {
        // Row `p` runs to degree `dim - 1 - p`: the triangle bound.
        let qmax = dim - 1 - p;
        let mut r = c(p, qmax);
        for q in (0..qmax).rev() {
            r = r * y + c(p, q);
        }
        s = s * x + r;
    }
    s
}

/// Evaluate the polynomial and both partial derivatives, as
/// `(value, d/dx, d/dy)`.
///
/// The derivatives follow the Horner recurrence of [`triangle`]. For a
/// step `s <- s * z + c`, the derivative satisfies `d <- d * z + s`,
/// with `s` taken from before the step. That recurrence in `y` gives
/// the `d/dy` of each row. The same recurrence in `x` gives `d/dx`
/// over the collapsed rows.
///
/// Differentiating term by term is exact and costs one pass. A central
/// difference costs four extra evaluations and carries a step-size
/// error. The Newton inverses in [`sip`](super::sip),
/// [`tpv`](super::tpv) and [`dss`](super::dss) each need the Jacobian
/// once per step.
///
/// The value returned is bit-identical to [`triangle`] on the same
/// input. The two share the recurrence and its operation order.
#[inline]
pub(crate) fn triangle_with_derivatives(
    dim: usize,
    c: impl Fn(usize, usize) -> f64,
    x: f64,
    y: f64,
) -> (f64, f64, f64) {
    let (mut s, mut dx, mut dy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for p in (0..dim).rev() {
        let qmax = dim - 1 - p;
        let mut r = c(p, qmax);
        let mut rd = 0.0_f64;
        for q in (0..qmax).rev() {
            rd = rd * y + r;
            r = r * y + c(p, q);
        }
        // `dx` consumes the previous `s`, so it updates first.
        dx = dx * x + s;
        s = s * x + r;
        dy = dy * x + rd;
    }
    (s, dx, dy)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(1 + 2x + 3y + 4x^2 + 5xy + 6y^2)` and its exact gradient.
    fn quad(p: usize, q: usize) -> f64 {
        match (p, q) {
            (0, 0) => 1.0,
            (1, 0) => 2.0,
            (0, 1) => 3.0,
            (2, 0) => 4.0,
            (1, 1) => 5.0,
            (0, 2) => 6.0,
            _ => 0.0,
        }
    }

    #[test]
    fn matches_direct_expansion() {
        for &(x, y) in &[(0.0, 0.0), (1.0, -1.0), (0.5, 0.25), (-3.0, 2.0)] {
            let want = 1.0 + 2.0 * x + 3.0 * y + 4.0 * x * x + 5.0 * x * y + 6.0 * y * y;
            assert!(
                (triangle(3, quad, x, y) - want).abs() < 1e-13,
                "at ({x}, {y})"
            );
        }
    }

    #[test]
    fn derivatives_match_closed_form() {
        for &(x, y) in &[(0.0, 0.0), (1.0, -1.0), (0.5, 0.25), (-3.0, 2.0)] {
            let (v, dx, dy) = triangle_with_derivatives(3, quad, x, y);
            assert_eq!(v, triangle(3, quad, x, y), "value diverged at ({x}, {y})");
            assert!(
                (dx - (2.0 + 8.0 * x + 5.0 * y)).abs() < 1e-13,
                "d/dx at ({x}, {y})"
            );
            assert!(
                (dy - (3.0 + 5.0 * x + 12.0 * y)).abs() < 1e-13,
                "d/dy at ({x}, {y})"
            );
        }
    }

    /// `dim = 0` is the empty polynomial, not an underflow.
    #[test]
    fn empty_is_zero() {
        assert_eq!(triangle(0, |_, _| 1.0, 2.0, 3.0), 0.0);
        assert_eq!(
            triangle_with_derivatives(0, |_, _| 1.0, 2.0, 3.0),
            (0.0, 0.0, 0.0)
        );
    }
}
