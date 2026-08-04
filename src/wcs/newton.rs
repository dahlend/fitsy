//! The 2x2 Newton iteration shared by the distortion inverses.
//!
//! [`Sip`](super::sip::Sip), [`Tpv`](super::tpv::Tpv),
//! [`Tnx`](super::tnx::Tnx) and [`Dss`](super::dss::Dss) each invert a
//! forward map `F: (x, y) -> (x', y')`. None of the four has a
//! closed-form inverse. All four solve `F(x, y) - (x', y') = 0` by
//! Newton iteration, with the 2x2 linear system taken by Cramer's
//! rule.
//!
//! Only the residual and the Jacobian differ between them. Those stay
//! in the callers. The step limit, the tolerance rule and the
//! singularity guard have one definition here.
//!
//! # Tolerance
//!
//! The tolerance scales with coordinate magnitude. The residual
//! `r = F(x, y) - (x', y')` subtracts two numbers of size `~|x'|`. Its
//! smallest representable magnitude is therefore the rounding floor
//! `~eps*|coord|`. A fixed absolute tolerance becomes unreachable once
//! `|coord|` passes a few hundred. A WISE frame evaluated past the
//! array edge reaches `|x'| ~ 1e3`, which puts the floor near 5e-13.
//! Newton reaches the correct root there. It then uses every remaining
//! iteration on rounding noise and reports a false non-convergence. A
//! relative tolerance tracks the floor instead. It stays far below any
//! sub-pixel accuracy that matters, near 1e-8 px at `|coord| ~ 1e3`.

use crate::error::{FitsError, Result};

/// Relative convergence tolerance, applied against [`residual_scale`].
const TOL_REL: f64 = 1e-11;

/// Iteration limit.
///
/// Newton doubles the number of correct digits per step near a simple
/// root. A converging point therefore finishes in a few steps. This
/// limit bounds the cost of a point that does not converge.
const MAX_ITERATIONS: usize = 32;

/// Smallest Jacobian determinant treated as invertible.
///
/// Below this value the Cramer solve amplifies rounding without bound.
/// The solver rejects the point instead of taking the step.
const MIN_DET: f64 = 1e-15;

/// The residual and Jacobian of the forward map at one point.
///
/// `rx`/`ry` are `F(x, y) - (x', y')`. `jNM` is `dF_N/d_M`, so the
/// matrix reads `[[j11, j12], [j21, j22]]`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Residual2 {
    /// First-axis residual.
    pub rx: f64,
    /// Second-axis residual.
    pub ry: f64,
    /// Row 1, column 1 of the Jacobian.
    pub j11: f64,
    /// Row 1, column 2.
    pub j12: f64,
    /// Row 2, column 1.
    pub j21: f64,
    /// Row 2, column 2.
    pub j22: f64,
}

/// The magnitude scale of a target point, for the [`solve`] tolerance.
///
/// The `1.0` floor keeps the tolerance finite at the origin. A purely
/// relative measure would demand an exact zero residual there.
pub(crate) fn residual_scale(x: f64, y: f64) -> f64 {
    1.0 + x.abs() + y.abs()
}

/// Solve a 2x2 system by Newton iteration.
///
/// The `label` argument names the distortion in an error message,
/// such as `"SIP"`. The `guess` argument is the starting point. The
/// `scale` argument sets the tolerance, and comes from
/// [`residual_scale`] applied to the target rather than to the guess.
/// The `step` argument returns the residual and the Jacobian at a
/// point.
///
/// `step` returns both parts together, once per iteration. Each caller
/// evaluates its polynomial and the derivatives in one pass, so a
/// split into two calls would double that work.
///
/// The solver tests convergence twice per iteration. It tests the
/// residual before the step, and the step size after it. A guess that
/// already sits on the root therefore returns without a wasted
/// iteration.
///
/// # Errors
///
/// [`FitsError::Wcs`] when the Jacobian is singular at an iterate.
///
/// [`FitsError::Wcs`] when the iteration reaches the step limit
/// without converging.
pub(crate) fn solve(
    label: &str,
    guess: (f64, f64),
    scale: f64,
    step: impl Fn(f64, f64) -> Residual2,
) -> Result<(f64, f64)> {
    let tol = TOL_REL * scale;
    let (mut x, mut y) = guess;
    for _ in 0..MAX_ITERATIONS {
        let r = step(x, y);
        if r.rx.abs() < tol && r.ry.abs() < tol {
            return Ok((x, y));
        }
        let det = r.j11 * r.j22 - r.j12 * r.j21;
        if det.abs() < MIN_DET {
            return Err(FitsError::Wcs(format!(
                "{label}: Jacobian singular during inverse iteration"
            )));
        }
        // Solve J * delta = r by Cramer's rule, then step against it.
        let dx = (r.j22 * r.rx - r.j12 * r.ry) / det;
        let dy = (-r.j21 * r.rx + r.j11 * r.ry) / det;
        x -= dx;
        y -= dy;
        if dx.abs() < tol && dy.abs() < tol {
            return Ok((x, y));
        }
    }
    Err(FitsError::Wcs(format!(
        "{label}: inverse iteration did not converge"
    )))
}
