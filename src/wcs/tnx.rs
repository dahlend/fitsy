//! IRAF TNX / ZPX polynomial distortion (non-standard).
//!
//! TNX is the IRAF Mosaic-pipeline extension to TAN; ZPX is the same
//! polynomial machinery on top of ZPN. The distortion is encoded in
//! the `WAT1_xxx` and `WAT2_xxx` keyword strings as
//!
//! ```text
//! wtype={tnx|zpx} axtype={ra|dec|...} [projp1=... projp2=...]
//!     lngcor = "<surface>"
//!     latcor = "<surface>"
//! ```
//!
//! where each `<surface>` body is a whitespace-separated record of
//!
//! ```text
//! function_type ni nj cross_term ximin ximax etamin etamax
//!     c00 c10 c01 c20 c11 c02 ...
//! ```
//!
//! and the corrections add to the (xi, eta) intermediate world
//! coordinates in degrees:
//!
//! ```text
//! xi'  = xi  + lngcor(xi, eta)
//! eta' = eta + latcor(xi, eta)
//! ```
//!
//! The distortion is applied between the linear matrix and the base
//! projection (TAN for TNX, ZPN for ZPX), exactly the slot occupied
//! by [`crate::wcs::tpv::Tpv`] for the TPV convention.
//!
//! ## References
//! - IRAF `noao$digiphot/lib/tnx.h` and the `MWCS` documentation.
//! - <http://iraf.noao.edu/projects/ccdmosaic/tnx.html>
//!
//! ## Validation
//!
//! This module follows the IRAF specification. No reference
//! implementation was available to compare against. The unit tests
//! cover three things instead:
//!
//! - The polynomial, Chebyshev and Legendre evaluators, against
//!   analytic ground truth.
//! - The derivatives of those evaluators, against the closed forms of
//!   the same polynomials. A finite-difference check cannot separate a
//!   wrong derivative from a shared error in the basis. This can.
//! - End-to-end round trips from pixel to world and back.

use crate::error::{FitsError, Result};
use crate::wcs::newton;

/// Surface basis function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TnxFunction {
    /// `function_type = 1`: Chebyshev polynomials of the 1st kind.
    Chebyshev,
    /// `function_type = 2`: Legendre polynomials.
    Legendre,
    /// `function_type = 3`: ordinary monomials `x^i * y^j`.
    Polynomial,
}

/// Cross-term policy controlling which `(i, j)` pairs are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TnxCrossTerm {
    /// `xterms = 0`: only `(i, 0)` for `i = 0..ni` and `(0, j)` for
    /// `j = 1..nj` (no mixed terms).
    None,
    /// `xterms = 1`: every `(i, j)` with `0 <= i < ni`, `0 <= j < nj`.
    Full,
    /// `xterms = 2`: every `(i, j)` with `i + j <= max(ni, nj) - 1`.
    Half,
}

/// One IRAF TNX/ZPX correction surface (`lngcor` or `latcor`).
#[derive(Debug, Clone)]
pub struct TnxSurface {
    /// Basis the surface is expanded in.
    pub function: TnxFunction,
    /// Number of basis functions in the xi (longitude) direction.
    pub ni: u32,
    /// Number of basis functions in the eta (latitude) direction.
    pub nj: u32,
    /// Which cross terms the expansion retains.
    pub cross: TnxCrossTerm,
    /// Lower bound of the `xi` normalization range.
    pub xi_min: f64,
    /// Upper bound of the `xi` normalization range.
    pub xi_max: f64,
    /// Lower bound of the `eta` normalization range.
    pub eta_min: f64,
    /// Upper bound of the `eta` normalization range.
    pub eta_max: f64,
    /// Coefficients in IRAF row-major order: outer loop over `j`,
    /// inner loop over `i`, restricted to entries selected by
    /// [`cross`](Self::cross).
    pub coeffs: Vec<f64>,
}

impl TnxSurface {
    /// Parse a surface body, meaning the text inside the quotes of a
    /// `lngcor` or `latcor` entry.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in five cases:
    ///
    /// - A mandatory token is absent.
    /// - A token does not parse as a number.
    /// - The function type or the cross-term code is unknown.
    /// - An order falls outside the supported range.
    /// - The coefficient count does not match the orders and the
    ///   cross-term code.
    pub fn parse(body: &str) -> Result<Self> {
        let mut tokens = body.split_ascii_whitespace();
        let next = |it: &mut std::str::SplitAsciiWhitespace<'_>, name: &str| -> Result<f64> {
            it.next()
                .ok_or_else(|| FitsError::Wcs(format!("TNX: missing token {name}")))?
                .parse::<f64>()
                .map_err(|e| FitsError::Wcs(format!("TNX: bad {name}: {e}")))
        };
        let ft = next(&mut tokens, "function_type")? as i32;
        let function = match ft {
            1 => TnxFunction::Chebyshev,
            2 => TnxFunction::Legendre,
            3 => TnxFunction::Polynomial,
            _ => return Err(FitsError::Wcs(format!("TNX: unknown function_type {ft}"))),
        };
        let ni = next(&mut tokens, "ni")? as i32;
        let nj = next(&mut tokens, "nj")? as i32;
        if !(1..=20).contains(&ni) || !(1..=20).contains(&nj) {
            return Err(FitsError::Wcs(format!(
                "TNX: implausible orders ni={ni}, nj={nj}"
            )));
        }
        let xt = next(&mut tokens, "cross_term")? as i32;
        let cross = match xt {
            0 => TnxCrossTerm::None,
            1 => TnxCrossTerm::Full,
            2 => TnxCrossTerm::Half,
            _ => return Err(FitsError::Wcs(format!("TNX: unknown cross_term {xt}"))),
        };
        let xi_min = next(&mut tokens, "xi_min")?;
        let xi_max = next(&mut tokens, "xi_max")?;
        let eta_min = next(&mut tokens, "eta_min")?;
        let eta_max = next(&mut tokens, "eta_max")?;
        if xi_max <= xi_min || eta_max <= eta_min {
            return Err(FitsError::Wcs(
                "TNX: degenerate normalization interval".into(),
            ));
        }
        let mut coeffs: Vec<f64> = Vec::new();
        for tok in tokens {
            coeffs.push(
                tok.parse::<f64>()
                    .map_err(|e| FitsError::Wcs(format!("TNX: bad coeff {tok}: {e}")))?,
            );
        }
        let expected = expected_coeff_count(ni as u32, nj as u32, cross);
        if coeffs.len() != expected {
            return Err(FitsError::Wcs(format!(
                "TNX: coefficient count {} != expected {} for ni={ni}, nj={nj}, cross={cross:?}",
                coeffs.len(),
                expected,
            )));
        }
        Ok(Self {
            function,
            ni: ni as u32,
            nj: nj as u32,
            cross,
            xi_min,
            xi_max,
            eta_min,
            eta_max,
            coeffs,
        })
    }

    /// Evaluate the surface at `(xi, eta)`.
    #[must_use]
    pub fn eval(&self, xi: f64, eta: f64) -> f64 {
        // Normalize into [-1, 1] for Chebyshev/Legendre only. IRAF's
        // `wf_gseval` passes RAW coordinates to the ordinary polynomial
        // basis (`wf_gsb1pol` takes no normalization constants at all).
        let (xn, yn) = if matches!(self.function, TnxFunction::Polynomial) {
            (xi, eta)
        } else {
            (
                (2.0 * xi - (self.xi_max + self.xi_min)) / (self.xi_max - self.xi_min),
                (2.0 * eta - (self.eta_max + self.eta_min)) / (self.eta_max - self.eta_min),
            )
        };
        let bx = basis(self.function, xn, self.ni as usize);
        let by = basis(self.function, yn, self.nj as usize);
        let mut sum = 0.0;
        let mut k = 0_usize;
        #[allow(
            clippy::needless_range_loop,
            reason = "nested (j, i) indices mirror the mathematical basis expansion"
        )]
        for j in 0..self.nj as usize {
            for i in 0..self.ni as usize {
                if !cross_includes(self.cross, i, j, self.ni as usize, self.nj as usize) {
                    continue;
                }
                sum += self.coeffs[k] * bx[i] * by[j];
                k += 1;
            }
        }
        sum
    }

    /// Evaluate the surface and both partials, as
    /// `(value, d/dxi, d/deta)`.
    ///
    /// The expansion is separable, `S = sum_k c_k B_i(xn) B_j(yn)`.
    /// Each partial therefore differentiates one factor and leaves the
    /// other unchanged. The normalization is affine and contributes a
    /// constant factor `dxn/dxi = 2 / (xi_max - xi_min)`. The ordinary
    /// polynomial basis reads raw coordinates, as [`Self::eval`]
    /// describes, and so contributes a factor of 1.
    ///
    /// Differentiating term by term is exact and costs one pass. The
    /// central difference this replaced cost four extra surface
    /// evaluations per axis. It also carried a step-size error, which
    /// bounded how closely the Newton inverse could converge.
    #[must_use]
    pub fn eval_with_derivatives(&self, xi: f64, eta: f64) -> (f64, f64, f64) {
        let (xn, yn, dxn_dxi, dyn_deta) = if matches!(self.function, TnxFunction::Polynomial) {
            (xi, eta, 1.0, 1.0)
        } else {
            (
                (2.0 * xi - (self.xi_max + self.xi_min)) / (self.xi_max - self.xi_min),
                (2.0 * eta - (self.eta_max + self.eta_min)) / (self.eta_max - self.eta_min),
                2.0 / (self.xi_max - self.xi_min),
                2.0 / (self.eta_max - self.eta_min),
            )
        };
        let (bx, dbx) = basis_with_derivative(self.function, xn, self.ni as usize);
        let (by, dby) = basis_with_derivative(self.function, yn, self.nj as usize);
        let (mut sum, mut d_xi, mut d_eta) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut k = 0_usize;
        #[allow(
            clippy::needless_range_loop,
            reason = "nested (j, i) indices mirror the mathematical basis expansion"
        )]
        for j in 0..self.nj as usize {
            for i in 0..self.ni as usize {
                if !cross_includes(self.cross, i, j, self.ni as usize, self.nj as usize) {
                    continue;
                }
                let c = self.coeffs[k];
                sum += c * bx[i] * by[j];
                d_xi += c * dbx[i] * by[j];
                d_eta += c * bx[i] * dby[j];
                k += 1;
            }
        }
        (sum, d_xi * dxn_dxi, d_eta * dyn_deta)
    }
}

fn expected_coeff_count(ni: u32, nj: u32, cross: TnxCrossTerm) -> usize {
    let ni = ni as usize;
    let nj = nj as usize;
    let mut n = 0;
    for j in 0..nj {
        for i in 0..ni {
            if cross_includes(cross, i, j, ni, nj) {
                n += 1;
            }
        }
    }
    n
}

fn cross_includes(cross: TnxCrossTerm, i: usize, j: usize, ni: usize, nj: usize) -> bool {
    match cross {
        TnxCrossTerm::Full => true,
        TnxCrossTerm::None => i == 0 || j == 0,
        TnxCrossTerm::Half => i + j < ni.max(nj),
    }
}

/// Basis vector and its derivative, `([B_k(x)], [dB_k/dx])`.
///
/// Each derivative comes from differentiating the three-term
/// recurrence [`basis`] runs. It therefore reuses the values already
/// computed, at one multiply-add per term:
///
/// ```text
/// monomial:  b'[k]   = k * b[k-1]
/// Chebyshev: b'[k+1] = 2 b[k] + 2x b'[k] - b'[k-1]
/// Legendre:  b'[k+1] = ((2k+1)(b[k] + x b'[k]) - k b'[k-1]) / (k+1)
/// ```
///
/// This uses the differentiated recurrence rather than a closed form.
/// The closed forms carry removable singularities inside the range
/// these surfaces cover. The Legendre form
/// `(1 - x^2) P'_n = n (P_{n-1} - x P_n)` divides by zero at
/// `x = +-1`, which is the edge of the normalization interval. The
/// recurrence holds at every point of that interval.
fn basis_with_derivative(f: TnxFunction, x: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut b = vec![0.0; n];
    let mut d = vec![0.0; n];
    if n == 0 {
        return (b, d);
    }
    // B_0 = 1 for every basis, so its derivative is 0.
    b[0] = 1.0;
    if n == 1 {
        return (b, d);
    }
    match f {
        TnxFunction::Polynomial => {
            for k in 1..n {
                // Both read the finished b[k-1]; d must use the value
                // before this step overwrites nothing it depends on.
                d[k] = (k as f64) * b[k - 1];
                b[k] = b[k - 1] * x;
            }
        }
        TnxFunction::Chebyshev => {
            b[1] = x;
            d[1] = 1.0;
            for k in 2..n {
                b[k] = 2.0 * x * b[k - 1] - b[k - 2];
                d[k] = 2.0 * b[k - 1] + 2.0 * x * d[k - 1] - d[k - 2];
            }
        }
        TnxFunction::Legendre => {
            b[1] = x;
            d[1] = 1.0;
            for k in 1..n - 1 {
                let kf = k as f64;
                d[k + 1] = ((2.0 * kf + 1.0) * (b[k] + x * d[k]) - kf * d[k - 1]) / (kf + 1.0);
                b[k + 1] = ((2.0 * kf + 1.0) * x * b[k] - kf * b[k - 1]) / (kf + 1.0);
            }
        }
    }
    (b, d)
}

/// Basis vector `[B_0(x), B_1(x), ..., B_{n-1}(x)]`.
fn basis(f: TnxFunction, x: f64, n: usize) -> Vec<f64> {
    let mut b = vec![0.0; n];
    if n == 0 {
        return b;
    }
    b[0] = 1.0;
    if n == 1 {
        return b;
    }
    match f {
        TnxFunction::Polynomial => {
            for k in 1..n {
                b[k] = b[k - 1] * x;
            }
        }
        TnxFunction::Chebyshev => {
            // T_0 = 1, T_1 = x, T_{k+1} = 2x*T_k - T_{k-1}.
            b[1] = x;
            for k in 2..n {
                b[k] = 2.0 * x * b[k - 1] - b[k - 2];
            }
        }
        TnxFunction::Legendre => {
            // P_0 = 1, P_1 = x, (k+1)*P_{k+1} = (2k+1)x*P_k - k*P_{k-1}.
            b[1] = x;
            for k in 1..n - 1 {
                let kf = k as f64;
                b[k + 1] = ((2.0 * kf + 1.0) * x * b[k] - kf * b[k - 1]) / (kf + 1.0);
            }
        }
    }
    b
}

/// One full TNX / ZPX axis pair.
#[derive(Debug, Clone)]
pub struct Tnx {
    /// Longitude correction surface, from `WAT1_nnn`.
    pub lngcor: Option<TnxSurface>,
    /// Latitude correction surface, from `WAT2_nnn`.
    pub latcor: Option<TnxSurface>,
}

impl Tnx {
    /// Parse the `lngcor` and `latcor` surfaces from the reassembled
    /// `WATi_` strings of the longitude axis and the latitude axis.
    /// The caller decides which string is which.
    ///
    /// Each surface is looked up by its own name, so a mislabeled
    /// string yields `None` rather than the wrong axis. The result is
    /// `Ok(None)` when neither surface is present.
    ///
    /// # Errors
    ///
    /// The conditions of [`TnxSurface::parse`], for whichever surface
    /// is present.
    pub fn from_wat_strings(wat_lon: Option<&str>, wat_lat: Option<&str>) -> Result<Option<Self>> {
        let lngcor = wat_lon
            .and_then(|w| extract_cor(w, "lngcor"))
            .map(TnxSurface::parse)
            .transpose()?;
        let latcor = wat_lat
            .and_then(|w| extract_cor(w, "latcor"))
            .map(TnxSurface::parse)
            .transpose()?;
        if lngcor.is_none() && latcor.is_none() {
            Ok(None)
        } else {
            Ok(Some(Self { lngcor, latcor }))
        }
    }

    /// Forward distortion, `(xi, eta) -> (xi + lngcor, eta + latcor)`.
    /// A surface that is absent contributes 0.
    #[must_use]
    pub fn forward(&self, xi: f64, eta: f64) -> (f64, f64) {
        let dxi = self.lngcor.as_ref().map_or(0.0, |s| s.eval(xi, eta));
        let deta = self.latcor.as_ref().map_or(0.0, |s| s.eval(xi, eta));
        (xi + dxi, eta + deta)
    }

    /// Inverse distortion, by Newton iteration on the forward map.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the Jacobian is singular, or when the
    /// iteration does not converge within its step limit.
    pub fn inverse(&self, xip: f64, etap: f64) -> Result<(f64, f64)> {
        newton::solve(
            "TNX",
            (xip, etap),
            newton::residual_scale(xip, etap),
            |xi, eta| {
                // Residual and exact Jacobian in one pass per surface.
                // F(xi, eta) = (xi + lngcor, eta + latcor), so the
                // identity part contributes the 1 on each diagonal --
                // the same shape as the SIP inverse.
                let (dxi, dxi_dxi, dxi_deta) = self
                    .lngcor
                    .as_ref()
                    .map_or((0.0, 0.0, 0.0), |s| s.eval_with_derivatives(xi, eta));
                let (deta, deta_dxi, deta_deta) = self
                    .latcor
                    .as_ref()
                    .map_or((0.0, 0.0, 0.0), |s| s.eval_with_derivatives(xi, eta));
                newton::Residual2 {
                    rx: (xi + dxi) - xip,
                    ry: (eta + deta) - etap,
                    j11: 1.0 + dxi_dxi,
                    j12: dxi_deta,
                    j21: deta_dxi,
                    j22: 1.0 + deta_deta,
                }
            },
        )
    }
}

/// Extract the body of the `<key> = "..."` clause (`key` is `lngcor` or
/// `latcor`) from a reassembled WAT string. Returns the substring between
/// the double quotes, or `None` if the clause is absent.
fn extract_cor<'a>(wat: &'a str, key: &str) -> Option<&'a str> {
    let after = &wat[wat.find(key)? + key.len()..];
    let q1 = after.find('"')?;
    let rest = &after[q1 + 1..];
    let q2 = rest.find('"')?;
    Some(&rest[..q2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a surface that holds a single unit coefficient.
    ///
    /// `eval` then returns one basis function of `x` alone, and the
    /// derivative of that function is known in closed form. The
    /// surface uses `ni = 4`, `nj = 1` and `xterms = 0`.
    fn single_term(ft: u32, k: usize) -> TnxSurface {
        let mut c = ["0."; 4];
        c[k] = "1.";
        TnxSurface::parse(&format!("{ft}. 4. 1. 0. -1. 1. -1. 1. {}", c.join(" ")))
            .expect("single-term surface parses")
    }

    /// Check the three derivative recurrences against closed-form
    /// ground truth.
    ///
    /// The closed forms are `T'_2 = 4x`, `T'_3 = 12x^2 - 3`,
    /// `P'_2 = 3x`, `P'_3 = (15x^2 - 3)/2` and
    /// `d(x^k)/dx = k x^(k-1)`.
    ///
    /// This module has no reference implementation, as the module docs
    /// state. The derivatives are therefore pinned to the published
    /// polynomials rather than to a finite difference. A
    /// finite-difference check can agree with a wrong analytic form
    /// when both share a mistake in the basis. This check cannot.
    #[test]
    fn derivative_recurrences_match_closed_form() {
        for &x in &[-1.0_f64, -0.9, -0.4, 0.0, 0.3, 0.75, 1.0] {
            let d = |ft: u32, k: usize| single_term(ft, k).eval_with_derivatives(x, 0.0).1;
            // Chebyshev of the first kind.
            assert!((d(1, 2) - 4.0 * x).abs() < 1e-14, "T'_2 at {x}");
            assert!(
                (d(1, 3) - (12.0 * x * x - 3.0)).abs() < 1e-14,
                "T'_3 at {x}"
            );
            // Legendre.
            assert!((d(2, 2) - 3.0 * x).abs() < 1e-14, "P'_2 at {x}");
            assert!(
                (d(2, 3) - (15.0 * x * x - 3.0) / 2.0).abs() < 1e-14,
                "P'_3 at {x}"
            );
            // Ordinary monomials.
            assert!((d(3, 2) - 2.0 * x).abs() < 1e-14, "d(x^2) at {x}");
            assert!((d(3, 3) - 3.0 * x * x).abs() < 1e-14, "d(x^3) at {x}");
        }
    }

    /// The fused pass must return what [`TnxSurface::eval`] returns,
    /// bit for bit.
    ///
    /// The two share the basis recurrence and the summation order. An
    /// inequality therefore means an operation moved, and the inverse
    /// would converge to a different point.
    #[test]
    fn fused_value_matches_eval() {
        for ft in [1_u32, 2, 3] {
            for k in 0..4 {
                let s = single_term(ft, k);
                for &x in &[-0.9_f64, -0.25, 0.0, 0.5, 1.0] {
                    for &y in &[-0.7_f64, 0.0, 0.4] {
                        assert_eq!(
                            s.eval_with_derivatives(x, y).0,
                            s.eval(x, y),
                            "ft={ft} k={k} at ({x}, {y})"
                        );
                    }
                }
            }
        }
    }

    /// Check the normalization chain rule.
    ///
    /// A Chebyshev surface on `[-2, 2]` has `dxn/dxi = 1/2`. Its
    /// derivative in `xi` is therefore half the value the same
    /// coefficients give on `[-1, 1]`. The monomial basis reads raw
    /// coordinates, so it must not scale.
    #[test]
    fn normalization_chain_rule_is_applied() {
        let wide = |ft: u32| {
            TnxSurface::parse(&format!("{ft}. 4. 1. 0. -2. 2. -2. 2. 0. 0. 1. 0.")).expect("parses")
        };
        // Chebyshev T_2 on [-2, 2]: xn = xi/2, so d/dxi = 4*xn * (1/2).
        let (_, dx, _) = wide(1).eval_with_derivatives(1.0, 0.0);
        assert!(
            (dx - 4.0 * 0.5 * 0.5).abs() < 1e-14,
            "chebyshev scaling: {dx}"
        );
        // Monomial x^2 ignores the interval: d/dxi = 2*xi = 2.
        let (_, dxp, _) = wide(3).eval_with_derivatives(1.0, 0.0);
        assert!((dxp - 2.0).abs() < 1e-14, "monomial must not scale: {dxp}");
    }

    #[test]
    fn polynomial_basis_matches_monomials() {
        let b = basis(TnxFunction::Polynomial, 0.7, 4);
        for (k, bk) in b.iter().enumerate() {
            assert!((bk - 0.7_f64.powi(k as i32)).abs() < 1e-15);
        }
    }

    #[test]
    fn chebyshev_basis_matches_known_values() {
        // T_2(x) = 2x^2 - 1; T_3(x) = 4x^3 - 3x.
        let x = 0.3;
        let b = basis(TnxFunction::Chebyshev, x, 4);
        assert!((b[2] - (2.0 * x * x - 1.0)).abs() < 1e-15);
        assert!((b[3] - (4.0 * x.powi(3) - 3.0 * x)).abs() < 1e-15);
    }

    #[test]
    fn legendre_basis_matches_known_values() {
        // P_2(x) = (3x^2 - 1)/2; P_3(x) = (5x^3 - 3x)/2.
        let x = 0.4;
        let b = basis(TnxFunction::Legendre, x, 4);
        assert!((b[2] - (3.0 * x * x - 1.0) / 2.0).abs() < 1e-15);
        assert!((b[3] - (5.0 * x.powi(3) - 3.0 * x) / 2.0).abs() < 1e-15);
    }

    #[test]
    fn surface_parses_and_evaluates_constant() {
        // function_type=3 (poly), ni=1, nj=1, cross=full, range -1..1,
        // single coeff 0.5 -> constant surface 0.5.
        let s = TnxSurface::parse("3 1 1 1 -1 1 -1 1 0.5").unwrap();
        assert_eq!(s.coeffs.len(), 1);
        assert!((s.eval(0.0, 0.0) - 0.5).abs() < 1e-15);
        assert!((s.eval(0.7, -0.3) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn surface_polynomial_xy_term() {
        // Surface = 1 + 2*x_norm + 3*y_norm + 4*x_norm*y_norm
        // (ni=2, nj=2, cross=full -> 4 coeffs in (j,i) row-major:
        // (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4). Range -1..1 so
        // x_norm = x.
        let s = TnxSurface::parse("3 2 2 1 -1 1 -1 1 1 2 3 4").unwrap();
        let (x, y) = (0.5, 0.25);
        let expected = 1.0 + 2.0 * x + 3.0 * y + 4.0 * x * y;
        assert!((s.eval(x, y) - expected).abs() < 1e-12);
    }

    #[test]
    fn surface_no_cross_terms() {
        // ni=3, nj=2, cross=none keeps (i,0) for i=0..2 and (0,j) for
        // j=1: total = 4 coeffs. IRAF order: j=0 -> (0,0),(1,0),(2,0);
        // j=1 -> (0,1) only.
        // Surface = 1 + 2x + 3x^2 + 4y.
        let s = TnxSurface::parse("3 3 2 0 -1 1 -1 1 1 2 3 4").unwrap();
        assert_eq!(s.coeffs.len(), 4);
        let (x, y) = (0.3, -0.7);
        let expected = 1.0 + 2.0 * x + 3.0 * x * x + 4.0 * y;
        assert!((s.eval(x, y) - expected).abs() < 1e-12);
    }

    #[test]
    fn surface_half_cross_terms() {
        // ni=3, nj=3, cross=half (i+j <= max(ni,nj)-1 = 2): 6 coeffs.
        // (j,i) order: (0,0),(0,1),(0,2),(1,0),(1,1),(2,0).
        let s = TnxSurface::parse("3 3 3 2 -1 1 -1 1 1 2 3 4 5 6").unwrap();
        assert_eq!(s.coeffs.len(), 6);
        let (x, y) = (0.5, -0.25);
        let expected = 1.0 + 2.0 * x + 3.0 * x * x + 4.0 * y + 5.0 * x * y + 6.0 * y * y;
        assert!((s.eval(x, y) - expected).abs() < 1e-12);
    }

    #[test]
    fn rejects_wrong_coeff_count() {
        // ni=2, nj=2, cross=full needs 4; supply 3.
        let err = TnxSurface::parse("3 2 2 1 -1 1 -1 1 1 2 3").unwrap_err();
        assert!(format!("{err:?}").contains("coefficient count"));
    }

    #[test]
    fn extract_cor_finds_quoted_body() {
        let s = "wtype=tnx axtype=ra projp1=0 lngcor = \"3 1 1 1 -1 1 -1 1 0.5\"";
        assert_eq!(extract_cor(s, "lngcor").unwrap(), "3 1 1 1 -1 1 -1 1 0.5");
        // A surface is only picked up under its own name.
        assert!(extract_cor(s, "latcor").is_none());
    }

    #[test]
    fn tnx_round_trip() {
        // Tiny additive distortion in xi only.
        let lng = TnxSurface::parse("3 2 2 1 -1 1 -1 1 0 1e-3 0 5e-4").unwrap();
        let lat = TnxSurface::parse("3 2 2 1 -1 1 -1 1 0 0 1e-3 -3e-4").unwrap();
        let t = Tnx {
            lngcor: Some(lng),
            latcor: Some(lat),
        };
        for &(xi, eta) in &[(0.0, 0.0), (0.3, -0.2), (-0.5, 0.4), (0.8, 0.7)] {
            let (xp, yp) = t.forward(xi, eta);
            let (xb, yb) = t.inverse(xp, yp).unwrap();
            assert!((xb - xi).abs() < 1e-10, "xi {xi} -> {xb}");
            assert!((yb - eta).abs() < 1e-10, "eta {eta} -> {yb}");
        }
    }
}
