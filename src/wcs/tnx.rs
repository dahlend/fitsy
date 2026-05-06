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
//! function_type ni nj cross_term ximin ximax etamin etamax c00 c10 c01 c20 c11 c02 ...
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
//! Implemented from the IRAF specification. No reference
//! implementation is available in WCSLIB or astropy, so unit tests
//! cover (a) the polynomial / Chebyshev / Legendre evaluators against
//! analytic ground truth and (b) end-to-end round-trips
//! `pix -> world -> pix`.

use crate::error::{FitsError, Result};

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
    pub function: TnxFunction,
    /// Number of basis functions in the xi (longitude) direction.
    pub ni: u32,
    /// Number of basis functions in the eta (latitude) direction.
    pub nj: u32,
    pub cross: TnxCrossTerm,
    pub xi_min: f64,
    pub xi_max: f64,
    pub eta_min: f64,
    pub eta_max: f64,
    /// Coefficients in IRAF row-major order: outer loop over `j`,
    /// inner loop over `i`, restricted to entries selected by
    /// [`cross`](Self::cross).
    pub coeffs: Vec<f64>,
}

impl TnxSurface {
    /// Parse a `<surface>` body (the part inside the `"..."`).
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
                "TNX: degenerate normalisation interval".into(),
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
        // Normalize into [-1, 1] for Chebyshev/Legendre; for the
        // ordinary polynomial basis IRAF uses the same normalization.
        let xn = (2.0 * xi - (self.xi_max + self.xi_min)) / (self.xi_max - self.xi_min);
        let yn = (2.0 * eta - (self.eta_max + self.eta_min)) / (self.eta_max - self.eta_min);
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
    pub lngcor: Option<TnxSurface>,
    pub latcor: Option<TnxSurface>,
}

impl Tnx {
    /// Parse the `lngcor` / `latcor` surfaces from the reassembled
    /// `WAT1` and `WAT2` strings. Returns `Ok(None)` when neither
    /// string carries an `lngcor`/`latcor` body.
    pub fn from_wat_strings(wat1: Option<&str>, wat2: Option<&str>) -> Result<Option<Self>> {
        let lngcor = wat1
            .and_then(extract_cor)
            .map(TnxSurface::parse)
            .transpose()?;
        let latcor = wat2
            .and_then(extract_cor)
            .map(TnxSurface::parse)
            .transpose()?;
        if lngcor.is_none() && latcor.is_none() {
            Ok(None)
        } else {
            Ok(Some(Self { lngcor, latcor }))
        }
    }

    /// Forward distortion `(xi, eta) -> (xi + lngcor, eta + latcor)`.
    /// Surfaces that are absent contribute zero.
    #[must_use]
    pub fn forward(&self, xi: f64, eta: f64) -> (f64, f64) {
        let dxi = self.lngcor.as_ref().map_or(0.0, |s| s.eval(xi, eta));
        let deta = self.latcor.as_ref().map_or(0.0, |s| s.eval(xi, eta));
        (xi + dxi, eta + deta)
    }

    /// Inverse via Newton iteration on the forward map.
    /// Returns `Wcs` error on non-convergence or singular Jacobian.
    pub fn inverse(&self, xip: f64, etap: f64) -> Result<(f64, f64)> {
        let (mut xi, mut eta) = (xip, etap);
        let h = 1e-6_f64.max(1e-9 * (xip.abs() + etap.abs() + 1.0));
        for _ in 0..32 {
            let (fx, fy) = self.forward(xi, eta);
            let rx = fx - xip;
            let ry = fy - etap;
            if rx.abs() < 1e-13 && ry.abs() < 1e-13 {
                return Ok((xi, eta));
            }
            let (fxp, fyp) = self.forward(xi + h, eta);
            let (fxm, fym) = self.forward(xi - h, eta);
            let (fxe, fye) = self.forward(xi, eta + h);
            let (fxq, fyq) = self.forward(xi, eta - h);
            let j11 = (fxp - fxm) / (2.0 * h);
            let j21 = (fyp - fym) / (2.0 * h);
            let j12 = (fxe - fxq) / (2.0 * h);
            let j22 = (fye - fyq) / (2.0 * h);
            let det = j11 * j22 - j12 * j21;
            if det.abs() < 1e-15 {
                return Err(FitsError::Wcs(
                    "TNX: Jacobian singular during inverse iteration".into(),
                ));
            }
            let dxi = (j22 * rx - j12 * ry) / det;
            let deta = (-j21 * rx + j11 * ry) / det;
            xi -= dxi;
            eta -= deta;
            if dxi.abs() < 1e-13 && deta.abs() < 1e-13 {
                return Ok((xi, eta));
            }
        }
        Err(FitsError::Wcs(
            "TNX: inverse iteration did not converge".into(),
        ))
    }
}

/// Extract the body of an `lngcor = "..."` or `latcor = "..."` clause
/// from a reassembled WAT string. Returns the substring between the
/// double quotes, or `None` if the clause is absent.
fn extract_cor(wat: &str) -> Option<&str> {
    // Look for either `lngcor` or `latcor` followed by `= "...".`.
    let key = if wat.contains("lngcor") {
        "lngcor"
    } else if wat.contains("latcor") {
        "latcor"
    } else {
        return None;
    };
    let after = &wat[wat.find(key)? + key.len()..];
    let q1 = after.find('"')?;
    let rest = &after[q1 + 1..];
    let q2 = rest.find('"')?;
    Some(&rest[..q2])
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(extract_cor(s).unwrap(), "3 1 1 1 -1 1 -1 1 0.5");
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
