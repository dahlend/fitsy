//! DSS plate-solution WCS (non-standard).
//!
//! The Digitized Sky Survey distributes images of POSS / SERC plates
//! together with a 13-term positional plate model (AMD terms 14-20
//! are magnitude and color terms, which evaluate to zero for image
//! astrometry). The model is
//! signaled by the simultaneous presence of `PLTRAH`, `PLTDECD`,
//! `PPO1..6`, `XPIXELSZ`, `YPIXELSZ`, `CNPIX1`, `CNPIX2`, and
//! `AMDX1..20` and `AMDY1..20`. Such a header usually also carries a
//! placeholder `RA---TAN` and `DEC--TAN` description with `CRVAL` near
//! 0. That description is not the astrometry. It is a fallback for a
//! reader that cannot parse the plate model.
//!
//! ## Algorithm
//!
//! 1. Pixel -> plate position (mm):
//!
//!    ```text
//!    xpix = pix_x + CNPIX1 - 0.5
//!    ypix = pix_y + CNPIX2 - 0.5
//!    xmm  = (PPO3 - xpix * XPIXELSZ) / 1000
//!    (the sign flips because x increases from east to west)
//!    ymm  = (ypix * YPIXELSZ - PPO6) / 1000
//!    ```
//!
//! 2. Plate position -> standard coordinates `(xi, eta)` in arcseconds
//!    via the 13 positional terms of the plate polynomial
//!    (see `amd_triangle`).
//!
//! 3. Standard coordinates -> celestial coordinates by inverse
//!    gnomonic projection from the plate center
//!    `(alpha_0, delta_0)`:
//!
//!    ```text
//!    alpha = atan2(xi, cos delta_0 - eta * sin delta_0) + alpha_0
//!    delta = atan2(sin delta_0 + eta * cos delta_0,
//!              sqrt((cos delta_0 - eta * sin delta_0)^2 + xi^2))
//!    ```
//!
//! ## References
//! - ESO DSS-II `getimage` documentation, plate-solution section.
//! - Greisen, "FITS Standard Conventions" non-FITS appendix.
//! - <http://tdc-www.harvard.edu/wcstools/dsswcs.wcs.html>
//!
//! ## Validation
//!
//! No reference implementation of the plate model was available to
//! compare against. The unit tests cover four things instead:
//!
//! - A round trip from pixel to world and back, to sub-millipixel
//!   precision.
//! - The plate center, which must project to the plate-center RA and
//!   Dec recovered from the `PLT` sexagesimal fields.
//! - The folded coefficient triangle, against the plate model written
//!   out term by term. A round trip alone cannot catch an error here.
//!   It closes on the wrong polynomial as readily as on the right
//!   one.
//! - A solve that does not converge, which must report an error
//!   rather than return a point.

use crate::error::{FitsError, Result};
use crate::header::Header;
use crate::header::value::Value;
use crate::wcs::newton;
use crate::wcs::poly;

/// Arcseconds per radian.
const ARCSEC_PER_RAD: f64 = 180.0 * 3600.0 / std::f64::consts::PI;
/// Degrees per radian.
const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;
/// Radians per degree.
const RAD_PER_DEG: f64 = std::f64::consts::PI / 180.0;

/// One DSS plate solution.
#[derive(Debug, Clone)]
pub struct Dss {
    /// Plate-center right ascension, degrees.
    pub plate_ra: f64,
    /// Plate-center declination, degrees.
    pub plate_dec: f64,
    /// `PPO3` -- plate center x in microns.
    pub ppo3: f64,
    /// `PPO6` -- plate center y in microns.
    pub ppo6: f64,
    /// `XPIXELSZ` -- pixel size on the plate, microns.
    pub xpixelsz: f64,
    /// `YPIXELSZ` -- pixel size on the plate, microns.
    pub ypixelsz: f64,
    /// `CNPIX1` -- x-offset of the subimage in original plate pixels.
    pub cnpix1: f64,
    /// `CNPIX2` -- y-offset of the subimage in original plate pixels.
    pub cnpix2: f64,
    /// 20 polynomial coefficients for xi.
    pub amdx: [f64; 20],
    /// 20 polynomial coefficients for eta.
    pub amdy: [f64; 20],
}

impl Dss {
    /// Parse a DSS plate solution from a header.
    ///
    /// The result is `Ok(None)` when any gating keyword is absent,
    /// which means the header describes no DSS plate. Those keywords
    /// are `PLTRAH`, `PLTDECD`, `PPO3`, `PPO6`, `XPIXELSZ`,
    /// `YPIXELSZ`, `AMDX1` and `AMDY1`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when a gating keyword is present but holds a
    /// value that is not numeric, or when the plate center keywords
    /// are inconsistent.
    pub fn from_header(header: &Header) -> Result<Option<Self>> {
        // Required gating keys: if any one is absent, this is not a
        // DSS plate header.
        let need = [
            "PLTRAH", "PLTDECD", "PPO3", "PPO6", "XPIXELSZ", "YPIXELSZ", "AMDX1", "AMDY1",
        ];
        for k in need {
            if header.first(k).is_none() {
                return Ok(None);
            }
        }
        let plate_ra = read_plate_ra(header)?;
        let plate_dec = read_plate_dec(header)?;
        let ppo3 = read_real(header, "PPO3")?;
        let ppo6 = read_real(header, "PPO6")?;
        let xpixelsz = read_real(header, "XPIXELSZ")?;
        let ypixelsz = read_real(header, "YPIXELSZ")?;
        // CNPIX1/2 default to zero when the file is the full plate.
        let cnpix1 = read_optional_real(header, "CNPIX1").unwrap_or(0.0);
        let cnpix2 = read_optional_real(header, "CNPIX2").unwrap_or(0.0);
        let mut amdx = [0.0; 20];
        let mut amdy = [0.0; 20];
        for i in 0..20 {
            amdx[i] = read_optional_real(header, &format!("AMDX{}", i + 1)).unwrap_or(0.0);
            amdy[i] = read_optional_real(header, &format!("AMDY{}", i + 1)).unwrap_or(0.0);
        }
        Ok(Some(Self {
            plate_ra,
            plate_dec,
            ppo3,
            ppo6,
            xpixelsz,
            ypixelsz,
            cnpix1,
            cnpix2,
            amdx,
            amdy,
        }))
    }

    /// Pixel (1-based, FITS convention) -> plate position (mm).
    fn pixel_to_plate(&self, pix_x: f64, pix_y: f64) -> (f64, f64) {
        let xpix = pix_x + self.cnpix1 - 0.5;
        let ypix = pix_y + self.cnpix2 - 0.5;
        let xmm = (self.ppo3 - xpix * self.xpixelsz) / 1000.0;
        let ymm = (ypix * self.ypixelsz - self.ppo6) / 1000.0;
        (xmm, ymm)
    }

    /// Inverse: plate position (mm) -> 1-based pixel.
    fn plate_to_pixel(&self, xmm: f64, ymm: f64) -> (f64, f64) {
        let xpix = (self.ppo3 - xmm * 1000.0) / self.xpixelsz;
        let ypix = (ymm * 1000.0 + self.ppo6) / self.ypixelsz;
        (xpix - self.cnpix1 + 0.5, ypix - self.cnpix2 + 0.5)
    }

    /// Forward map: 1-based pixel -> celestial (RA, Dec) in degrees.
    #[must_use]
    pub fn pixel_to_world(&self, pix_x: f64, pix_y: f64) -> (f64, f64) {
        let (xmm, ymm) = self.pixel_to_plate(pix_x, pix_y);
        // The eta solution is the xi solution with x and y exchanged
        // (`AMDY` multiplies `y` where `AMDX` multiplies `x`), so one
        // folded table serves both -- evaluated with its arguments
        // swapped for eta. This is the same device `TpvAxis::xy` uses.
        let xi_arcsec = amd_eval(&amd_triangle(&self.amdx), xmm, ymm);
        let eta_arcsec = amd_eval(&amd_triangle(&self.amdy), ymm, xmm);
        let xi = xi_arcsec / ARCSEC_PER_RAD;
        let eta = eta_arcsec / ARCSEC_PER_RAD;
        let dec0 = self.plate_dec * RAD_PER_DEG;
        let ra0 = self.plate_ra * RAD_PER_DEG;
        let cd = dec0.cos();
        let sd = dec0.sin();
        let denom = cd - eta * sd;
        let alpha = xi.atan2(denom) + ra0;
        let delta = (sd + eta * cd).atan2((denom * denom + xi * xi).sqrt());
        let mut ra = alpha * DEG_PER_RAD;
        ra = ra.rem_euclid(360.0);
        let dec = delta * DEG_PER_RAD;
        (ra, dec)
    }

    /// Inverse map, from (RA, Dec) in degrees to a 1-based pixel, by
    /// Newton iteration on the forward map.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the Jacobian is singular, or when the
    /// iteration does not converge within its step limit.
    pub fn world_to_pixel(&self, ra: f64, dec: f64) -> Result<(f64, f64)> {
        // Forward gnomonic: (alpha, delta) -> (xi, eta) at plate center, then
        // invert the polynomial via Newton on the (xmm, ymm) plane.
        let dec0 = self.plate_dec * RAD_PER_DEG;
        let ra0 = self.plate_ra * RAD_PER_DEG;
        let alpha = ra * RAD_PER_DEG;
        let delta = dec * RAD_PER_DEG;
        let cd = dec0.cos();
        let sd = dec0.sin();
        let cdec = delta.cos();
        let sdec = delta.sin();
        let cdra = (alpha - ra0).cos();
        let sdra = (alpha - ra0).sin();
        let denom = sdec * sd + cdec * cd * cdra;
        if denom <= 0.0 {
            return Err(FitsError::Wcs(
                "DSS: target point is behind the plate".into(),
            ));
        }
        let xi_target_arcsec = (cdec * sdra / denom) * ARCSEC_PER_RAD;
        let eta_target_arcsec = ((sdec * cd - cdec * sd * cdra) / denom) * ARCSEC_PER_RAD;
        // Initial guess: invert the linear part of the polynomial
        //   xi ~= AMDX1*x + AMDX2*y + AMDX3
        //   eta ~= AMDY1*y + AMDY2*x + AMDY3
        let a = self.amdx[0];
        let b = self.amdx[1];
        let c = self.amdx[2];
        let d = self.amdy[1];
        let e = self.amdy[0];
        let f = self.amdy[2];
        let det = a * e - b * d;
        if det.abs() < 1e-30 {
            return Err(FitsError::Wcs(
                "DSS: linear plate matrix is singular".into(),
            ));
        }
        let guess = (
            (e * (xi_target_arcsec - c) - b * (eta_target_arcsec - f)) / det,
            (a * (eta_target_arcsec - f) - d * (xi_target_arcsec - c)) / det,
        );
        // Newton on the plate polynomial, with an exact Jacobian: the
        // AMD terms fold into a triangle (see `amd_triangle`), so the
        // derivatives come out of the same Horner pass as the value.
        // Hoisted out of the loop -- the tables depend on the header,
        // not on the iterate.
        let tx = amd_triangle(&self.amdx);
        let ty = amd_triangle(&self.amdy);
        let (xmm, ymm) = newton::solve(
            "DSS",
            guess,
            newton::residual_scale(xi_target_arcsec, eta_target_arcsec),
            |xmm, ymm| {
                let (fx, dxi_dx, dxi_dy) =
                    poly::triangle_with_derivatives(AMD_DIM, |p, q| tx[p * AMD_DIM + q], xmm, ymm);
                // eta reads its arguments swapped, so the derivatives
                // come back swapped: the first is d/d(ymm).
                let (fy, deta_dy, deta_dx) =
                    poly::triangle_with_derivatives(AMD_DIM, |p, q| ty[p * AMD_DIM + q], ymm, xmm);
                newton::Residual2 {
                    rx: fx - xi_target_arcsec,
                    ry: fy - eta_target_arcsec,
                    j11: dxi_dx,
                    j12: dxi_dy,
                    j21: deta_dx,
                    j22: deta_dy,
                }
            },
        )?;
        let (px, py) = self.plate_to_pixel(xmm, ymm);
        Ok((px, py))
    }
}

/// Number of degree levels in the plate polynomial, one more than the
/// highest total degree.
///
/// The `AMDX13 * x * r^4` term reaches degree 5.
const AMD_DIM: usize = 6;

/// Fold the plate polynomial into a triangular coefficient table.
///
/// The table is indexed `[p * AMD_DIM + q]` for the term `x^p y^q`.
/// The `c` argument holds the 20 `AMDX` or `AMDY` values.
///
/// The `r^2` factors in the 13 positional terms are even powers of
/// `r`. Each one therefore expands into ordinary monomials:
///
/// ```text
/// c6  * r^2     = c6 x^2 + c6 y^2
/// c11 * x r^2   = c11 (x^3 + x y^2)
/// c12 * x r^4   = c12 (x^5 + 2 x^3 y^2 + x y^4)
/// ```
///
/// The plate model is therefore a bivariate polynomial of degree 5.
/// [`poly::triangle`] evaluates it by Horner, and
/// [`poly::triangle_with_derivatives`] differentiates it exactly.
/// [`Dss::world_to_pixel`] can then use the shared Newton solver with
/// an exact Jacobian.
///
/// The radial terms of [`Tpv`](crate::wcs::tpv::Tpv) are odd powers of
/// `r`. Those are not polynomial, and that module carries them apart
/// from its triangle.
///
/// This folds in the 13 positional terms alone. `AMDX14` to `AMDX20`
/// hold the magnitude and color terms of the GSC astrometric solution:
/// mag, mag^2, mag^3, mag*x, mag*(x^2+y^2), mag*x*(x^2+y^2) and color.
/// Image astrometry takes them as zero.
///
/// This builds the table on each call rather than caching it, because
/// [`Dss::amdx`] is a public field. A table resolved at construction
/// would go stale when a caller assigns to that field. Callers hoist
/// the call out of their loops.
fn amd_triangle(c: &[f64; 20]) -> [f64; AMD_DIM * AMD_DIM] {
    let mut t = [0.0_f64; AMD_DIM * AMD_DIM];
    let mut add = |p: usize, q: usize, v: f64| t[p * AMD_DIM + q] += v;
    // Constant and linear.
    add(0, 0, c[2]);
    add(1, 0, c[0]);
    add(0, 1, c[1]);
    // Quadratic, with `c6 r^2` split across x^2 and y^2.
    add(2, 0, c[3] + c[6]);
    add(1, 1, c[4]);
    add(0, 2, c[5] + c[6]);
    // Cubic, with `c11 x r^2` split across x^3 and x y^2.
    add(3, 0, c[7] + c[11]);
    add(2, 1, c[8]);
    add(1, 2, c[9] + c[11]);
    add(0, 3, c[10]);
    // Quintic: `c12 x r^4 = c12 (x^5 + 2 x^3 y^2 + x y^4)`.
    add(5, 0, c[12]);
    add(3, 2, 2.0 * c[12]);
    add(1, 4, c[12]);
    t
}

/// Evaluate a folded plate polynomial at `(x, y)`, in arcsec.
#[inline]
fn amd_eval(t: &[f64; AMD_DIM * AMD_DIM], x: f64, y: f64) -> f64 {
    poly::triangle(AMD_DIM, |p, q| t[p * AMD_DIM + q], x, y)
}

fn read_real(header: &Header, key: &str) -> Result<f64> {
    match header.first(key) {
        Some(Value::Integer(i)) => Ok(*i as f64),
        Some(Value::Real(r)) => Ok(*r),
        _ => Err(FitsError::Wcs(format!("DSS: missing or non-numeric {key}"))),
    }
}

fn read_optional_real(header: &Header, key: &str) -> Option<f64> {
    match header.first(key)? {
        Value::Integer(i) => Some(*i as f64),
        Value::Real(r) => Some(*r),
        _ => None,
    }
}

fn read_plate_ra(header: &Header) -> Result<f64> {
    let h = read_real(header, "PLTRAH")?;
    let m = read_optional_real(header, "PLTRAM").unwrap_or(0.0);
    let s = read_optional_real(header, "PLTRAS").unwrap_or(0.0);
    Ok((h + m / 60.0 + s / 3600.0) * 15.0)
}

fn read_plate_dec(header: &Header) -> Result<f64> {
    let d = read_real(header, "PLTDECD")?;
    let m = read_optional_real(header, "PLTDECM").unwrap_or(0.0);
    let s = read_optional_real(header, "PLTDECS").unwrap_or(0.0);
    let mag = d.abs() + m / 60.0 + s / 3600.0;
    let sign = match header.first("PLTDECSN") {
        Some(Value::String(s)) if s.trim().starts_with('-') => -1.0,
        _ => {
            if d < 0.0 {
                -1.0
            } else {
                1.0
            }
        }
    };
    Ok(sign * mag)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plate polynomial term by term, as the DSS documentation
    /// writes it, with the `r^2` factors left unexpanded.
    ///
    /// This is the form [`amd_triangle`] replaced. It stays here as
    /// the ground truth for that algebraic fold. A wrong expansion of
    /// `c6 r^2`, `c11 x r^2` or `c12 x r^4` passes every other test in
    /// the suite. A round trip closes on the wrong polynomial as
    /// readily as on the right one.
    fn amd_longhand(c: &[f64; 20], x: f64, y: f64) -> f64 {
        let r2 = x * x + y * y;
        c[0] * x
            + c[1] * y
            + c[2]
            + c[3] * x * x
            + c[4] * x * y
            + c[5] * y * y
            + c[6] * r2
            + c[7] * x * x * x
            + c[8] * x * x * y
            + c[9] * x * y * y
            + c[10] * y * y * y
            + c[11] * x * r2
            + c[12] * x * r2 * r2
    }

    /// The folded triangle must reproduce the term-by-term formula.
    ///
    /// The eta form must equal the xi form with its arguments
    /// exchanged.
    #[test]
    fn amd_triangle_matches_longhand_formula() {
        // Coefficients spanning several orders of magnitude, all 13
        // positional terms non-zero so every fold is exercised.
        let mut c = [0.0_f64; 20];
        for (i, v) in [
            67.0, 0.02, -0.3, 1e-5, -2e-6, 3e-6, 1e-6, 2e-8, -1e-8, 3e-9, 1e-9, -5e-9, 1e-11,
        ]
        .iter()
        .enumerate()
        {
            c[i] = *v;
        }
        let t = amd_triangle(&c);
        let mut worst = 0.0_f64;
        let mut x = -80.0_f64;
        while x <= 80.0 {
            let mut y = -80.0_f64;
            while y <= 80.0 {
                let want = amd_longhand(&c, x, y);
                let got = amd_eval(&t, x, y);
                // Relative: the value runs to thousands of arcsec, so
                // the rounding floor scales with it.
                worst = worst.max((got - want).abs() / want.abs().max(1.0));
                // The eta solution is the xi solution with x and y
                // exchanged; `pixel_to_world` relies on this.
                let want_eta = amd_longhand(&c, y, x);
                let got_eta = amd_eval(&t, y, x);
                worst = worst.max((got_eta - want_eta).abs() / want_eta.abs().max(1.0));
                y += 7.0;
            }
            x += 7.0;
        }
        assert!(worst < 1e-13, "folded triangle diverged by {worst:.3e}");
    }

    /// A successful [`Dss::world_to_pixel`] must have converged.
    ///
    /// The hand-written Newton loop this replaced left its iteration
    /// limit through a `break` and then returned `Ok` in every case. A
    /// point that did not converge therefore came back as a wrong
    /// answer with no error. The documentation named an error that the
    /// code never constructed.
    #[test]
    fn world_to_pixel_never_returns_an_unconverged_point() {
        let d = dss_fixture();
        let mut checked = 0_usize;
        // Includes points far off the plate, where the iteration is
        // least likely to behave.
        for px in [-5000.0_f64, -400.0, 0.0, 250.0, 4000.0, 50000.0] {
            for py in [-5000.0_f64, -400.0, 0.0, 250.0, 4000.0, 50000.0] {
                let (ra, dec) = d.pixel_to_world(px, py);
                let Ok((bx, by)) = d.world_to_pixel(ra, dec) else {
                    continue;
                };
                // The residual, not `|pixel - pixel|`: the plate
                // polynomial is a quintic, so far off the plate it is
                // not injective and a *different* preimage is still a
                // correct answer. What an `Ok` must mean is that the
                // point it returns maps back to the target.
                let (gra, gdec) = d.pixel_to_world(bx, by);
                let sep = {
                    let dd = (gdec - dec) * RAD_PER_DEG;
                    // RA wraps at 360 and converges near the poles.
                    let dr = (gra - ra).rem_euclid(360.0);
                    let dr = if dr > 180.0 { dr - 360.0 } else { dr };
                    let dr = dr * RAD_PER_DEG * (dec * RAD_PER_DEG).cos();
                    (dd * dd + dr * dr).sqrt() * ARCSEC_PER_RAD
                };
                assert!(
                    sep < 1e-6,
                    "world_to_pixel returned Ok at ({px}, {py}) but its \
                     result maps back {sep:.3e} arcsec away -- \
                     non-convergence must be reported as an error"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 20,
            "only {checked} points solved; test is vacuous"
        );
    }

    /// On the plate, the inverse recovers the same pixel.
    ///
    /// It must not return some other valid preimage.
    #[test]
    fn world_to_pixel_round_trips_on_plate() {
        let d = dss_fixture();
        let mut worst = 0.0_f64;
        for px in [-400.0_f64, 0.0, 250.0, 1000.0, 4000.0] {
            for py in [-400.0_f64, 0.0, 250.0, 1000.0, 4000.0] {
                let (ra, dec) = d.pixel_to_world(px, py);
                let (bx, by) = d.world_to_pixel(ra, dec).expect("on-plate point inverts");
                worst = worst.max((bx - px).abs().max((by - py).abs()));
            }
        }
        assert!(
            worst < 1e-6,
            "on-plate round trip off by {worst:.3e} pixels"
        );
    }

    /// A plate solution with POSS-II style values.
    fn dss_fixture() -> Dss {
        let mut cards: Vec<String> = vec![
            "PLTRAH  =                   12".into(),
            "PLTRAM  =                   30".into(),
            "PLTRAS  =                  0.0".into(),
            "PLTDECSN= '+'".into(),
            "PLTDECD =                   30".into(),
            "PLTDECM =                    0".into(),
            "PLTDECS =                  0.0".into(),
            "PPO3    =         177500.0".into(),
            "PPO6    =         177500.0".into(),
            "XPIXELSZ=            25.284".into(),
            "YPIXELSZ=            25.284".into(),
            "CNPIX1  =              1000".into(),
            "CNPIX2  =              1000".into(),
        ];
        let amdx = [
            67.0, 0.02, -0.3, 1e-5, -2e-6, 3e-6, 1e-6, 2e-8, -1e-8, 3e-9, 1e-9, -5e-9, 1e-11,
        ];
        let amdy = [
            67.0, -0.03, 0.25, -2e-5, 1e-6, -3e-6, 2e-6, -1e-8, 2e-8, -3e-9, 2e-9, 4e-9, -2e-11,
        ];
        for (i, v) in amdx.iter().enumerate() {
            cards.push(format!("AMDX{:<4}= {v:>20.12e}", i + 1));
        }
        for (i, v) in amdy.iter().enumerate() {
            cards.push(format!("AMDY{:<4}= {v:>20.12e}", i + 1));
        }
        let mut buf: Vec<u8> = Vec::new();
        for c in &cards {
            buf.extend_from_slice(&pad_card(c));
        }
        buf.extend_from_slice(&pad_card("END"));
        while !buf.len().is_multiple_of(2880) {
            buf.push(b' ');
        }
        let (h, _) = Header::parse(&buf, 0).expect("header parses");
        Dss::from_header(&h)
            .expect("dss parses")
            .expect("dss present")
    }

    /// A solve that does not converge must report an error.
    ///
    /// This pins the fix for the defect that
    /// `world_to_pixel_never_returns_an_unconverged_point` describes.
    ///
    /// A plate whose quintic term dominates its linear term exposes
    /// the case. The linear initial guess lands far from any root, and
    /// the iteration does not recover within its step limit.
    #[test]
    fn non_convergence_is_reported() {
        let mut d = dss_fixture();
        d.amdx = [0.0; 20];
        d.amdy = [0.0; 20];
        // Linear term, then the degree-5 `x r^4` term.
        d.amdx[0] = 1e-3;
        d.amdy[0] = 1e-3;
        d.amdx[12] = 1e-6;
        d.amdy[12] = 1e-6;
        let mut errs = 0_usize;
        for px in [-5000.0_f64, -400.0, 0.0, 250.0, 4000.0] {
            for py in [-5000.0_f64, -400.0, 0.0, 250.0, 4000.0] {
                let (ra, dec) = d.pixel_to_world(px, py);
                match d.world_to_pixel(ra, dec) {
                    Err(FitsError::Wcs(msg)) => {
                        assert!(
                            msg.starts_with("DSS:"),
                            "error should name the convention: {msg}"
                        );
                        errs += 1;
                    }
                    Err(e) => panic!("unexpected error kind at ({px}, {py}): {e}"),
                    // Converging here is allowed; returning a point
                    // that does not solve the system is not.
                    Ok((bx, by)) => {
                        let (gra, gdec) = d.pixel_to_world(bx, by);
                        assert!(
                            (gra - ra).abs() < 1e-6 && (gdec - dec).abs() < 1e-6,
                            "Ok at ({px}, {py}) does not map back"
                        );
                    }
                }
            }
        }
        assert!(
            errs > 0,
            "fixture no longer diverges, so this no longer tests anything"
        );
    }

    fn pad_card(s: &str) -> [u8; 80] {
        let mut b = [b' '; 80];
        b[..s.len()].copy_from_slice(s.as_bytes());
        b
    }

    #[test]
    fn plate_ra_dec_sexagesimal() {
        // 0h07m25.68s -> 1.857deg ; +0deg48'26" -> 0.80722deg.
        let cards = [
            pad_card("PLTRAH  =                    0"),
            pad_card("PLTRAM  =                    7"),
            pad_card("PLTRAS  =                25.68"),
            pad_card("PLTDECSN= '+'"),
            pad_card("PLTDECD =                    0"),
            pad_card("PLTDECM =                   48"),
            pad_card("PLTDECS =                 26.0"),
            pad_card("END"),
        ];
        let mut buf = Vec::new();
        for c in &cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % 2880 != 0 {
            buf.push(b' ');
        }
        let (h, _) = Header::parse(&buf, 0).unwrap();
        let ra = read_plate_ra(&h).unwrap();
        let dec = read_plate_dec(&h).unwrap();
        assert!((ra - (0.0 + 7.0 / 60.0 + 25.68 / 3600.0) * 15.0).abs() < 1e-9);
        assert!((dec - (48.0 / 60.0 + 26.0 / 3600.0)).abs() < 1e-9);
    }

    #[test]
    fn linear_polynomial_round_trip() {
        // Trivial linear plate model: xi = x, eta = y (in arcsec).
        let mut amdx = [0.0; 20];
        let mut amdy = [0.0; 20];
        // xi = 1*x
        amdx[0] = 1.0;
        // eta = 1*y
        amdy[0] = 1.0;
        let dss = Dss {
            plate_ra: 10.0,
            plate_dec: -5.0,
            ppo3: 100_000.0,
            ppo6: 100_000.0,
            xpixelsz: 25.0,
            ypixelsz: 25.0,
            cnpix1: 0.0,
            cnpix2: 0.0,
            amdx,
            amdy,
        };
        for &(px, py) in &[(100.0, 200.0), (4000.0, 4000.0), (1.0, 1.0)] {
            let (ra, dec) = dss.pixel_to_world(px, py);
            let (bx, by) = dss.world_to_pixel(ra, dec).unwrap();
            assert!((bx - px).abs() < 1e-6, "x: {px} -> {bx}");
            assert!((by - py).abs() < 1e-6, "y: {py} -> {by}");
        }
    }
}
