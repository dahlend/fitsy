//! DSS plate-solution WCS (non-standard).
//!
//! The Digitized Sky Survey distributes images of POSS / SERC plates
//! together with a 20-term astrometric plate model. The model is
//! signalled by the simultaneous presence of `PLTRAH`, `PLTDECD`,
//! `PPO1..6`, `XPIXELSZ`, `YPIXELSZ`, `CNPIX1`, `CNPIX2`, and
//! `AMDX1..20` / `AMDY1..20`. Headers usually also carry a dummy
//! `RA---TAN` / `DEC--TAN` description with `CRVAL` ~= 0, but that
//! description is **not** the astrometry -- it is a fallback for
//! readers that cannot parse the plate model.
//!
//! ## Algorithm
//!
//! 1. Pixel -> plate position (mm):
//!
//!    ```text
//!    xpix = pix_x + CNPIX1 - 0.5
//!    ypix = pix_y + CNPIX2 - 0.5
//!    xmm  = (PPO3 - xpix * XPIXELSZ) / 1000   (sign flip: x increases east -> west)
//!    ymm  = (ypix * YPIXELSZ - PPO6) / 1000
//!    ```
//!
//! 2. Plate position -> standard coordinates `(xi, eta)` in arcseconds
//!    via the 20-term polynomial (see `amd_xi` / `amd_eta`).
//!
//! 3. Standard coordinates -> celestial coordinates by inverse
//!    gnomonic projection from the plate center
//!    `(alpha_0, delta_0)`:
//!
//!    ```text
//!    alpha = atan2(-xi, cos delta_0 - eta * sin delta_0) + alpha_0
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
//! Astropy does not implement the DSS plate model -- it falls back to
//! the dummy TAN. We validate by (a) round-tripping
//! `pix -> world -> pix` to sub-millipixel precision and (b) checking
//! that the plate center projects to the plate-center RA/Dec
//! recovered from the `PLT*` sexagesimal fields.

use crate::error::{FitsError, Result};
use crate::header::Header;
use crate::header::value::Value;

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
    /// Parse from a header. Returns `Ok(None)` when the required
    /// plate keywords are not all present.
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
        let xi_arcsec = amd_xi(&self.amdx, xmm, ymm);
        let eta_arcsec = amd_eta(&self.amdy, xmm, ymm);
        let xi = xi_arcsec / ARCSEC_PER_RAD;
        let eta = eta_arcsec / ARCSEC_PER_RAD;
        let dec0 = self.plate_dec * RAD_PER_DEG;
        let ra0 = self.plate_ra * RAD_PER_DEG;
        let cd = dec0.cos();
        let sd = dec0.sin();
        let denom = cd - eta * sd;
        let alpha = (-xi).atan2(denom) + ra0;
        let delta = (sd + eta * cd).atan2((denom * denom + xi * xi).sqrt());
        let mut ra = alpha * DEG_PER_RAD;
        ra = ra.rem_euclid(360.0);
        let dec = delta * DEG_PER_RAD;
        (ra, dec)
    }

    /// Inverse map: (RA, Dec) in degrees -> 1-based pixel via Newton
    /// iteration on the forward map.
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
        let xi_target_arcsec = (-cdec * sdra / denom) * ARCSEC_PER_RAD;
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
        let mut xmm = (e * (xi_target_arcsec - c) - b * (eta_target_arcsec - f)) / det;
        let mut ymm = (a * (eta_target_arcsec - f) - d * (xi_target_arcsec - c)) / det;
        // Newton iteration on the 20-term polynomial.
        for _ in 0..32 {
            let fx = amd_xi(&self.amdx, xmm, ymm) - xi_target_arcsec;
            let fy = amd_eta(&self.amdy, xmm, ymm) - eta_target_arcsec;
            if fx.abs() < 1e-9 && fy.abs() < 1e-9 {
                break;
            }
            // Numerical Jacobian (the analytic version is messy).
            let h = 1e-4_f64.max(1e-9 * (xmm.abs() + ymm.abs() + 1.0));
            let jxx =
                (amd_xi(&self.amdx, xmm + h, ymm) - amd_xi(&self.amdx, xmm - h, ymm)) / (2.0 * h);
            let jxy =
                (amd_xi(&self.amdx, xmm, ymm + h) - amd_xi(&self.amdx, xmm, ymm - h)) / (2.0 * h);
            let jyx =
                (amd_eta(&self.amdy, xmm + h, ymm) - amd_eta(&self.amdy, xmm - h, ymm)) / (2.0 * h);
            let jyy =
                (amd_eta(&self.amdy, xmm, ymm + h) - amd_eta(&self.amdy, xmm, ymm - h)) / (2.0 * h);
            let jdet = jxx * jyy - jxy * jyx;
            if jdet.abs() < 1e-30 {
                return Err(FitsError::Wcs(
                    "DSS: Jacobian singular during inverse iteration".into(),
                ));
            }
            let dx = (jyy * fx - jxy * fy) / jdet;
            let dy = (-jyx * fx + jxx * fy) / jdet;
            xmm -= dx;
            ymm -= dy;
            if dx.abs() < 1e-12 && dy.abs() < 1e-12 {
                break;
            }
        }
        let (px, py) = self.plate_to_pixel(xmm, ymm);
        Ok((px, py))
    }
}

/// 20-term DSS plate polynomial for xi (arcsec). `x`, `y` are plate
/// position in mm relative to the plate center.
fn amd_xi(c: &[f64; 20], x: f64, y: f64) -> f64 {
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
        + c[13] * x.powi(5)
        + c[14] * x.powi(4) * y
        + c[15] * x.powi(3) * y * y
        + c[16] * x * x * y.powi(3)
        + c[17] * x * y.powi(4)
        + c[18] * y.powi(5)
        + c[19] * x * r2 * r2 * r2
}

/// 20-term DSS plate polynomial for eta (arcsec). Same monomial set as
/// [`amd_xi`] but with x and y swapped.
fn amd_eta(c: &[f64; 20], x: f64, y: f64) -> f64 {
    let r2 = x * x + y * y;
    c[0] * y
        + c[1] * x
        + c[2]
        + c[3] * y * y
        + c[4] * y * x
        + c[5] * x * x
        + c[6] * r2
        + c[7] * y * y * y
        + c[8] * y * y * x
        + c[9] * y * x * x
        + c[10] * x * x * x
        + c[11] * y * r2
        + c[12] * y * r2 * r2
        + c[13] * y.powi(5)
        + c[14] * y.powi(4) * x
        + c[15] * y.powi(3) * x * x
        + c[16] * y * y * x.powi(3)
        + c[17] * y * x.powi(4)
        + c[18] * x.powi(5)
        + c[19] * y * r2 * r2 * r2
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
