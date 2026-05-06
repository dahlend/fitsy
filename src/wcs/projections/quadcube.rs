//! Quadrilateralized cube projections -- Paper II Sec.5.6: TSC, CSC, QSC.

use crate::error::{FitsError, Result};
use crate::wcs::projection::Projection;
use crate::wcs::{D2R, R2D};

// -- TSC --------------------------------------------------------------

/// Tangential spherical cube `TSC` (Paper II Sec.5.6.1, eqs. 39-43).
/// Six tangent-plane faces in a +-shape; each edge 90deg.
#[derive(Debug, Clone, Copy)]
pub struct Tsc;
impl Tsc {
    /// Compute (face, xi, eta) for native coords. Face 0=+Z (north),
    /// 1..4=equatorial (phi = 0, 90deg, 180deg, 270deg), 5=-Z (south).
    fn face_coords(phi: f64, theta: f64) -> (u32, f64, f64) {
        let p = phi * D2R;
        let t = theta * D2R;
        let ct = t.cos();
        let l = ct * p.cos();
        let m = ct * p.sin();
        let n = t.sin();
        if n >= l.abs() && n >= m.abs() {
            (0, m / n, -l / n)
        } else if -n >= l.abs() && -n >= m.abs() {
            (5, m / (-n), l / (-n))
        } else if l >= m.abs() && l >= n.abs() {
            (1, m / l, n / l)
        } else if m >= l.abs() && m >= n.abs() {
            (2, -l / m, n / m)
        } else if -l >= m.abs() && -l >= n.abs() {
            (3, -m / (-l), n / (-l))
        } else {
            (4, l / (-m), n / (-m))
        }
    }
    fn face_to_xy(face: u32, xi: f64, eta: f64) -> (f64, f64) {
        let (cx, cy) = match face {
            0 => (0.0, 90.0),
            1 => (0.0, 0.0),
            2 => (-90.0, 0.0),
            3 => (-180.0, 0.0),
            4 => (-270.0, 0.0),
            5 => (0.0, -90.0),
            _ => unreachable!(),
        };
        (cx + xi * 45.0, cy + eta * 45.0)
    }
    fn xy_to_face(x: f64, y: f64) -> Option<(u32, f64, f64)> {
        let in_unit = |a: f64| a.abs() <= 1.0 + 1e-9;
        for (face, cx, cy) in [
            (0_u32, 0.0_f64, 90.0_f64),
            (1, 0.0, 0.0),
            (2, -90.0, 0.0),
            (3, -180.0, 0.0),
            (4, -270.0, 0.0),
            (5, 0.0, -90.0),
        ] {
            let xi = (x - cx) / 45.0;
            let eta = (y - cy) / 45.0;
            if in_unit(xi) && in_unit(eta) {
                return Some((face, xi.clamp(-1.0, 1.0), eta.clamp(-1.0, 1.0)));
            }
        }
        None
    }
    fn face_to_lmn(face: u32, xi: f64, eta: f64) -> (f64, f64, f64) {
        let norm = (1.0 + xi * xi + eta * eta).sqrt();
        match face {
            0 => (-eta / norm, xi / norm, 1.0 / norm),
            1 => (1.0 / norm, xi / norm, eta / norm),
            2 => (-xi / norm, 1.0 / norm, eta / norm),
            3 => (-1.0 / norm, -xi / norm, eta / norm),
            4 => (xi / norm, -1.0 / norm, eta / norm),
            5 => (eta / norm, xi / norm, -1.0 / norm),
            _ => unreachable!(),
        }
    }
}
impl Projection for Tsc {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let (face, xi, eta) = Self::face_coords(phi, theta);
        Ok(Self::face_to_xy(face, xi, eta))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (face, xi, eta) = Self::xy_to_face(x, y)
            .ok_or_else(|| FitsError::Wcs(format!("TSC: ({x}, {y}) is outside the cube layout")))?;
        let (l, m, n) = Self::face_to_lmn(face, xi, eta);
        let theta = n.clamp(-1.0, 1.0).asin() * R2D;
        let phi = if l == 0.0 && m == 0.0 {
            0.0
        } else {
            m.atan2(l) * R2D
        };
        Ok((phi, theta))
    }
}

// -- CSC --------------------------------------------------------------

// Chan & O'Neill (1975) forward polynomial coefficients.
const CSC_GSTAR: f64 = 1.37484847732;
const CSC_M: f64 = 0.004869491981;
const CSC_GAMMA: f64 = -0.13161671474;
const CSC_OMEGA1: f64 = -0.159596235474;
const CSC_D0: f64 = 0.0759196200467;
const CSC_D1: f64 = -0.0217762490699;
const CSC_C00: f64 = 0.141189631152;
const CSC_C10: f64 = 0.0809701286525;
const CSC_C01: f64 = -0.281528535557;
const CSC_C11: f64 = 0.15384112876;
const CSC_C20: f64 = -0.178251207466;
const CSC_C02: f64 = 0.106959469314;

// Inverse polynomial coefficients (deprojection).
const CSC_P00: f64 = -0.27292696;
const CSC_P10: f64 = -0.07629969;
const CSC_P20: f64 = -0.22797056;
const CSC_P30: f64 = 0.54852384;
const CSC_P40: f64 = -0.62930065;
const CSC_P50: f64 = 0.25795794;
const CSC_P60: f64 = 0.02584375;
const CSC_P01: f64 = -0.02819452;
const CSC_P11: f64 = -0.01471565;
const CSC_P21: f64 = 0.48051512;
const CSC_P31: f64 = -1.74114454;
const CSC_P41: f64 = 1.71547508;
const CSC_P51: f64 = -0.53022337;
const CSC_P02: f64 = 0.27058160;
const CSC_P12: f64 = -0.56800938;
const CSC_P22: f64 = 0.30803317;
const CSC_P32: f64 = 0.98938102;
const CSC_P42: f64 = -0.83180469;
const CSC_P03: f64 = -0.60441560;
const CSC_P13: f64 = 1.50880086;
const CSC_P23: f64 = -0.93678576;
const CSC_P33: f64 = 0.08693841;
const CSC_P04: f64 = 0.93412077;
const CSC_P14: f64 = -1.41601920;
const CSC_P24: f64 = 0.33887446;
const CSC_P05: f64 = -0.63915306;
const CSC_P15: f64 = 0.52032238;
const CSC_P06: f64 = 0.14381585;

/// COBE quadrilateralized spherical cube `CSC` (Paper II Sec.5.6.2).
/// The Chan & O'Neill (1975) polynomial is accurate to a few
/// arcseconds. The face layout follows the +-shape of [`Tsc`].
#[derive(Debug, Clone, Copy)]
pub struct Csc;
impl Csc {
    fn face_chi_psi(phi: f64, theta: f64) -> (u32, f64, f64) {
        let p = phi * D2R;
        let t = theta * D2R;
        let ct = t.cos();
        let l = ct * p.cos();
        let m = ct * p.sin();
        let n = t.sin();
        let mut face = 0_u32;
        let mut zeta = n;
        if l > zeta {
            face = 1;
            zeta = l;
        }
        if m > zeta {
            face = 2;
            zeta = m;
        }
        if -l > zeta {
            face = 3;
            zeta = -l;
        }
        if -m > zeta {
            face = 4;
            zeta = -m;
        }
        if -n > zeta {
            face = 5;
            zeta = -n;
        }
        let (xi, eta) = match face {
            0 => (m, -l),
            1 => (m, n),
            2 => (-l, n),
            3 => (-m, n),
            4 => (l, n),
            5 => (m, l),
            _ => unreachable!(),
        };
        (face, xi / zeta, eta / zeta)
    }
    fn forward_poly(chi: f64, psi: f64) -> f64 {
        let chi2 = chi * chi;
        let psi2 = psi * psi;
        let chi2co = 1.0 - chi2;
        let chi4 = chi2 * chi2;
        let psi4 = psi2 * psi2;
        chi * (chi2
            + chi2co
                * (CSC_GSTAR
                    + psi2
                        * (CSC_GAMMA * chi2co
                            + CSC_M * chi2
                            + (1.0 - psi2)
                                * (CSC_C00
                                    + CSC_C10 * chi2
                                    + CSC_C01 * psi2
                                    + CSC_C11 * chi2 * psi2
                                    + CSC_C20 * chi4
                                    + CSC_C02 * psi4))
                    + chi2 * (CSC_OMEGA1 - chi2co * (CSC_D0 + CSC_D1 * chi2))))
    }
    fn inverse_poly(xf: f64, yf: f64) -> f64 {
        let xx = xf * xf;
        let yy = yf * yf;
        let z0 = CSC_P00
            + xx * (CSC_P10
                + xx * (CSC_P20 + xx * (CSC_P30 + xx * (CSC_P40 + xx * (CSC_P50 + xx * CSC_P60)))));
        let z1 = CSC_P01
            + xx * (CSC_P11 + xx * (CSC_P21 + xx * (CSC_P31 + xx * (CSC_P41 + xx * CSC_P51))));
        let z2 = CSC_P02 + xx * (CSC_P12 + xx * (CSC_P22 + xx * (CSC_P32 + xx * CSC_P42)));
        let z3 = CSC_P03 + xx * (CSC_P13 + xx * (CSC_P23 + xx * CSC_P33));
        let z4 = CSC_P04 + xx * (CSC_P14 + xx * CSC_P24);
        let z5 = CSC_P05 + xx * CSC_P15;
        let chi = z0 + yy * (z1 + yy * (z2 + yy * (z3 + yy * (z4 + yy * (z5 + yy * CSC_P06)))));
        xf + xf * (1.0 - xx) * chi
    }
    /// Face-relative `(xf, yf)` -> projection-plane `(x, y)` in degrees
    /// (same +-layout as [`Tsc`]).
    pub(super) fn face_to_xy(face: u32, xf: f64, yf: f64) -> (f64, f64) {
        let (cx, cy) = match face {
            0 => (0.0, 90.0),
            1 => (0.0, 0.0),
            2 => (-90.0, 0.0),
            3 => (-180.0, 0.0),
            4 => (-270.0, 0.0),
            5 => (0.0, -90.0),
            _ => unreachable!(),
        };
        (cx + 45.0 * xf, cy + 45.0 * yf)
    }
    pub(super) fn xy_to_face(x: f64, y: f64) -> Option<(u32, f64, f64)> {
        let in_unit = |a: f64| a.abs() <= 1.0 + 1e-9;
        for (face, cx, cy) in [
            (0_u32, 0.0_f64, 90.0_f64),
            (1, 0.0, 0.0),
            (2, -90.0, 0.0),
            (3, -180.0, 0.0),
            (4, -270.0, 0.0),
            (5, 0.0, -90.0),
        ] {
            let xf = (x - cx) / 45.0;
            let yf = (y - cy) / 45.0;
            if in_unit(xf) && in_unit(yf) {
                return Some((face, xf.clamp(-1.0, 1.0), yf.clamp(-1.0, 1.0)));
            }
        }
        None
    }
}
impl Projection for Csc {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        let (face, chi, psi) = Self::face_chi_psi(phi, theta);
        let xf = Self::forward_poly(chi, psi).clamp(-1.0, 1.0);
        let yf = Self::forward_poly(psi, chi).clamp(-1.0, 1.0);
        Ok(Self::face_to_xy(face, xf, yf))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (face, xf, yf) = Self::xy_to_face(x, y)
            .ok_or_else(|| FitsError::Wcs(format!("CSC: ({x}, {y}) is outside the cube layout")))?;
        let chi = Self::inverse_poly(xf, yf);
        let psi = Self::inverse_poly(yf, xf);
        let t = 1.0 / (chi * chi + psi * psi + 1.0).sqrt();
        let (l, m, n) = match face {
            0 => (-psi * t, chi * t, t),
            1 => (t, chi * t, psi * t),
            2 => (-chi * t, t, psi * t),
            3 => (-t, -chi * t, psi * t),
            4 => (chi * t, -t, psi * t),
            5 => (psi * t, chi * t, -t),
            _ => unreachable!(),
        };
        let theta = n.clamp(-1.0, 1.0).asin() * R2D;
        let phi = if l == 0.0 && m == 0.0 {
            0.0
        } else {
            m.atan2(l) * R2D
        };
        Ok((phi, theta))
    }
}

// -- QSC --------------------------------------------------------------

/// Quadrilateralized spherical cube `QSC` (Paper II Sec.5.6.3,
/// eqs. 60-63). Equal-area exact form.
#[derive(Debug, Clone, Copy)]
pub struct Qsc;
impl Qsc {
    fn face_lmn(phi: f64, theta: f64) -> (u32, f64, f64, f64) {
        let p = phi * D2R;
        let t = theta * D2R;
        let ct = t.cos();
        let l = ct * p.cos();
        let m = ct * p.sin();
        let n = t.sin();
        let mut face = 0_u32;
        let mut zeta = n;
        if l > zeta {
            face = 1;
            zeta = l;
        }
        if m > zeta {
            face = 2;
            zeta = m;
        }
        if -l > zeta {
            face = 3;
            zeta = -l;
        }
        if -m > zeta {
            face = 4;
            zeta = -m;
        }
        if -n > zeta {
            let _ = zeta;
            face = 5;
        }
        (face, l, m, n)
    }
}
impl Projection for Qsc {
    fn theta0(&self) -> f64 {
        0.0
    }
    fn s2x(&self, phi: f64, theta: f64) -> Result<(f64, f64)> {
        if (theta - 90.0).abs() < 1e-12 {
            return Ok((0.0, 90.0));
        }
        if (theta + 90.0).abs() < 1e-12 {
            return Ok((0.0, -90.0));
        }
        let (face, l, m, n) = Self::face_lmn(phi, theta);
        let (xi, eta, zeta) = match face {
            1 => (m, n, l),
            2 => (-l, n, m),
            3 => (-m, n, -l),
            4 => (l, n, -m),
            5 => (m, l, -n),
            _ => (m, -l, n),
        };
        let zeco = 1.0 - zeta;
        let (xf, yf) = if xi == 0.0 && eta == 0.0 {
            (0.0, 0.0)
        } else if xi.abs() >= eta.abs() {
            let omega = eta / xi;
            let tau = 1.0 + omega * omega;
            let mag = (zeco / (1.0 - 1.0 / (1.0 + tau).sqrt())).sqrt();
            let xf = xi.signum() * mag;
            let inner = (omega.atan() - (omega / (tau + tau).sqrt()).asin()).to_degrees();
            (xf, (xf / 15.0) * inner)
        } else {
            let omega = xi / eta;
            let tau = 1.0 + omega * omega;
            let mag = (zeco / (1.0 - 1.0 / (1.0 + tau).sqrt())).sqrt();
            let yf = eta.signum() * mag;
            let inner = (omega.atan() - (omega / (tau + tau).sqrt()).asin()).to_degrees();
            ((yf / 15.0) * inner, yf)
        };
        Ok(Csc::face_to_xy(
            face,
            xf.clamp(-1.0, 1.0),
            yf.clamp(-1.0, 1.0),
        ))
    }
    fn x2s(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let (face, xf, yf) = Csc::xy_to_face(x, y)
            .ok_or_else(|| FitsError::Wcs(format!("QSC: ({x}, {y}) is outside the cube layout")))?;
        let direct = xf.abs() > yf.abs();
        let (omega, tau, zeta) = if direct && xf != 0.0 {
            let w = (15.0 * yf / xf).to_radians();
            let (sw, cw) = w.sin_cos();
            let o = sw / (cw - std::f64::consts::FRAC_1_SQRT_2);
            let t = 1.0 + o * o;
            (o, t, 1.0 - xf * xf * (1.0 - 1.0 / (1.0 + t).sqrt()))
        } else if !direct && yf != 0.0 {
            let w = (15.0 * xf / yf).to_radians();
            let (sw, cw) = w.sin_cos();
            let o = sw / (cw - std::f64::consts::FRAC_1_SQRT_2);
            let t = 1.0 + o * o;
            (o, t, 1.0 - yf * yf * (1.0 - 1.0 / (1.0 + t).sqrt()))
        } else {
            (0.0, 1.0, 1.0)
        };
        let zeta = zeta.clamp(-1.0, 1.0);
        let zeco = 1.0 - zeta;
        let w = (zeco * (2.0 - zeco) / tau).max(0.0).sqrt();
        let (l, m, n) = match face {
            1 => {
                let l = zeta;
                if direct {
                    let m = if xf < 0.0 { -w } else { w };
                    (l, m, m * omega)
                } else {
                    let n = if yf < 0.0 { -w } else { w };
                    (l, n * omega, n)
                }
            }
            2 => {
                let m = zeta;
                if direct {
                    let l = if xf > 0.0 { -w } else { w };
                    (l, m, -l * omega)
                } else {
                    let n = if yf < 0.0 { -w } else { w };
                    (-n * omega, m, n)
                }
            }
            3 => {
                let l = -zeta;
                if direct {
                    let m = if xf > 0.0 { -w } else { w };
                    (l, m, -m * omega)
                } else {
                    let n = if yf < 0.0 { -w } else { w };
                    (l, -n * omega, n)
                }
            }
            4 => {
                let m = -zeta;
                if direct {
                    let l = if xf < 0.0 { -w } else { w };
                    (l, m, l * omega)
                } else {
                    let n = if yf < 0.0 { -w } else { w };
                    (n * omega, m, n)
                }
            }
            5 => {
                let n = -zeta;
                if direct {
                    let m = if xf < 0.0 { -w } else { w };
                    (m * omega, m, n)
                } else {
                    let l = if yf < 0.0 { -w } else { w };
                    (l, l * omega, n)
                }
            }
            _ => {
                let n = zeta;
                if direct {
                    let m = if xf < 0.0 { -w } else { w };
                    (-m * omega, m, n)
                } else {
                    let l = if yf > 0.0 { -w } else { w };
                    (l, -l * omega, n)
                }
            }
        };
        let theta = n.clamp(-1.0, 1.0).asin() * R2D;
        let phi = if l == 0.0 && m == 0.0 {
            0.0
        } else {
            m.atan2(l) * R2D
        };
        Ok((phi, theta))
    }
}
