// Matrix-vector products read more naturally with explicit (i, j)
// indexing than with .iter().enumerate() chains.
#![allow(
    clippy::needless_range_loop,
    reason = "matrix-vector products are clearer with explicit (i, j) index notation"
)]
#![allow(
    clippy::doc_markdown,
    reason = "math formulae use backtick notation for subscripts within KaTeX blocks"
)]

//! Linear part of the WCS pipeline (Paper I Sec.2, Standard Sec.8.1).
//!
//! For an N-axis WCS with chosen alternate `a`, the pipeline maps a
//! pixel coordinate `p` to an *intermediate world* coordinate `x`:
//!
//! $$ `x_i` \;=\; \`sum_j` m_{ij}\,(`p_j` - \mathrm{CRPIX}_j) $$
//!
//! where `m_{ij}` is either `CDELT_i * PC_{ij}` (PC form) or `CD_{ij}`
//! (CD form). The two forms are mutually exclusive per Sec.8.2.1.
//! `CROTAi` is treated as a legacy way to construct a `PC` matrix
//! (see [`LinearTransform::from_crota`]).
//!
//! Intermediate world coordinates carry the *units* implied by
//! `CUNITi` and (for celestial axes) are degrees on the projection
//! plane; the projection layer maps them onto the celestial sphere.

use crate::error::{FitsError, Result};

/// Linear transform `x = M (p - crpix)` with both forward and inverse.
#[derive(Debug, Clone)]
pub struct LinearTransform {
    naxis: usize,
    crpix: Vec<f64>,
    /// Combined linear matrix in row-major order, `naxis x naxis`.
    /// PC form: `m[i][j] = cdelt[i] * pc[i][j]`. CD form: `m[i][j] = cd[i][j]`.
    matrix: Vec<f64>,
    /// Inverse of `matrix` (Gauss-Jordan, computed at construction).
    inverse: Vec<f64>,
}

impl LinearTransform {
    /// Build from CRPIX, CDELT, and PC matrix (row-major).
    #[allow(
        clippy::needless_pass_by_value,
        reason = "cdelt and pc are part of the public API; changing to &[f64] would be a breaking change"
    )]
    pub fn from_pc(crpix: Vec<f64>, cdelt: Vec<f64>, pc: Vec<f64>) -> Result<Self> {
        let n = crpix.len();
        if cdelt.len() != n || pc.len() != n * n {
            return Err(FitsError::Wcs(format!(
                "PC dimensions inconsistent: crpix={n}, cdelt={}, pc={}",
                cdelt.len(),
                pc.len()
            )));
        }
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] = cdelt[i] * pc[i * n + j];
            }
        }
        Self::from_matrix(crpix, m)
    }

    /// Build from CRPIX and CD matrix (row-major).
    pub fn from_cd(crpix: Vec<f64>, cd: Vec<f64>) -> Result<Self> {
        let n = crpix.len();
        if cd.len() != n * n {
            return Err(FitsError::Wcs(format!(
                "CD dimensions inconsistent: crpix={n}, cd={}",
                cd.len()
            )));
        }
        Self::from_matrix(crpix, cd)
    }

    /// 2-axis legacy `CROTA2` (Sec.8.2.1.4, deprecated). Builds the
    /// equivalent PC matrix
    /// $$ \mathrm{PC} \;=\; \begin{pmatrix}
    ///   \cos\rho & -\lambda\sin\rho \\
    ///   \sin\rho/\lambda &  \cos\rho
    /// \end{pmatrix}$$
    /// with `lambda = CDELT2/CDELT1`.
    pub fn from_crota(crpix: [f64; 2], cdelt: [f64; 2], crota_deg: f64) -> Result<Self> {
        if cdelt[0] == 0.0 {
            return Err(FitsError::Wcs("CDELT1 = 0 with CROTA2".into()));
        }
        let rho = crota_deg * super::D2R;
        let lam = cdelt[1] / cdelt[0];
        let pc = vec![rho.cos(), -lam * rho.sin(), rho.sin() / lam, rho.cos()];
        Self::from_pc(crpix.to_vec(), cdelt.to_vec(), pc)
    }

    fn from_matrix(crpix: Vec<f64>, matrix: Vec<f64>) -> Result<Self> {
        let n = crpix.len();
        let inverse = invert_matrix(&matrix, n)?;
        Ok(Self {
            naxis: n,
            crpix,
            matrix,
            inverse,
        })
    }

    #[must_use]
    pub fn naxis(&self) -> usize {
        self.naxis
    }

    /// Forward: `x = M (p - crpix)`. `pix` is **1-based** (FITS
    /// convention, Sec.3.3.4: pixel centers are at integer values
    /// starting from 1).
    pub fn pix_to_intermediate(&self, pix: &[f64]) -> Result<Vec<f64>> {
        if pix.len() != self.naxis {
            return Err(FitsError::Wcs(format!(
                "expected {} pixel coordinates, got {}",
                self.naxis,
                pix.len()
            )));
        }
        let n = self.naxis;
        let dp: Vec<f64> = (0..n).map(|j| pix[j] - self.crpix[j]).collect();
        let mut out = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                out[i] += self.matrix[i * n + j] * dp[j];
            }
        }
        Ok(out)
    }

    /// Inverse: `p = crpix + M^{-1} x`.
    pub fn intermediate_to_pix(&self, intermediate: &[f64]) -> Result<Vec<f64>> {
        if intermediate.len() != self.naxis {
            return Err(FitsError::Wcs(format!(
                "expected {} intermediate coords, got {}",
                self.naxis,
                intermediate.len()
            )));
        }
        let n = self.naxis;
        let mut dp = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                dp[i] += self.inverse[i * n + j] * intermediate[j];
            }
        }
        Ok((0..n).map(|i| self.crpix[i] + dp[i]).collect())
    }

    /// Read-only access to CRPIX (1-based reference pixel).
    #[must_use]
    pub fn crpix(&self) -> &[f64] {
        &self.crpix
    }

    /// Combined linear matrix in row-major order, length `naxis^2`.
    /// Row `i` holds the coefficients of intermediate axis `i`.
    #[must_use]
    pub fn matrix_row_major(&self) -> &[f64] {
        &self.matrix
    }

    /// Inverse of [`matrix_row_major`](Self::matrix_row_major), same
    /// row-major layout.
    #[must_use]
    pub fn inverse_row_major(&self) -> &[f64] {
        &self.inverse
    }

    /// Apply only the linear matrix `M * dp` (no CRPIX shift).
    /// Used by distortion conventions (SIP) that need to inject a
    /// polynomial between the CRPIX subtraction and the matrix.
    pub fn apply_matrix(&self, dp: &[f64]) -> Result<Vec<f64>> {
        if dp.len() != self.naxis {
            return Err(FitsError::Wcs(format!(
                "expected {} pixel offsets, got {}",
                self.naxis,
                dp.len()
            )));
        }
        let n = self.naxis;
        let mut out = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                out[i] += self.matrix[i * n + j] * dp[j];
            }
        }
        Ok(out)
    }

    /// Apply only `M^-1 * x` (no CRPIX add). Inverse counterpart of
    /// [`Self::apply_matrix`].
    pub fn apply_inverse_matrix(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.naxis {
            return Err(FitsError::Wcs(format!(
                "expected {} intermediate coords, got {}",
                self.naxis,
                x.len()
            )));
        }
        let n = self.naxis;
        let mut out = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                out[i] += self.inverse[i * n + j] * x[j];
            }
        }
        Ok(out)
    }

    /// Compose with a pre-pixel affine remap `p_phys = A * p_log + b`.
    ///
    /// Used to absorb the IRAF `LTV`/`LTM` subimage convention into
    /// the linear pipeline: the WCS-as-written refers to original
    /// (physical) detector pixels, but the array we are reading is a
    /// subimage in logical coordinates. Substituting `p_phys` into
    /// `x = M (p_phys - CRPIX_phys)` yields a new equivalent linear
    /// transform `x = M*A * (p_log - CRPIX_log)` with
    /// `CRPIX_log = A^-1 (CRPIX_phys - b)`.
    ///
    /// `a` is row-major `naxis x naxis`; `b` is length `naxis`.
    pub fn compose_with_input_affine(&self, a: &[f64], b: &[f64]) -> Result<Self> {
        let n = self.naxis;
        if a.len() != n * n || b.len() != n {
            return Err(FitsError::Wcs(format!(
                "compose_with_input_affine: expected {n}x{n} matrix and length-{n} vector"
            )));
        }
        // new_matrix[i][k] = Sigma_j matrix[i][j] * a[j][k]
        let mut new_m = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += self.matrix[i * n + j] * a[j * n + k];
                }
                new_m[i * n + k] = s;
            }
        }
        // new_crpix = A^-1 * (crpix - b)
        let a_inv = invert_matrix(a, n)?;
        let mut diff = vec![0.0; n];
        for i in 0..n {
            diff[i] = self.crpix[i] - b[i];
        }
        let mut new_crpix = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_crpix[i] += a_inv[i * n + j] * diff[j];
            }
        }
        Self::from_matrix(new_crpix, new_m)
    }
}

/// Gauss-Jordan inversion. Returns `Wcs` error if singular.
fn invert_matrix(m: &[f64], n: usize) -> Result<Vec<f64>> {
    debug_assert_eq!(
        m.len(),
        n * n,
        "matrix must be nxn; got {} elements for n={n}",
        m.len()
    );
    // Augmented [M | I].
    let mut a = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            a[i * 2 * n + j] = m[i * n + j];
        }
        a[i * 2 * n + n + i] = 1.0;
    }
    for i in 0..n {
        // Partial pivot.
        let mut pivot = i;
        let mut best = a[i * 2 * n + i].abs();
        for k in (i + 1)..n {
            let v = a[k * 2 * n + i].abs();
            if v > best {
                best = v;
                pivot = k;
            }
        }
        if best < 1e-300 {
            return Err(FitsError::Wcs(
                "linear matrix is singular (cannot invert)".into(),
            ));
        }
        if pivot != i {
            for j in 0..(2 * n) {
                a.swap(i * 2 * n + j, pivot * 2 * n + j);
            }
        }
        let inv_diag = 1.0 / a[i * 2 * n + i];
        for j in 0..(2 * n) {
            a[i * 2 * n + j] *= inv_diag;
        }
        for k in 0..n {
            if k == i {
                continue;
            }
            let factor = a[k * 2 * n + i];
            if factor == 0.0 {
                continue;
            }
            for j in 0..(2 * n) {
                a[k * 2 * n + j] -= factor * a[i * 2 * n + j];
            }
        }
    }
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = a[i * 2 * n + n + j];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_pc_round_trip() {
        let lt = LinearTransform::from_pc(vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 0.0, 0.0, 1.0])
            .unwrap();
        let p = vec![3.5, 7.25];
        let x = lt.pix_to_intermediate(&p).unwrap();
        let q = lt.intermediate_to_pix(&x).unwrap();
        for (a, b) in p.iter().zip(q.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn cd_matrix_round_trip() {
        // CD = [[0.001, 0], [0, 0.001]]
        let lt =
            LinearTransform::from_cd(vec![100.0, 200.0], vec![0.001, 0.0, 0.0, 0.001]).unwrap();
        let x = lt.pix_to_intermediate(&[150.0, 250.0]).unwrap();
        assert!((x[0] - 0.05).abs() < 1e-12);
        assert!((x[1] - 0.05).abs() < 1e-12);
        let p = lt.intermediate_to_pix(&x).unwrap();
        assert!((p[0] - 150.0).abs() < 1e-12);
        assert!((p[1] - 250.0).abs() < 1e-12);
    }

    #[test]
    fn crota_equivalent_to_pc() {
        // CROTA2 = 30deg; check (1,0) pixel offset rotates correctly.
        let lt = LinearTransform::from_crota([1.0, 1.0], [1.0, 1.0], 30.0).unwrap();
        let x = lt.pix_to_intermediate(&[2.0, 1.0]).unwrap();
        let c = (30_f64).to_radians().cos();
        let s = (30_f64).to_radians().sin();
        assert!((x[0] - c).abs() < 1e-12);
        assert!((x[1] - s).abs() < 1e-12);
    }

    #[test]
    fn singular_matrix_rejected() {
        let r = LinearTransform::from_cd(vec![0.0, 0.0], vec![1.0, 2.0, 2.0, 4.0]);
        assert!(r.is_err());
    }
}
