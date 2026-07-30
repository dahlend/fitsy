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
///
/// Owns the linear stage's per-axis data -- `CRPIX`, `CRVAL`, and the
/// combined matrix -- so they cannot fall out of step with `naxis`.
// `CRVAL` is held as data, not folded into the transform: the spectral
// and celestial stages need the CRVAL-free intermediate coordinate, and
// only plain linear axes add it back.
#[derive(Debug, Clone)]
pub struct LinearTransform {
    naxis: usize,
    crpix: Vec<f64>,
    /// `CRVALi`, in `CUNITi`. One per axis.
    crval: Vec<f64>,
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
    pub fn from_pc(
        crpix: Vec<f64>,
        crval: Vec<f64>,
        cdelt: Vec<f64>,
        pc: Vec<f64>,
    ) -> Result<Self> {
        let n = crpix.len();
        if cdelt.len() != n || pc.len() != n * n || crval.len() != n {
            return Err(FitsError::Wcs(format!(
                "PC dimensions inconsistent: crpix={n}, crval={}, cdelt={}, pc={}",
                crval.len(),
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
        Self::from_matrix(crpix, crval, m)
    }

    /// Build from CRPIX, CRVAL and CD matrix (row-major).
    pub fn from_cd(crpix: Vec<f64>, crval: Vec<f64>, cd: Vec<f64>) -> Result<Self> {
        let n = crpix.len();
        if cd.len() != n * n || crval.len() != n {
            return Err(FitsError::Wcs(format!(
                "CD dimensions inconsistent: crpix={n}, crval={}, cd={}",
                crval.len(),
                cd.len()
            )));
        }
        Self::from_matrix(crpix, crval, cd)
    }

    /// Legacy `CROTAi` (Sec.8.2, deprecated). Builds the equivalent
    /// `PC` matrix: the identity everywhere except the 2x2 block on
    /// the rotated axis pair `(lon, lat)`, which holds
    /// $$ \begin{pmatrix}
    ///   \cos\rho & -\lambda\sin\rho \\
    ///   \sin\rho/\lambda &  \cos\rho
    /// \end{pmatrix}$$
    /// with `lambda = CDELT_lat/CDELT_lon`.
    ///
    /// `CROTAi` is indexed and attaches to the latitude axis, so the
    /// rotated pair need not be axes 1 and 2, nor the image 2-D.
    pub fn from_crota(
        crpix: Vec<f64>,
        crval: Vec<f64>,
        cdelt: Vec<f64>,
        crota_deg: f64,
        lon: usize,
        lat: usize,
    ) -> Result<Self> {
        let n = crpix.len();
        if lon >= n || lat >= n || lon == lat {
            return Err(FitsError::Wcs(format!(
                "CROTA: rotated axis pair ({lon}, {lat}) is not valid for {n} axes"
            )));
        }
        if cdelt.len() != n {
            return Err(FitsError::Wcs(format!(
                "CROTA dimensions inconsistent: crpix={n}, cdelt={}",
                cdelt.len()
            )));
        }
        // Sec.8.2: "CDELTi ... The value must not be zero." A zero
        // here makes the ratio below infinite or NaN.
        for axis in [lon, lat] {
            if cdelt[axis] == 0.0 {
                return Err(FitsError::Wcs(format!(
                    "CDELT{} = 0 with CROTA (Sec.8.2: the value must not be zero)",
                    axis + 1
                )));
            }
        }
        let rho = crota_deg * super::D2R;
        let lam = cdelt[lat] / cdelt[lon];
        let mut pc = vec![0.0; n * n];
        for i in 0..n {
            pc[i * n + i] = 1.0;
        }
        pc[lon * n + lon] = rho.cos();
        pc[lon * n + lat] = -lam * rho.sin();
        pc[lat * n + lon] = rho.sin() / lam;
        pc[lat * n + lat] = rho.cos();
        Self::from_pc(crpix, crval, cdelt, pc)
    }

    fn from_matrix(crpix: Vec<f64>, crval: Vec<f64>, matrix: Vec<f64>) -> Result<Self> {
        let n = crpix.len();
        if crval.len() != n {
            return Err(FitsError::Wcs(format!(
                "linear transform: crpix has {n} entries but crval has {}",
                crval.len()
            )));
        }
        let inverse = invert_matrix(&matrix, n)?;
        Ok(Self {
            naxis: n,
            crpix,
            crval,
            matrix,
            inverse,
        })
    }

    /// Number of axes this transform covers.
    #[must_use]
    pub fn naxis(&self) -> usize {
        self.naxis
    }

    /// `CRVALi` for every axis, in `CUNITi`.
    #[must_use]
    pub fn crval(&self) -> &[f64] {
        &self.crval
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
        // `CRVAL` is a world value: the reference pixel moves, the
        // coordinate it names does not.
        Self::from_matrix(new_crpix, self.crval.clone(), new_m)
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
    // Every comparison against NaN is false, so the pivot test below
    // would pass a NaN matrix straight through and yield an all-NaN
    // inverse instead of the error Sec.8.2 calls for.
    if let Some(pos) = m.iter().position(|v| !v.is_finite()) {
        return Err(FitsError::Wcs(format!(
            "linear matrix element [{}][{}] is not finite ({})",
            pos / n,
            pos % n,
            m[pos]
        )));
    }
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
        let lt = LinearTransform::from_pc(
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
        )
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
        let lt = LinearTransform::from_cd(
            vec![100.0, 200.0],
            vec![0.0, 0.0],
            vec![0.001, 0.0, 0.0, 0.001],
        )
        .unwrap();
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
        let lt =
            LinearTransform::from_crota(vec![1.0, 1.0], vec![0.0, 0.0], vec![1.0, 1.0], 30.0, 0, 1)
                .unwrap();
        let x = lt.pix_to_intermediate(&[2.0, 1.0]).unwrap();
        let c = (30_f64).to_radians().cos();
        let s = (30_f64).to_radians().sin();
        assert!((x[0] - c).abs() < 1e-12);
        assert!((x[1] - s).abs() < 1e-12);
    }

    /// A cube's extra axes must pass through untouched while the sky
    /// plane still rotates.
    #[test]
    fn crota_rotates_only_the_celestial_pair_in_a_cube() {
        let lt = LinearTransform::from_crota(
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 7.0],
            30.0,
            0,
            1,
        )
        .unwrap();
        let x = lt.pix_to_intermediate(&[2.0, 1.0, 2.0]).unwrap();
        let c = (30_f64).to_radians().cos();
        let s = (30_f64).to_radians().sin();
        assert!((x[0] - c).abs() < 1e-12);
        assert!((x[1] - s).abs() < 1e-12);
        // Third axis keeps its own CDELT and picks up no rotation.
        assert!((x[2] - 7.0).abs() < 1e-12);
    }

    /// Sec.8.2: "CDELTi ... The value must not be zero." With CROTA a
    /// zero yields a NaN matrix that used to invert without complaint.
    #[test]
    fn crota_with_zero_cdelt_rejected() {
        for cdelt in [vec![0.0, 1.0], vec![1.0, 0.0]] {
            let r = LinearTransform::from_crota(vec![1.0, 1.0], vec![0.0, 0.0], cdelt, 30.0, 0, 1);
            assert!(r.is_err(), "zero CDELT accepted");
        }
    }

    #[test]
    fn singular_matrix_rejected() {
        let r = LinearTransform::from_cd(vec![0.0, 0.0], vec![0.0, 0.0], vec![1.0, 2.0, 2.0, 4.0]);
        assert!(r.is_err());
    }

    #[test]
    fn non_finite_matrix_rejected() {
        for bad in [f64::NAN, f64::INFINITY] {
            let r =
                LinearTransform::from_cd(vec![0.0, 0.0], vec![0.0, 0.0], vec![1.0, 0.0, bad, 1.0]);
            assert!(r.is_err(), "non-finite element {bad} accepted");
        }
    }
}
