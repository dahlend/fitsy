//! `nalgebra` interop. Enabled with `--features nalgebra`.

use ::nalgebra::{DMatrix, Scalar};

use crate::data::ImageData;
use crate::data::encoding::Pixel;
use crate::error::{FitsError, Result};
use crate::hdu::ImageBuilder;
use crate::wcs::Wcs;
use crate::wcs::linear::LinearTransform;

impl LinearTransform {
    /// Combined linear matrix `M` (PC*CDELT or CD), shaped
    /// `naxis x naxis`. Row `i` holds the coefficients that produce
    /// intermediate world axis `i` from `(p - CRPIX)`.
    #[must_use]
    pub fn matrix_na(&self) -> DMatrix<f64> {
        let n = self.naxis();
        // Internal storage is row-major; nalgebra is column-major.
        DMatrix::from_fn(n, n, |i, j| self.matrix_row_major()[i * n + j])
    }

    /// Inverse of [`matrix_na`](Self::matrix_na).
    #[must_use]
    pub fn inverse_na(&self) -> DMatrix<f64> {
        let n = self.naxis();
        DMatrix::from_fn(n, n, |i, j| self.inverse_row_major()[i * n + j])
    }
}

impl<T: Scalar + Copy> ImageData<T> {
    /// Re-shape a 2-D image into a [`DMatrix`] of size `NAXIS2 x NAXIS1`
    /// (rows = slow axis, columns = fast axis). Errors if the image
    /// is not 2-D.
    pub fn to_dmatrix(&self) -> Result<DMatrix<T>> {
        if self.axes().len() != 2 {
            return Err(FitsError::Data(format!(
                "to_dmatrix: image is {}-D, expected 2-D",
                self.axes().len()
            )));
        }
        // NAXIS1 (fast axis).
        let nx = self.axes()[0] as usize;
        // NAXIS2 (slow axis).
        let ny = self.axes()[1] as usize;
        // Memory is row-major over (y, x): data[y * nx + x].
        Ok(DMatrix::from_row_slice(ny, nx, self.as_slice()))
    }

    /// Build a 2-D image from a [`DMatrix`]. The matrix is interpreted
    /// as `nrows = NAXIS2` (slow axis), `ncols = NAXIS1` (fast axis) --
    /// the inverse of [`to_dmatrix`](Self::to_dmatrix), so the round-
    /// trip is the identity.
    ///
    /// nalgebra is column-major; FITS / `ImageData` is row-major over
    /// `(y, x)`. The conversion copies element-by-element.
    pub fn from_dmatrix(mat: &DMatrix<T>) -> Result<Self> {
        let ny = mat.nrows();
        let nx = mat.ncols();
        let mut data: Vec<T> = Vec::with_capacity(ny * nx);
        for r in 0..ny {
            for c in 0..nx {
                data.push(mat[(r, c)]);
            }
        }
        Self::new(data, vec![nx as u64, ny as u64])
    }
}

impl<T: Pixel + Scalar> ImageBuilder<T> {
    /// Build an [`ImageBuilder`] from a 2-D [`DMatrix`]. Same layout
    /// convention as [`ImageData::from_dmatrix`]: `nrows = NAXIS2`,
    /// `ncols = NAXIS1`.
    pub fn from_dmatrix(mat: &DMatrix<T>) -> Result<Self> {
        let ny = mat.nrows();
        let nx = mat.ncols();
        let mut data: Vec<T> = Vec::with_capacity(ny * nx);
        for r in 0..ny {
            for c in 0..nx {
                data.push(mat[(r, c)]);
            }
        }
        Self::new(vec![nx as u64, ny as u64], data)
    }
}

impl Wcs {
    /// Batched [`pixel_to_world`](Self::pixel_to_world). `pix` has shape
    /// `(naxis, n)` -- one point per column. Returns the same shape.
    pub fn pixel_to_world_na(&self, pix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let n = self.naxis;
        if pix.nrows() != n {
            return Err(FitsError::Wcs(format!(
                "pixel_to_world_na: matrix has {} rows, expected naxis = {n}",
                pix.nrows()
            )));
        }
        let m = pix.ncols();
        let mut out = DMatrix::<f64>::zeros(n, m);
        let mut buf = vec![0.0_f64; n];
        for j in 0..m {
            for i in 0..n {
                buf[i] = pix[(i, j)];
            }
            let world = self.pixel_to_world(&buf)?;
            for i in 0..n {
                out[(i, j)] = world[i];
            }
        }
        Ok(out)
    }

    /// Batched [`world_to_pixel`](Self::world_to_pixel). Same shape
    /// convention as [`pixel_to_world_na`](Self::pixel_to_world_na).
    pub fn world_to_pixel_na(&self, world: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let n = self.naxis;
        if world.nrows() != n {
            return Err(FitsError::Wcs(format!(
                "world_to_pixel_na: matrix has {} rows, expected naxis = {n}",
                world.nrows()
            )));
        }
        let m = world.ncols();
        let mut out = DMatrix::<f64>::zeros(n, m);
        let mut buf = vec![0.0_f64; n];
        for j in 0..m {
            for i in 0..n {
                buf[i] = world[(i, j)];
            }
            let pix = self.world_to_pixel(&buf)?;
            for i in 0..n {
                out[(i, j)] = pix[i];
            }
        }
        Ok(out)
    }
}
