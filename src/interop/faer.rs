//! `faer` interop. Enabled with `--features faer`.
//!
//! Mirrors the [`nalgebra`](super::nalgebra) integration: WCS matrix
//! accessors, an [`ImageData`] -> [`faer::Mat`] reshape, and batched
//! pixel<->world transforms with one point per column.

use ::faer::Mat;

use crate::data::ImageData;
use crate::error::{FitsError, Result};
use crate::wcs::Wcs;
use crate::wcs::linear::LinearTransform;

impl LinearTransform {
    /// Combined linear matrix `M`, `naxis x naxis`.
    #[must_use]
    pub fn matrix_faer(&self) -> Mat<f64> {
        let n = self.naxis();
        let m = self.matrix_row_major();
        Mat::from_fn(n, n, |i, j| m[i * n + j])
    }

    /// Inverse of [`matrix_faer`](Self::matrix_faer).
    #[must_use]
    pub fn inverse_faer(&self) -> Mat<f64> {
        let n = self.naxis();
        let inv = self.inverse_row_major();
        Mat::from_fn(n, n, |i, j| inv[i * n + j])
    }
}

impl<T: Clone> ImageData<T> {
    /// Re-shape a 2-D image into a [`Mat`] of size `NAXIS2 x NAXIS1`
    /// (rows = slow axis, columns = fast axis). Errors if the image
    /// is not 2-D.
    pub fn to_faer(&self) -> Result<Mat<T>> {
        if self.axes().len() != 2 {
            return Err(FitsError::Data(format!(
                "to_faer: image is {}-D, expected 2-D",
                self.axes().len()
            )));
        }
        // NAXIS1 (fast axis).
        let nx = self.axes()[0] as usize;
        // NAXIS2 (slow axis).
        let ny = self.axes()[1] as usize;
        let s = self.as_slice();
        Ok(Mat::from_fn(ny, nx, |r, c| s[r * nx + c].clone()))
    }
}

impl Wcs {
    /// Batched [`pixel_to_world`](Self::pixel_to_world). `pix` has shape
    /// `(naxis, n)` -- one point per column. Output has the same shape.
    pub fn pixel_to_world_faer(&self, pix: &Mat<f64>) -> Result<Mat<f64>> {
        let n = self.naxis;
        if pix.nrows() != n {
            return Err(FitsError::Wcs(format!(
                "pixel_to_world_faer: matrix has {} rows, expected naxis = {n}",
                pix.nrows()
            )));
        }
        let m = pix.ncols();
        let mut buf = vec![0.0_f64; n];
        let mut flat = vec![0.0_f64; n * m];
        for j in 0..m {
            for i in 0..n {
                buf[i] = pix[(i, j)];
            }
            let world = self.pixel_to_world(&buf)?;
            for (i, w) in world.iter().enumerate() {
                flat[j * n + i] = *w;
            }
        }
        // `flat` is column-major (n rows x m cols); build the Mat.
        Ok(Mat::from_fn(n, m, |i, j| flat[j * n + i]))
    }

    /// Batched [`world_to_pixel`](Self::world_to_pixel). Same shape
    /// convention as [`pixel_to_world_faer`](Self::pixel_to_world_faer).
    pub fn world_to_pixel_faer(&self, world: &Mat<f64>) -> Result<Mat<f64>> {
        let n = self.naxis;
        if world.nrows() != n {
            return Err(FitsError::Wcs(format!(
                "world_to_pixel_faer: matrix has {} rows, expected naxis = {n}",
                world.nrows()
            )));
        }
        let m = world.ncols();
        let mut buf = vec![0.0_f64; n];
        let mut flat = vec![0.0_f64; n * m];
        for j in 0..m {
            for i in 0..n {
                buf[i] = world[(i, j)];
            }
            let pix = self.world_to_pixel(&buf)?;
            for (i, p) in pix.iter().enumerate() {
                flat[j * n + i] = *p;
            }
        }
        Ok(Mat::from_fn(n, m, |i, j| flat[j * n + i]))
    }
}
