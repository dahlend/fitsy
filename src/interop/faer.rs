//! `faer` interop. Enabled with `--features faer`.
//!
//! Mirrors the [`nalgebra`](super::nalgebra) integration: WCS matrix
//! accessors, an [`ImageData`] -> [`faer::Mat`] reshape, and batched
//! pixel<->world transforms with one point per column.

use ::faer::Mat;

use crate::data::ImageData;
use crate::data::encoding::Pixel;
use crate::error::{FitsError, Result};
use crate::hdu::ImageBuilder;
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
    /// Re-shape a two-dimensional image into a [`Mat`] of size
    /// `NAXIS2` by `NAXIS1`. Rows follow the slow axis, and columns
    /// follow the fast axis.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when the image is not two-dimensional.
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

    /// Build a two-dimensional image from a [`Mat`].
    ///
    /// The matrix rows are `NAXIS2`, the slow axis, and its columns
    /// are `NAXIS1`, the fast axis. This inverts
    /// [`to_faer`](Self::to_faer), so the round trip is the identity.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when the element count overflows `u64`.
    pub fn from_faer(mat: &Mat<T>) -> Result<Self> {
        let ny = mat.nrows();
        let nx = mat.ncols();
        let mut data: Vec<T> = Vec::with_capacity(ny * nx);
        for r in 0..ny {
            for c in 0..nx {
                data.push(mat[(r, c)].clone());
            }
        }
        Self::new(data, vec![nx as u64, ny as u64])
    }
}

impl<T: Pixel> ImageBuilder<T> {
    /// Build an [`ImageBuilder`] from a two-dimensional [`Mat`].
    ///
    /// This uses the layout convention of [`ImageData::from_faer`]:
    /// matrix rows are `NAXIS2`, and matrix columns are `NAXIS1`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when the element count does not match the
    /// axis product.
    pub fn from_faer(mat: &Mat<T>) -> Result<Self> {
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
    /// Batched form of [`pixel_to_world`](Self::pixel_to_world).
    ///
    /// The `pix` argument has shape `(naxis, n)`, holding one point
    /// per column. The result has the same shape.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `pix` does not have `naxis` rows, and
    /// the conditions of [`pixel_to_world`](Self::pixel_to_world) for
    /// any point.
    pub fn pixel_to_world_faer(&self, pix: &Mat<f64>) -> Result<Mat<f64>> {
        let n = self.naxis();
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

    /// Batched form of [`world_to_pixel`](Self::world_to_pixel).
    ///
    /// This uses the shape convention of
    /// [`pixel_to_world_faer`](Self::pixel_to_world_faer).
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `world` does not have `naxis` rows, and
    /// the conditions of [`world_to_pixel`](Self::world_to_pixel) for
    /// any point.
    pub fn world_to_pixel_faer(&self, world: &Mat<f64>) -> Result<Mat<f64>> {
        let n = self.naxis();
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
