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
    /// Re-shape a two-dimensional image into a [`DMatrix`] of size
    /// `NAXIS2` by `NAXIS1`. Rows follow the slow axis, and columns
    /// follow the fast axis.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when the image is not two-dimensional.
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

    /// Build a two-dimensional image from a [`DMatrix`].
    ///
    /// The matrix rows are `NAXIS2`, the slow axis, and its columns
    /// are `NAXIS1`, the fast axis. This inverts
    /// [`to_dmatrix`](Self::to_dmatrix), so the round trip is the
    /// identity.
    ///
    /// `DMatrix` is column-major, while [`ImageData`] is row-major
    /// over `(y, x)`. The conversion therefore copies one element at a
    /// time.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when the element count overflows `u64`.
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
    /// Build an [`ImageBuilder`] from a two-dimensional [`DMatrix`].
    ///
    /// This uses the layout convention of
    /// [`ImageData::from_dmatrix`]: matrix rows are `NAXIS2`, and
    /// matrix columns are `NAXIS1`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when the element count does not match the
    /// axis product.
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
    pub fn pixel_to_world_na(&self, pix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let n = self.naxis();
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

    /// Batched form of [`world_to_pixel`](Self::world_to_pixel).
    ///
    /// This uses the shape convention of
    /// [`pixel_to_world_na`](Self::pixel_to_world_na).
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `world` does not have `naxis` rows, and
    /// the conditions of [`world_to_pixel`](Self::world_to_pixel) for
    /// any point.
    pub fn world_to_pixel_na(&self, world: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let n = self.naxis();
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
