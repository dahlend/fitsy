//! Conversions between `fitsy` types and external linear-algebra
//! crates.
//!
//! # Purpose
//!
//! Each integration sits behind its own cargo feature, so a build that
//! does not enable one compiles none of its code.
//!
//! - `nalgebra` -- re-shapes [`ImageData`] into a `nalgebra::DMatrix`,
//!   exposes the WCS pipeline matrices, and adds the batched
//!   [`Wcs::pixel_to_world_na`] and [`Wcs::world_to_pixel_na`].
//! - `faer` -- the same surface over `faer::Mat`.
//!
//! # Design constraints
//!
//! A batched coordinate transform is column-major per point. A matrix
//! of shape `(naxis, n)` holds `n` points, one per column. This
//! matches the native column-major layout of both matrix types.
//!
//! [`ImageData`]: crate::data::ImageData
//! [`Wcs::pixel_to_world_na`]: crate::Wcs::pixel_to_world_na
//! [`Wcs::world_to_pixel_na`]: crate::Wcs::world_to_pixel_na

#[cfg(feature = "nalgebra")]
pub mod nalgebra;

#[cfg(feature = "faer")]
pub mod faer;
