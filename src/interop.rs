//! Optional conversions between `fitsy` types and external linear-
//! algebra crates. Each integration lives behind its own feature
//! flag so non-users pay nothing.
//!
//! - `nalgebra` (feature `nalgebra`): re-shapes [`ImageData`] into a
//!   `nalgebra::DMatrix`, exposes the WCS pipeline matrices, and
//!   adds a batched [`Wcs::pixel_to_world_na`] / [`Wcs::world_to_pixel_na`].
//! - `faer` (feature `faer`): the same surface, mirrored onto
//!   `faer::Mat`.
//!
//! The convention for batched coordinate transforms is **column-major
//! per point**: a matrix of shape `(naxis, n)` represents `n` points,
//! one per column. This matches the native column-major layout of
//! both nalgebra and faer.
//!
//! [`ImageData`]: crate::data::ImageData
//! [`Wcs::pixel_to_world_na`]: crate::Wcs::pixel_to_world_na
//! [`Wcs::world_to_pixel_na`]: crate::Wcs::world_to_pixel_na

#[cfg(feature = "nalgebra")]
pub mod nalgebra;

#[cfg(feature = "faer")]
pub mod faer;
