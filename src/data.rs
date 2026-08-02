//! Raw image data encoding and decoding (Standard Sec.4.4.1.1,
//! Sec.5).
//!
//! # Purpose
//!
//! This module turns the raw bytes of a data section into typed pixel
//! values, and back.
//!
//! # Layout
//!
//! - [`Bitpix`] names the six `BITPIX` values and gives the byte size
//!   of each element.
//! - [`ImageData`] pairs a decoded pixel array with its axis shape.
//! - [`Scaling`] applies `BZERO`, `BSCALE` and `BLANK`.
//! - [`encoding`] holds the `Pixel` trait, which converts one element
//!   between its native type and big-endian bytes.
//! - [`ieee`] holds the IEEE special values that Sec.5 defines.
//! - [`unsigned`] holds the `BZERO` offsets that store unsigned data
//!   in a signed `BITPIX`.

pub mod encoding;
pub mod ieee;
pub mod scaling;
pub mod unsigned;

pub use encoding::{Bitpix, ImageData, Pixel};
pub use scaling::Scaling;
