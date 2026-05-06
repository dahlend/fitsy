//! Raw image data encoding and decoding (Standard Sec.4.4.1.1, Sec.5).
//!
//! [`Bitpix`] represents the six supported `BITPIX` values and gives
//! the byte size of each element. [`ImageData`] is the decoded pixel
//! array together with its axis shape. [`Scaling`] applies
//! `BZERO`/`BSCALE` and `BLANK` handling.

pub mod encoding;
pub mod ieee;
pub mod scaling;
pub mod unsigned;

pub use encoding::{Bitpix, ImageData, Pixel};
pub use scaling::Scaling;
