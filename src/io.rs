//! Block-level input and output.
//!
//! # Purpose
//!
//! A FITS file is a sequence of 2880-byte blocks (Standard Sec.3.1).
//! This module reads and writes those blocks.
//!
//! # Layout
//!
//! - [`block`] -- the block size and the padding arithmetic.
//! - [`writer`] -- [`FitsWriter`], the streaming write path, and the
//!   [`write()`] convenience function.
//! - [`source`] -- [`ByteSource`], the owning in-memory read buffer.
//! - [`append`] -- [`FitsAppender`], which adds extension HDUs to an
//!   existing file.
//! - [`update`] -- [`FitsUpdater`], which patches image pixels in
//!   place.
//!
//! # Design constraints
//!
//! [`ByteSource`] serves a read from memory. An on-disk read does not
//! pass through it: [`crate::FitsFile`] loads each data section lazily
//! with a positional read instead.
//!
//! The `append` and `update` submodules need a seekable file, so they
//! are absent on the `wasm32` target.

#[cfg(not(target_arch = "wasm32"))]
pub mod append;
pub mod block;
pub mod source;
#[cfg(not(target_arch = "wasm32"))]
pub mod update;
pub mod writer;

#[cfg(not(target_arch = "wasm32"))]
pub use append::FitsAppender;
pub use block::{BLOCK_SIZE, blocks_for_bytes, pad_to_block};
pub use source::ByteSource;
#[cfg(not(target_arch = "wasm32"))]
pub use update::FitsUpdater;
pub use writer::{FitsWriter, write};
