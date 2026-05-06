//! Block-level I/O and byte-source abstraction.
//!
//! FITS files are a sequence of 2880-byte blocks (Standard Sec.3.1).
//! [`FitsWriter`] is the main write-side entry point.
//! [`ByteSource`] is the in-memory read-side abstraction; on-disk
//! reads go through per-HDU lazy loads inside [`crate::FitsFile`].

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
