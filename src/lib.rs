//! Pure-Rust FITS file I/O and WCS coordinate transforms.
//!
//! `fitsy` reads and writes images, binary tables, ASCII tables, and
//! random-groups HDUs, parses the full FITS WCS suite (TAN/SIN/...,
//! SIP, TPV, TNX, DSS, `-TAB`), and works without linking against
//! CFITSIO or wcslib.
//!
//! # Quick start
//!
//! Read an image and apply `BZERO`/`BSCALE`:
//!
//! ```no_run
//! use fitsy::{FitsError, FitsFile, Hdu};
//!
//! let file = FitsFile::open("image.fits")?;
//! let Hdu::Image(img) = file.hdu(0)? else {
//!     return Err(FitsError::Header("HDU 0 is not an image".into()));
//! };
//! let pixels = img.read_physical()?; // f64, scaling applied
//! println!("{:?} bitpix={:?}", img.axes(), img.bitpix());
//! # Ok::<(), fitsy::FitsError>(())
//! ```
//!
//! Write a 2D image:
//!
//! ```no_run
//! use fitsy::{FitsWriter, ImageBuilder};
//!
//! let pixels: Vec<f32> = vec![0.0; 512 * 512];
//! let (header, data) = ImageBuilder::new(vec![512u64, 512], pixels)?
//!     .primary(true)
//!     .card("OBJECT", "M42", Some("target"))
//!     .build()?;
//! let mut out = std::fs::File::create("out.fits")?;
//! FitsWriter::new(&mut out).write_hdu(&header, &data)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Pixel &lt;-&gt; sky with WCS:
//!
//! ```no_run
//! use fitsy::FitsFile;
//!
//! let file = FitsFile::open("image.fits")?;
//! let wcs = file.wcs(0, ' ')?.expect("no WCS in HDU 0");
//! let (ra, dec) = wcs.pixel_to_celestial(512.0, 512.0)?;
//! # Ok::<(), fitsy::FitsError>(())
//! ```
//!
//! # Where to look next
//!
//! | If you want to...               | Start here                                          |
//! |---------------------------------|-----------------------------------------------------|
//! | Open a file                     | [`FitsFile`]                                        |
//! | Walk through HDUs               | [`Hdu`], [`FitsFile::iter`]                         |
//! | Read image pixels               | [`ImageHdu`]                                        |
//! | Read a binary table             | [`BinTableHdu`]                                     |
//! | Build a new image to write      | [`ImageBuilder`]                                    |
//! | Build a binary table to write   | [`BinTableHdu`] / [`hdu::BinTableBuilder`]          |
//! | Inspect header cards            | [`Header`]                                          |
//! | Convert pixels &lt;-&gt; sky            | [`wcs::Wcs`]                                        |
//! | Fit a WCS from pixel/sky pairs  | [`wcs::fit_celestial_wcs`]                          |
//!
//! # Runnable examples
//!
//! Each row below is a real file under
//! [`examples/`](https://github.com/dahlend/fitsy/tree/main/examples).
//! Clone the repo and run any of them with `cargo run --example NAME`:
//!
//! | Example                                                                              | Description                                              |
//! |--------------------------------------------------------------------------------------|----------------------------------------------------------|
//! | [`read_image`](https://github.com/dahlend/fitsy/blob/main/examples/read_image.rs)    | Open an image, inspect header, decode pixels             |
//! | [`read_table`](https://github.com/dahlend/fitsy/blob/main/examples/read_table.rs)    | Iterate columns of a binary table and decode cells       |
//! | [`write_image`](https://github.com/dahlend/fitsy/blob/main/examples/write_image.rs)  | Build and write a 2D image FITS file                     |
//! | [`write_table`](https://github.com/dahlend/fitsy/blob/main/examples/write_table.rs)  | Build and write a multi-column binary table              |
//! | [`wcs`](https://github.com/dahlend/fitsy/blob/main/examples/wcs.rs)                  | Pixel &lt;-&gt; sky transforms on the bundled NGC 2403 image     |
//! | [`fit_wcs`](https://github.com/dahlend/fitsy/blob/main/examples/fit_wcs.rs)          | Fit a celestial WCS from pixel/sky correspondences       |
//!
//! Sample data lives in
//! [`examples/data/`](https://github.com/dahlend/fitsy/tree/main/examples/data),
//! and parallel Python scripts (for the `python` feature) are under
//! [`examples/python/`](https://github.com/dahlend/fitsy/tree/main/examples/python).
//!
//! # Cargo features
//!
//! - `compression` *(default)* -- Rice / HCOMPRESS / PLIO / GZIP tile
//!   decompression and quantized-float decoding on read; `GZIP_1`
//!   tile compression on write via
//!   [`compress_image_to_hdu`]
//!   and [`FitsWriter::write_hdu_compressed`](FitsWriter::write_hdu_compressed).
//! - `nalgebra`, `faer` -- zero-copy interop adapters for those linear
//!   algebra crates (see the `interop` module).
//! - `python` -- `PyO3` bindings used to build the `fitsy` Python wheel
//!   via [maturin](https://www.maturin.rs).

// The crate is free of `unsafe` outside of the `python` module
// (where PyO3 macros expand to `unsafe` blocks). `deny` (rather than
// `forbid`) is required so that the inner `#![allow(unsafe_code)]`
// in `python.rs` can take effect.
#![deny(unsafe_code)]

pub mod data;
pub mod error;
pub mod hdu;
pub mod header;
pub mod io;
pub mod wcs;

pub mod checksum;
pub mod diff;

#[cfg(feature = "compression")]
pub mod compression;

#[cfg(any(feature = "nalgebra", feature = "faer"))]
pub mod interop;

#[cfg(feature = "python")]
pub mod python;

pub use data::{Bitpix, ImageData};
pub use error::FitsError;
pub use hdu::{
    AsciiCell, AsciiColumn, AsciiFormat, AsciiTableBuilder, AsciiTableHdu, BinColumn, BinFieldKind,
    BinFormat, BinTableBuilder, BinTableHdu, BinValue, FitsFile, FitsOpenOptions, Hdu,
    ImageBuilder, ImageHdu, ImagePixels,
};
pub use header::{Card, CommentaryKind, Header, IsoDateTime, Value};
#[cfg(not(target_arch = "wasm32"))]
pub use io::FitsAppender;
#[cfg(not(target_arch = "wasm32"))]
pub use io::FitsUpdater;
pub use io::{FitsWriter, write};
pub use wcs::Wcs;

#[cfg(feature = "compression")]
pub use compression::{CompressedImageHdu, OwnedImage, TileOpts, compress_image_to_hdu};
