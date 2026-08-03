//! Pure-Rust FITS input, output, and WCS coordinate transforms.
//!
//! # Purpose
//!
//! `fitsy` reads and writes FITS files. It also converts between pixel
//! coordinates and world coordinates. The crate is pure Rust and links
//! no C library.
//!
//! The read path covers images, binary tables, ASCII tables, and
//! random-groups HDUs. The write path covers images, binary tables,
//! and ASCII tables. An `XTENSION` that this crate does not recognize
//! arrives as [`hdu::ConformingHdu`], so a caller still reads its
//! header and its raw data bytes.
//!
//! # Layout
//!
//! Nine modules are always present. Three of them hold the usual entry
//! points:
//!
//! - [`hdu`] -- opens a file and reaches each HDU. Start at
//!   [`FitsFile`].
//! - [`header`] -- parses and builds the 80-byte cards of a header.
//!   Start at [`Header`].
//! - [`wcs`] -- parses a coordinate description and transforms
//!   coordinates. Start at [`Wcs`].
//!
//! The other six support them:
//!
//! - [`data`] -- decodes and encodes raw pixel bytes. Holds
//!   [`Bitpix`] and the `BZERO`/`BSCALE` scaling.
//! - [`io`] -- reads and writes the 2880-byte blocks. Holds
//!   [`FitsWriter`].
//! - [`error`] -- holds [`FitsError`], the single error type of the
//!   crate.
//! - [`units`] -- parses a unit string such as `BUNIT` or `CUNIT`
//!   into a scale factor and a set of dimensional exponents.
//! - [`checksum`] -- computes the `CHECKSUM` and `DATASUM` values.
//! - [`diff`] -- compares two files and reports the differences.
//!
//! Two more modules appear under a cargo feature. [`compression`]
//! reads and writes tile-compressed images, and inflates a gzipped
//! file. [`interop`] converts image data to the matrix type of
//! another crate, under the `nalgebra` or the `faer` feature.
//!
//! # Design constraints
//!
//! Three facts explain the shape of the API.
//!
//! First, an HDU borrows its file. [`FitsFile`] owns the bytes, and
//! [`Hdu`] holds a view into them. An HDU therefore cannot outlive the
//! [`FitsFile`] that produced it.
//!
//! Second, a read is lazy. [`FitsFile::open`] parses each header and
//! records the byte span of each data unit. It decodes no pixel data.
//! A call such as [`ImageHdu::read_physical`] decodes on demand.
//!
//! Third, [`Header`] and [`Wcs`] are separate layers. [`Header`]
//! preserves the file. It holds every card as written, including a
//! card that contradicts another card. [`Wcs`] is an interpretation.
//! It holds only the keywords that carry meaning in the description it
//! parsed. [`Wcs::to_header`] therefore emits that interpretation. Its
//! round-trip contract is `from_header(to_header(w)) == w`. It is not
//! byte fidelity to the source file.
//!
//! # Quick start
//!
//! Read an image. [`ImageHdu::read_physical`] applies `BZERO` and
//! `BSCALE` and returns `f64` pixels.
//!
//! ```
//! # use fitsy::{FitsWriter, ImageBuilder};
//! # let path = std::env::temp_dir().join("fitsy_doc_quickstart_read.fits");
//! # let (h, d) = ImageBuilder::new(vec![4_u64, 3], vec![1.0_f32; 12])?
//! #     .primary(true)
//! #     .build()?;
//! # let mut out = std::fs::File::create(&path)?;
//! # FitsWriter::new(&mut out).write_hdu(&h, &d)?;
//! use fitsy::{FitsError, FitsFile, Hdu};
//!
//! let file = FitsFile::open(&path)?;
//! let Hdu::Image(img) = file.hdu(0)? else {
//!     return Err(FitsError::Header("HDU 0 is not an image".into()));
//! };
//! let pixels = img.read_physical()?;
//!
//! assert_eq!(img.axes(), &[4, 3]);
//! assert_eq!(pixels.as_slice().len(), 12);
//! # std::fs::remove_file(&path)?;
//! # Ok::<(), fitsy::FitsError>(())
//! ```
//!
//! Write a two-dimensional image. [`ImageBuilder`] renders the header
//! and the big-endian data bytes, and [`FitsWriter`] writes them.
//!
//! ```
//! use fitsy::{FitsWriter, ImageBuilder};
//!
//! # let path = std::env::temp_dir().join("fitsy_doc_quickstart_write.fits");
//! let pixels: Vec<f32> = vec![0.0; 64 * 48];
//! let (header, data) = ImageBuilder::new(vec![64_u64, 48], pixels)?
//!     .primary(true)
//!     .card("OBJECT", "M42", Some("target"))
//!     .build()?;
//!
//! let mut out = std::fs::File::create(&path)?;
//! FitsWriter::new(&mut out).write_hdu(&header, &data)?;
//! # drop(out);
//! # assert_eq!(std::fs::metadata(&path)?.len() % 2880, 0);
//! # std::fs::remove_file(&path)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Convert a pixel coordinate to a sky coordinate. Pixel coordinates
//! in this API are 0-based, so the center of the first pixel is
//! `(0.0, 0.0)`.
//!
//! ```
//! # use fitsy::{FitsWriter, ImageBuilder};
//! # let path = std::env::temp_dir().join("fitsy_doc_quickstart_wcs.fits");
//! # let (h, d) = ImageBuilder::new(vec![64_u64, 48], vec![0.0_f32; 64 * 48])?
//! #     .primary(true)
//! #     .card("CTYPE1", "RA---TAN", None)
//! #     .card("CTYPE2", "DEC--TAN", None)
//! #     .card("CRPIX1", 32.0, None)
//! #     .card("CRPIX2", 24.0, None)
//! #     .card("CRVAL1", 150.0, None)
//! #     .card("CRVAL2", 2.5, None)
//! #     .card("CDELT1", -0.001, None)
//! #     .card("CDELT2", 0.001, None)
//! #     .build()?;
//! # let mut out = std::fs::File::create(&path)?;
//! # FitsWriter::new(&mut out).write_hdu(&h, &d)?;
//! use fitsy::FitsFile;
//!
//! let file = FitsFile::open(&path)?;
//! let wcs = file.wcs(0, ' ')?.expect("HDU 0 declares no WCS");
//! let world = wcs.pixel_to_world(&[31.0, 23.0])?;
//! let (ra, dec) = (world[0], world[1]);
//!
//! assert!((ra - 150.0).abs() < 1e-9);
//! assert!((dec - 2.5).abs() < 1e-9);
//! # std::fs::remove_file(&path)?;
//! # Ok::<(), fitsy::FitsError>(())
//! ```
//!
//! # Where to look next
//!
//! - Open a file: [`FitsFile::open`].
//! - Walk the HDUs: [`FitsFile::iter`] and [`Hdu`].
//! - Read image pixels: [`ImageHdu`].
//! - Read a binary table: [`BinTableHdu`].
//! - Read an ASCII table: [`AsciiTableHdu`].
//! - Build an image to write: [`ImageBuilder`].
//! - Build a binary table to write: [`BinTableBuilder`].
//! - Inspect header cards: [`Header`].
//! - Convert a pixel coordinate to a sky coordinate: [`Wcs`].
//! - Fit a WCS from pixel and sky pairs: [`wcs::fit_celestial_wcs`].
//!
//! # Runnable examples
//!
//! Each name below is a file under
//! [`examples/`](https://github.com/dahlend/fitsy/tree/main/examples).
//! Clone the repository and run one with `cargo run --example NAME`.
//!
//! - `read_image` -- opens an image, inspects the header, and decodes
//!   the pixels.
//! - `read_table` -- walks the columns of a binary table and decodes
//!   each cell.
//! - `write_image` -- builds and writes a two-dimensional image.
//! - `write_table` -- builds and writes a binary table of several
//!   columns.
//! - `wcs` -- transforms pixel coordinates to sky coordinates on the
//!   bundled NGC 2403 image.
//! - `fit_wcs` -- fits a celestial WCS from pixel and sky pairs.
//!
//! Sample data is in [`examples/data/`][data-dir]. Matching Python
//! scripts for the `python` feature are in [`examples/python/`][py-dir].
//!
//! [data-dir]: https://github.com/dahlend/fitsy/tree/main/examples/data
//! [py-dir]: https://github.com/dahlend/fitsy/tree/main/examples/python
//!
//! # Cargo features
//!
//! - `compression` *(default)* -- decodes `RICE_1`, `HCOMPRESS_1`,
//!   `PLIO_1`, `GZIP_1`, `GZIP_2`, and `NOCOMPRESS` tiles, and decodes
//!   quantized float tiles. Encodes `GZIP_1` tiles through
//!   [`compress_image_to_hdu`] and
//!   [`FitsWriter::write_hdu_compressed`](FitsWriter::write_hdu_compressed).
//!   Also inflates a whole-file gzip.
//! - `nalgebra` -- converts image data to `nalgebra::DMatrix` and adds
//!   batched coordinate transforms over that type.
//! - `faer` -- the same surface over `faer::Mat`.
//! - `python` -- builds the `PyO3` bindings for the `fitsy` Python
//!   wheel through [maturin](https://www.maturin.rs).

// The crate contains no `unsafe` outside the `python` module. The PyO3
// macros there expand to `unsafe` blocks, so `python.rs` carries an
// inner `#![allow(unsafe_code)]`. An inner allow has no effect under
// `forbid`, so this lint is `deny` instead.
#![deny(unsafe_code)]

pub mod data;
pub mod error;
pub mod hdu;
pub mod header;
pub mod io;
pub mod wcs;

pub mod checksum;
pub mod diff;
pub mod units;

#[cfg(feature = "compression")]
pub mod compression;

// Neither `nalgebra` nor `faer` is a default feature, so this module is
// absent from a default build. The crate docs above link to it, and
// `broken_intra_doc_links` is denied, so a build that enables neither
// feature fails to document. Build the documentation with
// `--all-features`, which is what the checklist and `docs.rs` both use
// (see `package.metadata.docs.rs` in Cargo.toml).
#[cfg(any(feature = "nalgebra", feature = "faer"))]
pub mod interop;

// Hidden from rustdoc. This module is the PyO3 binding surface, not
// Rust API. Its doc comments are numpydoc Python docstrings. They stay
// below four-space indentation so rustdoc does not collect them as
// doctests.
#[cfg(feature = "python")]
#[doc(hidden)]
pub mod python;

pub use data::{Bitpix, ImageData};
pub use error::FitsError;
pub use hdu::{
    AsciiCell, AsciiColumn, AsciiFormat, AsciiTableBuilder, AsciiTableHdu, BinColumn, BinFieldKind,
    BinFormat, BinTableBuilder, BinTableHdu, BinValue, FitsFile, Hdu, ImageBuilder, ImageHdu,
    ImagePixels,
};
pub use header::{Card, CommentaryKind, Diagnostic, Fix, Header, IsoDateTime, Level, Value};
#[cfg(not(target_arch = "wasm32"))]
pub use io::FitsAppender;
#[cfg(not(target_arch = "wasm32"))]
pub use io::FitsUpdater;
pub use io::{FitsWriter, write};
pub use wcs::{AxisKind, Wcs};

#[cfg(feature = "compression")]
pub use compression::{CompressedImageHdu, OwnedImage, TileOpts, compress_image_to_hdu};
