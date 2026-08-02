//! Header and Data Unit (HDU) layer (Standard Sec.3.4).
//!
//! # Purpose
//!
//! This module reads and builds the HDUs of a FITS file. [`FitsFile`]
//! opens a file and reaches each HDU through [`FitsFile::hdu`] or
//! [`FitsFile::iter`]. Each HDU arrives as a variant of [`Hdu`].
//!
//! # Layout
//!
//! Each submodule owns one kind of HDU, or one part of the read path:
//!
//! - [`mod@file`] -- [`FitsFile`], the reader that owns the file
//!   bytes.
//! - [`kind`] -- [`Hdu`], the enum that dispatches on HDU type, and
//!   [`ConformingHdu`] for an unrecognized `XTENSION`.
//! - [`image`] -- [`ImageHdu`] and its pixel decoders.
//! - [`bintable`] -- [`BinTableHdu`] and its cell decoders.
//! - [`ascii_table`] -- [`AsciiTableHdu`] and its cell decoders.
//! - [`random_groups`] -- [`RandomGroupsHdu`] (Standard Sec.6). This
//!   HDU type is read only.
//! - [`builder`] -- [`ImageBuilder`], [`BinTableBuilder`] and
//!   [`AsciiTableBuilder`], which render a header and data bytes for
//!   the write path.
//!
//! # Design constraints
//!
//! Every HDU type borrows its data bytes from the [`FitsFile`] that
//! produced it. The lifetime parameter on [`Hdu`] carries that borrow,
//! so an HDU cannot outlive its file. A caller that needs an owned
//! value decodes the HDU into an [`ImageData`](crate::data::ImageData)
//! or a cell vector first.

pub mod ascii_table;
pub mod bintable;
pub mod builder;
pub mod file;
pub mod image;
pub mod kind;
pub mod random_groups;
pub(crate) mod subarray;

pub use ascii_table::{AsciiCell, AsciiColumn, AsciiFormat, AsciiTableHdu};
pub use bintable::{BinColumn, BinFieldKind, BinFormat, BinTableHdu, BinValue, IntStorage};
pub use builder::{AsciiColumnData, AsciiTableBuilder, BinTableBuilder, ImageBuilder};
pub use file::FitsFile;
pub use image::{ImageHdu, ImagePixels};
pub use kind::{ConformingHdu, Hdu};
pub use random_groups::{GroupParameter, RandomGroupsHdu};
