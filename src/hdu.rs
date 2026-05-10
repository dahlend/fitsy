//! Hierarchical Data Unit (HDU) layer (Standard Sec.3.4).
//!
//! The top-level entry point is [`FitsFile`], which opens a file and
//! gives access to each HDU via [`FitsFile::hdu`] or [`FitsFile::iter`].
//! Each HDU is returned as an [`Hdu`] enum variant.

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
pub use file::{FitsFile, FitsOpenOptions};
pub use image::{ImageHdu, ImagePixels};
pub use kind::{ConformingHdu, Hdu};
pub use random_groups::RandomGroupsHdu;
