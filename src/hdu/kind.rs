//! The [`Hdu`] enum, which dispatches on the type of a parsed HDU.
//!
//! Each variant holds a borrowed view over the file bytes. The
//! lifetime `'a` is the lifetime of the [`FitsFile`](crate::FitsFile)
//! that produced the HDU.

use crate::header::Header;

use super::ascii_table::AsciiTableHdu;
use super::bintable::BinTableHdu;
use super::image::ImageHdu;
use super::random_groups::RandomGroupsHdu;

/// The HDU kinds that this crate recognizes.
///
/// An `XTENSION` value outside this set is not an error. It arrives as
/// [`Hdu::Conforming`], so a caller still reads its header and its raw
/// data bytes.
///
/// # Examples
///
/// ```
/// # use fitsy::{FitsWriter, ImageBuilder};
/// # let (h, d) = ImageBuilder::new(vec![4_u64, 3], vec![7_i16; 12])?
/// #     .primary(true)
/// #     .build()?;
/// # let mut buf: Vec<u8> = Vec::new();
/// # FitsWriter::new(&mut buf).write_hdu(&h, &d)?;
/// use fitsy::{FitsFile, Hdu};
///
/// let file = FitsFile::from_bytes(buf)?;
/// for hdu in file.iter() {
///     match hdu? {
///         Hdu::Image(img) => assert_eq!(img.axes(), &[4, 3]),
///         Hdu::BinTable(t) => println!("{} rows", t.n_rows()),
///         other => println!("other kind: {other:?}"),
///     }
/// }
/// # Ok::<(), fitsy::FitsError>(())
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Hdu<'a> {
    /// Primary array or `IMAGE` extension (Standard Sec.7.1).
    Image(ImageHdu<'a>),
    /// Random Groups primary HDU (Standard Sec.6).
    RandomGroups(RandomGroupsHdu<'a>),
    /// `TABLE` extension (Standard Sec.7.2).
    AsciiTable(AsciiTableHdu<'a>),
    /// `BINTABLE` extension (Standard Sec.7.3).
    BinTable(BinTableHdu<'a>),
    /// A `BINTABLE` carrying a tile-compressed image (`ZIMAGE = T`,
    /// Pence & Seaman 2010 / FITS standard 2016 Sec.7.4).
    /// Only available when the `compression` feature is enabled.
    #[cfg(feature = "compression")]
    CompressedImage(crate::compression::CompressedImageHdu<'a>),
    /// An `XTENSION` whose type this crate does not recognize. A
    /// caller still reads its header and its raw data bytes.
    Conforming(ConformingHdu<'a>),
}

/// An HDU whose `XTENSION` is not specifically handled by this crate.
///
/// Construct via [`ConformingHdu::new`]. Inspect via [`header`](Self::header),
/// [`data_bytes`](Self::data_bytes), and [`xtension`](Self::xtension).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ConformingHdu<'a> {
    header: Header,
    /// Raw data bytes (size already validated against header).
    data: &'a [u8],
    /// Value of the `XTENSION` keyword.
    xtension: String,
}

impl<'a> ConformingHdu<'a> {
    /// Construct from a parsed header and the raw data section.
    ///
    /// The `data` argument covers the data section without its
    /// trailing block padding. The `xtension` argument holds the
    /// `XTENSION` value, already trimmed of its padding spaces.
    #[must_use]
    pub fn new(header: Header, data: &'a [u8], xtension: String) -> Self {
        Self {
            header,
            data,
            xtension,
        }
    }

    /// The parsed header.
    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// The raw data bytes (no padding).
    #[must_use]
    pub fn data_bytes(&self) -> &[u8] {
        self.data
    }

    /// The `XTENSION` keyword value, trimmed.
    #[must_use]
    pub fn xtension(&self) -> &str {
        &self.xtension
    }
}

impl Hdu<'_> {
    /// Borrow the parsed header of this HDU.
    #[must_use]
    pub fn header(&self) -> &Header {
        match self {
            Hdu::Image(h) => h.header(),
            Hdu::RandomGroups(h) => h.header(),
            Hdu::AsciiTable(h) => h.header(),
            Hdu::BinTable(h) => h.header(),
            #[cfg(feature = "compression")]
            Hdu::CompressedImage(h) => h.as_bintable().header(),
            Hdu::Conforming(h) => h.header(),
        }
    }

    /// Borrow the raw data bytes of this HDU. The slice is the
    /// data section as it appears in the file (no padding).
    #[must_use]
    pub fn data_bytes(&self) -> &[u8] {
        match self {
            Hdu::Image(h) => h.raw_bytes(),
            Hdu::RandomGroups(h) => h.raw_bytes(),
            Hdu::AsciiTable(h) => h.data_bytes(),
            Hdu::BinTable(h) => h.data_bytes(),
            #[cfg(feature = "compression")]
            Hdu::CompressedImage(h) => h.as_bintable().data_bytes(),
            Hdu::Conforming(h) => h.data_bytes(),
        }
    }
}
