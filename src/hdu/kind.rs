//! The [`Hdu`] enum, which dispatches on the type of a parsed HDU.
//!
//! Each variant holds a borrowed view over the file bytes. The
//! lifetime `'a` is the lifetime of the [`FitsFile`](crate::FitsFile)
//! that produced the HDU.

use std::borrow::Cow;

use crate::header::Header;
use crate::header::value::Value;

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
/// # let hdu = ImageBuilder::new(vec![4_u64, 3], vec![7_i16; 12])?
/// #     .primary(true)
/// #     .build()?;
/// # let mut buf: Vec<u8> = Vec::new();
/// # FitsWriter::new(&mut buf).write_hdu(&hdu)?;
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
// `non_exhaustive` stays here even though the standard is complete.
// The `CompressedImage` variant is feature-gated. An exhaustive match
// written without the `compression` feature would stop compiling when
// any crate in the graph enables it, and features must stay additive.
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

/// One HDU's header and its data section.
///
/// Every HDU view implements this, and so does a `(Header, Vec<u8>)`
/// pair. [`FitsWriter::write_hdu`](crate::FitsWriter::write_hdu) takes
/// it, so what a builder returns goes to the writer whole.
pub trait HduBytes {
    /// The HDU's header.
    fn header(&self) -> &Header;
    /// The data section, without trailing block padding.
    fn data_bytes(&self) -> &[u8];
}

impl HduBytes for (Header, Vec<u8>) {
    fn header(&self) -> &Header {
        &self.0
    }
    fn data_bytes(&self) -> &[u8] {
        &self.1
    }
}

impl HduBytes for Hdu<'_> {
    fn header(&self) -> &Header {
        Self::header(self)
    }
    fn data_bytes(&self) -> &[u8] {
        Self::data_bytes(self)
    }
}

impl HduBytes for ImageHdu<'_> {
    fn header(&self) -> &Header {
        Self::header(self)
    }
    fn data_bytes(&self) -> &[u8] {
        self.raw_bytes()
    }
}

impl HduBytes for RandomGroupsHdu<'_> {
    fn header(&self) -> &Header {
        Self::header(self)
    }
    fn data_bytes(&self) -> &[u8] {
        self.raw_bytes()
    }
}

impl HduBytes for AsciiTableHdu<'_> {
    fn header(&self) -> &Header {
        Self::header(self)
    }
    fn data_bytes(&self) -> &[u8] {
        Self::data_bytes(self)
    }
}

impl HduBytes for BinTableHdu<'_> {
    fn header(&self) -> &Header {
        Self::header(self)
    }
    fn data_bytes(&self) -> &[u8] {
        Self::data_bytes(self)
    }
}

impl HduBytes for ConformingHdu<'_> {
    fn header(&self) -> &Header {
        Self::header(self)
    }
    fn data_bytes(&self) -> &[u8] {
        Self::data_bytes(self)
    }
}

/// What kind of HDU a header describes.
///
/// [`HduKind::from_header`] decides this from the header alone, so a
/// caller surveys a file without reading one data byte.
/// [`FitsFile::kind`](crate::FitsFile::kind) is the accessor that
/// applies it to an open file.
///
/// This mirrors the variants of [`Hdu`]. It differs in one place:
/// [`HduKind::CompressedImage`] is reported whether or not the
/// `compression` feature is enabled, because `ZIMAGE = T` is a header
/// fact and needs no codec.
///
/// # Examples
///
/// ```
/// # use fitsy::{FitsWriter, ImageBuilder};
/// # let hdu = ImageBuilder::new(vec![4_u64, 3], vec![7_i16; 12])?
/// #     .primary(true)
/// #     .build()?;
/// # let mut buf: Vec<u8> = Vec::new();
/// # FitsWriter::new(&mut buf).write_hdu(&hdu)?;
/// use fitsy::{FitsFile, HduKind};
///
/// let file = FitsFile::from_bytes(buf)?;
/// assert_eq!(file.kind(0)?, HduKind::Image);
/// # Ok::<(), fitsy::FitsError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum HduKind {
    /// Primary array or `IMAGE` extension (Standard Sec.7.1).
    Image,
    /// Random Groups primary HDU (Standard Sec.6).
    RandomGroups,
    /// `TABLE` extension (Standard Sec.7.2).
    AsciiTable,
    /// `BINTABLE` extension (Standard Sec.7.3).
    BinTable,
    /// A `BINTABLE` carrying a tile-compressed image (`ZIMAGE = T`,
    /// Standard Sec.10).
    CompressedImage,
    /// An `XTENSION` this crate does not model. Holds its value.
    Conforming(String),
    /// An extension header with no readable `XTENSION` card.
    Unknown,
}

impl HduKind {
    /// Decide the kind of the HDU that `header` describes.
    ///
    /// The `index` argument is the HDU's position in the file. Index 0
    /// is the primary HDU, which declares `SIMPLE` rather than
    /// `XTENSION`, so it is an image unless it declares Random Groups.
    #[must_use]
    pub fn from_header(header: &Header, index: usize) -> Self {
        if index == 0 {
            return if header.is_random_groups() {
                Self::RandomGroups
            } else {
                Self::Image
            };
        }
        match header.first("XTENSION") {
            Some(Value::String(s)) if s == "IMAGE" => Self::Image,
            Some(Value::String(s)) if s == "TABLE" => Self::AsciiTable,
            Some(Value::String(s)) if s == "BINTABLE" => {
                if matches!(header.first("ZIMAGE"), Some(Value::Logical(true))) {
                    Self::CompressedImage
                } else {
                    Self::BinTable
                }
            }
            Some(Value::String(s)) => Self::Conforming(s.clone()),
            _ => Self::Unknown,
        }
    }

    /// Whether this kind holds pixels, compressed or not.
    #[must_use]
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image | Self::CompressedImage)
    }
}

impl std::fmt::Display for HduKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Image => f.write_str("IMAGE"),
            Self::RandomGroups => f.write_str("RANDOM-GROUPS"),
            Self::AsciiTable => f.write_str("TABLE"),
            Self::BinTable => f.write_str("BINTABLE"),
            Self::CompressedImage => f.write_str("compressed IMAGE"),
            Self::Conforming(s) => write!(f, "{s}"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

/// An HDU whose `XTENSION` is not specifically handled by this crate.
///
/// Construct via [`ConformingHdu::new`]. Inspect via [`header`](Self::header),
/// [`data_bytes`](Self::data_bytes), and [`xtension`](Self::xtension).
#[derive(Debug, Clone)]
pub struct ConformingHdu<'a> {
    header: Header,
    /// Raw data bytes (size already validated against header).
    data: Cow<'a, [u8]>,
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
    pub fn new(header: Header, data: impl Into<Cow<'a, [u8]>>, xtension: String) -> Self {
        Self {
            header,
            data: data.into(),
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
        &self.data
    }

    /// The `XTENSION` keyword value, trimmed.
    #[must_use]
    pub fn xtension(&self) -> &str {
        &self.xtension
    }
}

impl<'a> Hdu<'a> {
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

    /// Borrow this HDU as a binary table, when it is one.
    ///
    /// A tile-compressed image is a `BINTABLE` that carries
    /// `ZIMAGE = T`, so it answers `Some` here too. A caller that
    /// reads columns, rows or the heap reaches both kinds through
    /// this one accessor.
    #[must_use]
    pub fn bintable(&self) -> Option<&BinTableHdu<'a>> {
        match self {
            Self::BinTable(t) => Some(t),
            #[cfg(feature = "compression")]
            Self::CompressedImage(c) => Some(c.as_bintable()),
            _ => None,
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
