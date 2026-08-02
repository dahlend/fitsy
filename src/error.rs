//! The error type of the crate.
//!
//! [`FitsError`] is the single error type that every fallible function
//! here returns. Its variants group by the level of the format they
//! belong to: block, card, value, header and data. A caller therefore
//! tells a structural violation from a semantic one without matching
//! on message text.
//!
//! The enum is `non_exhaustive`. A new variant is not a breaking
//! change, so a caller matches with a wildcard arm.

use std::fmt;
use std::io;

/// Every way that reading or writing a FITS file can fail.
///
/// The variants group by the level of the format they belong to:
/// block, card, value, header and data. A caller therefore tells a
/// structural violation from a semantic one without matching on
/// message text.
///
/// # Examples
///
/// ```
/// use fitsy::{FitsError, Header};
///
/// // A keyword outside the FITS character set is a header-level error.
/// let mut h = Header::empty();
/// let err = h.push("bad key!", 1_i64, None).unwrap_err();
///
/// match err {
///     FitsError::Header(msg) => assert!(msg.contains("bad key!")),
///     other => panic!("expected a header error, got {other:?}"),
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum FitsError {
    /// Underlying I/O error.
    Io(io::Error),

    /// Block-level violation (e.g., file size not a multiple of 2880,
    /// truncated block, padding not space/zero).
    Block {
        /// Byte offset of the offending block from the start of the file.
        offset: u64,
        /// What was wrong with it.
        msg: String,
    },

    /// Card-level violation (80-byte structure, keyword name).
    Card {
        /// Byte offset of the offending 80-byte card.
        offset: u64,
        /// What was wrong with it.
        msg: String,
    },

    /// Value parsing failed for a card.
    Value {
        /// Keyword whose value could not be read.
        keyword: String,
        /// Why the value was rejected.
        msg: String,
    },

    /// Generic header-level violation.
    Header(String),

    /// A mandatory keyword is missing or has the wrong type/value.
    MissingMandatory {
        /// The keyword the structure required.
        keyword: String,
    },

    /// `END` card not in the last header block of the HDU, or block
    /// after `END` not entirely ASCII spaces.
    EndCardMisplaced {
        /// Byte offset of the block where `END` was found.
        offset: u64,
    },

    /// HDU type mismatch (e.g., expected IMAGE got BINTABLE).
    HduMismatch {
        /// The HDU kind the caller asked for.
        expected: &'static str,
        /// The kind actually present.
        found: String,
    },

    /// Data section violation (size, scaling, blank handling).
    Data(String),

    /// Encountered a non-standard or unrecognized construct.
    NonStandard(String),

    /// WCS construction failed.
    Wcs(String),

    /// Checksum validation failed.
    Checksum(String),

    /// Wraps another error with the index of the HDU that produced
    /// it. Emitted by [`crate::FitsFile::iter`] (and friends) so a
    /// failure deep inside multi-HDU traversal can be located
    /// without rewinding the iterator.
    InHdu {
        /// Zero-based index of the HDU that produced `source`.
        index: usize,
        /// The underlying failure.
        source: Box<Self>,
    },
}

impl fmt::Display for FitsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "i/o error: {e}"),
            Self::Block { offset, msg } => write!(f, "block at byte {offset}: {msg}"),
            Self::Card { offset, msg } => write!(f, "card at byte {offset}: {msg}"),
            Self::Value { keyword, msg } => write!(f, "value for keyword `{keyword}`: {msg}"),
            Self::Header(m) => write!(f, "header: {m}"),
            Self::MissingMandatory { keyword } => {
                write!(f, "missing mandatory keyword `{keyword}`")
            }
            Self::EndCardMisplaced { offset } => {
                write!(f, "END card misplaced (block at byte {offset})")
            }
            Self::HduMismatch { expected, found } => {
                write!(f, "expected HDU type `{expected}`, found `{found}`")
            }
            Self::Data(m) => write!(f, "data: {m}"),
            Self::NonStandard(m) => write!(f, "non-standard construct: {m}"),
            Self::Wcs(m) => write!(f, "wcs: {m}"),
            Self::Checksum(m) => write!(f, "checksum: {m}"),
            Self::InHdu { index, source } => write!(f, "in HDU {index}: {source}"),
        }
    }
}

impl std::error::Error for FitsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::InHdu { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl From<io::Error> for FitsError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// `Result` with this crate's error type.
pub type Result<T> = std::result::Result<T, FitsError>;
