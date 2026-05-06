//! Error type for the crate.

use std::fmt;
use std::io;

#[derive(Debug)]
#[non_exhaustive]
pub enum FitsError {
    /// Underlying I/O error.
    Io(io::Error),

    /// Block-level violation (e.g., file size not a multiple of 2880,
    /// truncated block, padding not space/zero).
    Block { offset: u64, msg: String },

    /// Card-level violation (80-byte structure, keyword name).
    Card { offset: u64, msg: String },

    /// Value parsing failed for a card.
    Value { keyword: String, msg: String },

    /// Generic header-level violation.
    Header(String),

    /// A mandatory keyword is missing or has the wrong type/value.
    MissingMandatory { keyword: String },

    /// `END` card not in the last header block of the HDU, or block
    /// after `END` not entirely ASCII spaces.
    EndCardMisplaced { offset: u64 },

    /// HDU type mismatch (e.g., expected IMAGE got BINTABLE).
    HduMismatch {
        expected: &'static str,
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
    InHdu { index: usize, source: Box<Self> },
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

pub type Result<T> = std::result::Result<T, FitsError>;
