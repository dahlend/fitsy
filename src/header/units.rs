//! Units strings (Standard Sec.4.3).
//!
//! Units values (`BUNIT`, `CUNIT`, `TUNIT`, ...) are stored verbatim;
//! no IAU grammar validation is performed.

/// A units string as written in a header value (e.g. `BUNIT`,
/// `CUNIT`, `TUNIT`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitsString(pub String);

impl UnitsString {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
