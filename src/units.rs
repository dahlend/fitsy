//! FITS unit strings (Standard Sec.4.3).
//!
//! Parses the compound syntax of Table 6 into a scale factor and a set
//! of dimensional exponents, so a unit can be *converted* and, more
//! importantly, *checked*: a wavelength axis that declares `'Hz'` is a
//! broken header, not a wavelength in disguise.
//!
//! # Syntax
//!
//! ```text
//! str1 str2   str1*str2   str1.str2     multiplication
//! str1/str2                             division
//! str1**expr  str1^expr   str1expr      exponentiation
//! sqrt(str1)                            square root
//! (...)                                 grouping
//! 10**k  10^k  10+k  10-k               numeric multiplier
//! ```
//!
//! `expr` may be a signed integer, a decimal, or a ratio of integers;
//! the last two must be parenthesised. Multiplication and division
//! share a precedence level and associate left to right, so
//! `erg/s/cm2/Angstrom` reads as `erg s^-1 cm^-2 Angstrom^-1`, which
//! is the form a flux-calibrated spectrum uses.
//!
//! Case is significant, per the IAU convention that the standard
//! cites.
//!
//! # Canonical units
//!
//! [`Unit::scale`] is relative to metre, kilogram, second, degree,
//! kelvin, ampere, mole and candela. The angular base is degree rather
//! than radian, because the standard specifies every angular FITS
//! keyword in degrees.
//!
//! # Levels
//!
//! `mag`, `dB` and `Sun` are not dimensions. They are levels: a value
//! measured against a reference. For `mag` that measurement is
//! logarithmic.
//!
//! A level therefore converts by an offset, not by a factor.
//! `mag/arcsec2` and `mag/deg2` differ by an additive -17.78, not by
//! `3600^2`. Conversion here yields a [`Converter`] rather than an
//! `f64` for that reason. [`Level`] carries the log base, the
//! multiplier outside it, and the [`Reference`] that the ratio is
//! taken against.
//!
//! A prefix or numeric multiplier on a logarithmic level scales the
//! value of that level. 5 `mmag`, written also as `0.001 mag`, is
//! 0.005 `mag`. A dimensioned linear part, such as the `arcsec^-2`
//! above, instead remains the reference underneath the logarithm. That
//! is what keeps the conversion additive.
//!
//! # Equivalencies
//!
//! Some conversions are physics, not unit algebra: nm to Hz needs `c`.
//! Those are [`Equivalence`] implementations. [`Spectral`] lives here,
//! being parameterless and universal -- the base table already embeds
//! `c`, since a light year *is* `c` times a Julian year. Anything
//! carrying a parameter or a convention, like the Sec.8.4 doppler
//! definitions, stays with the standard that specifies it in
//! [`crate::wcs::spectral`].
//!
//! # The `[unit]` comment convention
//!
//! Sec.4.3.2 recommends recording a keyword's unit in its inline
//! comment, in square brackets. [`parse_comment_unit`] extracts that;
//! [`Header::keyword_unit`](crate::Header::keyword_unit) applies it.
//!
//! # Informal spellings
//!
//! [`parse_unit`] holds the formal keywords (`BUNIT`, `CUNITia`,
//! `TUNITn`) to the Sec.4.3 grammar. The `[unit]` comment convention
//! and `TIMEUNIT` carry looser spellings in practice -- `degrees`,
//! `sec`, `DAY`, `AU/day` -- so [`parse_unit_lenient`] and
//! [`factor_to_lenient`] accept those as well, trying the grammar
//! first.
//!
//! # Deliberate non-support
//!
//! `exp()` is recognized but rejected: it is not a unit. `log()` and
//! `ln()` *are* supported, as levels over their argument -- `log(Hz)`
//! agrees with `log(Hz)`, differs from `log(kHz)` by an additive 3, and
//! still refuses `Hz`.
//!
//! Converting a level to the linear quantity underneath it -- 0 `mag(AB)`
//! is 3630.78 `Jy` -- is exponential rather than affine and is left to
//! callers, who can read the [`Reference`] off the parsed unit. A unit
//! module should not be doing photometry.

pub mod constants;

mod convert;
mod dimension;
mod equivalencies;
mod lenient;
mod level;
mod parse;
mod quantity;
mod table;
mod unit;

#[cfg(test)]
mod tests;

pub use convert::{Equivalence, convert, convert_with, factor_to};
pub use dimension::{Dimension, dimensions};
pub use equivalencies::Spectral;
pub use lenient::{factor_to_lenient, parse_unit_lenient};
pub use level::{AB, Level, Object, Reference, ST, Zero};
pub use parse::{parse_comment_unit, parse_unit};
pub use quantity::Quantity;
pub use unit::{Converter, Unit};
