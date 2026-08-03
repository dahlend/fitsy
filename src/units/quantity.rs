//! [`Quantity`]: a number with a unit attached.
//!
//! [`Unit`] answers what kind of thing this is, and how big one of
//! them is. [`Quantity`] is a value measured in one.
//!
//! Arithmetic here is fallible rather than operator-overloaded.
//! Multiplying two levels yields no quantity, and neither does adding
//! a length to a time. A method returning [`Result`] reports those
//! cases. An operator could only panic on them.

use super::dimension::Dimension;
use super::unit::Unit;
use crate::error::{FitsError, Result};

/// A value together with the unit it is measured in.
///
/// ```
/// use fitsy::units::{Quantity, Unit, constants, dimensions, parse_unit};
///
/// // A wavelength, in nanometers.
/// let lambda = Quantity::new(500.0, parse_unit("nm")?);
/// assert!((lambda.to_canonical()? - 500e-9).abs() < 1e-20);
///
/// // Photon energy: E = h c / lambda, dimensions checked at every step.
/// let e = constants::PLANCK
///     .try_mul(constants::SPEED_OF_LIGHT)?
///     .try_div(lambda)?;
/// assert_eq!(e.unit.dimension, dimensions::ENERGY);
/// # Ok::<(), fitsy::FitsError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity {
    /// The number, as measured in [`Quantity::unit`].
    pub value: f64,
    /// What the number is measured in.
    pub unit: Unit,
}

impl Quantity {
    /// A value in `unit`.
    #[must_use]
    pub const fn new(value: f64, unit: Unit) -> Self {
        Self { value, unit }
    }

    /// A value in the canonical unit for `dimension` -- meter, kilogram,
    /// second, degree, kelvin, ampere, mole, candela.
    #[must_use]
    pub const fn canonical(value: f64, dimension: Dimension) -> Self {
        Self::new(value, Unit::new(1.0, dimension))
    }

    /// A pure number.
    #[must_use]
    pub const fn scalar(value: f64) -> Self {
        Self::new(value, Unit::scalar())
    }

    /// The same quantity expressed in `unit`.
    ///
    /// The conversion is affine, so this is correct for levels too: a
    /// surface brightness in `mag/arcsec2` becomes `mag/deg2` by a shift
    /// of -17.78, not by a factor.
    ///
    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if `unit` is not the same
    /// quantity -- a different dimension, or a level against a linear
    /// unit.
    pub fn to(self, unit: Unit) -> Result<Self> {
        Ok(Self::new(
            self.unit.converter_to(unit)?.apply(self.value),
            unit,
        ))
    }

    /// The value in canonical units, discarding the unit.
    ///
    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if this is a level, where
    /// "the canonical unit" has no meaning -- a magnitude is not a
    /// rescaled anything.
    pub fn to_canonical(self) -> Result<f64> {
        Ok(self.to(Unit::new(1.0, self.unit.dimension))?.value)
    }

    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if both sides carry a level.
    pub fn try_mul(self, other: Self) -> Result<Self> {
        Ok(Self::new(
            self.value * other.value,
            self.unit.try_mul(other.unit)?,
        ))
    }

    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if both sides carry a level,
    /// or the divisor is logarithmic.
    pub fn try_div(self, other: Self) -> Result<Self> {
        Ok(Self::new(
            self.value / other.value,
            self.unit.try_div(other.unit)?,
        ))
    }

    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if this is a logarithmic level.
    pub fn try_pow(self, e: f64) -> Result<Self> {
        Ok(Self::new(self.value.powf(e), self.unit.try_pow(e)?))
    }

    /// Add, converting `other` into this one's unit first.
    ///
    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if the two are not the same
    /// quantity. A length plus a time is not a mistake this should
    /// silently absorb.
    pub fn try_add(self, other: Self) -> Result<Self> {
        Ok(Self::new(
            self.value + other.to(self.unit)?.value,
            self.unit,
        ))
    }

    /// Subtract, converting `other` into this one's unit first.
    ///
    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if the two are not the same
    /// quantity.
    pub fn try_sub(self, other: Self) -> Result<Self> {
        Ok(Self::new(
            self.value - other.to(self.unit)?.value,
            self.unit,
        ))
    }

    /// Scale by a plain number, which no unit algebra can refuse.
    #[must_use]
    pub fn scaled(self, k: f64) -> Self {
        Self::new(self.value * k, self.unit)
    }

    /// True if this measures the same kind of thing as `other`, so the
    /// two can be added and compared.
    #[must_use]
    pub fn commensurable_with(self, other: Self) -> bool {
        self.unit.converter_to(other.unit).is_ok()
    }

    /// Parse a unit string and attach `value` to it.
    ///
    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if `unit` is not valid Sec.4.3
    /// syntax or names a unit outside Tables 4-5.
    pub fn parse(value: f64, unit: &str) -> Result<Self> {
        Ok(Self::new(value, super::parse::parse_unit(unit)?))
    }
}

/// `value unit`, e.g. `299792458 m s^-1`. The unit is the canonical
/// decomposition, as for [`Unit`] itself.
///
/// Values outside roughly `1e-4` to `1e7` render in exponential form,
/// since the constants in [`super::constants`] reach 1e-34 and writing
/// that out in full is unreadable rather than precise.
impl std::fmt::Display for Quantity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = self.value.abs();
        let unit = self.unit.to_string();
        let value = if m != 0.0 && !(1e-4..1e7).contains(&m) {
            format!("{:e}", self.value)
        } else {
            format!("{}", self.value)
        };
        if unit.is_empty() {
            f.write_str(&value)
        } else {
            write!(f, "{value} {unit}")
        }
    }
}

impl From<f64> for Quantity {
    fn from(value: f64) -> Self {
        Self::scalar(value)
    }
}

/// Not `PartialOrd`: two quantities are only comparable when they are
/// commensurable, and that is a fallible question.
impl Quantity {
    /// Compare against `other`, converting first.
    ///
    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if the two are not the same
    /// quantity.
    pub fn try_cmp(self, other: Self) -> Result<std::cmp::Ordering> {
        let o = other.to(self.unit)?;
        self.value
            .partial_cmp(&o.value)
            .ok_or_else(|| FitsError::Header("cannot order a NaN quantity".to_string()))
    }
}
