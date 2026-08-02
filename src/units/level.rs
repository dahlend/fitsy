//! Levels -- `mag`, `dB`, `Sun` -- and the references they are
//! measured against.

use super::dimension::{Dimension, F_LAMBDA, F_NU};

// -- levels -------------------------------------------------------------

/// A zero point fixed by definition, so a level against it names a real
/// physical quantity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zero {
    /// How it is spelled, e.g. `AB`.
    pub name: &'static str,
    /// Its value in the canonical units of [`Zero::dimension`].
    pub value: f64,
    /// What the zero point is a quantity *of*.
    pub dimension: Dimension,
}

/// `m_AB = -2.5 log10(f_nu) - 48.60`, i.e. 3630.78 Jy.
pub const AB: Zero = Zero {
    name: "AB",
    value: 3.630_780_547_701e-23,
    dimension: F_NU,
};

/// `m_ST = -2.5 log10(f_lambda) - 21.10`, i.e. 3.631e-9 erg/s/cm2/Angstrom.
pub const ST: Zero = Zero {
    name: "ST",
    value: 3.630_780_547_701e-2,
    dimension: F_LAMBDA,
};

/// An object a level is measured against, whose value this unit string
/// cannot pin down.
///
/// `Vega` is the reason this is a separate type rather than another
/// [`Zero`]. Its zero point depends on the passband and on the adopted
/// Vega spectrum, so no single number defines it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Object {
    /// Relative to the Sun -- `Sun`.
    Solar,
    /// Relative to Vega -- `mag(Vega)`.
    Vega,
}

/// What a level's ratio is taken against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reference {
    /// No stated zero point: bare `mag`, `dB`, `log()`.
    Unspecified,
    /// A zero point fixed by definition.
    Absolute(Zero),
    /// An object whose value is not fixed by the unit string.
    Object(Object),
}

/// A value that is a *level* rather than the quantity itself.
///
/// A value `v` in a unit with level `L` and linear part `scale`
/// stands for the physical quantity
///
/// ```text
/// base^(v / factor) * scale     logarithmic (`log_base` is `Some`)
/// v * scale                     a plain ratio (`Sun`)
/// ```
///
/// Which is what makes rescaling the linear part *additive*: it moves
/// inside the logarithm and comes out as an offset.
///
/// [`PartialEq`] compares with a tolerance, as [`super::Dimension`]
/// does. The `factor` field is the output of arithmetic once a prefix
/// or a numeric multiplier folds into it. Two spellings of the same
/// level can therefore differ in the last bit. Such a comparison is
/// not transitive, so this type implements no `Eq`.
#[derive(Debug, Clone, Copy)]
pub struct Level {
    /// The logarithm's base; `None` for a plain ratio.
    pub log_base: Option<f64>,
    /// The multiplier outside the logarithm. It is -2.5 for `mag`, 10
    /// for `dB`, and 1 for `Np` and for a bare `log()`.
    ///
    /// A prefix or numeric multiplier on the symbol scales the value
    /// of the level, so it folds in here. `mmag` therefore has factor
    /// -2500, because `v` mmag stands for `10^(v / -2500)`.
    pub factor: f64,
    /// What the ratio is taken against.
    pub reference: Reference,
}

impl PartialEq for Level {
    fn eq(&self, other: &Self) -> bool {
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
        self.reference == other.reference
            && close(self.factor, other.factor)
            && match (self.log_base, other.log_base) {
                (None, None) => true,
                (Some(a), Some(b)) => close(a, b),
                _ => false,
            }
    }
}

impl Level {
    pub(super) const fn log(base: f64, factor: f64, reference: Reference) -> Self {
        Self {
            log_base: Some(base),
            factor,
            reference,
        }
    }

    /// The astronomical magnitude: base 10, and the -2.5 that makes five
    /// magnitudes a factor of a hundred.
    pub const MAG: Self = Self::log(10.0, -2.5, Reference::Unspecified);
    /// `mag(AB)`.
    pub const AB_MAG: Self = Self::log(10.0, -2.5, Reference::Absolute(AB));
    /// `mag(ST)`.
    pub const ST_MAG: Self = Self::log(10.0, -2.5, Reference::Absolute(ST));
    /// `mag(Vega)`.
    pub const VEGA_MAG: Self = Self::log(10.0, -2.5, Reference::Object(Object::Vega));
    /// The decibel, on the power convention: `10 log10(ratio)`.
    pub const DB: Self = Self::log(10.0, 10.0, Reference::Unspecified);
    /// The neper: `ln(ratio)`.
    pub const NP: Self = Self::log(std::f64::consts::E, 1.0, Reference::Unspecified);
    /// `Sun` -- a plain ratio against a solar value, not a logarithm.
    pub const SUN: Self = Self {
        log_base: None,
        factor: 1.0,
        reference: Reference::Object(Object::Solar),
    };
}
