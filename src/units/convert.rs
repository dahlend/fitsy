//! Conversion: the narrow factor form, the general form, and the
//! [`Equivalence`] escape hatch for relations unit algebra cannot reach.

use super::dimension::Dimension;
use super::parse::parse_unit;
use super::unit::Unit;
use crate::error::{FitsError, Result};

/// Factor converting a value in `unit` to the canonical unit for
/// `canonical`, checking that the two describe the same physical
/// quantity.
///
/// A blank `unit` is taken to be canonical already (factor 1): `CUNIT`
/// defaults to blank, and the standard fixes the units of the keywords
/// where that matters.
///
/// # Errors
///
/// [`FitsError::Header`] if `unit` fails to parse or carries the wrong
/// dimension.
pub fn factor_to(unit: &str, canonical: Dimension) -> Result<f64> {
    let trimmed = unit.trim();
    if trimmed.is_empty() {
        return Ok(1.0);
    }
    let q = parse_unit(trimmed)?;
    let c = q.converter_to(Unit::new(1.0, canonical)).map_err(|e| {
        // Re-word with the offending string, which `converter_to` does
        // not have.
        FitsError::Header(format!("unit `{trimmed}`: {e}"))
    })?;
    c.as_factor().ok_or_else(|| {
        FitsError::Header(format!(
            "unit `{trimmed}` converts by a shift of {}, not a factor; use \
             `Unit::converter_to`",
            c.offset
        ))
    })
}

/// Convert one value between two unit strings, levels included.
///
/// The `value` argument is measured in the unit that `from` spells,
/// and the result is measured in the unit that `to` spells.
///
/// This is the general form. [`factor_to`] is the special case where a
/// plain multiplier suffices.
///
/// # Errors
///
/// [`FitsError::Header`] if either string fails to parse, or the two are
/// not the same quantity.
pub fn convert(value: f64, from: &str, to: &str) -> Result<f64> {
    Ok(parse_unit(from)?
        .converter_to(parse_unit(to)?)?
        .apply(value))
}

/// Convert one value, consulting `equivalencies` when the two units are
/// not directly commensurable.
///
/// The `value` argument is measured in the unit that `from` spells,
/// and the result is measured in the unit that `to` spells. The
/// `equivalencies` argument lists the physical relations to try.
///
/// A direct conversion always wins. The equivalencies are tried in
/// order only when it fails. This is how a wavelength reaches a
/// frequency. See [`Equivalence`].
///
/// # Errors
///
/// [`FitsError::Header`] if the strings fail to parse, and if no
/// equivalency bridges them.
pub fn convert_with(
    value: f64,
    from: &str,
    to: &str,
    equivalencies: &[&dyn Equivalence],
) -> Result<f64> {
    let (src, dst) = (parse_unit(from)?, parse_unit(to)?);
    if let Ok(c) = src.converter_to(dst) {
        return Ok(c.apply(value));
    }
    for eq in equivalencies {
        if let Some(v) = eq.convert(value, src, dst) {
            return Ok(v);
        }
    }
    Err(FitsError::Header(format!(
        "no conversion from `{from}` to `{to}`, with or without the \
         {} equivalencies supplied",
        equivalencies.len()
    )))
}

/// A conversion that unit algebra cannot reach on its own.
///
/// Wavelength to frequency needs `c`. Brightness temperature needs a
/// frequency and a solid angle. Both are physics, and both carry
/// parameters, such as a rest frequency or a beam size.
///
/// Each such conversion is therefore an implementation of this trait,
/// living in the module that owns that physics. None is an entry in a
/// unit table. This module never learns a speed of light.
///
/// See [`super::Spectral`].
pub trait Equivalence {
    /// Convert `value` from `from` to `to`, or `None` if this
    /// equivalency does not relate the two.
    fn convert(&self, value: f64, from: Unit, to: Unit) -> Option<f64>;
}
