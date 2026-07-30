//! Conversions that unit algebra cannot reach on its own.
//!
//! Only the parameterless, universal ones live here. A relation carrying
//! parameters or a convention -- the `VRAD`/`VOPT`/`VELO` doppler
//! definitions of Standard Sec.8.4, air-to-vacuum with its choice of
//! dispersion formula -- belongs with the standard that specifies it, in
//! [`crate::wcs::spectral`].

use super::constants::raw::{PLANCK, SPEED_OF_LIGHT};
use super::dimension::dimensions as d;
use super::unit::Unit;

/// The spectral equivalency: wavelength, frequency, energy and wavenumber
/// all describe the same photon, but no unit algebra relates them --
/// crossing between them needs `c` and `h`.
///
/// It sits in `units` rather than in `wcs` because it is parameterless
/// and universal, and because the table here already depends on `c`: a
/// light year *is* the speed of light times a Julian year. `astropy`
/// draws the line in the same place, with `u.spectral()` in its core
/// units package rather than in `astropy.wcs`.
///
/// Vacuum wavelength only. `AWAV` is air wavelength and needs a
/// refractive index, itself a function of the wavelength; that stays in
/// `air_to_vacuum` in [`crate::wcs::spectral`], where the choice of
/// dispersion formula belongs.
///
/// ```
/// use fitsy::units::{Spectral, convert_with};
/// // 500 nm as a frequency.
/// let hz = convert_with(500e-9, "m", "Hz", &[&Spectral]).unwrap();
/// assert!((hz - 5.99584916e14).abs() / hz < 1e-9);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Spectral;

impl super::convert::Equivalence for Spectral {
    fn convert(&self, value: f64, from: Unit, to: Unit) -> Option<f64> {
        // Everything routes through frequency in SI.
        let to_hz = |v: f64, dim| {
            if dim == d::FREQUENCY {
                Some(v)
            } else if dim == d::LENGTH {
                Some(SPEED_OF_LIGHT / v)
            } else if dim == d::WAVENUMBER {
                Some(v * SPEED_OF_LIGHT)
            } else if dim == d::ENERGY {
                Some(v / PLANCK)
            } else {
                None
            }
        };
        let from_hz = |f: f64, dim| {
            if dim == d::FREQUENCY {
                Some(f)
            } else if dim == d::LENGTH {
                Some(SPEED_OF_LIGHT / f)
            } else if dim == d::WAVENUMBER {
                Some(f / SPEED_OF_LIGHT)
            } else if dim == d::ENERGY {
                Some(f * PLANCK)
            } else {
                None
            }
        };
        // A level has no place here: the logarithm of a wavelength is not
        // a wavelength.
        if from.is_level() || to.is_level() {
            return None;
        }
        let hz = to_hz(value * from.scale, from.dimension)?;
        Some(from_hz(hz, to.dimension)? / to.scale)
    }
}

#[cfg(test)]
mod tests {
    use super::Spectral;
    use crate::units::{Equivalence, convert, convert_with};

    /// Wavelength, frequency, energy and wavenumber describe the same
    /// photon but share no dimension, so plain unit algebra refuses them
    /// and the equivalency is what bridges the gap.
    ///
    /// The reference values are `astropy`'s, from
    /// `(500*u.nm).to(u.Hz, equivalencies=u.spectral())`.
    #[test]
    fn spectral_equivalency_bridges_wavelength_and_frequency() {
        let eq: &[&dyn Equivalence] = &[&Spectral];
        // Without it, these are different quantities and stay that way.
        assert!(convert(500.0, "nm", "Hz").is_err());
        assert!(
            convert(500.0, "nm", "m").is_ok(),
            "same dimension is direct"
        );

        let hz = convert_with(500.0, "nm", "Hz", eq).unwrap();
        assert!((hz - 5.995_849_16e14).abs() / hz < 1e-8, "got {hz}");
        let ev = convert_with(500.0, "nm", "eV", eq).unwrap();
        assert!((ev - 2.479_683_968_664_005).abs() < 1e-9, "got {ev}");
        let wavn = convert_with(500.0, "nm", "m-1", eq).unwrap();
        assert!((wavn - 2e6).abs() / wavn < 1e-12, "got {wavn}");

        // Round trip.
        let back = convert_with(hz, "Hz", "nm", eq).unwrap();
        assert!((back - 500.0).abs() < 1e-9, "got {back}");

        // A direct conversion still wins over the equivalency.
        let km = convert_with(1.0, "km", "m", eq).unwrap();
        assert!((km - 1000.0).abs() < 1e-9);

        // The logarithm of a wavelength is not a wavelength.
        assert!(convert_with(1.0, "log(m)", "Hz", eq).is_err());
        // And an unrelated pair is still unrelated.
        assert!(convert_with(1.0, "kg", "Hz", eq).is_err());
    }
}
