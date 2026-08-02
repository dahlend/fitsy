//! Physical and astronomical constants, as [`Quantity`] values.
//!
//! Each carries its dimension, so a downstream calculation is checked
//! rather than trusted: `PLANCK.try_mul(frequency)` is an energy and the
//! compiler-adjacent machinery says so, where `PLANCK * f` on bare
//! floats would say nothing at all.
//!
//! # Provenance
//!
//! Fundamental constants are CODATA 2018. Five of them -- `c`, `h`, `k`,
//! `e` and `N_A` -- are *exact* by the 2019 SI redefinition, and are
//! marked as such below; the rest carry measurement uncertainty that
//! this module does not track.
//!
//! The solar, terrestrial and jovian values are the nominal values of
//! IAU 2015 Resolution B3. Those are conventional constants, fixed so
//! that a published result does not shift when the measurement
//! improves. They are not best estimates. Do not use them as such.
//! The Resolution defines `GM` for each body rather than a mass, because
//! `GM` is measured far better than `G` -- masses here are `GM/G` and
//! inherit `G`'s uncertainty, which is the worst of any constant here at
//! about 2.2e-5 relative.
//!
//! # Relationship to the unit table
//!
//! Standard Tables 4-5 tabulate some of the same values as *units* --
//! `solMass`, `AU`, `pc`, `eV`. Those stay written out in
//! the parser's own base table rather than being derived from here, so it
//! reproduces the Standard's numbers exactly.
//! `constants_agree_with_the_unit_table` in the module tests pins the two
//! together so they cannot drift.

use super::dimension::{
    ANGLE, CHARGE, Dimension, ENERGY, LENGTH, MASS, POWER, SUBSTANCE, TEMPERATURE, TIME,
};
use super::quantity::Quantity;

/// A constant in the canonical unit for its dimension.
const fn c(value: f64, dimension: Dimension) -> Quantity {
    Quantity::canonical(value, dimension)
}

// -- dimensions used only here ------------------------------------------

/// `m s^-1`.
const VELOCITY: Dimension = LENGTH.div(TIME);
/// `J s`, the dimension of action.
const ACTION: Dimension = ENERGY.mul(TIME);
/// `J K^-1`.
const HEAT_CAPACITY: Dimension = ENERGY.div(TEMPERATURE);
/// `mol^-1`.
const PER_SUBSTANCE: Dimension = SUBSTANCE.powf(-1.0);
/// `m^3 kg^-1 s^-2`, the dimension of `G`.
const GRAVITATION: Dimension = LENGTH.powf(3.0).div(MASS).div(TIME.powf(2.0));
/// `m^3 s^-2`, the dimension of a standard gravitational parameter `GM`.
const GRAVITATIONAL_PARAMETER: Dimension = LENGTH.powf(3.0).div(TIME.powf(2.0));
/// `W m^-2 K^-4`.
const STEFAN: Dimension = POWER.div(LENGTH.powf(2.0)).div(TEMPERATURE.powf(4.0));
/// `J mol^-1 K^-1`.
const MOLAR_HEAT_CAPACITY: Dimension = ENERGY.div(SUBSTANCE).div(TEMPERATURE);
/// `m K`, the dimension of Wien's displacement constant.
const LENGTH_TEMPERATURE: Dimension = LENGTH.mul(TEMPERATURE);
/// `m^-1`.
const WAVENUMBER: Dimension = LENGTH.powf(-1.0);

// -- exact by the 2019 SI definitions -----------------------------------

/// Speed of light in vacuum, `c`. This value is exact; it defines the metre.
pub const SPEED_OF_LIGHT: Quantity = c(299_792_458.0, VELOCITY);

/// Planck constant, `h`. This value is exact; it defines the kilogram.
pub const PLANCK: Quantity = c(6.626_070_15e-34, ACTION);

/// Reduced Planck constant, `hbar = h / 2 pi`.
pub const REDUCED_PLANCK: Quantity = c(6.626_070_15e-34 / (2.0 * std::f64::consts::PI), ACTION);

/// Boltzmann constant, `k`. This value is exact; it defines the kelvin.
pub const BOLTZMANN: Quantity = c(1.380_649e-23, HEAT_CAPACITY);

/// Elementary charge, `e`. This value is exact; it defines the ampere.
pub const ELEMENTARY_CHARGE: Quantity = c(1.602_176_634e-19, CHARGE);

/// Avogadro constant, `N_A`. This value is exact; it defines the mole.
pub const AVOGADRO: Quantity = c(6.022_140_76e23, PER_SUBSTANCE);

// -- CODATA 2018, measured ----------------------------------------------

/// Newtonian constant of gravitation, `G`. The least precisely known
/// constant here, at 2.2e-5 relative.
pub const GRAVITATIONAL_CONSTANT: Quantity = c(6.674_30e-11, GRAVITATION);

/// Stefan-Boltzmann constant, `sigma`.
pub const STEFAN_BOLTZMANN: Quantity = c(5.670_374_419e-8, STEFAN);

/// Molar gas constant, `R = N_A k`.
pub const GAS_CONSTANT: Quantity = c(8.314_462_618, MOLAR_HEAT_CAPACITY);

/// Wien displacement law constant, `b`, for the wavelength peak:
/// `lambda_max = b / T`.
pub const WIEN_DISPLACEMENT: Quantity = c(2.897_771_955e-3, LENGTH_TEMPERATURE);

/// Rydberg constant, `R_inf`.
pub const RYDBERG: Quantity = c(10_973_731.568_160, WAVENUMBER);

/// Fine-structure constant, `alpha`. Dimensionless.
pub const FINE_STRUCTURE: Quantity = Quantity::scalar(7.297_352_569_3e-3);

/// Electron rest mass, `m_e`.
pub const ELECTRON_MASS: Quantity = c(9.109_383_701_5e-31, MASS);

/// Proton rest mass, `m_p`.
pub const PROTON_MASS: Quantity = c(1.672_621_923_69e-27, MASS);

/// Neutron rest mass, `m_n`.
pub const NEUTRON_MASS: Quantity = c(1.674_927_498_04e-27, MASS);

/// Unified atomic mass unit, `u` -- one twelfth of a carbon-12 atom.
/// Tabulated as a unit too, as `u` in Table 5.
pub const ATOMIC_MASS: Quantity = c(1.660_539_066_60e-27, MASS);

// -- time ---------------------------------------------------------------

/// The day, s. Exactly 86400 SI seconds, which is *not* a rotation of the
/// Earth; see `UT1` for the one that is.
pub const DAY: Quantity = c(86_400.0, TIME);

/// The Julian year, s -- 365.25 days, the IAU definition behind both the
/// `a` unit and `lyr`.
pub const JULIAN_YEAR: Quantity = c(31_557_600.0, TIME);

/// The Julian century, s -- 36525 days. The time unit of most FITS
/// astrometric polynomials.
pub const JULIAN_CENTURY: Quantity = c(3_155_760_000.0, TIME);

// -- length -------------------------------------------------------------

/// Astronomical unit, `au`. This value is exact, by IAU 2012 Resolution B2.
pub const ASTRONOMICAL_UNIT: Quantity = c(1.495_978_707e11, LENGTH);

/// Parsec -- the distance subtending one arcsecond of parallax across one
/// astronomical unit, `au / tan(1")`.
pub const PARSEC: Quantity = c(3.085_677_581_491_367e16, LENGTH);

/// Light year -- `c` times a Julian year. Exactly 9.4607304725808e15 m.
pub const LIGHT_YEAR: Quantity = c(299_792_458.0 * 31_557_600.0, LENGTH);

// -- IAU 2015 Resolution B3 nominal values ------------------------------

/// Nominal solar radius, `R_sun`.
pub const SOLAR_RADIUS: Quantity = c(6.957e8, LENGTH);

/// Nominal solar luminosity, `L_sun`.
pub const SOLAR_LUMINOSITY: Quantity = c(3.828e26, POWER);

/// Nominal solar effective temperature, `T_eff,sun`.
pub const SOLAR_EFFECTIVE_TEMPERATURE: Quantity = c(5772.0, TEMPERATURE);

/// Nominal solar mass parameter, `GM_sun`. Measured far better than
/// [`SOLAR_MASS`], which is this divided by `G`.
pub const SOLAR_MASS_PARAMETER: Quantity = c(1.327_124_4e20, GRAVITATIONAL_PARAMETER);

/// Solar mass, `GM_sun / G`. Inherits `G`'s 2.2e-5 uncertainty; prefer
/// [`SOLAR_MASS_PARAMETER`] where the dynamics allow it.
pub const SOLAR_MASS: Quantity = c(1.327_124_4e20 / 6.674_30e-11, MASS);

/// Nominal equatorial radius of the Earth, `R_earth`.
pub const EARTH_RADIUS: Quantity = c(6.3781e6, LENGTH);

/// Nominal terrestrial mass parameter, `GM_earth`.
pub const EARTH_MASS_PARAMETER: Quantity = c(3.986_004e14, GRAVITATIONAL_PARAMETER);

/// Earth mass, `GM_earth / G`.
pub const EARTH_MASS: Quantity = c(3.986_004e14 / 6.674_30e-11, MASS);

/// Nominal equatorial radius of Jupiter, `R_jup`.
pub const JUPITER_RADIUS: Quantity = c(7.1492e7, LENGTH);

/// Nominal jovian mass parameter, `GM_jup`.
pub const JUPITER_MASS_PARAMETER: Quantity = c(1.266_865_3e17, GRAVITATIONAL_PARAMETER);

/// Jupiter mass, `GM_jup / G`.
pub const JUPITER_MASS: Quantity = c(1.266_865_3e17 / 6.674_30e-11, MASS);

// -- conversions the module itself leans on -----------------------------

/// Degrees per radian. The canonical angle here is the degree, so this
/// turns up wherever a formula is written in radians.
pub const DEGREES_PER_RADIAN: Quantity = c(180.0 / std::f64::consts::PI, ANGLE);

/// Raw `f64` forms, for the interior of numeric transforms where wrapping
/// every operand in a [`Quantity`] would cost more than it checks.
///
/// [`crate::wcs::spectral`] is the caller that matters. Its analytic
/// derivatives run once per coordinate. Their intermediate quantities
/// have no standard names to check against; `d(wavelength)/d(frequency)`
/// carries the dimension `m s`.
pub mod raw {
    /// [`super::SPEED_OF_LIGHT`] in m/s.
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
    /// [`super::PLANCK`] in J s.
    pub const PLANCK: f64 = 6.626_070_15e-34;
    /// [`super::JULIAN_YEAR`] in s.
    pub const JULIAN_YEAR: f64 = 31_557_600.0;
}
