//! Dimensional exponents: the vector every unit reduces to.

const N_DIM: usize = 15;

/// Indices into [`Dimension`].
pub(super) mod dim {
    #![allow(unreachable_pub, reason = "index constants are crate-internal")]
    pub const LENGTH: usize = 0;
    pub const MASS: usize = 1;
    pub const TIME: usize = 2;
    pub const ANGLE: usize = 3;
    pub const TEMPERATURE: usize = 4;
    pub const CURRENT: usize = 5;
    pub const SUBSTANCE: usize = 6;
    pub const LUMINOUS: usize = 7;
    // Not SI dimensions, but tracked so they cannot silently cancel
    // against one: `ct/s` must not compare equal to `Hz`, and
    // `Jy/beam` must not reduce to `Jy`.
    //
    // The set matches `wcslib`'s (`wcsunits.h`, `WCSUNITS_NTYPE`), which
    // separates several things that look alike but do not convert:
    //
    // * `BIT` from `COUNT`, so a data volume is not a photon count;
    // * `ADU` from both, because the ADU-to-electron factor is the
    //   detector gain -- per instrument, often per exposure -- so no
    //   universal conversion exists to offer;
    // * `VOXEL` from `PIXEL`.
    //
    // `wcslib` also gives `MAGNITUDE` and `SOLRATIO` slots here. Both are
    // levels rather than dimensions -- a magnitude is a logarithm and
    // `Sun` is a ratio against a reference -- so they live in [`Level`],
    // which is what lets them convert by an offset.
    pub const COUNT: usize = 8;
    pub const BIT: usize = 9;
    pub const ADU: usize = 10;
    pub const PIXEL: usize = 11;
    pub const VOXEL: usize = 12;
    pub const BEAM: usize = 13;
    pub const BIN: usize = 14;
}

/// Exponents of the base quantities. Fractional powers are legal
/// (`m**(3/2)`), so these are not integers.
///
/// [`PartialEq`] compares with a tolerance rather than exactly.
/// `m**(1/3)` cubed does not land on exactly 1.0, so an exact test
/// would answer no to a question whose answer is yes. A tolerant
/// comparison is not transitive, so this type implements no [`Eq`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Dimension([f64; N_DIM]);

impl PartialEq for Dimension {
    fn eq(&self, other: &Self) -> bool {
        self.matches(*other)
    }
}

impl Dimension {
    /// The dimensionless quantity.
    pub const NONE: Self = Self([0.0; N_DIM]);

    // `const` with `while` rather than iterators so the whole base table
    // and the [`dimensions`] set are built at compile time.
    pub(super) const fn base(index: usize) -> Self {
        let mut d = [0.0; N_DIM];
        d[index] = 1.0;
        Self(d)
    }

    pub(super) const fn mul(self, other: Self) -> Self {
        let mut out = self.0;
        let mut i = 0;
        while i < N_DIM {
            out[i] += other.0[i];
            i += 1;
        }
        Self(out)
    }

    pub(super) const fn powf(self, e: f64) -> Self {
        let mut out = self.0;
        let mut i = 0;
        while i < N_DIM {
            out[i] *= e;
            i += 1;
        }
        Self(out)
    }

    pub(super) const fn div(self, other: Self) -> Self {
        self.mul(other.powf(-1.0))
    }

    /// True if every exponent matches. Compared with a tolerance
    /// because `m**(1/3)` cubed will not land on exactly 1.0.
    #[must_use]
    pub fn matches(self, other: Self) -> bool {
        self.0
            .iter()
            .zip(other.0)
            .all(|(a, b)| (a - b).abs() < 1e-9)
    }

    /// True for a pure number -- every exponent zero.
    #[must_use]
    pub fn is_dimensionless(self) -> bool {
        self.matches(Self::NONE)
    }
}

/// Canonical symbol per dimension, index-parallel with [`dim`]. Every one
/// is a Table 4-5 spelling, so [`Dimension`]'s rendering re-parses.
const DIM_NAMES: [&str; N_DIM] = [
    "m", "kg", "s", "deg", "K", "A", "mol", "cd", "count", "bit", "adu", "pixel", "voxel", "beam",
    "bin",
];

/// The canonical decomposition, e.g. `m s^-1`.
///
/// Blank rather than `1` for the dimensionless case, since that is how a
/// dimensionless `CUNIT` is written; [`super::Unit`]'s rendering supplies the
/// scale in front.
impl std::fmt::Display for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut wrote = false;
        for (name, e) in DIM_NAMES.iter().zip(self.0) {
            if e.abs() < 1e-9 {
                continue;
            }
            if wrote {
                f.write_str(" ")?;
            }
            wrote = true;
            f.write_str(name)?;
            if (e - 1.0).abs() < 1e-9 {
                continue;
            }
            // Sec.4.3 permits a bare integer exponent but requires a
            // fractional one to be parenthesised, so `m1.5` is not a
            // spelling of three-halves and cannot be emitted as one.
            if (e - e.round()).abs() < 1e-9 {
                write!(f, "^{}", e.round())?;
            } else {
                write!(f, "^({e})")?;
            }
        }
        Ok(())
    }
}

// -- compound dimensions ------------------------------------------------
//
// Built at compile time rather than on entry to every table lookup.

pub(super) const LENGTH: Dimension = Dimension::base(dim::LENGTH);
pub(super) const MASS: Dimension = Dimension::base(dim::MASS);
pub(super) const TIME: Dimension = Dimension::base(dim::TIME);
pub(super) const ANGLE: Dimension = Dimension::base(dim::ANGLE);
pub(super) const TEMPERATURE: Dimension = Dimension::base(dim::TEMPERATURE);
pub(super) const CURRENT: Dimension = Dimension::base(dim::CURRENT);
pub(super) const SUBSTANCE: Dimension = Dimension::base(dim::SUBSTANCE);
pub(super) const LUMINOUS: Dimension = Dimension::base(dim::LUMINOUS);
pub(super) const ENERGY: Dimension = MASS.mul(LENGTH.powf(2.0)).mul(TIME.powf(-2.0));
pub(super) const FORCE: Dimension = MASS.mul(LENGTH).mul(TIME.powf(-2.0));
pub(super) const POWER: Dimension = ENERGY.mul(TIME.powf(-1.0));
pub(super) const CHARGE: Dimension = Dimension::base(dim::CURRENT).mul(TIME);
pub(super) const VOLTAGE: Dimension = ENERGY.div(CHARGE);
pub(super) const MAGNETIC_FLUX: Dimension = VOLTAGE.mul(TIME);
/// Spectral flux density per unit frequency -- the dimension of `Jy`.
pub(super) const F_NU: Dimension = POWER.mul(LENGTH.powf(-2.0)).mul(TIME);
/// Spectral flux density per unit wavelength.
pub(super) const F_LAMBDA: Dimension = POWER.mul(LENGTH.powf(-3.0));

pub(super) const DEG_PER_RAD: f64 = 180.0 / std::f64::consts::PI;
/// The canonical solid angle is deg^2, so anything defined through the
/// steradian carries this factor: `lm` has to equal `cd sr`.
pub(super) const SR_SCALE: f64 = DEG_PER_RAD * DEG_PER_RAD;

/// The dimensions callers need to check against.
///
/// Constants rather than functions: [`Dimension`]'s arithmetic is `const`,
/// so these are folded at compile time and usable in patterns and statics.
///
/// Anything not here is a product away: [`super::Unit::try_mul`] and
/// friends compose these into whatever a caller needs.
pub mod dimensions {
    use super::Dimension;

    /// `m`.
    pub const LENGTH: Dimension = super::LENGTH;
    /// `s`.
    pub const TIME: Dimension = super::TIME;
    /// `deg`, the canonical angle.
    pub const ANGLE: Dimension = super::ANGLE;
    /// `kg`.
    pub const MASS: Dimension = super::MASS;
    /// `K`.
    pub const TEMPERATURE: Dimension = super::TEMPERATURE;
    /// `A`.
    pub const CURRENT: Dimension = super::CURRENT;
    /// `mol`.
    pub const SUBSTANCE: Dimension = super::SUBSTANCE;
    /// `cd`.
    pub const LUMINOUS: Dimension = super::LUMINOUS;
    /// `kg m s^-2`, the dimension of `N`.
    pub const FORCE: Dimension = super::FORCE;
    /// `kg m^2 s^-3`, the dimension of `W`.
    pub const POWER: Dimension = super::POWER;
    /// `A s`, the dimension of `C`.
    pub const CHARGE: Dimension = super::CHARGE;
    /// `m^2`.
    pub const AREA: Dimension = super::LENGTH.powf(2.0);
    /// `m^3`.
    pub const VOLUME: Dimension = super::LENGTH.powf(3.0);
    /// `m s^-2`.
    pub const ACCELERATION: Dimension = super::LENGTH.mul(super::TIME.powf(-2.0));
    /// `s^-1`, the dimension of `Hz`.
    pub const FREQUENCY: Dimension = super::TIME.powf(-1.0);
    /// `m^-1`, the dimension of a wavenumber.
    pub const WAVENUMBER: Dimension = super::LENGTH.powf(-1.0);
    /// `m s^-1`.
    pub const VELOCITY: Dimension = super::LENGTH.mul(super::TIME.powf(-1.0));
    /// `kg m^2 s^-2`, the dimension of `J`.
    pub const ENERGY: Dimension = super::ENERGY;
    /// `kg s^-2`, the dimension of `Jy` -- flux per unit frequency.
    pub const SPECTRAL_FLUX_DENSITY: Dimension = super::F_NU;
    /// A pure number.
    pub const DIMENSIONLESS: Dimension = Dimension::NONE;
}

/// [`Dimension`] for an error message, naming the dimensionless case that
/// [`Dimension`]'s own rendering leaves blank.
pub(super) fn describe(d: Dimension) -> String {
    if d.is_dimensionless() {
        "dimensionless".to_string()
    } else {
        d.to_string()
    }
}
