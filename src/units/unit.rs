//! [`Unit`], the parsed result, and [`Converter`], the affine map
//! between two of them.

use super::dimension::{Dimension, describe};
use super::level::{Level, Object, Reference};
use crate::error::{FitsError, Result};

// -- the unit -----------------------------------------------------------

/// A parsed unit: a scale factor onto the canonical units, the dimensions
/// it carries, and its level if it has one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unit {
    /// Multiplier converting a value in this unit to the canonical
    /// units (meter, kilogram, second, degree, ...).
    ///
    /// For a level this is the linear part it is measured against: the
    /// `arcsec^-2` of `mag/arcsec2`.
    pub scale: f64,
    /// The physical quantity this unit measures.
    pub dimension: Dimension,
    /// Present when the value is a level rather than the quantity itself.
    pub level: Option<Level>,
}

/// The affine map between two units: `y = scale * x + offset`.
///
/// A linear unit always gives `offset == 0`. The offset exists for
/// levels.
///
/// This family stops at affine. A conversion that needs more than a
/// scale and a shift is physics, and belongs in an
/// [`super::Equivalence`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Converter {
    /// The multiplicative part.
    pub scale: f64,
    /// The additive part, nonzero only between levels.
    pub offset: f64,
}

impl Converter {
    /// Convert one value.
    #[must_use]
    pub fn apply(self, x: f64) -> f64 {
        self.scale * x + self.offset
    }

    /// The plain factor, when there is one. `None` if a shift is needed,
    /// which is the case callers that can only multiply have to refuse.
    #[must_use]
    pub fn as_factor(self) -> Option<f64> {
        (self.offset.abs() < 1e-12).then_some(self.scale)
    }
}

impl Unit {
    /// A linear unit: `scale` of the canonical unit for `dimension`.
    ///
    /// `Unit::new(1.0, dimensions::LENGTH)` is the meter;
    /// `Unit::new(1e3, dimensions::LENGTH)` is the kilometer.
    #[must_use]
    pub const fn new(scale: f64, dimension: Dimension) -> Self {
        Self {
            scale,
            dimension,
            level: None,
        }
    }

    /// The dimensionless unit -- a pure number.
    #[must_use]
    pub const fn scalar() -> Self {
        Self::new(1.0, Dimension::NONE)
    }

    /// A level over a dimensionless linear part: `mag`, `dB`, `Sun`.
    pub(super) const fn level(level: Level) -> Self {
        Self {
            scale: 1.0,
            dimension: Dimension::NONE,
            level: Some(level),
        }
    }

    /// True if this is a level, and so converts by a shift rather than
    /// only by a factor.
    #[must_use]
    pub const fn is_level(&self) -> bool {
        self.level.is_some()
    }

    /// Multiplication.
    ///
    /// At most one level survives a product: `mag/arcsec2` is a quantity
    /// -- a magnitude per unit area -- but `mag2` and `mag dB` are not.
    ///
    /// A pure number times a *logarithmic* level scales the level's
    /// value: `10**-3 mag` (like `mmag`) is a thousandth of a magnitude,
    /// not the magnitude of a thousandth of the flux. The number folds
    /// into [`Level::factor`] -- a value `v` in `k mag` stands for
    /// `10^(v k / -2.5)`, a level with factor `-2.5 / k`, so 5 of them
    /// convert to 0.005 `mag`. A
    /// *dimensioned* linear part multiplies `scale` instead, staying
    /// underneath the logarithm as the reference: that is what keeps
    /// `mag/arcsec2` to `mag/deg2` an additive -17.78.
    ///
    /// # Errors
    ///
    /// [`FitsError::Header`] if both sides carry a level.
    pub(super) fn mul(self, other: Self) -> Result<Self> {
        if self.level.is_some() && other.level.is_some() {
            return Err(FitsError::Header(
                "a product of two level units (`mag`, `dB`, `Sun`) is not a quantity".to_string(),
            ));
        }
        let (leveled, plain) = if self.level.is_some() {
            (self, other)
        } else {
            (other, self)
        };
        if let Some(l) = leveled.level
            && l.log_base.is_some()
            && plain.dimension.is_dimensionless()
        {
            return Ok(Self {
                level: Some(Level {
                    factor: l.factor / plain.scale,
                    ..l
                }),
                ..leveled
            });
        }
        Ok(Self {
            scale: self.scale * other.scale,
            dimension: self.dimension.mul(other.dimension),
            level: self.level.or(other.level),
        })
    }

    pub(super) fn div(self, other: Self) -> Result<Self> {
        self.mul(other.powf(-1.0)?)
    }

    /// # Errors
    ///
    /// [`FitsError::Header`] if this is a *logarithmic* level: a magnitude
    /// squared is not a quantity, and `sqrt(mag)` is not one either.
    ///
    /// A plain ratio is exempt. `Sun` is only a marker saying what the
    /// ratio is against, so it inverts with everything else and
    /// `solMass/Sun` -- a mass in solar units -- stays legal.
    pub(super) fn powf(self, e: f64) -> Result<Self> {
        if self.level.is_some_and(|l| l.log_base.is_some()) && (e - 1.0).abs() > 1e-9 {
            return Err(FitsError::Header(
                "a logarithmic unit (`mag`, `dB`) cannot be raised to a power".to_string(),
            ));
        }
        Ok(Self {
            scale: self.scale.powf(e),
            dimension: self.dimension.powf(e),
            level: self.level,
        })
    }

    /// Combine two units multiplicatively: `m` times `s^-1` is `m s^-1`.
    ///
    /// Fallible rather than a [`std::ops::Mul`] impl because levels make
    /// it so, and an operator that panics is worse than a `?`.
    ///
    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if both sides carry a level.
    pub fn try_mul(self, other: Self) -> Result<Self> {
        self.mul(other)
    }

    /// Divide two units: `m` by `s` is `m s^-1`.
    ///
    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if both sides carry a level, or
    /// if the divisor is logarithmic.
    pub fn try_div(self, other: Self) -> Result<Self> {
        self.div(other)
    }

    /// Raise a unit to a power. Fractional powers are legal -- `m**(3/2)`
    /// is a unit -- so this takes an `f64`.
    ///
    /// # Errors
    ///
    /// [`crate::error::FitsError::Header`] if this is a logarithmic level.
    pub fn try_pow(self, e: f64) -> Result<Self> {
        self.powf(e)
    }

    /// The map taking a value in `self` to a value in `target`.
    ///
    /// For two logarithmic levels, a value `v` stands for
    /// `a.base^(v/a.factor) * self.scale`. Equating that with the
    /// reading of the target gives an affine map. The offset of that
    /// map carries the linear rescale through the logarithm.
    ///
    /// `mag/arcsec2` to `mag/deg2` therefore comes out as `x - 17.78`,
    /// and `log(kHz)` to `log(Hz)` as `x + 3`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Header`] when the two are not the same quantity,
    /// in three cases:
    ///
    /// - They carry different dimensions.
    /// - One is a level and the other is linear.
    /// - Both are levels, against different references.
    pub fn converter_to(self, target: Self) -> Result<Converter> {
        if !self.dimension.matches(target.dimension) {
            return Err(FitsError::Header(format!(
                "unit is {}, but {} was required",
                describe(self.dimension),
                describe(target.dimension),
            )));
        }
        let linear = Converter {
            scale: self.scale / target.scale,
            offset: 0.0,
        };
        match (self.level, target.level) {
            (None, None) => Ok(linear),
            (Some(a), Some(b)) if a.reference != b.reference => Err(FitsError::Header(format!(
                "levels are measured against different references ({:?} and {:?})",
                a.reference, b.reference
            ))),
            (Some(a), Some(b)) => match (a.log_base, b.log_base) {
                (Some(ba), Some(bb)) => Ok(Converter {
                    scale: (b.factor * ba.ln()) / (a.factor * bb.ln()),
                    offset: b.factor * (self.scale / target.scale).ln() / bb.ln(),
                }),
                (None, None) => Ok(linear),
                _ => Err(FitsError::Header(
                    "no conversion relates a logarithmic level to a plain ratio".to_string(),
                )),
            },
            _ => Err(FitsError::Header(
                "no conversion relates a level unit to a linear one; read the \
                 reference off the unit and de-log it yourself"
                    .to_string(),
            )),
        }
    }
}

/// The scale as a Sec.4.3.1 numeric multiplier, or `None` when it is 1.
///
/// `{:e}` already splits a float into the mantissa and power of ten that
/// the section spells `10**k`; juxtaposition is multiplication, so the two
/// pieces need only a space between them. A plain `1.5e-3` would not
/// re-parse -- the grammar reads `e` as the start of a symbol.
fn scale_token(scale: f64) -> Option<String> {
    if (scale - 1.0).abs() < 1e-12 {
        return None;
    }
    let sci = format!("{scale:e}");
    let (mantissa, exp) = sci.split_once('e')?;
    Some(match (mantissa, exp) {
        ("1", "0") => return None,
        ("1", e) => format!("10**{e}"),
        (m, "0") => m.to_string(),
        (m, e) => format!("{m} 10**{e}"),
    })
}

/// How a level is spelled: `deg^-2 mag` juxtaposes, `log(Hz)` wraps.
enum Spelling {
    Prefix(String),
    Function(&'static str),
}

/// The family symbol, and the outer multiplier `k` making this level `k`
/// of that family's canonical one: `mmag` comes out as (`mag`, 1e-3).
fn spelling(l: Level) -> (Spelling, f64) {
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12 * a.abs().max(b.abs());
    match l.reference {
        Reference::Object(Object::Solar) => (Spelling::Prefix("Sun".to_string()), 1.0),
        Reference::Object(Object::Vega) => {
            (Spelling::Prefix("mag(Vega)".to_string()), -2.5 / l.factor)
        }
        // By name, so a hand-built zero point is not passed off as `ST`;
        // only `AB`, `ST` and `Vega` re-parse, but a wrong label is the
        // worse failure.
        Reference::Absolute(z) => (
            Spelling::Prefix(format!("mag({})", z.name)),
            -2.5 / l.factor,
        ),
        Reference::Unspecified => {
            if matches!(l.log_base, Some(b) if b == std::f64::consts::E) {
                if close(l.factor, 1.0) {
                    // `Np` and `ln()` are the same level; the function
                    // form is the one that always re-parses.
                    (Spelling::Function("ln"), 1.0)
                } else {
                    (Spelling::Prefix("Np".to_string()), 1.0 / l.factor)
                }
            } else if close(l.factor, -2.5) {
                (Spelling::Prefix("mag".to_string()), 1.0)
            } else if close(l.factor, 10.0) {
                (Spelling::Prefix("dB".to_string()), 1.0)
            } else if close(l.factor, 1.0) {
                (Spelling::Function("log"), 1.0)
            } else if l.factor < 0.0 {
                (Spelling::Prefix("mag".to_string()), -2.5 / l.factor)
            } else {
                (Spelling::Prefix("dB".to_string()), 10.0 / l.factor)
            }
        }
    }
}

/// The Table 7 prefix spelling of `k`, for fusing `mag 10**-3` into
/// `mmag`.
fn prefix_token(k: f64) -> Option<&'static str> {
    const TABLE: [(f64, &str); 20] = [
        (1e-24, "y"),
        (1e-21, "z"),
        (1e-18, "a"),
        (1e-15, "f"),
        (1e-12, "p"),
        (1e-9, "n"),
        (1e-6, "u"),
        (1e-3, "m"),
        (1e-2, "c"),
        (1e-1, "d"),
        (1e1, "da"),
        (1e2, "h"),
        (1e3, "k"),
        (1e6, "M"),
        (1e9, "G"),
        (1e12, "T"),
        (1e15, "P"),
        (1e18, "E"),
        (1e21, "Z"),
        (1e24, "Y"),
    ];
    TABLE
        .iter()
        .find(|(v, _)| (k / v - 1.0).abs() < 1e-12)
        .map(|(_, p)| *p)
}

/// The canonical form, e.g. `10**-26 kg s^-2` for `Jy`.
///
/// This is the *decomposition*, not the string it was parsed from: `Jy`
/// and `1e-26 kg s-2` are the same quantity and render alike. What it
/// guarantees instead is that the output re-parses through
/// [`super::parse_unit`] to an equal value, so it is safe to write into a card.
///
/// A level unit renders in the spelling of this module, such as
/// `deg^-2 mag(AB)`. Other spellings of the same quantity exist, and
/// no single spelling round-trips through every reader.
/// The linear part comes before the symbol, so its numeric factor stays
/// glued to the dimension it scales; an outer multiplier trails, where a
/// re-parse folds it back into the level's value -- except `k mag`, which
/// fuses into the prefixed symbol (`mmag`) whenever `k` is a Table 7
/// prefix.
impl std::fmt::Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dim = self.dimension.to_string();
        let body = match (scale_token(self.scale), dim.is_empty()) {
            (None, _) => dim,
            (Some(s), true) => s,
            (Some(s), false) => format!("{s} {dim}"),
        };
        let Some(level) = self.level else {
            return f.write_str(&body);
        };
        let (sp, k) = spelling(level);
        let mut outer = scale_token(k);
        match sp {
            Spelling::Prefix(mut name) => {
                // `mmag(AB)` would re-parse as an exponent, so only the
                // bare symbol takes a fused prefix.
                if name == "mag"
                    && outer.is_some()
                    && let Some(p) = prefix_token(k)
                {
                    name = format!("{p}mag");
                    outer = None;
                }
                if !body.is_empty() {
                    f.write_str(&body)?;
                    f.write_str(" ")?;
                }
                f.write_str(&name)?;
            }
            // A bare `log()` has no argument to wrap, so give it the 1 it
            // is a logarithm of.
            Spelling::Function(name) if body.is_empty() => write!(f, "{name}(1)")?,
            Spelling::Function(name) => write!(f, "{name}({body})")?,
        }
        match outer {
            Some(o) => write!(f, " {o}"),
            None => Ok(()),
        }
    }
}
