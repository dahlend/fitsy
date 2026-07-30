//! Standard Tables 4-5: the base symbols, the Table 7 prefixes, and
//! the rule for which symbols a prefix may attach to.

use super::constants::raw::{JULIAN_YEAR, SPEED_OF_LIGHT};
use super::dimension::{
    ANGLE, CHARGE, DEG_PER_RAD, Dimension, ENERGY, FORCE, LENGTH, MAGNETIC_FLUX, MASS, POWER,
    SR_SCALE, TIME, VOLTAGE, dim,
};
use super::level::Level;
use super::unit::Unit;
use crate::error::{FitsError, Result};

// -- base unit table ----------------------------------------------------

/// `(symbol, scale onto canonical units, dimension index or compound)`.
///
/// Ordered longest-symbol-first is not required: lookup is exact-match
/// first, then a single prefix character, so `Pa` resolves as pascal
/// rather than peta-year exactly as Standard Table 5 demands.
#[allow(
    clippy::match_same_arms,
    reason = "the arms mirror Standard Tables 4-5 one entry per row; merging those that happen to share a dimension would hide the correspondence"
)]
fn lookup_base(sym: &str) -> Option<Unit> {
    use dim::{
        ADU, BEAM, BIN, BIT, COUNT, CURRENT, LUMINOUS, PIXEL, SUBSTANCE, TEMPERATURE, VOXEL,
    };
    let d = Dimension::base;
    // Aliases onto the module constants, which are folded at compile time
    // rather than rebuilt on entry to every lookup.
    let (length, mass, time, angle) = (LENGTH, MASS, TIME, ANGLE);
    let (energy, force, power, charge) = (ENERGY, FORCE, POWER, CHARGE);
    let (voltage, magnetic_flux, sr_scale) = (VOLTAGE, MAGNETIC_FLUX, SR_SCALE);

    Some(match sym {
        // -- SI base and supplementary (Table 4) --
        "m" => Unit::new(1.0, length),
        "kg" => Unit::new(1.0, mass),
        "g" => Unit::new(1e-3, mass),
        "s" => Unit::new(1.0, time),
        "rad" => Unit::new(DEG_PER_RAD, angle),
        "sr" => Unit::new(sr_scale, angle.powf(2.0)),
        "K" => Unit::new(1.0, d(TEMPERATURE)),
        "A" => Unit::new(1.0, d(CURRENT)),
        "mol" => Unit::new(1.0, d(SUBSTANCE)),
        "cd" => Unit::new(1.0, d(LUMINOUS)),

        // -- IAU-recognized derived (Table 4) --
        "Hz" => Unit::new(1.0, time.powf(-1.0)),
        "J" => Unit::new(1.0, energy),
        "W" => Unit::new(1.0, power),
        "V" => Unit::new(1.0, voltage),
        "N" => Unit::new(1.0, force),
        "Pa" => Unit::new(1.0, force.mul(length.powf(-2.0))),
        "C" => Unit::new(1.0, charge),
        "Ohm" => Unit::new(1.0, voltage.div(d(CURRENT))),
        "S" => Unit::new(1.0, d(CURRENT).div(voltage)),
        "F" => Unit::new(1.0, charge.div(voltage)),
        "Wb" => Unit::new(1.0, magnetic_flux),
        "T" => Unit::new(1.0, magnetic_flux.mul(length.powf(-2.0))),
        "H" => Unit::new(1.0, magnetic_flux.div(d(CURRENT))),
        "lm" => Unit::new(sr_scale, d(LUMINOUS).mul(angle.powf(2.0))),
        "lx" => Unit::new(
            sr_scale,
            d(LUMINOUS).mul(angle.powf(2.0)).mul(length.powf(-2.0)),
        ),

        // -- angle (Table 5) --
        "deg" => Unit::new(1.0, angle),
        "arcmin" => Unit::new(1.0 / 60.0, angle),
        "arcsec" => Unit::new(1.0 / 3600.0, angle),
        "mas" => Unit::new(1.0 / 3_600_000.0, angle),

        // -- time (Table 5) --
        "min" => Unit::new(60.0, time),
        "h" => Unit::new(3600.0, time),
        "d" => Unit::new(86_400.0, time),
        // Julian year; `a` is the IAU spelling, `yr` the common one.
        "a" | "yr" => Unit::new(JULIAN_YEAR, time),
        // Sec.9.3 Table 30 adds the century to the Sec.4.3 set, plus
        // the tropical and Besselian years. The latter two are not
        // constants -- the standard gives each as a polynomial in the
        // epoch -- so these are their values at the epoch the
        // polynomial is expanded about (J2000 for `ta`, J1900 for
        // `Ba`), from Rots et al. (2015), A&A 574, A36, Sect.4.2:
        // 365.24219040211236 d and 365.2421987817 d respectively.
        // The standard discourages writing either, and warns
        // that data using them may not follow its definitions at all.
        "cy" => Unit::new(100.0 * JULIAN_YEAR, time),
        "ta" => Unit::new(365.242_190_402_112_4 * 86_400.0, time),
        "Ba" => Unit::new(365.242_198_781_7 * 86_400.0, time),

        // -- energy (Table 5) --
        "eV" => Unit::new(1.602_176_634e-19, energy),
        "erg" => Unit::new(1e-7, energy),
        // h c R_inf = 2.1798723611030e-18 J (CODATA), tabulated rather
        // than derived so this stays byte-identical to Table 5.
        "Ry" => Unit::new(2.179_872_361_1e-18, energy),

        // -- mass (Table 5) --
        "solMass" => Unit::new(1.988_41e30, mass),
        "u" => Unit::new(1.660_539_066_60e-27, mass),

        // -- luminosity (Table 5) --
        "solLum" => Unit::new(3.828e26, power),

        // -- length (Table 5) --
        "Angstrom" => Unit::new(1e-10, length),
        "solRad" => Unit::new(6.957e8, length),
        "AU" => Unit::new(1.495_978_707e11, length),
        // Exactly 9.4607304725808e15 m, and visibly so.
        "lyr" => Unit::new(SPEED_OF_LIGHT * JULIAN_YEAR, length),
        "pc" => Unit::new(3.085_677_581_491_367e16, length),

        // -- events, flux, and the rest (Table 5) --
        "count" | "ct" => Unit::new(1.0, d(COUNT)),
        "photon" | "ph" => Unit::new(1.0, d(COUNT)),
        // Jy = 1e-26 W m^-2 Hz^-1.
        "Jy" => Unit::new(1e-26, power.mul(length.powf(-2.0)).mul(time)),
        // A level, not a dimension: five magnitudes are a factor of a
        // hundred, so rescaling what it is measured against shifts it
        // additively. `mag(AB)` and friends are handled in `parse_atom`.
        "mag" => Unit::level(Level::MAG),
        // Not Table 5 symbols, but the same structure and the natural
        // spelling for anyone reaching for it: the decibel and the neper.
        "dB" => Unit::level(Level::DB),
        "Np" => Unit::level(Level::NP),
        // 1 R = 1e10 / 4pi photon m^-2 s^-1 sr^-1; the sr in the
        // denominator brings in 1/sr_scale on the way to deg^-2.
        "R" => Unit::new(
            1e10 / (4.0 * std::f64::consts::PI) / sr_scale,
            d(COUNT)
                .mul(length.powf(-2.0))
                .mul(time.powf(-1.0))
                .mul(angle.powf(-2.0)),
        ),
        "G" => Unit::new(1e-4, magnetic_flux.mul(length.powf(-2.0))),
        "pixel" | "pix" => Unit::new(1.0, d(PIXEL)),
        "voxel" => Unit::new(1.0, d(VOXEL)),
        "barn" => Unit::new(1e-28, length.powf(2.0)),
        "D" => Unit::new((1.0 / 3.0) * 1e-29, charge.mul(length)),
        "chan" => Unit::new(1.0, d(BIN)),
        "bin" => Unit::new(1.0, d(BIN)),
        // A data volume, not a photon count: `byte/s` is a rate of bits,
        // and must not convert into `ct/s`.
        "byte" => Unit::new(8.0, d(BIT)),
        "bit" => Unit::new(1.0, d(BIT)),
        // The ADU-to-electron factor is the detector gain, which varies
        // by instrument and often by exposure, so `adu` is left
        // incommensurable rather than declared equal to one count.
        "adu" => Unit::new(1.0, d(ADU)),
        "beam" => Unit::new(1.0, d(BEAM)),
        // `Sun` marks a quantity relative to the Sun (e.g. abundances).
        // A ratio against a reference, so a level -- and emphatically not
        // dimensionless, which would let it cancel out of every compound
        // it appears in.
        "Sun" => Unit::level(Level::SUN),
        _ => return None,
    })
}

/// Decimal prefix (Standard Table 7). `da` is the only two-character
/// one.
fn lookup_prefix(p: &str) -> Option<f64> {
    Some(match p {
        "y" => 1e-24,
        "z" => 1e-21,
        "a" => 1e-18,
        "f" => 1e-15,
        "p" => 1e-12,
        "n" => 1e-9,
        "u" => 1e-6,
        "m" => 1e-3,
        "c" => 1e-2,
        "d" => 1e-1,
        "da" => 1e1,
        "h" => 1e2,
        "k" => 1e3,
        "M" => 1e6,
        "G" => 1e9,
        "T" => 1e12,
        "P" => 1e15,
        "E" => 1e18,
        "Z" => 1e21,
        "Y" => 1e24,
        _ => return None,
    })
}

/// Whether a decimal prefix may be attached to this base symbol.
///
/// Sec.4.3 allows prefixes on the Table 4 (SI and IAU-derived) units
/// generally, but Table 5's `(dagger)` note restricts them to the
/// extended units it marks. So `keV` and `Mpc` are legal while
/// `marcsec` is not -- `arcsec` carries no dagger, and `mas` is a
/// tabulated symbol in its own right rather than a prefixed one.
/// `wcslib` draws the line in the same place.
fn allows_prefix(sym: &str) -> bool {
    matches!(
        sym,
        // Table 4: SI base, supplementary, and IAU-recognized derived.
        // `kg` is excluded -- the prefix goes on `g`.
        "m" | "g"
            | "s"
            | "rad"
            | "sr"
            | "K"
            | "A"
            | "mol"
            | "cd"
            | "Hz"
            | "J"
            | "W"
            | "V"
            | "N"
            | "Pa"
            | "C"
            | "Ohm"
            | "S"
            | "F"
            | "Wb"
            | "T"
            | "H"
            | "lm"
            | "lx"
            // Table 5, dagger-marked only.
            | "a"
            | "yr"
            | "eV"
            | "pc"
            | "Jy"
            | "mag"
            | "R"
            | "G"
            | "barn"
            | "bit"
            | "byte"
    )
}

/// Resolve a bare symbol, trying the exact spelling before any prefix.
///
/// The order matters and the standard says so: `Pa` is pascal, not
/// peta-annum, and Table 5 marks that reading "forbidden". The same
/// rule keeps `min`, `mas`, `cd`, `ct`, `pc`, `d`, `h` and `T` on their
/// tabulated meanings.
pub(super) fn resolve_symbol(sym: &str) -> Result<Unit> {
    if let Some(q) = lookup_base(sym) {
        return Ok(q);
    }
    // Two-character prefix first: `da` would otherwise resolve as
    // deci- applied to `a`.
    for cut in [2, 1] {
        if sym.len() > cut
            && sym.is_char_boundary(cut)
            && let Some(factor) = lookup_prefix(&sym[..cut])
            && let Some(q) = lookup_base(&sym[cut..])
            && allows_prefix(&sym[cut..])
        {
            return Ok(match q.level {
                // On a logarithmic level the prefix scales the level's
                // *value* -- `mmag` is a thousandth of a magnitude, not
                // the magnitude of a thousandth of the flux -- so it
                // folds into `Level::factor` exactly as a numeric
                // multiplier does in `Unit::mul`.
                Some(l) if l.log_base.is_some() => Unit {
                    level: Some(Level {
                        factor: l.factor / factor,
                        ..l
                    }),
                    ..q
                },
                // Spread rather than rebuild, keeping any ratio level.
                _ => Unit {
                    scale: q.scale * factor,
                    ..q
                },
            });
        }
    }
    Err(FitsError::Header(format!(
        "unrecognized unit `{sym}` (Standard Sec.4.3 Tables 4-5)"
    )))
}
