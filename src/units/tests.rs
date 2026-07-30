//! Cross-cutting tests: anything that exercises the parser, the table
//! and the rendering together.

use super::constants;
use super::dimension::{Dimension, dim, dimensions};
use super::level::{Level, Reference, Zero};
use super::unit::Unit;
use super::{
    convert, factor_to, factor_to_lenient, parse_comment_unit, parse_unit, parse_unit_lenient,
};

fn scale(s: &str) -> f64 {
    parse_unit(s).unwrap_or_else(|e| panic!("{s}: {e}")).scale
}

#[test]
fn base_units_and_prefixes() {
    assert!((scale("m") - 1.0).abs() < 1e-15);
    assert!((scale("km") - 1e3).abs() < 1e-9);
    assert!((scale("nm") - 1e-9).abs() < 1e-24);
    assert!((scale("Angstrom") - 1e-10).abs() < 1e-25);
    assert!((scale("GHz") - 1e9).abs() < 1e-3);
    assert!((scale("keV") - 1.602_176_634e-16).abs() < 1e-30);
}

/// Table 5 marks the peta-annum reading of `Pa` forbidden, so the
/// exact spelling has to win over any prefix split.
#[test]
fn exact_spellings_beat_prefix_splits() {
    assert!(
        parse_unit("Pa")
            .unwrap()
            .dimension
            .matches(dimensions::ENERGY.mul(dimensions::LENGTH.powf(-3.0)))
    );
    assert!((scale("min") - 60.0).abs() < 1e-12, "min is a minute");
    assert!((scale("d") - 86_400.0).abs() < 1e-9, "d is a day");
    assert!((scale("h") - 3600.0).abs() < 1e-9, "h is an hour");
    assert!((scale("mas") - 1.0 / 3_600_000.0).abs() < 1e-18);
    assert!(
        parse_unit("cd")
            .unwrap()
            .dimension
            .matches(Dimension::base(dim::LUMINOUS))
    );
}

/// The forms Sec.4.3 spells out for "per second" and "metres
/// squared" must all agree.
#[test]
fn every_spelling_of_a_power_agrees() {
    let want = scale("m2");
    for s in ["m**2", "m**+2", "m^2", "m^(+2)", "m**(2)"] {
        assert!((scale(s) - want).abs() < 1e-12, "{s}");
    }
    let per = scale("m-3");
    for s in ["m**-3", "m^(-3)", "/m3"] {
        assert!((scale(s) - per).abs() < 1e-12, "{s}");
    }
    // Three-halves: parenthesised forms only.
    for s in ["m(1.5)", "m^(1.5)", "m**(1.5)", "m(3/2)", "m**(3/2)"] {
        assert!(parse_unit(s).is_ok(), "{s} should parse");
    }
    // The standard explicitly excludes these two.
    assert!(parse_unit("m1.5").is_err(), "m1.5 is not valid");
}

/// The whole point of the exercise: `km s-1` is legal Sec.4.3 for
/// km/s, and used to be silently read as a factor of 1.
#[test]
fn space_is_multiplication() {
    let kms = scale("km/s");
    assert!((kms - 1e3).abs() < 1e-9);
    for s in ["km s-1", "km s**-1", "km.s**(-1)", "km*s^-1", "1000 m/s"] {
        assert!((scale(s) - kms).abs() < 1e-6, "{s} -> {}", scale(s));
    }
}

/// Division associates left to right, so the flux-calibrated
/// spectrum idiom means what its writers intend.
#[test]
fn repeated_division_associates_left() {
    let q = parse_unit("erg/s/cm2/Angstrom").unwrap();
    let want = dimensions::ENERGY
        .mul(dimensions::TIME.powf(-1.0))
        .mul(dimensions::LENGTH.powf(-2.0))
        .mul(dimensions::LENGTH.powf(-1.0));
    assert!(q.dimension.matches(want), "got {}", q.dimension);
    // erg = 1e-7 J, cm^-2 = 1e4 m^-2, Angstrom^-1 = 1e10 m^-1.
    assert!((q.scale - 1e-7 * 1e4 * 1e10).abs() / q.scale < 1e-12);
}

#[test]
fn angles_are_their_own_dimension() {
    assert!((scale("deg") - 1.0).abs() < 1e-15);
    assert!((scale("arcsec") - 1.0 / 3600.0).abs() < 1e-15);
    assert!((scale("rad") - 180.0 / std::f64::consts::PI).abs() < 1e-12);
    // An angle must not be interchangeable with a pure number.
    assert!(!parse_unit("deg").unwrap().dimension.is_dimensionless());
}

/// A count rate is not a frequency, however similar they look.
#[test]
fn pseudo_dimensions_do_not_cancel_into_si() {
    let ct_s = parse_unit("ct/s").unwrap().dimension;
    assert!(!ct_s.matches(dimensions::FREQUENCY));
    let jy_beam = parse_unit("Jy/beam").unwrap().dimension;
    assert!(!jy_beam.matches(parse_unit("Jy").unwrap().dimension));
}

#[test]
fn numeric_multiplier_and_grouping() {
    assert!((scale("10**3 m") - 1e3).abs() < 1e-9);
    assert!((scale("(km)") - 1e3).abs() < 1e-9);
    assert!((scale("sqrt(m2)") - 1.0).abs() < 1e-12);
}

/// `log()` and `ln()` are levels over their argument, so they parse
/// and check. `exp()` is not a unit at all.
#[test]
fn logarithmic_units_are_levels_not_refusals() {
    let log_hz = parse_unit("log(Hz)").unwrap();
    assert!(log_hz.is_level());
    // Agrees with itself ...
    assert!(log_hz.converter_to(parse_unit("log(Hz)").unwrap()).is_ok());
    // ... and is emphatically not a frequency.
    assert!(log_hz.converter_to(parse_unit("Hz").unwrap()).is_err());
    // A decade of the argument is an additive 1, not a factor of 10.
    let c = parse_unit("log(kHz)")
        .unwrap()
        .converter_to(log_hz)
        .unwrap();
    assert!((c.scale - 1.0).abs() < 1e-12);
    assert!((c.offset - 3.0).abs() < 1e-12, "offset {}", c.offset);
    assert!(parse_unit("ln(m)").unwrap().is_level());
    assert!(parse_unit("exp(s)").is_err(), "exp() is not a unit");
}

#[test]
fn unknown_and_malformed_are_errors_not_unity() {
    for s in ["AA", "furlong", "m/", "m**", "(m", "m)"] {
        assert!(parse_unit(s).is_err(), "{s} should not parse");
    }
}

#[test]
fn comment_annotations_are_extracted() {
    assert_eq!(parse_comment_unit("[AU] Distance to target"), Some("AU"));
    assert_eq!(parse_comment_unit("exposure time [s]"), Some("s"));
    assert!(parse_comment_unit("no brackets here").is_none());
    assert!(parse_comment_unit("[] empty brackets").is_none());
}

/// Spellings that turn up in real `[unit]` annotations.
#[test]
fn everyday_compound_spellings() {
    assert!((scale("km/h") - 1000.0 / 3600.0).abs() < 1e-10);
    // `d` is the Table 5 symbol for a day; `day` is not one.
    assert!((scale("AU/d") - 1.495_978_707e11 / 86_400.0).abs() < 1.0);
    assert!(parse_unit("AU/day").is_err());
    assert!((scale("AU") - 1.495_978_707e11).abs() < 1e3);
}

/// Sec.4.3.1 makes the numeric multiplier a prefix of the compound
/// string, and Table 7 makes juxtaposition multiplication, so no
/// separator is required between the two. `10**(46)erg/s` is the
/// section's own worked example.
///
/// Regression: only the whitespace-separated spelling parsed; every
/// abutting form was rejected as trailing text.
#[test]
fn numeric_multiplier_may_abut_its_units() {
    // 10^46 erg/s = 10^39 W.
    for s in [
        "10**(46)erg/s",
        "10^(46)erg/s",
        "10**46erg/s",
        "(10**46)erg/s",
        "10**(46) erg/s",
    ] {
        assert!(
            (scale(s) / 1e39 - 1.0).abs() < 1e-12,
            "{s}: got {}",
            scale(s)
        );
    }
    // The `10+-k` spelling, abutting.
    assert!((scale("10-3m") / 1e-3 - 1.0).abs() < 1e-12);
    assert!((scale("10+3m") / 1e3 - 1.0).abs() < 1e-12);
    assert!((scale("10**3m") / 1e3 - 1.0).abs() < 1e-12);
    // Dimensions survive the juxtaposition.
    let q = parse_unit("10**(-20)J/(s.m**2.Angstrom)").unwrap();
    assert!(
        q.dimension
            .matches(parse_unit("W/(m2.m)").unwrap().dimension)
    );
    // A symbol is still never split: `m2` is m squared, not m times 2.
    assert!((scale("m2") - 1.0).abs() < 1e-15);
    assert_eq!(
        parse_unit("m2").unwrap().dimension,
        parse_unit("m**2").unwrap().dimension
    );
    // Sec.4.3.2 still forbids `m1.5` for three-halves.
    assert!(parse_unit("m1.5").is_err());
}

/// The canonical solid angle is deg^2, so every steradian-derived
/// unit must agree with its own definition spelled through `sr`.
///
/// Regression: `lm`, `lx` and `R` were tabulated as if the
/// steradian itself were canonical, so `lm` and `cd sr` carried
/// the same dimension but scales (180/pi)^2 apart.
#[test]
fn steradian_derived_units_are_self_consistent() {
    let lm = parse_unit("lm").unwrap();
    let cdsr = parse_unit("cd sr").unwrap();
    assert!(lm.dimension.matches(cdsr.dimension));
    assert!((lm.scale / cdsr.scale - 1.0).abs() < 1e-12);

    let lx = parse_unit("lx").unwrap();
    let lm_m2 = parse_unit("lm/m2").unwrap();
    assert!(lx.dimension.matches(lm_m2.dimension));
    assert!((lx.scale / lm_m2.scale - 1.0).abs() < 1e-12);

    // 1 R = 1e10 / 4pi photon m^-2 s^-1 sr^-1, by definition.
    let r = parse_unit("R").unwrap();
    let def = parse_unit("ph m-2 s-1 sr-1").unwrap();
    assert!(r.dimension.matches(def.dimension));
    let in_def_units = r.scale / def.scale;
    let want = 1e10 / (4.0 * std::f64::consts::PI);
    assert!(
        (in_def_units / want - 1.0).abs() < 1e-12,
        "1 R = {in_def_units:e} ph/m2/s/sr, want {want:e}"
    );
}

/// The informal spellings real `[unit]` annotations and `TIMEUNIT`
/// values carry, which the strict grammar rightly refuses.
#[test]
fn informal_spellings_resolve_leniently() {
    let lenient = |s: &str| {
        parse_unit_lenient(s)
            .unwrap_or_else(|e| panic!("{s}: {e}"))
            .scale
    };
    assert!((lenient("degrees") - 1.0).abs() < 1e-15);
    assert!((lenient("DEG") - 1.0).abs() < 1e-15);
    assert!((lenient("arcsecs") - 1.0 / 3600.0).abs() < 1e-15);
    assert!((lenient("sec") - 1.0).abs() < 1e-15);
    assert!((lenient("DAY") - 86_400.0).abs() < 1e-9);
    assert!((lenient("KM/S") - 1e3).abs() < 1e-9);
    assert!((lenient("AU/day") - 1.495_978_707e11 / 86_400.0).abs() < 1.0);
    assert!((lenient("Angstroms") - 1e-10).abs() < 1e-25);
    assert!((lenient("microns") - 1e-6).abs() < 1e-21);
    assert!((lenient("uas") - 1.0 / 3_600_000_000.0).abs() < 1e-21);
    assert!((lenient("'") - 1.0 / 60.0).abs() < 1e-15);
    assert!((lenient("\"") - 1.0 / 3600.0).abs() < 1e-15);
    // A conforming string keeps its strict meaning untouched:
    // `d` is a day, not run through any alias.
    assert!((lenient("d") - 86_400.0).abs() < 1e-9);
    // The strict grammar still refuses all of these.
    for s in ["degrees", "DEG", "sec", "DAY", "Angstroms"] {
        assert!(parse_unit(s).is_err(), "{s} must stay invalid Sec.4.3");
    }
    // Gibberish still fails, reporting the strict error.
    assert!(parse_unit_lenient("furlong").is_err());
}

/// Detector spellings real `BUNIT` values carry: `COUNTS`, `ADU`,
/// `DN`, and the rest.
#[test]
fn detector_spellings_resolve_leniently() {
    let dim = |s: &str| {
        parse_unit_lenient(s)
            .unwrap_or_else(|e| panic!("{s}: {e}"))
            .dimension
    };
    assert!(dim("counts").matches(dim("count")));
    assert!(dim("CTS").matches(dim("count")));
    assert!(dim("counts/s").matches(dim("ct/s")));
    assert!(dim("photons").matches(dim("ph")));
    assert!(dim("ADU").matches(dim("adu")));
    assert!(dim("DN").matches(dim("adu")));
    assert!(dim("pixels").matches(dim("pix")));
    assert!(dim("PX").matches(dim("pix")));
    // `dN` is a decinewton to the grammar, and the grammar wins; only
    // the spellings it refuses fall back to the data-number alias.
    assert!(dim("dN").matches(dimensions::FORCE));
    let scale = |s: &str| {
        parse_unit_lenient(s)
            .unwrap_or_else(|e| panic!("{s}: {e}"))
            .scale
    };
    assert!((scale("ergs/s") - 1e-7).abs() < 1e-19);
    assert!((scale("Kelvin") - 1.0).abs() < 1e-15);
    assert!((scale("Gauss") - 1e-4).abs() < 1e-16);
    assert!((scale("Janskys") - 1e-26).abs() < 1e-38);
    assert!((scale("hertz") - 1.0).abs() < 1e-15);
    assert!((scale("msec") - 1e-3).abs() < 1e-15);
    assert!((scale("usec") - 1e-6).abs() < 1e-18);
}

/// `factor_to_lenient` may re-read a strict-but-wrong-dimension
/// spelling: `S` is siemens to the grammar, seconds to `TIMEUNIT`
/// writers, and `as` is an attosecond or an arcsecond depending on
/// what was asked for.
#[test]
fn factor_to_lenient_uses_the_dimension_to_disambiguate() {
    assert!((factor_to_lenient("S", dimensions::TIME).unwrap() - 1.0).abs() < 1e-15);
    assert!((factor_to_lenient("as", dimensions::TIME).unwrap() - 1e-18).abs() < 1e-33);
    assert!((factor_to_lenient("as", dimensions::ANGLE).unwrap() - 1.0 / 3600.0).abs() < 1e-15);
    // A genuine mismatch still refuses.
    assert!(factor_to_lenient("Hz", dimensions::LENGTH).is_err());
}

#[test]
fn blank_is_the_undefined_unit() {
    assert!((parse_unit("").unwrap().scale - 1.0).abs() < 1e-15);
    assert!((parse_unit("   ").unwrap().scale - 1.0).abs() < 1e-15);
}

#[test]
fn factor_to_checks_the_dimension() {
    assert!((factor_to("km", dimensions::LENGTH).unwrap() - 1e3).abs() < 1e-9);
    assert!((factor_to("arcsec", dimensions::ANGLE).unwrap() - 1.0 / 3600.0).abs() < 1e-15);
    // A wavelength axis declaring a frequency unit is a broken
    // header, not something to rescale.
    let e = factor_to("Hz", dimensions::LENGTH).unwrap_err();
    assert!(e.to_string().contains("required"), "{e}");
    // Blank passes through as canonical.
    assert!((factor_to("", dimensions::ANGLE).unwrap() - 1.0).abs() < 1e-15);
}

/// The angle units a celestial `CUNIT` can legally carry, and the
/// near-misses it cannot (Standard Sec.8.1 requires degrees; the
/// parser still has to convert the ones headers do use).
///
/// Every accept and reject below was checked against wcslib on an
/// `RA---TAN`/`DEC--TAN` pair and agrees with it -- the rejections
/// included. `uas` and `marcsec` are not Table 5 spellings:
/// `arcsec` carries no prefix marker and `mas` is a tabulated
/// symbol in its own right.
#[test]
fn celestial_angle_units_match_wcslib() {
    let deg = |u: &str| factor_to(u, dimensions::ANGLE);
    assert!((deg("deg").unwrap() - 1.0).abs() < 1e-15);
    assert!((deg("arcsec").unwrap() - 1.0 / 3600.0).abs() < 1e-15);
    assert!((deg("arcmin").unwrap() - 1.0 / 60.0).abs() < 1e-15);
    assert!((deg("mas").unwrap() - 1.0 / 3_600_000.0).abs() < 1e-18);
    assert!((deg("rad").unwrap() - 180.0 / std::f64::consts::PI).abs() < 1e-12);
    // Blank is the CUNIT default; Sec.8.1 fixes celestial units to
    // degrees, so it is taken as such.
    assert!((deg("").unwrap() - 1.0).abs() < 1e-15);
    assert!((deg("   ").unwrap() - 1.0).abs() < 1e-15);

    assert!(deg("Hz").is_err(), "a frequency is not an angle");
    assert!(deg("m").is_err(), "a length is not an angle");
    assert!(deg("DEG").is_err(), "case is significant");
    assert!(deg("arcsecs").is_err(), "not a Table 5 spelling");
    assert!(deg("uas").is_err(), "wcslib rejects this too");
    assert!(deg("marcsec").is_err(), "wcslib rejects this too");
}

/// Four quantities that used to share a slot with something they do
/// not convert into. The tracked set now matches `wcslib`'s
/// (`wcsunits.h`), which separates all four.
#[test]
fn incommensurable_tags_do_not_convert() {
    let dim = |s: &str| parse_unit(s).unwrap().dimension;
    // A data volume is not a photon count ...
    assert!(!dim("byte").matches(dim("count")), "byte is not a count");
    assert!(!dim("byte/s").matches(dim("ct/s")));
    // ... but a byte is still eight bits.
    assert!(dim("byte").matches(dim("bit")));
    assert!((scale("byte") / scale("bit") - 8.0).abs() < 1e-12);
    // The ADU-to-electron factor is the detector gain, so there is no
    // universal conversion to offer.
    assert!(!dim("adu").matches(dim("count")), "an ADU is not a photon");
    assert!(!dim("voxel").matches(dim("pixel")));
    // A solar ratio is a level, so it cannot cancel out of a compound
    // and leave a bare number behind.
    let sun = parse_unit("Sun").unwrap();
    assert!(sun.is_level());
    assert!(
        sun.converter_to(Unit::new(1.0, Dimension::NONE)).is_err(),
        "`Sun` must not reduce to a pure number"
    );
    assert!(
        parse_unit("solMass/Sun")
            .unwrap()
            .converter_to(parse_unit("solMass").unwrap())
            .is_err()
    );
}

/// The case that forces the level to wrap the linear part: surface
/// brightness. 1 deg^2 is 3600^2 arcsec^2, so the same sky is
/// 2.5 log10(3600^2) = 17.78 magnitudes brighter per square degree --
/// an additive shift, not a factor.
///
/// Regression: this reported a factor of 1.296e7, turning a
/// 20 mag/arcsec2 sky into 2.592e8 mag/deg2 rather than 2.2185.
#[test]
fn surface_brightness_converts_by_a_shift() {
    let c = parse_unit("mag/arcsec2")
        .unwrap()
        .converter_to(parse_unit("mag/deg2").unwrap())
        .unwrap();
    let want = -2.5 * (3600.0_f64 * 3600.0).log10();
    assert!((c.scale - 1.0).abs() < 1e-12, "scale {}", c.scale);
    assert!((c.offset - want).abs() < 1e-9, "offset {}", c.offset);
    assert!((c.apply(20.0) - 2.218_487_496_163_563_7).abs() < 1e-9);
    // `convert` is the same thing on strings.
    let v = convert(20.0, "mag/arcsec2", "mag/deg2").unwrap();
    assert!((v - 2.218_487_496_163_563_7).abs() < 1e-9);
    // `factor_to` takes a bare `Dimension`, which cannot say "a
    // magnitude", so a level has nothing there to convert *to* and
    // it refuses rather than returning the multiplicative part alone.
    let per_deg2 = parse_unit("mag/deg2").unwrap().dimension;
    let e = factor_to("mag/arcsec2", per_deg2).unwrap_err().to_string();
    assert!(e.contains("level"), "{e}");
    // A level against a linear unit stays an error at every entry.
    assert!(factor_to("mag", dimensions::DIMENSIONLESS).is_err());
    assert!(factor_to_lenient("mag", dimensions::DIMENSIONLESS).is_err());
    assert!(convert(1.0, "mag", "Jy").is_err());
    // Linear units are untouched.
    assert!(factor_to("arcsec", dimensions::ANGLE).is_ok());
}

/// A prefix or numeric multiplier on a logarithmic level scales the
/// level's *value*, so `mmag` is a thousandth of a magnitude.
///
/// Regression: the multiplier landed on the linear reference inside the
/// logarithm, so 5 mmag converted to 12.5 mag -- the magnitude of a
/// thousandth of the flux -- where astropy's `(5*u.mmag).to(u.mag)` and
/// wcslib's linear magnitude slot both give 0.005.
#[test]
fn prefixes_and_multipliers_scale_a_levels_value() {
    let v = convert(5.0, "mmag", "mag").unwrap();
    assert!((v - 0.005).abs() < 1e-15, "5 mmag -> {v} mag");
    // Every spelling of the same multiplier agrees.
    for s in ["0.001mag", "10**-3 mag", "mag 10**-3", "mag/1000"] {
        let v = convert(5.0, s, "mag").unwrap();
        assert!((v - 0.005).abs() < 1e-12, "{s}: got {v}");
    }
    // And back up.
    let v = convert(0.005, "mag", "mmag").unwrap();
    assert!((v - 5.0).abs() < 1e-9, "0.005 mag -> {v} mmag");
    // The dimensioned linear part still sits underneath the logarithm:
    // mmag/arcsec2 to mag/arcsec2 is exactly the 1e-3 ...
    let c = parse_unit("mmag/arcsec2")
        .unwrap()
        .converter_to(parse_unit("mag/arcsec2").unwrap())
        .unwrap();
    assert!((c.scale - 1e-3).abs() < 1e-15, "scale {}", c.scale);
    assert!(c.offset.abs() < 1e-12, "offset {}", c.offset);
    // ... and to mag/deg2 it picks up the same -17.78 shift the
    // unprefixed unit does.
    let c = parse_unit("mmag/arcsec2")
        .unwrap()
        .converter_to(parse_unit("mag/deg2").unwrap())
        .unwrap();
    let want = -2.5 * (3600.0_f64 * 3600.0).log10();
    assert!((c.scale - 1e-3).abs() < 1e-15, "scale {}", c.scale);
    assert!((c.offset - want).abs() < 1e-9, "offset {}", c.offset);
    // dB works the same way: 30 tenth-decibels are 3 dB.
    let v = convert(30.0, "0.1dB", "dB").unwrap();
    assert!((v - 3.0).abs() < 1e-12, "got {v}");
    // Inside log()'s parentheses a number is part of the argument, not
    // a value multiplier: log(0.001) to log(1) is an additive -3.
    let c = parse_unit("log(0.001)")
        .unwrap()
        .converter_to(parse_unit("log(1)").unwrap())
        .unwrap();
    assert!((c.scale - 1.0).abs() < 1e-12, "scale {}", c.scale);
    assert!((c.offset + 3.0).abs() < 1e-12, "offset {}", c.offset);
    // A prefixed magnitude is still logarithmic: no squaring it, and no
    // conversion into a linear unit.
    assert!(parse_unit("mmag").unwrap().try_pow(2.0).is_err());
    assert!(convert(1.0, "mmag", "Jy").is_err());
    // The canonical form fuses the multiplier back into the prefix.
    assert_eq!(parse_unit("mmag").unwrap().to_string(), "mmag");
    assert_eq!(parse_unit("0.001mag").unwrap().to_string(), "mmag");
}

/// Regression: a multi-byte character inside a parenthesised exponent
/// panicked on a char-boundary slice. Malformed input -- non-ASCII
/// included -- is an error, never a panic.
#[test]
fn non_ascii_input_is_an_error_not_a_panic() {
    for s in ["m**(µ)", "m**(¾)", "m**(3/2µ)", "Å", "µm", "log(µ)", "m**("] {
        assert!(parse_unit(s).is_err(), "`{s}` should fail to parse");
        assert!(
            parse_unit_lenient(s).is_err(),
            "`{s}` should fail leniently too"
        );
    }
}

/// A hand-built zero point renders by its own name rather than being
/// passed off as `mag(ST)`.
#[test]
fn custom_zero_points_render_by_name() {
    let u = Unit {
        scale: 1.0,
        dimension: Dimension::NONE,
        level: Some(Level {
            log_base: Some(10.0),
            factor: -2.5,
            reference: Reference::Absolute(Zero {
                name: "F606W",
                value: 1e-20,
                dimension: parse_unit("Jy").unwrap().dimension,
            }),
        }),
    };
    assert_eq!(u.to_string(), "mag(F606W)");
    // The tabulated two still spell themselves.
    assert_eq!(parse_unit("mag(AB)").unwrap().to_string(), "mag(AB)");
    assert_eq!(parse_unit("mag(ST)").unwrap().to_string(), "mag(ST)");
}

/// Levels convert between each other where the algebra allows, which
/// the dimension-slot design could not express at all.
#[test]
fn levels_interconvert_by_their_own_algebra() {
    // A magnitude is a factor 10^-0.4 in flux, i.e. -4 dB.
    let c = parse_unit("mag")
        .unwrap()
        .converter_to(parse_unit("dB").unwrap())
        .unwrap();
    assert!((c.scale - -4.0).abs() < 1e-12, "mag->dB {}", c.scale);
    // On the power convention, 1 Np is 10/ln(10) dB.
    let c = parse_unit("Np")
        .unwrap()
        .converter_to(parse_unit("dB").unwrap())
        .unwrap();
    assert!((c.scale - 10.0 / 10.0_f64.ln()).abs() < 1e-12);
}

/// `mag(AB)` is `astropy`'s own spelling, and the zero points are
/// real quantities: AB is 3630.78 Jy, ST is 3.631e-9 erg/s/cm2/A.
/// Vega is neither -- its zero point is passband-dependent, so it is
/// an `Object` with no number attached.
#[test]
fn magnitude_zero_points_are_distinguished() {
    let ab = parse_unit("mag(AB)").unwrap();
    let st = parse_unit("mag(ST)").unwrap();
    let vega = parse_unit("mag(Vega)").unwrap();
    let bare = parse_unit("mag").unwrap();
    // Each agrees with itself ...
    assert!(ab.converter_to(parse_unit("mag(AB)").unwrap()).is_ok());
    // ... and with nothing else, since the zero points differ.
    for (a, b) in [(ab, st), (ab, vega), (ab, bare), (vega, bare)] {
        assert!(a.converter_to(b).is_err(), "{a} must not convert to {b}");
    }
    // The AB zero point matches its definition, 3630.78 Jy.
    let Some(Reference::Absolute(z)) = ab.level.map(|l| l.reference) else {
        panic!("mag(AB) should carry an absolute zero point");
    };
    let jy = parse_unit("Jy").unwrap();
    assert!(z.dimension.matches(jy.dimension));
    assert!((z.value / jy.scale - 3_630.780_547_701).abs() < 1e-6);
    assert!(parse_unit("mag(Betelgeuse)").is_err());
}

/// One equality, not two: `PartialEq` *is* the tolerant comparison,
/// so a caller cannot reach for the exact one by accident. An exact
/// compare answers no here.
#[test]
fn dimension_equality_is_the_tolerant_one() {
    let cubed = parse_unit("m**(1/3)").unwrap().dimension.powf(3.0);
    assert_eq!(cubed, parse_unit("m").unwrap().dimension);
}

/// The rendered form is the canonical decomposition, not the spelling
/// it was parsed from -- but it has to re-parse to the same quantity,
/// since it is what gets written into a card.
#[test]
fn canonical_form_round_trips() {
    for s in [
        "m",
        "km",
        "Jy",
        "erg/s/cm2/Angstrom",
        "ct/s",
        "Jy/beam",
        "AU",
        "arcsec",
        "byte/s",
        "adu",
        "voxel",
        "Sun",
        "solMass/Sun",
        "mag",
        "mag/arcsec2",
        "mag(AB)",
        "mag(ST)",
        "mag(Vega)",
        "dB",
        "log(Hz)",
        "ln(m)",
        "mmag",
        "0.001mag",
        "mmag/arcsec2",
        "0.001 mag(AB)",
        "2dB",
        "Np",
        "log(0.001)",
        "0.5 log(Hz)",
        "m(3/2)",
        "sqrt(m)",
        "10**-20 J/(s.m**2.Angstrom)",
        "",
    ] {
        let q = parse_unit(s).unwrap();
        let rendered = q.to_string();
        let back =
            parse_unit(&rendered).unwrap_or_else(|e| panic!("`{s}` rendered as `{rendered}`: {e}"));
        assert!(back.dimension.matches(q.dimension), "`{s}` -> `{rendered}`");
        assert!(
            (back.scale / q.scale - 1.0).abs() < 1e-12,
            "`{s}` -> `{rendered}`: {} vs {}",
            back.scale,
            q.scale
        );
        // A level has to survive the round trip too, zero point
        // included -- otherwise `mag(AB)` comes back as `mag`.
        assert_eq!(back.level, q.level, "`{s}` -> `{rendered}` lost its level");
    }
    // The spelling itself, so a change to it is a deliberate one.
    assert_eq!(parse_unit("Jy").unwrap().to_string(), "10**-26 kg s^-2");
    assert_eq!(parse_unit("ct/s").unwrap().to_string(), "s^-1 count");
    assert_eq!(parse_unit("sqrt(m)").unwrap().to_string(), "m^(0.5)");
    // Blank stays blank: that is how an undefined `CUNIT` is written.
    assert_eq!(parse_unit("").unwrap().to_string(), "");
}

// -- constants and the Quantity algebra ---------------------------------

/// The Standard tabulates some of the same values as *units*, and this
/// module tabulates them as [`Quantity`] constants. They are written out
/// twice on purpose -- the parser has to reproduce the Standard's
/// numbers exactly -- so this is what stops the two drifting apart.
#[test]
fn constants_agree_with_the_unit_table() {
    let table = |s: &str| parse_unit(s).unwrap().scale;
    let agree = |a: f64, b: f64, tol: f64, what: &str| {
        assert!((a / b - 1.0).abs() < tol, "{what}: {a} vs {b}");
    };
    agree(constants::ASTRONOMICAL_UNIT.value, table("AU"), 1e-15, "au");
    agree(constants::PARSEC.value, table("pc"), 1e-15, "pc");
    agree(constants::LIGHT_YEAR.value, table("lyr"), 1e-15, "lyr");
    agree(
        constants::SOLAR_RADIUS.value,
        table("solRad"),
        1e-15,
        "R_sun",
    );
    agree(
        constants::SOLAR_LUMINOSITY.value,
        table("solLum"),
        1e-15,
        "L_sun",
    );
    agree(constants::ATOMIC_MASS.value, table("u"), 1e-15, "u");
    agree(
        constants::JULIAN_YEAR.value,
        table("a"),
        1e-15,
        "Julian year",
    );
    agree(constants::DAY.value, table("d"), 1e-15, "day");
    agree(
        constants::JULIAN_CENTURY.value,
        table("cy"),
        1e-15,
        "Julian century",
    );
    agree(
        constants::ELEMENTARY_CHARGE.value,
        table("eV"),
        1e-15,
        "e vs eV",
    );
    // `solMass` is GM_sun/G, so it inherits G's 2.2e-5 uncertainty -- the
    // one place the two are only expected to agree loosely.
    agree(constants::SOLAR_MASS.value, table("solMass"), 1e-4, "M_sun");
}

/// Each constant should carry the dimension its name claims, checked
/// through the parser rather than trusted from a doc comment.
#[test]
fn constants_carry_the_dimensions_their_names_claim() {
    let dim = |s: &str| parse_unit(s).unwrap().dimension;
    let same = |q: super::Quantity, s: &str| {
        assert!(
            q.unit.dimension.matches(dim(s)),
            "{q} should be {s}, got {}",
            q.unit.dimension
        );
    };
    same(constants::SPEED_OF_LIGHT, "m/s");
    same(constants::PLANCK, "J.s");
    same(constants::REDUCED_PLANCK, "J.s");
    same(constants::BOLTZMANN, "J/K");
    same(constants::ELEMENTARY_CHARGE, "C");
    same(constants::AVOGADRO, "/mol");
    same(constants::GRAVITATIONAL_CONSTANT, "m3/(kg.s2)");
    same(constants::STEFAN_BOLTZMANN, "W/(m2.K4)");
    same(constants::GAS_CONSTANT, "J/(mol.K)");
    same(constants::WIEN_DISPLACEMENT, "m.K");
    same(constants::RYDBERG, "/m");
    same(constants::ELECTRON_MASS, "kg");
    same(constants::SOLAR_MASS_PARAMETER, "m3/s2");
    same(constants::SOLAR_EFFECTIVE_TEMPERATURE, "K");
    same(constants::DEGREES_PER_RADIAN, "deg");
    assert!(constants::FINE_STRUCTURE.unit.dimension.is_dimensionless());
    // The raw forms are the same numbers.
    assert!((constants::raw::SPEED_OF_LIGHT - constants::SPEED_OF_LIGHT.value).abs() < 1e-9);
    assert!((constants::raw::PLANCK - constants::PLANCK.value).abs() < 1e-48);
}

/// Physics a downstream crate would actually write, with every step
/// dimension-checked.
#[test]
fn quantity_algebra_checks_the_physics() {
    use super::Quantity;
    // E = h c / lambda for a 500 nm photon, in eV.
    let lambda = Quantity::parse(500.0, "nm").unwrap();
    let e = constants::PLANCK
        .try_mul(constants::SPEED_OF_LIGHT)
        .unwrap()
        .try_div(lambda)
        .unwrap();
    assert!(e.unit.dimension.matches(dimensions::ENERGY));
    let ev = e.to(parse_unit("eV").unwrap()).unwrap();
    // astropy: (500*u.nm).to(u.eV, equivalencies=u.spectral()) = 2.4796839686640
    assert!((ev.value - 2.479_683_968_664).abs() < 1e-9, "got {ev}");

    // Stefan-Boltzmann: the Sun's luminosity from its radius and T_eff,
    // against the IAU nominal value it is meant to reproduce.
    let area = constants::SOLAR_RADIUS
        .try_pow(2.0)
        .unwrap()
        .scaled(4.0 * std::f64::consts::PI);
    let l = constants::STEFAN_BOLTZMANN
        .try_mul(area)
        .unwrap()
        .try_mul(constants::SOLAR_EFFECTIVE_TEMPERATURE.try_pow(4.0).unwrap())
        .unwrap();
    assert!(l.unit.dimension.matches(dimensions::POWER));
    let rel = (l.to_canonical().unwrap() / constants::SOLAR_LUMINOSITY.value - 1.0).abs();
    assert!(rel < 1e-3, "L_sun from sigma R^2 T^4 is off by {rel}");

    // Adding unlike quantities is refused, not silently absorbed.
    assert!(lambda.try_add(Quantity::parse(1.0, "s").unwrap()).is_err());
    // Adding like ones converts first: 500 nm + 0.5 um = 1000 nm.
    let sum = lambda.try_add(Quantity::parse(0.5, "um").unwrap()).unwrap();
    assert!((sum.value - 1000.0).abs() < 1e-9, "got {sum}");
    assert!(lambda.commensurable_with(Quantity::parse(1.0, "AU").unwrap()));
    assert!(!lambda.commensurable_with(Quantity::parse(1.0, "s").unwrap()));
}

/// A level is a quantity too, and converts by its shift.
#[test]
fn quantity_handles_levels() {
    use super::Quantity;
    let sb = Quantity::parse(20.0, "mag/arcsec2").unwrap();
    let per_deg2 = sb.to(parse_unit("mag/deg2").unwrap()).unwrap();
    assert!((per_deg2.value - 2.218_487_496_163_563_7).abs() < 1e-9);
    // Canonical has no meaning for a level.
    assert!(sb.to_canonical().is_err());
    // Nor does squaring one.
    assert!(sb.try_pow(2.0).is_err());
}

/// The constants reach 1e-34, so writing values out in full is
/// unreadable rather than precise.
#[test]
fn quantity_display_stays_readable_at_any_magnitude() {
    assert_eq!(constants::SPEED_OF_LIGHT.to_string(), "2.99792458e8 m s^-1");
    assert_eq!(constants::PLANCK.to_string(), "6.62607015e-34 m^2 kg s^-1");
    assert_eq!(super::Quantity::scalar(0.0).to_string(), "0");
    assert_eq!(
        super::Quantity::parse(1.5, "km").unwrap().to_string(),
        "1.5 10**3 m"
    );
}

/// Adversarial nesting has to come back as an `Err`: recursion costs
/// native stack, and a stack overflow aborts the process without
/// unwinding, so no caller can defend against it after the fact.
#[test]
fn runaway_nesting_is_an_error_not_a_crash() {
    let deep = format!("{}m{}", "(".repeat(5000), ")".repeat(5000));
    assert!(parse_unit(&deep).is_err());
    // Glued juxtaposition recurses too.
    let glued = format!("m{}", "s2".repeat(5000));
    assert!(parse_unit(&glued).is_err());
    // Real unit strings sit far inside the cap.
    assert!(parse_unit("10**(-20)erg/(s.cm**2.Angstrom)").is_ok());
}

/// A numeric divisor binds the units juxtaposed after it: `cts / 300 s`
/// is a count per 300 seconds, where reading the `s` as a separate
/// factor would put it in the numerator -- the wrong dimension, with
/// no error to show for it. Symbol divisors keep the left-to-right
/// rule documented on the parser.
#[test]
fn numeric_divisor_binds_the_units_after_it() {
    let rate = parse_unit_lenient("cts / 300 s").unwrap();
    let ct_s = parse_unit("ct/s").unwrap();
    assert_eq!(rate.dimension, ct_s.dimension);
    assert!((rate.scale * 300.0 - ct_s.scale).abs() < 1e-12);
    // The strict parser reads it the same way.
    let erg = parse_unit("erg / 300 s").unwrap();
    assert_eq!(erg.dimension, parse_unit("erg/s").unwrap().dimension);
    // Spelled without the space, nothing changes.
    assert_eq!(
        rate.dimension,
        parse_unit_lenient("cts/300s").unwrap().dimension
    );
    // A symbol divisor stays left to right: `m / s K` is `(m/s) K`.
    assert_eq!(
        parse_unit("m / s K").unwrap().dimension,
        parse_unit("m.K/s").unwrap().dimension
    );
}
