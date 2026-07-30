//! The informal spellings real headers carry, and the lenient entry
//! points that accept them.

use super::convert::factor_to;
use super::dimension::Dimension;
use super::parse::parse_unit;
use super::unit::Unit;
use crate::error::Result;

// -- informal spellings -------------------------------------------------

/// Canonical Table 4-5 spelling for one informal token, matched
/// case-insensitively. `None` leaves the token as written.
fn legacy_alias(lower: &str) -> Option<&'static str> {
    Some(match lower {
        "m" | "meter" | "meters" | "metre" | "metres" => "m",
        "km" | "kilometer" | "kilometers" | "kilometre" | "kilometres" => "km",
        "cm" | "centimeter" | "centimeters" | "centimetre" | "centimetres" => "cm",
        "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => "mm",
        "um" | "micrometer" | "micrometers" | "micrometre" | "micrometres" | "micron"
        | "microns" => "um",
        "nm" | "nanometer" | "nanometers" | "nanometre" | "nanometres" => "nm",
        "angstrom" | "angstroms" | "ang" => "Angstrom",
        "au" => "AU",
        "rad" | "radian" | "radians" => "rad",
        "deg" | "degree" | "degrees" => "deg",
        "arcmin" | "arcmins" | "amin" => "arcmin",
        "arcsec" | "arcsecs" | "asec" | "as" => "arcsec",
        "mas" => "mas",
        // Not a Table 5 symbol at all; micro- may not prefix `arcsec`
        // and `mas` is tabulated whole, so spell it as a numeric
        // multiple of the latter.
        "uas" => "0.001mas",
        "s" | "sec" | "second" | "seconds" => "s",
        "msec" => "ms",
        "usec" => "us",
        "min" | "minute" | "minutes" => "min",
        "h" | "hr" | "hour" | "hours" => "h",
        "d" | "day" | "days" => "d",
        "yr" | "year" | "years" => "yr",
        "hz" | "hertz" => "Hz",
        "khz" => "kHz",
        "mhz" => "MHz",
        "ghz" => "GHz",
        "thz" => "THz",
        "jy" | "jansky" | "janskys" => "Jy",
        "mjy" => "mJy",
        "ujy" => "uJy",
        // Detector spellings: `BUNIT = 'COUNTS'` and friends. `DN` is
        // the data number, which is the ADU by another name.
        "count" | "counts" | "cts" => "count",
        "photon" | "photons" => "photon",
        "adu" | "dn" => "adu",
        "pixel" | "pixels" | "px" => "pixel",
        "erg" | "ergs" => "erg",
        "kelvin" | "kelvins" => "K",
        "gauss" => "G",
        _ => return None,
    })
}

/// Rewrite the informal spellings in `s` onto their Table 4-5
/// equivalents, token by token, leaving everything else -- separators,
/// digits, unrecognized words -- exactly as written.
fn normalize_informal(s: &str) -> String {
    /// Append `token` to `out`, aliased if it is a known spelling.
    fn flush(out: &mut String, token: &mut String) {
        if token.is_empty() {
            return;
        }
        match legacy_alias(&token.to_ascii_lowercase()) {
            Some(canon) => out.push_str(canon),
            None => out.push_str(token),
        }
        token.clear();
    }
    let t = s.trim();
    // The two symbol spellings carry no alphabetic token to match.
    match t {
        "'" => return "arcmin".to_string(),
        "\"" => return "arcsec".to_string(),
        _ => {}
    }
    let mut out = String::with_capacity(t.len());
    let mut token = String::new();
    for c in t.chars() {
        if c.is_ascii_alphabetic() {
            token.push(c);
        } else {
            flush(&mut out, &mut token);
            out.push(c);
        }
    }
    flush(&mut out, &mut token);
    out
}

/// [`parse_unit`], falling back to the informal spellings real headers
/// put in `[unit]` comment annotations: `degrees`, `sec`, `AU/day`,
/// `KM/S`, ...
///
/// The strict grammar wins where it applies, so a conforming string
/// keeps exactly its [`parse_unit`] meaning. On failure the error is
/// the strict parser's.
///
/// # Errors
///
/// [`crate::error::FitsError::Header`] if neither reading resolves.
pub fn parse_unit_lenient(s: &str) -> Result<Unit> {
    let strict = parse_unit(s);
    if strict.is_ok() {
        return strict;
    }
    let normalized = normalize_informal(s);
    if normalized != s.trim()
        && let Ok(q) = parse_unit(&normalized)
    {
        return Ok(q);
    }
    strict
}

/// [`factor_to`] with the [`parse_unit_lenient`] fallback.
///
/// Also retries the aliases when the strict reading exists but carries
/// the wrong dimension: `TIMEUNIT = 'S'` is siemens to the grammar and
/// seconds to the files that write it, and `canonical` is what tells
/// the two apart.
///
/// # Errors
///
/// [`crate::error::FitsError::Header`] if neither reading has dimension `canonical`.
pub fn factor_to_lenient(unit: &str, canonical: Dimension) -> Result<f64> {
    let strict = factor_to(unit, canonical);
    if strict.is_ok() {
        return strict;
    }
    let normalized = normalize_informal(unit);
    if normalized != unit.trim()
        && let Ok(q) = parse_unit(&normalized)
        && let Ok(c) = q.converter_to(Unit::new(1.0, canonical))
        && let Some(f) = c.as_factor()
    {
        return Ok(f);
    }
    strict
}
