//! The Sec.4.3 recursive-descent parser.

use super::dimension::Dimension;
use super::level::{Level, Reference};
use super::table::resolve_symbol;
use super::unit::Unit;
use crate::error::{FitsError, Result};

// -- parser -------------------------------------------------------------

/// Deepest `parse_power` activation the parser will enter. Recursion
/// (glued juxtaposition, nested parentheses) costs native stack, and a
/// stack overflow aborts the process rather than unwinding into an
/// `Err`; real unit strings nest a handful of levels at most.
const MAX_DEPTH: usize = 64;

struct Parser<'a> {
    src: &'a str,
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            pos: 0,
            depth: 0,
        }
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn skip_spaces(&mut self) -> bool {
        let start = self.pos;
        while self.rest().starts_with(' ') {
            self.pos += 1;
        }
        self.pos != start
    }

    fn eat(&mut self, tok: &str) -> bool {
        if self.rest().starts_with(tok) {
            self.pos += tok.len();
            true
        } else {
            false
        }
    }

    fn err(&self, msg: &str) -> FitsError {
        FitsError::Header(format!("unit `{}`: {msg} at offset {}", self.src, self.pos))
    }

    /// `unit := power (op power)*`, where `op` is `*`, `.`, `/` or a
    /// space. Multiplication and division share precedence and
    /// associate left to right.
    fn parse_unit(&mut self) -> Result<Unit> {
        // A leading `/` means an implicit 1: Sec.4.3 gives `/m3` as a
        // spelling of "per meter cubed".
        self.skip_spaces();
        let mut acc = if self.rest().starts_with('/') {
            Unit::new(1.0, Dimension::NONE)
        } else {
            self.parse_power()?
        };
        loop {
            let had_space = self.skip_spaces();
            if self.eat("/") {
                self.skip_spaces();
                // A numeric divisor binds the units juxtaposed after
                // it: `cts / 300 s` is a count per 300 seconds, not a
                // count per 300 times a second. Symbol divisors keep
                // the left-to-right rule -- only a number changes the
                // reading, because "per <number> <unit>" is how the
                // quantity `300 s` is written in prose.
                let numeric = self.rest().starts_with(|c: char| c.is_ascii_digit());
                let mut divisor = self.parse_power()?;
                if numeric {
                    while self.skip_spaces() && self.starts_operand() {
                        divisor = divisor.mul(self.parse_power()?)?;
                    }
                }
                acc = acc.div(divisor)?;
            } else if self.eat("*") || self.eat(".") {
                self.skip_spaces();
                acc = acc.mul(self.parse_power()?)?;
            } else if had_space && self.starts_operand() {
                // A space is multiplication (Table 6).
                acc = acc.mul(self.parse_power()?)?;
            } else {
                return Ok(acc);
            }
        }
    }

    fn starts_operand(&self) -> bool {
        self.rest()
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '(')
    }

    /// `power := atom (('**' | '^') expr | bare_expr)? operand?`
    ///
    /// Every recursion in the grammar passes through here -- glued
    /// juxtaposition directly, parenthesized groups via `parse_atom`
    /// -- so this is where the [`MAX_DEPTH`] cap turns runaway input
    /// into an error instead of a stack overflow.
    fn parse_power(&mut self) -> Result<Unit> {
        if self.depth >= MAX_DEPTH {
            return Err(self.err("expression nests too deeply"));
        }
        self.depth += 1;
        let result = self.parse_power_inner();
        self.depth -= 1;
        result
    }

    fn parse_power_inner(&mut self) -> Result<Unit> {
        let base = self.parse_atom()?;
        let raised = if self.eat("**") || self.eat("^") {
            base.powf(self.parse_exponent()?)?
        } else if self
            .rest()
            .starts_with(|c: char| c.is_ascii_digit() || c == '+' || c == '-' || c == '(')
        {
            // `str1expr`: an exponent glued straight onto the symbol, as
            // in `m2`, `m-3`, `m(3/2)`. A digit or sign here can only be
            // an exponent -- the syntax has no addition.
            base.powf(self.parse_exponent()?)?
        } else {
            base
        };
        // Sec.4.3.1 makes the `10**k` / `10^k` / `10+-k` multiplier a
        // *prefix* of the compound string, so it may abut it with no
        // separator: `10**(46)erg/s` is the section's own worked example,
        // as is `10-3m`. Juxtaposition is multiplication (Table 7).
        //
        // A letter here is unambiguous. `parse_atom` consumes every
        // letter of a symbol, so a symbol is never followed by one; only
        // a numeric factor or a parenthesized group can be, and neither
        // can continue into a letter.
        if self.rest().starts_with(|c: char| c.is_ascii_alphabetic()) {
            return raised.mul(self.parse_power()?);
        }
        Ok(raised)
    }

    /// An exponent: a signed integer, or a parenthesized decimal or
    /// integer ratio.
    fn parse_exponent(&mut self) -> Result<f64> {
        if self.eat("(") {
            // `find` rather than a byte-stepping loop: stepping lands
            // mid-character on multi-byte input and the next slice
            // panics, where a bad exponent should only be an error.
            let start = self.pos;
            let Some(len) = self.rest().find(')') else {
                self.pos = self.src.len();
                return Err(self.err("unclosed exponent"));
            };
            self.pos = start + len;
            let body = &self.src[start..self.pos];
            let _ = self.eat(")");
            return parse_exponent_text(body).ok_or_else(|| {
                FitsError::Header(format!("unit `{}`: bad exponent `{body}`", self.src))
            });
        }
        let start = self.pos;
        if self.rest().starts_with('+') || self.rest().starts_with('-') {
            self.pos += 1;
        }
        while self.rest().starts_with(|c: char| c.is_ascii_digit()) {
            self.pos += 1;
        }
        let text = &self.src[start..self.pos];
        // Bare (unparenthesized) exponents must be integers: Sec.4.3
        // states that three-halves may *not* be written `m1.5`. Without
        // this the trailing `.5` would be swallowed as a `.`
        // multiplication by the numeric factor 5. `m2.s` stays legal --
        // there the `.` is followed by a symbol, not a digit.
        if self.rest().starts_with('.')
            && self.src[self.pos + 1..].starts_with(|c: char| c.is_ascii_digit())
        {
            return Err(self
                .err("a fractional exponent must be parenthesized, as in `m(1.5)` or `m**(3/2)`"));
        }
        text.parse::<i32>()
            .ok()
            .map(f64::from)
            .ok_or_else(|| self.err("expected an integer exponent"))
    }

    /// `atom := '(' unit ')' | func '(' unit ')' | number | symbol`
    fn parse_atom(&mut self) -> Result<Unit> {
        self.skip_spaces();
        if self.eat("(") {
            let inner = self.parse_unit()?;
            self.skip_spaces();
            if !self.eat(")") {
                return Err(self.err("unclosed ("));
            }
            return Ok(inner);
        }
        // `mag(AB)`, `mag(ST)`, `mag(Vega)`. An extension beyond Table 6,
        // whose function list holds only `log`, `ln`, `exp` and `sqrt`;
        // accepted because it is `astropy`'s own spelling -- `str(u.ABmag)`
        // is literally `mag(AB)` -- so a zero point survives the crossing.
        if self.rest().starts_with("mag(") {
            self.pos += "mag(".len();
            let start = self.pos;
            while self.rest().starts_with(|c: char| c.is_ascii_alphanumeric()) {
                self.pos += 1;
            }
            let name = &self.src[start..self.pos];
            let level = match name {
                "AB" => Level::AB_MAG,
                "ST" => Level::ST_MAG,
                // Vega's zero point is passband-dependent, so it stays an
                // `Object` rather than becoming a `Zero` with a number.
                "Vega" => Level::VEGA_MAG,
                _ => {
                    return Err(
                        self.err("`mag()` takes a zero point: `mag(AB)`, `mag(ST)` or `mag(Vega)`")
                    );
                }
            };
            if !self.eat(")") {
                return Err(self.err("unclosed ("));
            }
            return Ok(Unit::level(level));
        }
        // Functions.
        for name in ["sqrt", "log", "ln", "exp"] {
            if self.rest().starts_with(name) && self.src[self.pos + name.len()..].starts_with('(') {
                self.pos += name.len();
                let _ = self.eat("(");
                let inner = self.parse_unit()?;
                self.skip_spaces();
                if !self.eat(")") {
                    return Err(self.err("unclosed ("));
                }
                if name == "sqrt" {
                    return inner.powf(0.5);
                }
                if name == "exp" {
                    return Err(FitsError::Header(format!(
                        "unit `{}`: `exp()` is not a unit",
                        self.src
                    )));
                }
                // `log(X)` and `ln(X)` are levels over `X`: two `log(Hz)`
                // agree, `log(kHz)` differs from `log(Hz)` by an additive
                // 3, and neither converts into `Hz`.
                //
                // Built directly rather than through `Unit::mul`: inside
                // the parentheses a numeric factor is part of the
                // argument -- `log(0.001)` is a level over 1e-3 -- where
                // a factor *next to* a level symbol scales its value.
                if inner.is_level() {
                    return Err(FitsError::Header(format!(
                        "unit `{}`: the argument of `{name}()` cannot itself be a level unit",
                        self.src
                    )));
                }
                let base = if name == "log" {
                    10.0
                } else {
                    std::f64::consts::E
                };
                return Ok(Unit {
                    scale: inner.scale,
                    dimension: inner.dimension,
                    level: Some(Level::log(base, 1.0, Reference::Unspecified)),
                });
            }
        }
        // A leading number is the optional `10**k`-style multiplier, or
        // a bare numeric factor.
        if self.rest().starts_with(|c: char| c.is_ascii_digit()) {
            let start = self.pos;
            while self
                .rest()
                .starts_with(|c: char| c.is_ascii_digit() || c == '.')
            {
                self.pos += 1;
            }
            let text = &self.src[start..self.pos];
            let value: f64 = text
                .parse()
                .map_err(|_| self.err("malformed numeric factor"))?;
            return Ok(Unit::new(value, Dimension::NONE));
        }
        // Otherwise a symbol: letters only. Case is significant.
        let start = self.pos;
        while self.rest().starts_with(|c: char| c.is_ascii_alphabetic()) {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(self.err("expected a unit symbol"));
        }
        resolve_symbol(&self.src[start..self.pos])
    }
}

/// Parse a bracketed exponent body: `2`, `-3`, `1.5`, `3/2`.
fn parse_exponent_text(body: &str) -> Option<f64> {
    let body = body.trim();
    if let Some((num, den)) = body.split_once('/') {
        let n: f64 = num.trim().parse().ok()?;
        let d: f64 = den.trim().parse().ok()?;
        if d == 0.0 {
            return None;
        }
        return Some(n / d);
    }
    body.parse().ok()
}

/// Extract the first `[unit]` token from a FITS inline comment
/// (Standard Sec.4.3.2).
///
/// The standard recommends the brackets sit at the start of the
/// comment; this matches the first pair wherever it appears, since
/// writers put them at either end.
#[must_use]
pub fn parse_comment_unit(comment: &str) -> Option<&str> {
    let start = comment.find('[')? + 1;
    let end = comment[start..].find(']')? + start;
    let unit = comment[start..end].trim();
    if unit.is_empty() { None } else { Some(unit) }
}

/// Parse a FITS unit string into a scale factor and its dimensions.
///
/// An empty or all-blank string is the *undefined* unit (`CUNIT`'s
/// default): it yields a dimensionless factor of 1, leaving the caller
/// to apply whatever the standard mandates for that keyword.
///
/// # Errors
///
/// [`FitsError::Header`] if the string is not valid Sec.4.3 syntax or
/// names a unit outside Tables 4-5.
pub fn parse_unit(s: &str) -> Result<Unit> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Ok(Unit::new(1.0, Dimension::NONE));
    }
    let mut p = Parser::new(trimmed);
    let q = p.parse_unit()?;
    p.skip_spaces();
    if p.pos != trimmed.len() {
        return Err(FitsError::Header(format!(
            "unit `{trimmed}`: trailing text `{}`",
            &trimmed[p.pos..]
        )));
    }
    Ok(q)
}
