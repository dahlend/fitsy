//! Value parser for value cards (Standard Sec.4.2).

use crate::error::{FitsError, Result};

/// Parsed value of a value card.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `T` or `F` (Sec.4.2.2).
    Logical(bool),
    /// Decimal integer (Sec.4.2.3).
    Integer(i64),
    /// Floating point (Sec.4.2.4). `D` exponent normalized to `E`.
    Real(f64),
    /// Complex integer `(re, im)` (Sec.4.2.5).
    ComplexInteger(i64, i64),
    /// Complex floating-point `(re, im)` (Sec.4.2.6).
    ComplexReal(f64, f64),
    /// String literal (Sec.4.2.1) with quote escapes resolved and
    /// trailing spaces trimmed.
    String(String),
    /// Empty value field (Sec.4.2.7).
    Undefined,
}

// Ergonomic constructors. FITS has no narrower integer or float type,
// so all integers map to `Integer(i64)` and all floats to `Real(f64)`.
// `From<&str>` and `From<String>` produce string-valued cards; users
// who want an "undefined" card should construct `Value::Undefined`
// explicitly.

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Self::Logical(b)
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Self::Integer(i)
    }
}

impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Self::Integer(i64::from(i))
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Self::Real(f)
    }
}

impl From<f32> for Value {
    fn from(f: f32) -> Self {
        Self::Real(f64::from(f))
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

/// Component returned by [`split_value_and_comment`].
#[derive(Debug, Clone)]
pub struct ValueAndComment {
    pub value_field: String,
    pub comment: Option<String>,
}

/// Parse the value-and-comment portion of a value card (bytes 11..80).
pub fn parse(keyword: &str, body: &[u8]) -> Result<(Value, Option<String>)> {
    let parts = split_value_and_comment(body, keyword)?;
    let val = parse_value(&parts.value_field, keyword)?;
    Ok((val, parts.comment))
}

/// Split the body into value field and optional comment, respecting
/// the rule that a `/` inside a string literal is not a comment marker
/// (Sec.4.1.2.3).
pub fn split_value_and_comment(body: &[u8], keyword: &str) -> Result<ValueAndComment> {
    let s = std::str::from_utf8(body).map_err(|_| FitsError::Value {
        keyword: keyword.into(),
        msg: "non-UTF-8 body".into(),
    })?;

    // Skip leading spaces; remember offset for parser.
    let mut chars = s.char_indices().peekable();
    let mut value_end = s.len();
    let mut comment: Option<String> = None;

    // Detect the value type by the first non-space char to know whether
    // we are inside a string literal.
    let first_non_space = chars.find(|&(_, c)| c != ' ');

    match first_non_space {
        None => {
            return Ok(ValueAndComment {
                value_field: String::new(),
                comment: None,
            });
        }
        Some((_, '\'')) => {
            // Walk the string literal handling `''` escapes.
            let mut iter = s.char_indices().skip_while(|&(_, c)| c == ' ');
            // consume opening quote
            let _ = iter.next();
            let mut close_idx = None;
            while let Some((i, c)) = iter.next() {
                if c == '\'' {
                    // Possible escape: peek next.
                    let next = iter.clone().next();
                    if let Some((_, '\'')) = next {
                        let _ = iter.next();
                        continue;
                    }
                    close_idx = Some(i + c.len_utf8());
                    break;
                }
            }
            let close_idx = close_idx.ok_or_else(|| FitsError::Value {
                keyword: keyword.into(),
                msg: "unterminated string literal".into(),
            })?;
            value_end = close_idx;
            // Look for `/` after value end.
            let rest = &s[close_idx..];
            if let Some(slash_pos) = rest.find('/') {
                comment = Some(rest[slash_pos + 1..].trim().to_string());
            }
        }
        Some(_) => {
            // Non-string scalar. The first `/` ends the value.
            if let Some(slash_pos) = s.find('/') {
                value_end = slash_pos;
                comment = Some(s[slash_pos + 1..].trim().to_string());
            }
        }
    }

    Ok(ValueAndComment {
        value_field: s[..value_end].to_string(),
        comment,
    })
}

fn parse_value(field: &str, keyword: &str) -> Result<Value> {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return Ok(Value::Undefined);
    }

    // String literal.
    if trimmed.starts_with('\'') {
        return parse_string_literal(trimmed, keyword);
    }

    // Logical: Sec.4.2.2 says the value `T`/`F` appears in column 30.
    // We accept it as a single character anywhere in the value field.
    if trimmed == "T" {
        return Ok(Value::Logical(true));
    }
    if trimmed == "F" {
        return Ok(Value::Logical(false));
    }

    // Complex: parenthesised pair, e.g. `(1.0, 2.0)`.
    if let Some(inner) = trimmed.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        let mut parts = inner.split(',');
        let re = parts.next().ok_or_else(|| FitsError::Value {
            keyword: keyword.into(),
            msg: "complex value missing real part".into(),
        })?;
        let im = parts.next().ok_or_else(|| FitsError::Value {
            keyword: keyword.into(),
            msg: "complex value missing imaginary part".into(),
        })?;
        if parts.next().is_some() {
            return Err(FitsError::Value {
                keyword: keyword.into(),
                msg: "complex value has more than two components".into(),
            });
        }
        let re_t = re.trim();
        let im_t = im.trim();
        if let (Ok(r), Ok(i)) = (re_t.parse::<i64>(), im_t.parse::<i64>()) {
            return Ok(Value::ComplexInteger(r, i));
        }
        let r = parse_real(re_t).ok_or_else(|| FitsError::Value {
            keyword: keyword.into(),
            msg: format!("invalid real `{re_t}` in complex value"),
        })?;
        let i = parse_real(im_t).ok_or_else(|| FitsError::Value {
            keyword: keyword.into(),
            msg: format!("invalid real `{im_t}` in complex value"),
        })?;
        return Ok(Value::ComplexReal(r, i));
    }

    // Integer (no `.`, no `E`/`D`).
    if !trimmed
        .chars()
        .any(|c| matches!(c, '.' | 'e' | 'E' | 'd' | 'D'))
        && let Ok(i) = trimmed.parse::<i64>()
    {
        return Ok(Value::Integer(i));
    }

    if let Some(r) = parse_real(trimmed) {
        return Ok(Value::Real(r));
    }

    Err(FitsError::Value {
        keyword: keyword.into(),
        msg: format!("unrecognized value `{trimmed}`"),
    })
}

fn parse_real(s: &str) -> Option<f64> {
    // Sec.4.2.4: D exponent is permitted; replace with E for `f64::parse`.
    let normalized: String = s
        .chars()
        .map(|c| match c {
            'd' => 'e',
            'D' => 'E',
            other => other,
        })
        .collect();
    normalized.parse::<f64>().ok()
}

fn parse_string_literal(s: &str, keyword: &str) -> Result<Value> {
    debug_assert!(
        s.starts_with('\''),
        "string literal must start with a single quote; got {s:?}"
    );
    // Walk and resolve `''` escapes; the first unescaped `'` ends.
    let mut out = String::new();
    let bytes = s.as_bytes();
    // Skip the opening quote.
    let mut i = 1;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\'' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                out.push('\'');
                i += 2;
                continue;
            }
            // Trim trailing spaces per Sec.4.2.1.1.
            let trimmed = out.trim_end_matches(' ').to_string();
            return Ok(Value::String(trimmed));
        }
        out.push(c as char);
        i += 1;
    }
    Err(FitsError::Value {
        keyword: keyword.into(),
        msg: "unterminated string literal".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer() {
        let (v, _) = parse(
            "BITPIX",
            b"                   16 / number of bits per pixel  ",
        )
        .unwrap();
        assert_eq!(v, Value::Integer(16));
    }

    #[test]
    fn negative_integer() {
        let (v, _) = parse("BITPIX", b"                  -64").unwrap();
        assert_eq!(v, Value::Integer(-64));
    }

    #[test]
    fn real_e_exponent() {
        let (v, _) = parse("X", b"             1.5E+02").unwrap();
        assert_eq!(v, Value::Real(150.0));
    }

    #[test]
    fn real_d_exponent_normalized() {
        let (v, _) = parse("X", b"          1.234D+05").unwrap();
        assert_eq!(v, Value::Real(123_400.0));
    }

    #[test]
    fn logical_true() {
        let (v, _) = parse("SIMPLE", b"                    T").unwrap();
        assert_eq!(v, Value::Logical(true));
    }

    #[test]
    fn string_basic() {
        let (v, _) = parse("OBJECT", b"'NGC 1234'").unwrap();
        assert_eq!(v, Value::String("NGC 1234".into()));
    }

    #[test]
    fn string_with_escaped_quote() {
        let (v, _) = parse("OBJECT", b"'O''Brien'").unwrap();
        assert_eq!(v, Value::String("O'Brien".into()));
    }

    #[test]
    fn string_trailing_spaces_trimmed() {
        let (v, _) = parse("X", b"'hello   '").unwrap();
        assert_eq!(v, Value::String("hello".into()));
    }

    #[test]
    fn slash_inside_string_not_comment() {
        let (v, c) = parse("X", b"'a/b' / a comment").unwrap();
        assert_eq!(v, Value::String("a/b".into()));
        assert_eq!(c.as_deref(), Some("a comment"));
    }

    #[test]
    fn complex_real() {
        let (v, _) = parse("X", b"(1.0, -2.5)").unwrap();
        assert_eq!(v, Value::ComplexReal(1.0, -2.5));
    }

    #[test]
    fn complex_integer() {
        let (v, _) = parse("X", b"(3, 4)").unwrap();
        assert_eq!(v, Value::ComplexInteger(3, 4));
    }

    #[test]
    fn undefined_value() {
        let (v, _) = parse("X", b"                       ").unwrap();
        assert_eq!(v, Value::Undefined);
    }

    #[test]
    fn comment_extracted() {
        let (_, c) = parse("X", b"                   16 / bits").unwrap();
        assert_eq!(c.as_deref(), Some("bits"));
    }
}
