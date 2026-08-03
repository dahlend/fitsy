//! Value parser for value cards (Standard Sec.4.2).

use crate::error::{FitsError, Result};

/// Parsed value of a value card.
///
/// # Examples
///
/// ```
/// use fitsy::{Header, Value};
///
/// let mut h = Header::empty();
/// h.push("NAXIS", 2_i64, None)?;
/// h.push("OBJECT", "M42", None)?;
/// h.push("SIMPLE", true, None)?;
///
/// assert_eq!(h.first("NAXIS"), Some(&Value::Integer(2)));
/// assert_eq!(h.first("SIMPLE"), Some(&Value::Logical(true)));
///
/// match h.first("OBJECT") {
///     Some(Value::String(s)) => assert_eq!(s, "M42"),
///     other => panic!("expected a string value, got {other:?}"),
/// }
/// # Ok::<(), fitsy::FitsError>(())
/// ```
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
    /// A value field that could not be parsed as any of the standard
    /// types. Only produced by lenient parsing (see
    /// [`Header::parse_with`](crate::header::Header::parse_with)); strict
    /// parsing rejects such cards. Holds the raw value-field text
    /// (trailing spaces trimmed) so the card can still be inspected and
    /// re-encoded verbatim.
    Unparsed(String),
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
    /// Bytes 11..80 up to the comment separator, trimmed.
    pub value_field: String,
    /// Text after the `/`, trimmed. `None` when there is no comment.
    pub comment: Option<String>,
}

/// Parse the value-and-comment part of a value card, bytes 11 to 80.
///
/// The `keyword` argument names the card, and appears in an error
/// message. The `body` argument holds those bytes 11 to 80.
///
/// # Errors
///
/// [`FitsError::Value`] when the value field cannot be split from its
/// comment, such as for an unterminated string literal, or when the
/// field matches no standard value type.
pub fn parse(keyword: &str, body: &[u8]) -> Result<(Value, Option<String>)> {
    let parts = split_value_and_comment(body, keyword)?;
    let val = parse_value(&parts.value_field, keyword)?;
    Ok((val, parts.comment))
}

/// Fault-tolerant form of [`parse`], used by lenient header parsing.
///
/// The `keyword` and `body` arguments carry the same meaning they do
/// in [`parse`].
///
/// This function does not fail. When the value field cannot be split
/// from its comment, or matches no standard type, the raw field text
/// survives as [`Value::Unparsed`], and the rest of the header still
/// loads. When the split succeeds and only the value fails to parse,
/// this keeps the comment.
#[must_use]
pub fn parse_lenient(keyword: &str, body: &[u8]) -> (Value, Option<String>) {
    let Ok(parts) = split_value_and_comment(body, keyword) else {
        // Even splitting failed (unterminated string, non-UTF-8). Keep the
        // whole field verbatim, minus trailing padding.
        let raw = String::from_utf8_lossy(body).trim_end().to_string();
        return (Value::Unparsed(raw), None);
    };
    match parse_value(&parts.value_field, keyword) {
        Ok(val) => (val, parts.comment),
        Err(_) => (
            Value::Unparsed(parts.value_field.trim().to_string()),
            parts.comment,
        ),
    }
}

/// Split the body into a value field and an optional comment.
///
/// The `body` argument holds bytes 11 to 80 of the card, and the
/// `keyword` argument names the card for an error message.
///
/// A `/` inside a string literal does not start a comment
/// (Sec.4.1.2.3), and this function honors that rule.
///
/// The scan runs over raw bytes and looks only for ASCII characters,
/// so a non-ASCII byte counts as content. A comment is free text, and
/// a later stage sanitizes it, so a stray byte there never fails the
/// parse. The value field comes back verbatim and must be ASCII.
///
/// # Errors
///
/// [`FitsError::Value`] when a string literal has no closing quote, or
/// when the value field holds a byte that is not ASCII.
/// [`parse_lenient`] turns either case into [`Value::Unparsed`].
pub fn split_value_and_comment(body: &[u8], keyword: &str) -> Result<ValueAndComment> {
    // Skip leading spaces; the first non-space byte tells us whether the
    // value is a string literal (which may contain a `/`).
    let Some(start) = body.iter().position(|&b| b != b' ') else {
        return Ok(ValueAndComment {
            value_field: String::new(),
            comment: None,
        });
    };

    let (value_end, comment_start): (usize, Option<usize>) = if body[start] == b'\'' {
        // Walk the string literal handling `''` escapes.
        let mut i = start + 1;
        let mut close = None;
        while i < body.len() {
            if body[i] == b'\'' {
                if body.get(i + 1) == Some(&b'\'') {
                    i += 2;
                    continue;
                }
                close = Some(i + 1);
                break;
            }
            i += 1;
        }
        let close = close.ok_or_else(|| FitsError::Value {
            keyword: keyword.into(),
            msg: "unterminated string literal".into(),
        })?;
        // A `/` after the closing quote starts the comment.
        let slash = body[close..].iter().position(|&b| b == b'/');
        (close, slash.map(|p| close + p))
    } else {
        // Non-string scalar: the first `/` ends the value.
        match body.iter().position(|&b| b == b'/') {
            Some(p) => (p, Some(p)),
            None => (body.len(), None),
        }
    };

    // A value field must be printable ASCII (Sec.4.2). Reject any byte
    // outside 0x20..=0x7E -- this covers Latin-1 bytes (invalid UTF-8) and
    // valid-UTF-8 multibyte sequences alike, so `'caf\u{e9}'` and its
    // UTF-8 encoding are both rejected in strict mode. In lenient mode the
    // card scanner has already sanitized these bytes to spaces, so this
    // check never fires there.
    let value_bytes = &body[..value_end];
    if let Some(pos) = value_bytes
        .iter()
        .position(|&b| !(0x20..=0x7E).contains(&b))
    {
        return Err(FitsError::Value {
            keyword: keyword.into(),
            msg: format!("non-ASCII byte 0x{:02X} in value field", value_bytes[pos]),
        });
    }
    // Every byte is printable ASCII (validated above), so a direct
    // byte-to-char map is a lossless, panic-free conversion.
    let value_field: String = value_bytes.iter().map(|&b| b as char).collect();
    // The comment text follows the slash; sanitize and trim it.
    let comment = comment_start.map(|slash| sanitize_comment(&body[slash + 1..]));

    Ok(ValueAndComment {
        value_field,
        comment,
    })
}

/// Map every byte outside printable ASCII (0x20..=0x7E) to a space,
/// leaving printable bytes untouched. Used to make free-text fields
/// (comments, commentary cards) tolerant of the Latin-1 and control
/// bytes that non-conforming writers routinely leak into them.
pub(crate) fn sanitize_free_text(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&b| {
            if (0x20..=0x7E).contains(&b) {
                b as char
            } else {
                ' '
            }
        })
        .collect()
}

/// Sanitize and trim an inline comment (the text after a value card's
/// `/` comment marker).
fn sanitize_comment(bytes: &[u8]) -> String {
    sanitize_free_text(bytes).trim().to_string()
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

    // Complex: parenthesized pair, e.g. `(1.0, 2.0)`.
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

    #[test]
    fn parse_lenient_keeps_bad_value_and_comment() {
        // A malformed number: value is unparseable, but the comment still
        // splits cleanly and is retained.
        let (v, c) = parse_lenient("EXPTIME", b"             12.3.4.5 / seconds");
        assert_eq!(v, Value::Unparsed("12.3.4.5".into()));
        assert_eq!(c.as_deref(), Some("seconds"));
    }

    #[test]
    fn parse_lenient_keeps_unterminated_string() {
        // Splitting fails outright; the whole field is kept verbatim.
        let (v, c) = parse_lenient("OBJECT", b"'M31");
        assert_eq!(v, Value::Unparsed("'M31".into()));
        assert_eq!(c, None);
    }

    #[test]
    fn parse_lenient_passes_through_valid_values() {
        let (v, _) = parse_lenient("BITPIX", b"                   16");
        assert_eq!(v, Value::Integer(16));
    }

    #[test]
    fn non_ascii_in_value_rejected_strict_both_encodings() {
        // A non-ASCII byte in a value field is rejected in strict mode
        // whether it is a bare Latin-1 byte (invalid UTF-8) or a valid
        // UTF-8 multibyte sequence.
        assert!(parse("OBSERVER", b"'caf\xe9'").is_err()); // Latin-1 e-acute
        assert!(parse("OBSERVER", b"'caf\xc3\xa9'").is_err()); // UTF-8 e-acute
    }
}
