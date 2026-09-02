//! Card scanner (Standard Sec.4.1).
//!
//! A FITS header is a sequence of 80-byte cards. This module splits
//! one card into its keyword, its kind and its raw body bytes, and it
//! encodes those three parts back into 80 bytes.
//!
//! [`Card::parse_with`] does the split. It classifies each card as
//! [`CardKind::End`], [`CardKind::Commentary`], [`CardKind::Continue`]
//! or [`CardKind::Value`], and it leaves the body unparsed. The
//! [`value`](crate::header::value) module parses that body.

use crate::error::{FitsError, Result};

/// Length of a single FITS card in bytes (Sec.4.1).
pub const CARD_SIZE: usize = 80;

/// Length of the keyword name field (bytes 1-8 of a card).
pub(crate) const KEYWORD_LEN: usize = 8;

/// Length of the value indicator (`"= "`, bytes 9-10 of a value card).
pub(crate) const VALUE_INDICATOR_LEN: usize = 2;

/// The kind of a card, decided from its keyword field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardKind {
    /// `END` card.
    End,
    /// Commentary card (`COMMENT`, `HISTORY`, blank keyword).
    Commentary,
    /// `CONTINUE` long-string continuation.
    Continue,
    /// Value card.
    Value,
}

/// A single 80-byte card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Card {
    /// Zero to eight characters of upper-case ASCII, digits, `-` or
    /// `_`. This is empty for a blank keyword.
    pub keyword: String,
    /// Which of the Sec.4.1.2 card shapes this is.
    pub kind: CardKind,
    /// Raw bytes of the value-and-comment field. A value card puts
    /// this in bytes 11 to 80. A commentary or `END` card puts it in
    /// bytes 9 to 80. This keeps every trailing space.
    pub body: Vec<u8>,
}

impl Card {
    /// Parse a single 80-byte card.
    ///
    /// The `bytes` argument must be exactly 80 bytes long. The
    /// `offset` argument is the byte offset of the card within the
    /// file, and it appears in an error message.
    ///
    /// A `lenient` value of `true` turns each non-ASCII byte into a
    /// space and folds a lower-case keyword letter to upper case, so a
    /// corrupted card still loads. A value of `false` rejects such a
    /// byte in the keyword field or the value field.
    ///
    /// A comment is lenient under either value. This scanner never
    /// rejects one, and a later stage sanitizes it.
    ///
    /// # Errors
    ///
    /// [`FitsError::Card`] when `bytes` is not exactly 80 bytes long,
    /// or when `lenient` is `false` and the keyword field holds a byte
    /// outside upper-case ASCII, digits, `-`, `_` and space.
    pub(crate) fn parse_with(bytes: &[u8], offset: u64, lenient: bool) -> Result<Self> {
        if bytes.len() != CARD_SIZE {
            return Err(FitsError::Card {
                offset,
                msg: format!("expected {CARD_SIZE} bytes, got {}", bytes.len()),
            });
        }
        // Sec.3.2 makes headers ASCII 0x20..=0x7E, so NUL is never
        // legal -- but writers leak it constantly, from fixed-width
        // buffers left zero-initialized. Map every NUL to a space so
        // those cards parse; in `lenient` mode do the same for any
        // non-ASCII byte.
        //
        // Non-ASCII is not rejected outright here. Binary garbage is
        // still caught downstream, in the keyword field, the END body
        // and the value field. What survives is a stray byte in a
        // free-text comment, which carries no structural meaning and
        // should not sink an otherwise-valid file.
        let card = sanitized(bytes, lenient);
        let bytes: &[u8] = &card;

        let kw_field = &bytes[..KEYWORD_LEN];
        let keyword = parse_keyword_field(kw_field, offset, lenient)?;

        // `parse_keyword_field` folds lower-case letters to upper case in
        // lenient mode, so this comparison also recognizes a lower-case or
        // mixed-case `end` terminator there. In strict mode a non-upper-case
        // keyword is rejected before reaching this point, so only the
        // conformant upper-case `END` matches.
        if keyword == "END" {
            // Sec.4.4.1.2: bytes 9-80 of the END card must be ASCII spaces.
            for (i, &b) in bytes[KEYWORD_LEN..].iter().enumerate() {
                if b != b' ' {
                    return Err(FitsError::Card {
                        offset: offset + (KEYWORD_LEN + i) as u64,
                        msg: "non-space byte in END card body".into(),
                    });
                }
            }
            return Ok(Self {
                keyword,
                kind: CardKind::End,
                body: Vec::new(),
            });
        }

        if keyword.starts_with("HIERARCH") {
            return parse_hierarch(bytes, offset);
        }

        // Sec.4.1.2.2: `= ` in bytes 9 and 10 marks a value field,
        // "unless it is one of the commentary keywords ... which by
        // definition have no value". Sec.4.4.2 repeats the rule: a
        // commentary keyword "shall have no associated value even if
        // the value indicator characters appear in bytes 9 and 10",
        // and bytes 9 through 80 are its text. So the test below must
        // not reach `COMMENT`, `HISTORY` or a blank keyword; for those
        // the `= ` is part of the commentary.
        let kind = classify(bytes, &keyword);

        let body_start = if matches!(kind, CardKind::Value) {
            KEYWORD_LEN + VALUE_INDICATOR_LEN
        } else {
            KEYWORD_LEN
        };

        Ok(Self {
            keyword,
            kind,
            body: bytes[body_start..].to_vec(),
        })
    }

    /// True if this is the `END` card.
    #[must_use]
    pub(crate) fn is_end(&self) -> bool {
        matches!(self.kind, CardKind::End)
    }
}

#[inline]
fn is_ascii_text(b: u8) -> bool {
    // Standard Sec.3.2: ASCII text is 0x20..=0x7E.
    (0x20..=0x7E).contains(&b)
}

fn parse_keyword_field(field: &[u8], offset: u64, lenient: bool) -> Result<String> {
    debug_assert_eq!(
        field.len(),
        KEYWORD_LEN,
        "keyword field must be exactly {KEYWORD_LEN} bytes"
    );
    // Permitted keyword characters per Sec.4.1.2.1: upper-case letters,
    // digits, hyphen, underscore. Trailing spaces pad a short name.
    // A blank-keyword commentary card has all eight bytes as spaces.
    let trimmed_end = field.iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);
    let name = &field[..trimmed_end];
    // Trailing spaces only -- interior spaces are not permitted.
    let mut out = String::with_capacity(name.len());
    for (i, &b) in name.iter().enumerate() {
        let Some(mapped) = keyword_byte(b, lenient) else {
            return Err(FitsError::Card {
                offset: offset + i as u64,
                msg: format!("invalid character 0x{b:02X} in keyword name"),
            });
        };
        out.push(mapped as char);
    }
    Ok(out)
}

/// One byte of a keyword name, as it appears in the parsed keyword,
/// or `None` when strict mode rejects it.
///
/// This is the single rule the full parse and the cheap comparison in
/// [`keyword_matches`] both follow, so the two can never disagree
/// about what a card is called.
#[inline]
fn keyword_byte(b: u8, lenient: bool) -> Option<u8> {
    if is_keyword_char(b) {
        Some(b)
    } else if lenient && b.is_ascii_lowercase() {
        // Fold a lower-case keyword (e.g. `exptime`) to upper case;
        // this is the most common real-world keyword defect and the
        // fold preserves the intended keyword semantics.
        Some(b.to_ascii_uppercase())
    } else if lenient {
        // Any other stray byte (an interior space left by an earlier
        // non-ASCII sanitization, punctuation, ...) becomes `_` so the
        // card still carries a usable keyword instead of aborting the
        // whole header.
        Some(b'_')
    } else {
        None
    }
}

/// Rewrite a card's bytes as the parser reads them: NUL becomes a
/// space always, and in lenient mode so does every non-ASCII byte.
///
/// Sec.3.2 makes headers ASCII 0x20..=0x7E, so NUL is never legal --
/// but writers leak it constantly, from fixed-width buffers left
/// zero-initialized.
fn sanitized(bytes: &[u8], lenient: bool) -> [u8; CARD_SIZE] {
    let mut card = [b' '; CARD_SIZE];
    let n = bytes.len().min(CARD_SIZE);
    card[..n].copy_from_slice(&bytes[..n]);
    for b in &mut card {
        if *b == 0 || (lenient && !is_ascii_text(*b)) {
            *b = b' ';
        }
    }
    card
}

/// Which shape a card has, given its sanitized bytes and the keyword
/// already read from them. `END` and `HIERARCH` are decided before
/// this point.
fn classify(card: &[u8], keyword: &str) -> CardKind {
    // Sec.4.1.2.2: `= ` in bytes 9 and 10 marks a value field,
    // "unless it is one of the commentary keywords ... which by
    // definition have no value". Sec.4.4.2 repeats the rule: a
    // commentary keyword "shall have no associated value even if
    // the value indicator characters appear in bytes 9 and 10",
    // and bytes 9 through 80 are its text. So the test below must
    // not reach `COMMENT`, `HISTORY` or a blank keyword; for those
    // the `= ` is part of the commentary.
    let is_commentary_keyword = matches!(keyword, "COMMENT" | "HISTORY" | "");
    if !is_commentary_keyword && card[KEYWORD_LEN] == b'=' && card[KEYWORD_LEN + 1] == b' ' {
        CardKind::Value
    } else if keyword == "CONTINUE" {
        CardKind::Continue
    } else {
        CardKind::Commentary
    }
}

/// Read a keyword of eight characters or fewer into `buf`, returning
/// its length. `None` when the field holds a byte strict mode rejects.
fn short_keyword(card: &[u8], lenient: bool, buf: &mut [u8; KEYWORD_LEN]) -> Option<usize> {
    let field = &card[..KEYWORD_LEN];
    let end = field.iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);
    for (i, &b) in field[..end].iter().enumerate() {
        buf[i] = keyword_byte(b, lenient)?;
    }
    Some(end)
}

/// The shape of the card at `bytes`, without copying its body.
///
/// Answers from the first ten bytes what [`Card::parse_with`] answers
/// from all eighty, for a reader that only needs to know what kind of
/// card this is. A `HIERARCH` name reaches past those ten bytes, so
/// that one case defers to the full parse.
pub(crate) fn kind_of(bytes: &[u8], lenient: bool) -> CardKind {
    let card = sanitized(bytes, lenient);
    let mut buf = [0_u8; KEYWORD_LEN];
    let Some(n) = short_keyword(&card, lenient, &mut buf) else {
        return full_kind(bytes, lenient);
    };
    let Ok(keyword) = std::str::from_utf8(&buf[..n]) else {
        return full_kind(bytes, lenient);
    };
    if keyword == "END" {
        // Sec.4.4.1.2: bytes 9-80 of an END card are spaces. A card
        // that breaks that fails the full parse, in either mode.
        return if card[KEYWORD_LEN..].iter().all(|&b| b == b' ') {
            CardKind::End
        } else {
            full_kind(bytes, lenient)
        };
    }
    if keyword == "HIERARCH" {
        return full_kind(bytes, lenient);
    }
    classify(&card, keyword)
}

/// The kind the full parse reports, or `Commentary` when it fails --
/// which is how a reader treats a card it cannot otherwise describe.
fn full_kind(bytes: &[u8], lenient: bool) -> CardKind {
    Card::parse_with(bytes, 0, lenient).map_or(CardKind::Commentary, |c| c.kind)
}

/// The name of the card at `bytes`.
///
/// Reads the keyword field in place and copies out only the name, so
/// this does not touch the card's body. A `HIERARCH` name reaches past
/// that field and defers to the full parse.
pub(crate) fn keyword_of(bytes: &[u8], lenient: bool) -> String {
    let card = sanitized(bytes, lenient);
    let mut buf = [0_u8; KEYWORD_LEN];
    let Some(n) = short_keyword(&card, lenient, &mut buf) else {
        return full_keyword(bytes, lenient);
    };
    if &buf[..n] == b"HIERARCH" {
        return full_keyword(bytes, lenient);
    }
    // `short_keyword` emits only keyword characters, which are ASCII.
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

/// The name the full parse reports, or the empty name when it fails --
/// which reads as a blank-keyword commentary card, matching
/// [`kind_of`].
fn full_keyword(bytes: &[u8], lenient: bool) -> String {
    Card::parse_with(bytes, 0, lenient).map_or_else(|_| String::new(), |c| c.keyword)
}

/// True when `k` is one of `prefixes` followed by one or more digits,
/// such as `NAXIS3` for `["NAXIS"]`.
///
/// FITS names a whole family of cards this way, and every place that
/// has to recognise one asks here rather than slicing the string
/// itself.
#[must_use]
pub(crate) fn is_indexed(k: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| {
        k.strip_prefix(p)
            .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
    })
}

/// True when the card at `bytes` is named `keyword`.
///
/// Compares the keyword field in place, so asking after a card's name
/// costs no allocation. A `HIERARCH` name lives past that field and
/// defers to the full parse.
pub(crate) fn keyword_matches(bytes: &[u8], lenient: bool, keyword: &str) -> bool {
    if keyword.len() > KEYWORD_LEN || keyword.starts_with("HIERARCH") {
        return Card::parse_with(bytes, 0, lenient).is_ok_and(|c| c.keyword == keyword);
    }
    let card = sanitized(bytes, lenient);
    let mut buf = [0_u8; KEYWORD_LEN];
    let Some(n) = short_keyword(&card, lenient, &mut buf) else {
        return false;
    };
    if &buf[..n] == b"HIERARCH" {
        return Card::parse_with(bytes, 0, lenient).is_ok_and(|c| c.keyword == keyword);
    }
    buf[..n] == *keyword.as_bytes()
}

#[inline]
fn is_keyword_char(b: u8) -> bool {
    b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'-' || b == b'_'
}

/// Parse a HIERARCH card per the ESO convention
/// (<https://fits.gsfc.nasa.gov/registry/hierarch_keyword.html>):
/// `HIERARCH key1 key2 ... keyN = value / comment`. The name is
/// collapsed to one space-separated string. The convention allows
/// letters, digits, hyphen, underscore and period; we accept any
/// printable ASCII but `=`, to stay forgiving of real headers.
fn parse_hierarch(bytes: &[u8], offset: u64) -> Result<Card> {
    debug_assert_eq!(
        bytes.len(),
        CARD_SIZE,
        "card buffer must be exactly {CARD_SIZE} bytes"
    );
    // Find the value-indicator `= ` after byte 9. The first eight
    // bytes are "HIERARCH"; byte 8 must be a space.
    if bytes[KEYWORD_LEN] != b' ' {
        return Err(FitsError::Card {
            offset: offset + KEYWORD_LEN as u64,
            msg: "HIERARCH card: expected space after HIERARCH".into(),
        });
    }
    let mut eq: Option<usize> = None;
    let mut i = KEYWORD_LEN + 1;
    while i + 1 < CARD_SIZE {
        if bytes[i] == b'=' && bytes[i + 1] == b' ' {
            eq = Some(i);
            break;
        }
        i += 1;
    }
    let eq = eq.ok_or_else(|| FitsError::Card {
        offset,
        msg: "HIERARCH card has no `= ` value indicator".into(),
    })?;
    let raw_name = &bytes[KEYWORD_LEN + 1..eq];
    // Collapse runs of spaces, validate characters.
    let mut name = String::new();
    let mut prev_space = true;
    for (k, &b) in raw_name.iter().enumerate() {
        if b == b' ' {
            if !prev_space {
                name.push(' ');
            }
            prev_space = true;
        } else {
            if !is_hierarch_char(b) {
                return Err(FitsError::Card {
                    offset: offset + (KEYWORD_LEN + 1 + k) as u64,
                    msg: format!("HIERARCH: invalid keyword byte 0x{b:02X}"),
                });
            }
            name.push(b as char);
            prev_space = false;
        }
    }
    while name.ends_with(' ') {
        name.pop();
    }
    if name.is_empty() {
        return Err(FitsError::Card {
            offset,
            msg: "HIERARCH card has empty keyword name".into(),
        });
    }
    Ok(Card {
        keyword: format!("HIERARCH {name}"),
        kind: CardKind::Value,
        body: bytes[eq + VALUE_INDICATOR_LEN..].to_vec(),
    })
}

#[inline]
fn is_hierarch_char(b: u8) -> bool {
    // Printable ASCII other than `=` (which would terminate the name)
    // and space (handled separately as a separator).
    matches!(b, 0x21..=0x7E) && b != b'='
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_card(s: &str) -> [u8; CARD_SIZE] {
        let mut b = [b' '; CARD_SIZE];
        let bytes = s.as_bytes();
        assert!(bytes.len() <= CARD_SIZE);
        b[..bytes.len()].copy_from_slice(bytes);
        b
    }

    #[test]
    fn parse_simple_value_card() {
        let raw = make_card("BITPIX  =                   16");
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert_eq!(c.keyword, "BITPIX");
        assert_eq!(c.kind, CardKind::Value);
    }

    #[test]
    fn parse_end_card() {
        let raw = make_card("END");
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert!(c.is_end());
    }

    #[test]
    fn end_with_garbage_rejected() {
        let mut raw = make_card("END");
        raw[10] = b'X';
        assert!(Card::parse_with(&raw, 0, false).is_err());
    }

    #[test]
    fn end_card_nul_padded_accepted() {
        // Some writers zero-fill the END card body instead of space-padding.
        let mut raw = make_card("END");
        for b in &mut raw[3..] {
            *b = 0;
        }
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert!(c.is_end());
    }

    #[test]
    fn value_card_trailing_nul_padding_accepted() {
        // Trailing NUL padding after the value is normalized to spaces.
        let mut raw = make_card("OBJECT  = 'M31'");
        for b in &mut raw[15..] {
            *b = 0;
        }
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert_eq!(c.keyword, "OBJECT");
        assert_eq!(c.kind, CardKind::Value);
    }

    #[test]
    fn all_nul_card_is_blank_commentary() {
        // A whole zero card (post-END fill) normalizes to a blank card.
        let raw = [0_u8; CARD_SIZE];
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert_eq!(c.keyword, "");
        assert_eq!(c.kind, CardKind::Commentary);
    }

    #[test]
    fn nul_padded_keyword_field_accepted() {
        // Keyword field zero-padded instead of space-padded: "OBJECT\0\0".
        let mut raw = make_card("OBJECT  = 'M31'");
        raw[6] = 0;
        raw[7] = 0;
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert_eq!(c.keyword, "OBJECT");
        assert_eq!(c.kind, CardKind::Value);
    }

    #[test]
    fn nul_inside_quoted_string_accepted() {
        // A fixed-width string value zero-padded inside the quotes:
        // "'M31\0\0\0\0\0'". The embedded NULs map to spaces.
        let mut raw = make_card("OBJECT  = 'M31     '");
        for b in &mut raw[13..18] {
            *b = 0;
        }
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert_eq!(c.keyword, "OBJECT");
        assert_eq!(c.kind, CardKind::Value);
    }

    #[test]
    fn non_nul_binary_garbage_still_rejected() {
        // A non-printable byte that is not NUL still errors -- this is how
        // a non-FITS / binary file is caught rather than silently parsed.
        let mut raw = make_card("OBJECT  = 'M31'");
        raw[2] = 0xFF;
        assert!(Card::parse_with(&raw, 0, false).is_err());
    }

    #[test]
    fn comment_card() {
        let raw = make_card("COMMENT this is a comment");
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert_eq!(c.keyword, "COMMENT");
        assert_eq!(c.kind, CardKind::Commentary);
    }

    #[test]
    fn hierarch_accepted() {
        let raw = make_card("HIERARCH ESO TEL ALT = 1.0");
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert_eq!(c.keyword, "HIERARCH ESO TEL ALT");
        assert_eq!(c.kind, CardKind::Value);
    }

    #[test]
    fn hierarch_collapses_runs_of_spaces() {
        let raw = make_card("HIERARCH ESO   TEL  ALT  = 1.0");
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert_eq!(c.keyword, "HIERARCH ESO TEL ALT");
    }

    /// Sec.4.1.2.2 and Sec.4.4.2: a commentary keyword has no value
    /// even when `= ` sits in bytes 9 and 10, and those bytes are then
    /// part of its text. Reading such a card as a value card made a
    /// strict parse fail outright, and hid the card from
    /// `Header::history` / `comments` in lenient mode.
    #[test]
    fn commentary_keyword_with_value_indicator_stays_commentary() {
        for (raw, keyword, text) in [
            (
                "HISTORY = looks like a value",
                "HISTORY",
                "= looks like a value",
            ),
            ("HISTORY = 'quoted text'", "HISTORY", "= 'quoted text'"),
            (
                "COMMENT = also value-shaped",
                "COMMENT",
                "= also value-shaped",
            ),
            (
                "        = blank keyword shaped",
                "",
                "= blank keyword shaped",
            ),
        ] {
            let c = Card::parse_with(&make_card(raw), 0, false).unwrap();
            assert_eq!(c.kind, CardKind::Commentary, "{raw}");
            assert_eq!(c.keyword, keyword, "{raw}");
            let body = String::from_utf8(c.body.clone()).unwrap();
            assert_eq!(body.trim_end(), text, "{raw}");
        }
    }

    /// The rule must not swallow a genuine value card whose keyword
    /// merely starts like a commentary one.
    #[test]
    fn non_commentary_keyword_keeps_its_value_indicator() {
        let c = Card::parse_with(&make_card("HISTORYX=                    1"), 0, false).unwrap();
        assert_eq!(c.kind, CardKind::Value);
    }

    #[test]
    fn hierarch_without_value_indicator_rejected() {
        let raw = make_card("HIERARCH ESO TEL ALT 1.0");
        assert!(Card::parse_with(&raw, 0, false).is_err());
    }

    #[test]
    fn lowercase_keyword_rejected() {
        let raw = make_card("bitpix  =                   16");
        assert!(Card::parse_with(&raw, 0, false).is_err());
    }

    #[test]
    fn blank_keyword_is_commentary() {
        let raw = make_card("        a free-form comment");
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert_eq!(c.keyword, "");
        assert_eq!(c.kind, CardKind::Commentary);
    }

    #[test]
    fn wrong_length_rejected() {
        let raw = vec![b' '; 79];
        assert!(Card::parse_with(&raw, 0, false).is_err());
    }

    #[test]
    fn non_ascii_in_body_accepted_at_card_layer() {
        // The card scanner no longer rejects a non-ASCII byte in the
        // value/comment area even in strict mode -- comment sanitization
        // happens downstream (see value::split_value_and_comment). A stray
        // byte in a keyword field is still caught (see the test below).
        let s = "EXPTIME =                 30.0 / temp in C";
        let mut raw = make_card(s);
        raw[s.rfind('C').unwrap()] = 0xB0; // Latin-1 degree sign in comment
        let c = Card::parse_with(&raw, 0, false).unwrap();
        assert_eq!(c.keyword, "EXPTIME");
        assert_eq!(c.kind, CardKind::Value);
    }

    #[test]
    fn non_ascii_in_keyword_field_still_rejected_strict() {
        let mut raw = make_card("OBJECT  = 'M31'");
        raw[2] = 0xB0; // inside the keyword name
        assert!(Card::parse_with(&raw, 0, false).is_err());
    }

    #[test]
    fn lowercase_keyword_folded_when_lenient() {
        let raw = make_card("exptime =                 30.0");
        assert!(Card::parse_with(&raw, 0, false).is_err());
        let c = Card::parse_with(&raw, 0, true).unwrap();
        assert_eq!(c.keyword, "EXPTIME");
        assert_eq!(c.kind, CardKind::Value);
    }

    #[test]
    fn interior_space_keyword_becomes_underscore_when_lenient() {
        let raw = make_card("CD1 1   =                  1.0");
        assert!(Card::parse_with(&raw, 0, false).is_err());
        let c = Card::parse_with(&raw, 0, true).unwrap();
        assert_eq!(c.keyword, "CD1_1");
    }

    #[test]
    fn lenient_still_rejects_wrong_length() {
        // Sanitizing content does not excuse a structurally broken card.
        let raw = vec![b' '; 79];
        assert!(Card::parse_with(&raw, 0, true).is_err());
    }
}
