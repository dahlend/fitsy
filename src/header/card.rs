//! Card scanner (Standard Sec.4.1).

use crate::error::{FitsError, Result};

/// Length of a single FITS card in bytes (Sec.4.1).
pub const CARD_SIZE: usize = 80;

/// Length of the keyword name field (bytes 1-8 of a card).
pub const KEYWORD_LEN: usize = 8;

/// Length of the value indicator (`"= "`, bytes 9-10 of a value card).
pub const VALUE_INDICATOR_LEN: usize = 2;

/// Categorisation of a card by its keyword field.
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
pub struct Card {
    /// 0..=8 chars, ASCII upper/digit/`-`/`_`, or empty for blank.
    pub keyword: String,
    pub kind: CardKind,
    /// Raw bytes of the value-and-comment field (bytes 11..80) for
    /// value cards, or bytes 9..80 for commentary/`END`. Trailing
    /// spaces are kept.
    pub body: Vec<u8>,
}

impl Card {
    /// Parse a single 80-byte card. `offset` is the byte offset within
    /// the file (used in error messages).
    pub fn parse(bytes: &[u8], offset: u64) -> Result<Self> {
        if bytes.len() != CARD_SIZE {
            return Err(FitsError::Card {
                offset,
                msg: format!("expected {CARD_SIZE} bytes, got {}", bytes.len()),
            });
        }
        for (i, &b) in bytes.iter().enumerate() {
            if !is_ascii_text(b) {
                return Err(FitsError::Card {
                    offset: offset + i as u64,
                    msg: format!("non-ASCII-text byte 0x{b:02X}"),
                });
            }
        }

        let kw_field = &bytes[..KEYWORD_LEN];
        let keyword = parse_keyword_field(kw_field, offset)?;

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

        let has_value_indicator = bytes[KEYWORD_LEN] == b'=' && bytes[KEYWORD_LEN + 1] == b' ';

        let kind = if has_value_indicator {
            CardKind::Value
        } else {
            // Commentary or CONTINUE.
            if keyword == "CONTINUE" {
                CardKind::Continue
            } else {
                CardKind::Commentary
            }
        };

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
    pub fn is_end(&self) -> bool {
        matches!(self.kind, CardKind::End)
    }
}

#[inline]
fn is_ascii_text(b: u8) -> bool {
    // Standard Sec.3.2: ASCII text is 0x20..=0x7E.
    (0x20..=0x7E).contains(&b)
}

fn parse_keyword_field(field: &[u8], offset: u64) -> Result<String> {
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
    for (i, &b) in name.iter().enumerate() {
        if !is_keyword_char(b) {
            return Err(FitsError::Card {
                offset: offset + i as u64,
                msg: format!("invalid character 0x{b:02X} in keyword name"),
            });
        }
    }
    // Safe: validated to be ASCII subset.
    Ok(std::str::from_utf8(name)
        .expect("validated ASCII")
        .to_string())
}

#[inline]
fn is_keyword_char(b: u8) -> bool {
    b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'-' || b == b'_'
}

/// Parse a HIERARCH card per the ESO convention
/// (<https://fits.gsfc.nasa.gov/registry/hierarch_keyword.html>):
/// `HIERARCH key1 key2 ... keyN = value / comment`. The hierarchical
/// name is collapsed to a single space-separated string and stored as
/// the card's `keyword`. Subkeyword characters may include uppercase
/// letters, digits, hyphen, underscore, and period (`.`); we accept
/// any printable ASCII other than `=` to remain forgiving of
/// real-world headers.
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

/// Encode a card from its components into 80 bytes, padding with spaces.
/// `body` must not exceed `CARD_SIZE - body_offset` bytes.
pub fn encode(keyword: &str, kind: &CardKind, body: &[u8]) -> Result<[u8; CARD_SIZE]> {
    if keyword.len() > KEYWORD_LEN {
        return Err(FitsError::Card {
            offset: 0,
            msg: format!("keyword `{keyword}` exceeds {KEYWORD_LEN} chars"),
        });
    }
    for (i, b) in keyword.bytes().enumerate() {
        // Allow ASCII space inside non-empty keywords for the
        // HIERARCH form (`HIERARCH name1 name2 ...`); standard short
        // keywords don't contain spaces but the validator below
        // tolerates them so HIERARCH cards reuse this path.
        if !is_keyword_char(b) && b != b' ' {
            return Err(FitsError::Card {
                offset: i as u64,
                msg: format!("invalid keyword character 0x{b:02X}"),
            });
        }
    }
    let mut out = [b' '; CARD_SIZE];
    out[..keyword.len()].copy_from_slice(keyword.as_bytes());
    let body_start = match kind {
        CardKind::Value => {
            out[KEYWORD_LEN] = b'=';
            out[KEYWORD_LEN + 1] = b' ';
            KEYWORD_LEN + VALUE_INDICATOR_LEN
        }
        CardKind::End | CardKind::Commentary | CardKind::Continue => KEYWORD_LEN,
    };
    if body.len() > CARD_SIZE - body_start {
        return Err(FitsError::Card {
            offset: 0,
            msg: format!(
                "card body for `{keyword}` is {} bytes, max {}",
                body.len(),
                CARD_SIZE - body_start
            ),
        });
    }
    for &b in body {
        if !is_ascii_text(b) {
            return Err(FitsError::Card {
                offset: 0,
                msg: format!("non-ASCII-text byte 0x{b:02X} in card body"),
            });
        }
    }
    out[body_start..body_start + body.len()].copy_from_slice(body);
    Ok(out)
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
        let c = Card::parse(&raw, 0).unwrap();
        assert_eq!(c.keyword, "BITPIX");
        assert_eq!(c.kind, CardKind::Value);
    }

    #[test]
    fn parse_end_card() {
        let raw = make_card("END");
        let c = Card::parse(&raw, 0).unwrap();
        assert!(c.is_end());
    }

    #[test]
    fn end_with_garbage_rejected() {
        let mut raw = make_card("END");
        raw[10] = b'X';
        assert!(Card::parse(&raw, 0).is_err());
    }

    #[test]
    fn comment_card() {
        let raw = make_card("COMMENT this is a comment");
        let c = Card::parse(&raw, 0).unwrap();
        assert_eq!(c.keyword, "COMMENT");
        assert_eq!(c.kind, CardKind::Commentary);
    }

    #[test]
    fn hierarch_accepted() {
        let raw = make_card("HIERARCH ESO TEL ALT = 1.0");
        let c = Card::parse(&raw, 0).unwrap();
        assert_eq!(c.keyword, "HIERARCH ESO TEL ALT");
        assert_eq!(c.kind, CardKind::Value);
    }

    #[test]
    fn hierarch_collapses_runs_of_spaces() {
        let raw = make_card("HIERARCH ESO   TEL  ALT  = 1.0");
        let c = Card::parse(&raw, 0).unwrap();
        assert_eq!(c.keyword, "HIERARCH ESO TEL ALT");
    }

    #[test]
    fn hierarch_without_value_indicator_rejected() {
        let raw = make_card("HIERARCH ESO TEL ALT 1.0");
        assert!(Card::parse(&raw, 0).is_err());
    }

    #[test]
    fn lowercase_keyword_rejected() {
        let raw = make_card("bitpix  =                   16");
        assert!(Card::parse(&raw, 0).is_err());
    }

    #[test]
    fn blank_keyword_is_commentary() {
        let raw = make_card("        a free-form comment");
        let c = Card::parse(&raw, 0).unwrap();
        assert_eq!(c.keyword, "");
        assert_eq!(c.kind, CardKind::Commentary);
    }

    #[test]
    fn encode_round_trip() {
        let raw = encode("BITPIX", &CardKind::Value, b"                  16").unwrap();
        let c = Card::parse(&raw, 0).unwrap();
        assert_eq!(c.keyword, "BITPIX");
    }

    #[test]
    fn wrong_length_rejected() {
        let raw = vec![b' '; 79];
        assert!(Card::parse(&raw, 0).is_err());
    }
}
