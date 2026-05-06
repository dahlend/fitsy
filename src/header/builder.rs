//! Header construction and serialisation (Standard Sec.4).
//!
//! [`Header`] is parsed by [`Header::parse`]; this module adds the
//! complementary write side: builder methods to append cards
//! programmatically, and [`Header::to_bytes`] to render a valid,
//! block-padded byte stream that ends with `END`.
//!
//! Serialisation rules (Standard Sec.4):
//!
//! * Standard keywords up to 8 chars use the `KEYWORD = VALUE / COMMENT`
//!   form with the value indicator `"= "` in columns 9-10.
//! * Long string values (those that don't fit in 68 chars between the
//!   opening and closing quote) are emitted as a chain of `CONTINUE`
//!   cards (Sec.4.2.1.2). Each chunk except the last ends in `&`.
//! * `HIERARCH key1 key2 ...` cards place the `=` after the longest
//!   possible name; long-string `CONTINUE` for HIERARCH is not
//!   emitted (rare in practice).
//! * Commentary cards (`COMMENT`, `HISTORY`, blank keyword) carry up
//!   to 72 bytes of free text in columns 9-80; longer text is split
//!   across multiple cards verbatim (commentary is order-preserved).
//! * The output is always padded with ASCII spaces to the next
//!   2880-byte boundary (Standard Sec.3.1).

use crate::error::{FitsError, Result};
use crate::header::card::{CARD_SIZE, KEYWORD_LEN, VALUE_INDICATOR_LEN};
use crate::header::value::Value;
use crate::header::{Header, HeaderEntry};
use crate::io::block::BLOCK_SIZE;

/// Kind of a commentary card (Standard Sec.4.4.2.4-Sec.4.4.2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentaryKind {
    /// `COMMENT` keyword.
    Comment,
    /// `HISTORY` keyword.
    History,
    /// Blank (8-space) keyword.
    Blank,
}

impl CommentaryKind {
    fn keyword(self) -> &'static str {
        match self {
            Self::Comment => "COMMENT",
            Self::History => "HISTORY",
            Self::Blank => "",
        }
    }
}

impl Header {
    /// Empty header -- no cards, no `SIMPLE`, no `END`. Use the
    /// `push_*` methods to populate it; [`to_bytes`](Self::to_bytes)
    /// always appends an `END` card and pads to a 2880-byte block.
    #[must_use]
    pub fn empty() -> Self {
        // SAFETY-style note: the public constructor in the parent module
        // builds via `parse`. We mirror its zero-state here.
        Self::from_parts(Vec::new(), 0)
    }

    /// Append a value card.
    ///
    /// `keyword` may be a standard <= 8-char name (e.g. `"BITPIX"`)
    /// or an ESO HIERARCH long keyword in the form
    /// `"HIERARCH name1 name2 ..."`. If a card with the same keyword
    /// already exists, the new card is appended after it (i.e. the
    /// original first occurrence is returned by [`first`](Header::first)).
    pub fn push(
        &mut self,
        keyword: impl Into<String>,
        value: impl Into<Value>,
        comment: Option<&str>,
    ) -> Result<&mut Self> {
        let keyword = keyword.into();
        validate_keyword(&keyword)?;
        let entry = HeaderEntry {
            keyword,
            kind: crate::header::card::CardKind::Value,
            value: Some(value.into()),
            comment: comment.map(ToString::to_string),
            commentary: None,
        };
        self.append_entry(entry);
        Ok(self)
    }

    /// Append a commentary card. `text` may be any length; long text
    /// is split across multiple cards on serialisation.
    pub fn push_commentary(&mut self, kind: CommentaryKind, text: &str) -> &mut Self {
        let entry = HeaderEntry {
            keyword: kind.keyword().to_string(),
            kind: crate::header::card::CardKind::Commentary,
            value: None,
            comment: None,
            commentary: Some(text.to_string()),
        };
        self.append_entry(entry);
        self
    }

    /// Replace the value of the first existing entry with `keyword`,
    /// or append a new value card if none exists. Commentary cards
    /// are not affected. Returns `true` if an existing entry was
    /// updated, `false` if a new card was appended.
    pub fn set(
        &mut self,
        keyword: &str,
        value: impl Into<Value>,
        comment: Option<&str>,
    ) -> Result<bool> {
        validate_keyword(keyword)?;
        let value = value.into();
        if let Some(entry) = self.first_value_entry_mut(keyword) {
            entry.value = Some(value);
            if let Some(c) = comment {
                entry.comment = Some(c.to_string());
            }
            return Ok(true);
        }
        self.push(keyword.to_string(), value, comment)?;
        Ok(false)
    }

    /// Insert a value card at position `idx`, shifting subsequent
    /// cards right. If `idx` is past the end, the card is appended.
    pub fn insert(
        &mut self,
        idx: usize,
        keyword: impl Into<String>,
        value: impl Into<Value>,
        comment: Option<&str>,
    ) -> Result<&mut Self> {
        let keyword = keyword.into();
        validate_keyword(&keyword)?;
        let entry = HeaderEntry {
            keyword,
            kind: crate::header::card::CardKind::Value,
            value: Some(value.into()),
            comment: comment.map(ToString::to_string),
            commentary: None,
        };
        self.insert_entry(idx, entry);
        Ok(self)
    }

    /// Insert a value card immediately after the first card whose
    /// keyword equals `after`. Returns `Ok(false)` and appends at the
    /// end if no `after` card exists.
    pub fn set_after(
        &mut self,
        after: &str,
        keyword: impl Into<String>,
        value: impl Into<Value>,
        comment: Option<&str>,
    ) -> Result<bool> {
        let pos = self.first_value_index(after);
        let keyword = keyword.into();
        validate_keyword(&keyword)?;
        let entry = HeaderEntry {
            keyword,
            kind: crate::header::card::CardKind::Value,
            value: Some(value.into()),
            comment: comment.map(ToString::to_string),
            commentary: None,
        };
        if let Some(i) = pos {
            self.insert_entry(i + 1, entry);
            Ok(true)
        } else {
            self.append_entry(entry);
            Ok(false)
        }
    }

    /// Insert a value card immediately before the first card whose
    /// keyword equals `before`. Returns `Ok(false)` and appends at
    /// the end if no `before` card exists.
    pub fn set_before(
        &mut self,
        before: &str,
        keyword: impl Into<String>,
        value: impl Into<Value>,
        comment: Option<&str>,
    ) -> Result<bool> {
        let pos = self.first_value_index(before);
        let keyword = keyword.into();
        validate_keyword(&keyword)?;
        let entry = HeaderEntry {
            keyword,
            kind: crate::header::card::CardKind::Value,
            value: Some(value.into()),
            comment: comment.map(ToString::to_string),
            commentary: None,
        };
        if let Some(i) = pos {
            self.insert_entry(i, entry);
            Ok(true)
        } else {
            self.append_entry(entry);
            Ok(false)
        }
    }

    /// Rename every value card whose keyword equals `old` to use
    /// `new`. Returns the number of cards renamed.
    pub fn rename_keyword(&mut self, old: &str, new: &str) -> Result<usize> {
        validate_keyword(new)?;
        let mut count = 0_usize;
        for entry in self.cards_mut().iter_mut() {
            if matches!(entry.kind, crate::header::card::CardKind::Value) && entry.keyword == old {
                entry.keyword = new.to_string();
                count += 1;
            }
        }
        if count > 0 {
            self.rebuild_index();
        }
        Ok(count)
    }

    /// Render this header to padded bytes ending with `END` followed
    /// by ASCII spaces out to the next 2880-byte boundary.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out: Vec<u8> = Vec::with_capacity(BLOCK_SIZE);
        for entry in self.entries() {
            match entry.kind {
                crate::header::card::CardKind::Value => {
                    write_value_entry(&mut out, entry)?;
                }
                crate::header::card::CardKind::Commentary => {
                    write_commentary_entry(&mut out, entry)?;
                }
                crate::header::card::CardKind::Continue => {
                    // Should never appear in a built header -- parsed
                    // headers fold continuations into their parent
                    // string. If we see one, treat it as a stray
                    // commentary so we don't panic.
                    write_commentary_entry(&mut out, entry)?;
                }
                crate::header::card::CardKind::End => {
                    // Skip -- we always emit our own END card below.
                }
            }
        }
        // END card.
        let mut end = [b' '; CARD_SIZE];
        end[..3].copy_from_slice(b"END");
        out.extend_from_slice(&end);
        // Pad to block boundary with ASCII spaces.
        while !out.len().is_multiple_of(BLOCK_SIZE) {
            out.push(b' ');
        }
        Ok(out)
    }
}

// -- Card encoders ------------------------------------------------

fn write_value_entry(out: &mut Vec<u8>, entry: &HeaderEntry) -> Result<()> {
    let value = entry
        .value
        .as_ref()
        .ok_or_else(|| FitsError::Header(format!("value card `{}` has no value", entry.keyword)))?;

    if let Value::String(s) = value {
        write_string_value_with_continue(out, &entry.keyword, s, entry.comment.as_deref())?;
        return Ok(());
    }

    let value_field = format_scalar_value(value)?;
    let card = encode_single_value_card(&entry.keyword, &value_field, entry.comment.as_deref())?;
    out.extend_from_slice(&card);
    Ok(())
}

fn write_commentary_entry(out: &mut Vec<u8>, entry: &HeaderEntry) -> Result<()> {
    // 72 bytes of payload per card (columns 9..80).
    const PAYLOAD: usize = CARD_SIZE - KEYWORD_LEN;
    let text = entry.commentary.as_deref().unwrap_or("");
    let kw = entry.keyword.as_str();
    if kw.len() > KEYWORD_LEN {
        return Err(FitsError::Header(format!(
            "commentary keyword `{kw}` exceeds {KEYWORD_LEN} chars"
        )));
    }
    if text.is_empty() {
        let mut card = [b' '; CARD_SIZE];
        copy_keyword(&mut card, kw);
        out.extend_from_slice(&card);
        return Ok(());
    }
    // Split text into 72-byte chunks (columns 9..80).
    let bytes = text.as_bytes();
    for chunk in bytes.chunks(PAYLOAD) {
        for &b in chunk {
            if !is_ascii_text(b) {
                return Err(FitsError::Header(format!(
                    "commentary text contains non-printable byte 0x{b:02X}"
                )));
            }
        }
        let mut card = [b' '; CARD_SIZE];
        copy_keyword(&mut card, kw);
        card[KEYWORD_LEN..KEYWORD_LEN + chunk.len()].copy_from_slice(chunk);
        out.extend_from_slice(&card);
    }
    Ok(())
}

fn write_string_value_with_continue(
    out: &mut Vec<u8>,
    keyword: &str,
    s: &str,
    comment: Option<&str>,
) -> Result<()> {
    // Validate ASCII text up front; quote chars are escaped on emit.
    for &b in s.as_bytes() {
        if !is_ascii_text(b) {
            return Err(FitsError::Header(format!(
                "string value for `{keyword}` contains non-printable byte 0x{b:02X}"
            )));
        }
    }
    // Emit escaped form so we can reason about lengths in chars-as-bytes.
    let escaped = escape_string(s);

    // Available payload (chars between quotes) on the first card and
    // on each CONTINUE card. The first card's body starts at col 11,
    // a CONTINUE's body starts at col 9 -- but per Sec.4.2.1.2 CONTINUE
    // *also* uses the same "= " is replaced by two spaces, so it's
    // bytes 9..80 for both quotes and content.
    // Subtract 2 for the surrounding quote characters.
    let first_quote_budget = CARD_SIZE - keyword_body_offset(keyword)? - 2;
    // CONTINUE value field starts at column 11 (after the two-space
    // gap), runs to column 80; subtract two quote chars.
    let cont_quote_budget = CARD_SIZE - KEYWORD_LEN - VALUE_INDICATOR_LEN - 2;

    if is_hierarch(keyword) {
        // CONTINUE chaining is not standardised for HIERARCH. Either
        // the value fits on the single card (no 8-char minimum applies
        // to HIERARCH) or we hard-fail.
        if escaped.len() <= first_quote_budget {
            let value_field = format!("'{escaped}'");
            let card = encode_single_value_card(keyword, &value_field, comment)?;
            out.extend_from_slice(&card);
            return Ok(());
        }
        return Err(FitsError::Header(format!(
            "string value for HIERARCH keyword `{keyword}` is too long for one card \
             ({} chars); CONTINUE chaining is not emitted for HIERARCH",
            escaped.len()
        )));
    }

    // Try to fit on a single standard card first (no CONTINUE needed).
    let single_budget = if let Some(c) = comment {
        // Comment costs " / " + chars.
        first_quote_budget.saturating_sub(3 + c.len())
    } else {
        first_quote_budget
    };
    if escaped.len() <= single_budget {
        // Pad short strings to >= 8 chars between the quotes per
        // Sec.4.2.1.1 (standard keywords only; HIERARCH handled above).
        let mut padded = escaped.clone();
        while padded.len() < 8 {
            padded.push(' ');
        }
        let value_field = format!("'{padded}'");
        let card = encode_single_value_card(keyword, &value_field, comment)?;
        out.extend_from_slice(&card);
        return Ok(());
    }

    // Multi-card case. Reserve room for the trailing `&` on every
    // chunk except the last.
    let mut chunks: Vec<String> = Vec::new();
    let mut remaining = escaped.as_str();
    // Reserve one byte for the trailing `&` continuation marker.
    let first_chunk_budget = first_quote_budget.saturating_sub(1);
    let cont_chunk_budget = cont_quote_budget.saturating_sub(1);
    // First chunk.
    let take = char_floor(remaining, first_chunk_budget);
    chunks.push(remaining[..take].to_string());
    remaining = &remaining[take..];
    while !remaining.is_empty() {
        // For non-last chunks reserve `&`; for the last allow full budget.
        // We don't know if this is the last yet -- split greedily and
        // strip the `&` from the final chunk afterwards.
        let take = char_floor(remaining, cont_chunk_budget);
        chunks.push(remaining[..take].to_string());
        remaining = &remaining[take..];
    }

    // Try to fit a comment on the LAST card; if it doesn't fit, drop
    // the comment from this header rather than truncate it silently.
    // (Most callers don't put comments on long strings.)
    let last_idx = chunks.len() - 1;
    // Emit chunks with `&` on all but the last.
    for (i, chunk) in chunks.iter().enumerate() {
        let is_last = i == last_idx;
        let body = if is_last {
            format!("'{chunk}'")
        } else {
            format!("'{chunk}&'")
        };
        if i == 0 {
            let comment = if is_last { comment } else { None };
            let card = encode_single_value_card(keyword, &body, comment)?;
            out.extend_from_slice(&card);
        } else {
            let comment = if is_last { comment } else { None };
            let card = encode_continue_card(&body, comment)?;
            out.extend_from_slice(&card);
        }
    }
    Ok(())
}

fn encode_single_value_card(
    keyword: &str,
    value_field: &str,
    comment: Option<&str>,
) -> Result<[u8; CARD_SIZE]> {
    let body_offset = keyword_body_offset(keyword)?;
    let mut card = [b' '; CARD_SIZE];
    if is_hierarch(keyword) {
        // HIERARCH layout: `HIERARCH name1 name2 = value / comment`.
        let kw_bytes = keyword.as_bytes();
        if kw_bytes.len() + 3 + value_field.len() > CARD_SIZE {
            return Err(FitsError::Header(format!(
                "HIERARCH card for `{keyword}` would not fit in 80 bytes"
            )));
        }
        card[..kw_bytes.len()].copy_from_slice(kw_bytes);
        card[kw_bytes.len()] = b' ';
        card[kw_bytes.len() + 1] = b'=';
        card[kw_bytes.len() + 2] = b' ';
        let v_off = kw_bytes.len() + 3;
        let vb = value_field.as_bytes();
        validate_text(vb)?;
        card[v_off..v_off + vb.len()].copy_from_slice(vb);
        write_optional_comment(&mut card, v_off + vb.len(), comment)?;
    } else {
        copy_keyword(&mut card, keyword);
        card[KEYWORD_LEN] = b'=';
        card[KEYWORD_LEN + 1] = b' ';
        let vb = value_field.as_bytes();
        validate_text(vb)?;
        if body_offset + vb.len() > CARD_SIZE {
            return Err(FitsError::Header(format!(
                "value card for `{keyword}` does not fit in 80 bytes \
                 (value is {} chars)",
                vb.len()
            )));
        }
        card[body_offset..body_offset + vb.len()].copy_from_slice(vb);
        write_optional_comment(&mut card, body_offset + vb.len(), comment)?;
    }
    Ok(card)
}

fn encode_continue_card(value_field: &str, comment: Option<&str>) -> Result<[u8; CARD_SIZE]> {
    let mut card = [b' '; CARD_SIZE];
    card[..KEYWORD_LEN].copy_from_slice(b"CONTINUE");
    // Sec.4.2.1.2: CONTINUE has no value indicator; bytes 9..80 are
    // free-form value field. Conventional layout puts the string
    // literal starting at column 11 (i.e. two spaces of gap).
    let body_off = KEYWORD_LEN + VALUE_INDICATOR_LEN;
    let vb = value_field.as_bytes();
    validate_text(vb)?;
    if body_off + vb.len() > CARD_SIZE {
        return Err(FitsError::Header(
            "CONTINUE value field exceeds 70 bytes".into(),
        ));
    }
    card[body_off..body_off + vb.len()].copy_from_slice(vb);
    write_optional_comment(&mut card, body_off + vb.len(), comment)?;
    Ok(card)
}

fn write_optional_comment(
    card: &mut [u8; CARD_SIZE],
    cursor: usize,
    comment: Option<&str>,
) -> Result<()> {
    // FITS convention: the comment indicator `/` sits at column 32
    // (byte 31, 0-indexed) so that short string values align with
    // right-justified numeric values.  Advance `cursor` to the space
    // that precedes `/` when the value ends before that column.
    // If the value already occupies more than 30 bytes (e.g. a very
    // long real), we fall back to placing the comment right after it.
    // position of the space before `/`
    const COMMENT_START: usize = 30;

    let Some(c) = comment else {
        return Ok(());
    };
    if c.is_empty() {
        return Ok(());
    }
    let start = cursor.max(COMMENT_START);
    // Need " / " plus the comment text.
    let needed = 3 + c.len();
    if start + needed > CARD_SIZE {
        // Comment doesn't fit. Drop it silently -- losing a comment is
        // less bad than failing the whole serialize.
        return Ok(());
    }
    card[start] = b' ';
    card[start + 1] = b'/';
    card[start + 2] = b' ';
    let cb = c.as_bytes();
    validate_text(cb)?;
    card[start + 3..start + 3 + cb.len()].copy_from_slice(cb);
    Ok(())
}

fn copy_keyword(card: &mut [u8; CARD_SIZE], keyword: &str) {
    let kb = keyword.as_bytes();
    let n = kb.len().min(KEYWORD_LEN);
    card[..n].copy_from_slice(&kb[..n]);
}

fn keyword_body_offset(keyword: &str) -> Result<usize> {
    if is_hierarch(keyword) {
        // For length-budget purposes only; the actual offset depends
        // on the keyword length and is computed at emit time.
        Ok(keyword.len() + 3)
    } else if keyword.len() > KEYWORD_LEN {
        Err(FitsError::Header(format!(
            "keyword `{keyword}` exceeds {KEYWORD_LEN} chars (HIERARCH form required)"
        )))
    } else {
        Ok(KEYWORD_LEN + VALUE_INDICATOR_LEN)
    }
}

fn is_hierarch(keyword: &str) -> bool {
    keyword.starts_with("HIERARCH ")
}

fn validate_keyword(keyword: &str) -> Result<()> {
    if is_hierarch(keyword) {
        // Permit any printable ASCII (the parser already accepts these).
        for &b in keyword.as_bytes() {
            if !is_ascii_text(b) {
                return Err(FitsError::Header(format!(
                    "HIERARCH keyword contains non-printable byte 0x{b:02X}"
                )));
            }
        }
        return Ok(());
    }
    if keyword.is_empty() {
        return Err(FitsError::Header("keyword is empty".into()));
    }
    if keyword.len() > KEYWORD_LEN {
        return Err(FitsError::Header(format!(
            "keyword `{keyword}` exceeds {KEYWORD_LEN} chars"
        )));
    }
    for &b in keyword.as_bytes() {
        if !(b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'-' || b == b'_') {
            return Err(FitsError::Header(format!(
                "keyword `{keyword}` contains invalid byte 0x{b:02X}"
            )));
        }
    }
    Ok(())
}

#[inline]
fn is_ascii_text(b: u8) -> bool {
    (0x20..=0x7E).contains(&b)
}

fn validate_text(bytes: &[u8]) -> Result<()> {
    for &b in bytes {
        if !is_ascii_text(b) {
            return Err(FitsError::Header(format!(
                "header card contains non-printable byte 0x{b:02X}"
            )));
        }
    }
    Ok(())
}

fn escape_string(s: &str) -> String {
    // Sec.4.2.1.1: an embedded quote is escaped as `''`.
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c == '\'' {
            out.push('\'');
            out.push('\'');
        } else {
            out.push(c);
        }
    }
    out
}

/// Largest n <= `budget` such that `&s[..n]` is a char boundary.
fn char_floor(s: &str, budget: usize) -> usize {
    if budget >= s.len() {
        return s.len();
    }
    let mut n = budget;
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    n
}

fn format_scalar_value(v: &Value) -> Result<String> {
    Ok(match v {
        Value::Logical(b) => format!("{:>20}", if *b { "T" } else { "F" }),
        Value::Integer(i) => format!("{i:>20}"),
        Value::Real(r) => {
            if !r.is_finite() {
                return Err(FitsError::Header(format!(
                    "non-finite real `{r}` cannot appear in a FITS header"
                )));
            }
            format_real(*r)
        }
        Value::ComplexInteger(re, im) => format!("({re}, {im})"),
        Value::ComplexReal(re, im) => {
            if !re.is_finite() || !im.is_finite() {
                return Err(FitsError::Header(format!(
                    "non-finite complex value `({re}, {im})` cannot appear in a FITS header"
                )));
            }
            // Sec.4.2.4 requires a decimal point in every floating value;
            // ensure it survives a round-trip even for whole-number
            // components.
            let r = ensure_decimal(&format!("{re}"));
            let i = ensure_decimal(&format!("{im}"));
            format!("({r}, {i})")
        }
        Value::Undefined => String::new(),
        Value::String(_) => unreachable!("strings handled separately"),
    })
}

/// Render `r` for a fixed/free-format value field.
///
/// FITS Sec.4.2.4 requires a decimal point in every floating-point
/// value. Rust's `Display` for `f64` omits the trailing `.0` for
/// whole numbers (e.g. `1.0_f64 -> "1"`) and may emit exponential
/// forms without a decimal point (e.g. `1e100`). Both would round-trip
/// through the parser as `Integer`, silently corrupting the header.
/// We always re-insert a decimal point.
fn format_real(r: f64) -> String {
    let s = ensure_decimal(&format!("{r}"));
    if s.len() <= 20 {
        return format!("{s:>20}");
    }
    // Long form: explicit precision in `{:.NE}` always yields a
    // decimal point, so no further fix-up is needed.
    let long = format!("{r:.16E}");
    if long.len() <= 30 {
        long
    } else {
        format!("{r:.6E}")
    }
}

/// Insert `.0` into a numeric string that has no decimal point so it
/// parses back as a real, not an integer. Preserves any sign and
/// places the dot before the `e`/`E` exponent if present.
fn ensure_decimal(s: &str) -> String {
    if s.contains('.') {
        return s.to_string();
    }
    if let Some(pos) = s.find(['e', 'E']) {
        let mut out = String::with_capacity(s.len() + 2);
        out.push_str(&s[..pos]);
        out.push_str(".0");
        out.push_str(&s[pos..]);
        out
    } else {
        format!("{s}.0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_header_renders_to_one_block_with_end() {
        let h = Header::empty();
        let bytes = h.to_bytes().unwrap();
        assert_eq!(bytes.len(), BLOCK_SIZE);
        assert_eq!(&bytes[..3], b"END");
        // Everything after END must be spaces.
        assert!(bytes[3..].iter().all(|&b| b == b' '));
    }

    #[test]
    fn integer_card_round_trips() {
        let mut h = Header::empty();
        h.push("BITPIX", Value::Integer(16), None).unwrap();
        let bytes = h.to_bytes().unwrap();
        let (h2, _) = Header::parse(&bytes, 0).unwrap();
        assert_eq!(h2.first("BITPIX"), Some(&Value::Integer(16)));
    }

    #[test]
    fn string_card_round_trips() {
        let mut h = Header::empty();
        h.push("OBJECT", Value::String("M51".into()), Some("nice galaxy"))
            .unwrap();
        let bytes = h.to_bytes().unwrap();
        let (h2, _) = Header::parse(&bytes, 0).unwrap();
        match h2.first("OBJECT").unwrap() {
            // Sec.4.2.1.1: parsers may strip trailing spaces.
            Value::String(s) => assert!(s.starts_with("M51"), "got `{s}`"),
            other => panic!("not a string: {other:?}"),
        }
    }

    #[test]
    fn long_string_emits_continue_chain() {
        let long = "x".repeat(200);
        let mut h = Header::empty();
        h.push("OBJECT", Value::String(long.clone()), None).unwrap();
        let bytes = h.to_bytes().unwrap();
        // At least one CONTINUE card present.
        let n_continue = bytes
            .chunks_exact(CARD_SIZE)
            .filter(|c| c.starts_with(b"CONTINUE"))
            .count();
        assert!(
            n_continue >= 2,
            "expected >=2 CONTINUE cards, got {n_continue}"
        );
        let (h2, _) = Header::parse(&bytes, 0).unwrap();
        match h2.first("OBJECT").unwrap() {
            Value::String(s) => assert_eq!(s, &long),
            other => panic!("not a string: {other:?}"),
        }
    }

    #[test]
    fn commentary_round_trips_and_splits_long_text() {
        let mut h = Header::empty();
        // 150 chars > 72, so this should produce 3 HISTORY cards.
        let long = "y".repeat(150);
        h.push_commentary(CommentaryKind::History, &long);
        let bytes = h.to_bytes().unwrap();
        let n_history = bytes
            .chunks_exact(CARD_SIZE)
            .filter(|c| c.starts_with(b"HISTORY "))
            .count();
        assert_eq!(n_history, 3);
        let (h2, _) = Header::parse(&bytes, 0).unwrap();
        let joined: String = h2
            .entries()
            .iter()
            .filter(|e| e.keyword == "HISTORY")
            .filter_map(|e| e.commentary.clone())
            .collect();
        assert_eq!(joined, long);
    }

    #[test]
    fn logical_and_undefined_round_trip() {
        let mut h = Header::empty();
        h.push("SIMPLE", Value::Logical(true), None).unwrap();
        h.push("EMPTY", Value::Undefined, Some("absent")).unwrap();
        let bytes = h.to_bytes().unwrap();
        let (h2, _) = Header::parse(&bytes, 0).unwrap();
        assert_eq!(h2.first("SIMPLE"), Some(&Value::Logical(true)));
        assert_eq!(h2.first("EMPTY"), Some(&Value::Undefined));
    }

    #[test]
    fn rejects_lowercase_keyword() {
        let mut h = Header::empty();
        let err = h.push("naxis", Value::Integer(0), None).unwrap_err();
        assert!(matches!(err, FitsError::Header(_)));
    }

    #[test]
    fn rejects_nonfinite_real() {
        let mut h = Header::empty();
        h.push("BAD", Value::Real(f64::NAN), None).unwrap();
        assert!(h.to_bytes().is_err());
    }

    #[test]
    fn set_overwrites_existing() {
        let mut h = Header::empty();
        h.push("BITPIX", Value::Integer(8), None).unwrap();
        let updated = h.set("BITPIX", Value::Integer(16), None).unwrap();
        assert!(updated);
        let bytes = h.to_bytes().unwrap();
        let (h2, _) = Header::parse(&bytes, 0).unwrap();
        assert_eq!(h2.first("BITPIX"), Some(&Value::Integer(16)));
    }

    #[test]
    fn hierarch_round_trip() {
        let mut h = Header::empty();
        h.push(
            "HIERARCH ESO TEL ALT",
            Value::Real(45.5),
            Some("altitude in deg"),
        )
        .unwrap();
        let bytes = h.to_bytes().unwrap();
        let (h2, _) = Header::parse(&bytes, 0).unwrap();
        assert!(matches!(
            h2.first("HIERARCH ESO TEL ALT"),
            Some(Value::Real(_))
        ));
    }

    #[test]
    fn whole_number_real_round_trips_as_real() {
        // Sec.4.2.4: every floating-point value MUST contain a decimal
        // point. This regression test guards against `format_real`
        // emitting "1" for `1.0_f64`, which the parser would happily
        // re-classify as `Integer(1)` and silently break BSCALE,
        // BZERO, EQUINOX and many other commonly-real keywords.
        for r in [0.0_f64, 1.0, -1.0, 2000.0, 32768.0, 1.0e100, -1.0e-100] {
            let mut h = Header::empty();
            h.push("BSCALE", Value::Real(r), None).unwrap();
            let bytes = h.to_bytes().unwrap();
            let (h2, _) = Header::parse(&bytes, 0).unwrap();
            match h2.first("BSCALE") {
                Some(Value::Real(got)) => assert_eq!(*got, r, "wrong value"),
                other => panic!("BSCALE = {r} round-tripped as {other:?}, expected Real"),
            }
        }
    }

    #[test]
    fn whole_number_complex_real_round_trips_as_complex_real() {
        let mut h = Header::empty();
        h.push("ZVAL", Value::ComplexReal(1.0, 0.0), None).unwrap();
        let bytes = h.to_bytes().unwrap();
        let (h2, _) = Header::parse(&bytes, 0).unwrap();
        match h2.first("ZVAL") {
            Some(Value::ComplexReal(re, im)) => {
                assert_eq!(*re, 1.0);
                assert_eq!(*im, 0.0);
            }
            other => panic!("got {other:?}, expected ComplexReal"),
        }
    }

    #[test]
    fn ensure_decimal_inserts_dot_before_exponent() {
        assert_eq!(ensure_decimal("1"), "1.0");
        assert_eq!(ensure_decimal("-1"), "-1.0");
        assert_eq!(ensure_decimal("1e100"), "1.0e100");
        assert_eq!(ensure_decimal("-1E-7"), "-1.0E-7");
        assert_eq!(ensure_decimal("1.5"), "1.5");
        assert_eq!(ensure_decimal("1.5e10"), "1.5e10");
    }
}
