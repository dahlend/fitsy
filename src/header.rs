//! FITS header parsing and construction (Standard Sec.4).
//!
//! The main types are:
//! - [`Header`]: a parsed header, accessed via [`Header::first`],
//!   [`Header::entries`], and [`Header::contains`].
//! - [`Value`]: the parsed value of a header card.
//! - [`Card`]: a single 80-byte card (keyword + value + comment).
//! - [`Header::push`] / [`Header::to_bytes`]: the write path.

pub mod builder;
pub mod card;
pub mod observatory;
pub mod reserved;
pub mod time;
pub mod units;
pub mod validation;
pub mod value;

pub use builder::CommentaryKind;
pub use card::{CARD_SIZE, Card, CardKind};
pub use observatory::{ObsGeo, ObsGeodetic};
pub use time::IsoDateTime;
pub use validation::{Diagnostic, Fix, Level};
pub use value::Value;

use std::collections::BTreeMap;

use crate::error::{FitsError, Result};
use crate::io::block::{BLOCK_SIZE, CARDS_PER_BLOCK};

/// A parsed FITS header (a sequence of value cards plus commentary).
#[derive(Debug, Clone)]
pub struct Header {
    cards: Vec<HeaderEntry>,
    /// Number of header blocks (each 2880 bytes) consumed in the file,
    /// including padding through the END card.
    blocks: usize,
    /// First-card index per keyword for O(1) lookup. Multiple entries
    /// with the same keyword (e.g. `COMMENT`, `HISTORY`) keep only the
    /// first index here; use `find_all` for the full set.
    index: BTreeMap<String, usize>,
}

/// One entry in a parsed header.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HeaderEntry {
    pub keyword: String,
    pub kind: CardKind,
    pub value: Option<Value>,
    pub comment: Option<String>,
    /// Raw body bytes for commentary cards.
    pub commentary: Option<String>,
}

impl Header {
    /// Parse a header starting at the given byte offset within `bytes`.
    /// Returns the header and the number of bytes consumed (a multiple
    /// of 2880).
    pub fn parse(bytes: &[u8], start: u64) -> Result<(Self, u64)> {
        let start_usize = start as usize;
        if start_usize > bytes.len() || !(bytes.len() - start_usize).is_multiple_of(BLOCK_SIZE) {
            return Err(FitsError::Block {
                offset: start,
                msg: "header start not block-aligned".into(),
            });
        }

        let mut cards = Vec::new();
        let mut continuations: Vec<usize> = Vec::new();
        let mut block_idx = 0_usize;
        let mut end_seen = false;
        // (block, card_in_block)
        let mut end_card_pos = (0_usize, 0_usize);

        'outer: while start_usize + (block_idx + 1) * BLOCK_SIZE <= bytes.len() {
            let block_start = start_usize + block_idx * BLOCK_SIZE;
            for c in 0..CARDS_PER_BLOCK {
                let off = block_start + c * CARD_SIZE;
                let raw = &bytes[off..off + CARD_SIZE];
                let card = Card::parse(raw, off as u64)?;

                if end_seen {
                    // Sec.4.4.1.2: every card after END must be all spaces.
                    if raw.iter().any(|&b| b != b' ') {
                        return Err(FitsError::EndCardMisplaced { offset: off as u64 });
                    }
                    continue;
                }

                match card.kind {
                    CardKind::End => {
                        end_seen = true;
                        end_card_pos = (block_idx, c);
                    }
                    CardKind::Continue => {
                        let idx = cards.len();
                        cards.push(card_to_entry(card, off as u64)?);
                        continuations.push(idx);
                    }
                    CardKind::Commentary | CardKind::Value => {
                        cards.push(card_to_entry(card, off as u64)?);
                    }
                }
            }
            block_idx += 1;
            if end_seen {
                break 'outer;
            }
        }

        if !end_seen {
            return Err(FitsError::Header("no END card found in header".into()));
        }

        // Resolve CONTINUE long-string concatenation (Sec.4.2.1.2).
        merge_continuations(&mut cards, &continuations)?;

        // Build index.
        let mut index = BTreeMap::new();
        for (i, e) in cards.iter().enumerate() {
            if !e.keyword.is_empty()
                && !matches!(e.kind, CardKind::Commentary)
                && !index.contains_key(&e.keyword)
            {
                index.insert(e.keyword.clone(), i);
            }
        }

        let blocks = block_idx;
        let consumed = (blocks * BLOCK_SIZE) as u64;
        // Reserved for future diagnostics.
        let _ = end_card_pos;
        Ok((
            Self {
                cards,
                blocks,
                index,
            },
            consumed,
        ))
    }

    /// All entries in order.
    #[must_use]
    pub fn entries(&self) -> &[HeaderEntry] {
        &self.cards
    }

    /// Number of header blocks consumed.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks
    }

    /// Return `keyword` with `_` and `-` swapped, for use as a fallback query
    /// key. Returns `None` when the keyword contains neither character.
    pub(crate) fn alt_key(keyword: &str) -> Option<String> {
        if !keyword.contains(['-', '_']) {
            return None;
        }
        Some(
            keyword
                .chars()
                .map(|c| match c {
                    '-' => '_',
                    '_' => '-',
                    c => c,
                })
                .collect(),
        )
    }

    /// Find the value of the first occurrence of `keyword`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use fitsy::{FitsError, FitsFile, Hdu, Value};
    ///
    /// let f = FitsFile::open("image.fits")?;
    /// let Hdu::Image(img) = f.hdu(0)? else {
    ///     return Err(FitsError::Header("HDU 0 is not an image".into()));
    /// };
    /// if let Some(Value::String(obj)) = img.header().first("OBJECT") {
    ///     println!("OBJECT = {obj}");
    /// }
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    #[must_use]
    pub fn first(&self, keyword: &str) -> Option<&Value> {
        if let Some(&idx) = self.index.get(keyword) {
            return self.cards[idx].value.as_ref();
        }
        // Some files use '_' where the standard uses '-' (e.g. MJD_OBS for MJD-OBS).
        let &idx = self.index.get(Self::alt_key(keyword)?.as_str())?;
        self.cards[idx].value.as_ref()
    }

    /// True if `keyword` is present (with any kind).
    #[must_use]
    pub fn contains(&self, keyword: &str) -> bool {
        if self.cards.iter().any(|e| e.keyword == keyword) {
            return true;
        }
        Self::alt_key(keyword).is_some_and(|alt| self.cards.iter().any(|e| e.keyword == alt))
    }

    /// Iterate over the body text of every `COMMENT` card in the
    /// order they appear (Sec.4.4.2.1). Returns an empty iterator if
    /// none are present.
    pub fn comments(&self) -> impl Iterator<Item = &str> {
        self.commentary_iter("COMMENT")
    }

    /// Iterate over the body text of every `HISTORY` card in the
    /// order they appear (Sec.4.4.2.2).
    pub fn history(&self) -> impl Iterator<Item = &str> {
        self.commentary_iter("HISTORY")
    }

    /// Iterate over the body text of every blank-keyword commentary
    /// card (Sec.4.1.2.3 -- eight spaces in the keyword field).
    pub fn blank_commentary(&self) -> impl Iterator<Item = &str> {
        self.commentary_iter("")
    }

    fn commentary_iter<'a>(&'a self, keyword: &'a str) -> impl Iterator<Item = &'a str> {
        self.cards.iter().filter_map(move |e| {
            if matches!(e.kind, CardKind::Commentary) && e.keyword == keyword {
                e.commentary.as_deref()
            } else {
                None
            }
        })
    }

    /// Internal constructor used by the builder API in
    /// [`crate::header::builder`].
    pub(crate) fn from_parts(cards: Vec<HeaderEntry>, blocks: usize) -> Self {
        let mut index = BTreeMap::new();
        for (i, e) in cards.iter().enumerate() {
            if !e.keyword.is_empty()
                && !matches!(e.kind, CardKind::Commentary)
                && !index.contains_key(&e.keyword)
            {
                index.insert(e.keyword.clone(), i);
            }
        }
        Self {
            cards,
            blocks,
            index,
        }
    }

    /// Append a single entry, updating the keyword index.
    pub(crate) fn append_entry(&mut self, entry: HeaderEntry) {
        let idx = self.cards.len();
        if !entry.keyword.is_empty()
            && !matches!(entry.kind, CardKind::Commentary)
            && !self.index.contains_key(&entry.keyword)
        {
            self.index.insert(entry.keyword.clone(), idx);
        }
        self.cards.push(entry);
    }

    /// Mutable handle to the first value entry whose keyword matches.
    pub(crate) fn first_value_entry_mut(&mut self, keyword: &str) -> Option<&mut HeaderEntry> {
        let idx = *self.index.get(keyword)?;
        self.cards.get_mut(idx)
    }

    /// Position of the first value card whose keyword matches, or
    /// `None` if no such card exists. Used by the positional
    /// insertion methods on the builder.
    pub(crate) fn first_value_index(&self, keyword: &str) -> Option<usize> {
        self.index.get(keyword).copied()
    }

    /// Insert `entry` at the given position, shifting subsequent
    /// cards right by one.
    pub(crate) fn insert_entry(&mut self, idx: usize, entry: HeaderEntry) {
        let i = idx.min(self.cards.len());
        self.cards.insert(i, entry);
        self.rebuild_index();
    }

    /// Mutable handle to all cards. Callers must call
    /// [`rebuild_index`](Self::rebuild_index) afterwards if they
    /// modify any keyword text.
    pub(crate) fn cards_mut(&mut self) -> &mut Vec<HeaderEntry> {
        &mut self.cards
    }

    /// Recompute the keyword -> first-occurrence index from scratch.
    pub(crate) fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, e) in self.cards.iter().enumerate() {
            if !e.keyword.is_empty()
                && !matches!(e.kind, CardKind::Commentary)
                && !self.index.contains_key(&e.keyword)
            {
                self.index.insert(e.keyword.clone(), i);
            }
        }
    }

    /// Remove every value card whose keyword equals `keyword` (case
    /// sensitive). Commentary cards (`COMMENT`, `HISTORY`, blank) are
    /// not affected. Returns the number of cards removed.
    pub fn remove(&mut self, keyword: &str) -> usize {
        let before = self.cards.len();
        self.cards
            .retain(|e| !(matches!(e.kind, CardKind::Value) && e.keyword == keyword));
        let removed = before - self.cards.len();
        if removed > 0 {
            self.rebuild_index();
        }
        removed
    }

    /// Append every value card from `parent` whose keyword is not
    /// already present in `self`. Implements the (Goddard / IRAF)
    /// `INHERIT` convention: an extension header that carries
    /// `INHERIT = T` is meant to inherit non-structural keywords from
    /// the primary HDU.
    ///
    /// Mandatory structural keywords (`SIMPLE`, `XTENSION`, `BITPIX`,
    /// `NAXIS`, `NAXISn`, `PCOUNT`, `GCOUNT`, `EXTEND`, `END`,
    /// `INHERIT` itself) and `CHECKSUM`/`DATASUM` are never inherited
    /// -- the extension must carry its own. Commentary cards
    /// (`COMMENT`/`HISTORY`/blank) are also not inherited to avoid
    /// duplicating provenance.
    pub fn merge_inherited(&mut self, parent: &Self) {
        for entry in &parent.cards {
            if !matches!(entry.kind, CardKind::Value) {
                continue;
            }
            if is_structural_keyword(&entry.keyword) {
                continue;
            }
            if self.contains(&entry.keyword) {
                continue;
            }
            self.append_entry(entry.clone());
        }
    }
}

fn is_structural_keyword(kw: &str) -> bool {
    if matches!(
        kw,
        "SIMPLE"
            | "XTENSION"
            | "BITPIX"
            | "NAXIS"
            | "PCOUNT"
            | "GCOUNT"
            | "EXTEND"
            | "END"
            | "INHERIT"
            | "CHECKSUM"
            | "DATASUM"
            | "TFIELDS"
            | "GROUPS"
            | "ZIMAGE"
            | "ZBITPIX"
            | "ZNAXIS"
            | "ZCMPTYPE"
    ) {
        return true;
    }
    if kw.starts_with("NAXIS") && kw[5..].chars().all(|c| c.is_ascii_digit()) && kw.len() > 5 {
        return true;
    }
    if kw.starts_with("ZNAXIS") && kw[6..].chars().all(|c| c.is_ascii_digit()) && kw.len() > 6 {
        return true;
    }
    if kw.starts_with("TFORM")
        || kw.starts_with("TTYPE")
        || kw.starts_with("TUNIT")
        || kw.starts_with("TBCOL")
        || kw.starts_with("TDIM")
        || kw.starts_with("TNULL")
        || kw.starts_with("TSCAL")
        || kw.starts_with("TZERO")
        || kw.starts_with("TDISP")
        || kw.starts_with("THEAP")
    {
        return true;
    }
    false
}

fn card_to_entry(card: Card, _offset: u64) -> Result<HeaderEntry> {
    match card.kind {
        CardKind::Value | CardKind::Continue => {
            let (val, comment) = value::parse(&card.keyword, &card.body)?;
            Ok(HeaderEntry {
                keyword: card.keyword,
                kind: card.kind,
                value: Some(val),
                comment,
                commentary: None,
            })
        }
        CardKind::Commentary => {
            let text = std::str::from_utf8(&card.body)
                .map_err(|_| FitsError::Header("non-UTF-8 commentary card".into()))?
                .trim_end()
                .to_string();
            Ok(HeaderEntry {
                keyword: card.keyword,
                kind: card.kind,
                value: None,
                comment: None,
                commentary: Some(text),
            })
        }
        CardKind::End => unreachable!("END handled by caller"),
    }
}

/// Resolve `CONTINUE` long-string concatenation. A string value ending
/// with `&` is continued by the next `CONTINUE` card whose value is
/// itself a string (Sec.4.2.1.2). We merge each chain into the parent
/// card and remove the `CONTINUE` entries.
fn merge_continuations(cards: &mut Vec<HeaderEntry>, continuations: &[usize]) -> Result<()> {
    if continuations.is_empty() {
        return Ok(());
    }
    let to_remove: std::collections::BTreeSet<usize> = continuations.iter().copied().collect();
    // Walk forwards: for each CONTINUE, find the most recent prior
    // value card whose String ends in `&`, and append.
    for &cont_idx in continuations {
        // Find parent: walk backwards skipping any CONTINUE entries.
        let mut parent_idx: Option<usize> = None;
        for back in (0..cont_idx).rev() {
            if matches!(cards[back].kind, CardKind::Continue) {
                continue;
            }
            parent_idx = Some(back);
            break;
        }
        let parent_idx = parent_idx.ok_or_else(|| {
            FitsError::Header("CONTINUE card without preceding value card".into())
        })?;

        // Mutate parent string.
        let cont_text = match &cards[cont_idx].value {
            Some(Value::String(s)) => s.clone(),
            _ => {
                return Err(FitsError::Header(
                    "CONTINUE card value is not a string".into(),
                ));
            }
        };
        let cont_comment = cards[cont_idx].comment.clone();
        let parent = &mut cards[parent_idx];
        let Some(Value::String(parent_str)) = parent.value.as_mut() else {
            return Err(FitsError::Header(
                "CONTINUE follows a non-string value".into(),
            ));
        };
        if !parent_str.ends_with('&') {
            return Err(FitsError::Header(
                "CONTINUE follows a string that does not end with `&`".into(),
            ));
        }
        // Drop the trailing `&` continuation marker.
        parent_str.pop();
        parent_str.push_str(&cont_text);
        if let Some(c) = cont_comment {
            match parent.comment.as_mut() {
                Some(existing) => {
                    existing.push(' ');
                    existing.push_str(&c);
                }
                None => parent.comment = Some(c),
            }
        }
    }

    // Drop all CONTINUE entries in one pass.
    let mut i = 0_usize;
    cards.retain(|_| {
        let keep = !to_remove.contains(&i);
        i += 1;
        keep
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header(cards: &[&str]) -> Vec<u8> {
        let mut buf = Vec::new();
        for c in cards {
            let mut card = [b' '; CARD_SIZE];
            let bytes = c.as_bytes();
            assert!(bytes.len() <= CARD_SIZE);
            card[..bytes.len()].copy_from_slice(bytes);
            buf.extend_from_slice(&card);
        }
        // Pad to next 2880 boundary with spaces.
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(b' ');
        }
        buf
    }

    #[test]
    fn minimal_simple_header() {
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "END",
        ]);
        let (h, consumed) = Header::parse(&bytes, 0).unwrap();
        assert_eq!(consumed, BLOCK_SIZE as u64);
        assert_eq!(h.bitpix().unwrap(), 8);
        assert_eq!(h.naxis().unwrap(), 0);
    }

    #[test]
    fn missing_end_rejected() {
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
        ]);
        assert!(Header::parse(&bytes, 0).is_err());
    }

    #[test]
    fn garbage_after_end_rejected() {
        let mut bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "END",
        ]);
        // Sneak a non-space after END.
        bytes[4 * CARD_SIZE] = b'X';
        assert!(Header::parse(&bytes, 0).is_err());
    }

    #[test]
    fn continue_long_string() {
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "OBJECT  = 'this is a long &'",
            "CONTINUE  'tail'",
            "END",
        ]);
        let (h, _) = Header::parse(&bytes, 0).unwrap();
        match h.first("OBJECT").unwrap() {
            Value::String(s) => assert_eq!(s, "this is a long tail"),
            other => panic!("not a string: {other:?}"),
        }
    }

    #[test]
    fn continue_long_string_three_cards() {
        // Standard Sec.4.2.1.2: chained CONTINUEs concatenate when each
        // string up to the last ends in `&`. The terminator can omit
        // the trailing `&`.
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "OBJECT  = 'first part &'",
            "CONTINUE  'middle part &'",
            "CONTINUE  'final piece'",
            "END",
        ]);
        let (h, _) = Header::parse(&bytes, 0).unwrap();
        match h.first("OBJECT").unwrap() {
            Value::String(s) => assert_eq!(s, "first part middle part final piece"),
            other => panic!("not a string: {other:?}"),
        }
    }

    #[test]
    fn remove_clears_value_and_rebuilds_index() {
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "OBJECT  = 'eta carinae'",
            "EXPTIME =                  1.5",
            "COMMENT  this is a note",
            "END",
        ]);
        let (mut h, _) = Header::parse(&bytes, 0).unwrap();
        assert_eq!(h.remove("OBJECT"), 1);
        assert!(!h.contains("OBJECT"));
        // After removal, the next lookup must still find later cards.
        assert!(h.contains("EXPTIME"));
        assert_eq!(h.remove("OBJECT"), 0);
        // Commentary cards are not touched.
        assert_eq!(h.comments().count(), 1);
    }
}
