//! Read-only views over the bytes of a header.
//!
//! # Purpose
//!
//! A [`Header`](super::Header) stores the 80-byte cards it holds. It
//! stores no parsed form of them. The views here read a keyword, a
//! value, a comment or commentary text from those bytes on demand.
//!
//! A view holds no state and writes nothing back. The write path
//! emits the stored bytes, so a card that these views cannot describe
//! is still written as it was read.
//!
//! # Design constraints
//!
//! A view returns a value for every card. A value field that matches
//! no standard type reads back as [`Value::Unparsed`], which holds
//! the original text. The parser and the validator apply the rules of
//! the Standard to the bytes. A read applies none of them.

use super::card::{self, CARD_SIZE, Card, CardKind};
use super::value::{self, Value};

/// One logical card, borrowed from the header that holds its bytes.
///
/// Each method parses `bytes` when it is called. A view stores no
/// parsed form of the card.
#[derive(Debug, Clone, Copy)]
pub struct CardView<'a> {
    bytes: &'a [u8],
}

impl<'a> CardView<'a> {
    /// Wrap `bytes`, which must be one or more whole 80-byte cards.
    ///
    /// Deliberately not public. A view can only be obtained from a
    /// header, so every card a caller can see and splice is one the
    /// header already holds -- either read from a file, and so
    /// reproduced as it was, or built through the encoders, which
    /// refuse a card they cannot write. There is no route by which a
    /// caller assembles card bytes of its own.
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        debug_assert!(
            bytes.len().is_multiple_of(CARD_SIZE) && !bytes.is_empty(),
            "a logical card is one or more whole 80-byte cards"
        );
        Self { bytes }
    }

    /// The bytes this card occupies. A continued string includes
    /// every `CONTINUE` card of its chain.
    ///
    /// The header writes these bytes for the card.
    #[must_use]
    pub fn raw(&self) -> &'a [u8] {
        self.bytes
    }

    /// How many 80-byte cards this logical card occupies.
    #[must_use]
    pub fn physical_cards(&self) -> usize {
        self.bytes.len() / CARD_SIZE
    }

    /// The leading physical card, parsed.
    ///
    /// This does not fail. A card that the strict rules reject still
    /// returns its keyword and its raw body.
    fn head(&self) -> Card {
        Card::parse_with(&self.bytes[..CARD_SIZE], 0, true).unwrap_or_else(|_| Card {
            keyword: String::new(),
            kind: CardKind::Commentary,
            body: Vec::new(),
        })
    }

    /// Keyword, trimmed and upper-cased. Empty for a blank-keyword
    /// commentary card. A `HIERARCH` name comes back whole.
    #[must_use]
    pub fn keyword(&self) -> String {
        card::keyword_of(&self.bytes[..CARD_SIZE], true)
    }

    /// True when this card's keyword is `keyword`.
    ///
    /// Compares the keyword field in place, so this costs no
    /// allocation. Case-sensitive: keywords are upper-case.
    #[must_use]
    pub fn has_keyword(&self, keyword: &str) -> bool {
        card::keyword_matches(&self.bytes[..CARD_SIZE], true, keyword)
    }

    /// Which of the Sec.4.1.2 card shapes this is.
    ///
    /// A `CONTINUE` card at the head of a logical card is an orphan.
    /// A `CONTINUE` card that attaches to a value card is part of
    /// that card. Sec.4.2.1.2 reads an orphan as commentary, and this
    /// method reports it as commentary.
    #[must_use]
    pub fn kind(&self) -> CardKind {
        match card::kind_of(&self.bytes[..CARD_SIZE], true) {
            CardKind::Continue => CardKind::Commentary,
            kind => kind,
        }
    }

    /// True for `COMMENT`, `HISTORY` or a blank keyword.
    #[must_use]
    pub fn is_commentary(&self) -> bool {
        matches!(self.kind(), CardKind::Commentary)
    }

    /// The value and inline comment together, reading the card and any
    /// `CONTINUE` chain once.
    ///
    /// `None` for a card that carries neither, which is a commentary
    /// card, an `END` card, or an orphaned `CONTINUE`.
    #[must_use]
    pub fn parsed(&self) -> Option<(Value, Option<String>)> {
        let head = self.head();
        if matches!(
            head.kind,
            CardKind::Commentary | CardKind::End | CardKind::Continue
        ) {
            return None;
        }
        let (mut value, mut comment) = value::parse_lenient(&head.keyword, &head.body);
        for card in self.continuations() {
            let (cont, part) = value::parse_lenient(&card.keyword, &card.body);
            // Sec.4.2.1.2: a string ending in `&` continues on the next
            // card. Join the chain into one value.
            if let Value::String(ref mut s) = value
                && s.ends_with('&')
                && let Value::String(text) = cont
            {
                s.pop();
                s.push_str(&text);
            }
            // A comment may ride on any card of the chain; the reader
            // joins the fragments with a single space.
            if let Some(part) = part {
                match comment.as_mut() {
                    Some(existing) => {
                        existing.push(' ');
                        existing.push_str(&part);
                    }
                    None => comment = Some(part),
                }
            }
        }
        Some((value, comment))
    }

    /// The parsed value, with any `CONTINUE` chain joined, or `None`
    /// for a commentary card.
    ///
    /// A value field that matches no standard type returns
    /// [`Value::Unparsed`], which holds the original text.
    #[must_use]
    pub fn value(&self) -> Option<Value> {
        self.parsed().map(|(value, _)| value)
    }

    /// The inline comment, with any `CONTINUE` fragments joined by a
    /// single space, or `None` when the card carries none.
    #[must_use]
    pub fn comment(&self) -> Option<String> {
        self.parsed().and_then(|(_, comment)| comment)
    }

    /// The free text of a commentary card, or `None` for a value card.
    #[must_use]
    pub fn commentary(&self) -> Option<String> {
        let head = self.head();
        if !matches!(head.kind, CardKind::Commentary | CardKind::Continue) {
            return None;
        }
        Some(value::sanitize_free_text(&head.body).trim_end().to_string())
    }

    /// The physical cards after the first, parsed.
    fn continuations(&self) -> impl Iterator<Item = Card> + '_ {
        self.bytes
            .as_chunks::<CARD_SIZE>()
            .0
            .iter()
            .skip(1)
            .filter_map(|c| Card::parse_with(c, 0, true).ok())
    }
}
