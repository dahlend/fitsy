//! FITS header parsing and construction (Standard Sec.4).
//!
//! # Purpose
//!
//! A FITS header is a sequence of 80-byte cards. This module parses
//! that sequence into a [`Header`], reads values out of it, and
//! renders it back to bytes.
//!
//! [`Header`] stores the bytes of every card, in file order. It
//! stores no parsed form of them. It keeps a card that contradicts
//! another card, and a card that breaks the Standard.
//!
//! Interpretation happens elsewhere, in [`Wcs`](crate::Wcs) for a
//! coordinate description and in the [`reserved`] accessors for a
//! single typed keyword.
//!
//! # Layout
//!
//! Three types carry the data:
//!
//! - [`Header`] -- the cards of one header, as bytes. Read it with
//!   [`Header::first`], [`Header::cards`] and [`Header::contains`].
//! - [`CardView`] -- a read-only view of one logical card. It reads a
//!   keyword, a value or a comment from the bytes of that card.
//! - [`Value`] -- the parsed value of one card.
//!
//! Each submodule owns one part of the work:
//!
//! - [`value`] -- the value and comment parser.
//! - [`view`] -- [`CardView`], the read path over the stored bytes.
//! - [`builder`] -- card construction and editing. A card is encoded
//!   here when a caller creates or edits it.
//! - [`reserved`] -- typed accessors such as [`Header::bitpix`].
//! - [`validation`] -- structural checks that report a [`Diagnostic`].
//! - [`time`] -- the time keywords of Standard Sec.9.
//! - [`observatory`] -- the `OBSGEO` location keywords.
//! - [`units`] -- the `BUNIT` accessor built on [`crate::units`].
//!
//! # Design constraints
//!
//! [`Header::to_bytes`] writes the stored bytes. It encodes no card.
//! A card is encoded when a caller creates or edits it, so a value
//! that the header cannot write returns an error at that point. A
//! card read from a file is written back as it was read, and
//! [`Header::normalize`] is the one method that rewrites it.
//!
//! A keyword can repeat. `COMMENT` and `HISTORY` do so by design, and
//! a malformed file repeats a value keyword. [`Header`] keeps every
//! occurrence and indexes the first one. [`Header::first`] is
//! therefore a constant-time lookup, and [`Header::all`] reaches
//! every occurrence.
//!
//! Lookup falls back from `-` to `_`. Sec.4.1.2.1 makes the two
//! distinct, but real files write `MJD_OBS` for `MJD-OBS`. The
//! fallback runs in one direction only, so `CD1_1` never matches a
//! `CD1-1` card.

pub mod builder;
// The card scanner is an implementation detail of this module.
// [`CardView`] is how a caller reads a card; `Card` is how the parser
// splits one, and nothing outside the crate needs that distinction.
pub(crate) mod card;
pub mod observatory;
pub mod reserved;
pub mod time;
pub mod units;
pub mod validation;
pub mod value;
pub mod view;

pub use builder::CommentaryKind;
pub use card::{CARD_SIZE, CardKind};
pub use observatory::{ObsGeo, ObsGeodetic};
pub use time::IsoDateTime;
pub use validation::{Diagnostic, Fix, Level};
pub use value::Value;
pub use view::CardView;

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::OnceLock;

use crate::error::{FitsError, Result};
use crate::io::block::{BLOCK_SIZE, CARDS_PER_BLOCK};
use card::Card;

/// A parsed FITS header: a sequence of value cards plus commentary.
///
/// [`Header::parse`] reads one from bytes, and the builder methods in
/// [`builder`] construct one. [`Header::to_bytes`] renders it back.
///
/// # Examples
///
/// ```
/// use fitsy::{Header, Value};
///
/// let mut h = Header::empty();
/// h.push("SIMPLE", true, Some("conforming FITS file"))?;
/// h.push("OBJECT", "M42", None)?;
///
/// assert!(h.contains("OBJECT"));
/// assert_eq!(h.first("OBJECT"), Some(Value::String("M42".into())));
///
/// // Rendering pads to the 2880-byte block and appends END.
/// assert_eq!(h.to_bytes().len() % 2880, 0);
/// # Ok::<(), fitsy::FitsError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Header {
    /// Every physical card, back to back. The length is a multiple
    /// of [`CARD_SIZE`].
    ///
    /// This is the header. A card is a fixed 80 bytes, so one buffer
    /// holds them all. [`CardView`] reads a keyword, a value or a
    /// comment from these bytes and writes nothing back, and
    /// [`Header::to_bytes`] emits the buffer, so a card is written as
    /// it was stored.
    bytes: Vec<u8>,
    /// Everything derived from `bytes`: where each logical card lies,
    /// and where each keyword first appears.
    ///
    /// Built when first needed. A mutation either leaves this alone,
    /// when it cannot have changed what the layout says, or drops it.
    /// Nothing edits it into a different shape, so it cannot come to
    /// disagree with the bytes. A dropped layout costs a rebuild; a
    /// wrong one would return the wrong card.
    layout: OnceLock<Layout>,
}
impl Header {
    /// Parse a header from byte offset `start` within `bytes`, in
    /// strict mode.
    ///
    /// The result pairs the header with the number of bytes consumed,
    /// which is a multiple of 2880.
    ///
    /// # Errors
    ///
    /// The conditions of [`Header::parse_with`] with `lenient` set to
    /// `false`.
    pub fn parse(bytes: &[u8], start: u64) -> Result<(Self, u64)> {
        Self::parse_with(bytes, start, false)
    }

    /// Parse a header, keeping every card's bytes.
    ///
    /// The header stores the cards as they appear in the file. This
    /// function re-encodes no card. A card that this crate cannot
    /// fully describe is written back as it was read.
    ///
    /// When `lenient` is false, a card that breaks the Standard
    /// fails the parse. When `lenient` is true, the header keeps that
    /// card. A read of it returns [`Value::Unparsed`] rather than an
    /// error. [`Header::normalize`] rewrites it into conforming
    /// form.
    ///
    /// An `END` card is required under either value of `lenient`. It
    /// is the only marker of the boundary between header and data.
    ///
    /// # Errors
    ///
    /// - [`FitsError::Block`] when `start` is past the end of `bytes`.
    /// - [`FitsError::Header`] when no `END` card is found.
    /// - [`FitsError::EndCardMisplaced`] when a non-blank card follows
    ///   `END`, in strict mode.
    /// - In strict mode, any error from the card scanner or the value
    ///   parser.
    pub fn parse_with(bytes: &[u8], start: u64, lenient: bool) -> Result<(Self, u64)> {
        let start_usize = start as usize;
        if start_usize > bytes.len() {
            return Err(FitsError::Block {
                offset: start,
                msg: "header start beyond end of buffer".into(),
            });
        }
        // Note: the buffer may extend past the header (callers often hand us
        // a whole file) and may end in a partial block; only full 2880-byte
        // blocks are scanned, and a header whose END card is never found in
        // them is rejected below.
        let mut cards: Vec<u8> = Vec::new();
        let mut block_idx = 0_usize;
        let mut end_seen = false;

        'outer: while start_usize + (block_idx + 1) * BLOCK_SIZE <= bytes.len() {
            let block_start = start_usize + block_idx * BLOCK_SIZE;
            for c in 0..CARDS_PER_BLOCK {
                let off = block_start + c * CARD_SIZE;
                let raw = &bytes[off..off + CARD_SIZE];
                let card = Card::parse_with(raw, off as u64, lenient)?;

                if end_seen {
                    // Sec.4.4.1.2: every card after END must be all spaces.
                    // Tolerate NUL padding here too (see `Card::parse_with`): some
                    // writers zero-fill the remainder of the final header block.
                    // In lenient mode, ignore any other trailing bytes in the
                    // final header block (some writers leave stray fill there).
                    if !lenient && raw.iter().any(|&b| b != b' ' && b != 0) {
                        return Err(FitsError::EndCardMisplaced { offset: off as u64 });
                    }
                    continue;
                }

                // Strict mode rejects a value field the Standard does
                // not allow. Lenient mode keeps the card as written.
                if !lenient && matches!(card.kind, CardKind::Value | CardKind::Continue) {
                    value::parse(&card.keyword, &card.body)?;
                }

                // Every card but `END` is kept exactly as it appears.
                // Which of them join into one logical card is decided
                // by `Layout`, from these same bytes.
                if matches!(card.kind, CardKind::End) {
                    end_seen = true;
                } else {
                    cards.extend_from_slice(raw);
                }
            }
            block_idx += 1;
            if end_seen {
                break 'outer;
            }
        }

        if !end_seen {
            // Sec.4.4.1.2 requires an END card. It is enforced even in
            // lenient mode: END is the sole delimiter between the header and
            // the data section, so without it there is no way to know where
            // the header ends. (A lower-case or mixed-case `end` keyword is
            // still accepted in lenient mode -- the card scanner folds it to
            // `END` -- so this only fires when no END card is present at all.)
            return Err(FitsError::Header("no END card found in header".into()));
        }

        let consumed = (block_idx * BLOCK_SIZE) as u64;
        Ok((Self::from_bytes(cards), consumed))
    }

    /// Render the header: every card's bytes, then `END`, padded with
    /// spaces to a whole 2880-byte block.
    ///
    /// This function encodes no card. The header holds the bytes of
    /// every card, whether the card came from a file or from
    /// [`push`](Self::push). This function copies those bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(self.bytes.len() + BLOCK_SIZE);
        out.extend_from_slice(&self.bytes);
        let mut end = [b' '; CARD_SIZE];
        end[..3].copy_from_slice(b"END");
        out.extend_from_slice(&end);
        while !out.len().is_multiple_of(BLOCK_SIZE) {
            out.push(b' ');
        }
        out
    }

    /// Every card, in order, as a read-only view over its bytes.
    pub fn cards(&self) -> impl Iterator<Item = CardView<'_>> {
        self.layout()
            .spans
            .iter()
            .map(|span| CardView::new(&self.bytes[span.clone()]))
    }

    /// The layout of the current bytes, deriving it if it is not held.
    fn layout(&self) -> &Layout {
        self.layout.get_or_init(|| Layout::build(&self.bytes))
    }

    /// Drop the derived layout, because the bytes no longer match it.
    fn forget_layout(&mut self) {
        self.layout.take();
    }

    /// The card at `idx`, or `None` when the index is past the end.
    #[must_use]
    pub fn card(&self, idx: usize) -> Option<CardView<'_>> {
        let span = self.layout().spans.get(idx)?.clone();
        Some(CardView::new(&self.bytes[span]))
    }

    /// How many logical cards the header holds. A continued string
    /// counts once, however many `CONTINUE` cards it spans.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layout().spans.len()
    }

    /// True when the header holds no cards.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Return `keyword` with each `-` replaced by `_`, so a lookup for
    /// `MJD-OBS` also finds the `MJD_OBS` some writers emit. `None`
    /// when the keyword has no `-`.
    ///
    /// One direction only: Sec.4.1.2.1 makes `-` and `_` distinct, so
    /// `CD1_1` must not match a `CD1-1` card.
    pub(crate) fn alt_key(keyword: &str) -> Option<String> {
        if !keyword.contains('-') {
            return None;
        }
        Some(keyword.replace('-', "_"))
    }

    /// The value of the first card named `keyword`.
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
    pub fn first(&self, keyword: &str) -> Option<Value> {
        self.first_card(keyword).and_then(|c| c.value())
    }

    /// A view of the first card named `keyword`.
    #[must_use]
    pub fn first_card(&self, keyword: &str) -> Option<CardView<'_>> {
        if let Some(&idx) = self.layout().index.get(keyword) {
            return self.card(idx);
        }
        // Some files use '_' where the standard uses '-' (e.g. MJD_OBS for MJD-OBS).
        let &idx = self.layout().index.get(Self::alt_key(keyword)?.as_str())?;
        self.card(idx)
    }

    /// Every card named `keyword`, in order.
    ///
    /// The Standard allows a keyword to repeat. This returns every
    /// occurrence.
    pub fn all<'a>(&'a self, keyword: &'a str) -> impl Iterator<Item = CardView<'a>> {
        self.cards().filter(move |c| c.has_keyword(keyword))
    }

    /// True if `keyword` is present, with any card kind.
    #[must_use]
    pub fn contains(&self, keyword: &str) -> bool {
        self.has_exactly(keyword)
            || Self::alt_key(keyword).is_some_and(|alt| self.has_exactly(&alt))
    }

    /// True if a card is named exactly `keyword`, with no `-`/`_`
    /// fallback.
    ///
    /// The index answers for a value card without reading any card. A
    /// commentary keyword is not indexed, so a miss still has to look.
    fn has_exactly(&self, keyword: &str) -> bool {
        self.layout().index.contains_key(keyword) || self.cards().any(|c| c.has_keyword(keyword))
    }

    /// Iterate over the body text of every `COMMENT` card in the
    /// order they appear (Sec.4.4.2.1). Returns an empty iterator if
    /// none are present.
    pub fn comments(&self) -> impl Iterator<Item = String> {
        self.commentary_iter("COMMENT")
    }

    /// Iterate over the body text of every `HISTORY` card in the
    /// order they appear (Sec.4.4.2.2).
    pub fn history(&self) -> impl Iterator<Item = String> {
        self.commentary_iter("HISTORY")
    }

    /// Iterate over the body text of every blank-keyword commentary
    /// card (Sec.4.1.2.3 -- eight spaces in the keyword field).
    pub fn blank_commentary(&self) -> impl Iterator<Item = String> {
        self.commentary_iter("")
    }

    fn commentary_iter<'a>(&'a self, keyword: &'a str) -> impl Iterator<Item = String> + 'a {
        self.cards().filter_map(move |c| {
            if c.is_commentary() && c.has_keyword(keyword) {
                c.commentary()
            } else {
                None
            }
        })
    }

    /// A header over `bytes`, which must be whole 80-byte cards.
    pub(crate) fn from_bytes(bytes: Vec<u8>) -> Self {
        debug_assert!(
            bytes.len().is_multiple_of(CARD_SIZE),
            "a header is a whole number of 80-byte cards"
        );
        Self {
            bytes,
            layout: OnceLock::new(),
        }
    }

    /// Append `bytes`, one or more whole cards, as one logical card.
    pub(crate) fn append_card_bytes(&mut self, bytes: &[u8]) {
        debug_assert!(
            bytes.len().is_multiple_of(CARD_SIZE) && !bytes.is_empty(),
            "a logical card is one or more whole 80-byte cards"
        );
        let span = self.bytes.len()..self.bytes.len() + bytes.len();
        self.bytes.extend_from_slice(bytes);
        // Appending cannot invalidate what the layout already says, so
        // this is the one mutation that keeps it.
        if let Some(layout) = self.layout.get_mut() {
            layout.note_appended(span, &self.bytes);
        }
    }

    /// Insert `bytes` as a logical card at `idx`, shifting the rest.
    pub(crate) fn insert_card_bytes(&mut self, idx: usize, bytes: Vec<u8>) {
        let at = self
            .layout()
            .spans
            .get(idx)
            .map_or(self.bytes.len(), |span| span.start);
        self.bytes.splice(at..at, bytes);
        // Every card after the insertion point moved.
        self.forget_layout();
    }

    /// Replace the logical card at `idx`.
    pub(crate) fn replace_card_bytes(&mut self, idx: usize, bytes: Vec<u8>) {
        let Some(range) = self.layout().spans.get(idx).cloned() else {
            return;
        };
        // Replacing a card in place moves nothing and renames nothing,
        // so the layout still describes these bytes. Setting a value
        // under a keyword that is already there -- the ordinary case --
        // takes this path.
        let same_size = range.len() == bytes.len();
        let same_name =
            CardView::new(&bytes).has_keyword(&CardView::new(&self.bytes[range.clone()]).keyword());
        if same_size {
            self.bytes[range].copy_from_slice(&bytes);
        } else {
            self.bytes.splice(range, bytes);
        }
        if !(same_size && same_name) {
            self.forget_layout();
        }
    }

    /// Copy `card` onto the end of this header, bytes and all.
    ///
    /// One header takes a card from another with this method. The
    /// bytes move unchanged. The value, the comment and the layout of
    /// the card are all preserved.
    ///
    /// This method does not look the keyword up. It appends the card,
    /// so it does not replace a card that has the same keyword. Use
    /// [`set`](Self::set) to edit a card by keyword.
    pub fn splice(&mut self, card: &CardView<'_>) {
        self.append_card_bytes(card.raw());
    }

    /// Index of the first value card named `keyword`.
    pub(crate) fn first_value_index(&self, keyword: &str) -> Option<usize> {
        self.layout().index.get(keyword).copied()
    }

    /// Remove every card named `keyword`, returning how many went.
    pub fn remove(&mut self, keyword: &str) -> usize {
        // Value cards only. A commentary keyword names a class of
        // cards, not one card. A caller that names `COMMENT` does not
        // ask to delete every comment in the header.
        let keep: Vec<Range<usize>> = self
            .layout()
            .spans
            .iter()
            .filter(|span| {
                let card = CardView::new(&self.bytes[(*span).clone()]);
                !(matches!(card.kind(), CardKind::Value) && card.has_keyword(keyword))
            })
            .cloned()
            .collect();
        let removed = self.layout().spans.len() - keep.len();
        if removed > 0 {
            let mut bytes = Vec::with_capacity(self.bytes.len());
            for span in keep {
                bytes.extend_from_slice(&self.bytes[span]);
            }
            self.bytes = bytes;
            self.forget_layout();
        }
        removed
    }

    /// Rewrite every card that does not conform to the Standard into
    /// conforming form, leaving conforming cards byte-for-byte alone.
    ///
    /// This is the only method that re-encodes a card that was read
    /// from a file. A caller must call it. No other operation
    /// rewrites a card that it did not change.
    ///
    /// # Errors
    ///
    /// [`FitsError::Header`] when a non-conforming card cannot be
    /// re-encoded, which leaves the header untouched.
    pub fn normalize(&mut self) -> Result<usize> {
        let mut fixed = 0;
        let mut bytes: Vec<u8> = Vec::with_capacity(self.bytes.len());
        let spans = self.layout().spans.clone();
        for span in spans {
            let raw = &self.bytes[span];
            if raw
                .as_chunks::<CARD_SIZE>()
                .0
                .iter()
                .all(|c| card_is_conforming(c))
            {
                bytes.extend_from_slice(raw);
                continue;
            }
            let view = CardView::new(raw);
            let encoded = if view.is_commentary() {
                builder::encode_commentary_card(
                    &view.keyword(),
                    view.commentary().as_deref().unwrap_or(""),
                )?
            } else {
                let Some((value, comment)) = view.parsed() else {
                    bytes.extend_from_slice(raw);
                    continue;
                };
                // A field no standard type matched keeps its text, as
                // a string: repairing the card must not lose what it
                // said, only how it said it.
                let value = match value {
                    Value::Unparsed(text) => Value::String(text),
                    other => other,
                };
                builder::encode_value_card(&view.keyword(), &value, comment.as_deref())?
            };
            bytes.extend_from_slice(&encoded);
            fixed += 1;
        }
        self.bytes = bytes;
        self.forget_layout();
        Ok(fixed)
    }

    /// Merge value cards from `parent` that this header lacks.
    ///
    /// Adds a card only when this header has no card with that
    /// keyword. Skips commentary and structural keywords: a
    /// `HISTORY` chain belongs to the HDU that recorded it, and the
    /// structural cards describe this HDU's own data.
    pub fn merge_inherited(&mut self, parent: &Self) {
        // Read this header's keywords once. Asking `contains` per
        // parent card would rescan this header for every one of them.
        let mine: std::collections::HashSet<String> = self.cards().map(|c| c.keyword()).collect();
        let mut adopt: Vec<Vec<u8>> = Vec::new();
        for card in parent.cards() {
            if card.is_commentary() {
                continue;
            }
            let keyword = card.keyword();
            if is_structural_keyword(&keyword) || mine.contains(&keyword) {
                continue;
            }
            adopt.push(card.raw().to_vec());
        }
        for bytes in adopt {
            self.append_card_bytes(&bytes);
        }
    }
}

/// True when the last physical card of `span` leaves its string open,
/// so the `CONTINUE` card after it extends the same value
/// (Sec.4.2.1.2).
///
/// Only the last card of the run can be open, so this reads that one
/// card. Asking the whole logical card for its value would rejoin the
/// entire chain, once per card, to answer a question about its tail.
fn ends_open(bytes: &[u8], span: &Range<usize>) -> bool {
    let last = &bytes[span.end - CARD_SIZE..span.end];
    let Ok(card) = Card::parse_with(last, 0, true) else {
        return false;
    };
    let (value, _) = value::parse_lenient(&card.keyword, &card.body);
    matches!(value, Value::String(s) if s.ends_with('&'))
}

/// Where each logical card lies in a header's bytes, and where each
/// keyword first appears.
///
/// Both facts follow from the bytes, so this is a cache. It is built
/// in one place and never edited into a different shape, which is what
/// keeps it from disagreeing with the bytes it describes.
#[derive(Debug, Clone)]
struct Layout {
    /// Byte range of each logical card, in order.
    spans: Vec<Range<usize>>,
    /// Logical index of the first value card with each keyword.
    /// A commentary keyword names a class of cards rather than one
    /// card, so it is not listed.
    index: BTreeMap<String, usize>,
}

impl Layout {
    /// Derive the layout of `bytes`.
    fn build(bytes: &[u8]) -> Self {
        let mut spans: Vec<Range<usize>> = Vec::new();
        let mut at = 0_usize;
        while at + CARD_SIZE <= bytes.len() {
            // Sec.4.2.1.2: a `CONTINUE` extends the card before it when
            // that card's string is still open. The two are then one
            // logical card. A `CONTINUE` that attaches to nothing
            // stands alone, which the same section reads as commentary.
            let attaches = matches!(
                card::kind_of(&bytes[at..at + CARD_SIZE], true),
                CardKind::Continue
            ) && spans.last().is_some_and(|prev| ends_open(bytes, prev));
            if attaches {
                spans.last_mut().expect("checked by `attaches`").end += CARD_SIZE;
            } else {
                spans.push(at..at + CARD_SIZE);
            }
            at += CARD_SIZE;
        }
        let mut index = BTreeMap::new();
        for (i, span) in spans.iter().enumerate() {
            let view = CardView::new(&bytes[span.clone()]);
            if view.is_commentary() {
                continue;
            }
            let keyword = view.keyword();
            if !keyword.is_empty() {
                index.entry(keyword).or_insert(i);
            }
        }
        Self { spans, index }
    }

    /// Record a card appended at the end of the header.
    ///
    /// The only edit this type allows, and the only mutation that
    /// cannot invalidate an existing entry: every card already present
    /// keeps its span, and a card at the end can never displace the
    /// first occurrence of a keyword.
    fn note_appended(&mut self, span: Range<usize>, bytes: &[u8]) {
        let idx = self.spans.len();
        let view = CardView::new(&bytes[span.clone()]);
        if !view.is_commentary() {
            let keyword = view.keyword();
            if !keyword.is_empty() {
                self.index.entry(keyword).or_insert(idx);
            }
        }
        self.spans.push(span);
    }
}

/// True when a writer regenerates `kw` from the HDU it is building,
/// so a copy must not carry the source's version of it.
///
/// This covers the cards that describe the HDU's own structure, which
/// [`ImageBuilder::build`](crate::ImageBuilder::build) and the table
/// builders emit themselves, and the checksums, which the writer
/// recomputes over the bytes it actually wrote -- carrying a source
/// `CHECKSUM` forward would describe data that no longer exists.
///
/// Every other card belongs to the user and is copied verbatim.
/// This deliberately excludes `INHERIT`, which carries meaning of its
/// own, and the `Z` compression keywords, which the compression path
/// maps rather than drops.
#[must_use]
pub fn is_writer_owned_keyword(kw: &str) -> bool {
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
            | "TFIELDS"
            | "GROUPS"
            | "CHECKSUM"
            | "DATASUM"
    ) {
        return true;
    }
    card::is_indexed(kw, &["NAXIS"])
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
    if card::is_indexed(kw, &["NAXIS"]) {
        return true;
    }
    if card::is_indexed(kw, &["ZNAXIS"]) {
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

/// True when `raw`, one 80-byte card, conforms to the Standard as
/// written. A strict parse accepts such a card and repairs nothing.
///
/// [`Header::normalize`] uses this to select the cards it rewrites.
/// It leaves a conforming card unchanged.
fn card_is_conforming(raw: &[u8]) -> bool {
    if raw.len() != CARD_SIZE {
        return false;
    }
    // Sec.4.1.1: a card holds printable ASCII, 0x20 to 0x7e, only.
    if raw.iter().any(|&b| !(0x20..=0x7e).contains(&b)) {
        return false;
    }
    let Ok(card) = Card::parse_with(raw, 0, false) else {
        return false;
    };
    match card.kind {
        CardKind::Value | CardKind::Continue => value::parse(&card.keyword, &card.body).is_ok(),
        CardKind::Commentary | CardKind::End => true,
    }
}

#[cfg(test)]
mod tests {

    /// Setting a value under a keyword that is already there leaves
    /// every other card, and the lookup index, exactly as it was.
    #[test]
    fn set_keeps_the_index_and_its_neighbours() {
        let mut h = Header::empty();
        for i in 0..8 {
            h.push(format!("KEY{i}"), i64::from(i), None).unwrap();
        }
        h.set("KEY3", 300_i64, None).unwrap();
        assert_eq!(h.first("KEY3"), Some(Value::Integer(300)));
        for i in (0..8).filter(|&i| i != 3) {
            assert_eq!(
                h.first(&format!("KEY{i}")),
                Some(Value::Integer(i64::from(i)))
            );
        }
        assert_eq!(h.len(), 8);
    }

    /// A set whose new card needs more physical cards than the old one
    /// shifts every later card, and they must all still be found.
    #[test]
    fn set_to_a_continued_string_keeps_later_cards_readable() {
        let mut h = Header::empty();
        for i in 0..6 {
            h.push(format!("KEY{i}"), i64::from(i), None).unwrap();
        }
        let long = "x".repeat(150);
        h.set("KEY2", long.clone(), None).unwrap();
        assert_eq!(h.first("KEY2"), Some(Value::String(long)));
        assert!(h.first_card("KEY2").unwrap().physical_cards() > 1);
        for i in [0, 1, 3, 4, 5] {
            assert_eq!(
                h.first(&format!("KEY{i}")),
                Some(Value::Integer(i64::from(i)))
            );
        }
        // And the whole thing still round-trips.
        let (re, _) = Header::parse_with(&h.to_bytes(), 0, true).unwrap();
        assert_eq!(re.len(), h.len());
    }

    /// Inserting ahead of a card with the same keyword makes the new
    /// card the one `first` reports.
    #[test]
    fn insert_before_a_duplicate_becomes_the_first() {
        let mut h = Header::empty();
        h.push("OBJECT", "second", None).unwrap();
        h.insert(0, "OBJECT", "first", None).unwrap();
        assert_eq!(h.first("OBJECT"), Some(Value::String("first".into())));
        let all: Vec<Value> = h.all("OBJECT").filter_map(|c| c.value()).collect();
        assert_eq!(
            all,
            vec![
                Value::String("first".into()),
                Value::String("second".into())
            ]
        );
    }

    /// Inserting after one leaves the earlier card first, and every
    /// card that moved is still found under its own keyword.
    #[test]
    fn insert_after_a_duplicate_leaves_the_first_alone() {
        let mut h = Header::empty();
        h.push("OBJECT", "first", None).unwrap();
        h.push("TAIL", 1_i64, None).unwrap();
        h.insert(1, "OBJECT", "second", None).unwrap();
        assert_eq!(h.first("OBJECT"), Some(Value::String("first".into())));
        assert_eq!(h.first("TAIL"), Some(Value::Integer(1)));
    }

    /// A commentary card is not indexed, and inserting one still
    /// shifts the cards it moved.
    #[test]
    fn insert_of_a_commentary_card_shifts_the_index() {
        let mut h = Header::empty();
        h.push("OBJECT", "M31", None).unwrap();
        h.push("TAIL", 1_i64, None).unwrap();
        let card = builder::encode_commentary_card("COMMENT", "note").unwrap();
        h.insert_card_bytes(0, card);
        assert_eq!(h.first("OBJECT"), Some(Value::String("M31".into())));
        assert_eq!(h.first("TAIL"), Some(Value::Integer(1)));
        assert_eq!(h.comments().collect::<Vec<_>>(), vec!["note".to_string()]);
    }

    /// Renaming does change what a card is called, so the index has to
    /// be rebuilt for it.
    #[test]
    fn rename_updates_the_index() {
        let mut h = Header::empty();
        h.push("OLD", 1_i64, None).unwrap();
        h.push("KEEP", 2_i64, None).unwrap();
        assert_eq!(h.rename_keyword("OLD", "NEW").unwrap(), 1);
        assert_eq!(h.first("NEW"), Some(Value::Integer(1)));
        assert_eq!(h.first("OLD"), None);
        assert_eq!(h.first("KEEP"), Some(Value::Integer(2)));
    }

    /// The layout is derived from the bytes, so whatever a mutation
    /// does to them, what the header holds must match what a fresh
    /// derivation says. This walks a long mixed sequence of edits and
    /// checks that after every one.
    #[test]
    fn the_held_layout_always_matches_a_fresh_one() {
        // xorshift, so the sequence is reproducible without a dependency.
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..60 {
            let mut h = Header::empty();
            for step in 0..40 {
                let keyword = if rnd() % 4 == 0 {
                    format!("HIERARCH ESO DET CHIP{}", rnd() % 5)
                } else {
                    format!("K{:03}", rnd() % 12)
                };
                match rnd() % 6 {
                    0 => {
                        let _ = h.push(keyword, (rnd() % 1000) as i64, None);
                    }
                    1 => {
                        let _ = h.set(&keyword, (rnd() % 1000) as i64, None);
                    }
                    2 => {
                        let at = (rnd() as usize) % (h.len() + 1);
                        let _ = h.insert(at, keyword, (rnd() % 1000) as i64, None);
                    }
                    3 => {
                        h.remove(&keyword);
                    }
                    4 => {
                        let _ = h.push_commentary(CommentaryKind::History, "note");
                    }
                    // long enough to need a CONTINUE chain, which is
                    // the case that makes a logical card span several
                    // physical ones.
                    _ => {
                        let _ = h.set(&keyword, "z".repeat(90), None);
                    }
                }
                let fresh = Layout::build(&h.bytes);
                assert_eq!(
                    h.layout().spans,
                    fresh.spans,
                    "round {round} step {step}: spans drifted"
                );
                assert_eq!(
                    h.layout().index,
                    fresh.index,
                    "round {round} step {step}: index drifted"
                );
                // And the bytes still describe the same cards on the
                // way back in.
                let (re, _) = Header::parse_with(&h.to_bytes(), 0, true).unwrap();
                let mine: Vec<(String, Option<Value>)> =
                    h.cards().map(|c| (c.keyword(), c.value())).collect();
                let theirs: Vec<(String, Option<Value>)> =
                    re.cards().map(|c| (c.keyword(), c.value())).collect();
                assert_eq!(mine, theirs, "round {round} step {step}: round trip");
            }
        }
    }

    /// Editing a header must not cost more the larger it gets. A
    /// quadratic `set` made a 1600-card header take a quarter second
    /// to update; the bound here is loose enough for a noisy machine
    /// but far under what that regression would produce.
    #[test]
    fn editing_scales_with_the_number_of_edits_not_the_header() {
        use std::time::Instant;
        let elapsed = |n: usize| {
            let mut h = Header::empty();
            for i in 0..n {
                h.push(format!("KEY{i:05}"), i as i64, None).unwrap();
            }
            let t = Instant::now();
            for i in 0..n {
                h.set(&format!("KEY{i:05}"), (i * 2) as i64, None).unwrap();
            }
            t.elapsed().as_secs_f64()
        };
        let small = elapsed(200).max(1e-6);
        let large = elapsed(1600);
        // Linear work grows 8x with 8x the edits; quadratic grows 64x.
        assert!(
            large / small < 24.0,
            "editing looks super-linear: 200 keys took {small:.4}s, 1600 took {large:.4}s"
        );
    }

    /// Comment text too long for its card continues onto `CONTINUE`
    /// cards, and the reader rejoins every fragment.
    #[test]
    fn oversized_comment_round_trips_through_continuations() {
        // Two-letter words: many break points, so the chunking is
        // exercised rather than the single-split case.
        let comment = "ab ".repeat(60).trim_end().to_string();
        let mut h = Header::empty();
        h.push("OBJECT", "M31", Some(&comment)).unwrap();
        let (re, _) = Header::parse_with(&h.to_bytes(), 0, true).unwrap();
        let card = re.first_card("OBJECT").expect("card present");
        assert_eq!(card.value(), Some(Value::String("M31".into())));
        assert_eq!(card.comment().as_deref(), Some(comment.as_str()));
    }

    /// A value that cannot carry a `CONTINUE` chain spills its comment
    /// to a `COMMENT` card rather than to an orphan.
    #[test]
    fn unattachable_comment_spills_to_a_commentary_card() {
        let comment = "a long note about the exposure that will not fit on one card at all";
        let mut h = Header::empty();
        h.push("EXPTIME", 30.0_f64, Some(comment)).unwrap();
        let (re, _) = Header::parse_with(&h.to_bytes(), 0, true).unwrap();
        // No orphaned CONTINUE card is left behind.
        assert!(re.cards().all(|c| !c.has_keyword("CONTINUE")));
        let tail: Vec<String> = re.comments().collect();
        assert_eq!(tail.len(), 1);
        // Every word survives, across the card and its overflow.
        let head = re.first_card("EXPTIME").and_then(|c| c.comment()).unwrap();
        assert_eq!(format!("{head} {}", tail[0]), comment);
    }

    /// Commentary too long for one card becomes several cards, and a
    /// built header reads back exactly as a parsed one does.
    #[test]
    fn long_commentary_matches_its_parsed_form() {
        let mut h = Header::empty();
        h.push_commentary(CommentaryKind::History, &"y".repeat(150))
            .unwrap();
        let built: Vec<String> = h.history().collect();
        let (re, _) = Header::parse_with(&h.to_bytes(), 0, true).unwrap();
        let parsed: Vec<String> = re.history().collect();
        assert_eq!(built, parsed);
        assert_eq!(built.concat(), "y".repeat(150));
    }

    /// A header takes no card it could not write. Every construction
    /// path encodes the card first, so an invalid one is refused at
    /// the point it is offered rather than discovered on the way out.
    #[test]
    fn a_card_that_cannot_be_written_is_never_taken() {
        let mut h = Header::empty();
        // Non-printable bytes are forbidden on a card (Sec.4.1.1).
        let bad = "caf\u{e9}";
        assert!(h.push("OBJECT", bad, None).is_err());
        assert!(h.push("OBJECT", "M31", Some(bad)).is_err());
        assert!(h.push_commentary(CommentaryKind::Comment, bad).is_err());
        assert!(h.push_commentary(CommentaryKind::History, bad).is_err());
        assert!(h.set("OBJECT", bad, None).is_err());
        assert!(h.insert(0, "OBJECT", bad, None).is_err());
        // A refused card leaves nothing behind, and what the header
        // does hold still writes.
        assert!(h.is_empty());
        assert_eq!(h.to_bytes().len(), BLOCK_SIZE);
    }

    /// Reading alters nothing. A card the Standard rejects is still
    /// written back exactly as it was read; repairing it is something
    /// a caller asks for with `normalize`.
    #[test]
    fn a_card_read_is_a_card_written() {
        let mut raw = vec![b' '; BLOCK_SIZE];
        let card = b"OBJECT  = 'M31'    / caf\xe9 au lait";
        raw[..card.len()].copy_from_slice(card);
        raw[CARD_SIZE..CARD_SIZE + 3].copy_from_slice(b"END");
        let (mut h, _) = Header::parse_with(&raw, 0, true).unwrap();
        assert_eq!(h.to_bytes(), raw, "a read card is written back unaltered");
        // The caller can ask for the repair, and only then.
        assert_eq!(h.normalize().unwrap(), 1);
        assert!(!h.to_bytes().contains(&0xe9));
    }

    /// `remove` takes value cards. A commentary keyword names a
    /// class of cards, not one card, so `remove` leaves those cards
    /// in place.
    #[test]
    fn remove_leaves_commentary_cards_alone() {
        let mut h = Header::empty();
        h.push_commentary(CommentaryKind::Comment, "note").unwrap();
        h.push("OBJECT", "M31", None).unwrap();
        assert_eq!(h.remove("COMMENT"), 0);
        assert_eq!(h.comments().count(), 1);
        assert_eq!(h.remove("OBJECT"), 1);
    }

    /// A conforming card that is read and written unchanged comes
    /// back byte-for-byte. The write path emits the stored bytes and
    /// re-encodes no card.
    #[test]
    fn conforming_cards_round_trip_byte_for_byte() {
        let cards = [
            "SIMPLE  =                    T / conforms to FITS standard",
            "BITPIX  =                   16 / array data type",
            "NAXIS   =                    0",
            // A comment sitting where our own encoder would not put it.
            "OBSERVER= 'me'   / who observed",
            // Duplicate keywords: legal, order-significant, not a map.
            "NOTE    = 'one'                / first",
            "NOTE    = 'two'                / second",
            "COMMENT first light",
            "HISTORY flat-fielded",
            "END",
        ];
        let mut bytes = vec![b' '; BLOCK_SIZE];
        for (i, c) in cards.iter().enumerate() {
            bytes[i * CARD_SIZE..i * CARD_SIZE + c.len()].copy_from_slice(c.as_bytes());
        }
        let (h, _) = Header::parse_with(&bytes, 0, true).unwrap();
        assert_eq!(h.to_bytes(), bytes, "header did not round-trip verbatim");
        // Both duplicates survive, in order: a keyword is not a key.
        let notes: Vec<Value> = h.all("NOTE").filter_map(|c| c.value()).collect();
        assert_eq!(
            notes,
            vec![Value::String("one".into()), Value::String("two".into())]
        );
    }

    /// A long string and its `CONTINUE` cards are one logical card
    /// spanning several physical ones.
    #[test]
    fn continue_chain_is_one_logical_card() {
        let cards = [
            "LONGSTR = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa&'",
            "CONTINUE  'bbbbbbbbbbbbbbbb' / joined",
            "END",
        ];
        let mut bytes = vec![b' '; BLOCK_SIZE];
        for (i, c) in cards.iter().enumerate() {
            bytes[i * CARD_SIZE..i * CARD_SIZE + c.len()].copy_from_slice(c.as_bytes());
        }
        let (h, _) = Header::parse_with(&bytes, 0, true).unwrap();
        let card = h.first_card("LONGSTR").expect("card present");
        assert_eq!(card.physical_cards(), 2);
        assert_eq!(card.raw().len(), 2 * CARD_SIZE);
        match card.value() {
            Some(Value::String(s)) => assert!(s.starts_with('a') && s.ends_with('b')),
            other => panic!("expected a joined string, got {other:?}"),
        }
        assert_eq!(h.to_bytes(), bytes);
    }

    /// Normalizing rewrites a card the Standard rejects and leaves
    /// every conforming card exactly as it was.
    #[test]
    fn normalize_rewrites_only_the_non_conforming_card() {
        let cards = [
            "SIMPLE  =                    T",
            "WEIRD   =             12.3.4.5 / unparsable value",
            "OBSERVER= 'me'   / who observed",
            "END",
        ];
        let mut bytes = vec![b' '; BLOCK_SIZE];
        for (i, c) in cards.iter().enumerate() {
            bytes[i * CARD_SIZE..i * CARD_SIZE + c.len()].copy_from_slice(c.as_bytes());
        }
        let (mut h, _) = Header::parse_with(&bytes, 0, true).unwrap();
        // Until asked, even the malformed card is written back as read.
        assert_eq!(h.to_bytes(), bytes);

        assert_eq!(h.normalize().unwrap(), 1, "exactly one card needed fixing");
        let out = h.to_bytes();
        // The conforming neighbour is untouched; the repaired one is not.
        assert_eq!(
            &out[2 * CARD_SIZE..3 * CARD_SIZE],
            &bytes[2 * CARD_SIZE..3 * CARD_SIZE]
        );
        assert_ne!(
            &out[CARD_SIZE..2 * CARD_SIZE],
            &bytes[CARD_SIZE..2 * CARD_SIZE]
        );
        // Normalizing keeps the text it could not parse as a value.
        assert_eq!(h.first("WEIRD"), Some(Value::String("12.3.4.5".into())));
    }

    /// Editing one card rewrites that card and no other.
    #[test]
    fn editing_a_card_touches_only_that_card() {
        let cards = [
            "SIMPLE  =                    T / conforms to FITS standard",
            "OBSERVER= 'me'   / who observed",
            "END",
        ];
        let mut bytes = vec![b' '; BLOCK_SIZE];
        for (i, c) in cards.iter().enumerate() {
            bytes[i * CARD_SIZE..i * CARD_SIZE + c.len()].copy_from_slice(c.as_bytes());
        }
        let (mut h, _) = Header::parse_with(&bytes, 0, true).unwrap();
        h.set("OBSERVER", "someone else", None).unwrap();
        let out = h.to_bytes();
        assert_eq!(
            &out[..CARD_SIZE],
            &bytes[..CARD_SIZE],
            "SIMPLE was rewritten"
        );
        let observer = String::from_utf8_lossy(&out[CARD_SIZE..2 * CARD_SIZE]).to_string();
        assert!(observer.contains("someone else"));
        // An edit keeps the comment the caller did not replace.
        assert!(observer.contains("who observed"));
    }

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
    fn continue_without_ampersand_is_commentary_not_an_error() {
        // Sec.4.2.1.2: the `&` is what licenses the join. Without it the
        // parent's value is complete and the CONTINUE is an orphan, which
        // the standard says to read as commentary -- in both modes, since
        // it explicitly does not invalidate the file.
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "OBJECT  = 'first part'",
            "CONTINUE  ' and second'",
            "END",
        ]);
        for lenient in [false, true] {
            let (h, _) = Header::parse_with(&bytes, 0, lenient)
                .unwrap_or_else(|e| panic!("lenient={lenient}: {e}"));
            match h.first("OBJECT").unwrap() {
                Value::String(s) => assert_eq!(s, "first part", "lenient={lenient}"),
                other => panic!("not a string: {other:?}"),
            }
            // Demoted to commentary: it must not be indexed as a value.
            let orphan = h
                .cards()
                .find(|c| c.has_keyword("CONTINUE"))
                .expect("orphan card is kept");
            assert!(orphan.is_commentary());
            assert!(orphan.value().is_none());
        }
    }

    #[test]
    fn continue_with_non_string_parent_is_commentary() {
        // Same rule when there is no value card to attach to at all.
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "CONTINUE  'dangling'",
            "END",
        ]);
        for lenient in [false, true] {
            let (h, _) = Header::parse_with(&bytes, 0, lenient)
                .unwrap_or_else(|e| panic!("lenient={lenient}: {e}"));
            assert_eq!(h.naxis().unwrap(), 0);
            assert!(h.first("CONTINUE").is_none(), "not a value card");
        }
    }

    #[test]
    fn continue_chain_still_joins_and_drops_the_marker() {
        // The valid case must keep working: each `&` is consumed and the
        // substrings concatenate in order.
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "STRKEY  = 'This keyword value is continued &'",
            "CONTINUE  ' over multiple keyword records.&'",
            "CONTINUE  ''",
            "END",
        ]);
        let (h, _) = Header::parse(&bytes, 0).unwrap();
        assert_eq!(
            h.optional_string("STRKEY").as_deref(),
            Some("This keyword value is continued  over multiple keyword records.")
        );
    }

    #[test]
    fn missing_end_rejected_even_when_lenient() {
        // END is the only header/data delimiter, so it is required in both
        // modes: a header without it cannot be delimited.
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
        ]);
        assert!(Header::parse(&bytes, 0).is_err());
        assert!(Header::parse_with(&bytes, 0, true).is_err());
    }

    #[test]
    fn lowercase_end_accepted_when_lenient() {
        // A lower-case/mixed-case `end` keyword is folded to `END` in
        // lenient mode and recognized as the terminator; strict rejects it.
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "end",
        ]);
        assert!(Header::parse(&bytes, 0).is_err());
        let (h, _) = Header::parse_with(&bytes, 0, true).unwrap();
        assert_eq!(h.naxis().unwrap(), 0);
    }

    #[test]
    fn junk_after_end_ignored_lenient() {
        let mut bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "END",
        ]);
        // A stray non-space byte in the block after END.
        bytes[4 * CARD_SIZE] = b'X';
        assert!(Header::parse(&bytes, 0).is_err());
        assert!(Header::parse_with(&bytes, 0, true).is_ok());
    }

    #[test]
    fn malformed_value_aborts_strict_but_loads_lenient() {
        // A value that matches no standard type. Strict parsing rejects
        // the whole header; lenient keeps it as `Value::Unparsed` so the
        // rest of the header (and the HDU) still load.
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EXPTIME =              12.3.4.5",
            "OBJECT  = 'M31'",
            "END",
        ]);
        assert!(Header::parse(&bytes, 0).is_err());
        let (h, _) = Header::parse_with(&bytes, 0, true).unwrap();
        assert!(matches!(h.first("EXPTIME"), Some(Value::Unparsed(s)) if s == "12.3.4.5"));
        // A later card still parses normally.
        assert!(matches!(h.first("OBJECT"), Some(Value::String(s)) if s == "M31"));
    }

    #[test]
    fn unterminated_string_kept_lenient() {
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "OBJECT  = 'M31",
            "END",
        ]);
        assert!(Header::parse(&bytes, 0).is_err());
        let (h, _) = Header::parse_with(&bytes, 0, true).unwrap();
        assert!(matches!(h.first("OBJECT"), Some(Value::Unparsed(_))));
    }

    /// Build header bytes from raw card byte-slices (each padded/truncated
    /// to 80 bytes), append an END card, and pad to a block. Unlike
    /// `make_header`, this accepts non-ASCII bytes.
    fn make_header_raw(cards: &[&[u8]]) -> Vec<u8> {
        let mut buf = Vec::new();
        for c in cards {
            let mut card = [b' '; CARD_SIZE];
            let n = c.len().min(CARD_SIZE);
            card[..n].copy_from_slice(&c[..n]);
            buf.extend_from_slice(&card);
        }
        let mut end = [b' '; CARD_SIZE];
        end[..3].copy_from_slice(b"END");
        buf.extend_from_slice(&end);
        while buf.len() % BLOCK_SIZE != 0 {
            buf.push(b' ');
        }
        buf
    }

    #[test]
    fn non_ascii_in_comment_loads_in_strict_mode() {
        // A Latin-1 degree sign (0xB0) in a value card's comment must not
        // fail the default (strict) parse; comments are free text and are
        // sanitized to spaces.
        let mut exptime = b"EXPTIME =                 30.0 / temp in C".to_vec();
        let idx = exptime.iter().rposition(|&b| b == b'C').unwrap();
        exptime[idx] = 0xB0;
        let bytes = make_header_raw(&[
            b"SIMPLE  =                    T",
            b"BITPIX  =                    8",
            b"NAXIS   =                    0",
            &exptime,
        ]);
        // Default (strict) parse succeeds and the value is still numeric.
        let (h, _) = Header::parse(&bytes, 0).unwrap();
        assert_eq!(h.first("EXPTIME"), Some(Value::Real(30.0)));
    }

    #[test]
    fn non_ascii_in_commentary_card_loads_in_strict_mode() {
        let mut comment = b"COMMENT observed at 12 deg C".to_vec();
        let idx = comment.iter().rposition(|&b| b == b'C').unwrap();
        comment[idx] = 0xB0;
        let bytes = make_header_raw(&[
            b"SIMPLE  =                    T",
            b"BITPIX  =                    8",
            b"NAXIS   =                    0",
            &comment,
        ]);
        let (h, _) = Header::parse(&bytes, 0).unwrap();
        assert_eq!(h.comments().count(), 1);
    }

    #[test]
    fn non_ascii_in_string_value_needs_lenient() {
        // A non-ASCII byte in a *value* field is data, not free text: it
        // fails the strict parse and only loads under `lenient`.
        let mut obs = b"OBSERVER= 'Xose'".to_vec();
        let idx = obs.iter().position(|&b| b == b'X').unwrap();
        obs[idx] = 0xE9; // Latin-1 'e-acute'
        let bytes = make_header_raw(&[
            b"SIMPLE  =                    T",
            b"BITPIX  =                    8",
            b"NAXIS   =                    0",
            &obs,
        ]);
        assert!(Header::parse(&bytes, 0).is_err());
        let (h, _) = Header::parse_with(&bytes, 0, true).unwrap();
        assert!(matches!(h.first("OBSERVER"), Some(Value::String(_))));
    }

    #[test]
    fn unparsed_value_round_trips_verbatim() {
        let bytes = make_header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EXPTIME =              12.3.4.5",
            "END",
        ]);
        let (h, _) = Header::parse_with(&bytes, 0, true).unwrap();
        // Re-encoding preserves the raw value text unchanged.
        let out = h.to_bytes();
        let (h2, _) = Header::parse_with(&out, 0, true).unwrap();
        assert!(matches!(h2.first("EXPTIME"), Some(Value::Unparsed(s)) if s == "12.3.4.5"));
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
