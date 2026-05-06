//! IRAF `WAT*` multi-card string reassembly.
//!
//! IRAF encodes long auxiliary WCS strings (e.g. the TNX / ZPX
//! polynomial distortion records) as a sequence of FITS keywords
//! `<prefix>001`, `<prefix>002`, ... each holding a quoted string
//! fragment. The full record is the concatenation of those
//! fragments.
//!
//! ## Trailing-space caveat
//!
//! IRAF writes each WAT fragment padded with trailing spaces to a
//! fixed width and relies on those spaces being preserved when the
//! reader concatenates them. FITS 4.0 Sec.4.2.1.1 says trailing spaces
//! in quoted strings are *not significant*, so a strictly conforming
//! reader (such as ours) discards them. In practice IRAF also breaks
//! every fragment at a whitespace boundary, so joining the trimmed
//! fragments with a single space recovers the original record without
//! splitting tokens. The very rare pathological file that breaks a
//! numeric token mid-card (against IRAF's own writer) cannot be
//! reconstructed losslessly here; such files are not currently
//! supported.

use crate::header::Header;
use crate::header::value::Value;

/// Reassemble a WAT-style multi-card string.
///
/// Reads `<prefix>001`, `<prefix>002`, ... in order until the first gap
/// and concatenates their string values, separating fragments with a
/// single space. Returns `None` when no fragments are present.
#[must_use]
pub(crate) fn reassemble(header: &Header, prefix: &str) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for n in 1..=999 {
        let key = format!("{prefix}{n:03}");
        match header.first(&key) {
            Some(Value::String(s)) => parts.push(s.clone()),
            _ => break,
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;

    fn pad_card(s: &str) -> [u8; 80] {
        let mut b = [b' '; 80];
        b[..s.len()].copy_from_slice(s.as_bytes());
        b
    }

    #[test]
    fn reassembles_in_order_and_stops_at_gap() {
        let cards = [
            pad_card("WAT1_001= 'wtype=tnx axtype=ra'"),
            pad_card("WAT1_002= 'lngcor = \"3 4 4 1 -1 1 -1 1 0.1 0.0 0.0\"'"),
            // Skip 003: should stop here.
            pad_card("WAT1_004= 'should not appear'"),
            pad_card("END"),
        ];
        let mut buf = Vec::new();
        for c in &cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % 2880 != 0 {
            buf.push(b' ');
        }
        let (h, _) = Header::parse(&buf, 0).unwrap();
        let s = reassemble(&h, "WAT1_").unwrap();
        assert!(s.contains("wtype=tnx"));
        assert!(s.contains("lngcor"));
        assert!(!s.contains("should not appear"));
    }

    #[test]
    fn returns_none_when_absent() {
        let cards = [pad_card("OBJECT  = 'foo'"), pad_card("END")];
        let mut buf = Vec::new();
        for c in &cards {
            buf.extend_from_slice(c);
        }
        while buf.len() % 2880 != 0 {
            buf.push(b' ');
        }
        let (h, _) = Header::parse(&buf, 0).unwrap();
        assert!(reassemble(&h, "WAT1_").is_none());
    }
}
