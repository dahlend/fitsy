//! Typed accessors for reserved keywords (Standard Sec.4.4).

use crate::error::{FitsError, Result};
use crate::header::Header;
use crate::header::value::Value;

impl Header {
    /// `BITPIX` (Sec.4.4.1.1) as one of the legal values
    /// 8, 16, 32, 64, -32, -64.
    pub fn bitpix(&self) -> Result<i64> {
        let v = self.required_int("BITPIX")?;
        match v {
            8 | 16 | 32 | 64 | -32 | -64 => Ok(v),
            _ => Err(FitsError::Value {
                keyword: "BITPIX".into(),
                msg: format!("illegal BITPIX value {v}"),
            }),
        }
    }

    /// `NAXIS` (Sec.4.4.1.1).
    pub fn naxis(&self) -> Result<usize> {
        let v = self.required_int("NAXIS")?;
        if !(0..=999).contains(&v) {
            return Err(FitsError::Value {
                keyword: "NAXIS".into(),
                msg: format!("NAXIS must be in 0..=999, got {v}"),
            });
        }
        Ok(v as usize)
    }

    /// `NAXISn` for `n` in `1..=NAXIS`.
    pub fn naxisn(&self, n: usize) -> Result<u64> {
        if n == 0 || n > 999 {
            return Err(FitsError::Value {
                keyword: format!("NAXIS{n}"),
                msg: "axis index out of range".into(),
            });
        }
        let key = format!("NAXIS{n}");
        let v = self.required_int(&key)?;
        if v < 0 {
            return Err(FitsError::Value {
                keyword: key,
                msg: format!("NAXIS{n} must be non-negative, got {v}"),
            });
        }
        Ok(v as u64)
    }

    /// All `NAXISn` values in order.
    pub fn axes(&self) -> Result<Vec<u64>> {
        let n = self.naxis()?;
        (1..=n).map(|i| self.naxisn(i)).collect()
    }

    /// `BZERO` defaulting to 0.0 (Sec.4.4.2.5).
    #[must_use]
    pub fn bzero(&self) -> f64 {
        self.optional_real("BZERO").unwrap_or(0.0)
    }

    /// `BSCALE` defaulting to 1.0 (Sec.4.4.2.5).
    #[must_use]
    pub fn bscale(&self) -> f64 {
        self.optional_real("BSCALE").unwrap_or(1.0)
    }

    /// `BLANK` (Sec.4.4.2.4) for integer images. Returns `None` if
    /// absent **or** if the current header has a floating-point
    /// `BITPIX` (in which case BLANK has no defined meaning per
    /// Sec.4.4.2.4 -- undefined floats are signalled by IEEE NaN). A
    /// non-integer BLANK value is also treated as absent.
    #[must_use]
    pub fn blank(&self) -> Option<i64> {
        if let Ok(b) = self.bitpix()
            && b < 0
        {
            return None;
        }
        self.optional_int("BLANK")
    }

    /// `BUNIT` (Sec.4.4.2.6). Returns `None` if absent.
    #[must_use]
    pub fn bunit(&self) -> Option<&str> {
        self.first("BUNIT").and_then(|c| match c {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// Mandatory integer keyword. Errors on absence or wrong type.
    pub fn required_int(&self, key: &str) -> Result<i64> {
        match self.first(key).ok_or_else(|| FitsError::MissingMandatory {
            keyword: key.into(),
        })? {
            Value::Integer(i) => Ok(*i),
            other => Err(FitsError::Value {
                keyword: key.into(),
                msg: format!("expected integer, found {other:?}"),
            }),
        }
    }

    /// Optional integer keyword. Returns `None` if absent. A
    /// floating-point value is accepted if and only if it is an exact
    /// integer fitting in `i64` -- many real-world headers write\n    /// supposedly-integer reserved keywords (`BLANK`, `PCOUNT`,\n    /// `GCOUNT`, `BITPIX`, `NAXIS*`) as `1.0` or `-32768.` instead of\n    /// the spec-strict integer literal.
    #[must_use]
    pub fn optional_int(&self, key: &str) -> Option<i64> {
        match self.first(key)? {
            Value::Integer(i) => Some(*i),
            Value::Real(r) => {
                if r.is_finite()
                    && r.fract() == 0.0
                    && *r >= i64::MIN as f64
                    && *r <= i64::MAX as f64
                {
                    Some(*r as i64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Optional real keyword. Integers are widened to `f64`.
    #[must_use]
    pub fn optional_real(&self, key: &str) -> Option<f64> {
        match self.first(key)? {
            Value::Integer(i) => Some(*i as f64),
            Value::Real(r) => Some(*r),
            _ => None,
        }
    }

    /// Optional string keyword. Returns the unquoted, untrimmed value.
    #[must_use]
    pub fn optional_string(&self, key: &str) -> Option<&str> {
        match self.first(key)? {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::header::Header;

    fn hdr(cards: &[&str]) -> Header {
        let mut bytes: Vec<u8> = Vec::new();
        for c in cards {
            let mut card = c.as_bytes().to_vec();
            card.resize(80, b' ');
            bytes.extend_from_slice(&card);
        }
        let mut end = b"END".to_vec();
        end.resize(80, b' ');
        bytes.extend_from_slice(&end);
        // Pad to 2880.
        while !bytes.len().is_multiple_of(2880) {
            bytes.push(b' ');
        }
        Header::parse(&bytes, 0).expect("parse").0
    }

    #[test]
    fn optional_int_accepts_real_with_zero_fraction() {
        // Many real-world writers emit `BLANK = -32768.` instead of
        // `-32768`. We must accept that as the integer value.
        let h = hdr(&["BLANK   =             -32768."]);
        assert_eq!(h.optional_int("BLANK"), Some(-32768));
    }

    #[test]
    fn optional_int_rejects_non_integer_real() {
        let h = hdr(&["FOO     =              1.5"]);
        assert_eq!(h.optional_int("FOO"), None);
    }

    #[test]
    fn blank_returns_none_for_float_bitpix() {
        // BLANK is undefined for floating-point images (Sec.4.4.2.4):
        // even if a writer accidentally emits it, we must ignore it.
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    0",
            "BLANK   =                  -1",
        ]);
        assert_eq!(h.blank(), None);
    }

    #[test]
    fn blank_returned_for_integer_bitpix() {
        let h = hdr(&[
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    0",
            "BLANK   =               -32768",
        ]);
        assert_eq!(h.blank(), Some(-32768));
    }
}
