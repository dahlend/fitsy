//! Pixel scaling and `BLANK` handling (Standard Sec.4.4.2.4-Sec.4.4.2.5).

use crate::error::{FitsError, Result};

/// The parameters that convert a stored pixel into a physical value.
///
/// A physical value is `bzero + bscale * raw`, per Standard
/// Sec.4.4.2.5. An integer image may also declare a `blank` sentinel,
/// which Sec.4.4.2.4 marks as undefined; this type maps it to `NaN`.
#[derive(Debug, Clone, Copy)]
pub struct Scaling {
    /// `BZERO`, the additive offset. Defaults to 0.0.
    pub bzero: f64,
    /// `BSCALE`, the multiplicative factor. Defaults to 1.0.
    pub bscale: f64,
    /// Integer-image blank sentinel. `None` for floating-point images,
    /// where IEEE NaN serves as the undefined value.
    pub blank: Option<i64>,
}

impl Scaling {
    /// Apply `physical = BZERO + BSCALE * raw` (Standard Sec.4.4.2.5).
    /// All arithmetic is performed in `f64` to avoid intermediate
    /// rounding, even when `BZERO`/`BSCALE` are exact integers.
    #[inline]
    #[must_use]
    pub fn apply_int(&self, raw: i64) -> f64 {
        if let Some(blank) = self.blank
            && raw == blank
        {
            return f64::NAN;
        }
        self.bzero + self.bscale * (raw as f64)
    }

    /// Apply `physical = BZERO + BSCALE * raw` to a float pixel.
    ///
    /// There is no `BLANK` check: Sec.4.4.2.4 restricts the keyword to
    /// integer images, and an undefined float is already a NaN, which
    /// propagates through the arithmetic.
    #[inline]
    #[must_use]
    pub fn apply_real(&self, raw: f64) -> f64 {
        // NaN propagates through arithmetic.
        self.bzero + self.bscale * raw
    }

    /// Whether this scaling leaves a pixel unchanged.
    ///
    /// `BZERO = 0`, `BSCALE = 1` and no `BLANK` mean the stored value
    /// is the physical value, so the two read paths agree.
    #[inline]
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.bzero == 0.0 && self.bscale == 1.0 && self.blank.is_none()
    }

    /// Invert [`Self::apply_int`]: the integer to store for a physical
    /// value.
    ///
    /// This computes `(physical - BZERO) / BSCALE` and rounds to the
    /// nearest integer, half away from zero. A `NaN` physical value is
    /// undefined, and stores as the `BLANK` sentinel.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when `physical` is `NaN` and the header
    /// declares no `BLANK`, when `BSCALE` is zero, and when the
    /// rounded value does not fit `i64`.
    pub fn unapply_int(&self, physical: f64) -> Result<i64> {
        if physical.is_nan() {
            return self.blank.ok_or_else(|| {
                FitsError::Data(
                    "cannot store an undefined pixel in an integer image that \
                     declares no BLANK (Sec.4.4.2.4)"
                        .into(),
                )
            });
        }
        if self.bscale == 0.0 {
            return Err(FitsError::Data(
                "BSCALE is zero, so it cannot invert".into(),
            ));
        }
        let raw = ((physical - self.bzero) / self.bscale).round();
        if !raw.is_finite() || raw < i64::MIN as f64 || raw > i64::MAX as f64 {
            return Err(FitsError::Data(format!(
                "physical value {physical} scales to {raw}, which does not fit i64"
            )));
        }
        Ok(raw as i64)
    }

    /// Invert [`Self::apply_real`]: the float to store for a physical
    /// value.
    ///
    /// `NaN` passes through, because an undefined float pixel is
    /// already `NaN` and Sec.4.4.2.4 gives a float image no `BLANK`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when `BSCALE` is zero.
    pub fn unapply_real(&self, physical: f64) -> Result<f64> {
        if self.bscale == 0.0 {
            return Err(FitsError::Data(
                "BSCALE is zero, so it cannot invert".into(),
            ));
        }
        Ok((physical - self.bzero) / self.bscale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_scaling() {
        let s = Scaling {
            bzero: 0.0,
            bscale: 1.0,
            blank: None,
        };
        assert_eq!(s.apply_int(42), 42.0);
        assert_eq!(s.apply_real(1.5), 1.5);
    }

    #[test]
    fn unsigned_u16_offset() {
        let s = Scaling {
            bzero: 32_768.0,
            bscale: 1.0,
            blank: None,
        };
        assert_eq!(s.apply_int(-32_768), 0.0);
        assert_eq!(s.apply_int(0), 32_768.0);
        assert_eq!(s.apply_int(32_767), 65_535.0);
    }

    #[test]
    fn unapply_inverts_apply() {
        let s = Scaling {
            bzero: 100.0,
            bscale: 0.5,
            blank: Some(-32768),
        };
        for raw in [-5_i64, 0, 7, 1234] {
            let physical = s.apply_int(raw);
            assert_eq!(s.unapply_int(physical).unwrap(), raw);
        }
        // An undefined pixel round trips through BLANK.
        assert!(s.apply_int(-32768).is_nan());
        assert_eq!(s.unapply_int(f64::NAN).unwrap(), -32768);
        // Without a BLANK there is nowhere to put an undefined pixel.
        let no_blank = Scaling { blank: None, ..s };
        assert!(no_blank.unapply_int(f64::NAN).is_err());
        // Floats invert exactly for an exactly representable scale.
        assert_eq!(s.unapply_real(s.apply_real(3.25)).unwrap(), 3.25);
    }

    #[test]
    fn blank_becomes_nan() {
        let s = Scaling {
            bzero: 0.0,
            bscale: 1.0,
            blank: Some(-1),
        };
        assert!(s.apply_int(-1).is_nan());
        assert_eq!(s.apply_int(0), 0.0);
    }
}
