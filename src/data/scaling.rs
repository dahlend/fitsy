//! Pixel scaling and `BLANK` handling (Standard Sec.4.4.2.4-Sec.4.4.2.5).

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
