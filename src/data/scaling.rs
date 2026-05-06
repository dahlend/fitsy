//! Pixel scaling and `BLANK` handling (Standard Sec.4.4.2.4-Sec.4.4.2.5).

/// Scaling parameters for an image.
#[derive(Debug, Clone, Copy)]
pub struct Scaling {
    pub bzero: f64,
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
