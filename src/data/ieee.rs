//! IEEE 754 helpers that preserve NaN bit patterns (Standard
//! Sec.4.4.2.5: "the value of the NaN should not be modified").

#[inline]
#[must_use]
pub fn f32_from_be_bytes_preserving_nan(b: &[u8]) -> f32 {
    let bits = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
    f32::from_bits(bits)
}

#[inline]
#[must_use]
pub fn f64_from_be_bytes_preserving_nan(b: &[u8]) -> f64 {
    let bits = u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
    f64::from_bits(bits)
}

#[inline]
#[must_use]
pub fn f32_to_be_bytes_preserving_nan(x: f32) -> [u8; 4] {
    x.to_bits().to_be_bytes()
}

#[inline]
#[must_use]
pub fn f64_to_be_bytes_preserving_nan(x: f64) -> [u8; 8] {
    x.to_bits().to_be_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_signaling_nan_round_trip() {
        // Signalling NaN payload.
        let bits = 0x7fa0_0001_u32;
        let bytes = bits.to_be_bytes();
        let x = f32_from_be_bytes_preserving_nan(&bytes);
        assert!(x.is_nan());
        let out = f32_to_be_bytes_preserving_nan(x);
        assert_eq!(out, bytes);
    }

    #[test]
    fn f64_negative_zero_round_trip() {
        let bits = 0x8000_0000_0000_0000_u64;
        let bytes = bits.to_be_bytes();
        let x = f64_from_be_bytes_preserving_nan(&bytes);
        assert!(x == 0.0 && x.is_sign_negative());
        let out = f64_to_be_bytes_preserving_nan(x);
        assert_eq!(out, bytes);
    }
}
