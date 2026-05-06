//! 2880-byte block constants and helpers (Standard Sec.3.1).

/// Logical FITS block size in bytes (Sec.3.1).
pub const BLOCK_SIZE: usize = 2880;

/// Number of 80-byte cards per header block.
pub const CARDS_PER_BLOCK: usize = BLOCK_SIZE / 80;

/// Round a byte count up to the next 2880-byte boundary.
#[inline]
#[must_use]
pub fn pad_to_block(n: u64) -> u64 {
    let r = n % BLOCK_SIZE as u64;
    if r == 0 {
        n
    } else {
        n + (BLOCK_SIZE as u64 - r)
    }
}

/// Number of complete blocks needed to hold `n` bytes.
#[inline]
#[must_use]
pub fn blocks_for_bytes(n: u64) -> u64 {
    pad_to_block(n) / BLOCK_SIZE as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_zero_is_zero() {
        assert_eq!(pad_to_block(0), 0);
    }

    #[test]
    fn pad_one_byte() {
        assert_eq!(pad_to_block(1), BLOCK_SIZE as u64);
    }

    #[test]
    fn pad_exact_block() {
        assert_eq!(pad_to_block(BLOCK_SIZE as u64), BLOCK_SIZE as u64);
    }

    #[test]
    fn pad_just_over() {
        assert_eq!(pad_to_block(BLOCK_SIZE as u64 + 1), 2 * BLOCK_SIZE as u64);
    }

    #[test]
    fn cards_per_block_is_36() {
        assert_eq!(CARDS_PER_BLOCK, 36);
    }
}
