//! `HCOMPRESS_1` tile decompression (R. White 1992; FITS standard
//! 2016 Sec.10.4.5; Pence et al. 2010 Sec.3.3).
//!
//! 1. **Decode** the bit-packed quadtree-coded payload into a 2-D
//!    array of integer H-transform coefficients.
//! 2. **Un-digitize** by multiplying every coefficient by the integer
//!    `SCALE` factor that was used during compression
//!    (lossy -- `SCALE > 1` discards low-order bits).
//! 3. **Inverse H-transform** (`hinv`) reconstructs the image, with
//!    optional `hsmooth` smoothing of coefficients between transform
//!    levels (enabled by the `SMOOTH` parameter; off by default).
//!
//! The decoder is generic over the coefficient type via the [`Coeff`]
//! trait so a single implementation services both 32-bit and 64-bit
//! payloads (`fits_hdecompress` vs `fits_hdecompress64` in cfitsio).
//! Output pixel widths supported: 1 / 2 / 4 / 8 byte signed integers
//! (i.e. `ZBITPIX` 8, 16, 32 or 64, plus quantized floats which feed
//! through the i32 path).

use std::ops::{Add, AddAssign, BitAnd, BitOr, BitOrAssign, BitXor, Neg, Shl, Shr, Sub};

use crate::error::{FitsError, Result};

/// Decoded `HCOMPRESS_1` parameters carried in `ZNAMEn`/`ZVALn`.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct HcompressParams {
    /// Quantisation scale factor (1 = lossless, 0 = trust payload).
    pub scale: i32,
    /// Apply `hsmooth` between inverse-transform levels.
    pub smooth: bool,
}

/// Decompress an `HCOMPRESS_1` tile payload into `dst`.
///
/// `dst` must be `bp * n_pixels` bytes long. The payload self-
/// describes its `nx x ny` shape and `scale`; we cross-check
/// `nx*ny == n_pixels` and that the scale matches the value supplied
/// via `ZVALn` (when the caller provides one).
///
/// `bp in {1, 2, 4}` decodes through the i32 path; `bp == 8` decodes
/// through the i64 path.
pub(super) fn decompress_into(
    payload: &[u8],
    bp: usize,
    n_pixels: usize,
    params: HcompressParams,
    dst: &mut [u8],
) -> Result<()> {
    if !matches!(bp, 1 | 2 | 4 | 8) {
        return Err(FitsError::Data(format!(
            "HCOMPRESS_1 only supports 1/2/4/8-byte integer pixels, got bp={bp}"
        )));
    }
    let expected = n_pixels
        .checked_mul(bp)
        .ok_or_else(|| FitsError::Data("HCOMPRESS_1 destination size overflows usize".into()))?;
    if dst.len() != expected {
        return Err(FitsError::Data(format!(
            "HCOMPRESS_1 destination is {} bytes, expected {expected}",
            dst.len()
        )));
    }
    if n_pixels == 0 {
        return Ok(());
    }
    if bp == 8 {
        decompress_typed::<i64>(payload, n_pixels, params, bp, dst)
    } else {
        decompress_typed::<i32>(payload, n_pixels, params, bp, dst)
    }
}

fn decompress_typed<T: Coeff>(
    payload: &[u8],
    n_pixels: usize,
    params: HcompressParams,
    bp: usize,
    dst: &mut [u8],
) -> Result<()> {
    let mut state = State::new(payload);
    let (nx, ny, file_scale) = state.read_decode_header()?;
    if (nx as usize).checked_mul(ny as usize) != Some(n_pixels) {
        return Err(FitsError::Data(format!(
            "HCOMPRESS_1 payload describes {nx}x{ny} = {} pixels, tile expects {n_pixels}",
            i64::from(nx) * i64::from(ny)
        )));
    }
    if params.scale != 0 && params.scale != file_scale {
        return Err(FitsError::Data(format!(
            "HCOMPRESS_1 SCALE mismatch: ZVALn says {}, payload says {file_scale}",
            params.scale
        )));
    }
    let scale = file_scale;

    let mut a: Vec<T> = vec![T::ZERO; n_pixels];
    let sumall = state.read_sumall()?;
    let nbitplanes = state.read_nbitplanes()?;
    dodecode(&mut state, &mut a, nx, ny, nbitplanes)?;
    a[0] = T::from_i64(sumall);

    if scale > 1 {
        let s = T::from_i64(i64::from(scale));
        for v in &mut a {
            *v = (*v).wrapping_mul(s);
        }
    }
    hinv(&mut a, nx, ny, params.smooth, scale);

    // Narrow to the requested pixel width (big-endian). For bp < 8
    // we keep only the low `bp` bytes of the i64, matching the
    // signed two's-complement truncation cfitsio does.
    for (i, v) in a.into_iter().enumerate() {
        let bytes = v.to_i64().to_be_bytes();
        dst[i * bp..(i + 1) * bp].copy_from_slice(&bytes[8 - bp..]);
    }
    Ok(())
}

// -- Coeff trait --------------------------------------------------

/// Integer type used for H-transform coefficients. Implemented for
/// `i32` (matches cfitsio's `fits_hdecompress`) and `i64` (matches
/// `fits_hdecompress64`).
pub(super) trait Coeff:
    Copy
    + PartialOrd
    + Ord
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + Shl<i32, Output = Self>
    + Shr<i32, Output = Self>
    + AddAssign
    + BitOrAssign
{
    const ZERO: Self;
    const ONE: Self;
    fn from_i64(v: i64) -> Self;
    fn to_i64(self) -> i64;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
}

impl Coeff for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    fn from_i64(v: i64) -> Self {
        v as Self
    }
    fn to_i64(self) -> i64 {
        i64::from(self)
    }
    fn wrapping_add(self, other: Self) -> Self {
        Self::wrapping_add(self, other)
    }
    fn wrapping_mul(self, other: Self) -> Self {
        Self::wrapping_mul(self, other)
    }
}

impl Coeff for i64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    fn from_i64(v: i64) -> Self {
        v
    }
    fn to_i64(self) -> i64 {
        self
    }
    fn wrapping_add(self, other: Self) -> Self {
        Self::wrapping_add(self, other)
    }
    fn wrapping_mul(self, other: Self) -> Self {
        Self::wrapping_mul(self, other)
    }
}

// -- bit input ----------------------------------------------------

const CODE_MAGIC: [u8; 2] = [0xDD, 0x99];

struct State<'a> {
    src: &'a [u8],
    pos: usize,
    buffer: u32,
    bits_to_go: i32,
}

impl<'a> State<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            pos: 0,
            buffer: 0,
            bits_to_go: 0,
        }
    }

    fn need(&self, n: usize) -> Result<()> {
        if self.pos + n > self.src.len() {
            Err(FitsError::Data(format!(
                "HCOMPRESS_1 payload truncated: need {n} bytes at offset {} of {}",
                self.pos,
                self.src.len()
            )))
        } else {
            Ok(())
        }
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        let s = &self.src[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_int_be(&mut self) -> Result<i32> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_long_be(&mut self) -> Result<i64> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn start_inputing_bits(&mut self) {
        self.buffer = 0;
        self.bits_to_go = 0;
    }
    fn input_bit(&mut self) -> Result<i32> {
        if self.bits_to_go == 0 {
            self.need(1)?;
            self.buffer = u32::from(self.src[self.pos]);
            self.pos += 1;
            self.bits_to_go = 8;
        }
        self.bits_to_go -= 1;
        Ok(((self.buffer >> self.bits_to_go) & 1) as i32)
    }
    fn input_nbits(&mut self, n: i32) -> Result<i32> {
        debug_assert!((1..=8).contains(&n), "n={n} must be in range 1..=8");
        if self.bits_to_go < n {
            self.need(1)?;
            self.buffer = (self.buffer << 8) | u32::from(self.src[self.pos]);
            self.pos += 1;
            self.bits_to_go += 8;
        }
        self.bits_to_go -= n;
        let mask = (1_u32 << n) - 1;
        Ok(((self.buffer >> self.bits_to_go) & mask) as i32)
    }
    fn input_nybble(&mut self) -> Result<i32> {
        self.input_nbits(4)
    }
    fn input_nnybble(&mut self, n: usize, out: &mut [u8]) -> Result<()> {
        for slot in out.iter_mut().take(n) {
            *slot = self.input_nybble()? as u8;
        }
        Ok(())
    }
    fn input_huffman(&mut self) -> Result<i32> {
        let mut c = self.input_nbits(3)?;
        if c < 4 {
            return Ok(1 << c);
        }
        c = self.input_bit()? | (c << 1);
        if c < 13 {
            return Ok(match c {
                8 => 3,
                9 => 5,
                10 => 10,
                11 => 12,
                12 => 15,
                _ => unreachable!(),
            });
        }
        c = self.input_bit()? | (c << 1);
        if c < 31 {
            return Ok(match c {
                26 => 6,
                27 => 7,
                28 => 9,
                29 => 11,
                30 => 13,
                _ => unreachable!(),
            });
        }
        c = self.input_bit()? | (c << 1);
        Ok(if c == 62 { 0 } else { 14 })
    }
    fn read_decode_header(&mut self) -> Result<(i32, i32, i32)> {
        let magic = self.read_bytes(2)?;
        if magic != CODE_MAGIC {
            return Err(FitsError::Data(format!(
                "HCOMPRESS_1 bad magic bytes {:02X} {:02X} (expected DD 99)",
                magic[0], magic[1]
            )));
        }
        let nx = self.read_int_be()?;
        let ny = self.read_int_be()?;
        let scale = self.read_int_be()?;
        if nx <= 0 || ny <= 0 {
            return Err(FitsError::Data(format!(
                "HCOMPRESS_1 declares non-positive dimensions {nx}x{ny}"
            )));
        }
        Ok((nx, ny, scale))
    }
    fn read_sumall(&mut self) -> Result<i64> {
        self.read_long_be()
    }
    fn read_nbitplanes(&mut self) -> Result<[u8; 3]> {
        let b = self.read_bytes(3)?;
        Ok([b[0], b[1], b[2]])
    }
}

// -- decode -------------------------------------------------------

fn dodecode<T: Coeff>(
    state: &mut State<'_>,
    a: &mut [T],
    nx: i32,
    ny: i32,
    nbitplanes: [u8; 3],
) -> Result<()> {
    let nel = (nx * ny) as usize;
    let nx2 = (nx + 1) / 2;
    let ny2 = (ny + 1) / 2;

    state.start_inputing_bits();
    qtree_decode(state, a, 0, ny, nx2, ny2, nbitplanes[0])?;
    qtree_decode(state, a, ny2 as usize, ny, nx2, ny / 2, nbitplanes[1])?;
    qtree_decode(
        state,
        a,
        (ny * nx2) as usize,
        ny,
        nx / 2,
        ny2,
        nbitplanes[1],
    )?;
    qtree_decode(
        state,
        a,
        (ny * nx2 + ny2) as usize,
        ny,
        nx / 2,
        ny / 2,
        nbitplanes[2],
    )?;

    if state.input_nybble()? != 0 {
        return Err(FitsError::Data(
            "HCOMPRESS_1 dodecode: bad bit plane values".into(),
        ));
    }
    state.start_inputing_bits();
    for v in a.iter_mut().take(nel) {
        if *v != T::ZERO && state.input_bit()? != 0 {
            *v = -*v;
        }
    }
    Ok(())
}

fn qtree_decode<T: Coeff>(
    state: &mut State<'_>,
    a: &mut [T],
    offset: usize,
    n: i32,
    nqx: i32,
    nqy: i32,
    nbitplanes: u8,
) -> Result<()> {
    if nqx == 0 || nqy == 0 {
        return Ok(());
    }
    let nqmax = nqx.max(nqy);
    let mut log2n = f64::from(nqmax).log2().round() as i32;
    if nqmax > (1 << log2n) {
        log2n += 1;
    }
    let nqx2 = ((nqx + 1) / 2) as usize;
    let nqy2 = ((nqy + 1) / 2) as usize;
    let mut scratch = vec![0_u8; nqx2 * nqy2];

    for bit in (0..i32::from(nbitplanes)).rev() {
        let b = state.input_nybble()?;
        if b == 0 {
            state.input_nnybble(nqx2 * nqy2, &mut scratch)?;
            qtree_bitins(&scratch, nqx, nqy, a, offset, n, bit);
        } else if b != 0xF {
            return Err(FitsError::Data(format!(
                "HCOMPRESS_1 qtree_decode: bad format code 0x{b:X}"
            )));
        } else {
            scratch[0] = state.input_huffman()? as u8;
            let mut nx = 1_i32;
            let mut ny = 1_i32;
            let mut nfx = nqx;
            let mut nfy = nqy;
            let mut c = 1_i32 << log2n;
            for _ in 1..log2n {
                c >>= 1;
                nx <<= 1;
                ny <<= 1;
                if nfx <= c {
                    nx -= 1;
                } else {
                    nfx -= c;
                }
                if nfy <= c {
                    ny -= 1;
                } else {
                    nfy -= c;
                }
                qtree_expand(state, &mut scratch, nx, ny)?;
            }
            qtree_bitins(&scratch, nqx, nqy, a, offset, n, bit);
        }
    }
    Ok(())
}

fn qtree_expand(state: &mut State<'_>, a: &mut [u8], nx: i32, ny: i32) -> Result<()> {
    qtree_copy_inplace(a, nx, ny);
    let n = (nx * ny) as usize;
    for v in a.iter_mut().take(n).rev() {
        if *v != 0 {
            *v = state.input_huffman()? as u8;
        }
    }
    Ok(())
}

fn qtree_copy_inplace(a: &mut [u8], nx: i32, ny: i32) {
    let nx2 = (nx + 1) / 2;
    let ny2 = (ny + 1) / 2;
    let n = ny;

    let mut k = ny2 * (nx2 - 1) + ny2 - 1;
    for i in (0..nx2).rev() {
        let mut s00 = 2 * (n * i + ny2 - 1);
        for _ in (0..ny2).rev() {
            a[s00 as usize] = a[k as usize];
            k -= 1;
            s00 -= 2;
        }
    }

    let mut i = 0_i32;
    while i < nx - 1 {
        let s00 = n * i;
        let s10 = s00 + n;
        let mut j = 0_i32;
        while j < ny - 1 {
            let v = a[(s00 + j) as usize];
            a[(s10 + j + 1) as usize] = v & 1;
            a[(s10 + j) as usize] = (v >> 1) & 1;
            a[(s00 + j + 1) as usize] = (v >> 2) & 1;
            a[(s00 + j) as usize] = (v >> 3) & 1;
            j += 2;
        }
        if j < ny {
            let v = a[(s00 + j) as usize];
            a[(s10 + j) as usize] = (v >> 1) & 1;
            a[(s00 + j) as usize] = (v >> 3) & 1;
        }
        i += 2;
    }
    if i < nx {
        let s00 = n * i;
        let mut j = 0_i32;
        while j < ny - 1 {
            let v = a[(s00 + j) as usize];
            a[(s00 + j + 1) as usize] = (v >> 2) & 1;
            a[(s00 + j) as usize] = (v >> 3) & 1;
            j += 2;
        }
        if j < ny {
            let v = a[(s00 + j) as usize];
            a[(s00 + j) as usize] = (v >> 3) & 1;
        }
    }
}

fn qtree_bitins<T: Coeff>(
    scratch: &[u8],
    nqx: i32,
    nqy: i32,
    a: &mut [T],
    offset: usize,
    n: i32,
    bit: i32,
) {
    let plane_val: T = T::ONE << bit;
    let mut k: usize = 0;

    let mut i = 0_i32;
    while i < nqx - 1 {
        let s00 = (n * i) as usize + offset;
        let mut j = 0_i32;
        while j < nqy - 1 {
            let v = scratch[k];
            apply_2x2(a, s00 + j as usize, n as usize, v, plane_val);
            k += 1;
            j += 2;
        }
        if j < nqy {
            let v = scratch[k];
            if v & 0b1000 != 0 {
                a[s00 + j as usize] |= plane_val;
            }
            if v & 0b0010 != 0 {
                a[s00 + j as usize + n as usize] |= plane_val;
            }
            k += 1;
        }
        i += 2;
    }
    if i < nqx {
        let s00 = (n * i) as usize + offset;
        let mut j = 0_i32;
        while j < nqy - 1 {
            let v = scratch[k];
            if v & 0b1000 != 0 {
                a[s00 + j as usize] |= plane_val;
            }
            if v & 0b0100 != 0 {
                a[s00 + j as usize + 1] |= plane_val;
            }
            k += 1;
            j += 2;
        }
        if j < nqy {
            let v = scratch[k];
            if v & 0b1000 != 0 {
                a[s00 + j as usize] |= plane_val;
            }
        }
    }
}

#[inline]
fn apply_2x2<T: Coeff>(a: &mut [T], s00: usize, n: usize, v: u8, plane_val: T) {
    if v & 0b0001 != 0 {
        a[s00 + n + 1] |= plane_val;
    }
    if v & 0b0010 != 0 {
        a[s00 + n] |= plane_val;
    }
    if v & 0b0100 != 0 {
        a[s00 + 1] |= plane_val;
    }
    if v & 0b1000 != 0 {
        a[s00] |= plane_val;
    }
}

// -- inverse H-transform ------------------------------------------

fn hinv<T: Coeff>(a: &mut [T], nx: i32, ny: i32, smooth: bool, scale: i32) {
    let nmax = nx.max(ny);
    let mut log2n = f64::from(nmax).log2().round() as i32;
    if nmax > (1 << log2n) {
        log2n += 1;
    }
    let mut tmp: Vec<T> = vec![T::ZERO; ((nmax + 1) / 2) as usize];

    let mut shift: i32 = 1;
    let mut bit0: T = T::ONE << (log2n - 1);
    let mut bit1: T = bit0 << 1;
    let bit2: T = bit0 << 2;
    let mut mask0: T = -bit0;
    let mut mask1: T = mask0 << 1;
    let mask2: T = mask0 << 2;
    let mut prnd0: T = bit0 >> 1;
    let mut prnd1: T = bit1 >> 1;
    let prnd2: T = bit2 >> 1;
    let mut nrnd0: T = prnd0 - T::ONE;
    let mut nrnd1: T = prnd1 - T::ONE;
    let nrnd2: T = prnd2 - T::ONE;

    let a0 = a[0];
    a[0] = a0.wrapping_add(if a0 >= T::ZERO { prnd2 } else { nrnd2 }) & mask2;

    let mut nxtop = 1_i32;
    let mut nytop = 1_i32;
    let mut nxf = nx;
    let mut nyf = ny;
    let mut c = 1_i32 << log2n;

    for k in (0..log2n).rev() {
        c >>= 1;
        nxtop <<= 1;
        nytop <<= 1;
        if nxf <= c {
            nxtop -= 1;
        } else {
            nxf -= c;
        }
        if nyf <= c {
            nytop -= 1;
        } else {
            nyf -= c;
        }
        if k == 0 {
            nrnd0 = T::ZERO;
            shift = 2;
        }

        for i in 0..nxtop {
            unshuffle(a, (ny * i) as usize, nytop, 1, &mut tmp);
        }
        for j in 0..nytop {
            unshuffle(a, j as usize, nxtop, ny as usize, &mut tmp);
        }

        if smooth {
            hsmooth(a, nxtop, nytop, ny, scale);
        }

        let oddx = nxtop % 2;
        let oddy = nytop % 2;
        let mut i = 0_i32;
        while i < nxtop - oddx {
            let mut s00 = (ny * i) as usize;
            let mut s10 = s00 + ny as usize;
            let mut j = 0_i32;
            while j < nytop - oddy {
                let h0 = a[s00];
                let mut hx = a[s10];
                let mut hy = a[s00 + 1];
                let mut hc = a[s10 + 1];

                hx = hx.wrapping_add(if hx >= T::ZERO { prnd1 } else { nrnd1 }) & mask1;
                hy = hy.wrapping_add(if hy >= T::ZERO { prnd1 } else { nrnd1 }) & mask1;
                hc = hc.wrapping_add(if hc >= T::ZERO { prnd0 } else { nrnd0 }) & mask0;

                let lowbit0 = hc & bit0;
                hx = if hx >= T::ZERO {
                    hx - lowbit0
                } else {
                    hx + lowbit0
                };
                hy = if hy >= T::ZERO {
                    hy - lowbit0
                } else {
                    hy + lowbit0
                };
                let lowbit1 = (hc ^ hx ^ hy) & bit1;
                let h0 = if h0 >= T::ZERO {
                    h0 + lowbit0 - lowbit1
                } else if lowbit0 == T::ZERO {
                    h0 + lowbit1
                } else {
                    h0 + lowbit0 - lowbit1
                };

                a[s10 + 1] = (h0 + hx + hy + hc) >> shift;
                a[s10] = (h0 + hx - hy - hc) >> shift;
                a[s00 + 1] = (h0 - hx + hy - hc) >> shift;
                a[s00] = (h0 - hx - hy + hc) >> shift;

                s00 += 2;
                s10 += 2;
                j += 2;
            }
            if oddy != 0 {
                let h0 = a[s00];
                let mut hx = a[s10];
                hx = hx.wrapping_add(if hx >= T::ZERO { prnd1 } else { nrnd1 }) & mask1;
                let lowbit1 = hx & bit1;
                let h0 = if h0 >= T::ZERO {
                    h0 - lowbit1
                } else {
                    h0 + lowbit1
                };
                a[s10] = (h0 + hx) >> shift;
                a[s00] = (h0 - hx) >> shift;
            }
            i += 2;
        }
        if oddx != 0 {
            let mut s00 = (ny * i) as usize;
            let mut j = 0_i32;
            while j < nytop - oddy {
                let h0 = a[s00];
                let mut hy = a[s00 + 1];
                hy = hy.wrapping_add(if hy >= T::ZERO { prnd1 } else { nrnd1 }) & mask1;
                let lowbit1 = hy & bit1;
                let h0 = if h0 >= T::ZERO {
                    h0 - lowbit1
                } else {
                    h0 + lowbit1
                };
                a[s00 + 1] = (h0 + hy) >> shift;
                a[s00] = (h0 - hy) >> shift;
                s00 += 2;
                j += 2;
            }
            if oddy != 0 {
                let h0 = a[s00];
                a[s00] = h0 >> shift;
            }
        }

        bit1 = bit0;
        bit0 = bit0 >> 1;
        mask1 = mask0;
        mask0 = mask0 >> 1;
        prnd1 = prnd0;
        prnd0 = prnd0 >> 1;
        nrnd1 = nrnd0;
        nrnd0 = prnd0 - T::ONE;
    }
    let _ = (mask1, mask2, bit1);
}

fn unshuffle<T: Coeff>(a: &mut [T], offset: usize, n: i32, n2: usize, tmp: &mut [T]) {
    let nhalf = ((n + 1) >> 1) as usize;
    for i in 0..(n as usize - nhalf) {
        tmp[i] = a[offset + n2 * (nhalf + i)];
    }
    for i in (0..nhalf).rev() {
        a[offset + n2 * (2 * i)] = a[offset + n2 * i];
    }
    for i in 0..(n as usize - nhalf) {
        a[offset + n2 * (2 * i + 1)] = tmp[i];
    }
}

// -- hsmooth ------------------------------------------------------

fn hsmooth<T: Coeff>(a: &mut [T], nxtop: i32, nytop: i32, ny: i32, scale: i32) {
    let smax = T::from_i64(i64::from(scale >> 1));
    if smax <= T::ZERO {
        return;
    }
    let neg_smax = -smax;
    let ny2 = (ny << 1) as usize;
    let ny_us = ny as usize;
    let three: T = T::from_i64(3);

    // Adjust hx (x-difference).
    let mut i = 2_i32;
    while i < nxtop - 2 {
        let mut s00 = ny_us * i as usize;
        let mut s10 = s00 + ny_us;
        let mut j = 0_i32;
        while j < nytop {
            let hm = a[s00 - ny2];
            let h0 = a[s00];
            let hp = a[s00 + ny2];
            let mut diff = hp - hm;
            let dmax = (cmp_min(hp - h0, h0 - hm).max(T::ZERO)) << 2;
            let dmin = (cmp_max(hp - h0, h0 - hm).min(T::ZERO)) << 2;
            if dmin < dmax {
                diff = clamp(diff, dmin, dmax);
                let mut s = diff - (a[s10] << 3);
                s = if s >= T::ZERO {
                    s >> 3
                } else {
                    (s + T::from_i64(7)) >> 3
                };
                s = clamp(s, neg_smax, smax);
                a[s10] += s;
            }
            s00 += 2;
            s10 += 2;
            j += 2;
            let _ = three;
        }
        i += 2;
    }
    // Adjust hy.
    let mut i = 0_i32;
    while i < nxtop {
        let mut s00 = ny_us * i as usize + 2;
        let mut j = 2_i32;
        while j < nytop - 2 {
            let hm = a[s00 - 2];
            let h0 = a[s00];
            let hp = a[s00 + 2];
            let mut diff = hp - hm;
            let dmax = (cmp_min(hp - h0, h0 - hm).max(T::ZERO)) << 2;
            let dmin = (cmp_max(hp - h0, h0 - hm).min(T::ZERO)) << 2;
            if dmin < dmax {
                diff = clamp(diff, dmin, dmax);
                let mut s = diff - (a[s00 + 1] << 3);
                s = if s >= T::ZERO {
                    s >> 3
                } else {
                    (s + T::from_i64(7)) >> 3
                };
                s = clamp(s, neg_smax, smax);
                a[s00 + 1] += s;
            }
            s00 += 2;
            j += 2;
        }
        i += 2;
    }
    // Adjust hc.
    let mut i = 2_i32;
    while i < nxtop - 2 {
        let mut s00 = ny_us * i as usize + 2;
        let mut s10 = s00 + ny_us;
        let mut j = 2_i32;
        while j < nytop - 2 {
            let hmm = a[s00 - ny2 - 2];
            let hpm = a[s00 + ny2 - 2];
            let hmp = a[s00 - ny2 + 2];
            let hpp = a[s00 + ny2 + 2];
            let h0 = a[s00];
            let mut diff = hpp + hmm - hmp - hpm;
            let hx2 = a[s10] << 1;
            let hy2 = a[s00 + 1] << 1;
            let m1 = cmp_min(
                (hpp - h0).max(T::ZERO) - hx2 - hy2,
                (h0 - hpm).max(T::ZERO) + hx2 - hy2,
            );
            let m2 = cmp_min(
                (h0 - hmp).max(T::ZERO) - hx2 + hy2,
                (hmm - h0).max(T::ZERO) + hx2 + hy2,
            );
            let dmax = cmp_min(m1, m2) << 4;
            let m1 = cmp_max(
                (hpp - h0).min(T::ZERO) - hx2 - hy2,
                (h0 - hpm).min(T::ZERO) + hx2 - hy2,
            );
            let m2 = cmp_max(
                (h0 - hmp).min(T::ZERO) - hx2 + hy2,
                (hmm - h0).min(T::ZERO) + hx2 + hy2,
            );
            let dmin = cmp_max(m1, m2) << 4;
            if dmin < dmax {
                diff = clamp(diff, dmin, dmax);
                let mut s = diff - (a[s10 + 1] << 6);
                s = if s >= T::ZERO {
                    s >> 6
                } else {
                    (s + T::from_i64(63)) >> 6
                };
                s = clamp(s, neg_smax, smax);
                a[s10 + 1] += s;
            }
            s00 += 2;
            s10 += 2;
            j += 2;
        }
        i += 2;
    }
}

#[inline]
fn cmp_min<T: Coeff>(a: T, b: T) -> T {
    if a <= b { a } else { b }
}
#[inline]
fn cmp_max<T: Coeff>(a: T, b: T) -> T {
    if a >= b { a } else { b }
}
#[inline]
fn clamp<T: Coeff>(v: T, lo: T, hi: T) -> T {
    cmp_min(cmp_max(v, lo), hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_payload(nx: i32, ny: i32) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&CODE_MAGIC);
        p.extend_from_slice(&nx.to_be_bytes());
        p.extend_from_slice(&ny.to_be_bytes());
        p.extend_from_slice(&1_i32.to_be_bytes());
        p.extend_from_slice(&0_i64.to_be_bytes());
        p.extend_from_slice(&[0_u8, 0, 0]);
        p.push(0x00);
        p
    }

    #[test]
    fn rejects_bad_magic() {
        let payload = vec![0_u8; 32];
        let mut dst = vec![0_u8; 8];
        assert!(decompress_into(&payload, 2, 4, HcompressParams::default(), &mut dst,).is_err());
    }

    #[test]
    fn rejects_bad_pixel_width() {
        let mut dst = vec![0_u8; 4];
        let r = decompress_into(&[], 3, 4, HcompressParams::default(), &mut dst);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_dst_size_mismatch() {
        let mut dst = vec![0_u8; 5];
        assert!(
            decompress_into(&[0xDD, 0x99], 2, 4, HcompressParams::default(), &mut dst).is_err()
        );
    }

    #[test]
    fn all_zero_tile_round_trip_i16() {
        let p = zero_payload(2, 2);
        let mut dst = vec![0_u8; 8];
        decompress_into(&p, 2, 4, HcompressParams::default(), &mut dst).unwrap();
        assert_eq!(dst, vec![0_u8; 8]);
    }

    #[test]
    fn all_zero_tile_round_trip_i32() {
        let p = zero_payload(2, 2);
        let mut dst = vec![0_u8; 16];
        decompress_into(&p, 4, 4, HcompressParams::default(), &mut dst).unwrap();
        assert_eq!(dst, vec![0_u8; 16]);
    }

    #[test]
    fn all_zero_tile_round_trip_i64() {
        let p = zero_payload(2, 2);
        let mut dst = vec![0_u8; 32];
        decompress_into(&p, 8, 4, HcompressParams::default(), &mut dst).unwrap();
        assert_eq!(dst, vec![0_u8; 32]);
    }
}
