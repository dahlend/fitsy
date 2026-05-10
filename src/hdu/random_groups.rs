//! Random Groups primary HDU (Standard Sec.6).
//!
//! A Random Groups primary HDU is signalled by `NAXIS1 = 0`,
//! `NAXIS >= 2`, and `GROUPS = T` in the primary header. The data
//! section consists of `GCOUNT` repetitions of:
//!
//! 1. `PCOUNT` *parameter* values (BITPIX-typed, big-endian);
//! 2. `NAXIS2 x NAXIS3 x ... x NAXISn` *data array* values.
//!
//! Random Groups was originally devised for radio interferometry
//! visibilities. Its use is discouraged for new files (Standard
//! Sec.6.4) but a substantial corpus of legacy data still uses it.

use crate::data::encoding::{Bitpix, Pixel};
use crate::error::{FitsError, Result};
use crate::header::Header;

/// A Random Groups primary HDU.
#[derive(Debug)]
pub struct RandomGroupsHdu<'a> {
    header: Header,
    data: &'a [u8],
    bitpix: Bitpix,
    /// Number of parameters per group (`PCOUNT`).
    pcount: u64,
    /// Number of data values per group (prod NAXIS2..NAXISn).
    data_per_group: u64,
    /// Number of groups (`GCOUNT`).
    gcount: u64,
}

impl<'a> RandomGroupsHdu<'a> {
    pub fn new(header: Header, data: &'a [u8]) -> Result<Self> {
        let bitpix = Bitpix::from_i64(header.bitpix()?)?;
        let naxis = header.naxis()?;
        if naxis < 2 {
            return Err(FitsError::Data(format!(
                "NAXIS must be >= 2, got {naxis}"
            )));
        }
        if header.naxisn(1)? != 0 {
            return Err(FitsError::Data("NAXIS1 must be 0".into()));
        }
        let mut data_per_group: u64 = 1;
        for i in 2..=naxis {
            let n = header.naxisn(i)?;
            if n == 0 {
                data_per_group = 0;
                break;
            }
            data_per_group = data_per_group.checked_mul(n).ok_or_else(|| {
                FitsError::Data("NAXISn product overflowed u64".into())
            })?;
        }
        let pcount = match header.first("PCOUNT") {
            Some(crate::header::Value::Integer(p)) if *p >= 0 => *p as u64,
            _ => 0,
        };
        let gcount = match header.first("GCOUNT") {
            Some(crate::header::Value::Integer(g)) if *g >= 1 => *g as u64,
            _ => 1,
        };
        let bytes_per_elem = bitpix.byte_size() as u64;
        let needed = bytes_per_elem
            .checked_mul(gcount)
            .and_then(|v| v.checked_mul(pcount.checked_add(data_per_group)?))
            .ok_or_else(|| FitsError::Data("data size overflowed u64".into()))?;
        if data.len() as u64 != needed {
            return Err(FitsError::Data(format!(
                "data slice {} bytes does not match expected {needed}",
                data.len()
            )));
        }
        Ok(Self {
            header,
            data,
            bitpix,
            pcount,
            data_per_group,
            gcount,
        })
    }

    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }

    #[must_use]
    pub fn bitpix(&self) -> Bitpix {
        self.bitpix
    }

    /// Number of parameters per group (`PCOUNT`).
    #[must_use]
    pub fn pcount(&self) -> u64 {
        self.pcount
    }

    /// Number of data values per group (prod `NAXIS2..NAXISn`).
    #[must_use]
    pub fn data_per_group(&self) -> u64 {
        self.data_per_group
    }

    /// Number of groups (`GCOUNT`).
    #[must_use]
    pub fn n_groups(&self) -> u64 {
        self.gcount
    }

    /// Read group `g` as raw native-typed values: returns the
    /// `(parameters, data_array)` pair, both decoded big-endian
    /// without `BZERO`/`BSCALE`/`PZERO`/`PSCAL` applied.
    ///
    /// `T` must match `BITPIX`.
    pub fn group_raw<T: Pixel>(&self, g: u64) -> Result<(Vec<T>, Vec<T>)> {
        if T::BITPIX != self.bitpix {
            return Err(FitsError::Data(format!(
                "RandomGroupsHdu::group_raw: T does not match BITPIX (have {:?})",
                self.bitpix
            )));
        }
        if g >= self.gcount {
            return Err(FitsError::Data(format!(
                "RandomGroupsHdu::group_raw: group {g} out of range (n_groups = {})",
                self.gcount
            )));
        }
        let bsize = self.bitpix.byte_size();
        let group_elements = self.pcount + self.data_per_group;
        let group_bytes = (group_elements as usize) * bsize;
        let off = (g as usize) * group_bytes;
        let slice = &self.data[off..off + group_bytes];
        let mut params = Vec::with_capacity(self.pcount as usize);
        let mut data = Vec::with_capacity(self.data_per_group as usize);
        for (i, chunk) in slice.chunks_exact(bsize).enumerate() {
            let v = T::from_be_bytes(chunk);
            if (i as u64) < self.pcount {
                params.push(v);
            } else {
                data.push(v);
            }
        }
        Ok((params, data))
    }

    /// Raw data bytes for the entire data section.
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        self.data
    }
}
