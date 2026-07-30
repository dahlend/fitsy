//! Random Groups primary HDU (Standard Sec.6).
//!
//! A Random Groups primary HDU is signalled by `NAXIS1 = 0`,
//! `NAXIS >= 2`, and `GROUPS = T` in the primary header. The data
//! section consists of `GCOUNT` repetitions of:
//!
//! 1. `PCOUNT` *parameter* values (BITPIX-typed, big-endian);
//! 2. `NAXIS2 x NAXIS3 x ... x NAXISn` *data array* values.
//!
//! Random Groups was devised for radio interferometry visibilities.
//! Sec.6.4 discourages it for new files, but a lot of legacy data
//! still uses it.

use crate::data::encoding::{Bitpix, Pixel};
use crate::error::{FitsError, Result};
use crate::header::Header;

/// Description of one group parameter slot (Standard Sec.6.1.2).
///
/// `PTYPEn` names the parameter; `PSCALn` and `PZEROn` convert the
/// stored value to the physical one by eq. 6.1,
/// `physical = PZEROn + PSCALn x stored`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GroupParameter {
    /// 1-based parameter index, as in `PTYPEn`.
    pub index: usize,
    /// `PTYPEn`, trimmed; empty when the keyword is absent. Sec.6.1.2
    /// lets the same name repeat across slots, in which case the
    /// physical values are summed -- see
    /// [`RandomGroupsHdu::group_parameter_by_name`].
    pub name: String,
    /// `PSCALn`, default 1.0.
    pub pscal: f64,
    /// `PZEROn`, default 0.0.
    pub pzero: f64,
}

impl GroupParameter {
    /// Apply eq. 6.1 to a stored value.
    #[must_use]
    #[inline]
    pub fn physical(&self, stored: f64) -> f64 {
        self.pzero + self.pscal * stored
    }
}

/// A Random Groups primary HDU.
#[derive(Debug, Clone)]
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
    /// `PTYPEn`/`PSCALn`/`PZEROn` for each of the `PCOUNT` slots.
    parameters: Vec<GroupParameter>,
}

impl<'a> RandomGroupsHdu<'a> {
    /// Wrap a random-groups primary HDU.
    ///
    /// # Errors
    ///
    /// If the header is not random groups (Sec.6.1.1 requires
    /// `GROUPS = T` and `NAXIS1 = 0`), or `data` does not match the
    /// extent `GCOUNT`, `PCOUNT` and the axes imply.
    pub fn new(header: Header, data: &'a [u8]) -> Result<Self> {
        let bitpix = Bitpix::from_i64(header.bitpix()?)?;
        let naxis = header.naxis()?;
        if naxis < 2 {
            return Err(FitsError::Data(format!("NAXIS must be >= 2, got {naxis}")));
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
            data_per_group = data_per_group
                .checked_mul(n)
                .ok_or_else(|| FitsError::Data("NAXISn product overflowed u64".into()))?;
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
        let parameters = (1..=pcount as usize)
            .map(|i| GroupParameter {
                index: i,
                name: header
                    .optional_string(&format!("PTYPE{i}"))
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default(),
                pscal: header.optional_real(&format!("PSCAL{i}")).unwrap_or(1.0),
                pzero: header.optional_real(&format!("PZERO{i}")).unwrap_or(0.0),
            })
            .collect();
        Ok(Self {
            header,
            data,
            bitpix,
            pcount,
            data_per_group,
            gcount,
            parameters,
        })
    }

    #[must_use]
    /// The HDU's header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    #[must_use]
    /// Pixel encoding, from `BITPIX`.
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

    /// `PTYPEn`/`PSCALn`/`PZEROn` for each of the `PCOUNT` parameter
    /// slots, in order (Standard Sec.6.1.2).
    #[must_use]
    pub fn parameters(&self) -> &[GroupParameter] {
        &self.parameters
    }

    /// Distinct `PTYPEn` names, in order of first appearance, with
    /// unnamed slots skipped. A name that occurs more than once is
    /// listed once: Sec.6.1.2 makes the repeats components of a single
    /// higher-precision parameter.
    #[must_use]
    pub fn parameter_names(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for p in &self.parameters {
            if !p.name.is_empty() && !out.contains(&p.name.as_str()) {
                out.push(&p.name);
            }
        }
        out
    }

    /// Physical value of every parameter slot of group `g`, one entry
    /// per `PCOUNT` slot, with eq. 6.1 applied
    /// (`PZEROn + PSCALn x stored`).
    ///
    /// Slots are returned individually; the Sec.6.1.2 rule that repeated
    /// `PTYPEn` names are summed is *not* applied here, since the caller
    /// may want the components. Use
    /// [`Self::group_parameter_by_name`] for the summed value.
    pub fn group_parameters(&self, g: u64) -> Result<Vec<f64>> {
        let raw = self.group_parameters_raw(g)?;
        Ok(raw
            .iter()
            .zip(&self.parameters)
            .map(|(&v, p)| p.physical(v))
            .collect())
    }

    /// Physical value of the parameter called `name` in group `g`,
    /// or `None` if no `PTYPEn` carries that name.
    ///
    /// Sec.6.1.2: "If the `PTYPEn` keywords for more than one value of
    /// `n` have the same associated name in the value field, then the
    /// data value for the parameter of that name is to be obtained by
    /// adding the derived data values of the corresponding parameters."
    /// This is how AIPS splits a date across two slots to get more
    /// precision than `BITPIX` allows, so the sum is the only correct
    /// reading. Names are compared after trimming, case-sensitively.
    pub fn group_parameter_by_name(&self, g: u64, name: &str) -> Result<Option<f64>> {
        let want = name.trim();
        if !self.parameters.iter().any(|p| p.name == want) {
            return Ok(None);
        }
        let raw = self.group_parameters_raw(g)?;
        let sum = raw
            .iter()
            .zip(&self.parameters)
            .filter(|(_, p)| p.name == want)
            .map(|(&v, p)| p.physical(v))
            .sum();
        Ok(Some(sum))
    }

    /// Stored (unscaled) parameter values of group `g`, widened to
    /// `f64` whatever `BITPIX` is.
    fn group_parameters_raw(&self, g: u64) -> Result<Vec<f64>> {
        if g >= self.gcount {
            return Err(FitsError::Data(format!(
                "RandomGroupsHdu: group {g} out of range (n_groups = {})",
                self.gcount
            )));
        }
        let bsize = self.bitpix.byte_size();
        let group_bytes = ((self.pcount + self.data_per_group) as usize) * bsize;
        let off = (g as usize) * group_bytes;
        let params = &self.data[off..off + (self.pcount as usize) * bsize];
        Ok(params
            .chunks_exact(bsize)
            .map(|c| match self.bitpix {
                Bitpix::U8 => f64::from(u8::from_be_bytes([c[0]])),
                Bitpix::I16 => f64::from(i16::from_be_bytes([c[0], c[1]])),
                Bitpix::I32 => f64::from(i32::from_be_bytes([c[0], c[1], c[2], c[3]])),
                Bitpix::I64 => {
                    i64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64
                }
                Bitpix::F32 => f64::from(f32::from_be_bytes([c[0], c[1], c[2], c[3]])),
                Bitpix::F64 => f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]),
            })
            .collect())
    }

    /// Raw data bytes for the entire data section.
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        self.data
    }
}
