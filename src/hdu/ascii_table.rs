//! ASCII Table extension (`XTENSION = 'TABLE   '`, Standard Sec.7.2).
//!
//! ASCII tables store every cell as a fixed-width text field placed
//! at byte position `TBCOLn` within each row. Rows are exactly
//! `NAXIS1` bytes wide; there are `NAXIS2` rows.
//!
//! Field formats supported (Standard Table 15):
//! `Aw` (string), `Iw` (integer), `Fw.d` (fixed-point real),
//! `Ew.d` (real with E exponent), `Dw.d` (real with D exponent).

use crate::error::{FitsError, Result};
use crate::header::Header;

/// One ASCII-table column descriptor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AsciiColumn {
    /// 1-based field index.
    pub index: usize,
    /// `TTYPEn` (column name), trimmed; empty string if absent.
    pub name: String,
    /// `TUNITn`, trimmed; empty string if absent.
    pub unit: String,
    /// `TBCOLn` -- 1-based starting byte within the row.
    pub start: usize,
    /// Field format (Standard Table 15).
    pub format: AsciiFormat,
    /// `TSCALn` (default 1.0).
    pub tscal: f64,
    /// `TZEROn` (default 0.0).
    pub tzero: f64,
    /// `TNULLn` -- string indicating an undefined value.
    pub tnull: Option<String>,
}

impl AsciiColumn {
    /// Width of this field, in bytes.
    #[must_use]
    pub fn width(&self) -> usize {
        self.format.width()
    }
}

/// ASCII-table field format codes (Standard Table 15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsciiFormat {
    /// `Aw` -- character string of width `w`.
    A(usize),
    /// `Iw` -- decimal integer in `w` columns.
    I(usize),
    /// `Fw.d` -- fixed-point real.
    F(usize, usize),
    /// `Ew.d` -- real with `E` exponent.
    E(usize, usize),
    /// `Dw.d` -- real with `D` exponent.
    D(usize, usize),
}

impl AsciiFormat {
    #[must_use]
    pub fn width(self) -> usize {
        match self {
            Self::A(w) | Self::I(w) | Self::F(w, _) | Self::E(w, _) | Self::D(w, _) => w,
        }
    }

    /// Parse a `TFORMn` value such as `"A20"`, `"I10"`, `"F10.4"`,
    /// `"E15.7"`, `"D25.17"`.
    pub fn parse(s: &str) -> Result<Self> {
        let t = s.trim();
        let mut chars = t.chars();
        let kind = chars
            .next()
            .ok_or_else(|| FitsError::Value {
                keyword: "TFORM".into(),
                msg: "empty TFORM".into(),
            })?
            .to_ascii_uppercase();
        let rest: String = chars.collect();
        match kind {
            'A' | 'I' => {
                let w: usize = rest.trim().parse().map_err(|_| FitsError::Value {
                    keyword: "TFORM".into(),
                    msg: format!("invalid width in `{s}`"),
                })?;
                Ok(if kind == 'A' { Self::A(w) } else { Self::I(w) })
            }
            'F' | 'E' | 'D' => {
                let (w_s, d_s) = rest.split_once('.').ok_or_else(|| FitsError::Value {
                    keyword: "TFORM".into(),
                    msg: format!("`{s}` requires `w.d` form"),
                })?;
                let w: usize = w_s.trim().parse().map_err(|_| FitsError::Value {
                    keyword: "TFORM".into(),
                    msg: format!("invalid width in `{s}`"),
                })?;
                let d: usize = d_s.trim().parse().map_err(|_| FitsError::Value {
                    keyword: "TFORM".into(),
                    msg: format!("invalid precision in `{s}`"),
                })?;
                Ok(match kind {
                    'F' => Self::F(w, d),
                    'E' => Self::E(w, d),
                    _ => Self::D(w, d),
                })
            }
            other => Err(FitsError::Value {
                keyword: "TFORM".into(),
                msg: format!("ASCII tables: unsupported format kind `{other}`"),
            }),
        }
    }
}

/// One ASCII table HDU.
#[derive(Debug)]
pub struct AsciiTableHdu<'a> {
    header: Header,
    data: &'a [u8],
    row_size: usize,
    n_rows: usize,
    columns: Vec<AsciiColumn>,
}

impl<'a> AsciiTableHdu<'a> {
    /// Build from a parsed header and the raw data slice (no padding).
    pub fn new(header: Header, data: &'a [u8]) -> Result<Self> {
        // Validate the mandatory shape: BITPIX=8, NAXIS=2.
        if header.bitpix()? != 8 {
            return Err(FitsError::Value {
                keyword: "BITPIX".into(),
                msg: "ASCII table requires BITPIX = 8".into(),
            });
        }
        if header.naxis()? != 2 {
            return Err(FitsError::Value {
                keyword: "NAXIS".into(),
                msg: "ASCII table requires NAXIS = 2".into(),
            });
        }
        let row_size = header.naxisn(1)? as usize;
        let n_rows = header.naxisn(2)? as usize;
        let needed = row_size
            .checked_mul(n_rows)
            .ok_or_else(|| FitsError::Data("ASCII table size overflows usize".into()))?;
        if data.len() != needed {
            return Err(FitsError::Data(format!(
                "ASCII table data slice {} bytes does not match row_size*n_rows = {needed}",
                data.len()
            )));
        }

        let tfields = header.required_int("TFIELDS")? as usize;
        let mut columns = Vec::with_capacity(tfields);
        for i in 1..=tfields {
            let key_form = format!("TFORM{i}");
            let key_bcol = format!("TBCOL{i}");
            let format =
                AsciiFormat::parse(header.optional_string(&key_form).ok_or_else(|| {
                    FitsError::MissingMandatory {
                        keyword: key_form.clone(),
                    }
                })?)?;
            let start_i = header.required_int(&key_bcol)?;
            if start_i < 1 {
                return Err(FitsError::Value {
                    keyword: key_bcol,
                    msg: format!("TBCOL must be >= 1, got {start_i}"),
                });
            }
            let start = start_i as usize;
            let name = header
                .optional_string(&format!("TTYPE{i}"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            let unit = header
                .optional_string(&format!("TUNIT{i}"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            let tscal = header.optional_real(&format!("TSCAL{i}")).unwrap_or(1.0);
            let tzero = header.optional_real(&format!("TZERO{i}")).unwrap_or(0.0);
            // Sec.7.2.4: TNULL is meaningful only for I-format columns.
            let tnull = if matches!(format, AsciiFormat::I(_)) {
                header
                    .optional_string(&format!("TNULL{i}"))
                    .map(String::from)
            } else {
                None
            };
            columns.push(AsciiColumn {
                index: i,
                name,
                unit,
                start,
                format,
                tscal,
                tzero,
                tnull,
            });
        }

        // Validate column placement: each field must fit inside the
        // row (Standard Sec.7.2.2). The spec permits overlap, but it is
        // a strong indicator of a malformed file (a TBCOL off-by-one
        // is one of the most common real-world ASCII-table errors),
        // so we report it as a warning-style error: callers who need
        // the lenient behaviour can simply ignore the column-by-name
        // accessors and read raw row bytes themselves.
        let mut sorted: Vec<&AsciiColumn> = columns.iter().collect();
        sorted.sort_by_key(|c| c.start);
        for pair in sorted.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let a_end = a.start + a.width();
            if a_end > b.start {
                return Err(FitsError::Value {
                    keyword: format!("TBCOL{}", b.index),
                    msg: format!(
                        "ASCII-table fields overlap: column {} ends at byte {} \
                         but column {} starts at byte {} (Sec.7.2.2)",
                        a.index, a_end, b.index, b.start
                    ),
                });
            }
        }
        for c in &columns {
            let end = c
                .start
                .checked_add(c.width())
                .ok_or_else(|| FitsError::Value {
                    keyword: format!("TBCOL{}", c.index),
                    msg: "TBCOL + width overflows".into(),
                })?;
            if end > row_size + 1 {
                return Err(FitsError::Value {
                    keyword: format!("TBCOL{}", c.index),
                    msg: format!(
                        "field {} (TBCOL={}, width={}) extends past row end (NAXIS1={})",
                        c.index,
                        c.start,
                        c.width(),
                        row_size,
                    ),
                });
            }
        }

        Ok(Self {
            header,
            data,
            row_size,
            n_rows,
            columns,
        })
    }

    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }
    /// Raw data bytes (the entire data section, `n_rows * row_size`).
    #[must_use]
    pub fn data_bytes(&self) -> &[u8] {
        self.data
    }
    #[must_use]
    pub fn row_size(&self) -> usize {
        self.row_size
    }
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    #[must_use]
    pub fn columns(&self) -> &[AsciiColumn] {
        &self.columns
    }

    /// Find a column by `TTYPEn`, case-insensitive.
    #[must_use]
    pub fn column_by_name(&self, name: &str) -> Option<&AsciiColumn> {
        self.columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
    }

    /// Raw bytes of one row.
    pub fn row_bytes(&self, row: usize) -> Result<&[u8]> {
        if row >= self.n_rows {
            return Err(FitsError::Data(format!(
                "row {row} out of range (n_rows = {})",
                self.n_rows
            )));
        }
        let start = row * self.row_size;
        Ok(&self.data[start..start + self.row_size])
    }

    /// Raw bytes of one cell (after `TBCOLn` placement).
    pub fn cell_bytes(&self, row: usize, col: &AsciiColumn) -> Result<&[u8]> {
        let row_bytes = self.row_bytes(row)?;
        // TBCOLn is 1-based.
        let start = col.start - 1;
        Ok(&row_bytes[start..start + col.width()])
    }

    /// Decoded value of one cell. Returns `None` for `TNULL` matches
    /// or fields that are entirely whitespace.
    pub fn cell_value(&self, row: usize, col: &AsciiColumn) -> Result<Option<AsciiCell>> {
        let raw = self.cell_bytes(row, col)?;
        let s = std::str::from_utf8(raw).map_err(|_| {
            FitsError::Data(format!(
                "ASCII table row {row} col {} contains non-UTF8 bytes",
                col.index
            ))
        })?;
        // TNULL match: compare the *raw* (non-trimmed) field text.
        if let Some(tn) = col.tnull.as_deref()
            && s.trim_end() == tn.trim_end()
        {
            return Ok(None);
        }
        match col.format {
            AsciiFormat::A(_) => Ok(Some(AsciiCell::Str(s.to_string()))),
            AsciiFormat::I(_) => {
                let t = s.trim();
                if t.is_empty() {
                    return Ok(None);
                }
                let v: i64 = t.parse().map_err(|_| {
                    FitsError::Data(format!(
                        "ASCII table row {row} col {}: not an integer: `{s}`",
                        col.index
                    ))
                })?;
                let scaled = col.tzero + col.tscal * v as f64;
                if col.tscal == 1.0 && col.tzero == 0.0 {
                    Ok(Some(AsciiCell::Int(v)))
                } else {
                    Ok(Some(AsciiCell::Float(scaled)))
                }
            }
            AsciiFormat::F(_, _) | AsciiFormat::E(_, _) | AsciiFormat::D(_, _) => {
                let t = s.trim();
                if t.is_empty() {
                    return Ok(None);
                }
                // FITS allows D-exponent; Rust's f64::from_str does not.
                let normalized = t.replace(['D', 'd'], "E");
                let v: f64 = normalized.parse().map_err(|_| {
                    FitsError::Data(format!(
                        "ASCII table row {row} col {}: not a real: `{s}`",
                        col.index
                    ))
                })?;
                Ok(Some(AsciiCell::Float(col.tzero + col.tscal * v)))
            }
        }
    }
}

/// Decoded value of one ASCII-table cell.
#[derive(Debug, Clone, PartialEq)]
pub enum AsciiCell {
    Int(i64),
    Float(f64),
    Str(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_formats() {
        assert_eq!(AsciiFormat::parse("A8").unwrap(), AsciiFormat::A(8));
        assert_eq!(AsciiFormat::parse("I10").unwrap(), AsciiFormat::I(10));
        assert_eq!(AsciiFormat::parse("F10.4").unwrap(), AsciiFormat::F(10, 4));
        assert_eq!(AsciiFormat::parse("E15.7").unwrap(), AsciiFormat::E(15, 7));
        assert_eq!(
            AsciiFormat::parse("D25.17").unwrap(),
            AsciiFormat::D(25, 17)
        );
        assert!(AsciiFormat::parse("X10").is_err());
        assert!(AsciiFormat::parse("F10").is_err());
    }
}
