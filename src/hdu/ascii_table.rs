//! ASCII table extension (`XTENSION = 'TABLE   '`, Standard Sec.7.2).
//!
//! # Purpose
//!
//! [`AsciiTableHdu`] reads one ASCII table. Such a table stores every
//! cell as a fixed-width text field at byte position `TBCOLn` within
//! its row. Each row is `NAXIS1` bytes wide, and the table holds
//! `NAXIS2` rows.
//!
//! # Layout
//!
//! [`AsciiFormat`] holds one parsed `TFORMn` value. Standard Table 15
//! defines five forms, and this module reads each one:
//!
//! - `Aw` -- a string of width `w`.
//! - `Iw` -- an integer of width `w`.
//! - `Fw.d` -- a fixed-point real with `d` decimal places.
//! - `Ew.d` -- a real with an `E` exponent.
//! - `Dw.d` -- a real with a `D` exponent.
//!
//! [`AsciiTableHdu::cell_value`] decodes one cell into an
//! [`AsciiCell`], and [`AsciiTableHdu::cell_bytes`] returns its raw
//! bytes instead.
//!
//! # Design constraints
//!
//! `TNULLn` is the only way an ASCII table marks a value undefined.
//! Sec.7.2.5 gives a blank numeric field the value zero, so a blank
//! field cannot carry that meaning.
//!
//! The comparison against `TNULLn` trims surrounding space on both
//! sides. Sec.7.2.5 does not say how the sentinel is justified in its
//! field, and writers right-justify a numeric field. A `TNULLn` that
//! is itself all spaces therefore matches a blank field alone, which
//! is how a writer opts a column out of the blank-means-zero rule.

use std::borrow::Cow;

use crate::error::{FitsError, Result};
use crate::header::Header;

/// One column of an ASCII table.
///
/// This pairs the parsed `TFORMn` of the column with its 1-based start
/// column from `TBCOLn`, and with the `TSCALn`, `TZEROn` and `TNULLn`
/// cards that apply to it. [`AsciiTableHdu::columns`] returns one of
/// these per column.
#[derive(Debug, Clone)]
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
    /// `TDISPn`, trimmed (Standard Sec.7.2.5). `None` when the card
    /// is absent.
    ///
    /// This is the recommended display format for the field, held as
    /// written. It is a Fortran edit descriptor such as `F8.3`. It
    /// describes presentation only, so this crate stores it and never
    /// interprets it.
    pub tdisp: Option<String>,
}

impl AsciiColumn {
    /// Width of this field, in bytes.
    #[must_use]
    pub fn width(&self) -> usize {
        self.format.width()
    }
}

/// One parsed `TFORMn` value of an ASCII table (Standard Table 15).
///
/// Each variant carries its field width `w`. The three real forms also
/// carry the count `d` of decimal places. [`AsciiFormat::parse`]
/// builds one from the card text.
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
    /// Field width `w` in characters, from `TFORMn`.
    pub fn width(self) -> usize {
        match self {
            Self::A(w) | Self::I(w) | Self::F(w, _) | Self::E(w, _) | Self::D(w, _) => w,
        }
    }

    /// Parse a `TFORMn` value such as `"A20"`, `"I10"`, `"F10.4"`,
    /// `"E15.7"` or `"D25.17"`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Value`] with keyword `TFORM`, in four cases:
    ///
    /// - `s` is empty.
    /// - The leading character is not `A`, `I`, `F`, `E` or `D`.
    /// - The width does not parse as a number.
    /// - An `F`, `E` or `D` form omits its `.d` part.
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
///
/// This borrows the data section of the table from the
/// [`FitsFile`](crate::FitsFile) that produced it, so it cannot
/// outlive that file. Every cell is a fixed-width text field, placed
/// by the `TBCOLn` of its column.
///
/// Reach a column through [`columns`](Self::columns) or
/// [`column_by_name`](Self::column_by_name), then decode with
/// [`cell_value`](Self::cell_value).
///
/// # Examples
///
/// ```
/// use fitsy::hdu::builder::AsciiColumnData;
/// use fitsy::{AsciiCell, AsciiFormat, AsciiTableBuilder};
/// use fitsy::{FitsFile, FitsWriter, Hdu, ImageBuilder};
///
/// let primary = ImageBuilder::new(Vec::<u64>::new(), Vec::<f32>::new())?
///     .primary(true)
///     .build()?;
/// let mut b = AsciiTableBuilder::new();
/// b.add_column(
///     "COUNT",
///     AsciiFormat::I(6),
///     AsciiColumnData::Int(vec![Some(12), None, Some(37)]),
/// )?;
/// // TNULL is the only marker of an undefined ASCII-table value.
/// b.tnull("---")?;
/// b.extname("CATALOG");
/// let hdu = b.build()?;
///
/// let mut buf: Vec<u8> = Vec::new();
/// let mut w = FitsWriter::new(&mut buf);
/// w.write_hdu(&primary)?;
/// w.write_hdu(&hdu)?;
/// w.finish()?;
///
/// let file = FitsFile::from_bytes(buf)?;
/// let Hdu::AsciiTable(tbl) = file.hdu_by_name("CATALOG", None)? else {
///     panic!("CATALOG is not an ASCII table");
/// };
/// let col = tbl.column_by_name("COUNT").expect("COUNT column");
/// assert_eq!(tbl.cell_value(0, col)?, Some(AsciiCell::Int(12)));
/// assert_eq!(tbl.cell_value(1, col)?, None);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct AsciiTableHdu<'a> {
    header: Header,
    data: Cow<'a, [u8]>,
    row_size: usize,
    n_rows: usize,
    columns: Vec<AsciiColumn>,
}

impl<'a> AsciiTableHdu<'a> {
    /// Build from a parsed header and the raw data slice.
    ///
    /// The `data` slice must hold `NAXIS1 * NAXIS2` bytes, without
    /// trailing block padding.
    ///
    /// # Errors
    ///
    /// - [`FitsError::MissingMandatory`] when the header omits
    ///   `BITPIX`, `NAXIS`, `NAXIS1`, `NAXIS2`, `TFIELDS`, or a
    ///   `TFORMn` or `TBCOLn` card for a declared column.
    /// - [`FitsError::Value`] when `BITPIX` is not 8, when `NAXIS` is
    ///   not 2, when a `TBCOLn` is below 1, or when a `TFORMn` string
    ///   fails to parse.
    /// - [`FitsError::Data`] when the row size times the row count
    ///   overflows `usize`, or when `data.len()` does not equal that
    ///   product.
    pub fn new(header: Header, data: impl Into<Cow<'a, [u8]>>) -> Result<Self> {
        let data = data.into();
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
                AsciiFormat::parse(&header.optional_string(&key_form).ok_or_else(|| {
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
            // Sec.7.2.4 puts no type restriction on the ASCII-table
            // TNULLn -- it is "the character string that represents an
            // undefined value for field n", full stop. (The integer-only
            // rule belongs to the *binary* table form, Sec.7.3.4.) It has
            // to apply to real fields here, because Sec.7.2.5 defines a
            // blank numeric field as zero, leaving TNULLn as the only
            // marker for an undefined float.
            let tnull = if matches!(format, AsciiFormat::A(_)) {
                None
            } else {
                header.optional_string(&format!("TNULL{i}"))
            };
            let tdisp = header
                .optional_string(&format!("TDISP{i}"))
                .map(|s| s.trim().to_string());
            columns.push(AsciiColumn {
                index: i,
                name,
                unit,
                start,
                format,
                tscal,
                tzero,
                tnull,
                tdisp,
            });
        }

        // Validate column placement: each field must fit inside the
        // row (Standard Sec.7.2.2). The spec permits overlap, but it is
        // a strong indicator of a malformed file (a TBCOL off-by-one
        // is one of the most common real-world ASCII-table errors),
        // so we report it as a warning-style error: callers who need
        // the lenient behavior can simply ignore the column-by-name
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
    /// The HDU's header.
    pub fn header(&self) -> &Header {
        &self.header
    }
    /// Consume the HDU and return its header and data section.
    ///
    /// This is the inverse of [`new`](Self::new), and the escape hatch
    /// for an interface that holds the two apart, such as
    /// [`FitsWriter::write_hdu_parts`](crate::FitsWriter::write_hdu_parts).
    /// The bytes are copied when they are borrowed and moved when they
    /// are owned.
    #[must_use]
    pub fn into_parts(self) -> (Header, Vec<u8>) {
        (self.header, self.data.into_owned())
    }

    /// Raw data bytes (the entire data section, `n_rows * row_size`).
    #[must_use]
    pub fn data_bytes(&self) -> &[u8] {
        &self.data
    }
    #[must_use]
    /// Bytes per row, from `NAXIS1`.
    pub fn row_size(&self) -> usize {
        self.row_size
    }
    #[must_use]
    /// Row count, from `NAXIS2`.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    #[must_use]
    /// Every column, in `TFORMn` order.
    pub fn columns(&self) -> &[AsciiColumn] {
        &self.columns
    }

    /// Find a column whose `TTYPEn` equals `name`.
    ///
    /// The comparison ignores ASCII case. The result is `None` when no
    /// column carries that name.
    #[must_use]
    pub fn column_by_name(&self, name: &str) -> Option<&AsciiColumn> {
        self.columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
    }

    /// Raw bytes of one row.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when `row` is not less than the row count.
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

    /// Raw bytes of one cell, located by its `TBCOLn` position.
    ///
    /// The `row` argument is a 0-based row index. The `col` argument
    /// describes the column, as [`columns`](Self::columns) returns it.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when `row` is not less than the row count.
    pub fn cell_bytes(&self, row: usize, col: &AsciiColumn) -> Result<&[u8]> {
        let row_bytes = self.row_bytes(row)?;
        // TBCOLn is 1-based.
        let start = col.start - 1;
        Ok(&row_bytes[start..start + col.width()])
    }

    /// Decode one cell into an [`AsciiCell`].
    ///
    /// The `row` argument is a 0-based row index. The `col` argument
    /// describes the column, as [`columns`](Self::columns) returns it.
    ///
    /// The result is `Ok(None)` only when the field matches `TNULLn`.
    /// Sec.7.2.5 gives a blank numeric field the value zero, so
    /// `TNULLn` is the only marker of an undefined value here.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] in three cases:
    ///
    /// - `row` is not less than the row count.
    /// - The field holds bytes that are not valid UTF-8.
    /// - A numeric field does not parse as its declared format.
    pub fn cell_value(&self, row: usize, col: &AsciiColumn) -> Result<Option<AsciiCell>> {
        let raw = self.cell_bytes(row, col)?;
        let s = std::str::from_utf8(raw).map_err(|_| {
            FitsError::Data(format!(
                "ASCII table row {row} col {} contains non-UTF8 bytes",
                col.index
            ))
        })?;
        // TNULLn match. Sec.7.2.5 places the sentinel in the field
        // without saying how it is justified, and writers right-justify
        // numerics, so compare with surrounding space removed. A TNULLn
        // that is itself all spaces still matches only a blank field,
        // which is how a writer opts a column out of the
        // blank-means-zero rule.
        if let Some(tn) = col.tnull.as_deref()
            && s.trim() == tn.trim()
        {
            return Ok(None);
        }
        match col.format {
            AsciiFormat::A(_) => Ok(Some(AsciiCell::Str(s.to_string()))),
            AsciiFormat::I(_) => {
                // Sec.7.2.5 "Integer fields": a single optional sign then
                // decimal digits, surrounding spaces not significant, and
                // "a blank field has value 0".
                let t = s.trim();
                let v: i64 = if t.is_empty() {
                    0
                } else {
                    t.parse().map_err(|_| {
                        FitsError::Data(format!(
                            "ASCII table row {row} col {}: not an integer: `{s}`",
                            col.index
                        ))
                    })?
                };
                let scaled = col.tzero + col.tscal * v as f64;
                if col.tscal == 1.0 && col.tzero == 0.0 {
                    Ok(Some(AsciiCell::Int(v)))
                } else {
                    Ok(Some(AsciiCell::Float(scaled)))
                }
            }
            AsciiFormat::F(_, d) | AsciiFormat::E(_, d) | AsciiFormat::D(_, d) => {
                let v = parse_fortran_real(s, d).ok_or_else(|| {
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

/// Parse one `Fw.d` / `Ew.d` / `Dw.d` field per Standard Sec.7.2.5
/// "Real fields". `d` is the format's fractional-digit count, needed for
/// the implicit decimal point of rule 2.
///
/// The rules, in order: discard trailing spaces and right-justify; read
/// an optionally-signed digit string containing at most one decimal
/// point; if there was no explicit point, place it immediately before
/// the rightmost `d` digits; then read an exponent introduced by `E`,
/// `D`, or -- rule 3(a) -- by a bare `+`/`-`. Anything else, including
/// an embedded space, is forbidden. A blank field is zero.
///
/// The two forms that ordinary float parsing gets wrong are the implicit
/// decimal point (`F10.3` over `12345` is 12.345, not 12345.0) and the
/// bare-sign exponent (`1.234+05` is 1.234e5). Both are legal and both
/// occur in older tables; the standard deprecates writing the first but
/// still defines how to read it.
fn parse_fortran_real(field: &str, d: usize) -> Option<f64> {
    let t = field.trim();
    if t.is_empty() {
        // "The default exponent is zero and a blank field has value zero."
        return Some(0.0);
    }
    let b = t.as_bytes();
    let mut i = 0;
    let negative = match b.first()? {
        b'-' => {
            i += 1;
            true
        }
        b'+' => {
            i += 1;
            false
        }
        _ => false,
    };

    // Mantissa: decimal digits with at most one embedded point.
    let mut digits = String::new();
    let mut fraction_digits: Option<usize> = None;
    while i < b.len() {
        match b[i] {
            c @ b'0'..=b'9' => {
                digits.push(c as char);
                if let Some(n) = fraction_digits.as_mut() {
                    *n += 1;
                }
                i += 1;
            }
            b'.' if fraction_digits.is_none() => {
                fraction_digits = Some(0);
                i += 1;
            }
            _ => break,
        }
    }
    if digits.is_empty() {
        return None;
    }

    // Exponent, if the numeric string was terminated rather than ended.
    // Held as i64 with saturating arithmetic: the field bytes are
    // untrusted, and `exponent - scale` on i32 overflows for a crafted
    // exponent near i32::MIN, panicking in debug builds.
    let mut exponent: i64 = 0;
    if i < b.len() {
        let tail = match b[i] {
            b'E' | b'e' | b'D' | b'd' => {
                i += 1;
                &t[i..]
            }
            // Rule 3(a): a bare sign introduces the exponent.
            b'+' | b'-' => &t[i..],
            _ => return None,
        };
        if tail.is_empty() {
            return None;
        }
        exponent = tail.parse::<i64>().ok()?;
    }

    // Rule 2: with no explicit point the implicit one sits immediately
    // before the rightmost `d` digits.
    let scale = fraction_digits.map_or_else(|| i64::try_from(d).unwrap_or(i64::MAX), |n| n as i64);
    // Let the float parser do the decimal scaling so the result is
    // correctly rounded rather than accumulated through multiplication.
    // The float parser saturates an out-of-range exponent to 0/inf, so
    // the saturating subtraction only ever affects values already far
    // outside f64 range.
    let sign = if negative { "-" } else { "" };
    format!("{sign}{digits}e{}", exponent.saturating_sub(scale))
        .parse::<f64>()
        .ok()
}

/// The decoded value of one ASCII-table cell.
///
/// The variant follows the [`AsciiFormat`] of the column: `A` gives
/// [`AsciiCell::Str`], `I` gives [`AsciiCell::Int`], and `F`, `E` or
/// `D` gives [`AsciiCell::Float`]. A cell that matches `TNULLn` yields
/// `None` from [`AsciiTableHdu::cell_value`] rather than a variant
/// here.
#[derive(Debug, Clone, PartialEq)]
pub enum AsciiCell {
    /// An `I`-format field with no scaling applied.
    Int(i64),
    /// An `F`/`E`/`D` field, or an `I` field with `TSCAL`/`TZERO`.
    Float(f64),
    /// An `A`-format character field, untrimmed.
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

    /// Sec.7.2.5 "Real fields", the whole rule set.
    ///
    /// Regression: the field was handed to `f64::from_str` after only
    /// mapping `D`->`E`, which silently ignored the implicit decimal
    /// point (rule 2), rejected a bare-sign exponent (rule 3a), and read
    /// a blank field as undefined instead of zero.
    #[test]
    fn fortran_real_fields_follow_sec_7_2_5() {
        // Rule 2: no explicit point, so it sits before the rightmost d
        // digits, with leading zeros assumed if the string is short.
        assert_eq!(parse_fortran_real("     12345", 3), Some(12.345));
        assert_eq!(parse_fortran_real("        12", 3), Some(0.012));
        assert_eq!(parse_fortran_real("       -12", 3), Some(-0.012));
        assert_eq!(parse_fortran_real("     12345", 0), Some(12345.0));
        // An explicit point wins over d.
        assert_eq!(parse_fortran_real("    12.345", 3), Some(12.345));
        assert_eq!(parse_fortran_real("   1234.5  ", 3), Some(1234.5));

        // Rule 3(a): a bare `+`/`-` introduces the exponent.
        assert_eq!(parse_fortran_real("  1.234+05", 3), Some(1.234e5));
        assert_eq!(parse_fortran_real("  1.234-05", 3), Some(1.234e-5));
        // Rule 3(b): `E` or `D`, with an optional sign.
        assert_eq!(parse_fortran_real(" 1.234E+05", 3), Some(1.234e5));
        assert_eq!(parse_fortran_real(" 1.234D+05", 3), Some(1.234e5));
        assert_eq!(parse_fortran_real("  1.234E05", 3), Some(1.234e5));
        assert_eq!(parse_fortran_real("  1.234E-5", 3), Some(1.234e-5));
        // The implicit point and an exponent compose.
        assert_eq!(parse_fortran_real("    1234+2", 3), Some(123.4));

        // "The default exponent is zero and a blank field has value zero."
        assert_eq!(parse_fortran_real("          ", 3), Some(0.0));
        assert_eq!(parse_fortran_real("", 3), Some(0.0));

        // "Characters other than those specified above, including
        // embedded space characters, are forbidden."
        assert_eq!(parse_fortran_real("  1.2 34  ", 3), None);
        assert_eq!(parse_fortran_real("      abcd", 3), None);
        assert_eq!(parse_fortran_real("     1.2.3", 3), None);
        assert_eq!(parse_fortran_real("         .", 3), None);
        assert_eq!(parse_fortran_real("    1.2E+ ", 3), None);
        assert_eq!(parse_fortran_real("       1E ", 3), None);
    }

    /// The field bytes are untrusted: an exponent near `i32::MIN`, or
    /// an oversized `d` combined with any negative exponent, must not
    /// overflow the implicit-point subtraction (a debug-build panic
    /// before it was widened to saturating i64 arithmetic). Values far
    /// outside f64 range saturate the way float parsing always does.
    #[test]
    fn extreme_exponents_saturate_instead_of_overflowing() {
        assert_eq!(parse_fortran_real("1.0E-2147483648", 1), Some(0.0));
        assert_eq!(
            parse_fortran_real("1.0E+2147483647", 1),
            Some(f64::INFINITY)
        );
        assert_eq!(parse_fortran_real("       5-2", 4_000_000_000), Some(0.0));
    }
}
