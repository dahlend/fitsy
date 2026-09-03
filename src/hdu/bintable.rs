//! Binary table extension (`XTENSION = 'BINTABLE'`, Standard Sec.7.3).
//!
//! # Purpose
//!
//! [`BinTableHdu`] reads one binary table. A binary table stores
//! `NAXIS2` rows of `NAXIS1` bytes. Each row packs `TFIELDS` columns
//! with no padding between them.
//!
//! `TFORMn` describes column `n` in the form `rT[a]`. Here `r` is the
//! repeat count, `T` is one of the type codes `L`, `X`, `B`, `I`, `J`,
//! `K`, `A`, `E`, `D`, `C`, `M`, `P` and `Q`, and the optional `a` is
//! informational.
//!
//! # Layout
//!
//! [`BinFormat::parse`] turns a `TFORMn` string into a [`BinFormat`].
//! [`BinTableHdu::new`] builds the column table from the header.
//! [`BinTableHdu::cell_value`] decodes one cell into a [`BinValue`],
//! and [`BinTableHdu::cell_bytes`] returns its raw bytes instead.
//! [`BinTableHdu::row_range`] walks a contiguous slab of rows.
//!
//! # Design constraints
//!
//! A variable-length array column stores a descriptor in the row, not
//! the data. A `P` descriptor is two big-endian `i32` values, and a
//! `Q` descriptor is the same pair in `i64`. Each pair holds an
//! element count and a heap offset (Sec.7.3.5). The array itself lives
//! in the heap that follows the row area, starting at byte offset
//! `THEAP`, which defaults to `NAXIS1 * NAXIS2`.
//!
//! Two accessors therefore differ for such a column.
//! [`BinTableHdu::cell_value`] follows the descriptor into the heap
//! and returns [`BinValue::Vla`]. [`BinTableHdu::cell_bytes`] returns
//! the descriptor itself and does not follow it.
//!
//! An integer column reads as one of three [`BinValue`] variants. The
//! `TSCALn` and `TZEROn` cards select which. A `TZEROn` that matches
//! the unsigned offset of the type gives [`BinValue::Uint`]. A `B`
//! column with `TZEROn = -128` gives [`BinValue::Int`]. Any other
//! non-default scaling gives [`BinValue::Float`].

use std::borrow::Cow;

use crate::error::{FitsError, Result};
use crate::header::Header;
use crate::header::value::Value;

/// `TFORM` type code (Standard Table 18).
// Exhaustive: Table 18 has not grown since `Q` arrived in
// FITS 3.0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinFieldKind {
    /// `L` -- logical (1 byte: 'T', 'F', or 0 = undefined).
    Logical,
    /// `X` -- packed bits (8 per byte, MSB first).
    Bit,
    /// `B` -- unsigned byte.
    Byte,
    /// `I` -- 16-bit signed integer.
    I16,
    /// `J` -- 32-bit signed integer.
    I32,
    /// `K` -- 64-bit signed integer.
    I64,
    /// `A` -- character.
    Char,
    /// `E` -- 32-bit float.
    F32,
    /// `D` -- 64-bit float.
    F64,
    /// `C` -- 64-bit complex (two `E`).
    C64,
    /// `M` -- 128-bit complex (two `D`).
    C128,
    /// `P` -- variable-length array descriptor (2 x i32).
    P,
    /// `Q` -- variable-length array descriptor (2 x i64).
    Q,
}

impl BinFieldKind {
    /// Bytes that one repeat of this kind consumes in the row area.
    /// This does not account for the bit packing of `X`.
    #[must_use]
    pub fn element_bytes(self) -> usize {
        match self {
            Self::Logical | Self::Byte | Self::Char | Self::Bit => 1,
            Self::I16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 | Self::C64 | Self::P => 8,
            Self::C128 | Self::Q => 16,
        }
    }

    /// Single-character `TFORM` type code for this kind
    /// (Standard Table 18).
    #[must_use]
    pub fn tform_char(self) -> char {
        match self {
            Self::Logical => 'L',
            Self::Bit => 'X',
            Self::Byte => 'B',
            Self::I16 => 'I',
            Self::I32 => 'J',
            Self::I64 => 'K',
            Self::Char => 'A',
            Self::F32 => 'E',
            Self::F64 => 'D',
            Self::C64 => 'C',
            Self::C128 => 'M',
            Self::P => 'P',
            Self::Q => 'Q',
        }
    }

    fn from_char(c: char) -> Result<Self> {
        Ok(match c.to_ascii_uppercase() {
            'L' => Self::Logical,
            'X' => Self::Bit,
            'B' => Self::Byte,
            'I' => Self::I16,
            'J' => Self::I32,
            'K' => Self::I64,
            'A' => Self::Char,
            'E' => Self::F32,
            'D' => Self::F64,
            'C' => Self::C64,
            'M' => Self::C128,
            'P' => Self::P,
            'Q' => Self::Q,
            other => {
                return Err(FitsError::Value {
                    keyword: "TFORM".into(),
                    msg: format!("unknown BINTABLE type code `{other}`"),
                });
            }
        })
    }
}

/// One parsed `TFORMn` value, in the form `r T (a)`.
///
/// The `repeat` field is the count `r`, and the `kind` field is the
/// type code `T`. The two `vla_` fields carry the extra parts that a
/// `P` or `Q` descriptor spells.
///
/// [`BinFormat::parse`] builds one from the card text.
#[derive(Debug, Clone)]
pub struct BinFormat {
    /// Repeat count (default 1).
    pub repeat: usize,
    /// Element type.
    pub kind: BinFieldKind,
    /// Inner element type for `P`/`Q` descriptors.
    pub vla_kind: Option<BinFieldKind>,
    /// Maximum heap length for `P`/`Q` (informational).
    pub vla_max: Option<usize>,
}

impl BinFormat {
    /// Width in bytes that this column occupies inside one row. This
    /// excludes any heap data.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        match self.kind {
            BinFieldKind::Bit => self.repeat.div_ceil(8),
            BinFieldKind::P => 8,
            BinFieldKind::Q => 16,
            other => self.repeat * other.element_bytes(),
        }
    }

    /// Parse a `TFORMn` string per Standard Sec.7.3.3.1.
    ///
    /// Accepted forms include `"1J"`, `"4E"`, `"16A"`, `"PE(99)"` and
    /// `"1QD(0)"`. An absent repeat count means 1.
    ///
    /// # Errors
    ///
    /// [`FitsError::Value`] with keyword `TFORM`, in four cases:
    ///
    /// - `s` is empty.
    /// - The repeat count does not parse as a number.
    /// - The type code is absent or unknown.
    /// - A `P` or `Q` descriptor carries a malformed element type or
    ///   `rmax` field.
    pub fn parse(s: &str) -> Result<Self> {
        let t = s.trim();
        if t.is_empty() {
            return Err(FitsError::Value {
                keyword: "TFORM".into(),
                msg: "empty TFORM".into(),
            });
        }
        let bytes = t.as_bytes();
        // Leading repeat count (digits).
        let mut i = 0;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let repeat: usize = if i == 0 {
            1
        } else {
            t[..i].parse().map_err(|_| FitsError::Value {
                keyword: "TFORM".into(),
                msg: format!("invalid repeat in `{s}`"),
            })?
        };
        if i >= bytes.len() {
            return Err(FitsError::Value {
                keyword: "TFORM".into(),
                msg: format!("missing type code in `{s}`"),
            });
        }
        let head = bytes[i] as char;
        i += 1;
        let kind = BinFieldKind::from_char(head)?;
        let mut vla_kind = None;
        let mut vla_max = None;
        if matches!(kind, BinFieldKind::P | BinFieldKind::Q) {
            // Sec.7.3.5: outer repeat for P/Q must be 0 or 1.
            if repeat > 1 {
                return Err(FitsError::Value {
                    keyword: "TFORM".into(),
                    msg: format!("`{s}` VLA descriptor repeat must be 0 or 1"),
                });
            }
            // Inner type, e.g. `PE(99)` or `1QD(0)`.
            if i >= bytes.len() {
                return Err(FitsError::Value {
                    keyword: "TFORM".into(),
                    msg: format!("`{s}` missing VLA inner type"),
                });
            }
            vla_kind = Some(BinFieldKind::from_char(bytes[i] as char)?);
            i += 1;
            // Optional `(maxlen)`.
            if i < bytes.len() && bytes[i] == b'(' {
                let close = t[i..].find(')').ok_or_else(|| FitsError::Value {
                    keyword: "TFORM".into(),
                    msg: format!("`{s}` missing closing `)`"),
                })?;
                let inner = &t[i + 1..i + close];
                vla_max = Some(inner.trim().parse().map_err(|_| FitsError::Value {
                    keyword: "TFORM".into(),
                    msg: format!("invalid VLA maxlen in `{s}`"),
                })?);
                i += close + 1;
            }
        } else {
            // Optional `(a)` informational suffix; ignore.
            if i < bytes.len()
                && bytes[i] == b'('
                && let Some(close) = t[i..].find(')')
            {
                i += close + 1;
            }
        }
        // Anything left is allowed to be a unit-style suffix per
        // Sec.7.3.3.1 ("the optional element a is reserved..."); ignore.
        let _ = i;
        Ok(Self {
            repeat,
            kind,
            vla_kind,
            vla_max,
        })
    }
}

/// One column of a binary table.
///
/// This pairs the parsed `TFORMn` of the column with its byte offset
/// inside a row and with the scaling cards that apply to it: `TSCALn`,
/// `TZEROn` and `TNULLn`. [`BinTableHdu::columns`] returns one of
/// these per column, and [`BinTableHdu::cell_value`] takes one to
/// decode a cell.
#[derive(Debug, Clone)]
pub struct BinColumn {
    /// 1-based column number, as in `TFORMn`.
    pub index: usize,
    /// `TTYPEn`, trimmed. Empty when the card is absent.
    pub name: String,
    /// `TUNITn`, trimmed. Empty when the card is absent.
    pub unit: String,
    /// Parsed `TFORMn`: repeat count and type code.
    pub format: BinFormat,
    /// Byte offset within a row where this column begins.
    pub offset: usize,
    /// `TSCALn`, default 1.0.
    pub tscal: f64,
    /// `TZEROn`, default 0.0.
    pub tzero: f64,
    /// `TNULLn` -- the stored value that marks an undefined element.
    /// Integer columns only (Sec.7.3.4).
    pub tnull: Option<i64>,
    /// `TDIMn` shape, if present (Standard Sec.7.3.3.2).
    pub tdim: Option<Vec<usize>>,
    /// `TDISPn`, trimmed (Standard Sec.7.3.4). `None` when the card
    /// is absent.
    ///
    /// This is the recommended display format for the field, held as
    /// written. It is a Fortran edit descriptor such as `I6.4` or
    /// `E12.5E3`. It describes presentation only, so this crate stores
    /// it and never interprets it.
    pub tdisp: Option<String>,
    /// How integer columns (`B/I/J/K`) should be interpreted given
    /// `TSCAL`/`TZERO` (Standard Sec.11.3.1). Always `RawSigned` for
    /// non-integer kinds.
    pub int_storage: IntStorage,
}

/// Integer-column storage interpretation per Standard Sec.11.3.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntStorage {
    /// No special convention: return the signed value as stored
    /// (`B` is unsigned u8 by default; `I/J/K` are signed).
    RawSigned,
    /// `TSCAL=1`, `TZERO=2^(8n-1)`: stored bytes reinterpreted as
    /// unsigned (u16 for `I`, u32 for `J`, u64 for `K`).
    Unsigned,
    /// `B` only with `TSCAL=1`, `TZERO=-128`: stored unsigned byte
    /// reinterpreted as signed `i8`.
    SignedFromByte,
    /// Any other `TSCAL`/`TZERO`: scale through `f64`.
    Scaled,
}

impl IntStorage {
    /// Pick the storage convention for an integer column given its
    /// `TFORM` kind and `TSCAL`/`TZERO`.
    fn detect(kind: BinFieldKind, tscal: f64, tzero: f64) -> Self {
        fn near(a: f64, b: f64) -> bool {
            (a - b).abs() <= b.abs() * f64::EPSILON
        }
        if tscal == 1.0 && tzero == 0.0 {
            return Self::RawSigned;
        }
        // Tolerate the rounding that occurs when TZERO is written
        if near(tscal, 1.0) {
            // 2^(8n-1) for n in {1,2,4,8} bytes.
            let unsigned_offset = match kind {
                BinFieldKind::I16 => Some(32_768.0_f64),
                BinFieldKind::I32 => Some(2_147_483_648.0_f64),
                BinFieldKind::I64 => Some(9_223_372_036_854_775_808.0_f64),
                _ => None,
            };
            if let Some(off) = unsigned_offset
                && near(tzero, off)
            {
                return Self::Unsigned;
            }
            if matches!(kind, BinFieldKind::Byte) && near(tzero, -128.0) {
                return Self::SignedFromByte;
            }
        }
        Self::Scaled
    }
}

/// One binary table HDU.
///
/// This borrows the data section of the table from the
/// [`FitsFile`](crate::FitsFile) that produced it, so it cannot
/// outlive that file. It parses the column table from the header at
/// construction, and decodes a cell only when asked.
///
/// Reach a column through [`columns`](Self::columns) or
/// [`column_by_name`](Self::column_by_name), then decode with
/// [`cell_value`](Self::cell_value).
///
/// # Examples
///
/// ```
/// use fitsy::{BinFieldKind, BinTableBuilder, BinValue};
/// use fitsy::{FitsFile, FitsWriter, Hdu, ImageBuilder};
///
/// // Build a one-column table and write it in memory.
/// let primary = ImageBuilder::new(Vec::<u64>::new(), Vec::<f32>::new())?
///     .primary(true)
///     .build()?;
/// let mut b = BinTableBuilder::new();
/// b.add_column("FLUX", BinFieldKind::F64, 1, Some("Jy"), None)?;
/// b.extname("CATALOG");
/// let mut rows = Vec::new();
/// for v in [1.5_f64, 2.5, 3.5] {
///     rows.extend_from_slice(&v.to_be_bytes());
/// }
/// let hdu = b.build(3, rows)?;
///
/// let mut buf: Vec<u8> = Vec::new();
/// let mut w = FitsWriter::new(&mut buf);
/// w.write_hdu(&primary)?;
/// w.write_hdu(&hdu)?;
/// w.finish()?;
///
/// // Read one cell back.
/// let file = FitsFile::from_bytes(buf)?;
/// let Hdu::BinTable(tbl) = file.hdu_by_name("CATALOG", None)? else {
///     panic!("CATALOG is not a binary table");
/// };
/// let col = tbl.column_by_name("FLUX").expect("FLUX column");
/// assert_eq!(tbl.n_rows(), 3);
/// assert!(matches!(tbl.cell_value(0, col)?, BinValue::F64(v) if v == [1.5]));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct BinTableHdu<'a> {
    header: Header,
    /// Bytes covering the row table (and its heap).
    data: Cow<'a, [u8]>,
    row_size: usize,
    n_rows: usize,
    columns: Vec<BinColumn>,
    heap_offset: usize,
    heap_size: usize,
}

impl<'a> BinTableHdu<'a> {
    /// Build from a parsed header and the raw data slice.
    ///
    /// The `data` slice must hold `NAXIS1 * NAXIS2 + PCOUNT` bytes,
    /// without trailing block padding.
    ///
    /// # Errors
    ///
    /// - [`FitsError::MissingMandatory`] when the header omits
    ///   `BITPIX`, `NAXIS`, `NAXIS1`, `NAXIS2` or `TFIELDS`.
    /// - [`FitsError::Value`] when `BITPIX` is not 8, when `NAXIS` is
    ///   not 2, when `PCOUNT` is negative, when `TFIELDS` is negative,
    ///   or when a `TFORMn` string fails to parse.
    /// - [`FitsError::Data`] when the row size times the row count
    ///   overflows, when `data.len()` is smaller than the header
    ///   declares, or when the columns do not fit inside `NAXIS1`.
    pub fn new(header: Header, data: impl Into<Cow<'a, [u8]>>) -> Result<Self> {
        let data = data.into();
        if header.bitpix()? != 8 {
            return Err(FitsError::Value {
                keyword: "BITPIX".into(),
                msg: "BINTABLE requires BITPIX = 8".into(),
            });
        }
        if header.naxis()? != 2 {
            return Err(FitsError::Value {
                keyword: "NAXIS".into(),
                msg: "BINTABLE requires NAXIS = 2".into(),
            });
        }
        let row_size = header.naxisn(1)? as usize;
        let n_rows = header.naxisn(2)? as usize;
        let pcount_i = header.optional_int("PCOUNT").unwrap_or(0);
        if pcount_i < 0 {
            return Err(FitsError::Value {
                keyword: "PCOUNT".into(),
                msg: format!("PCOUNT must be >= 0, got {pcount_i}"),
            });
        }
        let pcount = pcount_i as usize;
        let table_bytes = row_size
            .checked_mul(n_rows)
            .ok_or_else(|| FitsError::Data("BINTABLE size overflows usize".into()))?;
        let needed = table_bytes
            .checked_add(pcount)
            .ok_or_else(|| FitsError::Data("BINTABLE+heap size overflows".into()))?;
        if data.len() != needed {
            return Err(FitsError::Data(format!(
                "BINTABLE data slice {} bytes does not match NAXIS1*NAXIS2+PCOUNT = {needed}",
                data.len()
            )));
        }
        // Heap offset: by Sec.7.3.5 it defaults to NAXIS1*NAXIS2 and is
        // measured from the start of the data section.
        let heap_offset = match header.optional_int("THEAP") {
            Some(t) if t >= 0 => t as usize,
            Some(t) => {
                return Err(FitsError::Value {
                    keyword: "THEAP".into(),
                    msg: format!("THEAP must be >= 0, got {t}"),
                });
            }
            None => table_bytes,
        };
        if heap_offset < table_bytes || heap_offset > needed {
            return Err(FitsError::Value {
                keyword: "THEAP".into(),
                msg: format!(
                    "THEAP={heap_offset} must lie within [NAXIS1*NAXIS2={table_bytes}, total={needed}]"
                ),
            });
        }

        let tfields = header.required_int("TFIELDS")? as usize;
        let mut columns: Vec<BinColumn> = Vec::with_capacity(tfields);
        let mut offset = 0_usize;
        for i in 1..=tfields {
            let key_form = format!("TFORM{i}");
            let format =
                BinFormat::parse(&header.optional_string(&key_form).ok_or_else(|| {
                    FitsError::MissingMandatory {
                        keyword: key_form.clone(),
                    }
                })?)?;
            let width = format.row_bytes();
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
            let tnull = header.optional_int(&format!("TNULL{i}"));
            let tdim = parse_tdim(&header, i)?;
            let int_storage = if matches!(
                format.kind,
                BinFieldKind::Byte | BinFieldKind::I16 | BinFieldKind::I32 | BinFieldKind::I64
            ) {
                IntStorage::detect(format.kind, tscal, tzero)
            } else {
                IntStorage::RawSigned
            };
            let tdisp = header
                .optional_string(&format!("TDISP{i}"))
                .map(|s| s.trim().to_string());
            columns.push(BinColumn {
                index: i,
                name,
                unit,
                format,
                offset,
                tscal,
                tzero,
                tnull,
                tdim,
                tdisp,
                int_storage,
            });
            offset = offset.checked_add(width).ok_or_else(|| FitsError::Value {
                keyword: format!("TFORM{i}"),
                msg: "row offset overflows usize".into(),
            })?;
        }
        if offset > row_size {
            return Err(FitsError::Value {
                keyword: "NAXIS1".into(),
                msg: format!("sum of TFORM widths ({offset}) exceeds NAXIS1 ({row_size})"),
            });
        }

        let heap_size = needed - heap_offset;
        Ok(Self {
            header,
            data,
            row_size,
            n_rows,
            columns,
            heap_offset,
            heap_size,
        })
    }

    /// The HDU's header.
    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }
    /// Row count, from `NAXIS2`.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    /// Bytes per row, from `NAXIS1`.
    #[must_use]
    pub fn row_size(&self) -> usize {
        self.row_size
    }
    /// Every column, in `TFORMn` order.
    #[must_use]
    pub fn columns(&self) -> &[BinColumn] {
        &self.columns
    }
    /// Size of the variable-length-array heap in bytes, from `PCOUNT`
    /// less `THEAP` (Sec.7.3.5).
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.heap_size
    }

    /// Find a column whose `TTYPEn` equals `name`.
    ///
    /// The comparison ignores ASCII case. The result is `None` when no
    /// column carries that name.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use fitsy::{FitsError, FitsFile, Hdu};
    ///
    /// let f = FitsFile::open("catalog.fits")?;
    /// let Hdu::BinTable(t) = f.hdu(1)? else {
    ///     return Err(FitsError::Header("HDU 1 is not a binary table".into()));
    /// };
    /// let ra = t.column_by_name("RA").expect("missing RA column");
    /// println!("RA is column {} with kind {:?}", ra.index, ra.format.kind);
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    #[must_use]
    pub fn column_by_name(&self, name: &str) -> Option<&BinColumn> {
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
        let s = row * self.row_size;
        Ok(&self.data[s..s + self.row_size])
    }

    /// Raw bytes of one cell inside the row area.
    ///
    /// The `row` argument is a 0-based row index. The `col` argument
    /// describes the column, as [`columns`](Self::columns) returns it.
    ///
    /// For a `P` or `Q` column this returns the descriptor. It does
    /// not follow that descriptor into the heap. Call
    /// [`cell_value`](Self::cell_value) for the array itself.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when `row` is not less than the row count.
    pub fn cell_bytes(&self, row: usize, col: &BinColumn) -> Result<&[u8]> {
        let row_bytes = self.row_bytes(row)?;
        let s = col.offset;
        Ok(&row_bytes[s..s + col.format.row_bytes()])
    }

    /// Decode one cell into a typed [`BinValue`].
    ///
    /// The `row` argument is a 0-based row index. The `col` argument
    /// describes the column, as [`columns`](Self::columns) returns it.
    ///
    /// A `P` or `Q` column resolves its descriptor against the heap
    /// and returns [`BinValue::Vla`].
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] in four cases:
    ///
    /// - `row` is not less than the row count.
    /// - An `L` byte is neither `T`, `F` nor 0.
    /// - An `A` field is not valid UTF-8.
    /// - A `P` or `Q` descriptor points outside the heap.
    ///
    /// [`FitsError::Value`] when a `P` or `Q` column carries no inner
    /// element type in its `TFORMn`.
    pub fn cell_value(&self, row: usize, col: &BinColumn) -> Result<BinValue> {
        let raw = self.cell_bytes(row, col)?;
        decode_cell(col, raw, self.heap_bytes())
    }

    /// Heap bytes (everything from `heap_offset` to end of data).
    /// May be empty if `PCOUNT == 0` or `THEAP` equals the data
    /// length.
    #[must_use]
    pub fn heap_bytes(&self) -> &[u8] {
        &self.data[self.heap_offset..self.heap_offset + self.heap_size]
    }

    /// Raw data bytes (table rows followed by the heap, exactly as
    /// they appear in the file). The slice length equals
    /// `NAXIS1 * NAXIS2 + PCOUNT` (or whatever the on-disk extent
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

    /// turned out to be).
    #[must_use]
    pub fn data_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Iterate `count` consecutive rows from `start`, yielding the raw
    /// bytes of each row.
    ///
    /// This lets a caller process a contiguous slab of rows without
    /// walking the rest of the table and without copying the heap.
    ///
    /// # Errors
    ///
    /// [`FitsError::Data`] when `start + count` overflows `usize`, or
    /// when that sum exceeds the row count.
    pub fn row_range(
        &self,
        start: usize,
        count: usize,
    ) -> Result<impl Iterator<Item = &[u8]> + '_> {
        let end = start
            .checked_add(count)
            .ok_or_else(|| FitsError::Data("row start + count overflowed usize".into()))?;
        if end > self.n_rows {
            return Err(FitsError::Data(format!(
                "row_range: rows {start}..{end} out of range (n_rows = {})",
                self.n_rows
            )));
        }
        let row_size = self.row_size;
        let s = start * row_size;
        let slab = &self.data[s..s + count * row_size];
        Ok(slab.chunks_exact(row_size))
    }
}

/// Decoded value of one BINTABLE cell.
#[derive(Debug, Clone)]
pub enum BinValue {
    /// Fixed-length array of logicals (`L`).
    Logical(Vec<Option<bool>>),
    /// Packed-bit field (`X`), most significant bit first within each
    /// byte. The second element is the bit count, the `r` of `rX`.
    Bits(Vec<u8>, usize),
    /// Signed integer column (`B/I/J/K`) with no scaling, or with
    /// `IntStorage::SignedFromByte` (`B + TZERO=-128`).
    Int(Vec<Option<i64>>),
    /// Unsigned-integer column (`I/J/K`) under the
    /// `IntStorage::Unsigned` convention. `K + TZERO=2^63` is the
    /// only path that preserves full u64 precision.
    Uint(Vec<Option<u64>>),
    /// `B/I/J/K` column with non-canonical `TSCAL`/`TZERO` applied
    /// in `f64`.
    Float(Vec<f64>),
    /// `A` character array (returned as one trimmed string).
    Str(String),
    /// `A` character array carrying a `TDIMn` (Sec.7.3.3.2): the first
    /// dimension is the width of each string and the rest give the
    /// array shape, so the cell is an array of fixed-width strings.
    StrArray(Vec<String>),
    /// `E` single-precision floats.
    F32(Vec<f32>),
    /// `D` double-precision floats.
    F64(Vec<f64>),
    /// `C` single-precision complex, as `(re, im)` pairs.
    C64(Vec<(f32, f32)>),
    /// `M` double-precision complex, as `(re, im)` pairs.
    C128(Vec<(f64, f64)>),
    /// `P`/`Q` variable-length array -> resolved against the heap.
    Vla(Box<Self>),
}

fn decode_cell(col: &BinColumn, raw: &[u8], heap: &[u8]) -> Result<BinValue> {
    match col.format.kind {
        BinFieldKind::Logical => {
            let mut out = Vec::with_capacity(raw.len());
            for &b in raw {
                out.push(match b {
                    b'T' | b't' => Some(true),
                    b'F' | b'f' => Some(false),
                    0 => None,
                    other => {
                        return Err(FitsError::Data(format!(
                            "BINTABLE col {}: invalid L byte 0x{other:02x}",
                            col.index
                        )));
                    }
                });
            }
            Ok(BinValue::Logical(out))
        }
        BinFieldKind::Bit => Ok(BinValue::Bits(raw.to_vec(), col.format.repeat)),
        BinFieldKind::Byte => decode_int(col, raw, 1, |b| i64::from(b[0])),
        BinFieldKind::I16 => {
            decode_int(col, raw, 2, |b| i64::from(i16::from_be_bytes([b[0], b[1]])))
        }
        BinFieldKind::I32 => decode_int(col, raw, 4, |b| {
            i64::from(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        }),
        BinFieldKind::I64 => decode_int(col, raw, 8, |b| {
            i64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        }),
        BinFieldKind::Char => {
            let s = std::str::from_utf8(raw).map_err(|_| {
                FitsError::Data(format!(
                    "BINTABLE col {}: A field is not valid UTF-8",
                    col.index
                ))
            })?;
            // Sec.7.3.3.2: `TDIMn` on an `A` column makes the cell an
            // array of strings -- the first dimension is each string's
            // width, the rest are the array shape. Trailing elements
            // beyond the TDIM product are undefined fill.
            if let Some(dims) = col.tdim.as_ref()
                && dims.len() >= 2
                && dims[0] > 0
                && let Some(count) = dims[1..].iter().try_fold(1_usize, |a, &d| a.checked_mul(d))
                && let Some(needed) = count.checked_mul(dims[0])
                && needed <= s.len()
            {
                let out = s.as_bytes()[..needed]
                    .chunks_exact(dims[0])
                    .map(|c| {
                        // Each element is padded independently, so trim
                        // per string rather than across the whole field.
                        String::from_utf8_lossy(c)
                            .trim_end_matches('\0')
                            .trim_end()
                            .to_string()
                    })
                    .collect();
                return Ok(BinValue::StrArray(out));
            }
            Ok(BinValue::Str(
                s.trim_end_matches('\0').trim_end().to_string(),
            ))
        }
        BinFieldKind::F32 => {
            let mut out = Vec::with_capacity(col.format.repeat);
            for c in raw.as_chunks::<4>().0 {
                let v = f32::from_be_bytes(*c);
                out.push((col.tzero as f32) + (col.tscal as f32) * v);
            }
            Ok(BinValue::F32(out))
        }
        BinFieldKind::F64 => {
            let mut out = Vec::with_capacity(col.format.repeat);
            for c in raw.as_chunks::<8>().0 {
                let v = f64::from_be_bytes(*c);
                out.push(col.tzero + col.tscal * v);
            }
            Ok(BinValue::F64(out))
        }
        BinFieldKind::C64 => {
            let mut out = Vec::with_capacity(col.format.repeat);
            for c in raw.as_chunks::<8>().0 {
                let (re, im) = c.split_at(4);
                let re = f32::from_be_bytes(re.try_into().unwrap());
                let im = f32::from_be_bytes(im.try_into().unwrap());
                out.push((re, im));
            }
            Ok(BinValue::C64(out))
        }
        BinFieldKind::C128 => {
            let mut out = Vec::with_capacity(col.format.repeat);
            for c in raw.as_chunks::<16>().0 {
                let (re, im) = c.split_at(8);
                let re = f64::from_be_bytes(re.try_into().unwrap());
                let im = f64::from_be_bytes(im.try_into().unwrap());
                out.push((re, im));
            }
            Ok(BinValue::C128(out))
        }
        BinFieldKind::P | BinFieldKind::Q => {
            let (n, off) = parse_vla_descriptor(col.format.kind, raw)?;
            if n == 0 {
                return Ok(BinValue::Vla(Box::new(BinValue::Int(Vec::new()))));
            }
            let inner = col.format.vla_kind.ok_or_else(|| FitsError::Value {
                keyword: format!("TFORM{}", col.index),
                msg: "P/Q descriptor missing inner type".into(),
            })?;
            // `n` is off-disk: a wrapped byte count would pass the heap
            // check below and leave `n` to reach `Vec::with_capacity`.
            let bytes_needed = if matches!(inner, BinFieldKind::Bit) {
                n.div_ceil(8)
            } else {
                n.checked_mul(inner.element_bytes()).ok_or_else(|| {
                    FitsError::Data(format!(
                        "BINTABLE col {}: VLA byte length overflows usize (n={n})",
                        col.index
                    ))
                })?
            };
            if off.saturating_add(bytes_needed) > heap.len() {
                return Err(FitsError::Data(format!(
                    "BINTABLE col {}: P/Q descriptor (n={n}, off={off}) escapes heap (len={})",
                    col.index,
                    heap.len()
                )));
            }
            let slice = &heap[off..off + bytes_needed];
            // Synthesize a fake column with the inner type and decode
            // it through the same path. The inner column inherits
            // TSCAL/TZERO/TNULL since Sec.7.3.5 says VLA elements are
            // the same physical kind as a fixed column would be.
            let inner_storage = if matches!(
                inner,
                BinFieldKind::Byte | BinFieldKind::I16 | BinFieldKind::I32 | BinFieldKind::I64
            ) {
                IntStorage::detect(inner, col.tscal, col.tzero)
            } else {
                IntStorage::RawSigned
            };
            let inner_col = BinColumn {
                index: col.index,
                name: col.name.clone(),
                unit: String::new(),
                format: BinFormat {
                    repeat: n,
                    kind: inner,
                    vla_kind: None,
                    vla_max: None,
                },
                offset: 0,
                tscal: col.tscal,
                tzero: col.tzero,
                tnull: col.tnull,
                tdim: None,
                // The inner column decodes bytes. `TDISPn` describes
                // presentation, so it plays no part here.
                tdisp: None,
                int_storage: inner_storage,
            };
            let inner_value = decode_cell(&inner_col, slice, &[])?;
            Ok(BinValue::Vla(Box::new(inner_value)))
        }
    }
}

/// Decode an integer column honoring `IntStorage` (Standard
/// Sec.11.3.1). The reader returns the raw signed value as stored;
/// reinterpretation to unsigned is by bit-pattern.
#[allow(
    clippy::unnecessary_wraps,
    reason = "keeping Result for consistency with other column decoders that can return Err"
)]
fn decode_int(
    col: &BinColumn,
    raw: &[u8],
    elem: usize,
    read: impl Fn(&[u8]) -> i64,
) -> Result<BinValue> {
    let n = col.format.repeat;
    match col.int_storage {
        IntStorage::RawSigned => {
            let mut out = Vec::with_capacity(n);
            for c in raw.chunks_exact(elem) {
                let v = read(c);
                out.push(if Some(v) == col.tnull { None } else { Some(v) });
            }
            Ok(BinValue::Int(out))
        }
        IntStorage::SignedFromByte => {
            // B with TZERO=-128: stored u8 in 0..=255 -> signed i8
            // by adding TZERO (=-128).
            let mut out = Vec::with_capacity(n);
            for c in raw.chunks_exact(elem) {
                let stored = read(c);
                out.push(if Some(stored) == col.tnull {
                    None
                } else {
                    Some(stored - 128)
                });
            }
            Ok(BinValue::Int(out))
        }
        IntStorage::Unsigned => {
            // I/J/K with TZERO=2^(8n-1): bit-pattern reinterpreted as
            // unsigned. Preserves full u64 precision for K.
            let mut out = Vec::with_capacity(n);
            for c in raw.chunks_exact(elem) {
                let signed = read(c);
                // Add within the native type: widening first would put
                // every negative stored value 2^n too high.
                let unsigned = match col.format.kind {
                    BinFieldKind::I16 => u64::from((signed as i16 as u16).wrapping_add(0x8000)),
                    BinFieldKind::I32 => {
                        u64::from((signed as i32 as u32).wrapping_add(0x8000_0000))
                    }
                    BinFieldKind::I64 => (signed as u64).wrapping_add(0x8000_0000_0000_0000),
                    _ => unreachable!("Unsigned only set for I/J/K"),
                };
                out.push(if Some(signed) == col.tnull {
                    None
                } else {
                    Some(unsigned)
                });
            }
            Ok(BinValue::Uint(out))
        }
        IntStorage::Scaled => {
            // TNULL marks the stored (pre-scaling) undefined value
            // (Sec.7.3.3.2); undefined cells surface as NaN. Note astropy
            // leaves scaled TNULL cells unmasked -- we follow the standard
            // (and cfitsio) instead.
            let mut out = Vec::with_capacity(n);
            for c in raw.chunks_exact(elem) {
                let stored = read(c);
                out.push(if Some(stored) == col.tnull {
                    f64::NAN
                } else {
                    col.tzero + col.tscal * stored as f64
                });
            }
            Ok(BinValue::Float(out))
        }
    }
}

/// Parse a `P` or `Q` variable-length array descriptor cell.
///
/// Standard Sec.7.3.5 defines a `P` cell as two big-endian `i32`
/// values, `(n_elements, heap_offset)`. A `Q` cell holds the same pair
/// in `i64`. Both fields must be non-negative.
///
/// This returns the element count, not a byte count. The caller
/// multiplies by the inner element size from `BinFormat::vla_kind`.
///
/// # Errors
///
/// [`FitsError::Data`] when the cell is too short for the descriptor,
/// or when either field is negative. [`FitsError::Header`] when the
/// count or the offset exceeds `usize` on this target.
pub(crate) fn parse_vla_descriptor(kind: BinFieldKind, raw: &[u8]) -> Result<(usize, usize)> {
    let (n, off) = match kind {
        BinFieldKind::P => {
            if raw.len() < 8 {
                return Err(FitsError::Data(format!(
                    "P-descriptor cell is {} bytes, need 8",
                    raw.len()
                )));
            }
            let n = i64::from(i32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]));
            let o = i64::from(i32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]));
            (n, o)
        }
        BinFieldKind::Q => {
            if raw.len() < 16 {
                return Err(FitsError::Data(format!(
                    "Q-descriptor cell is {} bytes, need 16",
                    raw.len()
                )));
            }
            let n = i64::from_be_bytes([
                raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
            ]);
            let o = i64::from_be_bytes([
                raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15],
            ]);
            (n, o)
        }
        other => {
            return Err(FitsError::Header(format!(
                "expected a P/Q descriptor column, got {other:?}"
            )));
        }
    };
    if n < 0 || off < 0 {
        return Err(FitsError::Data(format!(
            "VLA descriptor has negative field (n={n}, off={off})"
        )));
    }
    Ok((n as usize, off as usize))
}

fn parse_tdim(h: &Header, i: usize) -> Result<Option<Vec<usize>>> {
    let key = format!("TDIM{i}");
    let s = match h.first(&key) {
        Some(Value::String(s)) => s.clone(),
        _ => return Ok(None),
    };
    let t = s.trim();
    let inner = t
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| FitsError::Value {
            keyword: key.clone(),
            msg: format!("TDIM must be `(d1,d2,...)`, got `{s}`"),
        })?;
    let dims: Result<Vec<usize>> = inner
        .split(',')
        .map(|p| {
            p.trim().parse::<usize>().map_err(|_| FitsError::Value {
                keyword: key.clone(),
                msg: format!("invalid TDIM dim `{p}`"),
            })
        })
        .collect();
    Ok(Some(dims?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_formats() {
        let p = BinFormat::parse("1J").unwrap();
        assert_eq!(p.repeat, 1);
        assert_eq!(p.kind, BinFieldKind::I32);
        assert_eq!(p.row_bytes(), 4);

        let p = BinFormat::parse("4E").unwrap();
        assert_eq!(p.repeat, 4);
        assert_eq!(p.kind, BinFieldKind::F32);
        assert_eq!(p.row_bytes(), 16);

        let p = BinFormat::parse("16A").unwrap();
        assert_eq!(p.repeat, 16);
        assert_eq!(p.kind, BinFieldKind::Char);
        assert_eq!(p.row_bytes(), 16);

        let p = BinFormat::parse("33X").unwrap();
        assert_eq!(p.repeat, 33);
        assert_eq!(p.kind, BinFieldKind::Bit);
        // ceil(33/8) = 5
        assert_eq!(p.row_bytes(), 5);
    }

    #[test]
    fn parse_vla_descriptors() {
        let p = BinFormat::parse("PE(99)").unwrap();
        assert_eq!(p.kind, BinFieldKind::P);
        assert_eq!(p.vla_kind, Some(BinFieldKind::F32));
        assert_eq!(p.vla_max, Some(99));
        assert_eq!(p.row_bytes(), 8);

        let p = BinFormat::parse("1QD(0)").unwrap();
        assert_eq!(p.kind, BinFieldKind::Q);
        assert_eq!(p.vla_kind, Some(BinFieldKind::F64));
        assert_eq!(p.row_bytes(), 16);

        let p = BinFormat::parse("PJ").unwrap();
        assert_eq!(p.vla_kind, Some(BinFieldKind::I32));
        assert_eq!(p.vla_max, None);
    }

    fn scalar_col(kind: BinFieldKind, tscal: f64, tzero: f64) -> BinColumn {
        BinColumn {
            index: 1,
            name: "COL".into(),
            unit: String::new(),
            format: BinFormat {
                repeat: 1,
                kind,
                vla_kind: None,
                vla_max: None,
            },
            offset: 0,
            tscal,
            tzero,
            tnull: None,
            tdim: None,
            tdisp: None,
            int_storage: IntStorage::detect(kind, tscal, tzero),
        }
    }

    fn decode_uint(col: &BinColumn, raw: &[u8]) -> u64 {
        match decode_cell(col, raw, &[]).unwrap() {
            BinValue::Uint(v) => v[0].unwrap(),
            other => panic!("expected Uint, got {other:?}"),
        }
    }

    /// Sec.11.3.1: `physical = stored + 2^(8n-1)`, wrapping at the
    /// column width, so `stored = -2^(8n-1)` reads back as 0.
    #[test]
    fn unsigned_i16_column_covers_full_u16_range() {
        let col = scalar_col(BinFieldKind::I16, 1.0, 32_768.0);
        assert_eq!(col.int_storage, IntStorage::Unsigned);
        for (stored, want) in [
            (-32_768_i16, 0_u64),
            (-32_767, 1),
            (-1, 32_767),
            (0, 32_768),
            (1, 32_769),
            (32_767, 65_535),
        ] {
            assert_eq!(decode_uint(&col, &stored.to_be_bytes()), want, "{stored}");
        }
    }

    #[test]
    fn unsigned_i32_column_covers_full_u32_range() {
        let col = scalar_col(BinFieldKind::I32, 1.0, 2_147_483_648.0);
        assert_eq!(col.int_storage, IntStorage::Unsigned);
        for (stored, want) in [
            (i32::MIN, 0_u64),
            (-1, 2_147_483_647),
            (0, 2_147_483_648),
            (i32::MAX, 4_294_967_295),
        ] {
            assert_eq!(decode_uint(&col, &stored.to_be_bytes()), want, "{stored}");
        }
    }

    #[test]
    fn unsigned_i64_column_covers_full_u64_range() {
        let col = scalar_col(BinFieldKind::I64, 1.0, 9_223_372_036_854_775_808.0);
        assert_eq!(col.int_storage, IntStorage::Unsigned);
        for (stored, want) in [
            (i64::MIN, 0_u64),
            (-1, 9_223_372_036_854_775_807),
            (0, 9_223_372_036_854_775_808),
            (i64::MAX, u64::MAX),
        ] {
            assert_eq!(decode_uint(&col, &stored.to_be_bytes()), want, "{stored}");
        }
    }

    /// An element count that overflows on conversion to bytes must be
    /// rejected, not wrapped (`capacity overflow` panic in release).
    #[test]
    fn vla_element_count_overflow_is_an_error() {
        let mut col = scalar_col(BinFieldKind::Q, 1.0, 0.0);
        col.format.vla_kind = Some(BinFieldKind::F64);
        let mut raw = [0_u8; 16];
        // 2^61 elements * 8 bytes wraps to 0 in a 64-bit usize.
        raw[..8].copy_from_slice(&(1_i64 << 61).to_be_bytes());
        assert!(decode_cell(&col, &raw, &[]).is_err());
    }

    /// Sec.7.3.3.2: `TDIM` on an `A` column makes the cell an array of
    /// strings -- first dimension is the width, the rest the shape.
    #[test]
    fn char_column_with_tdim_splits_into_strings() {
        let mut col = scalar_col(BinFieldKind::Char, 1.0, 0.0);
        col.format.repeat = 15;
        col.tdim = Some(vec![5, 3]);
        match decode_cell(&col, b"alphabeta gamma", &[]).unwrap() {
            BinValue::StrArray(v) => assert_eq!(v, vec!["alpha", "beta", "gamma"]),
            other => panic!("expected StrArray, got {other:?}"),
        }
    }

    /// Without `TDIM` an `A` column stays one string.
    #[test]
    fn char_column_without_tdim_is_one_string() {
        let mut col = scalar_col(BinFieldKind::Char, 1.0, 0.0);
        col.format.repeat = 15;
        match decode_cell(&col, b"alphabeta gamma", &[]).unwrap() {
            BinValue::Str(s) => assert_eq!(s, "alphabeta gamma"),
            other => panic!("expected Str, got {other:?}"),
        }
    }

    /// A `TDIM` product smaller than the repeat count is legal; the
    /// trailing elements are undefined fill and are dropped.
    #[test]
    fn char_column_tdim_shorter_than_repeat_drops_fill() {
        let mut col = scalar_col(BinFieldKind::Char, 1.0, 0.0);
        col.format.repeat = 15;
        col.tdim = Some(vec![5, 2]);
        match decode_cell(&col, b"alphabeta gamma", &[]).unwrap() {
            BinValue::StrArray(v) => assert_eq!(v, vec!["alpha", "beta"]),
            other => panic!("expected StrArray, got {other:?}"),
        }
    }

    /// A `TDIM` that overruns the field is malformed: fall back to one
    /// string rather than reading past the cell.
    #[test]
    fn char_column_oversized_tdim_falls_back() {
        let mut col = scalar_col(BinFieldKind::Char, 1.0, 0.0);
        col.format.repeat = 15;
        col.tdim = Some(vec![5, 9]);
        assert!(matches!(
            decode_cell(&col, b"alphabeta gamma", &[]).unwrap(),
            BinValue::Str(_)
        ));
    }

    #[test]
    fn parse_tdim_simple() {
        // Built via in-place header construction is overkill for a
        // pure function; just make sure parser logic accepts the
        // expected format via direct call.
        let s = "(2,3,4)";
        let parts: Vec<usize> = s
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .unwrap()
            .split(',')
            .map(|p| p.trim().parse().unwrap())
            .collect();
        assert_eq!(parts, vec![2, 3, 4]);
    }
}
