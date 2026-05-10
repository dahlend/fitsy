//! Typed HDU builders.
//!
//! These builders compose the mandatory keywords for the supported
//! HDU kinds so callers don't have to remember the exact spellings
//! of `BITPIX`/`NAXIS`/`PCOUNT`/`GCOUNT`/`TFORMn`/etc. They emit
//! `(Header, Vec<u8>)` pairs that can be fed directly to
//! [`FitsWriter::write_hdu`](crate::FitsWriter::write_hdu).

use crate::data::encoding::Pixel;
use crate::data::unsigned::{BZERO_U16, BZERO_U32, BZERO_U64_F64};
use crate::error::{FitsError, Result};
use crate::hdu::bintable::BinFieldKind;
use crate::header::{CommentaryKind, Header, Value};

/// Builder for an `IMAGE` HDU (primary or extension).
///
/// Pixel data is supplied as a typed slice; the builder serializes it
/// big-endian, sets `BITPIX`/`NAXIS`/`NAXISn` consistently, and
/// optionally adds `BSCALE`/`BZERO`/`BUNIT`/`BLANK` cards.
///
/// # Unsigned-integer images
///
/// FITS `BITPIX` values are signed (16/32/64); unsigned data is stored
/// per Standard Sec.4.4.2.5 as the offset-encoded signed value plus a
/// `BZERO` card (with `BSCALE = 1`). Use the dtype-specific
/// constructors [`from_u16`](Self::from_u16), [`from_u32`](Self::from_u32),
/// and [`from_u64`](Self::from_u64) -- they encode the offset and
/// emit the required `BSCALE`/`BZERO` cards automatically.
#[derive(Debug)]
pub struct ImageBuilder<T: Pixel> {
    axes: Vec<u64>,
    pixels: Vec<T>,
    is_primary: bool,
    extras: Vec<(String, Value, Option<String>)>,
    history: Vec<String>,
    /// When `Some`, the builder emits `BSCALE = 1` and `BZERO = ...`
    /// cards for unsigned-integer images that have already been
    /// re-encoded into the signed `T` storage.
    unsigned_bzero: Option<Value>,
}

impl<T: Pixel> ImageBuilder<T> {
    /// Construct a new IMAGE-shaped HDU. `axes` lists axis lengths in
    /// FITS order -- `axes[0]` is `NAXIS1`, the fastest-varying axis.
    /// `pixels.len()` must equal the product of `axes`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fitsy::ImageBuilder;
    ///
    /// let pixels: Vec<f32> = (0..(64 * 48)).map(|i| i as f32).collect();
    /// let (header, data) = ImageBuilder::new(vec![64_u64, 48], pixels)?
    ///     .primary(true)
    ///     .build()?;
    /// assert_eq!(data.len(), 64 * 48 * 4);
    /// assert_eq!(header.naxis()?, 2);
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    pub fn new(axes: impl Into<Vec<u64>>, pixels: impl Into<Vec<T>>) -> Result<Self> {
        let axes: Vec<u64> = axes.into();
        let pixels: Vec<T> = pixels.into();
        let expected: u64 = axes.iter().copied().product::<u64>();
        let got = pixels.len() as u64;
        if !axes.is_empty() && expected != got {
            return Err(FitsError::Data(format!(
                "ImageBuilder: pixel count {got} does not match axes product {expected}"
            )));
        }
        Ok(Self {
            axes,
            pixels,
            is_primary: false,
            extras: Vec::new(),
            history: Vec::new(),
            unsigned_bzero: None,
        })
    }

    /// Mark this image as the primary HDU (`SIMPLE = T` rather than
    /// `XTENSION = 'IMAGE   '`). At most one primary per file.
    #[must_use]
    pub fn primary(mut self, yes: bool) -> Self {
        self.is_primary = yes;
        self
    }

    /// Append an arbitrary value card (e.g. `BUNIT`, `OBJECT`).
    /// Validated on [`build`](Self::build).
    #[must_use]
    pub fn card(
        mut self,
        keyword: impl Into<String>,
        value: impl Into<Value>,
        comment: Option<&str>,
    ) -> Self {
        self.extras.push((
            keyword.into(),
            value.into(),
            comment.map(ToString::to_string),
        ));
        self
    }

    /// Append a `HISTORY` card.
    #[must_use]
    pub fn history(mut self, text: impl Into<String>) -> Self {
        self.history.push(text.into());
        self
    }

    /// Render this builder to a `(Header, data)` pair ready for
    /// [`FitsWriter::write_hdu`](crate::FitsWriter::write_hdu). The
    /// returned data is big-endian raw bytes.
    pub fn build(self) -> Result<(Header, Vec<u8>)> {
        let bitpix = T::BITPIX.as_i64();
        let mut h = Header::empty();
        if self.is_primary {
            h.push("SIMPLE", Value::Logical(true), Some("conforming FITS file"))?;
        } else {
            h.push("XTENSION", Value::String("IMAGE".into()), None)?;
        }
        h.push("BITPIX", Value::Integer(bitpix), None)?;
        h.push("NAXIS", Value::Integer(self.axes.len() as i64), None)?;
        for (i, n) in self.axes.iter().enumerate() {
            h.push(format!("NAXIS{}", i + 1), Value::Integer(*n as i64), None)?;
        }
        // Standard Sec.4.4.1.1: EXTEND must follow the last NAXISn card.
        // Emit only for the primary HDU, and only when the user has not
        // already supplied it via .card().
        if self.is_primary && !self.extras.iter().any(|(k, _, _)| k == "EXTEND") {
            h.push(
                "EXTEND",
                Value::Logical(true),
                Some("FITS dataset may contain extensions"),
            )?;
        }
        if !self.is_primary {
            h.push("PCOUNT", Value::Integer(0), None)?;
            h.push("GCOUNT", Value::Integer(1), None)?;
        }
        if let Some(bzero) = self.unsigned_bzero.clone() {
            h.push(
                "BSCALE",
                Value::Integer(1),
                Some("unsigned integer encoding"),
            )?;
            h.push("BZERO", bzero, Some("unsigned integer offset"))?;
        }
        for (k, v, c) in self.extras {
            if self.unsigned_bzero.is_some() && matches!(k.as_str(), "BSCALE" | "BZERO") {
                return Err(FitsError::Header(format!(
                    "ImageBuilder::from_u* manages {k} automatically; do not add it via card()"
                )));
            }
            h.push(k, v, c.as_deref())?;
        }
        for line in self.history {
            h.push_commentary(CommentaryKind::History, &line);
        }

        // Serialize pixels big-endian.
        let bpp = T::BITPIX.byte_size();
        let mut data = Vec::with_capacity(self.pixels.len() * bpp);
        for p in &self.pixels {
            p.write_be(&mut data);
        }
        Ok((h, data))
    }
}

impl ImageBuilder<i16> {
    /// Build a `u16` image (`BITPIX = 16`, `BZERO = 32768`,
    /// `BSCALE = 1`). Pixels are offset-encoded into `i16` and the
    /// matching `BSCALE`/`BZERO` cards are emitted automatically.
    pub fn from_u16(axes: impl Into<Vec<u64>>, pixels: &[u16]) -> Result<Self> {
        let signed: Vec<i16> = pixels
            .iter()
            .map(|&p| (i32::from(p) - BZERO_U16 as i32) as i16)
            .collect();
        let mut b = Self::new(axes, signed)?;
        b.unsigned_bzero = Some(Value::Integer(BZERO_U16));
        Ok(b)
    }
}

impl ImageBuilder<i32> {
    /// Build a `u32` image (`BITPIX = 32`, `BZERO = 2147483648`,
    /// `BSCALE = 1`).
    pub fn from_u32(axes: impl Into<Vec<u64>>, pixels: &[u32]) -> Result<Self> {
        let signed: Vec<i32> = pixels
            .iter()
            .map(|&p| (i64::from(p) - BZERO_U32) as i32)
            .collect();
        let mut b = Self::new(axes, signed)?;
        b.unsigned_bzero = Some(Value::Integer(BZERO_U32));
        Ok(b)
    }
}

impl ImageBuilder<i64> {
    /// Build a `u64` image (`BITPIX = 64`, `BZERO = 9.22e18` real-valued
    /// because the magnitude `2^63` does not fit in `i64`,
    /// `BSCALE = 1`).
    pub fn from_u64(axes: impl Into<Vec<u64>>, pixels: &[u64]) -> Result<Self> {
        let signed: Vec<i64> = pixels
            .iter()
            .map(|&p| p.wrapping_sub(1_u64 << 63) as i64)
            .collect();
        let mut b = Self::new(axes, signed)?;
        b.unsigned_bzero = Some(Value::Real(BZERO_U64_F64));
        Ok(b)
    }
}

/// Builder for a `BINTABLE` HDU (Standard Sec.7.3).
///
/// Columns are described once via [`add_column`](Self::add_column);
/// rows are then handed in as a flat byte buffer in the same order
/// the columns were declared. The builder generates `XTENSION`,
/// `BITPIX`, `NAXIS{,1,2}`, `PCOUNT`, `GCOUNT`, `TFIELDS`, plus
/// `TTYPEn`, `TFORMn`, and (optionally) `TUNITn`/`TDIMn`.
#[derive(Debug, Default)]
pub struct BinTableBuilder {
    columns: Vec<ColSpec>,
    extras: Vec<(String, Value, Option<String>)>,
    history: Vec<String>,
    extname: Option<String>,
}

#[derive(Debug)]
struct ColSpec {
    name: String,
    kind: BinFieldKind,
    repeat: usize,
    unit: Option<String>,
    tdim: Option<String>,
    /// For VLA columns (`kind == P` or `kind == Q`), the element type
    /// stored in the heap (e.g. `I32` for `1PJ`). `None` for fixed
    /// columns.
    vla_element: Option<BinFieldKind>,
    /// For VLA columns, the optional bound `r` in `1PT(r)` written
    /// into `TFORMn`. Informational only -- the builder does not
    /// enforce per-row VLA lengths.
    vla_max: Option<usize>,
}

impl ColSpec {
    fn tform(&self) -> String {
        let code = self.kind.tform_char();
        if let Some(elt) = self.vla_element {
            debug_assert!(
                !matches!(elt, BinFieldKind::P | BinFieldKind::Q),
                "VLA of VLA disallowed",
            );
            let elt_code = elt.tform_char();
            // Standard Sec.7.3.5: "1PT(rmax)" or "1QT(rmax)".
            match self.vla_max {
                Some(m) => format!("{}{}{}({})", self.repeat, code, elt_code, m),
                None => format!("{}{}{}", self.repeat, code, elt_code),
            }
        } else {
            format!("{}{}", self.repeat, code)
        }
    }

    fn row_bytes(&self) -> usize {
        match self.kind {
            BinFieldKind::Bit => self.repeat.div_ceil(8),
            BinFieldKind::P => 8 * self.repeat.max(1),
            BinFieldKind::Q => 16 * self.repeat.max(1),
            other => self.repeat * other.element_bytes(),
        }
    }
}

impl BinTableBuilder {
    /// Empty builder; add columns via [`add_column`](Self::add_column).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a fixed-shape column. `repeat` is the `r` in `rT` (1 for
    /// scalars, `n` for fixed `n`-character strings of kind
    /// [`BinFieldKind::Char`], or whatever array length applies). For
    /// variable-length (`P`/`Q`) columns use
    /// [`add_vla_column`](Self::add_vla_column).
    pub fn add_column(
        &mut self,
        name: impl Into<String>,
        kind: BinFieldKind,
        repeat: usize,
        unit: Option<&str>,
        tdim: Option<&str>,
    ) -> Result<&mut Self> {
        if matches!(kind, BinFieldKind::P | BinFieldKind::Q) {
            return Err(FitsError::Data(
                "BinTableBuilder: use add_vla_column for P/Q descriptors".into(),
            ));
        }
        if repeat == 0 {
            return Err(FitsError::Data(
                "BinTableBuilder: column repeat must be >= 1".into(),
            ));
        }
        self.columns.push(ColSpec {
            name: name.into(),
            kind,
            repeat,
            unit: unit.map(ToString::to_string),
            tdim: tdim.map(ToString::to_string),
            vla_element: None,
            vla_max: None,
        });
        Ok(self)
    }

    /// Append a variable-length (heap) column described by a `P` or
    /// `Q` descriptor (Standard Sec.7.3.5).
    ///
    /// * `descriptor` must be [`BinFieldKind::P`] (32-bit count/offset
    ///   pairs) or [`BinFieldKind::Q`] (64-bit pairs).
    /// * `element` is the type of the values stored in the heap (any
    ///   non-`P`/`Q` kind).
    /// * `max` is the optional `rmax` advertised in `TFORMn`.
    ///
    /// The row area always carries one descriptor per row; the heap
    /// itself is supplied to [`build_with_heap`](Self::build_with_heap).
    pub fn add_vla_column(
        &mut self,
        name: impl Into<String>,
        descriptor: BinFieldKind,
        element: BinFieldKind,
        unit: Option<&str>,
        max: Option<usize>,
    ) -> Result<&mut Self> {
        if !matches!(descriptor, BinFieldKind::P | BinFieldKind::Q) {
            return Err(FitsError::Data(
                "BinTableBuilder::add_vla_column: descriptor must be P or Q".into(),
            ));
        }
        if matches!(element, BinFieldKind::P | BinFieldKind::Q) {
            return Err(FitsError::Data(
                "BinTableBuilder::add_vla_column: VLA element type cannot itself be P/Q".into(),
            ));
        }
        self.columns.push(ColSpec {
            name: name.into(),
            kind: descriptor,
            repeat: 1,
            unit: unit.map(ToString::to_string),
            tdim: None,
            vla_element: Some(element),
            vla_max: max,
        });
        Ok(self)
    }

    /// Encode a `P`-descriptor (`[count: i32, offset: i32]`, big-endian).
    #[must_use]
    pub fn p_descriptor(count: u32, offset: u32) -> [u8; 8] {
        let mut out = [0_u8; 8];
        out[..4].copy_from_slice(&(count as i32).to_be_bytes());
        out[4..].copy_from_slice(&(offset as i32).to_be_bytes());
        out
    }

    /// Encode a `Q`-descriptor (`[count: i64, offset: i64]`, big-endian).
    #[must_use]
    pub fn q_descriptor(count: u64, offset: u64) -> [u8; 16] {
        let mut out = [0_u8; 16];
        out[..8].copy_from_slice(&(count as i64).to_be_bytes());
        out[8..].copy_from_slice(&(offset as i64).to_be_bytes());
        out
    }

    /// Append an arbitrary value card (e.g. `OBJECT`).
    pub fn card(
        &mut self,
        keyword: impl Into<String>,
        value: impl Into<Value>,
        comment: Option<&str>,
    ) -> &mut Self {
        self.extras.push((
            keyword.into(),
            value.into(),
            comment.map(ToString::to_string),
        ));
        self
    }

    /// Append a `HISTORY` card.
    pub fn history(&mut self, text: impl Into<String>) -> &mut Self {
        self.history.push(text.into());
        self
    }

    /// Set `EXTNAME`.
    pub fn extname(&mut self, name: impl Into<String>) -> &mut Self {
        self.extname = Some(name.into());
        self
    }

    /// Set (or replace) the `TUNITn` of the most recently added column.
    /// Errors if no column has been added yet.
    ///
    /// This is sugar so callers can chain
    /// `.add_column(name, kind, repeat, None, None)?.unit("m")?` instead of
    /// repeating the column name in `extras`.
    pub fn unit(&mut self, unit: impl Into<String>) -> Result<&mut Self> {
        let last = self
            .columns
            .last_mut()
            .ok_or_else(|| FitsError::Data("no column added yet".into()))?;
        last.unit = Some(unit.into());
        Ok(self)
    }

    /// Set (or replace) the `TDIMn` of the most recently added column.
    /// Errors if no column has been added yet.
    pub fn tdim(&mut self, tdim: impl Into<String>) -> Result<&mut Self> {
        let last = self
            .columns
            .last_mut()
            .ok_or_else(|| FitsError::Data("no column added yet".into()))?;
        last.tdim = Some(tdim.into());
        Ok(self)
    }

    /// Width of one row in bytes (matches `NAXIS1`).
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.columns.iter().map(ColSpec::row_bytes).sum()
    }

    /// Render to `(Header, data)`. `data` must be exactly
    /// `n_rows * row_bytes()` long; rows are packed column-by-column
    /// in the order declared.
    ///
    /// For tables containing variable-length columns use
    /// [`build_with_heap`](Self::build_with_heap) instead -- this
    /// method emits `PCOUNT = 0`.
    pub fn build(self, n_rows: usize, data: Vec<u8>) -> Result<(Header, Vec<u8>)> {
        self.build_with_heap(n_rows, data, &[])
    }

    /// Render to `(Header, data + heap)` for tables that include `P`
    /// or `Q` (variable-length) columns.
    ///
    /// * `row_bytes` must be exactly `n_rows * self.row_bytes()` long.
    /// * `heap_bytes` is appended verbatim after the row area; the
    ///   builder sets `PCOUNT = heap_bytes.len()` and `THEAP =
    ///   row_area_size`.
    /// * Per-row `P`/`Q` descriptor cells must already encode their
    ///   `(count, offset)` pairs (use [`Self::p_descriptor`] /
    ///   [`Self::q_descriptor`] to build them).
    pub fn build_with_heap(
        self,
        n_rows: usize,
        row_bytes: Vec<u8>,
        heap_bytes: &[u8],
    ) -> Result<(Header, Vec<u8>)> {
        let row = self.row_bytes();
        let expected = row
            .checked_mul(n_rows)
            .ok_or_else(|| FitsError::Data("row x column count overflowed usize".into()))?;
        if row_bytes.len() != expected {
            return Err(FitsError::Data(format!(
                "BinTableBuilder: row data is {} bytes; expected {n_rows} rows x {row} bytes = {expected}",
                row_bytes.len()
            )));
        }
        let pcount = heap_bytes.len();
        let mut h = Header::empty();
        h.push("XTENSION", Value::String("BINTABLE".into()), None)?;
        h.push("BITPIX", Value::Integer(8), None)?;
        h.push("NAXIS", Value::Integer(2), None)?;
        h.push("NAXIS1", Value::Integer(row as i64), None)?;
        h.push("NAXIS2", Value::Integer(n_rows as i64), None)?;
        h.push("PCOUNT", Value::Integer(pcount as i64), None)?;
        h.push("GCOUNT", Value::Integer(1), None)?;
        h.push("TFIELDS", Value::Integer(self.columns.len() as i64), None)?;
        for (i, c) in self.columns.iter().enumerate() {
            let n = i + 1;
            h.push(format!("TTYPE{n}"), Value::String(c.name.clone()), None)?;
            h.push(format!("TFORM{n}"), Value::String(c.tform()), None)?;
            if let Some(u) = &c.unit {
                h.push(format!("TUNIT{n}"), Value::String(u.clone()), None)?;
            }
            if let Some(d) = &c.tdim {
                h.push(format!("TDIM{n}"), Value::String(d.clone()), None)?;
            }
        }
        if pcount > 0 {
            // Sec.7.3.5: THEAP defaults to the row-area size; emit it
            // explicitly so readers don't have to infer it.
            h.push("THEAP", Value::Integer(expected as i64), None)?;
        }
        if let Some(name) = &self.extname {
            h.push("EXTNAME", Value::String(name.clone()), None)?;
        }
        for (k, v, c) in self.extras {
            h.push(k, v, c.as_deref())?;
        }
        for line in self.history {
            h.push_commentary(CommentaryKind::History, &line);
        }
        let mut out = row_bytes;
        out.extend_from_slice(heap_bytes);
        Ok((h, out))
    }
}

// ---------------------------------------------------------------------
// AsciiTableBuilder
// ---------------------------------------------------------------------

/// Per-column data for an [`AsciiTableBuilder`]. Variants must match
/// the column's [`crate::AsciiFormat`] kind: `Int` <-> `I`, `Float` <-> `F`/`E`/`D`,
/// `Str` <-> `A`. All columns in a builder must carry the same number
/// of rows.
#[derive(Debug, Clone)]
pub enum AsciiColumnData {
    /// Integer column (`I` format). `None` cells render as `TNULLn`.
    Int(Vec<Option<i64>>),
    /// Floating-point column (`F`/`E`/`D` format). `NaN` cells render
    /// as a blank field.
    Float(Vec<f64>),
    /// String column (`A` format).
    Str(Vec<String>),
}

impl AsciiColumnData {
    fn len(&self) -> usize {
        match self {
            Self::Int(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::Str(v) => v.len(),
        }
    }
}

#[derive(Debug)]
struct AsciiColSpec {
    name: String,
    format: crate::hdu::ascii_table::AsciiFormat,
    data: AsciiColumnData,
    unit: Option<String>,
    /// Required for `I` columns that contain `None` cells; rendered
    /// into the row whenever the value is missing. The string is
    /// padded/truncated to the field width.
    tnull: Option<String>,
}

/// Builder for an `ASCII TABLE` HDU (Standard Sec.7.2).
///
/// Columns are described once and packed contiguously (no inter-field
/// padding). The builder generates `XTENSION = 'TABLE   '`,
/// `BITPIX = 8`, `NAXIS{,1,2}`, `PCOUNT = 0`, `GCOUNT = 1`, `TFIELDS`,
/// plus per-column `TFORMn`, `TBCOLn`, `TTYPEn`, optional `TUNITn`,
/// and `TNULLn` (for `I` columns with a configured null sentinel).
///
/// All columns must report the same row count. Use
/// [`AsciiFormat`](crate::AsciiFormat) variants matching the data
/// kind: `I` for integers, `F`/`E`/`D` for floats, `A` for strings.
#[derive(Debug, Default)]
pub struct AsciiTableBuilder {
    columns: Vec<AsciiColSpec>,
    extras: Vec<(String, Value, Option<String>)>,
    history: Vec<String>,
    extname: Option<String>,
}

impl AsciiTableBuilder {
    /// Empty builder; add columns via [`add_column`](Self::add_column).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an `A`/`I`/`F`/`E`/`D` column. The variant of `data`
    /// must match the kind of `format`.
    pub fn add_column(
        &mut self,
        name: impl Into<String>,
        format: crate::hdu::ascii_table::AsciiFormat,
        data: AsciiColumnData,
    ) -> Result<&mut Self> {
        use crate::hdu::ascii_table::AsciiFormat;
        let kind_ok = matches!(
            (&format, &data),
            (AsciiFormat::A(_), AsciiColumnData::Str(_))
                | (AsciiFormat::I(_), AsciiColumnData::Int(_))
                | (
                    AsciiFormat::F(_, _) | AsciiFormat::E(_, _) | AsciiFormat::D(_, _),
                    AsciiColumnData::Float(_),
                )
        );
        if !kind_ok {
            return Err(FitsError::Data(format!(
                "AsciiTableBuilder: column `{}`: data kind does not match format {:?}",
                name.into(),
                format,
            )));
        }
        self.columns.push(AsciiColSpec {
            name: name.into(),
            format,
            data,
            unit: None,
            tnull: None,
        });
        Ok(self)
    }

    /// Set `TUNITn` on the most recently added column.
    pub fn unit(&mut self, unit: impl Into<String>) -> Result<&mut Self> {
        self.last_mut("unit")?.unit = Some(unit.into());
        Ok(self)
    }

    /// Set `TNULLn` on the most recently added column. Only legal for
    /// `I` columns; the sentinel string is padded/truncated to the
    /// field width when written.
    pub fn tnull(&mut self, tnull: impl Into<String>) -> Result<&mut Self> {
        let last = self.last_mut("tnull")?;
        if !matches!(last.format, crate::hdu::ascii_table::AsciiFormat::I(_)) {
            return Err(FitsError::Data(
                "AsciiTableBuilder::tnull: TNULL is only meaningful for `I` columns".into(),
            ));
        }
        last.tnull = Some(tnull.into());
        Ok(self)
    }

    fn last_mut(&mut self, what: &str) -> Result<&mut AsciiColSpec> {
        self.columns
            .last_mut()
            .ok_or_else(|| FitsError::Data(format!("{what}: no column added yet")))
    }

    /// Append an arbitrary value card (e.g. `OBJECT`).
    pub fn card(
        &mut self,
        keyword: impl Into<String>,
        value: impl Into<Value>,
        comment: Option<&str>,
    ) -> &mut Self {
        self.extras.push((
            keyword.into(),
            value.into(),
            comment.map(ToString::to_string),
        ));
        self
    }

    /// Append a `HISTORY` card.
    pub fn history(&mut self, text: impl Into<String>) -> &mut Self {
        self.history.push(text.into());
        self
    }

    /// Set `EXTNAME`.
    pub fn extname(&mut self, name: impl Into<String>) -> &mut Self {
        self.extname = Some(name.into());
        self
    }

    /// Render to `(Header, data)`. Returns an error if the columns
    /// disagree on row count, if any cell does not fit its field
    /// width, or if an `I` column contains `None` cells without
    /// [`tnull`](Self::tnull) set.
    pub fn build(self) -> Result<(Header, Vec<u8>)> {
        if self.columns.is_empty() {
            return Err(FitsError::Data(
                "AsciiTableBuilder: at least one column is required".into(),
            ));
        }
        let n_rows = self.columns[0].data.len();
        for c in &self.columns {
            if c.data.len() != n_rows {
                return Err(FitsError::Data(format!(
                    "AsciiTableBuilder: column `{}` has {} rows; expected {n_rows}",
                    c.name,
                    c.data.len()
                )));
            }
        }

        // Pack columns contiguously: TBCOL1 = 1, TBCOL2 = 1+w1, ...
        let mut tbcol = Vec::with_capacity(self.columns.len());
        let mut start: usize = 1;
        for c in &self.columns {
            tbcol.push(start);
            start += c.format.width();
        }
        let row_size = start - 1;

        let mut data = vec![
            b' ';
            row_size.checked_mul(n_rows).ok_or_else(|| {
                FitsError::Data("row x column count overflowed usize".into())
            })?
        ];
        for r in 0..n_rows {
            for (c, &col_start) in self.columns.iter().zip(tbcol.iter()) {
                let dst = &mut data[r * row_size + (col_start - 1)..][..c.format.width()];
                render_ascii_cell(c, r, dst)?;
            }
        }

        let mut h = Header::empty();
        h.push("XTENSION", Value::String("TABLE".into()), None)?;
        h.push("BITPIX", Value::Integer(8), None)?;
        h.push("NAXIS", Value::Integer(2), None)?;
        h.push("NAXIS1", Value::Integer(row_size as i64), None)?;
        h.push("NAXIS2", Value::Integer(n_rows as i64), None)?;
        h.push("PCOUNT", Value::Integer(0), None)?;
        h.push("GCOUNT", Value::Integer(1), None)?;
        h.push("TFIELDS", Value::Integer(self.columns.len() as i64), None)?;
        for (i, c) in self.columns.iter().enumerate() {
            let n = i + 1;
            h.push(format!("TTYPE{n}"), Value::String(c.name.clone()), None)?;
            h.push(
                format!("TFORM{n}"),
                Value::String(format_ascii_tform(c.format)),
                None,
            )?;
            h.push(format!("TBCOL{n}"), Value::Integer(tbcol[i] as i64), None)?;
            if let Some(u) = &c.unit {
                h.push(format!("TUNIT{n}"), Value::String(u.clone()), None)?;
            }
            if let Some(tn) = &c.tnull {
                h.push(format!("TNULL{n}"), Value::String(tn.clone()), None)?;
            }
        }
        if let Some(name) = &self.extname {
            h.push("EXTNAME", Value::String(name.clone()), None)?;
        }
        for (k, v, c) in self.extras {
            h.push(k, v, c.as_deref())?;
        }
        for line in self.history {
            h.push_commentary(CommentaryKind::History, &line);
        }
        Ok((h, data))
    }
}

fn format_ascii_tform(f: crate::hdu::ascii_table::AsciiFormat) -> String {
    use crate::hdu::ascii_table::AsciiFormat;
    match f {
        AsciiFormat::A(w) => format!("A{w}"),
        AsciiFormat::I(w) => format!("I{w}"),
        AsciiFormat::F(w, d) => format!("F{w}.{d}"),
        AsciiFormat::E(w, d) => format!("E{w}.{d}"),
        AsciiFormat::D(w, d) => format!("D{w}.{d}"),
    }
}

fn render_ascii_cell(c: &AsciiColSpec, row: usize, dst: &mut [u8]) -> Result<()> {
    use crate::hdu::ascii_table::AsciiFormat;
    let w = c.format.width();
    debug_assert_eq!(
        dst.len(),
        w,
        "dst buffer length {0} must match column width {w}",
        dst.len()
    );
    let s: String = match (&c.format, &c.data) {
        (AsciiFormat::A(_), AsciiColumnData::Str(rows)) => {
            let v = &rows[row];
            if v.len() > w {
                return Err(FitsError::Data(format!(
                    "AsciiTableBuilder: column `{}` row {row}: string `{v}` exceeds A{w}",
                    c.name
                )));
            }
            // A-fields are left-justified, padded with spaces.
            format!("{v:<w$}")
        }
        (AsciiFormat::I(_), AsciiColumnData::Int(rows)) => match rows[row] {
            Some(v) => format!("{v:>w$}"),
            None => match c.tnull.as_deref() {
                Some(tn) => {
                    if tn.len() > w {
                        return Err(FitsError::Data(format!(
                            "AsciiTableBuilder: column `{}`: TNULL `{tn}` exceeds I{w}",
                            c.name
                        )));
                    }
                    format!("{tn:>w$}")
                }
                None => {
                    return Err(FitsError::Data(format!(
                        "AsciiTableBuilder: column `{}` row {row} is None but no TNULL set",
                        c.name
                    )));
                }
            },
        },
        (AsciiFormat::F(_, d), AsciiColumnData::Float(rows)) => {
            let v = rows[row];
            if v.is_nan() {
                " ".repeat(w)
            } else {
                format!("{v:>w$.d$}", w = w, d = *d)
            }
        }
        (AsciiFormat::E(_, d), AsciiColumnData::Float(rows)) => {
            let v = rows[row];
            if v.is_nan() {
                " ".repeat(w)
            } else {
                let raw = format!("{v:.d$E}", d = *d);
                format!("{raw:>w$}")
            }
        }
        (AsciiFormat::D(_, d), AsciiColumnData::Float(rows)) => {
            let v = rows[row];
            if v.is_nan() {
                " ".repeat(w)
            } else {
                // Standard Sec.7.2.5: D-format uses `D` exponent.
                let raw = format!("{v:.d$E}", d = *d).replace('E', "D");
                format!("{raw:>w$}")
            }
        }
        // Mismatch was rejected at add_column time; treat as a bug.
        _ => unreachable!("AsciiTableBuilder: format/data mismatch slipped past add_column"),
    };
    if s.len() != w {
        return Err(FitsError::Data(format!(
            "AsciiTableBuilder: column `{}` row {row}: rendered `{s}` is {} bytes, field width is {w}",
            c.name,
            s.len()
        )));
    }
    dst.copy_from_slice(s.as_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_primary_image_round_trips_via_writer() {
        use crate::{FitsFile, FitsWriter, Hdu};

        let pixels: Vec<i16> = (0..12).map(|i| i as i16 * 10).collect();
        let (h, data) = ImageBuilder::<i16>::new(vec![4_u64, 3], pixels.clone())
            .unwrap()
            .primary(true)
            .card("OBJECT", Value::String("synthetic".into()), Some("test"))
            .history("created by ImageBuilder")
            .build()
            .unwrap();

        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu(&h, &data).unwrap();
        w.finish().unwrap();

        let parsed = FitsFile::from_bytes(buf).unwrap();
        let img = match parsed.hdu(0).unwrap() {
            Hdu::Image(i) => i,
            other => panic!("not an image: {other:?}"),
        };
        assert_eq!(img.axes(), &[4_u64, 3]);
        let got: Vec<i16> = img.read_raw::<i16>().unwrap().into_vec();
        assert_eq!(got, pixels);
    }

    #[test]
    fn build_image_extension_with_f32() {
        use crate::{FitsFile, FitsWriter, Hdu};

        let pixels: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
        let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
            .unwrap()
            .primary(true)
            .build()
            .unwrap();
        let ext = ImageBuilder::<f32>::new(vec![2_u64, 2], pixels.clone())
            .unwrap()
            .primary(false)
            .card("EXTNAME", Value::String("DATA".into()), None)
            .build()
            .unwrap();

        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu(&primary.0, &primary.1).unwrap();
        w.write_hdu(&ext.0, &ext.1).unwrap();
        w.finish().unwrap();

        let parsed = FitsFile::from_bytes(buf).unwrap();
        let img = match parsed.hdu(1).unwrap() {
            Hdu::Image(i) => i,
            other => panic!("not an image: {other:?}"),
        };
        let got: Vec<f32> = img.read_raw::<f32>().unwrap().into_vec();
        assert_eq!(got, pixels);
    }

    #[test]
    fn build_bintable_round_trips() {
        use crate::hdu::BinValue;
        use crate::{FitsFile, FitsWriter, Hdu};

        let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
            .unwrap()
            .primary(true)
            .build()
            .unwrap();

        let mut bt = BinTableBuilder::new();
        bt.add_column("ID", BinFieldKind::I32, 1, None, None)
            .unwrap();
        bt.add_column("FLUX", BinFieldKind::F64, 1, Some("Jy"), None)
            .unwrap();
        bt.add_column("NAME", BinFieldKind::Char, 8, None, None)
            .unwrap();
        // Three rows: (1, 1.5, "ALPHA   "), (2, 2.5, "BETA    "), (3, 3.5, "GAMMA   ").
        let mut row_bytes: Vec<u8> = Vec::new();
        for (id, flux, name) in [
            (1_i32, 1.5_f64, b"ALPHA   "),
            (2, 2.5, b"BETA    "),
            (3, 3.5, b"GAMMA   "),
        ] {
            row_bytes.extend_from_slice(&id.to_be_bytes());
            row_bytes.extend_from_slice(&flux.to_bits().to_be_bytes());
            row_bytes.extend_from_slice(name);
        }
        let (h, data) = bt.build(3, row_bytes).unwrap();

        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu(&primary.0, &primary.1).unwrap();
        w.write_hdu(&h, &data).unwrap();
        w.finish().unwrap();

        let parsed = FitsFile::from_bytes(buf).unwrap();
        let table = match parsed.hdu(1).unwrap() {
            Hdu::BinTable(t) => t,
            other => panic!("not a bintable: {other:?}"),
        };
        assert_eq!(table.n_rows(), 3);
        assert_eq!(table.columns().len(), 3);
        let id_col = table.column_by_name("ID").unwrap();
        let v = table.cell_value(2, id_col).unwrap();
        match v {
            BinValue::Int(xs) => assert_eq!(xs, vec![Some(3_i64)]),
            other => panic!("ID col is not Int: {other:?}"),
        }
    }

    #[test]
    fn unsigned_u16_round_trip() {
        use crate::{FitsFile, FitsWriter, Hdu};
        let pixels: Vec<u16> = vec![0, 1, 32_768, 65_535, 12_345, 54_321];
        let (h, data) = ImageBuilder::<i16>::from_u16(vec![3_u64, 2], &pixels)
            .unwrap()
            .primary(true)
            .build()
            .unwrap();
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu(&h, &data).unwrap();
        w.finish().unwrap();

        let parsed = FitsFile::from_bytes(buf).unwrap();
        let Hdu::Image(img) = parsed.hdu(0).unwrap() else {
            panic!("not image");
        };
        let phys = img.read_physical().unwrap().into_vec();
        let expected: Vec<f64> = pixels.iter().map(|&v| f64::from(v)).collect();
        assert_eq!(phys, expected);
    }

    #[test]
    fn unsigned_u32_round_trip() {
        use crate::{FitsFile, FitsWriter, Hdu};
        let pixels: Vec<u32> = vec![0, 1, 2_147_483_648, u32::MAX, 100_000];
        let (h, data) = ImageBuilder::<i32>::from_u32(vec![5_u64], &pixels)
            .unwrap()
            .primary(true)
            .build()
            .unwrap();
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu(&h, &data).unwrap();
        w.finish().unwrap();
        let parsed = FitsFile::from_bytes(buf).unwrap();
        let Hdu::Image(img) = parsed.hdu(0).unwrap() else {
            panic!("not image");
        };
        let phys = img.read_physical().unwrap().into_vec();
        let expected: Vec<f64> = pixels.iter().map(|&v| f64::from(v)).collect();
        assert_eq!(phys, expected);
    }

    #[test]
    fn unsigned_u64_round_trip() {
        use crate::{FitsFile, FitsWriter, Hdu};
        let pixels: Vec<u64> = vec![0, 1, 1_u64 << 32, 1_u64 << 50, u64::MAX];
        let (h, data) = ImageBuilder::<i64>::from_u64(vec![5_u64], &pixels)
            .unwrap()
            .primary(true)
            .build()
            .unwrap();
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu(&h, &data).unwrap();
        w.finish().unwrap();
        let parsed = FitsFile::from_bytes(buf).unwrap();
        let Hdu::Image(img) = parsed.hdu(0).unwrap() else {
            panic!("not image");
        };
        // f64 cannot represent BZERO + small offsets exactly because
        // 2^63 has only 53 bits of mantissa; verify via the raw i64
        // payload + manual offset instead. (Standard Sec.4.4.2.5 notes
        // this -- readers that need full u64 fidelity must avoid f64.)
        let raw: Vec<i64> = img.read_raw::<i64>().unwrap().into_vec();
        let recovered: Vec<u64> = raw
            .iter()
            .map(|&s| (s as u64).wrapping_add(1_u64 << 63))
            .collect();
        assert_eq!(recovered, pixels);
    }

    #[test]
    fn build_bintable_with_vla_p_column_round_trips() {
        use crate::hdu::BinValue;
        use crate::{FitsFile, FitsWriter, Hdu};

        let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
            .unwrap()
            .primary(true)
            .build()
            .unwrap();

        let mut bt = BinTableBuilder::new();
        bt.add_column("ID", BinFieldKind::I32, 1, None, None)
            .unwrap();
        // Variable-length J (i32) column.
        bt.add_vla_column("ARR", BinFieldKind::P, BinFieldKind::I32, None, Some(8))
            .unwrap();

        // Three rows with VLA payloads of length 0, 1, 3.
        let payloads: Vec<Vec<i32>> = vec![vec![], vec![42], vec![10, 20, 30]];

        let mut heap = Vec::<u8>::new();
        let mut row_data = Vec::<u8>::new();
        for (id, payload) in payloads.iter().enumerate() {
            let id = id as i32 + 1;
            row_data.extend_from_slice(&id.to_be_bytes());
            let count = payload.len() as u32;
            let offset = heap.len() as u32;
            row_data.extend_from_slice(&BinTableBuilder::p_descriptor(count, offset));
            for &v in payload {
                heap.extend_from_slice(&v.to_be_bytes());
            }
        }

        let (h, data) = bt.build_with_heap(payloads.len(), row_data, &heap).unwrap();

        // Sanity check on the emitted header.
        assert!(matches!(h.first("PCOUNT"), Some(Value::Integer(p)) if *p > 0));
        match h.first("TFORM2") {
            Some(Value::String(s)) => assert!(s.contains('P') && s.contains('J')),
            _ => panic!("missing TFORM2"),
        }

        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu(&primary.0, &primary.1).unwrap();
        w.write_hdu(&h, &data).unwrap();
        w.finish().unwrap();

        let parsed = FitsFile::from_bytes(buf).unwrap();
        let Hdu::BinTable(t) = parsed.hdu(1).unwrap() else {
            panic!("not bintable");
        };
        assert_eq!(t.n_rows(), payloads.len());
        let arr = t.column_by_name("ARR").unwrap();
        for (row, expected) in payloads.iter().enumerate() {
            let v = t.cell_value(row, arr).unwrap();
            match v {
                BinValue::Vla(inner) => match *inner {
                    BinValue::Int(xs) => {
                        let got: Vec<i32> = xs.into_iter().map(|x| x.unwrap() as i32).collect();
                        assert_eq!(&got, expected, "row {row}");
                    }
                    other => panic!("row {row}: VLA inner not Int: {other:?}"),
                },
                other => panic!("row {row}: not VLA: {other:?}"),
            }
        }
    }

    #[test]
    fn build_bintable_with_vla_q_column_round_trips() {
        use crate::hdu::BinValue;
        use crate::{FitsFile, FitsWriter, Hdu};

        let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
            .unwrap()
            .primary(true)
            .build()
            .unwrap();

        let mut bt = BinTableBuilder::new();
        bt.add_vla_column("DATA", BinFieldKind::Q, BinFieldKind::F64, None, None)
            .unwrap();

        let payloads: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5]];
        let mut heap = Vec::<u8>::new();
        let mut row_data = Vec::<u8>::new();
        for payload in &payloads {
            let count = payload.len() as u64;
            let offset = heap.len() as u64;
            row_data.extend_from_slice(&BinTableBuilder::q_descriptor(count, offset));
            for &v in payload {
                heap.extend_from_slice(&v.to_bits().to_be_bytes());
            }
        }

        let (h, data) = bt.build_with_heap(payloads.len(), row_data, &heap).unwrap();
        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu(&primary.0, &primary.1).unwrap();
        w.write_hdu(&h, &data).unwrap();
        w.finish().unwrap();

        let parsed = FitsFile::from_bytes(buf).unwrap();
        let Hdu::BinTable(t) = parsed.hdu(1).unwrap() else {
            panic!("not bintable");
        };
        let col = t.column_by_name("DATA").unwrap();
        for (row, expected) in payloads.iter().enumerate() {
            let v = t.cell_value(row, col).unwrap();
            match v {
                BinValue::Vla(inner) => match *inner {
                    BinValue::F64(xs) => assert_eq!(&xs, expected, "row {row}"),
                    other => panic!("row {row}: VLA inner not F64: {other:?}"),
                },
                other => panic!("row {row}: not VLA: {other:?}"),
            }
        }
    }

    #[test]
    fn build_ascii_table_round_trips() {
        use crate::hdu::ascii_table::{AsciiCell, AsciiFormat};
        use crate::{FitsFile, FitsWriter, Hdu};

        let primary = ImageBuilder::<u8>::new(Vec::<u64>::new(), Vec::<u8>::new())
            .unwrap()
            .primary(true)
            .build()
            .unwrap();

        let mut bt = AsciiTableBuilder::new();
        bt.add_column(
            "ID",
            AsciiFormat::I(5),
            AsciiColumnData::Int(vec![Some(1), Some(2), None, Some(99999)]),
        )
        .unwrap()
        .tnull("-9999")
        .unwrap();
        bt.add_column(
            "FLUX",
            AsciiFormat::F(10, 3),
            AsciiColumnData::Float(vec![1.5, 2.25, -3.125, f64::NAN]),
        )
        .unwrap()
        .unit("Jy")
        .unwrap();
        bt.add_column(
            "NAME",
            AsciiFormat::A(8),
            AsciiColumnData::Str(vec![
                "alpha".into(),
                "beta".into(),
                "g".into(),
                "delta".into(),
            ]),
        )
        .unwrap();
        bt.extname("CAT");
        let (h, data) = bt.build().unwrap();

        let mut buf = Vec::new();
        let mut w = FitsWriter::new(&mut buf);
        w.write_hdu(&primary.0, &primary.1).unwrap();
        w.write_hdu(&h, &data).unwrap();
        w.finish().unwrap();

        let parsed = FitsFile::from_bytes(buf).unwrap();
        let Hdu::AsciiTable(t) = parsed.hdu(1).unwrap() else {
            panic!("not ascii table");
        };
        assert_eq!(t.n_rows(), 4);

        let id = t.column_by_name("ID").unwrap();
        let flux = t.column_by_name("FLUX").unwrap();
        let name = t.column_by_name("NAME").unwrap();
        assert_eq!(flux.unit, "Jy");

        // Row 0: 1, 1.5, "alpha"
        assert!(matches!(
            t.cell_value(0, id).unwrap(),
            Some(AsciiCell::Int(1))
        ));
        match t.cell_value(0, flux).unwrap() {
            Some(AsciiCell::Float(v)) => assert!((v - 1.5).abs() < 1e-12),
            other => panic!("row 0 flux: {other:?}"),
        }
        match t.cell_value(0, name).unwrap() {
            Some(AsciiCell::Str(s)) => assert_eq!(s.trim_end(), "alpha"),
            other => panic!("row 0 name: {other:?}"),
        }
        // Row 2: TNULL -> None for ID; "g" for name
        assert!(t.cell_value(2, id).unwrap().is_none());
        match t.cell_value(2, name).unwrap() {
            Some(AsciiCell::Str(s)) => assert_eq!(s.trim_end(), "g"),
            other => panic!("row 2 name: {other:?}"),
        }
        // Row 3: NaN -> blank -> None for FLUX
        assert!(t.cell_value(3, flux).unwrap().is_none());

        // EXTNAME round-trips.
        assert_eq!(
            t.header()
                .first("EXTNAME")
                .and_then(|v| match v {
                    Value::String(s) => Some(s.trim().to_string()),
                    _ => None,
                })
                .as_deref(),
            Some("CAT")
        );
    }

    #[test]
    fn ascii_table_builder_rejects_format_data_mismatch() {
        use crate::hdu::ascii_table::AsciiFormat;
        let mut bt = AsciiTableBuilder::new();
        let err = bt
            .add_column("X", AsciiFormat::I(5), AsciiColumnData::Float(vec![1.0]))
            .unwrap_err();
        assert!(format!("{err}").contains("does not match format"));
    }

    #[test]
    fn ascii_table_builder_rejects_overflow_string() {
        use crate::hdu::ascii_table::AsciiFormat;
        let mut bt = AsciiTableBuilder::new();
        bt.add_column(
            "X",
            AsciiFormat::A(3),
            AsciiColumnData::Str(vec!["toolong".into()]),
        )
        .unwrap();
        let err = bt.build().unwrap_err();
        assert!(format!("{err}").contains("exceeds A3"));
    }
}
