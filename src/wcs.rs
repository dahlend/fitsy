//! World Coordinate System (Standard Sec.8, Greisen & Calabretta 2002,
//! Calabretta & Greisen 2002).
//!
//! # Purpose
//!
//! [`Wcs`] describes the coordinates of one HDU and transforms between
//! pixel and world values. It runs the four-step pipeline of Paper I
//! Sec.8.1: pixel offset from `CRPIX`, the linear matrix, the
//! projection, and the spherical rotation.
//!
//! [`Wcs::from_header`] parses a [`Header`](crate::Header). For a file
//! with a tabular `-TAB` axis, call
//! [`FitsFile::wcs`](crate::FitsFile::wcs) instead. That method
//! resolves the table data as well.
//!
//! # Layout
//!
//! - [`linear`] -- the `CRPIX`, `CDELT`, `PCi_j` and `CDi_j` step.
//! - [`projection`] -- the celestial projections of Paper II Table 13,
//!   selected by the `CTYPE` code.
//! - [`celestial`] -- the spherical rotation and the reference frame.
//! - [`spectral`] -- the spectral `CTYPE` codes and the non-linear
//!   algorithms of Paper III Sec.3.3 (Greisen et al. 2006).
//! - [`tab`] -- the `-TAB` lookup axes of Paper III Sec.6.
//! - [`time`] -- the time axis of Sec.9.5.3.
//! - [`distortion`] -- SIP, TPV, TNX and the DSS plate solution, the
//!   conventions that sit outside the `PVi_m` family.
//! - [`table`] -- the table-resident forms of Sec.8.2, Table 22.
//! - [`fit_celestial_wcs`] -- fits a celestial WCS from pixel and sky
//!   pairs.
//!
//! # Design constraints
//!
//! [`Wcs`] is the interpreted layer of the crate.
//! [`Header`](crate::Header) preserves the file and holds every card
//! as written, including a card that contradicts another. [`Wcs`]
//! holds only the keywords that carry meaning in the description it
//! parsed. `ZSOURCE` on a header with no spectral axis is dropped
//! here, and so is `EQUINOX` under `RADESYS = 'ICRS'`, which defines
//! the equinox away. The original cards remain in the `Header` they
//! were parsed from.
//!
//! [`Wcs::to_header`] therefore emits that interpretation, not a
//! reproduction of the source. Its round-trip contract is
//! `from_header(to_header(w)) == w`. It is not byte fidelity to the
//! file.
//!
//! Parsing is lenient in one direction only. A keyword outside the
//! description is discarded rather than rejected. A `CTYPE` that names
//! an algorithm code outside Sec.8.4 Table 26 is an error, because
//! evaluating such an axis as a linear one would report a wrong
//! coordinate as though it were right.
//!
//! A time axis transforms linearly, which is what Sec.9.5.3 defines.
//! It therefore reports elapsed time, not an absolute epoch.

pub mod celestial;
pub mod celestial_block;
pub mod distortion;
pub mod linear;
pub mod projection;
pub mod spectral;
pub mod tab;
pub mod table;
pub mod time;

mod fit;
mod parse;
mod serialize;

pub use celestial::{CelestialFrame, RadeSys};
pub use celestial_block::{CelestialBlock, CelestialPair};
// The distortion conventions are applied by the transform pipeline
// here, so a caller inspecting one needs it at the same level.
pub use distortion::{Dss, Sip, Tnx, Tpv};
pub use fit::{WcsFit, WcsFitOptions, fit_celestial_wcs};
pub use linear::LinearTransform;
// The projection structs are the payloads of `Projection`'s variants
// and the arguments to its `From` impls, so a caller writing
// `Projection::from(Tan)` needs them at the same level as the enum.
pub use projection::{
    Air, Ait, Arc, Azp, Bon, Car, Cea, Cod, Coe, Coo, Cop, Csc, Cyp, Hpx, Mer, Mol, Par, Pco,
    Projection, Qsc, Sfl, Sin, Stg, Szp, Tan, Tsc, Xph, Zea, Zpn,
};
// `Linearized` and `Grism` are reachable from `SpectralAlgorithm`'s
// variants, so a caller matching on one needs them at the same level.
pub use spectral::{
    Grism, Linearized, SourceFrame, SpectralAlgorithm, SpectralAxis, SpectralFrame, SpectralKind,
};
pub use tab::{TabGroup, TabSpec};
pub use table::TableWcs;
pub use time::{PhaseAxis, TimeAxis};

use crate::error::{FitsError, Result};

/// Degrees -> radians.
pub(crate) const D2R: f64 = std::f64::consts::PI / 180.0;
/// Radians -> degrees.
pub(crate) const R2D: f64 = 180.0 / std::f64::consts::PI;

/// The suffix an alternate code contributes to a WCS keyword: nothing
/// for the primary description, the letter itself otherwise.
///
/// # Errors
///
/// [`FitsError::Wcs`] when `alt` is neither a space nor an ASCII
/// uppercase letter. Sec.8.2 admits no other code.
// Shared by every entry point so the rule is stated once.
pub(crate) fn alt_suffix(alt: char) -> Result<String> {
    if alt == ' ' {
        return Ok(String::new());
    }
    if !alt.is_ascii_uppercase() {
        return Err(FitsError::Wcs(format!(
            "WCS alternate code must be ' ' or 'A'..'Z' (got {alt:?})"
        )));
    }
    Ok(alt.to_string())
}

/// A parsed WCS for one alternate descriptor (`' '`, `'A'` to `'Z'`).
///
/// # Examples
///
/// ```
/// use fitsy::{AxisKind, Header, Wcs};
///
/// let mut h = Header::empty();
/// h.push("NAXIS", 2_i64, None)?;
/// h.push("CTYPE1", "RA---TAN", None)?;
/// h.push("CTYPE2", "DEC--TAN", None)?;
/// h.push("CRPIX1", 32.0_f64, None)?;
/// h.push("CRPIX2", 24.0_f64, None)?;
/// h.push("CRVAL1", 150.0_f64, None)?;
/// h.push("CRVAL2", 2.5_f64, None)?;
/// h.push("CDELT1", -0.001_f64, None)?;
/// h.push("CDELT2", 0.001_f64, None)?;
///
/// let wcs = Wcs::from_header(&h, ' ')?.expect("header declares a WCS");
///
/// // Pixel coordinates are 0-based, so CRPIX 32 is pixel 31.
/// // `pixel_to_world` returns one value per axis, in axis order;
/// // `axis_kinds` says which value is which.
/// let world = wcs.pixel_to_world(&[31.0, 23.0])?;
/// assert_eq!(wcs.axis_kinds(), vec![AxisKind::Longitude, AxisKind::Latitude]);
/// assert!((world[0] - 150.0).abs() < 1e-9);
/// assert!((world[1] - 2.5).abs() < 1e-9);
/// # Ok::<(), fitsy::FitsError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Wcs {
    // Private so `naxis()` can be its length: public parallel vectors
    // could be truncated out of step and the pipeline indexed them
    // unchecked. Read via `axes()`, `axis()`, `ctype()`, `cunit()`.
    axes: Vec<Axis>,
    /// Linear stage of the pipeline: `CRPIX`, `CRVAL`, and the combined
    /// `CDELT`/`PC` or `CD` matrix.
    ///
    /// Private for the same reason as `axes`. The per-point bodies
    /// index `CRPIX` unchecked and zip the matrix against `naxis`
    /// rows, so a transform of a different rank must be impossible
    /// rather than checked per call. [`Self::new`] and
    /// [`Self::set_linear`] validate the rank once. Read via
    /// [`Self::linear`].
    linear: LinearTransform,
    /// Celestial axis pair plus everything that depends on it
    /// (projection, native<->celestial rotation, optional SIP/TPV).
    /// Either every component is present or `celestial` is `None` --
    /// the type system enforces the all-or-nothing rule.
    pub celestial: Option<CelestialBlock>,
    /// Spectral axes (Paper III). Each entry is keyed by its
    /// zero-based axis index (`SpectralAxis::axis`); axes not in
    /// this list are treated as plain linear coordinates.
    pub spectral: Vec<SpectralAxis>,
    /// `RADESYS` keyword (Paper II Sec.3.1) -- only meaningful for
    /// equatorial frames.
    pub radesys: RadeSys,
    /// `EQUINOX` keyword (Julian or Besselian epoch, depending on
    /// `radesys`).
    pub equinox: Option<f64>,
    /// `MJD-OBS` (Modified Julian Date of observation, days).
    pub mjd_obs: Option<f64>,
    /// `WCSNAME` keyword (Standard Sec.8.2.6) -- free-form name for this
    /// alternate coordinate description. `None` when not supplied.
    pub wcsname: Option<String>,
    /// Spectral reference frames (Paper III Sec.7, Standard
    /// Sec.8.4.3): `SPECSYS`, `SSYSOBS`, `VELOSYS`, and the `ZSOURCE`
    /// source-frame group. `None` when the description has no spectral
    /// axis -- see the module-level layering note.
    pub spectral_frame: Option<SpectralFrame>,
    /// Optional DSS plate solution (non-standard). When present it
    /// replaces the standard celestial pipeline for the celestial
    /// axis pair: pixels go straight through `Dss::pixel_to_world`,
    /// bypassing CRPIX, the linear matrix, SIP, TPV, TNX, and the
    /// projection.
    pub dss: Option<Dss>,
    /// Tabular `-TAB` axes parsed out of the header. Each entry
    /// records the binary-table extension and column names; the
    /// actual coordinate data is loaded by [`Self::resolve_tab`].
    /// Empty when the header has no `-TAB` axes.
    pub tab_specs: Vec<TabSpec>,
    /// Resolved `-TAB` lookup tables (populated by
    /// [`Self::resolve_tab`] or by [`crate::FitsFile::wcs`]). One
    /// entry per table, driving the *M* axes that share it. While a
    /// `-TAB` axis is parsed but unresolved, `pixel_to_world` /
    /// `world_to_pixel` return a clear error rather than silently
    /// dropping the lookup.
    pub tab: Vec<TabGroup>,
    /// Time axis (Standard Sec.9.5.3), if a `CTYPE` names one,
    /// carrying the `TREFPOS`/`TREFDIR`/`PLEPHEM` reference-position
    /// trio with it.
    ///
    /// This field is descriptive. Sec.9.5.3 defines the transform as
    /// linear, so the pipeline handles a time axis as it handles any
    /// other linear one. This records which axis is time, and on what
    /// scale.
    pub time: Option<TimeAxis>,
    /// Phase axes (Sec.9.6, `CTYPE = 'PHASE'`), each carrying its
    /// `CZPHSia`/`CPERIia` pair. Empty when no axis is one.
    pub phase: Vec<PhaseAxis>,
    /// The celestial axis pair, if the CTYPEs name one.
    ///
    /// Set even when [`Self::celestial`] is `None`, which happens for
    /// a `RA---TAB`/`DEC--TAB` pair: those carry no projection at all,
    /// their coordinates coming straight from the lookup table.
    ///
    /// When the block exists it holds the same pair. Construction
    /// rejects a disagreement.
    ///
    /// Classification queries and the serializer read this field.
    /// The transform pipeline reads the block's copy.
    pub celestial_pair: Option<CelestialPair>,
    /// Snapshot of the `NAXISn` cards, in FITS axis order, for callers
    /// like [`Self::footprint`] that need the image extent. Not part of
    /// the coordinate description: nothing in the pipeline reads it,
    /// and it is not re-checked, so a cropped or rebinned image leaves
    /// it stale.
    ///
    /// `None` for a fitted WCS or a header without `NAXISn`. Its
    /// length can differ from [`Self::naxis`] when `WCSAXES` declares
    /// a different number of coordinate axes.
    pub pixel_shape: Option<Vec<u64>>,
}

/// Everything the standard attaches to a single coordinate axis
/// (Sec.8.2, Table 22).
///
/// One entry per axis; [`Wcs::naxis`] is the length of the list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Axis {
    /// `CTYPEia` -- axis type and, past the fourth character, the
    /// algorithm code. Blank means a plain linear axis.
    pub ctype: String,
    /// `CUNITia` -- units of `CRVALia` and `CDELTia`. Empty when
    /// absent, which means the units Sec.8.1 fixes for the axis type.
    pub cunit: String,
    /// `CNAMEia` -- free-form name for the axis, the per-axis
    /// counterpart of `WCSNAMEa`. Descriptive only; the standard
    /// attaches no meaning to the string.
    pub cname: Option<String>,
    /// `CRDERia` -- random error in the coordinate. Sec.9.4.3 makes
    /// this the per-axis override of `TIMRDER`.
    pub crder: Option<f64>,
    /// `CSYERia` -- systematic error in the coordinate; the per-axis
    /// override of `TIMSYER`.
    pub csyer: Option<f64>,
}

/// Everything [`Wcs::new`] needs beyond the axis list and the linear
/// transform. The fields move into the `Wcs` unchanged.
// A bundle rather than sixteen positional parameters, so the two
// constructors read as named fields.
#[derive(Debug, Default)]
pub(crate) struct WcsParts {
    /// Celestial axis pair and everything attached to it.
    pub celestial: Option<CelestialBlock>,
    /// Spectral axes, keyed by zero-based axis index.
    pub spectral: Vec<SpectralAxis>,
    /// `RADESYS` keyword.
    pub radesys: RadeSys,
    /// `EQUINOX` keyword.
    pub equinox: Option<f64>,
    /// `MJD-OBS` keyword, in days.
    pub mjd_obs: Option<f64>,
    /// `WCSNAME` keyword.
    pub wcsname: Option<String>,
    /// `SPECSYS` group. `None` without a spectral axis.
    pub spectral_frame: Option<SpectralFrame>,
    /// DSS plate solution, which replaces the celestial pipeline.
    pub dss: Option<Dss>,
    /// Parsed but unresolved `-TAB` axis pointers.
    pub tab_specs: Vec<TabSpec>,
    /// Resolved `-TAB` lookup tables.
    pub tab: Vec<TabGroup>,
    /// Time axis, if a `CTYPE` names one.
    pub time: Option<TimeAxis>,
    /// Phase axes (Sec.9.6).
    pub phase: Vec<PhaseAxis>,
    /// Which two axes form the celestial pair.
    pub celestial_pair: Option<CelestialPair>,
    /// `NAXISn` lengths, when the caller supplied them.
    pub pixel_shape: Option<Vec<u64>>,
}

/// The kind of coordinate an axis carries.
///
/// This names the *type* half of `CTYPEia`, the part before the
/// algorithm code (Sec.8.1). The algorithm half describes how the
/// coordinate is computed, not what it is, so `RA---TAN` and
/// `RA---TAB` are both [`AxisKind::Longitude`]. Ask
/// [`Wcs::is_tabular`] for the `-TAB` case, which is the one algorithm
/// a caller usually has to plan around.
///
/// Obtain one with [`Wcs::axis_kind`], or the whole set with
/// [`Wcs::axis_kinds`].
///
/// This enum is exhaustive. Sec.8.1 fixes the set of coordinate types,
/// and a caller matching on it should get a compile error rather than
/// a silent fallthrough if that set ever grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisKind {
    /// Celestial longitude, such as `RA` or `GLON`.
    ///
    /// This kind comes from the celestial pair, not from the type code
    /// alone. Sec.8.2 defines a spherical projection over two axes. A
    /// longitude with no matching latitude therefore has no projection.
    /// The pipeline transforms that axis linearly. [`Wcs::axis_kind`]
    /// reports [`AxisKind::Linear`] for it. Use
    /// [`Wcs::celestial_axes`] to locate the pair itself.
    Longitude,
    /// Celestial latitude, such as `DEC` or `GLAT`.
    ///
    /// This kind pairs with [`AxisKind::Longitude`] and follows the
    /// same rule.
    Latitude,
    /// Spectral axis (Paper III): `FREQ`, `WAVE`, `ENER`, and the
    /// rest of [`SpectralKind`].
    Spectral,
    /// Time axis (Sec.9.5.3).
    Time,
    /// Phase axis (Sec.9.6).
    Phase,
    /// Stokes polarization (Sec.8.1 Table 25).
    Stokes,
    /// A plain linear axis, and any type this crate does not
    /// recognize. `CDELT` and `CRVAL` still describe it.
    Linear,
}

/// Working buffers for one coordinate transform.
///
/// A transform needs four vectors of length `NAXIS`. Allocating them
/// per point dominates the cost of a batch call. The batch entry
/// points therefore build one `Scratch` and reuse it for every point.
#[derive(Debug)]
struct Scratch {
    /// Forward: 1-based pixel coordinates. Unused by the inverse.
    pix: Vec<f64>,
    /// Pixel offsets from `CRPIX`, before or after the linear matrix.
    dp: Vec<f64>,
    /// Intermediate world coordinates.
    intermediate: Vec<f64>,
    /// Result. World coordinates forward, pixel coordinates inverse.
    out: Vec<f64>,
    /// Values for one multi-axis `-TAB` group. Grows to the size of
    /// the largest group and then stays there.
    tab: Vec<f64>,
}

impl Scratch {
    fn new(naxis: usize) -> Self {
        Self {
            pix: vec![0.0; naxis],
            dp: vec![0.0; naxis],
            intermediate: vec![0.0; naxis],
            out: vec![0.0; naxis],
            tab: Vec::new(),
        }
    }
}

/// A two-axis celestial WCS with at most one pixel-space and one
/// intermediate-space distortion attached, every per-point input
/// resolved.
///
/// [`Wcs::pixel_to_world_into`] is written for the general case, so it
/// re-derives loop-invariant state for every point. It probes several
/// `Option` fields. It walks the spectral and `-TAB` lists. It reads
/// `CRPIX` and the matrix through slice-returning accessors and
/// indexes the celestial pair with runtime indices. None of that
/// varies across a batch.
///
/// This resolves all of it once. The parts it removes are worth about a
/// third of the per-point cost on a `TAN` batch. That saving is spread
/// across several small terms rather than concentrated in one.
///
/// This path carries SIP, TPV and TNX rather than declining them.
/// SIP and TPV are the two common distortions on survey images. Each
/// costs a few multiply-adds of its own. TNX occupies the same
/// pipeline slot as TPV, with the same analytic-Jacobian inverse
/// shape. Sending any of them to the general body added the
/// per-point overhead above on top of that cost.
///
/// The saving is smaller for TPV and TNX than for SIP, because their
/// inverses iterate and that iteration dominates what the
/// specialization removes. It is still a saving, and it costs the
/// other paths nothing.
///
/// The two paths must agree bit for bit. `fast_path_matches_general` in
/// `tests/wcs.rs` is what holds them together.
struct FastCelestial<'a> {
    /// 1-based reference pixel, both axes.
    crpix: [f64; 2],
    /// Combined linear matrix, row-major.
    matrix: [f64; 4],
    /// Index of the longitude axis, 0 or 1. Latitude is the other.
    lon: usize,
    /// `CUNIT` scaling to degrees, longitude then latitude.
    cunit_to_deg: (f64, f64),
    /// Projection-plane offset of the fiducial point.
    fiducial_offset: (f64, f64),
    /// Inverse of `matrix`, row-major.
    inverse: [f64; 4],
    /// Borrowed, so each point pays one match dispatch and no clone
    /// of the projection's parameters.
    projection: &'a Projection,
    /// Native to celestial rotation.
    rotation: &'a celestial::CelestialRotation,
    /// Pixel-space SIP distortion, when the header carries one.
    sip: Option<&'a Sip>,
    /// Intermediate-space TPV distortion, when the header carries one.
    ///
    /// Borrowed rather than copied: the two coefficient tables are 640
    /// bytes, which is not something to move per point.
    tpv: Option<&'a Tpv>,
    /// Intermediate-space TNX/ZPX distortion, when the header carries
    /// one. Occupies the same pipeline slot as TPV; the parser sets at
    /// most one of the two.
    tnx: Option<&'a Tnx>,
}

impl FastCelestial<'_> {
    /// One point, or `None` when the projection rejects it.
    ///
    /// Mirrors the surviving steps of [`Wcs::pixel_to_world_into`] in
    /// the same order, so the two produce identical bits.
    #[inline]
    fn point(&self, px: f64, py: f64) -> Option<(f64, f64)> {
        // 0-based -> 1-based, then offset from CRPIX.
        let mut d0 = px + 1.0 - self.crpix[0];
        let mut d1 = py + 1.0 - self.crpix[1];
        // SIP pixel-space distortion, which the general body applies to
        // the celestial pair in `(lon, lat)` order and writes back to
        // the same two slots.
        if let Some(sip) = self.sip {
            if self.lon == 0 {
                (d0, d1) = sip.forward(d0, d1);
            } else {
                (d1, d0) = sip.forward(d1, d0);
            }
        }
        // Linear matrix.
        let i0 = self.matrix[0] * d0 + self.matrix[1] * d1;
        let i1 = self.matrix[2] * d0 + self.matrix[3] * d1;
        // The general body writes `CRVAL + intermediate` for every axis
        // and then overwrites both celestial slots. With two celestial
        // axes those writes are dead, so this omits them.
        let (ilon, ilat) = if self.lon == 0 { (i0, i1) } else { (i1, i0) };
        let (fx, fy) = self.cunit_to_deg;
        let (mut x, mut y) = (ilon * fx, ilat * fy);
        // TPV sits between the linear stage and the projection, and
        // TNX shares that slot; the general body applies TPV first.
        if let Some(tpv) = self.tpv {
            (x, y) = tpv.forward(x, y);
        }
        if let Some(tnx) = self.tnx {
            (x, y) = tnx.forward(x, y);
        }
        let (phi, theta) = self
            .projection
            .x2s(x + self.fiducial_offset.0, y + self.fiducial_offset.1)
            .ok()?;
        Some(self.rotation.native_to_celestial(phi, theta))
    }

    /// One point of the inverse, or `None` when the projection or the
    /// SIP inverse rejects it.
    ///
    /// Mirrors the surviving steps of [`Wcs::world_to_pixel_into`].
    /// `CRVAL` does not appear. The general body seeds `intermediate`
    /// with `world - CRVAL`, then overwrites both celestial slots from
    /// the projection. With two celestial axes that seed is dead.
    #[inline]
    fn inverse_point(&self, lon_deg: f64, lat_deg: f64) -> Option<(f64, f64)> {
        let (phi, theta) = self.rotation.celestial_to_native(lon_deg, lat_deg);
        let (x_proj, y_proj) = self.projection.s2x(phi, theta).ok()?;
        let mut x = x_proj - self.fiducial_offset.0;
        let mut y = y_proj - self.fiducial_offset.1;
        // Inverse TNX then inverse TPV, before the `CUNIT` rescaling,
        // which is the order the general body uses. Each iterates, so
        // each can fail to converge -- a per-point rejection like the
        // projection's.
        if let Some(tnx) = self.tnx {
            (x, y) = tnx.inverse(x, y).ok()?;
        }
        if let Some(tpv) = self.tpv {
            (x, y) = tpv.inverse(x, y).ok()?;
        }
        let (fx, fy) = self.cunit_to_deg;
        let (ilon, ilat) = (x / fx, y / fy);
        // Back into axis order before the inverse matrix.
        let (i0, i1) = if self.lon == 0 {
            (ilon, ilat)
        } else {
            (ilat, ilon)
        };
        let mut d0 = self.inverse[0] * i0 + self.inverse[1] * i1;
        let mut d1 = self.inverse[2] * i0 + self.inverse[3] * i1;
        // Inverse SIP, after the inverse matrix and before `CRPIX`,
        // which is the order the general body uses. It iterates, so it
        // can fail to converge. That is a per-point rejection, like the
        // one the projection reports.
        if let Some(sip) = self.sip {
            if self.lon == 0 {
                (d0, d1) = sip.inverse(d0, d1).ok()?;
            } else {
                (d1, d0) = sip.inverse(d1, d0).ok()?;
            }
        }
        // 1-based -> 0-based.
        Some((self.crpix[0] + d0 - 1.0, self.crpix[1] + d1 - 1.0))
    }

    /// The inverse batch, in the flat layout of
    /// [`Wcs::world_to_pixel_many`].
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `world` is not a whole number of points.
    fn world_to_pixel_many(&self, world: &[f64]) -> Result<Vec<f64>> {
        if !world.len().is_multiple_of(2) {
            return Err(FitsError::Wcs(format!(
                "expected a whole multiple of 2 world coordinates, got {}",
                world.len()
            )));
        }
        let mut out = vec![0.0; world.len()];
        let lat = 1 - self.lon;
        for (src, dst) in world
            .as_chunks::<2>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<2>().0.iter_mut())
        {
            match self.inverse_point(src[self.lon], src[lat]) {
                Some((px, py)) => {
                    dst[0] = px;
                    dst[1] = py;
                }
                None => dst.fill(f64::NAN),
            }
        }
        Ok(out)
    }

    /// The batch, in the flat layout of [`Wcs::pixel_to_world_many`].
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `pix` is not a whole number of points.
    fn pixel_to_world_many(&self, pix: &[f64]) -> Result<Vec<f64>> {
        if !pix.len().is_multiple_of(2) {
            return Err(FitsError::Wcs(format!(
                "expected a whole multiple of 2 pixel coordinates, got {}",
                pix.len()
            )));
        }
        let mut out = vec![0.0; pix.len()];
        let lat = 1 - self.lon;
        for (src, dst) in pix
            .as_chunks::<2>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<2>().0.iter_mut())
        {
            match self.point(src[0], src[1]) {
                Some((alpha, delta)) => {
                    dst[self.lon] = alpha;
                    dst[lat] = delta;
                }
                None => dst.fill(f64::NAN),
            }
        }
        Ok(out)
    }
}

impl Wcs {
    /// Resolve the fast path, or `None` when the general body is
    /// needed.
    ///
    /// The conditions cover what this path reproduces: two axes, both
    /// celestial, and no spectral, tabular or plate-solution stage.
    /// This path carries SIP, TPV and TNX.
    fn fast_celestial(&self) -> Option<FastCelestial<'_>> {
        if self.naxis() != 2
            || !self.spectral.is_empty()
            || !self.tab.is_empty()
            || !self.tab_specs.is_empty()
            || self.dss.is_some()
        {
            return None;
        }
        let c = self.celestial.as_ref()?;
        // Both axes must be the celestial pair, or an axis would need
        // the `CRVAL` step this path skips.
        if c.pair.lon.max(c.pair.lat) > 1 || c.pair.lon == c.pair.lat {
            return None;
        }
        let m = self.linear.matrix_row_major();
        let inv = self.linear.inverse_row_major();
        let crpix = self.linear.crpix();
        if m.len() != 4 || inv.len() != 4 || crpix.len() != 2 {
            return None;
        }
        Some(FastCelestial {
            crpix: [crpix[0], crpix[1]],
            matrix: [m[0], m[1], m[2], m[3]],
            inverse: [inv[0], inv[1], inv[2], inv[3]],
            lon: c.pair.lon,
            cunit_to_deg: c.cunit_to_deg,
            fiducial_offset: c.fiducial_offset,
            projection: &c.projection,
            rotation: &c.rotation,
            sip: c.sip.as_ref(),
            tpv: c.tpv.as_ref(),
            tnx: c.tnx.as_ref(),
        })
    }
}

impl Wcs {
    /// Largest `NAXIS` [`Self::footprint`] accepts.
    ///
    /// The corner count doubles with every axis. `NAXIS` comes from
    /// the header, so an unchecked value would let a file ask for an
    /// unbounded allocation. Sixteen axes is 65536 corners, which
    /// exceeds any WCS in use.
    pub const MAX_FOOTPRINT_AXES: usize = 16;

    /// Number of coordinate axes.
    ///
    /// This is the WCS axis count, which `WCSAXESa` may set higher than
    /// `NAXIS` (Sec.8.2).
    #[must_use]
    pub fn naxis(&self) -> usize {
        self.axes.len()
    }

    /// Per-axis descriptions, in FITS order.
    #[must_use]
    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// Axis `i` (zero-based), or `None` past the end.
    #[must_use]
    pub fn axis(&self, i: usize) -> Option<&Axis> {
        self.axes.get(i)
    }

    /// `CTYPEia` for axis `i` (zero-based); `""` past the end.
    #[must_use]
    pub fn ctype(&self, i: usize) -> &str {
        self.axes.get(i).map_or("", |a| a.ctype.as_str())
    }

    /// `CUNITia` for axis `i` (zero-based); `""` past the end, which
    /// is also what an absent card yields.
    #[must_use]
    pub fn cunit(&self, i: usize) -> &str {
        self.axes.get(i).map_or("", |a| a.cunit.as_str())
    }

    /// `CRVALia` for every axis, in the matching [`Axis::cunit`].
    #[must_use]
    pub fn crval(&self) -> &[f64] {
        self.linear.crval()
    }

    /// Assemble a `Wcs` from its parts.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in three cases:
    ///
    /// - The linear transform and the axis list disagree on the axis
    ///   count.
    /// - A spectral, phase, time, `-TAB` or celestial entry names an
    ///   axis index the description does not have.
    /// - `celestial` and `celestial_pair` disagree.
    // `LinearTransform` and `Axis` each validate themselves, so what
    // is left is the agreement between them, plus the axis indices in
    // `SpectralAxis`, `PhaseAxis`, `TabSpec`, `TabGroup` and
    // `CelestialPair` -- each of which indexes a `naxis`-long slice.
    pub(crate) fn new(axes: Vec<Axis>, linear: LinearTransform, parts: WcsParts) -> Result<Self> {
        let naxis = axes.len();
        if linear.naxis() != naxis {
            return Err(FitsError::Wcs(format!(
                "WCS has {naxis} axis description(s) but a {}-axis linear transform",
                linear.naxis(),
            )));
        }
        let check = |what: &str, axis: usize| -> Result<()> {
            if axis >= naxis {
                return Err(FitsError::Wcs(format!(
                    "WCS {what} names axis {} of a {naxis}-axis description",
                    axis + 1,
                )));
            }
            Ok(())
        };
        for sx in &parts.spectral {
            check("spectral axis", sx.axis)?;
        }
        for p in &parts.phase {
            check("phase axis", p.axis)?;
        }
        for s in &parts.tab_specs {
            check("-TAB axis", s.axis)?;
        }
        for g in &parts.tab {
            for &a in &g.axes {
                check("-TAB group axis", a)?;
            }
        }
        if let Some(t) = &parts.time {
            check("time axis", t.axis)?;
        }
        if let Some(p) = &parts.celestial_pair {
            check("celestial longitude axis", p.lon)?;
            check("celestial latitude axis", p.lat)?;
        }
        // The pair is stored twice when a block exists. `celestial_pair`
        // answers classification queries (`axis_kind`, `is_celestial`,
        // `celestial_axes`, the serializer's frame gate). `celestial.pair`
        // drives the transform pipeline. Both fields are public, so a
        // disagreement between them is otherwise possible. It would send
        // the two sets of consumers to different axes. Check it here,
        // the one choke point both construction paths pass through.
        if let Some(c) = &parts.celestial {
            match &parts.celestial_pair {
                Some(p) if *p == c.pair => {}
                Some(p) => {
                    return Err(FitsError::Wcs(format!(
                        "WCS celestial pair {p:?} disagrees with its \
                         celestial block's pair {:?}",
                        c.pair,
                    )));
                }
                None => {
                    return Err(FitsError::Wcs(
                        "WCS has a celestial block but no celestial pair".into(),
                    ));
                }
            }
        }
        Ok(Self {
            axes,
            linear,
            celestial: parts.celestial,
            spectral: parts.spectral,
            radesys: parts.radesys,
            equinox: parts.equinox,
            mjd_obs: parts.mjd_obs,
            wcsname: parts.wcsname,
            spectral_frame: parts.spectral_frame,
            dss: parts.dss,
            tab_specs: parts.tab_specs,
            tab: parts.tab,
            time: parts.time,
            phase: parts.phase,
            celestial_pair: parts.celestial_pair,
            pixel_shape: parts.pixel_shape,
        })
    }

    /// Transform pixel coordinates to world coordinates.
    ///
    /// # Pixel indexing convention
    ///
    /// Pixel coordinates in this API are 0-based. The center of the
    /// first pixel is `(0.0, 0.0, ...)`. FITS itself is 1-based
    /// (Sec.3.3.4), so subtract 1 from a coordinate that came from a
    /// 1-based source before passing it here.
    ///
    /// This convention holds for every pixel-coordinate method on
    /// `Wcs`: [`pixel_to_world`], [`world_to_pixel`],
    /// [`pixel_to_world_many`], [`world_to_pixel_many`],
    /// [`footprint`] and [`pixel_scale_at`].
    ///
    /// This method transforms one point and fails on a point it cannot
    /// transform. [`pixel_to_world_many`] transforms a batch and marks
    /// such a point `NaN` instead.
    ///
    /// A world value comes back in the unit its `CUNIT` names. A
    /// celestial axis therefore reports degrees.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in four cases:
    ///
    /// - `pix.len()` does not equal `NAXIS`.
    /// - A `-TAB` axis remains unresolved.
    /// - The point lies outside the domain of the projection.
    /// - A spectral or tabular axis cannot evaluate the point.
    ///
    /// [`pixel_to_world`]: Self::pixel_to_world
    /// [`world_to_pixel`]: Self::world_to_pixel
    /// [`pixel_to_world_many`]: Self::pixel_to_world_many
    /// [`world_to_pixel_many`]: Self::world_to_pixel_many
    /// [`footprint`]: Self::footprint
    /// [`pixel_scale_at`]: Self::pixel_scale_at
    pub fn pixel_to_world(&self, pix: &[f64]) -> Result<Vec<f64>> {
        self.check_tab_resolved()?;
        let mut scratch = Scratch::new(self.naxis());
        self.pixel_to_world_into(pix, &mut scratch)?;
        Ok(scratch.out)
    }

    /// The linear stage of the pipeline: `CRPIX`, `CRVAL`, and the
    /// combined `CDELT`/`PC` or `CD` matrix.
    #[must_use]
    pub fn linear(&self) -> &LinearTransform {
        &self.linear
    }

    /// Replace the linear stage.
    ///
    /// The per-point bodies index `CRPIX` unchecked and zip the matrix
    /// against `naxis` rows, so the rank is validated here, once,
    /// rather than on every transform call.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `linear` does not describe `naxis`
    /// axes.
    pub fn set_linear(&mut self, linear: LinearTransform) -> Result<()> {
        let n = self.naxis();
        if linear.naxis() != n {
            return Err(FitsError::Wcs(format!(
                "WCS describes {n} axes but the linear transform describes {}",
                linear.naxis()
            )));
        }
        self.linear = linear;
        Ok(())
    }

    /// Forward transform of one point, writing into `s.out`.
    ///
    /// This holds the body of [`Self::pixel_to_world`]. It exists so a
    /// batch loop can reuse one [`Scratch`] instead of allocating a
    /// vector per point.
    ///
    /// The caller runs [`Self::check_tab_resolved`] first. This is
    /// the per-point work alone.
    ///
    /// # Errors
    ///
    /// The conditions of [`Self::pixel_to_world`] that belong to the
    /// point: a wrong length, a point outside the projection, and a
    /// spectral or tabular axis that cannot evaluate it.
    fn pixel_to_world_into(&self, pix: &[f64], s: &mut Scratch) -> Result<()> {
        let n = self.naxis();
        if pix.len() != n {
            return Err(FitsError::Wcs(format!(
                "expected {n} pixel coordinates, got {}",
                pix.len()
            )));
        }
        // Step 1: 0-based -> 1-based, then offset relative to CRPIX.
        // See the doc comment on `pixel_to_world`.
        let crpix = self.linear.crpix();
        for j in 0..n {
            s.pix[j] = pix[j] + 1.0;
            s.dp[j] = s.pix[j] - crpix[j];
        }
        // Step 2: SIP pixel-space distortion (celestial pair only).
        if let Some(c) = self.celestial.as_ref()
            && let Some(sip) = c.sip.as_ref()
        {
            let (u, v) = (s.dp[c.pair.lon], s.dp[c.pair.lat]);
            let (up, vp) = sip.forward(u, v);
            s.dp[c.pair.lon] = up;
            s.dp[c.pair.lat] = vp;
        }
        // Step 3: linear matrix, accumulated in place. `apply_matrix`
        // would return a fresh vector on every point.
        let m = self.linear.matrix_row_major();
        // Guaranteed at construction: `Wcs::new` and `set_linear`
        // validate the rank, and the field is private.
        // below would silently keep the previous point's values if a
        // short matrix made `chunks_exact` yield too few rows.
        debug_assert_eq!(m.len(), n * n, "linear matrix does not match NAXIS");
        let dp = &s.dp;
        for (out, row) in s.intermediate.iter_mut().zip(m.chunks_exact(n)) {
            *out = row.iter().zip(dp).map(|(a, b)| a * b).sum();
        }
        // Step 4: assemble world; celestial axes go through projection.
        // `crval` hoisted: one contiguous slice for the whole loop,
        // and the bounds check happens once instead of per axis.
        let crval = self.linear.crval();
        for ((out, cv), inter) in s.out.iter_mut().zip(crval).zip(&s.intermediate) {
            *out = cv + inter;
        }
        // Spectral axes: replace the linear value with the algorithm's
        // forward transform (Paper III Sec.3.3).
        for sx in &self.spectral {
            s.out[sx.axis] = sx.intermediate_to_world(s.intermediate[sx.axis])?;
        }
        // Tabular axes (Paper III Sec.6): the lookup replaces the
        // linear pass output with an interpolated world value.
        // The lookup operates on the full intermediate world
        // coordinate (CRVAL + linear_intermediate), which is
        // exactly `s.out[axis]` at this point.
        //
        // `check_tab_resolved` guards this loop, and every entry point
        // runs it once before reaching here. It is hoisted because an
        // unresolved `-TAB` axis is a property of the WCS, not of the
        // point, so a batch must fail the whole call rather than mark
        // each point `NaN`.
        for group in &self.tab {
            // A separable axis takes the scalar path: no per-point
            // allocation, same numbers.
            if let [axis] = group.axes[..] {
                s.out[axis] = group.forward_scalar(s.out[axis])?;
            } else {
                s.tab.clear();
                s.tab.extend(group.axes.iter().map(|&a| s.out[a]));
                for (&axis, value) in group.axes.iter().zip(group.forward(&s.tab)?) {
                    s.out[axis] = value;
                }
            }
        }
        if let Some(c) = self.celestial.as_ref() {
            // DSS plate solution: bypass the entire standard
            // celestial pipeline for the celestial axis pair.
            if let Some(dss) = self.dss.as_ref() {
                let (ra, dec) = dss.pixel_to_world(s.pix[c.pair.lon], s.pix[c.pair.lat]);
                s.out[c.pair.lon] = ra;
                s.out[c.pair.lat] = dec;
                return Ok(());
            }
            // Convert the celestial intermediate coords to degrees
            // before feeding the projection inverse, honoring any
            // non-degree CUNIT (Paper I Sec.3.1). Resolved at parse
            // time; see `CelestialBlock::cunit_to_deg`.
            let (fx, fy) = c.cunit_to_deg;
            let mut x = s.intermediate[c.pair.lon] * fx;
            let mut y = s.intermediate[c.pair.lat] * fy;
            // TPV polynomial sits between linear and projection.
            if let Some(tpv) = c.tpv.as_ref() {
                let (xp, yp) = tpv.forward(x, y);
                x = xp;
                y = yp;
            }
            // TNX/ZPX additive distortion in the same slot as TPV.
            if let Some(tnx) = c.tnx.as_ref() {
                let (xp, yp) = tnx.forward(x, y);
                x = xp;
                y = yp;
            }
            // Intermediate coordinates are zero at the reference
            // point, the projection measures from its own origin.
            // These differ only if PVi_1/PVi_2 moved the fiducial
            // point (Sec.8.2).
            let (phi, theta) = c
                .projection
                .x2s(x + c.fiducial_offset.0, y + c.fiducial_offset.1)?;
            let (alpha, delta) = c.rotation.native_to_celestial(phi, theta);
            s.out[c.pair.lon] = alpha;
            s.out[c.pair.lat] = delta;
        }
        Ok(())
    }

    /// Transform world coordinates to pixel coordinates.
    ///
    /// The result is 0-based.
    /// [`pixel_to_world`](Self::pixel_to_world) states the indexing
    /// convention.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in four cases:
    ///
    /// - `world.len()` does not equal `NAXIS`.
    /// - A `-TAB` axis remains unresolved.
    /// - The point lies outside the domain of the projection.
    /// - An iterative inverse fails to converge.
    pub fn world_to_pixel(&self, world: &[f64]) -> Result<Vec<f64>> {
        self.check_tab_resolved()?;
        let mut scratch = Scratch::new(self.naxis());
        self.world_to_pixel_into(world, &mut scratch)?;
        Ok(scratch.out)
    }

    /// Inverse transform of one point, writing into `s.out`.
    ///
    /// This holds the body of [`Self::world_to_pixel`]. It exists so a
    /// batch loop can reuse one [`Scratch`] instead of allocating a
    /// vector per point.
    ///
    /// The caller runs [`Self::check_tab_resolved`] first, as in
    /// [`Self::pixel_to_world_into`].
    ///
    /// # Errors
    ///
    /// The conditions of [`Self::world_to_pixel`] that belong to the
    /// point: a wrong length, a point the projection cannot represent,
    /// and an iterative inverse that does not converge.
    fn world_to_pixel_into(&self, world: &[f64], s: &mut Scratch) -> Result<()> {
        let n = self.naxis();
        if world.len() != n {
            return Err(FitsError::Wcs(format!(
                "expected {n} world coordinates, got {}",
                world.len()
            )));
        }
        let crval = self.linear.crval();
        for i in 0..n {
            s.intermediate[i] = world[i] - crval[i];
        }
        // Spectral axes: invert the algorithm.
        for sx in &self.spectral {
            s.intermediate[sx.axis] = sx.world_to_intermediate(world[sx.axis])?;
        }
        // Tabular axes: invert the lookup. Same all-or-nothing
        // rule as the forward pass, and the same hoisted
        // `check_tab_resolved` guarding it.
        // The lookup yields the full intermediate world coordinate;
        // subtract CRVAL to get back to the linear-pipeline space.
        for group in &self.tab {
            if let [axis] = group.axes[..] {
                s.intermediate[axis] = group.inverse_scalar(world[axis])? - crval[axis];
            } else {
                s.tab.clear();
                s.tab.extend(group.axes.iter().map(|&a| world[a]));
                for (&axis, psi) in group.axes.iter().zip(group.inverse(&s.tab)?) {
                    s.intermediate[axis] = psi - crval[axis];
                }
            }
        }
        // A DSS plate replaces the celestial pipeline, but the other
        // axes still need the linear one below. Resolve the pair now
        // and splice it in at the end.
        let dss_pixel = match (self.celestial.as_ref(), self.dss.as_ref()) {
            (Some(c), Some(dss)) => Some((
                c.pair,
                dss.world_to_pixel(world[c.pair.lon], world[c.pair.lat])?,
            )),
            _ => None,
        };
        if let Some((pair, _)) = dss_pixel {
            // The celestial slots still hold `world - CRVAL`, which is
            // meaningless here. Zero them so the inverse matrix cannot
            // mix them into the axes it is still responsible for.
            s.intermediate[pair.lon] = 0.0;
            s.intermediate[pair.lat] = 0.0;
        }
        if let Some(c) = self.celestial.as_ref()
            && dss_pixel.is_none()
        {
            let alpha = world[c.pair.lon];
            let delta = world[c.pair.lat];
            let (phi, theta) = c.rotation.celestial_to_native(alpha, delta);
            let (x_proj, y_proj) = c.projection.s2x(phi, theta)?;
            // Back to intermediate coordinates; see the forward pass.
            let (mut x, mut y) = (x_proj - c.fiducial_offset.0, y_proj - c.fiducial_offset.1);
            // Inverse TNX/ZPX (Newton on the additive surface).
            if let Some(tnx) = c.tnx.as_ref() {
                let (xp, yp) = tnx.inverse(x, y)?;
                x = xp;
                y = yp;
            }
            // Inverse TPV: undistort intermediate coords.
            if let Some(tpv) = c.tpv.as_ref() {
                let (xp, yp) = tpv.inverse(x, y)?;
                x = xp;
                y = yp;
            }
            // Convert degrees back to the header's CUNIT.
            let (fx, fy) = c.cunit_to_deg;
            s.intermediate[c.pair.lon] = x / fx;
            s.intermediate[c.pair.lat] = y / fy;
        }
        // Inverse linear matrix, accumulated in place.
        // `apply_inverse_matrix` would return a fresh vector per point.
        let inv = self.linear.inverse_row_major();
        // See the matching assertion in `pixel_to_world_into`.
        debug_assert_eq!(inv.len(), n * n, "inverse matrix does not match NAXIS");
        let intermediate = &s.intermediate;
        for (out, row) in s.dp.iter_mut().zip(inv.chunks_exact(n)) {
            *out = row.iter().zip(intermediate).map(|(a, b)| a * b).sum();
        }
        // Inverse SIP.
        if let Some(c) = self.celestial.as_ref()
            && dss_pixel.is_none()
            && let Some(sip) = c.sip.as_ref()
        {
            let (u, v) = sip.inverse(s.dp[c.pair.lon], s.dp[c.pair.lat])?;
            s.dp[c.pair.lon] = u;
            s.dp[c.pair.lat] = v;
        }
        let crpix = self.linear.crpix();
        // 1-based -> 0-based: see pixel_to_world doc.
        for ((out, cr), d) in s.out.iter_mut().zip(crpix).zip(&s.dp) {
            *out = cr + d - 1.0;
        }
        if let Some((pair, (px, py))) = dss_pixel {
            // DSS works in 1-based coords internally; the public API
            // is 0-based.
            s.out[pair.lon] = px - 1.0;
            s.out[pair.lat] = py - 1.0;
        }
        Ok(())
    }

    /// Transform many pixel coordinates to world coordinates.
    ///
    /// `pix` holds the points end to end, `NAXIS` values per point, so
    /// its length is a whole multiple of `NAXIS`. The result uses the
    /// same layout and the same length. This flat form keeps the whole
    /// batch in two allocations, one for the input and one for the
    /// result.
    ///
    /// The transform matches [`Self::pixel_to_world`] value for value.
    /// It reuses one set of working buffers across the batch, so the
    /// per-point cost carries no allocation.
    ///
    /// # Out-of-domain points
    ///
    /// A point the WCS cannot transform fills its own `NAXIS` slots
    /// with `f64::NAN` and does not fail the call. Most projections
    /// cover part of the plane alone, so a wide field routinely mixes
    /// valid and invalid pixels. Call [`Self::pixel_to_world`] on one
    /// point to read the error message for that point.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the whole batch cannot transform:
    ///
    /// - `pix.len()` is not a multiple of `NAXIS`.
    /// - `NAXIS` is zero.
    /// - A `-TAB` axis remains unresolved.
    pub fn pixel_to_world_many(&self, pix: &[f64]) -> Result<Vec<f64>> {
        if let Some(fast) = self.fast_celestial() {
            return fast.pixel_to_world_many(pix);
        }
        self.transform_many(pix, "pixel", Self::pixel_to_world_into)
    }

    /// Transform many world coordinates to pixel coordinates.
    ///
    /// This mirrors [`Self::pixel_to_world_many`], including the flat
    /// layout and the `NaN` treatment of a point that does not
    /// transform. The result is 0-based.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the whole batch cannot transform:
    ///
    /// - `world.len()` is not a multiple of `NAXIS`.
    /// - `NAXIS` is zero.
    /// - A `-TAB` axis remains unresolved.
    pub fn world_to_pixel_many(&self, world: &[f64]) -> Result<Vec<f64>> {
        if let Some(fast) = self.fast_celestial() {
            return fast.world_to_pixel_many(world);
        }
        self.transform_many(world, "world", Self::world_to_pixel_into)
    }

    /// Shared body of [`Self::pixel_to_world_many`] and
    /// [`Self::world_to_pixel_many`].
    ///
    /// `kind` names the input in an error message. `step` is the
    /// single-point transform to run for each point.
    ///
    /// # Errors
    ///
    /// The conditions of [`Self::pixel_to_world_many`].
    fn transform_many(
        &self,
        input: &[f64],
        kind: &str,
        step: fn(&Self, &[f64], &mut Scratch) -> Result<()>,
    ) -> Result<Vec<f64>> {
        let n = self.naxis();
        if n == 0 {
            return Err(FitsError::Wcs("WCS has no axes".into()));
        }
        if !input.len().is_multiple_of(n) {
            return Err(FitsError::Wcs(format!(
                "expected a whole multiple of {n} {kind} coordinates, got {}",
                input.len()
            )));
        }
        // Hoisted out of the loop so a per-point `Err` means "this
        // point is outside the projection", never "this WCS cannot
        // transform at all". The latter must fail the whole call.
        self.check_tab_resolved()?;
        let mut scratch = Scratch::new(n);
        let mut out = vec![0.0; input.len()];
        for (src, dst) in input.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
            match step(self, src, &mut scratch) {
                Ok(()) => dst.copy_from_slice(&scratch.out),
                Err(_) => dst.fill(f64::NAN),
            }
        }
        Ok(out)
    }

    /// Kind of coordinate axis `i` carries, or `None` when the WCS has
    /// no axis `i`.
    ///
    /// Use this to find an axis by meaning rather than by position.
    /// [`Self::pixel_to_world`] returns one value per axis in axis
    /// order, and this says what each of those values is.
    ///
    /// The kind comes from the type half of `CTYPEia`, so an axis
    /// driven by a `-TAB` lookup still reports its coordinate type.
    /// [`Self::is_tabular`] reports the lookup itself.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fitsy::{FitsWriter, ImageBuilder};
    /// # let path = std::env::temp_dir().join("fitsy_doc_axis_kind.fits");
    /// # let (h, d) = ImageBuilder::new(vec![4_u64, 3], vec![0.0_f32; 12])?
    /// #     .primary(true)
    /// #     .card("CTYPE1", "RA---TAN", None)
    /// #     .card("CTYPE2", "DEC--TAN", None)
    /// #     .card("CRPIX1", 1.0, None)
    /// #     .card("CRPIX2", 1.0, None)
    /// #     .card("CRVAL1", 10.0, None)
    /// #     .card("CRVAL2", 20.0, None)
    /// #     .card("CDELT1", -0.001, None)
    /// #     .card("CDELT2", 0.001, None)
    /// #     .build()?;
    /// # let mut out = std::fs::File::create(&path)?;
    /// # FitsWriter::new(&mut out).write_hdu(&h, &d)?;
    /// use fitsy::{AxisKind, FitsFile};
    ///
    /// let file = FitsFile::open(&path)?;
    /// let wcs = file.wcs(0, ' ')?.expect("HDU 0 declares a WCS");
    ///
    /// assert_eq!(wcs.axis_kind(0), Some(AxisKind::Longitude));
    /// assert_eq!(wcs.axis_kind(1), Some(AxisKind::Latitude));
    /// assert_eq!(wcs.axis_kind(2), None);
    /// # std::fs::remove_file(&path)?;
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    #[must_use]
    pub fn axis_kind(&self, i: usize) -> Option<AxisKind> {
        if i >= self.naxis() {
            return None;
        }
        if let Some(p) = self.celestial_pair {
            if p.lon == i {
                return Some(AxisKind::Longitude);
            }
            if p.lat == i {
                return Some(AxisKind::Latitude);
            }
        }
        let ctype = self.ctype(i).trim();
        // Read the type code, not the parsed algorithm state. A
        // `WAVE-TAB` axis is spectral, yet the parser files it under
        // `tab_specs` alone, because the lookup rather than a spectral
        // algorithm supplies its coordinates.
        if SpectralKind::from_code(parse::first4(ctype)).is_some() {
            return Some(AxisKind::Spectral);
        }
        if self.time.as_ref().is_some_and(|t| t.axis == i) {
            return Some(AxisKind::Time);
        }
        if self.phase.iter().any(|p| p.axis == i) {
            return Some(AxisKind::Phase);
        }
        if ctype.eq_ignore_ascii_case("STOKES") {
            return Some(AxisKind::Stokes);
        }
        Some(AxisKind::Linear)
    }

    /// Kind of every axis, in axis order.
    ///
    /// The result has one entry per axis, so it lines up with the
    /// vector [`Self::pixel_to_world`] returns. See
    /// [`Self::axis_kind`].
    #[must_use]
    pub fn axis_kinds(&self) -> Vec<AxisKind> {
        // `map`, not `filter_map`: `axis_kind` returns `None` only past
        // the end, so every index below `naxis` yields a kind and the
        // result is one entry per axis. A `filter_map` would let a
        // future `None` shorten the vector out of step with the world
        // vector it names.
        (0..self.naxis())
            .map(|i| self.axis_kind(i).unwrap_or(AxisKind::Linear))
            .collect()
    }

    /// True when axis `i` takes its coordinate from a `-TAB` lookup
    /// (Paper III Sec.6).
    ///
    /// This is a property of the algorithm, not of the coordinate, so
    /// it is independent of [`Self::axis_kind`]. A tabular axis needs
    /// its binary table loaded before it can transform; open the file
    /// with [`crate::FitsFile::wcs`], or call
    /// [`Self::resolve_tab`].
    #[must_use]
    pub fn is_tabular(&self, i: usize) -> bool {
        self.tab_specs.iter().any(|s| s.axis == i)
    }

    /// Indices of the celestial longitude / latitude axes, if any.
    /// Convenience for callers who do not want to reach into
    /// `self.celestial`.
    #[must_use]
    pub fn celestial_axes(&self) -> Option<(usize, usize)> {
        self.celestial_pair.map(|p| (p.lon, p.lat))
    }

    /// True iff this WCS has a celestial axis pair.
    #[must_use]
    pub fn is_celestial(&self) -> bool {
        self.celestial_pair.is_some()
    }

    /// World coordinates of the corner pixels of the image.
    ///
    /// The result holds `2^k` corners end to end, `NAXIS` values per
    /// corner. This is the flat layout of
    /// [`Self::pixel_to_world_many`]. `k` is the number of axes
    /// [`Self::pixel_shape`] covers, which is `NAXIS` for a normal
    /// image. Each corner is a pixel center, not the outer edge of the
    /// grid. For the outer edge, call [`Self::pixel_to_world`] with
    /// `-0.5` and `n - 0.5`.
    ///
    /// Corners come back in Gray-code order, so consecutive corners
    /// differ on one axis alone. A two-axis image therefore yields
    /// `(0, 0)`, `(nx-1, 0)`, `(nx-1, ny-1)`, `(0, ny-1)`, which walks
    /// the image counter-clockwise in pixel space and closes the ring.
    ///
    /// This reports corners, not an axis-aligned bounding box. A
    /// rotated image has corners outside the box its own minimum and
    /// maximum describe, and a celestial axis that crosses zero makes
    /// such a box meaningless.
    ///
    /// # Degenerate axes
    ///
    /// `WCSAXESa` may exceed `NAXIS` (Sec.8.2). A coordinate axis past
    /// the end of the image shape then has no length to take a corner
    /// from. That axis holds its reference pixel for every corner. The
    /// corner count follows the image, and every corner still carries a
    /// full world vector.
    ///
    /// # Out-of-domain corners
    ///
    /// This runs [`Self::pixel_to_world_many`]. A corner outside the
    /// domain of the projection therefore fills its own `NAXIS` slots
    /// with `f64::NAN` instead of failing the call. A wide-field `SIN`
    /// or `AZP` image can put every corner outside that domain. Every
    /// value then comes back `f64::NAN`. Call
    /// [`Self::pixel_to_world`] on one corner to read the reason.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when:
    ///
    /// - [`Self::pixel_shape`] is absent, which is the case for a
    ///   fitted WCS and for a header without `NAXISn` cards. The
    ///   shape is a parse-time snapshot, so a cropped image yields
    ///   the original corners.
    /// - Any axis the shape covers has length zero.
    /// - `NAXIS` exceeds [`Self::MAX_FOOTPRINT_AXES`], since the
    ///   corner count doubles with every axis.
    /// - [`Self::pixel_to_world_many`] rejects the whole batch.
    pub fn footprint(&self) -> Result<Vec<f64>> {
        let n = self.naxis();
        // Checked before anything else: this one bounds the allocation
        // the rest of the function makes. The corner count doubles per
        // covered axis. `covered` never exceeds `n`, so this bound
        // holds for a degenerate axis too.
        if n > Self::MAX_FOOTPRINT_AXES {
            return Err(FitsError::Wcs(format!(
                "footprint: {n} axes would need 2^{n} corners; the limit is {} axes",
                Self::MAX_FOOTPRINT_AXES
            )));
        }
        let shape = self.pixel_shape.as_ref().ok_or_else(|| {
            FitsError::Wcs(
                "footprint: this WCS carries no image shape (fitted, or a header \
                 without NAXISn cards)"
                    .into(),
            )
        })?;
        // A WCS axis past the end of the image shape is degenerate:
        // `WCSAXESa > NAXIS`. It has no length to take a corner from,
        // so it sits at its reference pixel instead. `CRPIX` is 1-based
        // and this API is 0-based, hence the shift.
        let covered = shape.len().min(n);
        let crpix = self.linear.crpix();
        let mut point: Vec<f64> = crpix.iter().map(|c| c - 1.0).collect();
        // The far corner on each covered axis. `NAXISn` counts pixels
        // and the API is 0-based, hence the same shift.
        let far: Vec<f64> = shape[..covered]
            .iter()
            .map(|&len| {
                if len == 0 {
                    Err(FitsError::Wcs(
                        "footprint: image has a zero-length axis".into(),
                    ))
                } else {
                    Ok((len - 1) as f64)
                }
            })
            .collect::<Result<_>>()?;
        let corners = 1_usize << covered;
        let mut pixels = Vec::with_capacity(corners * n);
        for k in 0..corners {
            // Gray code: bit `j` of `k ^ (k >> 1)` selects the near or
            // far end of axis `j`. Consecutive codes differ in one bit,
            // which is what makes the two-axis case a closed ring.
            let gray = k ^ (k >> 1);
            for (j, &f) in far.iter().enumerate() {
                point[j] = if gray & (1 << j) == 0 { 0.0 } else { f };
            }
            pixels.extend_from_slice(&point);
        }
        self.pixel_to_world_many(&pixels)
    }

    /// Local pixel scale at `(px, py)`, in arcseconds per pixel along
    /// the longitude axis and the latitude axis.
    ///
    /// This measures a finite difference on the sphere, so it includes
    /// the `cos(dec)` foreshortening, the distortion and any local
    /// skew.
    ///
    /// The result is a great-circle distance per pixel, not the signed
    /// `CDELT`. An image with a flipped RA axis therefore reports a
    /// positive scale.
    ///
    /// `px` and `py` index the longitude and latitude axes, whichever
    /// positions those hold. See [`Self::axis_kinds`].
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the WCS has no celestial axis pair, and
    /// the conditions of [`Self::pixel_to_world`] evaluated at
    /// `(px, py)` and at the two neighboring pixels.
    pub fn pixel_scale_at(&self, px: f64, py: f64) -> Result<(f64, f64)> {
        let (lon, lat) = self
            .celestial_axes()
            .ok_or_else(|| FitsError::Wcs("pixel_scale_at: WCS has no celestial pair".into()))?;
        // Any axis outside the pair holds its reference pixel: the
        // scale describes the celestial pair, measured where the rest
        // of the WCS sits by default. `CRPIX` is 1-based and this API
        // is 0-based, hence the shift. Built once and reused for all
        // three points.
        let mut point: Vec<f64> = self.linear.crpix().iter().map(|c| c - 1.0).collect();
        let mut sky = |x: f64, y: f64| -> Result<(f64, f64)> {
            point[lon] = x;
            point[lat] = y;
            let w = self.pixel_to_world(&point)?;
            Ok((w[lon], w[lat]))
        };
        let (ra0, dec0) = sky(px, py)?;
        let (ra_x, dec_x) = sky(px + 1.0, py)?;
        let (ra_y, dec_y) = sky(px, py + 1.0)?;
        let dx_arcsec = great_circle_arcsec(ra0, dec0, ra_x, dec_x);
        let dy_arcsec = great_circle_arcsec(ra0, dec0, ra_y, dec_y);
        Ok((dx_arcsec, dy_arcsec))
    }

    /// How many `-TAB` axes the resolved groups cover. One group
    /// drives every axis that shares its coordinate array, so this is
    /// not the group count.
    fn resolved_tab_axes(&self) -> usize {
        self.tab.iter().map(|g| g.axes.len()).sum()
    }

    /// Reject a WCS whose `-TAB` axes parsed but never loaded.
    ///
    /// Returning the linear approximation instead would be a wrong
    /// answer that looks like a right one: the per-point bodies walk
    /// `self.tab`, so an unresolved axis leaves the lookup out
    /// silently.
    ///
    /// Every entry point runs this once per call, before the
    /// per-point body: it describes the WCS rather than the point, so
    /// a batch must fail the whole call rather than mark each point
    /// `NaN`. The fast path has no check of its own;
    /// [`Self::fast_celestial`] declines a WCS that carries any
    /// `-TAB` spec.
    fn check_tab_resolved(&self) -> Result<()> {
        let resolved = self.resolved_tab_axes();
        if self.tab_specs.len() == resolved {
            return Ok(());
        }
        Err(FitsError::Wcs(format!(
            "WCS has {} unresolved -TAB axis spec(s); \
             call FitsFile::wcs() or Wcs::resolve_tab() to load them",
            // `saturating_sub`: `resolve_tab` never leaves `tab` longer
            // than `tab_specs`, but both are public fields and a caller
            // that pushed into `tab` directly should land on this
            // error, not an overflow panic.
            self.tab_specs.len().saturating_sub(resolved),
        )))
    }

    /// Resolve every parsed `-TAB` axis against the binary tables in
    /// `file`.
    ///
    /// The result is the number of axes resolved. A second call after
    /// a successful one changes nothing and returns 0.
    ///
    /// Most callers do not call this. [`crate::FitsFile::wcs`]
    /// resolves the axes on their behalf.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in four cases:
    ///
    /// - The extension a `TabSpec` names is absent from `file`.
    /// - That extension is not a binary table.
    /// - The named column is absent.
    /// - The coordinate array has a shape the axis count does not
    ///   match.
    pub fn resolve_tab(&mut self, file: &crate::FitsFile) -> Result<usize> {
        if self.tab_specs.len() == self.resolved_tab_axes() {
            return Ok(0);
        }
        // Sec.6.2: axes naming the same coordinate array in the same
        // extension are one non-separable group and must be looked up
        // together. Grouping keeps their first-seen order stable.
        let mut order: Vec<(String, i64, i64, String)> = Vec::new();
        let mut groups: Vec<Vec<&TabSpec>> = Vec::new();
        for spec in &self.tab_specs {
            let key = spec.group_key();
            if let Some(i) = order.iter().position(|k| *k == key) {
                groups[i].push(spec);
            } else {
                order.push(key);
                groups.push(vec![spec]);
            }
        }
        let mut resolved = Vec::with_capacity(groups.len());
        for group in &groups {
            resolved.push(load_tab_group(file, group)?);
        }
        self.tab = resolved;
        Ok(self.resolved_tab_axes())
    }
}

/// Load one `-TAB` group -- every WCS axis sharing a coordinate array
/// -- from its referenced binary table (Paper III Sec.6.2).
fn load_tab_group(file: &crate::FitsFile, specs: &[&TabSpec]) -> Result<TabGroup> {
    let first = specs[0];
    let hdu = file.hdu_by_name(&first.extname, Some(first.extver))?;
    let crate::Hdu::BinTable(bin) = hdu else {
        return Err(FitsError::Wcs(format!(
            "-TAB axis {}: extension `{}` (EXTVER {}) is not a BINTABLE",
            first.axis + 1,
            first.extname,
            first.extver,
        )));
    };
    // Sec.6.2 identifies the table by the EXTNAME/EXTVER/EXTLEVEL
    // triple; `hdu_by_name` matches the first two, so confirm the third.
    let found_level = match bin.header().first("EXTLEVEL") {
        Some(crate::header::value::Value::Integer(v)) => v,
        _ => 1,
    };
    if found_level != first.extlevel {
        return Err(FitsError::Wcs(format!(
            "-TAB axis {}: extension `{}` (EXTVER {}) has EXTLEVEL {found_level}, but \
             PV{}_2 asks for {}",
            first.axis + 1,
            first.extname,
            first.extver,
            first.axis + 1,
            first.extlevel,
        )));
    }
    if bin.n_rows() != 1 {
        return Err(FitsError::Wcs(format!(
            "-TAB: BINTABLE `{}` has {} rows; Sec.6.2 requires exactly one",
            first.extname,
            bin.n_rows(),
        )));
    }

    let (coord, tdim) = read_tab_column(&bin, &first.coord_column, first)?;
    // `TDIM` is `(M, K_1, ..., K_M)`, fastest axis first. Without it
    // the array can only be the one-axis case, where the repeat count
    // is K and the leading degenerate axis is implied.
    let dims: Vec<usize> = match tdim {
        Some(t) if t.len() >= 2 => t[1..].to_vec(),
        _ => vec![coord.len()],
    };
    let rank = dims.len();
    if specs.len() != rank {
        return Err(FitsError::Wcs(format!(
            "-TAB: coordinate array `{}` describes {rank} axes, but {} WCS axes \
             reference it; Sec.6.2 requires the PVi_3 values to account for all of them",
            first.coord_column,
            specs.len(),
        )));
    }

    // Place each axis in the slot its `PVi_3` names.
    let mut axes = vec![usize::MAX; rank];
    let mut index: Vec<Option<Vec<f64>>> = vec![None; rank];
    for spec in specs {
        let m = spec.coord_axis as usize;
        if m < 1 || m > rank {
            return Err(FitsError::Wcs(format!(
                "-TAB axis {}: PV{}_3 = {m} is outside 1..={rank}",
                spec.axis + 1,
                spec.axis + 1,
            )));
        }
        if axes[m - 1] != usize::MAX {
            return Err(FitsError::Wcs(format!(
                "-TAB: axes {} and {} both claim coordinate-array axis {m}",
                axes[m - 1] + 1,
                spec.axis + 1,
            )));
        }
        axes[m - 1] = spec.axis;
        if let Some(name) = &spec.index_column {
            index[m - 1] = Some(read_tab_column(&bin, name, spec)?.0);
        }
    }

    let group = TabGroup {
        axes,
        dims,
        index,
        coord,
    };
    group.validate()?;
    Ok(group)
}

/// Read a 1-D float column from a single-row BINTABLE, with its
/// `TDIMn` if present.
fn read_tab_column(
    bin: &crate::hdu::bintable::BinTableHdu<'_>,
    name: &str,
    spec: &TabSpec,
) -> Result<(Vec<f64>, Option<Vec<usize>>)> {
    use crate::hdu::bintable::BinValue;
    let col = bin.column_by_name(name).ok_or_else(|| {
        FitsError::Wcs(format!(
            "-TAB axis {}: BINTABLE `{}` has no column `{name}`",
            spec.axis + 1,
            spec.extname,
        ))
    })?;
    let raw = bin.cell_value(0, col)?;
    let v = match raw {
        BinValue::F64(v) | BinValue::Float(v) => v,
        BinValue::F32(v) => v.into_iter().map(f64::from).collect(),
        BinValue::Int(v) => v
            .into_iter()
            .map(|o| o.map_or(f64::NAN, |i| i as f64))
            .collect(),
        other => {
            return Err(FitsError::Wcs(format!(
                "-TAB axis {}: column `{name}` has unsupported type {other:?}",
                spec.axis + 1,
            )));
        }
    };
    Ok((v, col.tdim.clone()))
}

/// Great-circle separation between two (RA, Dec) points in degrees,
/// returned in arcseconds. Uses the Vincenty form so it stays
/// well-conditioned for both small and antipodal separations.
fn great_circle_arcsec(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let d2r = std::f64::consts::PI / 180.0;
    let (s1, c1) = (dec1 * d2r).sin_cos();
    let (s2, c2) = (dec2 * d2r).sin_cos();
    let dra = (ra2 - ra1) * d2r;
    let (sd, cd) = dra.sin_cos();
    let num = ((c2 * sd).powi(2) + (c1 * s2 - s1 * c2 * cd).powi(2)).sqrt();
    let den = s1 * s2 + c1 * c2 * cd;
    num.atan2(den) / d2r * 3600.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both construction paths pass through [`Wcs::new`]. It is the
    /// one place that can hold `celestial` and `celestial_pair` to
    /// agreement. A disagreement would send the classification
    /// queries and the transform pipeline to different axes.
    #[test]
    fn celestial_block_and_pair_must_agree() {
        let pair = CelestialPair {
            lon: 0,
            lat: 1,
            frame: CelestialFrame::Equatorial,
        };
        let block = || CelestialBlock {
            pair,
            projection: Projection::default(),
            rotation: celestial::CelestialRotation::new(150.0, 30.0, None, None, 0.0, 90.0)
                .unwrap(),
            sip: None,
            tpv: None,
            tnx: None,
            cunit_to_deg: (1.0, 1.0),
            fiducial_offset: (0.0, 0.0),
        };
        let linear = || {
            LinearTransform::from_pc(
                vec![1.0, 1.0],
                vec![150.0, 30.0],
                vec![1.0, 1.0],
                vec![1.0, 0.0, 0.0, 1.0],
            )
            .unwrap()
        };
        let axes = || vec![Axis::default(), Axis::default()];

        // Agreeing pair: accepted.
        let parts = WcsParts {
            celestial: Some(block()),
            celestial_pair: Some(pair),
            ..WcsParts::default()
        };
        assert!(Wcs::new(axes(), linear(), parts).is_ok());

        // Swapped axes on the outer pair: rejected.
        let parts = WcsParts {
            celestial: Some(block()),
            celestial_pair: Some(CelestialPair {
                lon: 1,
                lat: 0,
                frame: CelestialFrame::Equatorial,
            }),
            ..WcsParts::default()
        };
        let err = Wcs::new(axes(), linear(), parts).unwrap_err();
        assert!(err.to_string().contains("disagrees"), "got: {err}");

        // A frame-only disagreement is a disagreement too.
        let parts = WcsParts {
            celestial: Some(block()),
            celestial_pair: Some(CelestialPair {
                lon: 0,
                lat: 1,
                frame: CelestialFrame::Galactic,
            }),
            ..WcsParts::default()
        };
        let err = Wcs::new(axes(), linear(), parts).unwrap_err();
        assert!(err.to_string().contains("disagrees"), "got: {err}");

        // A block with no pair at all: rejected.
        let parts = WcsParts {
            celestial: Some(block()),
            celestial_pair: None,
            ..WcsParts::default()
        };
        let err = Wcs::new(axes(), linear(), parts).unwrap_err();
        assert!(err.to_string().contains("no celestial pair"), "got: {err}");
    }
}
