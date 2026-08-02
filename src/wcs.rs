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
//! - [`projection`] and [`projections`] -- the celestial projections
//!   of Paper II Table 13, selected by the `CTYPE` code.
//! - [`celestial`] -- the spherical rotation and the reference frame.
//! - [`spectral`] -- the spectral `CTYPE` codes and the non-linear
//!   algorithms of Paper III Sec.3.3 (Greisen et al. 2006).
//! - [`tab`] -- the `-TAB` lookup axes of Paper III Sec.6.
//! - [`time`] -- the time axis of Sec.9.5.3.
//! - [`sip`], [`tpv`], [`tnx`], [`dss`] -- distortion conventions that
//!   sit outside the `PVi_m` family.
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
pub mod dss;
pub mod linear;
pub mod projection;
pub mod sip;
pub mod spectral;
pub mod tab;
pub mod table;
pub mod time;
pub mod tnx;
pub mod tpv;

pub mod projections;
pub(crate) mod wat;

mod fit;
mod parse;
mod serialize;

pub use celestial::{CelestialFrame, RadeSys};
pub use celestial_block::{CelestialBlock, CelestialPair};
pub use dss::Dss;
pub use fit::{WcsFit, WcsFitOptions, fit_celestial_wcs};
pub use linear::LinearTransform;
pub use projection::{Projection, ProjectionKind};
// `Linearised` and `Grism` are reachable from `SpectralAlgorithm`'s
// variants, so a caller matching on one needs them at the same level.
pub use spectral::{
    Grism, Linearised, SourceFrame, SpectralAlgorithm, SpectralAxis, SpectralFrame, SpectralKind,
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
/// Sec.8.2 restricts the code to a space or `A`-`Z`; anything else is
/// an error.
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
/// use fitsy::{Header, Wcs};
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
/// let (ra, dec) = wcs.pixel_to_celestial(31.0, 23.0)?;
/// assert!((ra - 150.0).abs() < 1e-9);
/// assert!((dec - 2.5).abs() < 1e-9);
/// # Ok::<(), fitsy::FitsError>(())
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Wcs {
    // Private so `naxis()` can be its length: public parallel vectors
    // could be truncated out of step and the pipeline indexed them
    // unchecked. Read via `axes()`, `axis()`, `ctype()`, `cunit()`.
    axes: Vec<Axis>,
    /// Linear stage of the pipeline: `CRPIX`, `CRVAL`, and the combined
    /// `CDELT`/`PC` or `CD` matrix.
    pub linear: LinearTransform,
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
#[non_exhaustive]
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

impl Wcs {
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

    /// Assemble a `Wcs`, rejecting pieces that disagree on the axis
    /// count or carry an out-of-range axis index.
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
    /// [`pixel_to_celestial`], [`celestial_to_pixel`],
    /// [`pixel_to_celestial_many`], [`celestial_to_pixel_many`] and
    /// [`pixel_scale_at`].
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
    /// [`pixel_to_celestial`]: Self::pixel_to_celestial
    /// [`celestial_to_pixel`]: Self::celestial_to_pixel
    /// [`pixel_to_celestial_many`]: Self::pixel_to_celestial_many
    /// [`celestial_to_pixel_many`]: Self::celestial_to_pixel_many
    /// [`pixel_scale_at`]: Self::pixel_scale_at
    pub fn pixel_to_world(&self, pix: &[f64]) -> Result<Vec<f64>> {
        if pix.len() != self.naxis() {
            return Err(FitsError::Wcs(format!(
                "expected {} pixel coordinates, got {}",
                self.naxis(),
                pix.len()
            )));
        }
        // 0-based -> 1-based: see the doc comment above.
        let pix: Vec<f64> = pix.iter().map(|p| p + 1.0).collect();
        let pix = pix.as_slice();
        // Step 1: pixel offset relative to CRPIX.
        let crpix = self.linear.crpix();
        let mut dp: Vec<f64> = (0..self.naxis()).map(|j| pix[j] - crpix[j]).collect();
        // Step 2: SIP pixel-space distortion (celestial pair only).
        if let Some(c) = self.celestial.as_ref()
            && let Some(sip) = c.sip.as_ref()
        {
            let (u, v) = (dp[c.pair.lon], dp[c.pair.lat]);
            let (up, vp) = sip.forward(u, v);
            dp[c.pair.lon] = up;
            dp[c.pair.lat] = vp;
        }
        // Step 3: linear matrix.
        let intermediate = self.linear.apply_matrix(&dp)?;
        // Step 4: assemble world; celestial axes go through projection.
        // `crval` hoisted: one contiguous slice for the whole loop,
        // and the bounds check happens once instead of per axis.
        let crval = self.linear.crval();
        let mut world: Vec<f64> = (0..self.naxis())
            .map(|i| crval[i] + intermediate[i])
            .collect();
        // Spectral axes: replace the linear value with the algorithm's
        // forward transform (Paper III Sec.3.3).
        for sx in &self.spectral {
            world[sx.axis] = sx.intermediate_to_world(intermediate[sx.axis])?;
        }
        // Tabular axes (Paper III Sec.6): the lookup replaces the
        // linear pass output with an interpolated world value.
        // The lookup operates on the full intermediate world
        // coordinate (CRVAL + linear_intermediate), which is
        // exactly `world[axis]` at this point.
        self.check_tab_resolved()?;
        for group in &self.tab {
            // A separable axis takes the scalar path: no per-point
            // allocation, same numbers.
            if let [axis] = group.axes[..] {
                world[axis] = group.forward_scalar(world[axis])?;
            } else {
                let psi: Vec<f64> = group.axes.iter().map(|&a| world[a]).collect();
                for (&axis, value) in group.axes.iter().zip(group.forward(&psi)?) {
                    world[axis] = value;
                }
            }
        }
        if let Some(c) = self.celestial.as_ref() {
            // DSS plate solution: bypass the entire standard
            // celestial pipeline for the celestial axis pair.
            if let Some(dss) = self.dss.as_ref() {
                let (ra, dec) = dss.pixel_to_world(pix[c.pair.lon], pix[c.pair.lat]);
                world[c.pair.lon] = ra;
                world[c.pair.lat] = dec;
                return Ok(world);
            }
            // Convert the celestial intermediate coords to degrees
            // before feeding the projection inverse, honoring any
            // non-degree CUNIT (Paper I Sec.3.1). Resolved at parse
            // time; see `CelestialBlock::cunit_to_deg`.
            let (fx, fy) = c.cunit_to_deg;
            let mut x = intermediate[c.pair.lon] * fx;
            let mut y = intermediate[c.pair.lat] * fy;
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
            world[c.pair.lon] = alpha;
            world[c.pair.lat] = delta;
        }
        Ok(world)
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
        if world.len() != self.naxis() {
            return Err(FitsError::Wcs(format!(
                "expected {} world coordinates, got {}",
                self.naxis(),
                world.len()
            )));
        }
        let crval = self.linear.crval();
        let mut intermediate: Vec<f64> = (0..self.naxis()).map(|i| world[i] - crval[i]).collect();
        // Spectral axes: invert the algorithm.
        for sx in &self.spectral {
            intermediate[sx.axis] = sx.world_to_intermediate(world[sx.axis])?;
        }
        // Tabular axes: invert the lookup. Same all-or-nothing
        // rule as the forward pass. The lookup yields the full
        // intermediate world coordinate; subtract CRVAL to get
        // back to the linear-pipeline space.
        self.check_tab_resolved()?;
        for group in &self.tab {
            if let [axis] = group.axes[..] {
                intermediate[axis] = group.inverse_scalar(world[axis])? - crval[axis];
            } else {
                let target: Vec<f64> = group.axes.iter().map(|&a| world[a]).collect();
                for (&axis, psi) in group.axes.iter().zip(group.inverse(&target)?) {
                    intermediate[axis] = psi - crval[axis];
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
            intermediate[pair.lon] = 0.0;
            intermediate[pair.lat] = 0.0;
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
            intermediate[c.pair.lon] = x / fx;
            intermediate[c.pair.lat] = y / fy;
        }
        // Inverse linear matrix.
        let mut dp = self.linear.apply_inverse_matrix(&intermediate)?;
        // Inverse SIP.
        if let Some(c) = self.celestial.as_ref()
            && dss_pixel.is_none()
            && let Some(sip) = c.sip.as_ref()
        {
            let (u, v) = sip.inverse(dp[c.pair.lon], dp[c.pair.lat])?;
            dp[c.pair.lon] = u;
            dp[c.pair.lat] = v;
        }
        let crpix = self.linear.crpix();
        // 1-based -> 0-based: see pixel_to_world doc.
        let mut out: Vec<f64> = (0..self.naxis()).map(|i| crpix[i] + dp[i] - 1.0).collect();
        if let Some((pair, (px, py))) = dss_pixel {
            // DSS works in 1-based coords internally; the public API
            // is 0-based.
            out[pair.lon] = px - 1.0;
            out[pair.lat] = py - 1.0;
        }
        Ok(out)
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

    /// Sky positions of the image's four corner pixels, as
    /// `(lon, lat)` in degrees.
    ///
    /// These are corner pixel centers: `(0, 0)`, `(nx-1, 0)`,
    /// `(nx-1, ny-1)` and `(0, ny-1)`, counter-clockwise from the
    /// origin. For the outer edge of the grid instead, call
    /// [`Self::pixel_to_celestial`] with `-0.5` and `n - 0.5`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] if the WCS has no celestial pair, or if
    /// [`Self::pixel_shape`] is absent or does not cover both
    /// celestial axes. The shape is a parse-time snapshot, so a
    /// cropped image yields the original corners.
    pub fn footprint(&self) -> Result<[(f64, f64); 4]> {
        let (lon, lat) = self
            .celestial_axes()
            .ok_or_else(|| FitsError::Wcs("footprint: WCS has no celestial axis pair".into()))?;
        let shape = self.pixel_shape.as_ref().ok_or_else(|| {
            FitsError::Wcs(
                "footprint: this WCS carries no image shape (fitted, or a header \
                 without NAXISn cards)"
                    .into(),
            )
        })?;
        let (Some(&nx), Some(&ny)) = (shape.get(lon), shape.get(lat)) else {
            return Err(FitsError::Wcs(format!(
                "footprint: image shape has {} axes, which does not cover the \
                 celestial pair (axes {lon} and {lat})",
                shape.len()
            )));
        };
        if nx == 0 || ny == 0 {
            return Err(FitsError::Wcs(
                "footprint: image has a zero-length celestial axis".into(),
            ));
        }
        let (x1, y1) = ((nx - 1) as f64, (ny - 1) as f64);
        Ok([
            self.pixel_to_celestial(0.0, 0.0)?,
            self.pixel_to_celestial(x1, 0.0)?,
            self.pixel_to_celestial(x1, y1)?,
            self.pixel_to_celestial(0.0, y1)?,
        ])
    }

    /// Check what applies to a whole batch, so the batch helpers can
    /// tell "this WCS cannot transform at all" (an `Err`) from "this
    /// point is outside the projection" (a `NaN` slot). These are the
    /// only two: every other failure is per-point.
    fn batch_precheck(&self) -> Result<()> {
        if self.celestial_pair.is_none() {
            return Err(FitsError::Wcs("WCS has no celestial axis pair".into()));
        }
        self.check_tab_resolved()
    }

    /// Transform many pixel pairs to (RA, Dec) pairs.
    ///
    /// This runs the transform of [`Self::pixel_to_celestial`] and
    /// shares the per-call setup across every point.
    ///
    /// # Out-of-domain points
    ///
    /// A point outside the domain of the projection yields
    /// `(f64::NAN, f64::NAN)`. It does not fail the call. Most
    /// projections cover part of the plane alone, so a wide field
    /// routinely mixes valid and invalid pixels. Call
    /// [`Self::pixel_to_celestial`] on one point to read the error
    /// message for that point.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the whole WCS cannot transform, which
    /// means it has no celestial axis pair, or it has a `-TAB` axis
    /// that remains unresolved.
    pub fn pixel_to_celestial_many(&self, pixels: &[(f64, f64)]) -> Result<Vec<(f64, f64)>> {
        self.batch_precheck()?;
        Ok(pixels
            .iter()
            .map(|&(px, py)| {
                self.pixel_to_celestial(px, py)
                    .unwrap_or((f64::NAN, f64::NAN))
            })
            .collect())
    }

    /// Transform many (RA, Dec) pairs to pixel pairs.
    ///
    /// This mirrors [`Self::pixel_to_celestial_many`], including its
    /// `NaN` treatment of an out-of-domain point.
    ///
    /// # Errors
    ///
    /// The same conditions as [`Self::pixel_to_celestial_many`].
    pub fn celestial_to_pixel_many(&self, sky: &[(f64, f64)]) -> Result<Vec<(f64, f64)>> {
        self.batch_precheck()?;
        Ok(sky
            .iter()
            .map(|&(ra, dec)| {
                self.celestial_to_pixel(ra, dec)
                    .unwrap_or((f64::NAN, f64::NAN))
            })
            .collect())
    }

    /// Transform one celestial pixel pair to (RA, Dec) in degrees.
    ///
    /// This serves the common case of a two-axis image. Any further
    /// axis evaluates at its reference pixel, which is the only
    /// defined choice when the caller supplies no value for it.
    ///
    /// The `px` and `py` arguments are 0-based. See
    /// [`pixel_to_world`](Self::pixel_to_world).
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the WCS has no celestial axis pair, or
    /// when the conditions of
    /// [`pixel_to_world`](Self::pixel_to_world) apply.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fitsy::{FitsWriter, ImageBuilder};
    /// # let path = std::env::temp_dir().join("fitsy_doc_p2c.fits");
    /// # let px = vec![0.0_f32; 64 * 48];
    /// # let (h, d) = ImageBuilder::new(vec![64_u64, 48], px)?
    /// #     .primary(true)
    /// #     .card("CTYPE1", "RA---TAN", None)
    /// #     .card("CTYPE2", "DEC--TAN", None)
    /// #     .card("CRPIX1", 32.0, None)
    /// #     .card("CRPIX2", 24.0, None)
    /// #     .card("CRVAL1", 202.469, None)
    /// #     .card("CRVAL2", 47.195, None)
    /// #     .card("CDELT1", -0.001, None)
    /// #     .card("CDELT2", 0.001, None)
    /// #     .build()?;
    /// # let mut out = std::fs::File::create(&path)?;
    /// # FitsWriter::new(&mut out).write_hdu(&h, &d)?;
    /// use fitsy::FitsFile;
    ///
    /// let f = FitsFile::open(&path)?;
    /// let wcs = f.wcs(0, ' ')?.expect("HDU 0 declares no WCS");
    /// let (ra, dec) = wcs.pixel_to_celestial(31.0, 23.0)?;
    ///
    /// assert!((ra - 202.469).abs() < 1e-9);
    /// assert!((dec - 47.195).abs() < 1e-9);
    /// # std::fs::remove_file(&path)?;
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    pub fn pixel_to_celestial(&self, px: f64, py: f64) -> Result<(f64, f64)> {
        let (lon, lat) = self
            .celestial_axes()
            .ok_or_else(|| FitsError::Wcs("WCS has no celestial axis pair".into()))?;
        let crpix = self.linear.crpix();
        // CRPIX is 1-based per FITS; this API is 0-based, so the
        // "sit at reference pixel" filler for the non-celestial axes
        // is `crpix - 1`.
        let mut pix: Vec<f64> = crpix.iter().map(|c| c - 1.0).collect();
        pix[lon] = px;
        pix[lat] = py;
        let world = self.pixel_to_world(&pix)?;
        Ok((world[lon], world[lat]))
    }

    /// Transform one (RA, Dec) pair in degrees to a celestial pixel
    /// pair. This mirrors [`Self::pixel_to_celestial`].
    ///
    /// The returned `(px, py)` is 0-based. Any further axis is held at
    /// its `CRVAL`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the WCS has no celestial axis pair, or
    /// when the conditions of
    /// [`world_to_pixel`](Self::world_to_pixel) apply.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fitsy::{FitsWriter, ImageBuilder};
    /// # let path = std::env::temp_dir().join("fitsy_doc_c2p.fits");
    /// # let px = vec![0.0_f32; 64 * 48];
    /// # let (h, d) = ImageBuilder::new(vec![64_u64, 48], px)?
    /// #     .primary(true)
    /// #     .card("CTYPE1", "RA---TAN", None)
    /// #     .card("CTYPE2", "DEC--TAN", None)
    /// #     .card("CRPIX1", 32.0, None)
    /// #     .card("CRPIX2", 24.0, None)
    /// #     .card("CRVAL1", 202.469, None)
    /// #     .card("CRVAL2", 47.195, None)
    /// #     .card("CDELT1", -0.001, None)
    /// #     .card("CDELT2", 0.001, None)
    /// #     .build()?;
    /// # let mut out = std::fs::File::create(&path)?;
    /// # FitsWriter::new(&mut out).write_hdu(&h, &d)?;
    /// use fitsy::FitsFile;
    ///
    /// let f = FitsFile::open(&path)?;
    /// let wcs = f.wcs(0, ' ')?.expect("HDU 0 declares no WCS");
    /// let (px, py) = wcs.celestial_to_pixel(202.469, 47.195)?;
    ///
    /// assert!((px - 31.0).abs() < 1e-6);
    /// assert!((py - 23.0).abs() < 1e-6);
    /// # std::fs::remove_file(&path)?;
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    pub fn celestial_to_pixel(&self, ra: f64, dec: f64) -> Result<(f64, f64)> {
        let (lon, lat) = self
            .celestial_axes()
            .ok_or_else(|| FitsError::Wcs("WCS has no celestial axis pair".into()))?;
        // Build a world vector with the celestial pair set and the
        // other axes at CRVAL (zero, since CRVAL is absorbed into the
        // celestial rotation and the spectral algorithms).
        let mut world = self.linear.crval().to_vec();
        world[lon] = ra;
        world[lat] = dec;
        let pix = self.world_to_pixel(&world)?;
        Ok((pix[lon], pix[lat]))
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
    /// # Errors
    ///
    /// The conditions of [`Self::pixel_to_celestial`], evaluated at
    /// `(px, py)` and at the two neighboring pixels.
    pub fn pixel_scale_at(&self, px: f64, py: f64) -> Result<(f64, f64)> {
        let (ra0, dec0) = self.pixel_to_celestial(px, py)?;
        let (ra_x, dec_x) = self.pixel_to_celestial(px + 1.0, py)?;
        let (ra_y, dec_y) = self.pixel_to_celestial(px, py + 1.0)?;
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
    /// answer that looks like a right one. Both transform directions
    /// therefore check this before they reach the lookup.
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
        Some(crate::header::value::Value::Integer(v)) => *v,
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
