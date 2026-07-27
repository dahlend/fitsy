//! World Coordinate System (Standard Sec.8, Greisen & Calabretta 2002,
//! Calabretta & Greisen 2002).
//!
//! This module implements the four-step WCS pipeline (Paper I Sec.8.1)
//! and every celestial projection in Paper II Table 13, plus the
//! spectral CTYPE codes and non-linear regridding algorithms of
//! Paper III Sec.3.3 (Greisen et al. 2006). SIP and TPV distortion
//! conventions are supported in addition to the standard `PVi_m`
//! parameter family.
//!
//! The typical entry point is [`Wcs::from_header`], which parses the
//! WCS from a [`Header`](crate::Header). For files with tabular (`-TAB`)
//! axes, use [`FitsFile::wcs`](crate::FitsFile::wcs) instead, which
//! resolves the table data automatically.
//!
//! Time WCS (Sec.9) and the grism algorithms `-GRI`/`-GRA` are not
//! supported. Tabular WCS (`-TAB`) is supported for the common
//! single-axis 1-D case via [`tab::TabAxis`].

pub mod celestial;
pub mod celestial_block;
pub mod dss;
pub mod linear;
pub mod projection;
pub mod sip;
pub mod spectral;
pub mod tab;
pub mod tnx;
pub mod tpv;

pub mod projections;
pub(crate) mod units;
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
pub use spectral::{SpectralAlgorithm, SpectralAxis, SpectralKind};
pub use tab::{TabAxis, TabSpec};

use crate::error::{FitsError, Result};
use crate::wcs::units::to_degrees_factor;

/// Degrees -> radians.
pub(crate) const D2R: f64 = std::f64::consts::PI / 180.0;
/// Radians -> degrees.
pub(crate) const R2D: f64 = 180.0 / std::f64::consts::PI;

/// A parsed WCS for a single alternate (`' '`, `'A'`, ..., `'Z'`).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Wcs {
    pub naxis: usize,
    pub linear: LinearTransform,
    /// Per-axis CTYPE values, uppercase.
    pub ctype: Vec<String>,
    /// Per-axis CUNIT values; empty string if not given.
    pub cunit: Vec<String>,
    /// Per-axis CRVAL reference value, in `cunit[i]` units. Always the
    /// true as-parsed/as-fit value, including on the celestial pair
    /// (also mirrored into `celestial`'s rotation) and spectral axes
    /// (also mirrored into the matching `SpectralAxis::crval_si`) --
    /// the pixel<->world pipeline recomputes those axes from the
    /// rotation/spectral algorithm rather than from this field.
    pub crval: Vec<f64>,
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
    /// `SPECSYS` keyword (Paper III Sec.7) -- spectral reference frame
    /// in which the spectral coordinates are expressed (`TOPOCENT`,
    /// `BARYCENT`, `LSRK`, ...). Stored verbatim; not applied.
    pub specsys: Option<String>,
    /// `SSYSOBS` keyword (Paper III Sec.7) -- spectral reference frame
    /// of the observation. Stored verbatim; not applied.
    pub ssysobs: Option<String>,
    /// `VELOSYS` keyword (Paper III Sec.7) -- relative radial velocity
    /// (m/s) between the observer and `SSYSOBS`. Stored verbatim;
    /// not applied.
    pub velosys: Option<f64>,
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
    /// Resolved `-TAB` axes (populated by [`Self::resolve_tab`] or
    /// by [`crate::FitsFile::wcs`]). When a `-TAB` axis is parsed
    /// but unresolved, `pixel_to_world` / `world_to_pixel` return a
    /// clear error rather than silently dropping the lookup.
    pub tab: Vec<TabAxis>,
    /// Size of the image this WCS was read from, in FITS axis order
    /// (`NAXIS1` first) -- a **snapshot of the `NAXISn` cards**, not
    /// part of the coordinate description.
    ///
    /// `None` when there was no image to read it from, which
    /// includes every WCS produced by
    /// [`fit_celestial_wcs`] and any
    /// header without `NAXISn` cards. Its length is the image's
    /// `NAXIS`, which can differ from [`Self::naxis`] when `WCSAXES`
    /// declares a different number of coordinate axes.
    ///
    /// Nothing in the pixel<->world pipeline reads this field, and
    /// nothing validates against it: the image it describes may have
    /// been cropped or rebinned since. It exists so callers that do
    /// need the extent -- [`Self::footprint`] and the like -- can get
    /// it without threading the shape through every call. Treat a
    /// stale value as the caller's problem, the same way the writer
    /// treats layout cards on a user-supplied header.
    pub pixel_shape: Option<Vec<u64>>,
}

impl Wcs {
    /// Pixel -> world coordinates.
    ///
    /// # Pixel indexing convention
    ///
    /// **Pixel coordinates in this API are 0-based** (numpy / C / row-major
    /// convention): the center of the first pixel is `(0.0, 0.0, ...)`.
    /// The underlying FITS standard (Sec.3.3.4) defines pixels as 1-based,
    /// so this method internally adds `1.0` to every input before
    /// evaluating the WCS pipeline. If you are porting from a FITS-native
    /// tool (cfitsio, wcslib, IRAF) that expects 1-based pixels, subtract
    /// 1 from those coordinates before calling. Astropy's `wcs.WCS` uses
    /// the same 0-based default (its `origin=0` argument).
    ///
    /// Applies to every pixel-coordinate method on `Wcs`: [`pixel_to_world`],
    /// [`world_to_pixel`], [`pixel_to_celestial`], [`celestial_to_pixel`],
    /// [`pixel_to_celestial_many`], [`celestial_to_pixel_many`], and
    /// [`pixel_scale_at`].
    ///
    /// World values are returned in their CUNIT (degrees for celestial
    /// axes).
    ///
    /// [`pixel_to_world`]: Self::pixel_to_world
    /// [`world_to_pixel`]: Self::world_to_pixel
    /// [`pixel_to_celestial`]: Self::pixel_to_celestial
    /// [`celestial_to_pixel`]: Self::celestial_to_pixel
    /// [`pixel_to_celestial_many`]: Self::pixel_to_celestial_many
    /// [`celestial_to_pixel_many`]: Self::celestial_to_pixel_many
    /// [`pixel_scale_at`]: Self::pixel_scale_at
    pub fn pixel_to_world(&self, pix: &[f64]) -> Result<Vec<f64>> {
        if pix.len() != self.naxis {
            return Err(FitsError::Wcs(format!(
                "expected {} pixel coordinates, got {}",
                self.naxis,
                pix.len()
            )));
        }
        // 0-based -> 1-based: see the doc comment above.
        let pix: Vec<f64> = pix.iter().map(|p| p + 1.0).collect();
        let pix = pix.as_slice();
        // Step 1: pixel offset relative to CRPIX.
        let crpix = self.linear.crpix();
        let mut dp: Vec<f64> = (0..self.naxis).map(|j| pix[j] - crpix[j]).collect();
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
        let mut world = vec![0.0; self.naxis];
        for i in 0..self.naxis {
            world[i] = self.crval[i] + intermediate[i];
        }
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
        for tab in &self.tab {
            world[tab.axis] = tab.forward(world[tab.axis])?;
        }
        // Any unresolved -TAB axis is a hard error: silently
        // returning the linear approximation would be a wrong
        // answer disguised as a right one.
        if self.tab_specs.len() != self.tab.len() {
            return Err(FitsError::Wcs(format!(
                "WCS has {} unresolved -TAB axis spec(s); \
                 call FitsFile::wcs() or Wcs::resolve_tab() to load them",
                // `saturating_sub`: `resolve_tab` never leaves `tab`
                // longer than `tab_specs`, but both are public fields
                // and a caller that pushed into `tab` directly should
                // land on this error, not an overflow panic.
                self.tab_specs.len().saturating_sub(self.tab.len()),
            )));
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
            // non-degree CUNIT (Paper I Sec.3.1).
            let fx = to_degrees_factor(&self.cunit[c.pair.lon]);
            let fy = to_degrees_factor(&self.cunit[c.pair.lat]);
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
            let (phi, theta) = c.projection.x2s(x, y)?;
            let (alpha, delta) = c.rotation.native_to_celestial(phi, theta);
            world[c.pair.lon] = alpha;
            world[c.pair.lat] = delta;
        }
        Ok(world)
    }

    /// World -> pixel coordinates.
    ///
    /// Returns 0-based pixel coordinates. See [`pixel_to_world`](Self::pixel_to_world)
    /// for the indexing convention.
    pub fn world_to_pixel(&self, world: &[f64]) -> Result<Vec<f64>> {
        if world.len() != self.naxis {
            return Err(FitsError::Wcs(format!(
                "expected {} world coordinates, got {}",
                self.naxis,
                world.len()
            )));
        }
        let mut intermediate = vec![0.0; self.naxis];
        for i in 0..self.naxis {
            intermediate[i] = world[i] - self.crval[i];
        }
        // Spectral axes: invert the algorithm.
        for sx in &self.spectral {
            intermediate[sx.axis] = sx.world_to_intermediate(world[sx.axis])?;
        }
        // Tabular axes: invert the lookup. Same all-or-nothing
        // rule as the forward pass. The lookup yields the full
        // intermediate world coordinate; subtract CRVAL to get
        // back to the linear-pipeline space.
        for tab in &self.tab {
            intermediate[tab.axis] = tab.inverse(world[tab.axis])? - self.crval[tab.axis];
        }
        if self.tab_specs.len() != self.tab.len() {
            return Err(FitsError::Wcs(format!(
                "WCS has {} unresolved -TAB axis spec(s); \
                 call FitsFile::wcs() or Wcs::resolve_tab() to load them",
                // `saturating_sub`: `resolve_tab` never leaves `tab`
                // longer than `tab_specs`, but both are public fields
                // and a caller that pushed into `tab` directly should
                // land on this error, not an overflow panic.
                self.tab_specs.len().saturating_sub(self.tab.len()),
            )));
        }
        if let Some(c) = self.celestial.as_ref() {
            // DSS plate solution: bypass the standard inverse and
            // hand the world coordinates straight to the plate
            // model. Other (non-celestial) axes still flow through
            // the linear pipeline below.
            if let Some(dss) = self.dss.as_ref() {
                let (px, py) = dss.world_to_pixel(world[c.pair.lon], world[c.pair.lat])?;
                let mut out = vec![0.0; self.naxis];
                // DSS works in 1-based coords internally; the public
                // API is 0-based.
                out[c.pair.lon] = px - 1.0;
                out[c.pair.lat] = py - 1.0;
                return Ok(out);
            }
            let alpha = world[c.pair.lon];
            let delta = world[c.pair.lat];
            let (phi, theta) = c.rotation.celestial_to_native(alpha, delta);
            let (mut x, mut y) = c.projection.s2x(phi, theta)?;
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
            let fx = to_degrees_factor(&self.cunit[c.pair.lon]);
            let fy = to_degrees_factor(&self.cunit[c.pair.lat]);
            intermediate[c.pair.lon] = x / fx;
            intermediate[c.pair.lat] = y / fy;
        }
        // Inverse linear matrix.
        let mut dp = self.linear.apply_inverse_matrix(&intermediate)?;
        // Inverse SIP.
        if let Some(c) = self.celestial.as_ref()
            && let Some(sip) = c.sip.as_ref()
        {
            let (u, v) = sip.inverse(dp[c.pair.lon], dp[c.pair.lat])?;
            dp[c.pair.lon] = u;
            dp[c.pair.lat] = v;
        }
        let crpix = self.linear.crpix();
        // 1-based -> 0-based: see pixel_to_world doc.
        Ok((0..self.naxis).map(|i| crpix[i] + dp[i] - 1.0).collect())
    }

    /// Indices of the celestial longitude / latitude axes, if any.
    /// Convenience for callers who do not want to reach into
    /// `self.celestial`.
    #[must_use]
    pub fn celestial_axes(&self) -> Option<(usize, usize)> {
        self.celestial.as_ref().map(|c| (c.pair.lon, c.pair.lat))
    }

    /// True iff this WCS has a celestial axis pair.
    #[must_use]
    pub fn is_celestial(&self) -> bool {
        self.celestial.is_some()
    }

    /// Sky positions of the image's four corner pixels, as
    /// `(lon, lat)` in degrees.
    ///
    /// Corners are the **centers of the corner pixels** -- `(0, 0)`,
    /// `(nx-1, 0)`, `(nx-1, ny-1)`, `(0, ny-1)` in this crate's
    /// 0-based convention -- matching the default of astropy's
    /// `WCS.calc_footprint()`. For the outer edge of the pixel grid
    /// instead, call [`Self::pixel_to_celestial`] with `-0.5` and
    /// `n - 0.5` directly.
    ///
    /// Returns the corners counter-clockwise in pixel space, starting
    /// at the origin.
    ///
    /// # Errors
    ///
    /// Returns [`FitsError::Wcs`] if the WCS has no celestial axis
    /// pair, or if [`Self::pixel_shape`] is absent (a fitted WCS, or
    /// a header without `NAXISn`) or does not cover both celestial
    /// axes. Note that the shape is a snapshot taken at parse time:
    /// if the image has since been cropped or rebinned, the corners
    /// describe the original.
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

    /// Validate the preconditions that apply to a whole batch rather
    /// than to one point, so the batch helpers can distinguish
    /// "this WCS cannot transform anything" (an `Err`) from "this
    /// particular point is outside the projection" (a `NaN` slot).
    ///
    /// These two are the only structural failures reachable from the
    /// celestial helpers: the linear matrix is inverted once at parse
    /// time, and the coordinate vectors are built here with the right
    /// length, so every remaining error is per-point.
    fn batch_precheck(&self) -> Result<()> {
        if self.celestial.is_none() {
            return Err(FitsError::Wcs("WCS has no celestial axis pair".into()));
        }
        if self.tab_specs.len() != self.tab.len() {
            return Err(FitsError::Wcs(format!(
                "WCS has {} unresolved -TAB axis spec(s); \
                 call FitsFile::wcs() or Wcs::resolve_tab() to load them",
                // `saturating_sub`: `resolve_tab` never leaves `tab`
                // longer than `tab_specs`, but both are public fields
                // and a caller that pushed into `tab` directly should
                // land on this error, not an overflow panic.
                self.tab_specs.len().saturating_sub(self.tab.len()),
            )));
        }
        Ok(())
    }

    /// Batch pixel -> (RA, Dec). Same transform as
    /// [`Self::pixel_to_celestial`] but amortises the per-call setup
    /// across `pixels.len()` points and writes into a caller-owned
    /// `Vec` so a tight catalog-projection loop pays no per-point
    /// allocation cost.
    ///
    /// # Out-of-domain points
    ///
    /// Points that fall outside the projection's valid domain yield
    /// `(f64::NAN, f64::NAN)` in their slot instead of aborting the
    /// call, matching `wcslib` and `astropy.wcs`. Most all-sky
    /// projections cover only part of the plane (SIN's unit circle,
    /// ZPN below `PV2_0`, AZP beyond the horizon), so a wide field
    /// routinely mixes valid and invalid pixels; failing the batch
    /// would throw away every good result for one bad point. Use
    /// [`Self::pixel_to_celestial`] on a single point when you want
    /// the diagnostic message instead.
    ///
    /// An `Err` here means the *whole* WCS cannot transform: no
    /// celestial axis pair, or unresolved `-TAB` axes.
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

    /// Batch (RA, Dec) -> pixel. Mirror of
    /// [`Self::pixel_to_celestial_many`], including the `NaN`
    /// treatment of out-of-domain points.
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

    /// 2D celestial pixel -> (RA, Dec) in degrees. Convenience for the
    /// overwhelmingly common case of a 2-axis celestial image. Returns
    /// `Err` if the WCS has no celestial pair. Non-celestial axes (if
    /// any) are evaluated at their reference pixel, which is the only
    /// well-defined choice when the caller has not supplied them.
    ///
    /// `(px, py)` are 0-based -- see [`pixel_to_world`](Self::pixel_to_world).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use fitsy::FitsFile;
    ///
    /// let f = FitsFile::open("image.fits")?;
    /// let wcs = f.wcs(0, ' ')?.expect("no WCS in HDU 0");
    /// let (ra, dec) = wcs.pixel_to_celestial(512.0, 512.0)?;
    /// println!("RA={ra:.6} Dec={dec:.6}");
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

    /// 2D (RA, Dec) in degrees -> celestial pixel. Mirror of
    /// [`Self::pixel_to_celestial`].
    ///
    /// Returned `(px, py)` are 0-based.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use fitsy::FitsFile;
    ///
    /// let f = FitsFile::open("image.fits")?;
    /// let wcs = f.wcs(0, ' ')?.expect("no WCS in HDU 0");
    /// let (px, py) = wcs.celestial_to_pixel(202.469, 47.195)?;
    /// println!("pixel = ({px:.2}, {py:.2})");
    /// # Ok::<(), fitsy::FitsError>(())
    /// ```
    pub fn celestial_to_pixel(&self, ra: f64, dec: f64) -> Result<(f64, f64)> {
        let (lon, lat) = self
            .celestial_axes()
            .ok_or_else(|| FitsError::Wcs("WCS has no celestial axis pair".into()))?;
        // Build a world vector with the celestial pair set and the
        // other axes at CRVAL (zero, since CRVAL is absorbed into the
        // celestial rotation and the spectral algorithms).
        let mut world = self.crval.clone();
        world[lon] = ra;
        world[lat] = dec;
        let pix = self.world_to_pixel(&world)?;
        Ok((pix[lon], pix[lat]))
    }

    /// Local pixel scale at pixel `(px, py)`, in **arcseconds per
    /// pixel** along the longitude and latitude axes respectively.
    /// Computed by finite difference on the unit sphere, so the
    /// figures account for `cos(dec)` foreshortening, distortion
    /// (SIP/TPV/TNX), and any local skew of the projection.
    ///
    /// Returns the magnitude of the great-circle distance per pixel
    /// along each axis, *not* the signed `CDELT` value: a
    /// flipped-RA image still reports a positive scale. Callers
    /// asking "what's the pixel scale of this image?" want this.
    pub fn pixel_scale_at(&self, px: f64, py: f64) -> Result<(f64, f64)> {
        let (ra0, dec0) = self.pixel_to_celestial(px, py)?;
        let (ra_x, dec_x) = self.pixel_to_celestial(px + 1.0, py)?;
        let (ra_y, dec_y) = self.pixel_to_celestial(px, py + 1.0)?;
        let dx_arcsec = great_circle_arcsec(ra0, dec0, ra_x, dec_x);
        let dy_arcsec = great_circle_arcsec(ra0, dec0, ra_y, dec_y);
        Ok((dx_arcsec, dy_arcsec))
    }

    /// Resolve every parsed `-TAB` axis against the binary tables
    /// in `file`. Idempotent: calling twice has no effect after the
    /// first successful resolution. Returns the number of axes
    /// resolved. Most callers do not call this directly -- use
    /// [`crate::FitsFile::wcs`] which resolves transparently.
    pub fn resolve_tab(&mut self, file: &crate::FitsFile) -> Result<usize> {
        if self.tab_specs.len() == self.tab.len() {
            return Ok(0);
        }
        let mut resolved = Vec::with_capacity(self.tab_specs.len());
        for spec in &self.tab_specs {
            let axis = load_tab_axis(file, spec)?;
            resolved.push(axis);
        }
        let n = resolved.len();
        self.tab = resolved;
        Ok(n)
    }
}

/// Load one `-TAB` axis from its referenced binary table.
fn load_tab_axis(file: &crate::FitsFile, spec: &TabSpec) -> Result<TabAxis> {
    let hdu = file.hdu_by_name(&spec.extname, Some(spec.extver))?;
    let crate::Hdu::BinTable(bin) = hdu else {
        return Err(FitsError::Wcs(format!(
            "-TAB axis {}: extension `{}` (EXTVER {}) is not a BINTABLE",
            spec.axis + 1,
            spec.extname,
            spec.extver,
        )));
    };
    let coord = read_tab_column(&bin, &spec.coord_column, spec)?;
    let index = match &spec.index_column {
        Some(name) => Some(read_tab_column(&bin, name, spec)?),
        None => None,
    };
    if let Some(idx) = index.as_ref()
        && idx.len() != coord.len()
    {
        return Err(FitsError::Wcs(format!(
            "-TAB axis {}: index column `{}` length {} != coord column `{}` length {}",
            spec.axis + 1,
            spec.index_column.as_deref().unwrap_or(""),
            idx.len(),
            spec.coord_column,
            coord.len(),
        )));
    }
    Ok(TabAxis {
        axis: spec.axis,
        coord,
        index,
    })
}

/// Read a single 1-D float column from a BINTABLE as `Vec<f64>`. The
/// column is expected to hold one row whose cell is the full lookup
/// array (the canonical Paper III layout for single-axis -TAB).
fn read_tab_column(
    bin: &crate::hdu::bintable::BinTableHdu<'_>,
    name: &str,
    spec: &TabSpec,
) -> Result<Vec<f64>> {
    use crate::hdu::bintable::BinValue;
    let col = bin.column_by_name(name).ok_or_else(|| {
        FitsError::Wcs(format!(
            "-TAB axis {}: BINTABLE `{}` has no column `{name}`",
            spec.axis + 1,
            spec.extname,
        ))
    })?;
    if bin.n_rows() != 1 {
        return Err(FitsError::Wcs(format!(
            "-TAB axis {}: BINTABLE `{}` has {} rows; only single-row \
             1-D lookup tables are supported",
            spec.axis + 1,
            spec.extname,
            bin.n_rows(),
        )));
    }
    if spec.coord_axis != 1 {
        return Err(FitsError::Wcs(format!(
            "-TAB axis {}: PV{}_3 = {} requests a multi-dimensional \
             lookup which is not supported",
            spec.axis + 1,
            spec.axis + 1,
            spec.coord_axis,
        )));
    }
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
    Ok(v)
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
