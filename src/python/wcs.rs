//! `PyWcs` -- Python wrapper around `crate::wcs::Wcs`.

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::wcs::Wcs;

use super::IntoPyResult;
use super::header::PyHeader;

/// Validate the `origin` argument and return the offset to add to a
/// caller-supplied pixel coordinate before handing it to the
/// (now 0-based) Rust API.
///
/// `origin = 0` (default, numpy / C convention): caller pixels are
/// 0-based, matching the Rust API, so we add 0.
///
/// `origin = 1` (FITS convention): caller pixels are 1-based, so we
/// subtract 1 to convert to the Rust API's 0-based convention.
///
/// Matches the semantics of `astropy.wcs.WCS.{wcs_,all_}pix2world`.
fn pixel_offset(origin: u8) -> PyResult<f64> {
    match origin {
        0 => Ok(0.0),
        1 => Ok(-1.0),
        other => Err(PyValueError::new_err(format!(
            "origin must be 0 (numpy/C, default) or 1 (FITS); got {other}"
        ))),
    }
}

/// World Coordinate System for an HDU.
///
/// Constructed via :meth:`FitsFile.wcs`, :meth:`ImageHdu.wcs`, or
/// directly from a header. Supports celestial, spectral, time and
/// generic linear axes; SIP, TPV, TNX, ``-TAB`` and DSS distortion
/// conventions are recognized.
///
/// Examples
/// --------
/// >>> with fitsy.open("image.fits") as f:
/// ...     wcs = f[0].wcs()
/// ...     ra, dec = wcs.pixel_to_celestial(512.0, 512.0)
#[pyclass(name = "Wcs", module = "fitsy")]
#[derive(Debug)]
pub struct PyWcs {
    pub(crate) inner: Wcs,
}

impl From<Wcs> for PyWcs {
    fn from(w: Wcs) -> Self {
        Self { inner: w }
    }
}

impl PyWcs {
    /// Astropy-style multi-line summary used by ``__repr__`` and
    /// ``__str__``. Mirrors the ``WCS Keywords`` block printed by
    /// :class:`astropy.wcs.WCS` so users get the same at-a-glance
    /// information (CTYPE / CUNIT / CRVAL / CRPIX / CD or PC).
    fn format_summary(&self) -> String {
        use std::fmt::Write as _;
        let w = &self.inner;
        let n = w.naxis;
        let mut out = String::with_capacity(256);
        out.push_str("WCS Keywords\n\n");
        let _ = writeln!(out, "Number of WCS axes: {n}");

        // Quoted, space-separated lists of the per-axis string
        // metadata, matching astropy's exact formatting.
        let quoted = |items: &[String]| -> String {
            let mut s = String::new();
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    s.push(' ');
                }
                s.push('\'');
                s.push_str(v);
                s.push('\'');
            }
            s
        };
        let nums = |items: &[f64]| -> String {
            let mut s = String::new();
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    s.push(' ');
                }
                let _ = write!(s, "{v}");
            }
            s
        };

        let _ = writeln!(out, "CTYPE : {}", quoted(&w.ctype));
        if !w.cunit.is_empty() && w.cunit.iter().any(|s| !s.is_empty()) {
            let _ = writeln!(out, "CUNIT : {}", quoted(&w.cunit));
        }

        // CRVAL is stored zeroed-out for celestial / spectral axes
        // (those values are absorbed into the celestial rotation /
        // spectral helper). Reconstruct the user-facing CRVAL.
        let mut crval = w.crval.clone();
        if let Some(c) = w.celestial.as_ref() {
            if c.pair.lon < crval.len() {
                crval[c.pair.lon] = c.rotation.alpha0;
            }
            if c.pair.lat < crval.len() {
                crval[c.pair.lat] = c.rotation.delta0;
            }
        }
        for sx in &w.spectral {
            if sx.axis < crval.len() {
                // Reverse the SI conversion done at parse time.
                let unit = if sx.unit_to_si == 0.0 {
                    1.0
                } else {
                    sx.unit_to_si
                };
                crval[sx.axis] = sx.crval_si / unit;
            }
        }
        let _ = writeln!(out, "CRVAL : {}", nums(&crval));
        let _ = writeln!(out, "CRPIX : {}", nums(w.linear.crpix()));

        // Linear matrix as CD<i>_<j> rows. We don't carry the
        // PC-vs-CD distinction at this layer, so always label as
        // CD; mirrors what astropy prints for SIP/CD-based headers.
        let m = w.linear.matrix_row_major();
        if m.len() == n * n && n > 0 {
            for i in 0..n {
                let mut header_label = String::new();
                let mut row = String::new();
                for j in 0..n {
                    if j > 0 {
                        header_label.push(' ');
                        row.push(' ');
                    }
                    let _ = write!(header_label, "CD{}_{}", i + 1, j + 1);
                    let _ = write!(row, "{}", m[i * n + j]);
                }
                let _ = writeln!(out, "{header_label} : {row}");
            }
        }

        if let Some(name) = w.wcsname.as_ref() {
            let _ = writeln!(out, "WCSNAME : '{name}'");
        }
        if w.celestial.is_some() {
            let _ = writeln!(out, "RADESYS : {:?}", w.radesys);
            if let Some(eq) = w.equinox {
                let _ = writeln!(out, "EQUINOX : {eq}");
            }
        }
        out
    }
}

#[pymethods]
impl PyWcs {
    /// Construct a WCS by parsing a header.
    ///
    /// Parameters
    /// ----------
    /// header : Header
    ///     The HDU header to parse.
    /// alt : str, optional
    ///     Alternate-WCS letter; ``' '`` for the primary description.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the header carries no WCS for ``alt``.
    ///
    /// Notes
    /// -----
    /// ``-TAB`` axes are *not* auto-resolved by this constructor;
    /// use :meth:`FitsFile.wcs` for that.
    #[new]
    #[pyo3(signature = (header, alt=' '))]
    fn py_new(header: &PyHeader, alt: char) -> PyResult<Self> {
        let inner = Wcs::from_header(&header.lock(), alt)
            .into_py_result()?
            .ok_or_else(|| PyValueError::new_err("header carries no WCS"))?;
        Ok(Self { inner })
    }

    /// Number of axes (``NAXIS``).
    #[getter]
    fn naxis(&self) -> usize {
        self.inner.naxis
    }

    /// Per-axis ``CTYPE`` strings.
    #[getter]
    fn ctype(&self) -> Vec<String> {
        self.inner.ctype.clone()
    }

    /// Per-axis ``CUNIT`` strings.
    #[getter]
    fn cunit(&self) -> Vec<String> {
        self.inner.cunit.clone()
    }

    /// Per-axis ``CRVAL`` reference values.
    #[getter]
    fn crval(&self) -> Vec<f64> {
        self.inner.crval.clone()
    }

    /// True when the WCS has a celestial axis pair.
    #[getter]
    fn is_celestial(&self) -> bool {
        self.inner.is_celestial()
    }

    /// Indices of the celestial axes.
    ///
    /// Returns
    /// -------
    /// tuple of int or None
    ///     ``(lon_axis, lat_axis)`` (zero-based), or ``None`` if no
    ///     celestial pair is declared.
    fn celestial_axes(&self) -> Option<(usize, usize)> {
        self.inner.celestial_axes()
    }

    /// Forward transform a single pixel coordinate.
    ///
    /// Parameters
    /// ----------
    /// pix : sequence of float
    ///     Length-``naxis`` pixel coordinate.
    /// origin : int, optional
    ///     ``0`` (default) treats ``pix`` as 0-based (numpy/C
    ///     convention, matching ``astropy.wcs``); ``1`` treats
    ///     it as 1-based FITS coordinates.
    ///
    /// Returns
    /// -------
    /// list of float
    ///     World coordinates with units given by :attr:`cunit`.
    #[pyo3(signature = (pix, origin=0))]
    fn pixel_to_world(&self, pix: Vec<f64>, origin: u8) -> PyResult<Vec<f64>> {
        let off = pixel_offset(origin)?;
        let shifted: Vec<f64> = pix.iter().map(|p| p + off).collect();
        self.inner.pixel_to_world(&shifted).into_py_result()
    }

    /// Inverse transform world to pixel.
    ///
    /// Parameters
    /// ----------
    /// world : sequence of float
    ///     Length-``naxis`` world coordinate.
    /// origin : int, optional
    ///     ``0`` (default) returns 0-based pixel coordinates,
    ///     ``1`` returns 1-based FITS coordinates.
    ///
    /// Returns
    /// -------
    /// list of float
    ///     Pixel coordinate in the chosen origin.
    #[pyo3(signature = (world, origin=0))]
    fn world_to_pixel(&self, world: Vec<f64>, origin: u8) -> PyResult<Vec<f64>> {
        let off = pixel_offset(origin)?;
        let mut out = self.inner.world_to_pixel(&world).into_py_result()?;
        for p in &mut out {
            *p -= off;
        }
        Ok(out)
    }

    /// Forward transform a single celestial pixel.
    ///
    /// Parameters
    /// ----------
    /// px, py : float
    ///     Pixel coordinates.
    /// origin : int, optional
    ///     ``0`` (default) treats inputs as 0-based; ``1`` as
    ///     1-based FITS coordinates.
    ///
    /// Returns
    /// -------
    /// tuple of float
    ///     ``(ra, dec)`` (or ``(lon, lat)``) in degrees.
    #[pyo3(signature = (px, py, origin=0))]
    fn pixel_to_celestial(&self, px: f64, py: f64, origin: u8) -> PyResult<(f64, f64)> {
        let off = pixel_offset(origin)?;
        self.inner
            .pixel_to_celestial(px + off, py + off)
            .into_py_result()
    }

    /// Inverse celestial transform.
    ///
    /// Parameters
    /// ----------
    /// ra, dec : float
    ///     Sky coordinates in degrees.
    /// origin : int, optional
    ///     ``0`` (default) returns 0-based ``(px, py)``; ``1``
    ///     returns 1-based FITS pixel coordinates.
    ///
    /// Returns
    /// -------
    /// tuple of float
    ///     Pixel coordinates in the chosen origin.
    #[pyo3(signature = (ra, dec, origin=0))]
    fn celestial_to_pixel(&self, ra: f64, dec: f64, origin: u8) -> PyResult<(f64, f64)> {
        let off = pixel_offset(origin)?;
        let (px, py) = self.inner.celestial_to_pixel(ra, dec).into_py_result()?;
        Ok((px - off, py - off))
    }

    /// Local pixel scale at ``(px, py)``.
    ///
    /// Parameters
    /// ----------
    /// px, py : float
    ///     Pixel coordinates.
    /// origin : int, optional
    ///     ``0`` (default) treats inputs as 0-based; ``1`` as
    ///     1-based FITS coordinates.
    ///
    /// Returns
    /// -------
    /// tuple of float
    ///     Pixel scale in degrees per pixel along the two
    ///     celestial axes.
    #[pyo3(signature = (px, py, origin=0))]
    fn pixel_scale_at(&self, px: f64, py: f64, origin: u8) -> PyResult<(f64, f64)> {
        let off = pixel_offset(origin)?;
        self.inner
            .pixel_scale_at(px + off, py + off)
            .into_py_result()
    }

    /// Batch celestial forward transform.
    ///
    /// Parameters
    /// ----------
    /// pixels : numpy.ndarray
    ///     Shape ``(N, 2)`` array of pixel coordinates.
    /// origin : int, optional
    ///     ``0`` (default) treats inputs as 0-based; ``1`` as
    ///     1-based FITS coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Shape ``(N, 2)`` array of ``(ra, dec)`` in degrees.
    #[pyo3(signature = (pixels, origin=0))]
    fn pixel_to_celestial_many<'py>(
        &self,
        py: Python<'py>,
        pixels: PyReadonlyArray2<'_, f64>,
        origin: u8,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let off = pixel_offset(origin)?;
        let view = pixels.as_array();
        if view.ncols() != 2 {
            return Err(PyValueError::new_err(
                "pixel_to_celestial_many expects an (N, 2) array",
            ));
        }
        let pairs: Vec<(f64, f64)> = view
            .outer_iter()
            .map(|row| (row[0] + off, row[1] + off))
            .collect();
        let out = self
            .inner
            .pixel_to_celestial_many(&pairs)
            .into_py_result()?;
        let mut flat = Vec::with_capacity(out.len() * 2);
        for (a, b) in out {
            flat.push(a);
            flat.push(b);
        }
        let n = pairs.len();
        let arr = flat.into_pyarray(py);
        Ok(arr.reshape([n, 2]).expect("reshape (N,2)"))
    }

    /// Batch celestial inverse transform.
    ///
    /// Parameters
    /// ----------
    /// sky : numpy.ndarray
    ///     Shape ``(N, 2)`` array of ``(ra, dec)`` in degrees.
    /// origin : int, optional
    ///     ``0`` (default) returns 0-based pixel coordinates,
    ///     ``1`` returns 1-based FITS coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Shape ``(N, 2)`` array of pixel coordinates.
    #[pyo3(signature = (sky, origin=0))]
    fn celestial_to_pixel_many<'py>(
        &self,
        py: Python<'py>,
        sky: PyReadonlyArray2<'_, f64>,
        origin: u8,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let off = pixel_offset(origin)?;
        let view = sky.as_array();
        if view.ncols() != 2 {
            return Err(PyValueError::new_err(
                "celestial_to_pixel_many expects an (N, 2) array",
            ));
        }
        let pairs: Vec<(f64, f64)> = view.outer_iter().map(|row| (row[0], row[1])).collect();
        let out = self
            .inner
            .celestial_to_pixel_many(&pairs)
            .into_py_result()?;
        let mut flat = Vec::with_capacity(out.len() * 2);
        for (a, b) in out {
            flat.push(a - off);
            flat.push(b - off);
        }
        let n = pairs.len();
        let arr = flat.into_pyarray(py);
        Ok(arr.reshape([n, 2]).expect("reshape (N,2)"))
    }

    fn __repr__(&self) -> String {
        self.format_summary()
    }

    fn __str__(&self) -> String {
        self.format_summary()
    }

    /// Serialize this WCS to a fresh :class:`Header`.
    ///
    /// Parameters
    /// ----------
    /// alt : str, optional
    ///     ``' '`` (default) for the primary description, or
    ///     ``'A'`` through ``'Z'`` for an alternate.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     For spectral, ``-TAB``, TPV, TNX or DSS WCSs (not
    ///     supported on the write path).
    #[pyo3(signature = (alt=' '))]
    fn to_header(&self, alt: char) -> PyResult<PyHeader> {
        let h = self.inner.to_header(alt).into_py_result()?;
        Ok(PyHeader::from_header_with(&h, false))
    }
}

/// Result of :func:`fit_wcs`.
///
/// Carries the fitted :class:`Wcs` and per-point residuals.
///
/// Attributes
/// ----------
/// wcs : Wcs
///     The fitted world coordinate system.
/// rms_arcsec : float
///     Root-mean-square residual across all reference points (arcsec).
/// max_arcsec : float
///     Largest single-point residual (arcsec).
#[pyclass(name = "WcsFit", module = "fitsy")]
#[derive(Debug)]
pub struct PyWcsFit {
    #[pyo3(get)]
    pub wcs: Py<PyWcs>,
    /// Per-point residuals as an `(N, 2)` array of
    /// `(delta_alpha*cos delta, delta_dec)` in arcseconds.
    residuals: Vec<(f64, f64)>,
    #[pyo3(get)]
    pub rms_arcsec: f64,
    #[pyo3(get)]
    pub max_arcsec: f64,
}

#[pymethods]
impl PyWcsFit {
    /// Per-point residuals as a numpy array.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Shape ``(N, 2)`` of ``(delta_alpha * cos(delta), delta_dec)``
    ///     in arcseconds.
    #[getter]
    fn residuals_arcsec<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.residuals.len();
        let mut flat = Vec::with_capacity(n * 2);
        for &(a, b) in &self.residuals {
            flat.push(a);
            flat.push(b);
        }
        let arr = flat.into_pyarray(py);
        arr.reshape([n, 2]).expect("reshape (N,2)")
    }

    fn __repr__(&self) -> String {
        format!(
            "WcsFit(rms={:.4e} arcsec, max={:.4e} arcsec, n={})",
            self.rms_arcsec,
            self.max_arcsec,
            self.residuals.len()
        )
    }
}

/// Fit a celestial WCS to ``(pixel, sky)`` reference correspondences.
///
/// Parameters
/// ----------
/// pixels : numpy.ndarray
///     Shape ``(N, 2)`` array of pixel coordinates.
/// sky : numpy.ndarray
///     Shape ``(N, 2)`` array of ``(ra, dec)`` in degrees, or more
///     generally ``(lon, lat)`` in the chosen ``frame``.
/// projection : str, optional
///     Three-letter projection code (default ``"TAN"``).
/// crpix : tuple of float, optional
///     Pin the reference pixel; otherwise solved as part of the fit.
///     Interpreted in the same ``origin`` as ``pixels``.
/// crval : tuple of float, optional
///     Pin the tangent point; otherwise defaults to the spherical
///     centroid of the sky points.
/// sip_order : int, optional
///     Enable a SIP polynomial distortion fit of the given order
///     (typically 2-4).
/// fit_sip_inverse : bool, optional
///     When ``sip_order`` is given, also fit the AP/BP inverse
///     polynomial. Default True.
/// frame : {'equatorial', 'galactic', 'ecliptic', 'supergalactic', 'helioecliptic'}, optional
///     Celestial frame for the sky coordinates. ``'equatorial'`` is
///     the default and emits ``RA-/DEC-`` CTYPE pairs.
/// origin : int, optional
///     ``0`` (default, numpy/C convention) treats ``pixels`` and
///     ``crpix`` as 0-based; ``1`` treats them as 1-based FITS
///     coordinates. The fitted WCS itself always carries 1-based
///     ``CRPIX`` values per the FITS standard.
///
/// Returns
/// -------
/// WcsFit
///     Fitted WCS, residuals, and summary statistics.
///
/// Raises
/// ------
/// ValueError
///     On shape mismatches or unknown ``projection``/``frame``.
///
/// Examples
/// --------
/// >>> import numpy as np, fitsy
/// >>> pix = np.array([[100.0, 100.0], [200.0, 100.0], [100.0, 200.0]])
/// >>> sky = np.array([[10.00, -5.00], [10.05, -5.00], [10.00, -4.95]])
/// >>> fit = fitsy.fit_wcs(pix, sky, projection="TAN")
/// >>> fit.rms_arcsec
/// 0.0
#[pyfunction]
#[pyo3(signature = (
    pixels,
    sky,
    projection="TAN",
    crpix=None,
    crval=None,
    sip_order=None,
    fit_sip_inverse=true,
    frame="equatorial",
    origin=0,
))]
#[allow(
    clippy::too_many_arguments,
    reason = "WCS fitting requires many distinct input parameters; grouping into a struct would worsen Python ergonomics"
)]
pub fn fit_wcs(
    py: Python<'_>,
    pixels: PyReadonlyArray2<'_, f64>,
    sky: PyReadonlyArray2<'_, f64>,
    projection: &str,
    crpix: Option<(f64, f64)>,
    crval: Option<(f64, f64)>,
    sip_order: Option<u32>,
    fit_sip_inverse: bool,
    frame: &str,
    origin: u8,
) -> PyResult<PyWcsFit> {
    let off = pixel_offset(origin)?;
    let pv = pixels.as_array();
    let sv = sky.as_array();
    if pv.ncols() != 2 || sv.ncols() != 2 {
        return Err(PyValueError::new_err(
            "fit_wcs: pixels and sky must both be (N, 2) arrays",
        ));
    }
    if pv.nrows() != sv.nrows() {
        return Err(PyValueError::new_err(
            "fit_wcs: pixels and sky must have the same number of rows",
        ));
    }
    let pixels_v: Vec<(f64, f64)> = pv.outer_iter().map(|r| (r[0] + off, r[1] + off)).collect();
    let sky_v: Vec<(f64, f64)> = sv.outer_iter().map(|r| (r[0], r[1])).collect();
    let crpix = crpix.map(|(x, y)| (x + off, y + off));

    let proj_kind =
        crate::wcs::projection::ProjectionKind::from_code(&projection.to_ascii_uppercase())
            .into_py_result()?;
    let frame_kind = match frame.to_ascii_lowercase().as_str() {
        "equatorial" | "icrs" | "fk5" | "fk4" => crate::wcs::CelestialFrame::Equatorial,
        "galactic" => crate::wcs::CelestialFrame::Galactic,
        "ecliptic" => crate::wcs::CelestialFrame::Ecliptic,
        "supergalactic" => crate::wcs::CelestialFrame::Supergalactic,
        "helioecliptic" => crate::wcs::CelestialFrame::HelioEcliptic,
        other => {
            return Err(PyValueError::new_err(format!(
                "fit_wcs: unknown frame {other:?}"
            )));
        }
    };

    let opts = crate::wcs::WcsFitOptions {
        projection: proj_kind,
        crpix,
        crval,
        frame: frame_kind,
        sip_order,
        fit_sip_inverse,
    };
    let fit = crate::wcs::fit_celestial_wcs(&pixels_v, &sky_v, &opts).into_py_result()?;
    let wcs = Py::new(py, PyWcs { inner: fit.wcs })?;
    Ok(PyWcsFit {
        wcs,
        residuals: fit.residuals_arcsec,
        rms_arcsec: fit.rms_arcsec,
        max_arcsec: fit.max_arcsec,
    })
}
