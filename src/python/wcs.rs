//! `PyWcs` -- Python wrapper around `crate::wcs::Wcs`.

use numpy::{AllowTypeChange, IntoPyArray, PyArray2, PyArrayLike2, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::wcs::{AxisKind, Wcs};

use super::IntoPyResult;
use super::header::PyHeader;

/// Validate `origin` and return the offset that converts a
/// caller-supplied pixel coordinate to the 0-based Rust API.
///
/// `origin = 0` (default) is the numpy and C convention. Caller
/// pixels are already 0-based, so the offset is `0`.
///
/// `origin = 1` is the FITS convention. Caller pixels are 1-based,
/// so the offset is `-1`, which converts them to 0-based.
///
/// # Errors
///
/// Returns [`PyValueError`] if `origin` is neither `0` nor `1`.
fn pixel_offset(origin: u8) -> PyResult<f64> {
    match origin {
        0 => Ok(0.0),
        1 => Ok(-1.0),
        other => Err(PyValueError::new_err(format!(
            "origin must be 0 (numpy/C, default) or 1 (FITS); got {other}"
        ))),
    }
}

/// Check the column count of a 2-D batch against `naxis`.
///
/// The Rust batch entry points take the points flat. They can only
/// test the total length. A `(N, 3)` array on a two-axis WCS therefore
/// reads as `3N/2` points whenever `N` is even. The result reshapes
/// back to `(N, 3)` and reports no error. A transposed `(naxis, N)`
/// array passes the same test and pairs the wrong values together.
/// Only the caller's shape separates these two cases from a real
/// batch, so this layer holds the check.
///
/// # Errors
///
/// Returns [`PyValueError`] if `cols` is not `naxis`.
fn check_batch_cols(method: &str, cols: usize, naxis: usize) -> PyResult<()> {
    if cols == naxis {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{method}: expected a batch of shape (N, {naxis}) for a {naxis}-axis WCS, \
         got {cols} columns; pass the transpose if the points run down the columns"
    )))
}

/// World Coordinate System for an HDU.
///
/// Constructed via :meth:`FitsFile.wcs`, :meth:`ImageHdu.wcs`, or
/// directly from a header. Supports celestial, spectral, time,
/// phase and generic linear axes. Recognizes the SIP, TPV,
/// TNX/ZPX, ``-TAB`` and DSS distortion conventions.
///
/// Examples
/// --------
/// >>> with fitsy.open("image.fits") as f:
/// ...     wcs = f[0].wcs()
/// ...     kinds = wcs.axis_kinds()
/// ...     world = wcs.pixel_to_world([512.0, 512.0])
/// ...     ra = world[kinds.index("longitude")]
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
    /// Multi-line summary of the WCS keywords, used by `__repr__`
    /// and `__str__`. Lists `CTYPE`, `CUNIT`, `CRVAL`, `CRPIX` and
    /// the linear matrix, always labeled `CD`.
    fn format_summary(&self) -> String {
        use std::fmt::Write as _;
        let w = &self.inner;
        let n = w.naxis();
        let mut out = String::with_capacity(256);
        out.push_str("WCS Keywords\n\n");
        let _ = writeln!(out, "Number of WCS axes: {n}");

        // Quoted, space-separated lists of the per-axis string
        // metadata.
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

        let ctype: Vec<String> = w.axes().iter().map(|a| a.ctype.clone()).collect();
        let cunit: Vec<String> = w.axes().iter().map(|a| a.cunit.clone()).collect();
        let _ = writeln!(out, "CTYPE : {}", quoted(&ctype));
        if cunit.iter().any(|s| !s.is_empty()) {
            let _ = writeln!(out, "CUNIT : {}", quoted(&cunit));
        }

        let _ = writeln!(out, "CRVAL : {}", nums(w.crval()));
        let _ = writeln!(out, "CRPIX : {}", nums(w.linear().crpix()));

        // Linear matrix as CD<i>_<j> rows. This layer does not
        // carry the PC-vs-CD distinction, so it always labels the
        // matrix as CD.
        let m = w.linear().matrix_row_major();
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
    ///   The HDU header to parse.
    /// alt : str, optional
    ///   Alternate-WCS letter; ``' '`` for the primary description.
    ///
    /// Raises
    /// ------
    /// FitsError
    ///   If ``alt`` is not ``' '`` or ``'A'``-``'Z'``, or if the
    ///   header carries a malformed WCS description.
    /// ValueError
    ///   If the header carries no WCS for ``alt``.
    ///
    /// Notes
    /// -----
    /// This constructor does not resolve a ``-TAB`` axis, because a
    /// header alone does not reach the table it names. Use
    /// :meth:`FitsFile.wcs` or :meth:`ImageHdu.wcs` for that.
    #[new]
    #[pyo3(signature = (header, alt=' '))]
    fn py_new(header: &PyHeader, alt: char) -> PyResult<Self> {
        let inner = Wcs::from_header(&header.lock(), alt)
            .into_py_result()?
            .ok_or_else(|| PyValueError::new_err("header carries no WCS"))?;
        Ok(Self { inner })
    }

    /// Number of WCS axes. May exceed the header's ``NAXIS`` when
    /// ``WCSAXESa`` sets a higher axis count.
    #[getter]
    fn naxis(&self) -> usize {
        self.inner.naxis()
    }

    /// Per-axis ``CTYPE`` strings.
    #[getter]
    fn ctype(&self) -> Vec<String> {
        self.inner.axes().iter().map(|a| a.ctype.clone()).collect()
    }

    /// Per-axis ``CUNIT`` strings. Empty for an axis whose header
    /// carries no ``CUNIT`` card; that axis then uses the default
    /// unit for its axis type.
    #[getter]
    fn cunit(&self) -> Vec<String> {
        self.inner.axes().iter().map(|a| a.cunit.clone()).collect()
    }

    /// Per-axis ``CRVAL`` reference values, in the unit given by
    /// :attr:`cunit`.
    #[getter]
    fn crval(&self) -> Vec<f64> {
        self.inner.crval().to_vec()
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
    ///   ``(lon_axis, lat_axis)`` (zero-based), or ``None`` if no
    ///   celestial pair is declared.
    fn celestial_axes(&self) -> Option<(usize, usize)> {
        self.inner.celestial_axes()
    }

    /// Kind of coordinate each axis carries, in axis order.
    ///
    /// Use this to find an axis by meaning rather than by position.
    /// :meth:`pixel_to_world` returns one value per axis in the same
    /// order, so entry ``i`` here names value ``i`` there.
    ///
    /// Returns
    /// -------
    /// list of str
    ///   One entry per axis, each one of ``'longitude'``,
    ///   ``'latitude'``, ``'spectral'``, ``'time'``, ``'phase'``,
    ///   ``'stokes'`` or ``'linear'``.
    ///
    /// Notes
    /// -----
    /// The kind comes from the type half of ``CTYPEia``, so an axis
    /// driven by a ``-TAB`` lookup still reports its coordinate type.
    /// :meth:`is_tabular` reports the lookup itself.
    ///
    /// Examples
    /// --------
    /// Find the spectral axis of a cube and read its world value:
    ///
    /// >>> with fitsy.open("cube.fits") as f:
    /// ...     wcs = f[0].wcs()
    /// ...     kinds = wcs.axis_kinds()
    /// ...     world = wcs.pixel_to_world([31.0, 23.0, 4.0])
    /// ...     freq = world[kinds.index("spectral")]
    fn axis_kinds(&self) -> Vec<&'static str> {
        self.inner
            .axis_kinds()
            .into_iter()
            .map(|k| match k {
                AxisKind::Longitude => "longitude",
                AxisKind::Latitude => "latitude",
                AxisKind::Spectral => "spectral",
                AxisKind::Time => "time",
                AxisKind::Phase => "phase",
                AxisKind::Stokes => "stokes",
                AxisKind::Linear => "linear",
            })
            .collect()
    }

    /// Whether an axis takes its coordinate from a ``-TAB`` lookup.
    ///
    /// Parameters
    /// ----------
    /// axis : int
    ///   Zero-based axis index.
    ///
    /// Returns
    /// -------
    /// bool
    ///   ``True`` when axis ``axis`` is tabular. ``False`` for any
    ///   other axis, and for an index this WCS does not have.
    ///
    /// Notes
    /// -----
    /// This is a property of the algorithm, not of the coordinate, so
    /// it is independent of :meth:`axis_kinds`. A tabular axis needs
    /// its binary table loaded before it can transform.
    /// :meth:`fitsy.FitsFile.wcs` and :meth:`fitsy.ImageHdu.wcs` load
    /// it. ``fitsy.Wcs(header)`` cannot, because a header alone does
    /// not reach the table.
    fn is_tabular(&self, axis: usize) -> bool {
        self.inner.is_tabular(axis)
    }

    /// Size of the image this WCS came from, in FITS axis order
    /// (``NAXIS1`` first), or ``None`` when unknown.
    ///
    /// This is a snapshot of the ``NAXISn`` cards taken when the WCS
    /// was parsed. It is not part of the coordinate description: no
    /// transform reads it, and no transform checks a pixel
    /// coordinate against it. It is ``None`` for a WCS from
    /// :func:`fit_wcs`, which has no image, or for a header without
    /// ``NAXISn`` cards. A cropped or rebinned image leaves this
    /// value stale.
    #[getter]
    fn pixel_shape<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.inner
            .pixel_shape
            .as_ref()
            .map(|s| PyTuple::new(py, s))
            .transpose()
    }

    /// World coordinates of the image's corner pixels.
    ///
    /// Corners are pixel centers, not the outer edge of the grid. For
    /// the outer edge, call :meth:`pixel_to_world` with ``-0.5`` and
    /// ``n - 0.5``.
    ///
    /// Corners come back in Gray-code order, so consecutive corners
    /// differ on one axis alone. A two-axis image therefore yields
    /// ``(0, 0)``, ``(nx-1, 0)``, ``(nx-1, ny-1)``, ``(0, ny-1)``,
    /// which walks the image counter-clockwise in pixel space and
    /// closes the ring.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///   Shape ``(2**k, naxis)``. ``k`` is the number of axes
    ///   :attr:`pixel_shape` covers, which is :attr:`naxis` for a
    ///   normal image. A two-axis image gives ``(4, 2)`` of
    ///   ``(ra, dec)`` in degrees. A corner the WCS cannot transform
    ///   comes back as ``nan``. See the notes below.
    ///
    /// Raises
    /// ------
    /// FitsError
    ///   In three cases:
    ///
    ///   - :attr:`pixel_shape` is ``None``. A fitted WCS has no image
    ///     to take corners from.
    ///   - An axis :attr:`pixel_shape` covers has length zero.
    ///   - The WCS has more than 16 axes.
    ///
    /// Notes
    /// -----
    /// This reports corners, not an axis-aligned bounding box. A
    /// rotated image has corners outside the box its own minimum and
    /// maximum describe, and an ``RA`` axis that crosses zero makes
    /// such a box meaningless.
    ///
    /// Corners go through :meth:`pixel_to_world` in its batch form.
    /// A corner outside the projection's domain therefore fills its row
    /// with ``nan`` instead of raising. A wide-field ``SIN`` or ``AZP``
    /// image can put every corner outside that domain. The whole array
    /// is then ``nan``. Test the result with ``numpy.isfinite``. Pass
    /// one corner to :meth:`pixel_to_world` to read the reason it
    /// failed.
    ///
    /// ``WCSAXES`` may exceed ``NAXIS``. A coordinate axis past the end
    /// of :attr:`pixel_shape` then has no length to take a corner from.
    /// That axis holds its reference pixel for every corner. The corner
    /// count follows the image, and every corner still carries a full
    /// world vector.
    fn footprint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let flat = self.inner.footprint().into_py_result()?;
        let naxis = self.inner.naxis();
        let corners = flat.len().checked_div(naxis).unwrap_or(0);
        flat.into_pyarray(py).reshape([corners, naxis])
    }

    /// Forward transform pixel coordinates to world coordinates.
    ///
    /// Accepts one point or many. A length-``naxis`` sequence is one
    /// point and returns a list. An ``(N, naxis)`` array is ``N``
    /// points and returns an ``(N, naxis)`` array.
    ///
    /// Parameters
    /// ----------
    /// pix : array-like
    ///   Shape ``(naxis,)`` for a single point, or ``(N, naxis)`` for
    ///   a batch.
    /// origin : int, optional
    ///   ``0`` (default) treats ``pix`` as 0-based, ``1`` treats it as
    ///   1-based FITS coordinates.
    ///
    /// Returns
    /// -------
    /// list of float or numpy.ndarray
    ///   World coordinates with units given by :attr:`cunit`. A list
    ///   for a single point, an ``(N, naxis)`` array for a batch.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   In three cases:
    ///
    ///   - ``origin`` is neither ``0`` nor ``1``.
    ///   - ``pix`` is neither 1-D nor 2-D.
    ///   - A batch does not have exactly :attr:`naxis` columns. An
    ///     ``(naxis, N)`` array is the transpose of a batch, not a
    ///     batch of ``N`` points.
    /// FitsError
    ///   In three cases:
    ///
    ///   - A single point does not have :attr:`naxis` elements.
    ///   - The WCS has unresolved ``-TAB`` axes.
    ///   - ``pix`` falls outside the projection's valid domain. This
    ///     applies to a single point only.
    ///
    /// Notes
    /// -----
    /// The two forms differ on a point the WCS cannot transform. The
    /// single-point form raises, so the message names the reason. The
    /// batch form fills that point with ``nan`` and keeps going,
    /// because a wide field routinely mixes valid and invalid pixels.
    #[pyo3(signature = (pix, origin=0))]
    fn pixel_to_world<'py>(
        &self,
        py: Python<'py>,
        pix: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
        origin: u8,
    ) -> PyResult<Bound<'py, PyAny>> {
        let off = pixel_offset(origin)?;
        let view = pix.as_array();
        // `origin = 0` makes the shift a no-op, and a C-contiguous
        // array already holds the points in the layout the transform
        // wants. Borrow it in that case: a batch copy is a whole second
        // array, and it buys nothing here.
        //
        // Otherwise fall back. `iter()` walks an ndarray view in
        // logical row-major order whatever its memory layout, so a
        // sliced or non-contiguous input still arrives one whole point
        // at a time.
        let owned;
        let flat: &[f64] = if let Some(s) = view.as_slice().filter(|_| off == 0.0) {
            s
        } else {
            owned = view.iter().map(|p| p + off).collect::<Vec<f64>>();
            &owned
        };
        match *view.shape() {
            [_] => {
                let out = self.inner.pixel_to_world(flat).into_py_result()?;
                Ok(PyList::new(py, out)?.into_any())
            }
            [rows, cols] => {
                check_batch_cols("pixel_to_world", cols, self.inner.naxis())?;
                let out = self.inner.pixel_to_world_many(flat).into_py_result()?;
                Ok(out.into_pyarray(py).reshape([rows, cols])?.into_any())
            }
            ref s => Err(PyValueError::new_err(format!(
                "pixel_to_world: expected a 1-D or 2-D array, got {}-D",
                s.len()
            ))),
        }
    }

    /// Inverse transform world coordinates to pixel coordinates.
    ///
    /// Accepts one point or many, and mirrors :meth:`pixel_to_world`
    /// in both shape handling and error handling.
    ///
    /// Parameters
    /// ----------
    /// world : array-like
    ///   Shape ``(naxis,)`` for a single point, or ``(N, naxis)`` for
    ///   a batch.
    /// origin : int, optional
    ///   ``0`` (default) returns 0-based pixel coordinates,
    ///   ``1`` returns 1-based FITS coordinates.
    ///
    /// Returns
    /// -------
    /// list of float or numpy.ndarray
    ///   Pixel coordinates in the chosen origin. A list for a single
    ///   point, an ``(N, naxis)`` array for a batch.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   In three cases:
    ///
    ///   - ``origin`` is neither ``0`` nor ``1``.
    ///   - ``world`` is neither 1-D nor 2-D.
    ///   - A batch does not have exactly :attr:`naxis` columns.
    /// FitsError
    ///   In three cases:
    ///
    ///   - A single point does not have :attr:`naxis` elements.
    ///   - The WCS has unresolved ``-TAB`` axes.
    ///   - The inverse transform does not converge. This applies to a
    ///     single point only.
    ///
    /// Notes
    /// -----
    /// As in :meth:`pixel_to_world`, the batch form yields ``nan`` for
    /// a point that does not transform rather than raising.
    #[pyo3(signature = (world, origin=0))]
    fn world_to_pixel<'py>(
        &self,
        py: Python<'py>,
        world: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
        origin: u8,
    ) -> PyResult<Bound<'py, PyAny>> {
        let off = pixel_offset(origin)?;
        let view = world.as_array();
        // `origin` shifts the *result* here, not the input, so a
        // C-contiguous array is borrowed whatever the origin. See
        // `pixel_to_world` for the fallback.
        let owned;
        let flat: &[f64] = if let Some(s) = view.as_slice() {
            s
        } else {
            owned = view.iter().copied().collect::<Vec<f64>>();
            &owned
        };
        match *view.shape() {
            [_] => {
                let mut out = self.inner.world_to_pixel(flat).into_py_result()?;
                for p in &mut out {
                    *p -= off;
                }
                Ok(PyList::new(py, out)?.into_any())
            }
            [rows, cols] => {
                check_batch_cols("world_to_pixel", cols, self.inner.naxis())?;
                let mut out = self.inner.world_to_pixel_many(flat).into_py_result()?;
                for p in &mut out {
                    *p -= off;
                }
                Ok(out.into_pyarray(py).reshape([rows, cols])?.into_any())
            }
            ref s => Err(PyValueError::new_err(format!(
                "world_to_pixel: expected a 1-D or 2-D array, got {}-D",
                s.len()
            ))),
        }
    }

    /// Local pixel scale at ``(px, py)``.
    ///
    /// Parameters
    /// ----------
    /// px, py : float
    ///   Pixel coordinates.
    /// origin : int, optional
    ///   ``0`` (default) treats inputs as 0-based; ``1`` as
    ///   1-based FITS coordinates.
    ///
    /// Returns
    /// -------
    /// tuple of float
    ///   Pixel scale in arcseconds per pixel along the two
    ///   celestial axes.
    ///
    /// Raises
    /// ------
    /// FitsError
    ///   If the WCS has no celestial axis pair, if it has
    ///   unresolved ``-TAB`` axes, or if ``(px, py)`` or an
    ///   adjacent pixel used for the finite difference falls
    ///   outside the projection's valid domain.
    ///
    /// Notes
    /// -----
    /// fitsy measures this by finite difference on the sphere, so
    /// the result includes projection distortion and any local
    /// skew. It is a great-circle distance, always positive, not
    /// the signed ``CDELT`` value. An image with flipped RA still
    /// reports a positive scale.
    #[pyo3(signature = (px, py, origin=0))]
    fn pixel_scale_at(&self, px: f64, py: f64, origin: u8) -> PyResult<(f64, f64)> {
        let off = pixel_offset(origin)?;
        self.inner
            .pixel_scale_at(px + off, py + off)
            .into_py_result()
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
    ///   ``' '`` (default) for the primary description, or
    ///   ``'A'`` through ``'Z'`` for an alternate.
    ///
    /// Returns
    /// -------
    /// Header
    ///   A new header holding every WCS keyword fitsy's reader
    ///   understands: the linear pipeline, ``LONPOLE``/``LATPOLE``
    ///   and the projection's ``PV`` parameters, SIP, TPV, TNX/ZPX,
    ///   DSS plate solutions, spectral rest quantities, and the
    ///   ``-TAB`` pointer cards.
    ///
    /// Raises
    /// ------
    /// FitsError
    ///   If ``alt`` is not ``' '`` or ``'A'``-``'Z'``.
    ///
    /// Notes
    /// -----
    /// Parsing the returned header reproduces this WCS. Two things a
    /// bare header cannot carry:
    ///
    /// - ``NAXISn`` are written as zero placeholders, because a WCS
    ///   carries no image dimensions. Merge the result into a
    ///   header that already has the real values.
    /// - A ``-TAB`` axis names its lookup table by ``EXTNAME``. That
    ///   BINTABLE is a separate HDU and must be written alongside
    ///   this header.
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
///   The fitted world coordinate system.
/// rms_arcsec : float
///   Root-mean-square residual across all reference points (arcsec).
/// max_arcsec : float
///   Largest single-point residual (arcsec).
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
    /// Per-point residuals as a numpy array of shape ``(N, 2)``,
    /// holding ``(delta_alpha * cos(delta), delta_dec)`` in
    /// arcseconds.
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
/// pixels : array-like
///   Shape ``(N, 2)`` array-like of pixel coordinates (numpy
///   array, list of lists, tuple of tuples, etc.).
/// sky : array-like
///   Shape ``(N, 2)`` array-like of ``(ra, dec)`` in degrees, or
///   more generally ``(lon, lat)`` in the chosen ``frame``
///   (numpy array, list of lists, tuple of tuples, etc.).
/// projection : str, optional
///   Three-letter projection code (Paper II Table 13),
///   case-insensitive. One of
///   ``AZP``, ``SZP``, ``TAN``, ``STG``, ``SIN``, ``ARC``, ``ZPN``,
///   ``ZEA``, ``AIR``, ``CYP``, ``CEA``, ``CAR``, ``MER``, ``SFL``,
///   ``PAR``, ``MOL``, ``AIT``, ``COP``, ``COE``, ``COD``, ``COO``,
///   ``BON``, ``PCO``, ``TSC``, ``CSC``, ``QSC``, ``HPX`` or
///   ``XPH``. Default ``"TAN"``.
/// crpix : tuple of float, optional
///   Pin the reference pixel. Interpreted in the same ``origin`` as
///   ``pixels``. Default ``None``, which solves for the reference
///   pixel as part of the fit.
/// crval : tuple of float, optional
///   Pin the tangent point, in degrees. Default ``None``, which
///   uses the spherical centroid of the sky points.
/// sip_order : int, optional
///   Order of a SIP polynomial distortion fit. Valid range is 2 to
///   9. Default ``None``, which fits no SIP distortion. Orders 0
///   and 1 are rejected, because ``CRPIX`` and the ``CD`` matrix
///   already absorb those terms.
/// fit_sip_inverse : bool, optional
///   When ``sip_order`` is given, also fit the AP/BP inverse
///   polynomial. Default ``True``.
/// frame : str, optional
///   Celestial frame for the sky coordinates, case-insensitive.
///   ``'equatorial'`` (default), ``'icrs'``, ``'fk5'`` and
///   ``'fk4'`` are synonyms; all four emit the ``RA--``/``DEC-``
///   CTYPE prefixes. The other accepted values are
///   ``'galactic'``, ``'ecliptic'``, ``'supergalactic'`` and
///   ``'helioecliptic'``.
/// origin : int, optional
///   ``0`` (default, numpy/C convention) treats ``pixels`` and
///   ``crpix`` as 0-based; ``1`` treats them as 1-based FITS
///   coordinates. The fitted WCS itself always carries 1-based
///   ``CRPIX`` values per the FITS standard.
///
/// Returns
/// -------
/// WcsFit
///   Fitted WCS, residuals, and summary statistics.
///
/// Raises
/// ------
/// ValueError
///   If ``pixels`` or ``sky`` is not an ``(N, 2)`` array, if they
///   have different numbers of rows, or if ``frame`` names none of
///   the values above.
/// FitsError
///   If ``projection`` names none of the codes above, if fewer
///   than 2 points are given (or fewer than 3 without a pinned
///   ``crpix``), or if the fit is ill-conditioned, for example from
///   collinear points.
///
/// Examples
/// --------
/// >>> import numpy as np, fitsy
/// >>> pix = np.array([[100.0, 100.0], [200.0, 100.0], [100.0, 200.0]])
/// >>> sky = np.array([[10.00, -5.00], [10.05, -5.00], [10.00, -4.95]])
/// >>> fit = fitsy.fit_wcs(pix, sky, projection="TAN")
/// >>> fit.rms_arcsec < 1e-6
/// True
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
pub fn fit_wcs<'py>(
    py: Python<'py>,
    pixels: PyArrayLike2<'py, f64, AllowTypeChange>,
    sky: PyArrayLike2<'py, f64, AllowTypeChange>,
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

    let proj = crate::wcs::projection::Projection::from_code(projection, &[]).into_py_result()?;
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
        projection: proj,
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
