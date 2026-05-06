//! Fit a celestial WCS from `(pixel, sky)` reference correspondences.
//!
//! The algorithm inverts the standard WCS pipeline (Greisen &
//! Calabretta 2002, Paper I Sec.8):
//!
//! 1. Choose a tangent point `CRVAL` (user-supplied, or the spherical
//!    centroid of the sky points).
//! 2. De-project each sky point to intermediate world coords
//!    `(xi_i, eta_i)` in degrees on the projection plane.
//! 3. Linear least-squares for the CD matrix (and CRPIX, when free)
//!    against the observed pixel coordinates.
//! 4. Optionally fit SIP polynomial distortion to the residuals.
//! 5. Compose a [`Wcs`] and report per-point and RMS residuals
//!    (round-tripped through the fitted model, in arcseconds).
//!
//! The solver is a self-contained Householder QR (no external linalg
//! dependency); it handles the small dense systems (<= ~120 unknowns
//! for an order-9 SIP fit) we encounter in practice.

use crate::error::{FitsError, Result};
use crate::wcs::Wcs;
use crate::wcs::celestial::{CelestialFrame, CelestialRotation, RadeSys};
use crate::wcs::celestial_block::{CelestialBlock, CelestialPair};
use crate::wcs::linear::LinearTransform;
use crate::wcs::projection::{self, Projection, ProjectionKind};
use crate::wcs::sip::{SIP_MAX_ORDER, Sip, SipPoly};
use crate::wcs::{D2R, R2D};

/// User-supplied configuration for [`fit_celestial_wcs`].
#[derive(Debug, Clone)]
pub struct WcsFitOptions {
    /// Projection code (`TAN` is the usual choice).
    pub projection: ProjectionKind,
    /// Optional fixed reference pixel `(CRPIX1, CRPIX2)` in **0-based**
    /// pixel coordinates (numpy / C convention -- see
    /// [`crate::Wcs::pixel_to_world`] for the rationale). When `None`,
    /// CRPIX is solved as part of the
    /// linear fit.
    pub crpix: Option<(f64, f64)>,
    /// Optional fixed tangent point `(CRVAL1, CRVAL2)` in degrees.
    /// When `None`, the spherical centroid of the sky points is used.
    pub crval: Option<(f64, f64)>,
    /// Output frame for the celestial axes (sets `CTYPE` prefix and
    /// `RADESYS`). Defaults to ICRS.
    pub frame: CelestialFrame,
    /// SIP polynomial order (`p + q <= order`). `None` skips SIP.
    /// Order 0/1 are rejected -- those terms are absorbed by CRPIX/CD
    /// per Shupe et al. (2005) Sec.3.
    pub sip_order: Option<u32>,
    /// When fitting SIP, also fit `AP`/`BP` (the inverse polynomial)
    /// using a separate least-squares pass on the inverted points.
    /// Recommended; the inverse iteration is much faster when AP/BP
    /// give a good initial guess.
    pub fit_sip_inverse: bool,
}

impl Default for WcsFitOptions {
    fn default() -> Self {
        Self {
            projection: ProjectionKind::Tan,
            crpix: None,
            crval: None,
            frame: CelestialFrame::Equatorial,
            sip_order: None,
            fit_sip_inverse: true,
        }
    }
}

/// Result of [`fit_celestial_wcs`].
#[derive(Debug)]
pub struct WcsFit {
    /// Fitted WCS, ready for `pixel_to_celestial` / `celestial_to_pixel`.
    pub wcs: Wcs,
    /// Per-point residual `(delta_alpha*cos delta, delta_dec)` in **arcseconds**, computed
    /// by mapping each input pixel through the fitted WCS and
    /// comparing to the input sky coord. Index aligned with the
    /// inputs.
    pub residuals_arcsec: Vec<(f64, f64)>,
    /// RMS of `sqrt(delta_alpha^2*cos^2delta + delta_dec^2)` across all points, arcsec.
    pub rms_arcsec: f64,
    /// Maximum per-point residual magnitude, arcsec.
    pub max_arcsec: f64,
}

/// Fit a celestial WCS to a set of `(pixel, sky)` correspondences.
///
/// `pixels` are **0-based** pixel coordinates (numpy / C convention,
/// matching [`crate::Wcs::pixel_to_world`]). `sky` are
/// `(RA, Dec)` in degrees (or, more generally, `(lon, lat)` in the
/// chosen [`WcsFitOptions::frame`]).
///
/// At least 3 points are required for a free-CRPIX linear fit
/// (6 unknowns / 2 components per point); 2 suffice when CRPIX is
/// fixed. SIP fits need enough additional points to over-determine
/// the polynomial: a forward fit of order `n` introduces
/// `n(n+3)/2 = (n+1)(n+2)/2 - 3` extra parameters per axis, so at
/// least that many *additional* points are required for a sensible
/// fit. The function does not enforce a strict minimum beyond what
/// the linear solver tolerates.
pub fn fit_celestial_wcs(
    pixels: &[(f64, f64)],
    sky: &[(f64, f64)],
    opts: &WcsFitOptions,
) -> Result<WcsFit> {
    if pixels.len() != sky.len() {
        return Err(FitsError::Wcs(format!(
            "fit_celestial_wcs: pixel/sky length mismatch ({} vs {})",
            pixels.len(),
            sky.len()
        )));
    }
    let n = pixels.len();
    if n < 3 && opts.crpix.is_none() {
        return Err(FitsError::Wcs(
            "fit_celestial_wcs: need >=3 points (or pin CRPIX) for a linear fit".into(),
        ));
    }
    if n < 2 {
        return Err(FitsError::Wcs(
            "fit_celestial_wcs: need >=2 points even with CRPIX pinned".into(),
        ));
    }

    // 0-based -> 1-based: the rest of this function (and the CRPIX
    // values it ultimately writes into the header) work in the FITS
    // 1-based pixel convention. See `Wcs::pixel_to_world` doc.
    let pixels_1based: Vec<(f64, f64)> = pixels.iter().map(|(x, y)| (x + 1.0, y + 1.0)).collect();
    let pixels_0based = pixels;
    let pixels = pixels_1based.as_slice();
    let opts_crpix = opts.crpix.map(|(x, y)| (x + 1.0, y + 1.0));

    // 1. Tangent point.
    let (alpha0, delta0) = match opts.crval {
        Some(p) => p,
        None => spherical_centroid(sky),
    };

    // 2. De-project sky -> intermediate world (xi, eta) in degrees.
    let projection = projection::build(opts.projection, &[])?;
    let theta0 = projection.theta0();
    let rotation = CelestialRotation::new(alpha0, delta0, None, None, theta0)?;
    let mut iw: Vec<(f64, f64)> = Vec::with_capacity(n);
    for &(ra, dec) in sky {
        let (phi, theta) = rotation.celestial_to_native(ra, dec);
        let (xi, eta) = projection.s2x(phi, theta)?;
        iw.push((xi, eta));
    }

    // 3. Linear LS for CD matrix (and CRPIX when free).
    let (cd, crpix1, crpix2) = match opts_crpix {
        Some((cx, cy)) => {
            let cd = fit_cd_fixed_crpix(pixels, &iw, cx, cy)?;
            (cd, cx, cy)
        }
        None => fit_cd_with_crpix(pixels, &iw)?,
    };

    // 4. Optional SIP fit.
    let sip = match opts.sip_order {
        Some(order) => Some(fit_sip(
            pixels,
            &iw,
            &cd,
            crpix1,
            crpix2,
            order,
            opts.fit_sip_inverse,
        )?),
        None => None,
    };

    // 5. Assemble the Wcs.
    let wcs = build_wcs(
        opts.projection,
        opts.frame,
        crpix1,
        crpix2,
        cd,
        alpha0,
        delta0,
        sip,
        rotation,
        projection,
    )?;

    // Residuals (round-tripped through the final Wcs).
    // Use the caller-provided (0-based) pixels here; `Wcs::pixel_to_celestial`
    // is itself 0-based so we must not feed it the shifted version.
    let mut residuals = Vec::with_capacity(n);
    let mut sumsq = 0.0_f64;
    let mut max_r = 0.0_f64;
    for (i, &(px, py)) in pixels_0based.iter().enumerate() {
        let (ra_pred, dec_pred) = wcs.pixel_to_celestial(px, py)?;
        let (ra_obs, dec_obs) = sky[i];
        let cosd = dec_obs.to_radians().cos();
        let dra = wrap_lon_deg(ra_pred - ra_obs) * cosd * 3600.0;
        let dde = (dec_pred - dec_obs) * 3600.0;
        let r2 = dra * dra + dde * dde;
        sumsq += r2;
        if r2 > max_r {
            max_r = r2;
        }
        residuals.push((dra, dde));
    }
    let rms = (sumsq / n as f64).sqrt();

    Ok(WcsFit {
        wcs,
        residuals_arcsec: residuals,
        rms_arcsec: rms,
        max_arcsec: max_r.sqrt(),
    })
}

/// Spherical centroid of `(ra, dec)` points in degrees. Computes the
/// mean unit vector and converts back, which is well-defined even
/// when the inputs straddle the RA=0/360 wrap.
fn spherical_centroid(sky: &[(f64, f64)]) -> (f64, f64) {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;
    for &(ra, dec) in sky {
        let r = ra * D2R;
        let d = dec * D2R;
        let cd = d.cos();
        x += cd * r.cos();
        y += cd * r.sin();
        z += d.sin();
    }
    let mut ra = y.atan2(x) * R2D;
    if ra < 0.0 {
        ra += 360.0;
    }
    let dec = z.atan2((x * x + y * y).sqrt()) * R2D;
    (ra, dec)
}

/// Wrap a longitude difference into `(-180, 180]`.
fn wrap_lon_deg(d: f64) -> f64 {
    let mut x = d % 360.0;
    if x > 180.0 {
        x -= 360.0;
    } else if x <= -180.0 {
        x += 360.0;
    }
    x
}

/// Solve, by LS, the affine map
/// `xi = a x + b y + e`, `eta = c x + d y + f`. Then back out CRPIX
/// from the offsets so the final form is `(xi, eta) = CD * (x - c1, y - c2)`.
fn fit_cd_with_crpix(pixels: &[(f64, f64)], iw: &[(f64, f64)]) -> Result<([f64; 4], f64, f64)> {
    let n = pixels.len();
    // Stack: rows = 2N, cols = 6. Unknowns = [a, b, e, c, d, f].
    let m = 2 * n;
    let cols = 6;
    let mut mat = vec![0.0_f64; m * cols];
    let mut rhs = vec![0.0_f64; m];
    for (i, (&(x, y), &(xi, eta))) in pixels.iter().zip(iw.iter()).enumerate() {
        // xi row: a*x + b*y + 1*e + 0*c + 0*d + 0*f = xi
        let r1 = 2 * i;
        mat[r1 * cols] = x;
        mat[r1 * cols + 1] = y;
        mat[r1 * cols + 2] = 1.0;
        rhs[r1] = xi;
        // eta row: 0+0+0 + c*x + d*y + 1*f = eta
        let r2 = 2 * i + 1;
        mat[r2 * cols + 3] = x;
        mat[r2 * cols + 4] = y;
        mat[r2 * cols + 5] = 1.0;
        rhs[r2] = eta;
    }
    let beta = lstsq_qr(&mut mat, &mut rhs, m, cols)?;
    let (a, b, e, c, d, f) = (beta[0], beta[1], beta[2], beta[3], beta[4], beta[5]);
    // Back out CRPIX from (a*c1 + b*c2 = -e, c*c1 + d*c2 = -f), i.e.
    // CD * (c1, c2) = -(e, f). (Matching xi = a(x-c1)+b(y-c2) => e = -a*c1 - b*c2.)
    let det = a * d - b * c;
    if det.abs() < 1e-30 {
        return Err(FitsError::Wcs(
            "fit_celestial_wcs: degenerate CD matrix; reference points may be collinear".into(),
        ));
    }
    let crpix1 = (b * f - d * e) / det;
    let crpix2 = (c * e - a * f) / det;
    Ok(([a, b, c, d], crpix1, crpix2))
}

/// LS for the 2x2 CD matrix at fixed CRPIX. Solves
/// `(xi, eta) = CD * (x - cx, y - cy)`.
fn fit_cd_fixed_crpix(
    pixels: &[(f64, f64)],
    iw: &[(f64, f64)],
    cx: f64,
    cy: f64,
) -> Result<[f64; 4]> {
    let n = pixels.len();
    let m = 2 * n;
    let cols = 4;
    let mut mat = vec![0.0_f64; m * cols];
    let mut rhs = vec![0.0_f64; m];
    for (i, (&(x, y), &(xi, eta))) in pixels.iter().zip(iw.iter()).enumerate() {
        let u = x - cx;
        let v = y - cy;
        let r1 = 2 * i;
        mat[r1 * cols] = u;
        mat[r1 * cols + 1] = v;
        rhs[r1] = xi;
        let r2 = 2 * i + 1;
        mat[r2 * cols + 2] = u;
        mat[r2 * cols + 3] = v;
        rhs[r2] = eta;
    }
    let beta = lstsq_qr(&mut mat, &mut rhs, m, cols)?;
    Ok([beta[0], beta[1], beta[2], beta[3]])
}

/// Fit SIP A/B (and optionally AP/BP) of the given order.
///
/// Forward problem: with `(u, v) = (x - CRPIX, y - CRPIX2)` and
/// `(u', v') = CD^-1 * (xi, eta)` (the "ideal" undistorted pixel offsets
/// implied by the sky), the SIP convention says
/// `u' = u + A(u, v)`, `v' = v + B(u, v)` with `A`, `B` polynomials
/// of order `>= 2`. Predictors are the monomials `u^p * v^q` with
/// `2 <= p + q <= order`; targets are `u' - u` and `v' - v`.
fn fit_sip(
    pixels: &[(f64, f64)],
    iw: &[(f64, f64)],
    cd: &[f64; 4],
    cx: f64,
    cy: f64,
    order: u32,
    fit_inverse: bool,
) -> Result<Sip> {
    if order < 2 {
        return Err(FitsError::Wcs(format!(
            "SIP fit: order {order} too low (constant/linear absorbed by CRPIX/CD)"
        )));
    }
    if order > SIP_MAX_ORDER {
        return Err(FitsError::Wcs(format!(
            "SIP fit: order {order} exceeds SIP_MAX_ORDER ({SIP_MAX_ORDER})"
        )));
    }
    let n = pixels.len();

    // CD inverse for going (xi, eta) -> (u', v').
    let det = cd[0] * cd[3] - cd[1] * cd[2];
    if det.abs() < 1e-30 {
        return Err(FitsError::Wcs(
            "SIP fit: CD matrix is singular; cannot invert".into(),
        ));
    }
    let inv = [cd[3] / det, -cd[1] / det, -cd[2] / det, cd[0] / det];

    // Collect (u, v, u'-u, v'-v) per point.
    let mut samples: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(n);
    for (&(x, y), &(xi, eta)) in pixels.iter().zip(iw.iter()) {
        let u = x - cx;
        let v = y - cy;
        let up = inv[0] * xi + inv[1] * eta;
        let vp = inv[2] * xi + inv[3] * eta;
        samples.push((u, v, up - u, vp - v));
    }

    // Predictor terms: (p, q) with 2 <= p+q <= order, in row-major
    // order on (p, q). We fit A and B independently against the
    // same design matrix.
    let terms: Vec<(u32, u32)> = (0..=order)
        .flat_map(|p| (0..=order - p).map(move |q| (p, q)))
        .filter(|(p, q)| p + q >= 2)
        .collect();
    let k = terms.len();
    let m = n;

    // Build design matrix once.
    let mut design = vec![0.0_f64; m * k];
    for (i, &(u, v, _, _)) in samples.iter().enumerate() {
        for (j, &(p, q)) in terms.iter().enumerate() {
            design[i * k + j] = pow_u32(u, p) * pow_u32(v, q);
        }
    }

    let mut mat_a = design.clone();
    let mut rhs_a: Vec<f64> = samples.iter().map(|s| s.2).collect();
    let beta_a = lstsq_qr(&mut mat_a, &mut rhs_a, m, k)?;

    let mut mat_b = design;
    let mut rhs_b: Vec<f64> = samples.iter().map(|s| s.3).collect();
    let beta_b = lstsq_qr(&mut mat_b, &mut rhs_b, m, k)?;

    let a_terms: Vec<(u32, u32, f64)> = terms
        .iter()
        .zip(beta_a.iter())
        .map(|(&(p, q), &c)| (p, q, c))
        .collect();
    let b_terms: Vec<(u32, u32, f64)> = terms
        .iter()
        .zip(beta_b.iter())
        .map(|(&(p, q), &c)| (p, q, c))
        .collect();

    let a = SipPoly::from_terms(order, &a_terms)?;
    let b = SipPoly::from_terms(order, &b_terms)?;

    let (ap, bp) = if fit_inverse {
        // Inverse fit: predictors at (u', v'), targets are (u - u', v - v').
        let mut design_inv = vec![0.0_f64; m * k];
        for (i, &(u, v, du, dv)) in samples.iter().enumerate() {
            let up = u + du;
            let vp = v + dv;
            for (j, &(p, q)) in terms.iter().enumerate() {
                design_inv[i * k + j] = pow_u32(up, p) * pow_u32(vp, q);
            }
        }
        let mut mat_ap = design_inv.clone();
        let mut rhs_ap: Vec<f64> = samples.iter().map(|s| -s.2).collect();
        let beta_ap = lstsq_qr(&mut mat_ap, &mut rhs_ap, m, k)?;
        let mut mat_bp = design_inv;
        let mut rhs_bp: Vec<f64> = samples.iter().map(|s| -s.3).collect();
        let beta_bp = lstsq_qr(&mut mat_bp, &mut rhs_bp, m, k)?;
        let ap_terms: Vec<(u32, u32, f64)> = terms
            .iter()
            .zip(beta_ap.iter())
            .map(|(&(p, q), &c)| (p, q, c))
            .collect();
        let bp_terms: Vec<(u32, u32, f64)> = terms
            .iter()
            .zip(beta_bp.iter())
            .map(|(&(p, q), &c)| (p, q, c))
            .collect();
        (
            Some(SipPoly::from_terms(order, &ap_terms)?),
            Some(SipPoly::from_terms(order, &bp_terms)?),
        )
    } else {
        (None, None)
    };

    Ok(Sip { a, b, ap, bp })
}

/// Integer-exponent power that avoids `f64::powi`'s sign-handling
/// hot path for small exponents (which dominate SIP fits).
fn pow_u32(x: f64, p: u32) -> f64 {
    match p {
        0 => 1.0,
        1 => x,
        2 => x * x,
        3 => x * x * x,
        _ => x.powi(p as i32),
    }
}

/// Assemble the final [`Wcs`] from the fitted pieces.
#[allow(
    clippy::too_many_arguments,
    reason = "all parameters are required to assemble the WCS from its fitted components"
)]
fn build_wcs(
    proj_kind: ProjectionKind,
    frame: CelestialFrame,
    crpix1: f64,
    crpix2: f64,
    cd: [f64; 4],
    alpha0: f64,
    delta0: f64,
    sip: Option<Sip>,
    rotation: CelestialRotation,
    projection: Box<dyn Projection>,
) -> Result<Wcs> {
    let proj_code = ProjectionCode::for_kind(proj_kind);
    let suffix = if sip.is_some() { "-SIP" } else { "" };
    let (lon_prefix, lat_prefix) = ctype_prefixes(frame);
    let ctype1 = format!("{lon_prefix}-{}{}", proj_code.0, suffix);
    let ctype2 = format!("{lat_prefix}-{}{}", proj_code.0, suffix);
    // Pad to 8 chars per FITS string convention used elsewhere.
    let pad = |s: String| {
        let mut t = s;
        while t.len() < 8 {
            t.push(' ');
        }
        t
    };
    let ctype = vec![pad(ctype1), pad(ctype2)];
    let cunit = vec!["deg".into(), "deg".into()];
    let crval = vec![alpha0, delta0];
    let linear = LinearTransform::from_cd(vec![crpix1, crpix2], cd.to_vec())?;

    let pair = CelestialPair {
        lon: 0,
        lat: 1,
        frame,
    };
    let celestial = Some(CelestialBlock {
        pair,
        projection,
        rotation,
        sip,
        tpv: None,
        tnx: None,
    });

    Ok(Wcs {
        naxis: 2,
        linear,
        ctype,
        cunit,
        crval,
        celestial,
        spectral: Vec::new(),
        radesys: match frame {
            CelestialFrame::Equatorial => RadeSys::Icrs,
            _ => RadeSys::Other,
        },
        equinox: None,
        mjd_obs: None,
        wcsname: None,
        specsys: None,
        ssysobs: None,
        velosys: None,
        dss: None,
        tab_specs: Vec::new(),
        tab: Vec::new(),
    })
}

struct ProjectionCode(&'static str);

fn ctype_prefixes(frame: CelestialFrame) -> (&'static str, &'static str) {
    match frame {
        CelestialFrame::Equatorial => ("RA--", "DEC-"),
        CelestialFrame::Galactic => ("GLON", "GLAT"),
        CelestialFrame::Ecliptic => ("ELON", "ELAT"),
        CelestialFrame::Supergalactic => ("SLON", "SLAT"),
        CelestialFrame::HelioEcliptic => ("HLON", "HLAT"),
        CelestialFrame::Other => ("XLON", "XLAT"),
    }
}

impl ProjectionCode {
    fn for_kind(kind: ProjectionKind) -> Self {
        Self(match kind {
            ProjectionKind::Tan => "TAN",
            ProjectionKind::Stg => "STG",
            ProjectionKind::Sin => "SIN",
            ProjectionKind::Arc => "ARC",
            ProjectionKind::Zea => "ZEA",
            ProjectionKind::Zpn => "ZPN",
            ProjectionKind::Azp => "AZP",
            ProjectionKind::Szp => "SZP",
            ProjectionKind::Air => "AIR",
            ProjectionKind::Cyp => "CYP",
            ProjectionKind::Cea => "CEA",
            ProjectionKind::Car => "CAR",
            ProjectionKind::Mer => "MER",
            ProjectionKind::Sfl => "SFL",
            ProjectionKind::Par => "PAR",
            ProjectionKind::Mol => "MOL",
            ProjectionKind::Ait => "AIT",
            ProjectionKind::Cop => "COP",
            ProjectionKind::Coe => "COE",
            ProjectionKind::Cod => "COD",
            ProjectionKind::Coo => "COO",
            ProjectionKind::Bon => "BON",
            ProjectionKind::Pco => "PCO",
            ProjectionKind::Tsc => "TSC",
            ProjectionKind::Csc => "CSC",
            ProjectionKind::Qsc => "QSC",
            ProjectionKind::Hpx => "HPX",
            ProjectionKind::Xph => "XPH",
        })
    }
}

// ---- Self-contained least-squares solver -------------------------------

/// Solve `min ||A * beta - b||^2` for `A` (`m x n`, `m >= n`) by Householder
/// QR. `mat` is `m x n` row-major and is overwritten; `rhs` (length
/// `m`) is overwritten with `Q^T b`. Returns `beta` (length `n`).
///
/// We hand-roll this rather than pulling in a dep -- the systems
/// involved (<= ~100 unknowns) are tiny, and keeping the solver in
/// the crate avoids gating WCS fitting on the optional `nalgebra`
/// feature.
fn lstsq_qr(mat: &mut [f64], rhs: &mut [f64], m: usize, n: usize) -> Result<Vec<f64>> {
    if m < n {
        return Err(FitsError::Wcs(format!(
            "least-squares: under-determined system ({m} equations, {n} unknowns)"
        )));
    }
    let idx = |r: usize, c: usize| r * n + c;

    for k in 0..n {
        // Compute the Householder vector for column k below the diagonal.
        let mut norm2 = 0.0_f64;
        for i in k..m {
            norm2 += mat[idx(i, k)] * mat[idx(i, k)];
        }
        let norm = norm2.sqrt();
        if norm < 1e-30 {
            return Err(FitsError::Wcs(format!(
                "least-squares: rank-deficient column {k} (norm ~= 0); inputs may be collinear"
            )));
        }
        // Sign chosen to avoid catastrophic cancellation.
        let alpha = if mat[idx(k, k)] >= 0.0 { -norm } else { norm };
        // v = x - alpha e_k, but we store x[k] - alpha in mat[k,k].
        let mut v = vec![0.0_f64; m - k];
        v[0] = mat[idx(k, k)] - alpha;
        for i in 1..(m - k) {
            v[i] = mat[idx(k + i, k)];
        }
        let v_norm2 = v.iter().map(|x| x * x).sum::<f64>();
        if v_norm2 < 1e-300 {
            // Already in upper-triangular shape; nothing to do.
            mat[idx(k, k)] = alpha;
            continue;
        }
        let two_over_v2 = 2.0 / v_norm2;

        // Apply H = I - (2/v^2) v v^T to remaining columns of mat and to rhs.
        for j in k..n {
            let mut dot = 0.0_f64;
            for i in 0..(m - k) {
                dot += v[i] * mat[idx(k + i, j)];
            }
            let f = dot * two_over_v2;
            for i in 0..(m - k) {
                mat[idx(k + i, j)] -= f * v[i];
            }
        }
        let mut dot = 0.0_f64;
        for i in 0..(m - k) {
            dot += v[i] * rhs[k + i];
        }
        let f = dot * two_over_v2;
        for i in 0..(m - k) {
            rhs[k + i] -= f * v[i];
        }

        // Place alpha on the diagonal explicitly (we overwrote it via H above
        // implicitly, but the simpler algebra is to set it here and zero out
        // the sub-diagonal). The above multiplication has set
        // mat[k..m, k] = -alpha * e1; just fix mat[k,k] for the back-substitution.
        mat[idx(k, k)] = alpha;
        for i in (k + 1)..m {
            mat[idx(i, k)] = 0.0;
        }
    }

    // Back-substitution on the upper-triangular block.
    let mut beta = vec![0.0_f64; n];
    for k in (0..n).rev() {
        let mut s = rhs[k];
        for j in (k + 1)..n {
            s -= mat[idx(k, j)] * beta[j];
        }
        let d = mat[idx(k, k)];
        if d.abs() < 1e-30 {
            return Err(FitsError::Wcs(format!(
                "least-squares: zero pivot at row {k}; inputs may be degenerate"
            )));
        }
        beta[k] = s / d;
    }
    Ok(beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a known TAN WCS, sample a grid, and verify the fit
    /// recovers it to high precision.
    #[test]
    fn fit_recovers_known_tan_wcs() {
        let truth_header = [("CTYPE1", "RA---TAN"), ("CTYPE2", "DEC--TAN")];
        // Suppress unused-variable warning; the array is here for documentation.
        let _ = truth_header;
        // CRPIX = (100, 80), CRVAL = (123.4, +5.6), 0.5"/pix scale,
        // 30deg rotation.
        let crpix = (100.0_f64, 80.0_f64);
        let crval = (123.4_f64, 5.6_f64);
        // 0.5 arcsec/pixel in degrees.
        let scale = 0.5 / 3600.0;
        let theta = 30.0_f64.to_radians();
        let cd = [
            -scale * theta.cos(),
            scale * theta.sin(),
            scale * theta.sin(),
            scale * theta.cos(),
        ];

        let truth = synthesize_wcs(crpix, crval, cd, None);

        // Sample a 5x5 grid.
        let mut pixels = Vec::new();
        let mut sky = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                let px = 20.0 + 40.0 * f64::from(i);
                let py = 15.0 + 30.0 * f64::from(j);
                let (ra, dec) = truth.pixel_to_celestial(px, py).unwrap();
                pixels.push((px, py));
                sky.push((ra, dec));
            }
        }

        let fit = fit_celestial_wcs(&pixels, &sky, &WcsFitOptions::default()).unwrap();
        // Sub-microarcsec for clean synthetic input.
        assert!(fit.rms_arcsec < 1e-6, "rms = {} arcsec", fit.rms_arcsec);
        // CRVAL/CRPIX won't bit-equal the truth (we re-tangent at
        // the spherical centroid, not the input CRVAL), but the
        // recovered model must reproduce the original mapping to
        // sub-microarcsec -- the rms check above already enforces it.
        let _ = (crpix, crval);
    }

    #[test]
    fn fit_with_fixed_crpix_and_crval() {
        // The header-level CRPIX1/2 (FITS 1-based).
        let crpix_fits = (200.0_f64, 200.0_f64);
        // Same point in the 0-based pixel convention used by the
        // public Wcs / fit_celestial_wcs APIs.
        let crpix_0based = (crpix_fits.0 - 1.0, crpix_fits.1 - 1.0);
        let crval = (45.0_f64, 30.0_f64);
        let scale = 0.2 / 3600.0;
        let cd = [-scale, 0.0, 0.0, scale];
        let truth = synthesize_wcs(crpix_fits, crval, cd, None);
        let mut pixels = Vec::new();
        let mut sky = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let px = 50.0 + 70.0 * f64::from(i);
                let py = 60.0 + 50.0 * f64::from(j);
                let (ra, dec) = truth.pixel_to_celestial(px, py).unwrap();
                pixels.push((px, py));
                sky.push((ra, dec));
            }
        }
        let opts = WcsFitOptions {
            crpix: Some(crpix_0based),
            crval: Some(crval),
            ..Default::default()
        };
        let fit = fit_celestial_wcs(&pixels, &sky, &opts).unwrap();
        assert!(fit.rms_arcsec < 1e-6);
        // `linear.crpix()` returns the FITS 1-based stored value.
        assert!((fit.wcs.linear.crpix()[0] - crpix_fits.0).abs() < 1e-12);
        assert!((fit.wcs.linear.crpix()[1] - crpix_fits.1).abs() < 1e-12);
    }

    #[test]
    fn fit_recovers_sip_distortion() {
        let crpix = (100.0_f64, 100.0_f64);
        let crval = (10.0_f64, 20.0_f64);
        let scale = 0.3 / 3600.0;
        let cd = [scale, 0.0, 0.0, scale];
        // Fabricate a small order-2 SIP distortion.
        let a = SipPoly::from_terms(2, &[(2, 0, 1e-6), (1, 1, -2e-6), (0, 2, 5e-7)]).unwrap();
        let b = SipPoly::from_terms(2, &[(2, 0, -3e-7), (1, 1, 4e-6), (0, 2, -1e-6)]).unwrap();
        let sip_truth = Sip {
            a,
            b,
            ap: None,
            bp: None,
        };

        let truth = synthesize_wcs(crpix, crval, cd, Some(sip_truth));
        // Need lots of points spread over the chip for a stable
        // polynomial fit.
        let mut pixels = Vec::new();
        let mut sky = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                let px = 10.0 + 18.0 * f64::from(i);
                let py = 10.0 + 18.0 * f64::from(j);
                let (ra, dec) = truth.pixel_to_celestial(px, py).unwrap();
                pixels.push((px, py));
                sky.push((ra, dec));
            }
        }
        let opts = WcsFitOptions {
            sip_order: Some(2),
            ..Default::default()
        };
        let fit = fit_celestial_wcs(&pixels, &sky, &opts).unwrap();
        // SIP recovery on synthetic data. The fit re-tangents and
        // also has slight numerical noise from the polynomial LS;
        // ~10 milli-arcsec is comfortably below astrometric needs.
        assert!(
            fit.rms_arcsec < 1e-2,
            "rms = {} arcsec (expected SIP recovery)",
            fit.rms_arcsec
        );
        assert!(fit.wcs.celestial.as_ref().unwrap().sip.is_some());
    }

    #[test]
    fn rejects_low_sip_order() {
        let crpix = (50.0, 50.0);
        let crval = (0.0, 0.0);
        let scale = 1.0 / 3600.0;
        let cd = [scale, 0.0, 0.0, scale];
        let truth = synthesize_wcs(crpix, crval, cd, None);
        let mut pixels = Vec::new();
        let mut sky = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                let px = 10.0 + 20.0 * f64::from(i);
                let py = 10.0 + 20.0 * f64::from(j);
                let (ra, dec) = truth.pixel_to_celestial(px, py).unwrap();
                pixels.push((px, py));
                sky.push((ra, dec));
            }
        }
        let opts = WcsFitOptions {
            sip_order: Some(1),
            ..Default::default()
        };
        let r = fit_celestial_wcs(&pixels, &sky, &opts);
        assert!(r.is_err());
    }

    fn synthesize_wcs(crpix: (f64, f64), crval: (f64, f64), cd: [f64; 4], sip: Option<Sip>) -> Wcs {
        let projection = projection::build(ProjectionKind::Tan, &[]).unwrap();
        let theta0 = projection.theta0();
        let rotation = CelestialRotation::new(crval.0, crval.1, None, None, theta0).unwrap();
        build_wcs(
            ProjectionKind::Tan,
            CelestialFrame::Equatorial,
            crpix.0,
            crpix.1,
            cd,
            crval.0,
            crval.1,
            sip,
            rotation,
            projection,
        )
        .unwrap()
    }
}
