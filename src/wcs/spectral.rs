//! Spectral WCS axis (Greisen et al. 2006, Paper III; Standard
//! Sec.8.4).
//!
//! Implements every spectral CTYPE code of Paper III Table 25: the
//! linear types, the `-LOG` regridding, the twelve `X2P` algorithms
//! (`F2W`, `F2V`, `F2A`, `W2F`, `W2V`, `W2A`, `A2F`, `A2W`, `A2V`,
//! `V2F`, `V2W`, `V2A`) including the air<->vacuum dispersion relation
//! of Sec.4, and the `-GRI` / `-GRA` grism functions of Sec.5.1.
//!
//! Every internal computation runs in SI units:
//!   - the frequency class (`FREQ`, `ENER`, `WAVN`) uses Hz;
//!   - the wavelength class (`WAVE`, `AWAV`) uses m;
//!   - the velocity class (`VRAD`, `VOPT`, `VELO`) uses m/s;
//!   - the dimensionless class (`ZOPT`, `BETA`) uses no unit.
//!
//! `CUNIT`-to-SI conversion is applied on the boundary by
//! [`to_si_factor`]. Grism parameters are always SI regardless of
//! `CUNIT`, per Paper III Table 7.
//!
//! Two deliberate departures from the letter of Paper III, both
//! documented where they live:
//!
//! * the air refractivity uses the Cox (2000) form at 15 degC rather
//!   than the near-0 degC IUGG formula of the paper. See
//!   [`air_refractive_index`];
//! * every algorithm is evaluated by pivoting through absolute
//!   frequency rather than the paper's `S -> P -> X` chain, so a
//!   velocity-like type needs `RESTFRQ`/`RESTWAV` for codes whose
//!   closed forms cancel the rest quantity out -- see
//!   [`SpectralAxis::new`].
//!
//! Tabular spectral axes (`WAVE-TAB`, `FREQ-TAB`, ...) are handled by
//! [`crate::wcs::tab`] rather than here, since the lookup mechanism is
//! shared with every other axis type.

use crate::error::{FitsError, Result};
use crate::wcs::D2R;

/// Speed of light in vacuum, m/s (CODATA 2018).
pub use crate::units::constants::raw::SPEED_OF_LIGHT;

/// Planck constant, J*s (CODATA 2018, exact since 2019 SI redef).
pub use crate::units::constants::raw::PLANCK;

/// Refractive index of dry air. The `lambda_a` argument is in metres.
///
/// This is the Cox (2000) form, from Allen's Astrophysical Quantities,
/// for air at 15 degrees Celsius and 760 mmHg:
///
/// ```text
/// n = 1 + 1e-6 (64.328 + 29498.1/(146 - s^2) + 255.4/(41 - s^2))
/// ```
///
/// with `s = 1/lambda_a` in inverse micrometres.
///
/// This is not Paper III eq. (7). That equation describes air near 0
/// degrees Celsius, and so gives a refractivity about 5 percent
/// higher. The two forms differ by about 15 parts per million of
/// wavelength, which is 0.08 A at 5000 A. The difference appears on
/// any code that crosses between air and vacuum.
///
/// # Domain
///
/// The resonant denominators vanish at 0.0828 um and 0.1562 um. The
/// formula is therefore meaningless in the far UV, and it applies to
/// `lambda_a > 200 nm`. Each conversion built on it rejects an
/// argument near a pole rather than returning a diverging value.
#[must_use]
pub fn air_refractive_index(lambda_a_m: f64) -> f64 {
    let (_, d1, d2) = air_terms(lambda_a_m);
    1.0 + 1e-6 * (64.328 + 29_498.1 / d1 + 255.4 / d2)
}

/// `sigma^2` and the two resonant denominators, shared by the index
/// and its derivative.
fn air_terms(lambda_a_m: f64) -> (f64, f64, f64) {
    let sigma2 = (1.0 / (lambda_a_m * 1e6)).powi(2);
    (sigma2, 146.0 - sigma2, 41.0 - sigma2)
}

/// Reject wavelengths at or below the resonances, where the formula
/// diverges instead of describing air.
fn check_air_domain(lambda_a_m: f64) -> Result<()> {
    let (_, d1, d2) = air_terms(lambda_a_m);
    if d1 <= 1e-3 || d2 <= 1e-3 {
        return Err(FitsError::Wcs(format!(
            "spectral: air wavelength {lambda_a_m} m is at or below the refractivity \
             resonances (0.156 um); the relation is defined for lambda > 200 nm"
        )));
    }
    Ok(())
}

/// `d(lambda) / d(lambda_a)`, needed by the `dX/dw` chain of Paper III
/// eq. (4).
///
/// Differentiating `lambda = n(lambda_a) lambda_a` gives
/// `1 + 1e-6 (f(s) - s f'(s))`, the counterpart of Paper III eq. (8)
/// for the index used here.
#[must_use]
fn air_to_vacuum_derivative(lambda_a_m: f64) -> f64 {
    let (sigma2, d1, d2) = air_terms(lambda_a_m);
    let f = 64.328 + 29_498.1 / d1 + 255.4 / d2;
    // s f'(s) = 2 s^2 [ 29498.1/(146-s^2)^2 + 255.4/(41-s^2)^2 ]
    let sf = 2.0 * sigma2 * (29_498.1 / (d1 * d1) + 255.4 / (d2 * d2));
    1.0 + 1e-6 * (f - sf)
}

/// Air wavelength -> vacuum wavelength, metres: `lambda = n(lambda_a)
/// lambda_a`, the shape of Paper III eq. (6) with the index of
/// [`air_refractive_index`].
fn air_to_vacuum(lambda_a: f64) -> Result<f64> {
    if lambda_a <= 0.0 {
        return Err(FitsError::Wcs(
            "spectral: air wavelength must be positive".into(),
        ));
    }
    check_air_domain(lambda_a)?;
    Ok(air_refractive_index(lambda_a) * lambda_a)
}

/// Vacuum wavelength -> air wavelength, metres: the exact inverse of
/// [`air_to_vacuum`].
// Paper III eq. (9) offers the shortcut `lambda / n(lambda)`, but that
// is only an approximate inverse of eq. (6) -- out by 4e-9 relative,
// enough that an `AWAV` axis stops reporting `CRVAL` at its own
// reference point. So the shortcut seeds a Newton refinement instead;
// `n` is within 3e-4 of unity, so it converges in one or two steps.
fn vacuum_to_air(lambda: f64) -> Result<f64> {
    if lambda <= 0.0 {
        return Err(FitsError::Wcs(
            "spectral: vacuum wavelength must be positive".into(),
        ));
    }
    check_air_domain(lambda)?;
    let mut lambda_a = lambda / air_refractive_index(lambda);
    for _ in 0..8 {
        let residual = air_refractive_index(lambda_a) * lambda_a - lambda;
        let slope = air_to_vacuum_derivative(lambda_a);
        if slope == 0.0 {
            break;
        }
        let step = residual / slope;
        lambda_a -= step;
        if step.abs() <= 1e-16 * lambda_a.abs() {
            break;
        }
    }
    if lambda_a <= 0.0 {
        return Err(FitsError::Wcs(
            "spectral: air wavelength solution is not positive".into(),
        ));
    }
    Ok(lambda_a)
}

/// User-facing spectral coordinate type (the `S` in Paper III).
// `non_exhaustive`: Table 25 registers new types over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpectralKind {
    /// `FREQ` -- frequency, Hz.
    Freq,
    /// `ENER` -- photon energy, J.
    Ener,
    /// `WAVN` -- wavenumber, 1/m.
    Wavn,
    /// `WAVE` -- vacuum wavelength, m.
    Wave,
    /// `AWAV` -- wavelength in dry air, m. Conversions to and from
    /// every other type go through [`air_refractive_index`].
    Awav,
    /// `VRAD` -- radio velocity, m/s. Requires `RESTFRQ`.
    Vrad,
    /// `VOPT` -- optical velocity, m/s. Requires `RESTWAV` (or
    /// `RESTFRQ`).
    Vopt,
    /// `ZOPT` -- redshift, dimensionless.
    Zopt,
    /// `VELO` -- apparent (relativistic) radial velocity, m/s.
    Velo,
    /// `BETA` -- apparent radial velocity / c, dimensionless.
    Beta,
}

impl SpectralKind {
    /// Recognize the leading 4-char code (case-insensitive). Returns
    /// `None` for non-spectral CTYPE.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        let c = code.trim().to_ascii_uppercase();
        Some(match c.as_str() {
            "FREQ" => Self::Freq,
            "ENER" => Self::Ener,
            "WAVN" => Self::Wavn,
            "WAVE" => Self::Wave,
            "AWAV" => Self::Awav,
            "VRAD" => Self::Vrad,
            "VOPT" => Self::Vopt,
            "ZOPT" => Self::Zopt,
            "VELO" => Self::Velo,
            "BETA" => Self::Beta,
            _ => return None,
        })
    }

    /// What "class" of variable this is (frequency, wavelength, or
    /// velocity-like). Determines which rest-quantity is required and
    /// which intermediate variables are valid for the algorithm code.
    fn class(self) -> SpectralClass {
        match self {
            Self::Freq | Self::Ener | Self::Wavn => SpectralClass::F,
            Self::Wave | Self::Awav => SpectralClass::W,
            Self::Vrad | Self::Vopt | Self::Velo | Self::Zopt | Self::Beta => SpectralClass::V,
        }
    }

    /// The *associate variable* `P` this type is linearly related to,
    /// as the letter used in an algorithm code (Standard Table 25,
    /// "Assoc. variable").
    ///
    /// Paper III Sec.3.3.1 introduced `P` to cut down the number of
    /// legal `X2P` combinations, and fixes one per type: `VRAD` goes
    /// with frequency, `VOPT` and `ZOPT` with wavelength, `VELO` and
    /// `BETA` with velocity, and so on.
    pub(crate) fn associate_letter(self) -> char {
        match self {
            Self::Freq | Self::Ener | Self::Wavn | Self::Vrad => 'F',
            Self::Wave | Self::Vopt | Self::Zopt => 'W',
            Self::Awav => 'A',
            Self::Velo | Self::Beta => 'V',
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpectralClass {
    F,
    W,
    V,
}

/// Linearised intermediate variable `X` for a non-linear algorithm
/// (Paper III Sec.3.3: the *first* letter of the algorithm code).
// `non_exhaustive`: Table 25 registers new codes over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Linearised {
    /// Frequency-linear (`-F2*`).
    Freq,
    /// Vacuum-wavelength-linear (`-W2*`).
    Wave,
    /// Air-wavelength-linear (`-A2*`, Paper III Sec.4).
    AirWave,
    /// Apparent-velocity-linear (`-V2*`).
    Velo,
}

/// Non-linear regridding algorithm (Paper III Sec.3.3 Table 25,
/// Sec.5.1 for the grism codes).
// `non_exhaustive`: Table 25 registers new codes over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpectralAlgorithm {
    /// `-LOG` -- `S` is logarithmic in pixel:
    /// `S = S_r * exp(w / S_r)`.
    Log,
    /// `-X2Y` -- the variable `X` is linear in pixel; `S` is
    /// recovered through the F<->W<->A<->V conversions.
    Linear(Linearised),
    /// `-GRI` / `-GRA` -- a grating, prism or grism disperser
    /// (Paper III Sec.5.1). The axis is linear in the grism parameter
    /// `Gamma`, the detector offset in camera focal lengths. The
    /// disperser itself is in [`SpectralAxis::grism`], from `PVk_m`.
    Grism {
        /// `true` for `-GRA` (dispersion in dry air at STP), `false`
        /// for `-GRI` (in vacuum).
        air: bool,
    },
}

impl SpectralAlgorithm {
    /// Recognize the 3-char algorithm code that follows `S-` in the
    /// 8-char CTYPE field. `None` for `TAB` (driven by
    /// [`crate::wcs::tab`]) and for unregistered codes.
    // Only the first letter selects the transform; the third names the
    // associate variable, which Paper III Sec.3.3.1 fixes per type.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        let c = code.trim().to_ascii_uppercase();
        Some(match c.as_str() {
            "LOG" => Self::Log,
            "F2W" | "F2V" | "F2A" => Self::Linear(Linearised::Freq),
            "W2F" | "W2V" | "W2A" => Self::Linear(Linearised::Wave),
            "A2F" | "A2W" | "A2V" => Self::Linear(Linearised::AirWave),
            "V2F" | "V2W" | "V2A" => Self::Linear(Linearised::Velo),
            "GRI" => Self::Grism { air: false },
            "GRA" => Self::Grism { air: true },
            _ => return None,
        })
    }
}

/// Disperser parameters for a `-GRI` / `-GRA` axis (Paper III
/// Sec.5.1.3 Table 7), read from `PVk_m` on the spectral axis `k`.
///
/// Angles are in degrees, as written in the header. The defaults
/// describe a degenerate disperser, which is rejected at construction.
///
/// Construct one with
/// `Grism { density, order, alpha, ..Default::default() }`.
// Not `non_exhaustive`: Table 7 fixes the disperser at these seven
// parameters, and callers need to build one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grism {
    /// `PVk_0` -- grating ruling density `G`, 1/m. Zero for a pure prism.
    pub density: f64,
    /// `PVk_1` -- interference order `m`. Zero for a pure prism.
    pub order: f64,
    /// `PVk_2` -- angle of incidence `alpha`, degrees.
    pub alpha: f64,
    /// `PVk_3` -- index of refraction at the reference wavelength,
    /// `n_r`. Unity for a reflection or transmission grating.
    pub n_r: f64,
    /// `PVk_4` -- `dn/dlambda` at the reference wavelength, 1/m.
    pub n_r_prime: f64,
    /// `PVk_5` -- `epsilon`, the angle between the grating normal and
    /// the dispersion plane, degrees.
    pub epsilon: f64,
    /// `PVk_6` -- `theta`, reference ray to camera axis, degrees.
    pub theta: f64,
}

impl Default for Grism {
    /// Paper III Table 7 defaults: `n_r` is 1, everything else 0.
    fn default() -> Self {
        Self {
            density: 0.0,
            order: 0.0,
            alpha: 0.0,
            n_r: 1.0,
            n_r_prime: 0.0,
            epsilon: 0.0,
            theta: 0.0,
        }
    }
}

impl Grism {
    /// `Gm / cos(epsilon) - n'_r sin(alpha)`, the denominator of the
    /// grism equation (Paper III eq. 15), which "must not be zero".
    fn denominator(self) -> Result<f64> {
        let cos_eps = (self.epsilon * D2R).cos();
        if cos_eps == 0.0 {
            return Err(FitsError::Wcs(
                "grism: PVk_5 (epsilon) = +/-90 deg leaves G*m/cos(epsilon) undefined".into(),
            ));
        }
        let d = self.density * self.order / cos_eps - self.n_r_prime * (self.alpha * D2R).sin();
        if d == 0.0 {
            return Err(FitsError::Wcs(
                "grism: G*m/cos(epsilon) - n'_r sin(alpha) is zero (Paper III eq. 15 \
                 requires a non-zero denominator)"
                    .into(),
            ));
        }
        Ok(d)
    }

    /// Reject a parameter set that cannot describe a disperser.
    fn validate(self) -> Result<()> {
        self.denominator()?;
        Ok(())
    }

    /// `gamma_r`, the exit angle of the reference ray, in radians
    /// (Paper III eq. 16).
    fn gamma_r(self, lambda_r: f64) -> Result<f64> {
        let cos_eps = (self.epsilon * D2R).cos();
        let s =
            self.density * self.order * lambda_r / cos_eps - self.n_r * (self.alpha * D2R).sin();
        if !(-1.0..=1.0).contains(&s) {
            return Err(FitsError::Wcs(format!(
                "grism: sin(gamma_r) = {s} is out of range -- the disperser parameters \
                 and the reference wavelength are inconsistent (Paper III eq. 16)"
            )));
        }
        Ok(s.asin())
    }

    /// Medium wavelength (metres) from the exit angle (Paper III
    /// eq. 15).
    fn lambda_from_gamma(self, gamma: f64, lambda_r: f64) -> Result<f64> {
        let numer = (self.n_r - self.n_r_prime * lambda_r) * (self.alpha * D2R).sin() + gamma.sin();
        Ok(numer / self.denominator()?)
    }

    /// Exit angle from the medium wavelength -- the inverse of
    /// [`Self::lambda_from_gamma`], given in Paper III Sec.5.1.3.
    fn gamma_from_lambda(self, lambda: f64, lambda_r: f64) -> Result<f64> {
        let s = lambda * self.denominator()?
            - (self.n_r - self.n_r_prime * lambda_r) * (self.alpha * D2R).sin();
        if !(-1.0..=1.0).contains(&s) {
            return Err(FitsError::Wcs(format!(
                "grism: sin(gamma) = {s} is out of range -- wavelength {lambda} m is \
                 outside the disperser's domain"
            )));
        }
        Ok(s.asin())
    }

    /// `(dGamma/dlambda)_r` (Paper III eq. 26), with `lambda` the
    /// wavelength in the dispersing medium.
    fn dgamma_dlambda(self, gamma_r: f64) -> Result<f64> {
        let cos_theta = (self.theta * D2R).cos();
        let denom = gamma_r.cos() * cos_theta * cos_theta;
        if denom == 0.0 {
            return Err(FitsError::Wcs(
                "grism: cos(gamma_r) cos^2(theta) is zero, so dGamma/dlambda diverges".into(),
            ));
        }
        Ok(self.denominator()? / denom)
    }
}

/// The variable a non-linear algorithm samples linearly: the `X` of
/// Paper III eq. (2), or the grism parameter `Gamma` of eq. (24), which
/// enters the chain identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XKind {
    Lin(Linearised),
    Grism { air: bool },
}

/// Wavelength in the dispersing medium (Paper III Sec.5.1.5): vacuum
/// for `-GRI`, dry air at STP for `-GRA`.
fn medium_wavelength(air: bool, f: f64) -> Result<f64> {
    if f <= 0.0 {
        return Err(FitsError::Wcs("grism: frequency must be positive".into()));
    }
    let lambda = SPEED_OF_LIGHT / f;
    if air {
        vacuum_to_air(lambda)
    } else {
        Ok(lambda)
    }
}

/// Inverse of [`medium_wavelength`].
fn freq_from_medium_wavelength(air: bool, lambda_medium: f64) -> Result<f64> {
    let vacuum = if air {
        air_to_vacuum(lambda_medium)?
    } else {
        lambda_medium
    };
    if vacuum <= 0.0 {
        return Err(FitsError::Wcs(
            "grism: solved wavelength is not positive -- outside the disperser's domain".into(),
        ));
    }
    Ok(SPEED_OF_LIGHT / vacuum)
}

/// A parsed spectral axis ready to apply the forward (`pix -> S`) /
/// inverse (`S -> pix`) transforms to its intermediate world
/// coordinate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpectralAxis {
    /// Zero-based axis index.
    pub axis: usize,
    /// User-facing coordinate type `S`.
    pub kind: SpectralKind,
    /// Non-linear regridding algorithm, if any. `None` => linear in
    /// pixel.
    pub algorithm: Option<SpectralAlgorithm>,
    /// `CRVAL` value of `S`, in SI units.
    pub crval_si: f64,
    /// Rest frequency (Hz), if supplied via `RESTFRQ`.
    pub restfrq: Option<f64>,
    /// Rest wavelength (m), if supplied via `RESTWAV`.
    pub restwav: Option<f64>,
    /// Multiplier converting CUNIT -> SI (e.g. 1e9 for `GHz`).
    pub unit_to_si: f64,
    /// Disperser parameters (`PVk_0`..`PVk_6`) for a `-GRI` / `-GRA`
    /// axis. `None` for every other algorithm.
    pub grism: Option<Grism>,
}

impl SpectralAxis {
    /// Build a spectral axis from its parsed pieces.
    ///
    /// The `crval_user` and `cunit` arguments hold the `CRVAL` and
    /// `CUNIT` values of this axis as the header writes them.
    /// [`to_si_factor`] converts them to SI.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `cunit` names no unit that matches the
    /// spectral kind, when the algorithm needs a rest frequency or a
    /// rest wavelength that the header omits, or when a grism
    /// algorithm carries incomplete `PVi_m` parameters.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per independent FITS keyword; grouping them would only move the same fields behind a builder"
    )]
    pub fn new(
        axis: usize,
        kind: SpectralKind,
        algorithm: Option<SpectralAlgorithm>,
        crval_user: f64,
        cunit: &str,
        restfrq_hz: Option<f64>,
        restwav_m: Option<f64>,
        grism: Option<Grism>,
    ) -> Result<Self> {
        let unit_to_si = to_si_factor(kind, cunit)?;
        let crval_si = crval_user * unit_to_si;
        // A grism axis is defined by its disperser; without the
        // parameters there is no coordinate function at all.
        match (algorithm, grism) {
            (Some(SpectralAlgorithm::Grism { .. }), None) => {
                return Err(FitsError::Wcs(format!(
                    "spectral axis {} (-GRI/-GRA) has no PV{}_m disperser parameters \
                     (Paper III Sec.5.1.3)",
                    axis + 1,
                    axis + 1,
                )));
            }
            (Some(SpectralAlgorithm::Grism { .. }), Some(g)) => g.validate()?,
            _ => {}
        }
        // A rest quantity is only needed when a non-linear algorithm
        // forces a trip through absolute frequency: the linearised
        // variable is velocity, or the user-facing type is
        // velocity-like and has to be related to its associate
        // variable. A bare `VRAD`/`VOPT`/`ZOPT`/`VELO`/`BETA` axis is
        // `S = S_r + w` and a `-LOG` one is `S_r exp(w/S_r)`; neither
        // touches `F`, so demanding RESTFRQ there rejected headers the
        // standard permits (Paper III Sec.3.3.4 names only the
        // `F2V`/`V2F`/`W2V`/`V2W`/`A2V`/`V2A` codes; Standard Sec.8.4
        // is a `should`).
        //
        // Still stricter than Paper III for `VOPT-F2W`/`ZOPT-F2W`: its
        // closed forms (Table 5) cancel `lambda_0` out, while pivoting
        // through absolute frequency as we do here cannot.
        //
        // A grism chain always evaluates `S -> F` at the reference
        // point (the disperser's `lambda_r`), so a velocity-like `S`
        // needs the rest quantity there too -- caught here rather than
        // on the first transform.
        let needs_rest = match algorithm {
            Some(SpectralAlgorithm::Linear(lx)) => {
                kind.class() == SpectralClass::V || lx == Linearised::Velo
            }
            Some(SpectralAlgorithm::Grism { .. }) => kind.class() == SpectralClass::V,
            _ => false,
        };
        if needs_rest && restfrq_hz.is_none() && restwav_m.is_none() {
            return Err(FitsError::Wcs(format!(
                "spectral axis {} (CTYPE {:?}, algorithm {:?}) requires RESTFRQ or RESTWAV",
                axis + 1,
                kind,
                algorithm,
            )));
        }
        Ok(Self {
            axis,
            kind,
            algorithm,
            crval_si,
            restfrq: restfrq_hz,
            restwav: restwav_m,
            unit_to_si,
            grism,
        })
    }

    /// Forward step, from an intermediate world coordinate to the
    /// world value `S`. Both are in `CUNIT`, and the input is relative
    /// to `CRVAL`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when a `-LOG` axis has `CRVAL = 0`, or when
    /// the point lies outside the domain of the algorithm.
    pub fn intermediate_to_world(&self, w_user: f64) -> Result<f64> {
        let w_si = w_user * self.unit_to_si;
        let s_si = match self.algorithm {
            None => self.crval_si + w_si,
            Some(SpectralAlgorithm::Log) => {
                // Paper III Sec.3.2: S = S_r * exp(w / S_r).
                if self.crval_si == 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral -LOG axis: CRVAL must be non-zero".into(),
                    ));
                }
                self.crval_si * (w_si / self.crval_si).exp()
            }
            // Paper III eq. (2): X = X_r + w (dX/dw), then back to S.
            // The grism function of Sec.5.1 has the same shape with X
            // replaced by the grism parameter Gamma (eq. 24), so it
            // shares the chain rather than duplicating it.
            Some(SpectralAlgorithm::Linear(lin)) => self.forward_via_x(XKind::Lin(lin), w_si)?,
            Some(SpectralAlgorithm::Grism { air }) => {
                self.forward_via_x(XKind::Grism { air }, w_si)?
            }
        };
        Ok(s_si / self.unit_to_si)
    }

    /// Inverse step, from the world value `S` to an intermediate
    /// world coordinate. Both are in `CUNIT`, and the result is
    /// relative to `CRVAL`.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] in three cases:
    ///
    /// - A `-LOG` axis has `CRVAL = 0`.
    /// - `S` and `CRVAL` have opposite signs on a `-LOG` axis.
    /// - The point lies outside the domain of the algorithm.
    pub fn world_to_intermediate(&self, s_user: f64) -> Result<f64> {
        let s_si = s_user * self.unit_to_si;
        let w_si = match self.algorithm {
            None => s_si - self.crval_si,
            Some(SpectralAlgorithm::Log) => {
                // `S = S_r exp(w/S_r)` keeps the sign of `S_r`, so the
                // inverse is defined whenever `S` and `CRVAL` are
                // non-zero and share a sign -- a negative reference
                // value (an approaching source on a velocity axis) is
                // as legal here as it is in the forward direction.
                if self.crval_si == 0.0 || s_si / self.crval_si <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral -LOG inverse: CRVAL and S must be non-zero and share a sign"
                            .into(),
                    ));
                }
                self.crval_si * (s_si / self.crval_si).ln()
            }
            Some(SpectralAlgorithm::Linear(lin)) => self.inverse_via_x(XKind::Lin(lin), s_si)?,
            Some(SpectralAlgorithm::Grism { air }) => {
                self.inverse_via_x(XKind::Grism { air }, s_si)?
            }
        };
        Ok(w_si / self.unit_to_si)
    }

    /// `w -> S` for every algorithm that linearises some variable `X`
    /// (Paper III eq. 2 and, for grisms, eq. 24).
    fn forward_via_x(&self, x_kind: XKind, w_si: f64) -> Result<f64> {
        let f_r = self.f_at_reference()?;
        let x_r = self.linearised_from_freq(x_kind, f_r)?;
        let dxds = self.dxds_at_reference(x_kind, f_r)?;
        let x = x_r + dxds * w_si;
        let f = self.freq_from_linearised(x_kind, x)?;
        self.s_from_freq(f)
    }

    /// The inverse chain, traversed in the reverse direction.
    fn inverse_via_x(&self, x_kind: XKind, s_si: f64) -> Result<f64> {
        let f_r = self.f_at_reference()?;
        let x_r = self.linearised_from_freq(x_kind, f_r)?;
        let dxds = self.dxds_at_reference(x_kind, f_r)?;
        if dxds == 0.0 {
            return Err(FitsError::Wcs(
                "spectral inverse: dX/dS at reference is zero".into(),
            ));
        }
        let f = self.freq_from_s(s_si)?;
        let x = self.linearised_from_freq(x_kind, f)?;
        Ok((x - x_r) / dxds)
    }

    /// Disperser parameters, or a clear error. `new` rejects a grism
    /// axis without them, so this only fires for a hand-built axis.
    fn grism_params(&self) -> Result<Grism> {
        self.grism.ok_or_else(|| {
            FitsError::Wcs(format!(
                "spectral axis {}: -GRI/-GRA without disperser parameters",
                self.axis + 1
            ))
        })
    }

    /// `lambda_r`, the reference wavelength expressed in the
    /// dispersing medium.
    fn grism_lambda_r(&self, air: bool) -> Result<f64> {
        medium_wavelength(air, self.f_at_reference()?)
    }

    // ---- internal: F <-> S converters (all SI) ------------------------

    /// `S -> F` for the user's coordinate type.
    fn freq_from_s(&self, s: f64) -> Result<f64> {
        Ok(match self.kind {
            SpectralKind::Freq => s,
            SpectralKind::Ener => s / PLANCK,
            SpectralKind::Wavn => s * SPEED_OF_LIGHT,
            SpectralKind::Wave => {
                if s <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral: wavelength must be positive".into(),
                    ));
                }
                SPEED_OF_LIGHT / s
            }
            // Air wavelengths run through the dispersion relation of
            // Paper III Sec.4 rather than straight to `c / lambda`.
            SpectralKind::Awav => SPEED_OF_LIGHT / air_to_vacuum(s)?,
            SpectralKind::Vrad => {
                let f0 = self.rest_freq()?;
                f0 * (1.0 - s / SPEED_OF_LIGHT)
            }
            SpectralKind::Vopt => {
                // Optical velocity has no upper bound, but `-c` maps to
                // infinite frequency and anything below it to a
                // negative one.
                if 1.0 + s / SPEED_OF_LIGHT <= 0.0 {
                    return Err(FitsError::Wcs("spectral VOPT: v must be > -c".into()));
                }
                let f0 = self.rest_freq()?;
                f0 / (1.0 + s / SPEED_OF_LIGHT)
            }
            SpectralKind::Zopt => {
                if 1.0 + s <= 0.0 {
                    return Err(FitsError::Wcs("spectral ZOPT: z must be > -1".into()));
                }
                let f0 = self.rest_freq()?;
                f0 / (1.0 + s)
            }
            SpectralKind::Velo => {
                let beta = s / SPEED_OF_LIGHT;
                if beta.abs() >= 1.0 {
                    return Err(FitsError::Wcs("spectral VELO: |v| must be < c".into()));
                }
                let f0 = self.rest_freq()?;
                f0 * ((1.0 - beta) / (1.0 + beta)).sqrt()
            }
            SpectralKind::Beta => {
                if s.abs() >= 1.0 {
                    return Err(FitsError::Wcs("spectral BETA: |beta| must be < 1".into()));
                }
                let f0 = self.rest_freq()?;
                f0 * ((1.0 - s) / (1.0 + s)).sqrt()
            }
        })
    }

    /// `F -> S` for the user's coordinate type.
    fn s_from_freq(&self, f: f64) -> Result<f64> {
        Ok(match self.kind {
            SpectralKind::Freq => f,
            SpectralKind::Ener => f * PLANCK,
            SpectralKind::Wavn => f / SPEED_OF_LIGHT,
            SpectralKind::Wave => {
                if f <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral: frequency must be positive".into(),
                    ));
                }
                SPEED_OF_LIGHT / f
            }
            SpectralKind::Awav => {
                if f <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral: frequency must be positive".into(),
                    ));
                }
                vacuum_to_air(SPEED_OF_LIGHT / f)?
            }
            SpectralKind::Vrad => {
                let f0 = self.rest_freq()?;
                SPEED_OF_LIGHT * (f0 - f) / f0
            }
            SpectralKind::Vopt => {
                let f0 = self.rest_freq()?;
                if f <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral VOPT: frequency must be positive".into(),
                    ));
                }
                SPEED_OF_LIGHT * (f0 / f - 1.0)
            }
            SpectralKind::Zopt => {
                let f0 = self.rest_freq()?;
                if f <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral ZOPT: frequency must be positive".into(),
                    ));
                }
                f0 / f - 1.0
            }
            SpectralKind::Velo => {
                let f0 = self.rest_freq()?;
                if f <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral VELO: frequency must be positive".into(),
                    ));
                }
                let r2 = (f / f0).powi(2);
                SPEED_OF_LIGHT * (1.0 - r2) / (1.0 + r2)
            }
            SpectralKind::Beta => {
                let f0 = self.rest_freq()?;
                if f <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral BETA: frequency must be positive".into(),
                    ));
                }
                let r2 = (f / f0).powi(2);
                (1.0 - r2) / (1.0 + r2)
            }
        })
    }

    fn linearised_from_freq(&self, x_kind: XKind, f: f64) -> Result<f64> {
        Ok(match x_kind {
            XKind::Lin(Linearised::Freq) => f,
            XKind::Lin(Linearised::Wave) => SPEED_OF_LIGHT / f,
            XKind::Lin(Linearised::AirWave) => vacuum_to_air(SPEED_OF_LIGHT / f)?,
            // Paper III Sec.5.1.3 steps 2-3 run backwards: medium
            // wavelength -> exit angle -> grism parameter (eq. 23).
            XKind::Grism { air } => {
                let g = self.grism_params()?;
                let lambda_r = self.grism_lambda_r(air)?;
                let gamma = g.gamma_from_lambda(medium_wavelength(air, f)?, lambda_r)?;
                (gamma - g.gamma_r(lambda_r)? - g.theta * D2R).tan()
            }
            XKind::Lin(Linearised::Velo) => {
                // Apparent velocity of radiation at frequency f
                // relative to the rest frequency, m/s.
                //
                // `rest_freq()`, not `self.restfrq`: `new()` only
                // guarantees that *one* of RESTFRQ / RESTWAV is
                // present, so a header giving only RESTWAV (legal for
                // e.g. `WAVE-V2W`) has `restfrq == None` and used to
                // panic here.
                let f0 = self.rest_freq()?;
                let r2 = (f / f0).powi(2);
                SPEED_OF_LIGHT * (1.0 - r2) / (1.0 + r2)
            }
        })
    }

    fn freq_from_linearised(&self, x_kind: XKind, x: f64) -> Result<f64> {
        Ok(match x_kind {
            XKind::Lin(Linearised::Freq) => x,
            XKind::Lin(Linearised::Wave) => {
                if x <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral: linearised wavelength must be positive".into(),
                    ));
                }
                SPEED_OF_LIGHT / x
            }
            XKind::Lin(Linearised::AirWave) => SPEED_OF_LIGHT / air_to_vacuum(x)?,
            // Paper III Sec.5.1.3 steps 2-3: grism parameter -> exit
            // angle -> medium wavelength.
            XKind::Grism { air } => {
                let g = self.grism_params()?;
                let lambda_r = self.grism_lambda_r(air)?;
                let gamma = x.atan() + g.gamma_r(lambda_r)? + g.theta * D2R;
                freq_from_medium_wavelength(air, g.lambda_from_gamma(gamma, lambda_r)?)?
            }
            XKind::Lin(Linearised::Velo) => {
                let beta = x / SPEED_OF_LIGHT;
                if beta.abs() >= 1.0 {
                    return Err(FitsError::Wcs(
                        "spectral: linearised |v| must be < c".into(),
                    ));
                }
                // `rest_freq()`, not `self.restfrq`: RESTWAV alone is
                // a legal way to specify the rest quantity and
                // `new()` accepts it, so demanding RESTFRQ here
                // rejected headers the parser had already approved.
                let f0 = self.rest_freq()?;
                f0 * ((1.0 - beta) / (1.0 + beta)).sqrt()
            }
        })
    }

    /// `(dX/dS)|_r` evaluated at the reference frequency.
    ///
    /// This is Paper III eq. (4), `dX/dw = (dP/dS)_r / (dP/dX)_r`,
    /// written as the equivalent chain `(dX/dF)(dF/dS)` because every
    /// variable here is reached through absolute frequency.
    fn dxds_at_reference(&self, x_kind: XKind, f_r: f64) -> Result<f64> {
        // Chain rule: dX/dS = (dX/dF) * (dF/dS).
        let dxdf = match x_kind {
            XKind::Lin(Linearised::Freq) => 1.0,
            XKind::Lin(Linearised::Wave) => -SPEED_OF_LIGHT / (f_r * f_r),
            // dlambda_a/dF = (dlambda/dF) / (dlambda/dlambda_a).
            XKind::Lin(Linearised::AirWave) => {
                let lambda_a = vacuum_to_air(SPEED_OF_LIGHT / f_r)?;
                -SPEED_OF_LIGHT / (f_r * f_r) / air_to_vacuum_derivative(lambda_a)
            }
            // Paper III eq. (27): dGamma/dw = (dGamma/dlambda)
            // (dlambda/dP)(dP/dS). The last two factors are the medium
            // wavelength's own dX/dF times the dF/dS below.
            XKind::Grism { air } => {
                let g = self.grism_params()?;
                let lambda_r = self.grism_lambda_r(air)?;
                let dlambda_df = if air {
                    -SPEED_OF_LIGHT / (f_r * f_r) / air_to_vacuum_derivative(lambda_r)
                } else {
                    -SPEED_OF_LIGHT / (f_r * f_r)
                };
                g.dgamma_dlambda(g.gamma_r(lambda_r)?)? * dlambda_df
            }
            XKind::Lin(Linearised::Velo) => {
                let f0 = self.rest_freq()?;
                // V = c*(1 - r^2)/(1 + r^2) where r = F/F_0
                // dV/dF = -4*c*F/(F_0^2*(1 + r^2)^2)
                let r2 = (f_r / f0).powi(2);
                -4.0 * SPEED_OF_LIGHT * f_r / (f0 * f0 * (1.0 + r2).powi(2))
            }
        };
        let dfds = match self.kind {
            SpectralKind::Freq => 1.0,
            SpectralKind::Ener => 1.0 / PLANCK,
            SpectralKind::Wavn => SPEED_OF_LIGHT,
            SpectralKind::Wave => {
                let w_r = SPEED_OF_LIGHT / f_r;
                -SPEED_OF_LIGHT / (w_r * w_r)
            }
            // dF/dlambda_a = (dF/dlambda)(dlambda/dlambda_a).
            SpectralKind::Awav => {
                let w_r = SPEED_OF_LIGHT / f_r;
                -SPEED_OF_LIGHT / (w_r * w_r) * air_to_vacuum_derivative(vacuum_to_air(w_r)?)
            }
            SpectralKind::Vrad => {
                let f0 = self.rest_freq()?;
                -f0 / SPEED_OF_LIGHT
            }
            SpectralKind::Vopt => {
                let f0 = self.rest_freq()?;
                // V = c*(F_0/F - 1) => dF/dV = -F^2/(c*F_0)
                -(f_r * f_r) / (SPEED_OF_LIGHT * f0)
            }
            SpectralKind::Zopt => {
                let f0 = self.rest_freq()?;
                -(f_r * f_r) / f0
            }
            SpectralKind::Velo => {
                // Inverse of the dV/dF expression above.
                let f0 = self.rest_freq()?;
                let r2 = (f_r / f0).powi(2);
                -(f0 * f0) * (1.0 + r2).powi(2) / (4.0 * SPEED_OF_LIGHT * f_r)
            }
            SpectralKind::Beta => {
                let f0 = self.rest_freq()?;
                let r2 = (f_r / f0).powi(2);
                -(f0 * f0) * (1.0 + r2).powi(2) / (4.0 * f_r)
            }
        };
        Ok(dxdf * dfds)
    }

    /// Reference frequency derived from `CRVAL`.
    fn f_at_reference(&self) -> Result<f64> {
        self.freq_from_s(self.crval_si)
    }

    /// Rest frequency, derived from RESTFRQ (preferred) or RESTWAV.
    fn rest_freq(&self) -> Result<f64> {
        if let Some(f0) = self.restfrq {
            // A pipeline that has no rest frequency often writes
            // `RESTFRQ = 0.0` rather than omitting the card; unchecked
            // it reaches every `/ f0` and comes back as `Ok(NaN)`.
            if f0 <= 0.0 || !f0.is_finite() {
                return Err(FitsError::Wcs("RESTFRQ must be positive".into()));
            }
            Ok(f0)
        } else if let Some(w0) = self.restwav {
            if w0 <= 0.0 {
                return Err(FitsError::Wcs("RESTWAV must be positive".into()));
            }
            Ok(SPEED_OF_LIGHT / w0)
        } else {
            Err(FitsError::Wcs(
                "spectral: RESTFRQ or RESTWAV required for this transform".into(),
            ))
        }
    }
}

/// Spectral reference frames of the description as a whole (Paper III
/// Sec.7, Standard Sec.8.4.3): which frame the coordinates are
/// expressed in, and the observation-side velocity relating it to the
/// telescope. Stored; not applied -- no frame transformation is
/// performed.
///
/// Present on a [`Wcs`](crate::Wcs) only when the description has a
/// spectral axis; see the layering note in [`crate::wcs`].
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct SpectralFrame {
    /// `SPECSYS` -- frame the coordinates are expressed in
    /// (`TOPOCENT`, `BARYCENT`, `LSRK`, ..., Table 27).
    pub specsys: Option<String>,
    /// `SSYSOBS` -- frame constant during the observation, in which
    /// the spectral characteristics of the instrument are fixed.
    pub ssysobs: Option<String>,
    /// `VELOSYS` -- relative radial velocity (m/s) between the
    /// observer and `SSYSOBS`.
    pub velosys: Option<f64>,
    /// The source's own systemic velocity, when `ZSOURCE` gives one.
    pub source: Option<SourceFrame>,
}

/// The source-rest-frame description (Standard Sec.8.4.3). `ZSOURCE`
/// is the parent: `SSYSSRC` and `VELANGL` describe it, so neither is
/// retained without it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SourceFrame {
    /// `ZSOURCE` -- systemic velocity of the source as a unitless
    /// redshift.
    pub zsource: f64,
    /// `SSYSSRC` -- frame `zsource` is expressed in. Any Table 27
    /// value except `SOURCE`.
    pub ssyssrc: Option<String>,
    /// `VELANGL` -- orientation of the space velocity vector with
    /// respect to the plane of the sky, degrees, default +90. Only
    /// meaningful at relativistic velocities.
    pub velangl: Option<f64>,
}

/// The dimension a given spectral type must carry, per Standard
/// Table 25's "default units" column.
///
/// Keyed on the *type*, not the class: `FREQ`, `ENER` and `WAVN` all
/// linearise through frequency but are `s^-1`, `J` and `m^-1`
/// respectively, and confusing them is exactly the mistake worth
/// catching.
fn required_dimension(kind: SpectralKind) -> crate::units::Dimension {
    use crate::units::dimensions;
    match kind {
        SpectralKind::Freq => dimensions::FREQUENCY,
        SpectralKind::Ener => dimensions::ENERGY,
        SpectralKind::Wavn => dimensions::WAVENUMBER,
        SpectralKind::Wave | SpectralKind::Awav => dimensions::LENGTH,
        SpectralKind::Vrad | SpectralKind::Vopt | SpectralKind::Velo => dimensions::VELOCITY,
        SpectralKind::Zopt | SpectralKind::Beta => dimensions::DIMENSIONLESS,
    }
}

/// Multiplier converting a value in `cunit` to the canonical SI unit
/// for the given spectral type, parsed per Standard Sec.4.3.
///
/// A blank `CUNIT` is the standard's default and is taken to be the
/// canonical unit for the type (Table 25's default units column).
///
/// # Errors
///
/// [`FitsError::Header`] if `cunit` is not valid Sec.4.3 syntax, or
/// does not carry the dimension the spectral type requires.
pub fn to_si_factor(kind: SpectralKind, cunit: &str) -> Result<f64> {
    crate::units::factor_to(cunit, required_dimension(kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, rel: f64) {
        let denom = a.abs().max(b.abs()).max(1e-30);
        assert!(
            (a - b).abs() / denom < rel,
            "expected {a} ~= {b} (rel tol {rel})"
        );
    }

    #[test]
    fn linear_freq_passthrough() {
        let ax = SpectralAxis::new(2, SpectralKind::Freq, None, 1.420e9, "Hz", None, None, None)
            .unwrap();
        let s = ax.intermediate_to_world(1e6).unwrap();
        approx(s, 1.421e9, 1e-15);
        let w = ax.world_to_intermediate(1.421e9).unwrap();
        approx(w, 1e6, 1e-12);
    }

    #[test]
    fn cunit_ghz_converts_to_si() {
        let ax =
            SpectralAxis::new(2, SpectralKind::Freq, None, 1.420, "GHz", None, None, None).unwrap();
        approx(ax.crval_si, 1.420e9, 1e-15);
        // intermediate = 0.001 GHz => S = 1.421 GHz.
        let s = ax.intermediate_to_world(0.001).unwrap();
        approx(s, 1.421, 1e-12);
    }

    #[test]
    fn log_round_trip() {
        let ax = SpectralAxis::new(
            2,
            SpectralKind::Wave,
            Some(SpectralAlgorithm::Log),
            500e-9,
            "m",
            None,
            None,
            None,
        )
        .unwrap();
        for &s in &[400e-9_f64, 500e-9, 600e-9, 700e-9] {
            let w = ax.world_to_intermediate(s).unwrap();
            let s2 = ax.intermediate_to_world(w).unwrap();
            approx(s2, s, 1e-13);
        }
    }

    #[test]
    fn wave_f2w_round_trip() {
        // CTYPE = "WAVE-F2W": user wants WAVE, frequency is linear in
        // pixel.
        let ax = SpectralAxis::new(
            2,
            SpectralKind::Wave,
            Some(SpectralAlgorithm::Linear(Linearised::Freq)),
            500e-9,
            "m",
            Some(SPEED_OF_LIGHT / 500e-9),
            None,
            None,
        )
        .unwrap();
        for &s in &[480e-9_f64, 500e-9, 520e-9, 600e-9] {
            let w = ax.world_to_intermediate(s).unwrap();
            let s2 = ax.intermediate_to_world(w).unwrap();
            approx(s2, s, 1e-12);
        }
        // At reference, intermediate = 0 => S = CRVAL exactly.
        let s0 = ax.intermediate_to_world(0.0).unwrap();
        approx(s0, 500e-9, 1e-15);
    }

    #[test]
    fn freq_w2f_round_trip() {
        let ax = SpectralAxis::new(
            2,
            SpectralKind::Freq,
            Some(SpectralAlgorithm::Linear(Linearised::Wave)),
            6.0e14,
            "Hz",
            None,
            None,
            None,
        )
        .unwrap();
        for &s in &[5.5e14_f64, 6.0e14, 6.5e14] {
            let w = ax.world_to_intermediate(s).unwrap();
            let s2 = ax.intermediate_to_world(w).unwrap();
            approx(s2, s, 1e-11);
        }
    }

    #[test]
    fn vopt_f2w_round_trip() {
        // VOPT-F2W: user wants optical velocity, wavelength linear
        // in pixel. RESTWAV = 21cm line.
        let restwav = 0.211_061_141_0;
        let ax = SpectralAxis::new(
            2,
            SpectralKind::Vopt,
            Some(SpectralAlgorithm::Linear(Linearised::Wave)),
            0.0,
            "m/s",
            None,
            Some(restwav),
            None,
        )
        .unwrap();
        for &v in &[-1e6_f64, -1e5, 0.0, 1e5, 1e6] {
            let w = ax.world_to_intermediate(v).unwrap();
            let v2 = ax.intermediate_to_world(w).unwrap();
            approx(v2, v, 1e-9);
        }
    }

    #[test]
    fn velo_f2v_round_trip() {
        // VELO-F2V: relativistic velocity, freq-linear pixel.
        let ax = SpectralAxis::new(
            2,
            SpectralKind::Velo,
            Some(SpectralAlgorithm::Linear(Linearised::Freq)),
            0.0,
            "m/s",
            Some(1.420e9),
            None,
            None,
        )
        .unwrap();
        for &v in &[-1e7_f64, 0.0, 1e7] {
            let w = ax.world_to_intermediate(v).unwrap();
            let v2 = ax.intermediate_to_world(w).unwrap();
            approx(v2, v, 1e-8);
        }
    }

    #[test]
    fn vrad_definition_paper_iii_eq_2() {
        // VRAD: v = c(F_0 - F)/F_0; at F = F_0 => v = 0.
        let ax = SpectralAxis::new(
            2,
            SpectralKind::Vrad,
            None,
            0.0,
            "m/s",
            Some(1.420e9),
            None,
            None,
        )
        .unwrap();
        // V -> F -> V identity at known points.
        // 1 km/s offset from rest frequency.
        let v = ax.intermediate_to_world(1e3).unwrap();
        approx(v, 1e3, 1e-15);
    }

    #[test]
    fn missing_rest_for_velocity_algorithm_is_error() {
        // VOPT-F2W has to reach absolute frequency, so it needs a rest
        // quantity; without one this must fail at construction.
        let r = SpectralAxis::new(
            2,
            SpectralKind::Vopt,
            Some(SpectralAlgorithm::Linear(Linearised::Freq)),
            0.0,
            "m/s",
            None,
            None,
            None,
        );
        assert!(r.is_err());
        // ... and so does a wavelength axis linearised in velocity.
        let r = SpectralAxis::new(
            2,
            SpectralKind::Wave,
            Some(SpectralAlgorithm::Linear(Linearised::Velo)),
            500e-9,
            "m",
            None,
            None,
            None,
        );
        assert!(r.is_err());
    }

    /// Paper III Sec.3.3.4 requires RESTFRQ/RESTWAV only for the
    /// `F2V`/`V2F`/`W2V`/`V2W`/`A2V`/`V2A` codes. A *linear* velocity
    /// axis is `S = S_r + w`, which needs neither.
    ///
    /// Regression: every velocity-like CTYPE used to be rejected
    /// without a rest quantity, and the error propagated out of
    /// `Wcs::from_header`, so a plain `CTYPE3 = 'VRAD'` radio cube had
    /// no WCS at all.
    #[test]
    fn linear_velocity_axis_needs_no_rest_quantity() {
        // `ZOPT` and `BETA` are ratios, so their CUNIT is blank, not
        // `m/s` -- Table 25's default-units column.
        for (kind, unit) in [
            (SpectralKind::Vrad, "m/s"),
            (SpectralKind::Vopt, "m/s"),
            (SpectralKind::Zopt, ""),
            (SpectralKind::Velo, "m/s"),
            (SpectralKind::Beta, ""),
        ] {
            let ax = SpectralAxis::new(2, kind, None, 1000.0, unit, None, None, None)
                .unwrap_or_else(|e| panic!("linear {kind:?} rejected: {e}"));
            // Purely linear: S = CRVAL + w, both ways.
            approx(ax.intermediate_to_world(250.0).unwrap(), 1250.0, 1e-15);
            approx(ax.world_to_intermediate(1250.0).unwrap(), 250.0, 1e-12);
        }
    }

    /// `-LOG` is `S = S_r exp(w / S_r)` -- also no rest quantity.
    #[test]
    fn log_velocity_axis_needs_no_rest_quantity() {
        let ax = SpectralAxis::new(
            2,
            SpectralKind::Velo,
            Some(SpectralAlgorithm::Log),
            1000.0,
            "m/s",
            None,
            None,
            None,
        )
        .expect("VELO-LOG without RESTFRQ must be accepted");
        let w = ax.world_to_intermediate(1500.0).unwrap();
        approx(ax.intermediate_to_world(w).unwrap(), 1500.0, 1e-12);
    }

    /// `S = S_r exp(w / S_r)` keeps the sign of `S_r`, so a negative
    /// reference value -- an approaching source on a velocity axis --
    /// must invert as readily as it transforms forward.
    #[test]
    fn log_axis_round_trips_a_negative_reference() {
        let ax = SpectralAxis::new(
            2,
            SpectralKind::Velo,
            Some(SpectralAlgorithm::Log),
            -1000.0,
            "m/s",
            None,
            None,
            None,
        )
        .unwrap();
        let s = ax.intermediate_to_world(50.0).unwrap();
        approx(ax.world_to_intermediate(s).unwrap(), 50.0, 1e-9);
        // The reference point itself.
        approx(ax.world_to_intermediate(-1000.0).unwrap(), 0.0, 1e-12);
        // A world value on the wrong side of zero stays an error.
        assert!(ax.world_to_intermediate(1000.0).is_err());
    }

    /// A pipeline with no line often writes `RESTFRQ = 0.0` rather
    /// than omitting the card. That must be an error, not `Ok(NaN)`.
    #[test]
    fn degenerate_rest_frequency_is_an_error_not_nan() {
        for f0 in [0.0, -1.4e9, f64::NAN] {
            let ax = SpectralAxis::new(
                0,
                SpectralKind::Vopt,
                Some(SpectralAlgorithm::Linear(Linearised::Freq)),
                0.0,
                "m/s",
                Some(f0),
                None,
                None,
            )
            .unwrap();
            assert!(
                ax.intermediate_to_world(1000.0).is_err(),
                "RESTFRQ = {f0} must not produce a coordinate"
            );
        }
    }

    /// `VOPT = -c` and `ZOPT = -1` sit on the pole of the frequency
    /// map; at and beyond them the axis must refuse like VELO and
    /// BETA already do, not return an infinite or negative frequency.
    #[test]
    fn vopt_and_zopt_reject_their_singularities() {
        let z = SpectralAxis::new(
            0,
            SpectralKind::Zopt,
            Some(SpectralAlgorithm::Linear(Linearised::Freq)),
            -1.0,
            "",
            Some(1.4e9),
            None,
            None,
        )
        .unwrap();
        // The reference point itself is the singular value here.
        assert!(z.intermediate_to_world(0.0).is_err());
        let v = SpectralAxis::new(
            0,
            SpectralKind::Vopt,
            Some(SpectralAlgorithm::Linear(Linearised::Freq)),
            0.0,
            "m/s",
            Some(1.4e9),
            None,
            None,
        )
        .unwrap();
        assert!(v.world_to_intermediate(-SPEED_OF_LIGHT).is_err());
        assert!(v.world_to_intermediate(-2.0 * SPEED_OF_LIGHT).is_err());
    }

    /// The Cox (2000) index, evaluated independently.
    #[test]
    fn air_index_matches_cox_2000() {
        // n = 1 + 1e-6 (64.328 + 29498.1/(146-s^2) + 255.4/(41-s^2)),
        // s = 1/lambda_a in um^-1. These are also `wcslib`'s values to
        // machine precision.
        approx(air_refractive_index(0.5e-6), 1.000_278_963_801_294_3, 1e-15);
        approx(air_refractive_index(0.7e-6), 1.000_275_789_575_434_8, 1e-15);
        // Refractivity falls with wavelength and sits near 2.8e-4 --
        // ~5% below the ~0 degC value Paper III eq. (7) would give.
        assert!(air_refractive_index(0.35e-6) > air_refractive_index(2.0e-6));
        assert!((air_refractive_index(1.0e-6) - 1.0 - 2.74e-4).abs() < 1e-6);
    }

    /// The two resonant denominators make the formula diverge in the
    /// far UV, so the conversions refuse rather than answering with a
    /// number that looks plausible.
    #[test]
    fn air_conversions_reject_the_resonant_region() {
        assert!(air_to_vacuum(1.0e-7).is_err(), "below both poles");
        assert!(
            air_to_vacuum(1.5e-7).is_err(),
            "just below the 0.156 um pole"
        );
        assert!(air_to_vacuum(2.0e-7).is_ok(), "200 nm is in range");
        assert!(vacuum_to_air(1.0e-7).is_err());
        assert!(vacuum_to_air(2.0e-7).is_ok());
    }

    /// Paper III eq. (9) is only an approximate inverse of eq. (6).
    /// Using it as written left an `AWAV` axis reporting something
    /// other than CRVAL at its own reference point, so the inverse is
    /// solved properly; this pins that down.
    #[test]
    fn air_vacuum_round_trip_is_exact() {
        for &lambda_a in &[3.5e-7_f64, 5e-7, 6.5e-7, 1e-6, 2e-6] {
            let vac = air_to_vacuum(lambda_a).unwrap();
            assert!(vac > lambda_a, "vacuum wavelength must be the longer one");
            approx(vacuum_to_air(vac).unwrap(), lambda_a, 1e-15);
        }
        for &lambda in &[3.5e-7_f64, 5e-7, 6.5e-7, 1e-6, 2e-6] {
            approx(
                air_to_vacuum(vacuum_to_air(lambda).unwrap()).unwrap(),
                lambda,
                1e-15,
            );
        }
        // The eq. (9) shortcut alone is off by ~4e-9 relative -- small
        // physically, fatal to the reference point.
        let lambda_a = 5e-7;
        let shortcut = air_to_vacuum(lambda_a).unwrap()
            / air_refractive_index(air_to_vacuum(lambda_a).unwrap());
        assert!((shortcut - lambda_a).abs() / lambda_a > 1e-9);
    }

    /// A bare `AWAV` axis never converts, so it stays exactly linear.
    #[test]
    fn linear_air_axis_is_untouched_by_the_dispersion_relation() {
        let ax =
            SpectralAxis::new(2, SpectralKind::Awav, None, 5e-7, "m", None, None, None).unwrap();
        approx(ax.intermediate_to_world(2.047e-7).unwrap(), 7.047e-7, 1e-15);
    }

    /// Every `X2P` code must round-trip, including the six that involve
    /// air wavelengths.
    ///
    /// The offsets are a percent or so of CRVAL, i.e. what a real
    /// dispersion axis actually spans. Much smaller than that and the
    /// `x - x_r` of the inverse cancels away most of its significant
    /// digits -- inherent to Paper III's linearise-then-subtract form,
    /// not something this implementation can avoid.
    #[test]
    fn every_linearised_variable_round_trips() {
        const W_NM: [f64; 4] = [-2e-8, -5e-9, 5e-9, 2e-8];
        const W_HZ: [f64; 4] = [-2e13, -5e12, 5e12, 2e13];
        const W_MS: [f64; 4] = [-1e7, -1e6, 1e6, 1e7];

        // (kind, X, CRVAL, CUNIT, intermediate offsets)
        let cases: &[(SpectralKind, Linearised, f64, &str, [f64; 4])] = &[
            (SpectralKind::Awav, Linearised::Freq, 5e-7, "m", W_NM),
            (SpectralKind::Awav, Linearised::Wave, 5e-7, "m", W_NM),
            (SpectralKind::Awav, Linearised::Velo, 5e-7, "m", W_NM),
            (SpectralKind::Wave, Linearised::AirWave, 5e-7, "m", W_NM),
            (SpectralKind::Freq, Linearised::AirWave, 6e14, "Hz", W_HZ),
            (SpectralKind::Velo, Linearised::AirWave, 0.0, "m/s", W_MS),
        ];
        for &(kind, lin, crval, unit, offsets) in cases {
            let ax = SpectralAxis::new(
                2,
                kind,
                Some(SpectralAlgorithm::Linear(lin)),
                crval,
                unit,
                None,
                Some(5e-7),
                None,
            )
            .unwrap_or_else(|e| panic!("{kind:?}/{lin:?}: {e}"));
            for w in offsets {
                let s = ax.intermediate_to_world(w).unwrap();
                let back = ax.world_to_intermediate(s).unwrap();
                approx(back, w, 1e-9);
            }
        }
    }

    /// A grism axis is meaningless without its disperser, and a
    /// degenerate parameter set must be refused rather than divided by.
    #[test]
    fn grism_requires_usable_parameters() {
        let mk = |g: Option<Grism>| {
            SpectralAxis::new(
                2,
                SpectralKind::Wave,
                Some(SpectralAlgorithm::Grism { air: false }),
                5e-7,
                "m",
                None,
                None,
                g,
            )
        };
        assert!(mk(None).is_err(), "no PVk_m at all");
        // All defaults: G = m = n'_r = 0, so the eq. (15) denominator
        // vanishes.
        assert!(mk(Some(Grism::default())).is_err(), "degenerate disperser");
        assert!(
            mk(Some(Grism {
                density: 3.16e5,
                order: 1.0,
                alpha: 13.9,
                ..Grism::default()
            }))
            .is_ok(),
            "a plain grating must be accepted"
        );
    }

    /// A velocity-typed grism axis reaches absolute frequency through
    /// its reference point, so the rest quantity is required at
    /// construction, like the `Linear` codes -- not discovered on the
    /// first transform.
    #[test]
    fn grism_velocity_axis_requires_rest_at_construction() {
        let grating = Some(Grism {
            density: 3.16e5,
            order: 1.0,
            alpha: 13.9,
            ..Grism::default()
        });
        let r = SpectralAxis::new(
            2,
            SpectralKind::Vopt,
            Some(SpectralAlgorithm::Grism { air: false }),
            0.0,
            "m/s",
            None,
            None,
            grating,
        );
        assert!(r.is_err(), "VOPT-GRI without a rest quantity must fail");
        // With one, it constructs and transforms.
        let ax = SpectralAxis::new(
            2,
            SpectralKind::Vopt,
            Some(SpectralAlgorithm::Grism { air: false }),
            0.0,
            "m/s",
            None,
            Some(5e-7),
            grating,
        )
        .unwrap();
        // CRVAL = 0 m/s at the reference point, up to roundoff.
        assert!(ax.intermediate_to_world(0.0).unwrap().abs() < 1e-6);
    }

    /// The reference point maps to CRVAL for every algorithm -- Paper
    /// III eq. (3) with w = 0. Cheap, and it catches a mis-seeded
    /// chain that a round-trip test would happily confirm.
    #[test]
    fn reference_point_maps_to_crval() {
        let grating = Some(Grism {
            density: 3.16e5,
            order: 1.0,
            alpha: 13.9,
            ..Grism::default()
        });
        let cases: &[(SpectralKind, Option<SpectralAlgorithm>, Option<Grism>)] = &[
            (SpectralKind::Awav, None, None),
            (SpectralKind::Awav, Some(SpectralAlgorithm::Log), None),
            (
                SpectralKind::Awav,
                Some(SpectralAlgorithm::Linear(Linearised::Freq)),
                None,
            ),
            (
                SpectralKind::Wave,
                Some(SpectralAlgorithm::Linear(Linearised::AirWave)),
                None,
            ),
            (
                SpectralKind::Wave,
                Some(SpectralAlgorithm::Grism { air: false }),
                grating,
            ),
            (
                SpectralKind::Awav,
                Some(SpectralAlgorithm::Grism { air: true }),
                grating,
            ),
        ];
        for &(kind, algo, grism) in cases {
            let ax = SpectralAxis::new(2, kind, algo, 5e-7, "m", None, Some(5e-7), grism)
                .unwrap_or_else(|e| panic!("{kind:?}/{algo:?}: {e}"));
            approx(ax.intermediate_to_world(0.0).unwrap(), 5e-7, 1e-14);
            approx(ax.world_to_intermediate(5e-7).unwrap(), 0.0, 1e-14);
        }
    }
}
