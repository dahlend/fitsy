//! Spectral WCS axis (Greisen et al. 2006, Paper III; Standard
//! Sec.8.4).
//!
//! Implements both the linear spectral CTYPE codes and the
//! non-linear regridding algorithms `-LOG`, `-F2W`, `-W2F`, `-F2V`,
//! `-V2F`, `-W2V`, `-V2W` from Paper III Table 25.
//!
//! Units: all internal computation is performed in **SI**:
//!   - frequency-class (`FREQ`, `ENER`, `WAVN`) in Hz;
//!   - wavelength-class (`WAVE`, `AWAV`) in m;
//!   - velocity-class (`VRAD`, `VOPT`, `VELO`) in m/s;
//!   - dimensionless (`ZOPT`, `BETA`).
//!
//! `CUNIT`-to-SI conversion is applied on the boundary by
//! [`to_si_factor`].
//!
//! Air wavelengths (`AWAV`) and the grism algorithms (`-GRI`/`-GRA`)
//! are out of scope for this release. Tabular spectral axes
//! (`WAVE-TAB`, `FREQ-TAB`, ...) are handled by [`crate::wcs::tab`]
//! rather than here, since the lookup mechanism is shared with
//! every other axis type.

use crate::error::{FitsError, Result};

/// Speed of light in vacuum, m/s (CODATA 2018).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Planck constant, J*s (CODATA 2018, exact since 2019 SI redef).
pub const PLANCK: f64 = 6.626_070_15e-34;

/// User-facing spectral coordinate type (the `S` in Paper III).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralKind {
    /// `FREQ` -- frequency, Hz.
    Freq,
    /// `ENER` -- photon energy, J.
    Ener,
    /// `WAVN` -- wavenumber, 1/m.
    Wavn,
    /// `WAVE` -- vacuum wavelength, m.
    Wave,
    /// `AWAV` -- air wavelength, m. Linear-only; the air<->vacuum
    /// dispersion relation is not applied here.
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpectralClass {
    F,
    W,
    V,
}

/// Linearised intermediate variable for a non-linear algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linearised {
    /// Frequency-linear (Paper III Sec.3.3, `-F2*`).
    Freq,
    /// Wavelength-linear (`-W2*`).
    Wave,
    /// Apparent-velocity-linear (`-V2*`).
    Velo,
}

/// Non-linear regridding algorithm (Paper III Sec.3.3, Table 25).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralAlgorithm {
    /// `-LOG` -- `S` is logarithmic in pixel:
    /// `S = S_r * exp(w / S_r)`.
    Log,
    /// `-X2Y` -- the variable `x` is linear in pixel; `S` is
    /// recovered through the F<->W<->V conversion.
    Linear(Linearised),
}

impl SpectralAlgorithm {
    /// Recognize the 3-char algorithm code that follows `S-` in the
    /// 8-char CTYPE field. Returns `None` for unknown / unsupported
    /// codes (`GRI`, `GRA`, `TAB`).
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        let c = code.trim().to_ascii_uppercase();
        Some(match c.as_str() {
            "LOG" => Self::Log,
            "F2W" | "F2V" => Self::Linear(Linearised::Freq),
            "W2F" | "W2V" => Self::Linear(Linearised::Wave),
            "V2F" | "V2W" => Self::Linear(Linearised::Velo),
            _ => return None,
        })
    }
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
    /// `CRVAL` value of `S` in **SI units**.
    pub crval_si: f64,
    /// Rest frequency (Hz), if supplied via `RESTFRQ`.
    pub restfrq: Option<f64>,
    /// Rest wavelength (m), if supplied via `RESTWAV`.
    pub restwav: Option<f64>,
    /// Multiplier converting CUNIT -> SI (e.g. 1e9 for `GHz`).
    pub unit_to_si: f64,
}

impl SpectralAxis {
    /// Build a spectral axis from its parsed pieces.
    ///
    /// `crval_user` and `cunit` are taken as the user-supplied
    /// CRVAL/CUNIT for this axis; they are converted to SI via
    /// [`to_si_factor`].
    pub fn new(
        axis: usize,
        kind: SpectralKind,
        algorithm: Option<SpectralAlgorithm>,
        crval_user: f64,
        cunit: &str,
        restfrq_hz: Option<f64>,
        restwav_m: Option<f64>,
    ) -> Result<Self> {
        let unit_to_si = to_si_factor(kind, cunit);
        let crval_si = crval_user * unit_to_si;
        // Validate: non-linear algorithms whose linearised variable
        // crosses spectral classes require a rest quantity.
        if let Some(SpectralAlgorithm::Linear(lx)) = algorithm {
            let want_rest = matches!(
                (kind.class(), lx),
                (SpectralClass::F | SpectralClass::W, Linearised::Velo)
                    | (SpectralClass::V, Linearised::Freq | Linearised::Wave)
            );
            if want_rest && restfrq_hz.is_none() && restwav_m.is_none() {
                return Err(FitsError::Wcs(format!(
                    "spectral axis {} ({:?}-{:?}) requires RESTFRQ or RESTWAV",
                    axis + 1,
                    kind,
                    lx,
                )));
            }
        }
        // VRAD/VOPT/ZOPT/VELO/BETA without a rest quantity is
        // meaningless (the user-facing value cannot be computed).
        if matches!(
            kind,
            SpectralKind::Vrad
                | SpectralKind::Vopt
                | SpectralKind::Zopt
                | SpectralKind::Velo
                | SpectralKind::Beta
        ) && restfrq_hz.is_none()
            && restwav_m.is_none()
        {
            return Err(FitsError::Wcs(format!(
                "spectral axis {} (CTYPE {:?}) requires RESTFRQ or RESTWAV",
                axis + 1,
                kind,
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
        })
    }

    /// Forward: intermediate world coord (in CUNIT, relative to
    /// CRVAL) -> user-facing world value `S` (in CUNIT).
    pub fn intermediate_to_world(&self, w_user: f64) -> Result<f64> {
        let w_si = w_user * self.unit_to_si;
        let s_si = match self.algorithm {
            None => self.crval_si + w_si,
            Some(SpectralAlgorithm::Log) => {
                // Paper III Sec.5.1: S = S_r * exp(w / S_r).
                if self.crval_si == 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral -LOG axis: CRVAL must be non-zero".into(),
                    ));
                }
                self.crval_si * (w_si / self.crval_si).exp()
            }
            Some(SpectralAlgorithm::Linear(lin)) => {
                let f_r = self.f_at_reference()?;
                let x_r = self.linearised_from_freq(lin, f_r);
                let dxds = self.dxds_at_reference(lin, f_r)?;
                let x = x_r + dxds * w_si;
                let f = self.freq_from_linearised(lin, x)?;
                self.s_from_freq(f)?
            }
        };
        Ok(s_si / self.unit_to_si)
    }

    /// Inverse: user-facing world value `S` (in CUNIT) -> intermediate
    /// world coord (in CUNIT, relative to CRVAL).
    pub fn world_to_intermediate(&self, s_user: f64) -> Result<f64> {
        let s_si = s_user * self.unit_to_si;
        let w_si = match self.algorithm {
            None => s_si - self.crval_si,
            Some(SpectralAlgorithm::Log) => {
                if self.crval_si == 0.0 || s_si <= 0.0 || self.crval_si <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral -LOG inverse: CRVAL and S must be positive".into(),
                    ));
                }
                self.crval_si * (s_si / self.crval_si).ln()
            }
            Some(SpectralAlgorithm::Linear(lin)) => {
                let f_r = self.f_at_reference()?;
                let x_r = self.linearised_from_freq(lin, f_r);
                let dxds = self.dxds_at_reference(lin, f_r)?;
                let f = self.freq_from_s(s_si)?;
                let x = self.linearised_from_freq(lin, f);
                if dxds == 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral inverse: dX/dS at reference is zero".into(),
                    ));
                }
                (x - x_r) / dxds
            }
        };
        Ok(w_si / self.unit_to_si)
    }

    // ---- internal: F <-> S converters (all SI) ------------------------

    /// `S -> F` for the user's coordinate type.
    fn freq_from_s(&self, s: f64) -> Result<f64> {
        Ok(match self.kind {
            SpectralKind::Freq => s,
            SpectralKind::Ener => s / PLANCK,
            SpectralKind::Wavn => s * SPEED_OF_LIGHT,
            SpectralKind::Wave | SpectralKind::Awav => {
                if s <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral: wavelength must be positive".into(),
                    ));
                }
                SPEED_OF_LIGHT / s
            }
            SpectralKind::Vrad => {
                let f0 = self.rest_freq()?;
                f0 * (1.0 - s / SPEED_OF_LIGHT)
            }
            SpectralKind::Vopt => {
                let f0 = self.rest_freq()?;
                f0 / (1.0 + s / SPEED_OF_LIGHT)
            }
            SpectralKind::Zopt => {
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
            SpectralKind::Wave | SpectralKind::Awav => {
                if f <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral: frequency must be positive".into(),
                    ));
                }
                SPEED_OF_LIGHT / f
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

    fn linearised_from_freq(&self, lin: Linearised, f: f64) -> f64 {
        match lin {
            Linearised::Freq => f,
            Linearised::Wave => SPEED_OF_LIGHT / f,
            Linearised::Velo => {
                // Apparent velocity of radiation at frequency f
                // relative to the rest frequency, m/s.
                let f0 = self.restfrq.expect("validated in new()");
                let r2 = (f / f0).powi(2);
                SPEED_OF_LIGHT * (1.0 - r2) / (1.0 + r2)
            }
        }
    }

    fn freq_from_linearised(&self, lin: Linearised, x: f64) -> Result<f64> {
        Ok(match lin {
            Linearised::Freq => x,
            Linearised::Wave => {
                if x <= 0.0 {
                    return Err(FitsError::Wcs(
                        "spectral: linearised wavelength must be positive".into(),
                    ));
                }
                SPEED_OF_LIGHT / x
            }
            Linearised::Velo => {
                let beta = x / SPEED_OF_LIGHT;
                if beta.abs() >= 1.0 {
                    return Err(FitsError::Wcs(
                        "spectral: linearised |v| must be < c".into(),
                    ));
                }
                let f0 = self.restfrq.ok_or_else(|| {
                    FitsError::Wcs("spectral V2*: RESTFRQ required to invert".into())
                })?;
                f0 * ((1.0 - beta) / (1.0 + beta)).sqrt()
            }
        })
    }

    /// `(dX/dS)|_r` evaluated at the reference frequency.
    fn dxds_at_reference(&self, lin: Linearised, f_r: f64) -> Result<f64> {
        // Chain rule: dX/dS = (dX/dF) * (dF/dS).
        let dxdf = match lin {
            Linearised::Freq => 1.0,
            Linearised::Wave => -SPEED_OF_LIGHT / (f_r * f_r),
            Linearised::Velo => {
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
            SpectralKind::Wave | SpectralKind::Awav => {
                let w_r = SPEED_OF_LIGHT / f_r;
                -SPEED_OF_LIGHT / (w_r * w_r)
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

/// Multiplier converting a value in `cunit` to the canonical SI unit
/// for the given spectral kind. Unknown units pass through (factor 1).
#[must_use]
#[allow(
    clippy::match_same_arms,
    reason = "explicit SI base-unit arms document recognized units even when their conversion factor equals the wildcard's"
)]
pub fn to_si_factor(kind: SpectralKind, cunit: &str) -> f64 {
    let u = cunit.trim();
    if u.is_empty() {
        return 1.0;
    }
    let lower = u.to_ascii_lowercase();
    match kind.class() {
        SpectralClass::F => match lower.as_str() {
            "hz" => 1.0,
            "khz" => 1e3,
            "mhz" => 1e6,
            "ghz" => 1e9,
            "thz" => 1e12,
            // ENER:
            "j" => 1.0,
            "ev" => 1.602_176_634e-19,
            "kev" => 1.602_176_634e-16,
            "mev" => 1.602_176_634e-13,
            // WAVN:
            "1/m" | "m**-1" | "m^-1" => 1.0,
            "1/cm" | "cm**-1" | "cm^-1" => 100.0,
            _ => 1.0,
        },
        SpectralClass::W => match lower.as_str() {
            "m" => 1.0,
            "cm" => 1e-2,
            "mm" => 1e-3,
            "um" | "micron" | "microns" => 1e-6,
            "nm" => 1e-9,
            "angstrom" | "angstroms" | "a" | "ang" => 1e-10,
            "pm" => 1e-12,
            _ => 1.0,
        },
        SpectralClass::V => match lower.as_str() {
            "m/s" => 1.0,
            "km/s" => 1e3,
            "" => 1.0,
            _ => 1.0,
        },
    }
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
        let ax = SpectralAxis::new(2, SpectralKind::Freq, None, 1.420e9, "Hz", None, None).unwrap();
        let s = ax.intermediate_to_world(1e6).unwrap();
        approx(s, 1.421e9, 1e-15);
        let w = ax.world_to_intermediate(1.421e9).unwrap();
        approx(w, 1e6, 1e-12);
    }

    #[test]
    fn cunit_ghz_converts_to_si() {
        let ax = SpectralAxis::new(2, SpectralKind::Freq, None, 1.420, "GHz", None, None).unwrap();
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
        let ax = SpectralAxis::new(2, SpectralKind::Vrad, None, 0.0, "m/s", Some(1.420e9), None)
            .unwrap();
        // V -> F -> V identity at known points.
        // 1 km/s offset from rest frequency.
        let v = ax.intermediate_to_world(1e3).unwrap();
        approx(v, 1e3, 1e-15);
    }

    #[test]
    fn missing_rest_for_velocity_is_error() {
        let r = SpectralAxis::new(2, SpectralKind::Vopt, None, 0.0, "m/s", None, None);
        assert!(r.is_err());
    }
}
