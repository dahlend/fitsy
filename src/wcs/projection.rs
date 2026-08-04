//! The [`Projection`] enum and its dispatcher (Paper II Sec.8.3).
//!
//! A [`Projection`] maps native spherical coordinates `(phi, theta)` (in
//! degrees) to projection-plane coordinates `(x, y)` (also in degrees,
//! per Paper I) and back. Each projection has a *reference native
//! latitude* `theta0` used by the celestial-rotation defaults
//! (Paper II Sec.2.4).
//!
//! All Paper II projection codes are implemented natively, in one
//! submodule per family:
//!
//! - `zenithal` -- TAN, STG, SIN, ZPN, AZP, ARC, ZEA, SZP and AIR
//!   (Paper II Sec.5.1).
//! - `cylindrical` -- CAR, CEA, MER and CYP (Sec.5.2).
//! - `pseudocyl` -- SFL, PAR, MOL and AIT (Sec.5.3).
//! - `conic` -- COP, COE, COD and COO (Sec.5.4).
//! - `polyconic` -- BON and PCO (Sec.5.5).
//! - `quadcube` -- TSC, CSC and QSC (Sec.5.6).
//! - `healpix` -- HPX and XPH (Calabretta & Roukema 2007).
//!
//! The families group the source files. Table 13 is a flat set of
//! three-letter codes, so this module re-exports every projection
//! type. Reach one as `wcs::projection::Tan`. The XPH inverse has a
//! known face-disambiguation limit. An `#[ignore]`d test records it.
//!
//! # The four methods
//!
//! Each projection type carries the same four methods. The
//! [`Projection`] enum dispatches to them.
//!
//! - `theta0` returns the reference native latitude in degrees.
//! - `pv2` returns the `(m, value)` pairs that rebuild the projection
//!   through [`Projection::from_code`].
//! - `s2x` maps native `(phi, theta)` to plane `(x, y)`.
//! - `x2s` maps plane `(x, y)` back to native `(phi, theta)`.
//!
//! `s2x` and `x2s` both return [`Result`]. The error case is the
//! contract that matters: `s2x` must refuse a point that `x2s` cannot
//! return, or the pair reports a wrong coordinate as a right one. Each
//! method states its own domain.
//!
//! The set of projections is one data-carrying enum rather than a
//! trait object. Table 13 is complete, so a closed set costs nothing.
//! The enum buys three things. Every dispatch site is exhaustive, so
//! no projection can be missed. The three-letter code is a method on
//! the value, so the code and the parameters cannot fall out of step.
//! The per-point dispatch is a match, which the compiler can inline
//! through; it cannot inline through a vtable call.

use crate::error::{FitsError, Result};

mod conic;
mod cylindrical;
mod healpix;
mod polyconic;
mod pseudocyl;
mod quadcube;
mod zenithal;

#[cfg(test)]
mod testing;

pub use conic::{Cod, Coe, Coo, Cop};
pub use cylindrical::{Car, Cea, Cyp, Mer};
pub use healpix::{Hpx, Xph};
pub use polyconic::{Bon, Pco};
pub use pseudocyl::{Ait, Mol, Par, Sfl};
pub use quadcube::{Csc, Qsc, Tsc};
pub use zenithal::{Air, Arc, Azp, Sin, Stg, Szp, Tan, Zea, Zpn};

/// Run `$body` once for the live variant, with `$p` bound to its
/// payload. One definition serves every method below, so a variant
/// cannot be dispatched in one method and forgotten in another.
macro_rules! for_each_projection {
    ($self:expr, $p:ident => $body:expr) => {
        match $self {
            Self::Azp($p) => $body,
            Self::Szp($p) => $body,
            Self::Tan($p) => $body,
            Self::Stg($p) => $body,
            Self::Sin($p) => $body,
            Self::Arc($p) => $body,
            Self::Zpn($p) => $body,
            Self::Zea($p) => $body,
            Self::Air($p) => $body,
            Self::Cyp($p) => $body,
            Self::Cea($p) => $body,
            Self::Car($p) => $body,
            Self::Mer($p) => $body,
            Self::Sfl($p) => $body,
            Self::Par($p) => $body,
            Self::Mol($p) => $body,
            Self::Ait($p) => $body,
            Self::Cop($p) => $body,
            Self::Coe($p) => $body,
            Self::Cod($p) => $body,
            Self::Coo($p) => $body,
            Self::Bon($p) => $body,
            Self::Pco($p) => $body,
            Self::Tsc($p) => $body,
            Self::Csc($p) => $body,
            Self::Qsc($p) => $body,
            Self::Hpx($p) => $body,
            Self::Xph($p) => $body,
        }
    };
}

/// A Paper II Table 13 projection with its parameters resolved.
///
/// Build one with [`Self::from_code`], or wrap a projection struct
/// directly. Every payload type converts with `From`, so
/// `Projection::from(Tan)` and `Tan.into()` both work.
#[derive(Debug, Clone)]
pub enum Projection {
    /// `AZP` -- zenithal/azimuthal perspective (Paper II Sec.5.1.1).
    Azp(Azp),
    /// `SZP` -- slant zenithal perspective (Sec.5.1.2).
    Szp(Szp),
    /// `TAN` -- gnomonic, the tangent plane (Sec.5.1.3).
    Tan(Tan),
    /// `STG` -- stereographic (Sec.5.1.4).
    Stg(Stg),
    /// `SIN` -- orthographic / slant orthographic, the radio
    /// interferometry projection (Sec.5.1.5).
    Sin(Sin),
    /// `ARC` -- zenithal equidistant (Sec.5.1.6).
    Arc(Arc),
    /// `ZPN` -- zenithal polynomial (Sec.5.1.7).
    Zpn(Zpn),
    /// `ZEA` -- zenithal equal-area (Sec.5.1.8).
    Zea(Zea),
    /// `AIR` -- Airy (Sec.5.1.9).
    Air(Air),
    /// `CYP` -- cylindrical perspective (Sec.5.2.1).
    Cyp(Cyp),
    /// `CEA` -- cylindrical equal-area (Sec.5.2.2).
    Cea(Cea),
    /// `CAR` -- plate carree, the equirectangular projection
    /// (Sec.5.2.3).
    Car(Car),
    /// `MER` -- Mercator (Sec.5.2.4).
    Mer(Mer),
    /// `SFL` -- Sanson-Flamsteed, a.k.a. global sinusoidal
    /// (Sec.5.3.1).
    Sfl(Sfl),
    /// `PAR` -- parabolic (Sec.5.3.2).
    Par(Par),
    /// `MOL` -- Mollweide (Sec.5.3.3).
    Mol(Mol),
    /// `AIT` -- Hammer-Aitoff (Sec.5.3.4).
    Ait(Ait),
    /// `COP` -- conic perspective (Sec.5.4.1).
    Cop(Cop),
    /// `COE` -- conic equal-area (Sec.5.4.2).
    Coe(Coe),
    /// `COD` -- conic equidistant (Sec.5.4.3).
    Cod(Cod),
    /// `COO` -- conic orthomorphic (Sec.5.4.4).
    Coo(Coo),
    /// `BON` -- Bonne's equal-area (Sec.5.5.1).
    Bon(Bon),
    /// `PCO` -- polyconic (Sec.5.5.2).
    Pco(Pco),
    /// `TSC` -- tangential spherical cube (Sec.5.6.1).
    Tsc(Tsc),
    /// `CSC` -- COBE quadrilateralized spherical cube (Sec.5.6.2).
    ///
    /// The forward map is a polynomial approximation accurate to about
    /// an arcminute, so a round trip does not return to machine
    /// precision.
    Csc(Csc),
    /// `QSC` -- quadrilateralized spherical cube (Sec.5.6.3).
    Qsc(Qsc),
    /// `HPX` -- `HEALPix` grid (Calabretta & Roukema 2007).
    Hpx(Hpx),
    /// `XPH` -- polar `HEALPix`, the "butterfly" layout (Calabretta &
    /// Roukema 2007 Sec.6).
    Xph(Xph),
}

impl Projection {
    /// Build the projection a three-letter code names, matched
    /// case-insensitively.
    ///
    /// The `pv2` argument is the table of `PV2_m` keyword values for
    /// the latitude axis, indexed by `m` from 0. An entry that no card
    /// supplies holds 0. A projection without parameters ignores it.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the code is not in Table 13, or when
    /// the parameters in `pv2` are invalid for that projection. The
    /// `from_pv` constructor of each projection states its own
    /// conditions. The distortion pseudo-codes (`TPV`, `TNX`, `ZPX`)
    /// are handled a layer up.
    pub fn from_code(code: &str, pv2: &[f64]) -> Result<Self> {
        Ok(match code.to_uppercase().as_str() {
            "AZP" => Self::Azp(Azp::from_pv(pv2)?),
            "SZP" => Self::Szp(Szp::from_pv(pv2)?),
            "TAN" => Self::Tan(Tan),
            "STG" => Self::Stg(Stg),
            "SIN" => Self::Sin(Sin::from_pv(pv2)?),
            "ARC" => Self::Arc(Arc),
            "ZPN" => Self::Zpn(Zpn::from_pv(pv2)?),
            "ZEA" => Self::Zea(Zea),
            "AIR" => Self::Air(Air::from_pv(pv2)?),
            "CYP" => Self::Cyp(Cyp::from_pv(pv2)?),
            "CEA" => Self::Cea(Cea::from_pv(pv2)?),
            "CAR" => Self::Car(Car),
            "MER" => Self::Mer(Mer),
            "SFL" => Self::Sfl(Sfl),
            "PAR" => Self::Par(Par),
            "MOL" => Self::Mol(Mol),
            "AIT" => Self::Ait(Ait),
            "COP" => Self::Cop(Cop::from_pv(pv2)?),
            "COE" => Self::Coe(Coe::from_pv(pv2)?),
            "COD" => Self::Cod(Cod::from_pv(pv2)?),
            "COO" => Self::Coo(Coo::from_pv(pv2)?),
            "BON" => Self::Bon(Bon::from_pv(pv2)?),
            "PCO" => Self::Pco(Pco),
            "TSC" => Self::Tsc(Tsc),
            "CSC" => Self::Csc(Csc),
            "QSC" => Self::Qsc(Qsc),
            "HPX" => Self::Hpx(Hpx::from_pv(pv2)?),
            "XPH" => Self::Xph(Xph),
            _ => {
                return Err(FitsError::Wcs(format!("unknown projection code `{code}`")));
            }
        })
    }

    /// Three-letter code for this projection (Paper II Table 13).
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Azp(_) => "AZP",
            Self::Szp(_) => "SZP",
            Self::Tan(_) => "TAN",
            Self::Stg(_) => "STG",
            Self::Sin(_) => "SIN",
            Self::Arc(_) => "ARC",
            Self::Zpn(_) => "ZPN",
            Self::Zea(_) => "ZEA",
            Self::Air(_) => "AIR",
            Self::Cyp(_) => "CYP",
            Self::Cea(_) => "CEA",
            Self::Car(_) => "CAR",
            Self::Mer(_) => "MER",
            Self::Sfl(_) => "SFL",
            Self::Par(_) => "PAR",
            Self::Mol(_) => "MOL",
            Self::Ait(_) => "AIT",
            Self::Cop(_) => "COP",
            Self::Coe(_) => "COE",
            Self::Cod(_) => "COD",
            Self::Coo(_) => "COO",
            Self::Bon(_) => "BON",
            Self::Pco(_) => "PCO",
            Self::Tsc(_) => "TSC",
            Self::Csc(_) => "CSC",
            Self::Qsc(_) => "QSC",
            Self::Hpx(_) => "HPX",
            Self::Xph(_) => "XPH",
        }
    }

    /// Reference native latitude `theta_0` in degrees
    /// (Paper II Sec.2.4).
    #[must_use]
    pub fn theta0(&self) -> f64 {
        for_each_projection!(self, p => p.theta0())
    }

    /// Forward step, from native `(phi, theta)` to plane `(x, y)`.
    /// Both pairs are in degrees.
    ///
    /// # Errors
    ///
    /// [`crate::FitsError::Wcs`] when the point lies outside the
    /// domain that this projection covers.
    pub fn s2x(&self, phi_deg: f64, theta_deg: f64) -> Result<(f64, f64)> {
        for_each_projection!(self, p => p.s2x(phi_deg, theta_deg))
    }

    /// Inverse step, from plane `(x, y)` to native `(phi, theta)`.
    /// Both pairs are in degrees.
    ///
    /// # Errors
    ///
    /// [`crate::FitsError::Wcs`] when the point lies outside the
    /// region of the plane that this projection fills.
    pub fn x2s(&self, x_deg: f64, y_deg: f64) -> Result<(f64, f64)> {
        for_each_projection!(self, p => p.x2s(x_deg, y_deg))
    }

    /// The `(m, value)` pairs that [`Self::from_code`] needs to
    /// reconstruct this projection, which are the `PV2_m` cards it was
    /// parsed from. A projection without parameters returns an empty
    /// vector.
    ///
    /// A parameterized projection that returned the wrong pairs would
    /// serialize to a header that re-parses with different parameters,
    /// and nothing would report that. The
    /// `from_code_round_trips_every_code` test pins the pairing.
    #[must_use]
    pub fn pv2(&self) -> Vec<(u32, f64)> {
        for_each_projection!(self, p => p.pv2())
    }
}

/// The default is TAN, the projection of a plain undistorted image.
impl Default for Projection {
    fn default() -> Self {
        Self::Tan(Tan)
    }
}

macro_rules! impl_from {
    ($($variant:ident: $ty:ty),* $(,)?) => {$(
        impl From<$ty> for Projection {
            fn from(p: $ty) -> Self {
                Self::$variant(p)
            }
        }
    )*};
}
impl_from!(
    Azp: Azp, Szp: Szp, Tan: Tan, Stg: Stg, Sin: Sin, Arc: Arc, Zpn: Zpn,
    Zea: Zea, Air: Air, Cyp: Cyp, Cea: Cea, Car: Car, Mer: Mer, Sfl: Sfl,
    Par: Par, Mol: Mol, Ait: Ait, Cop: Cop, Coe: Coe, Cod: Cod, Coo: Coo,
    Bon: Bon, Pco: Pco, Tsc: Tsc, Csc: Csc, Qsc: Qsc, Hpx: Hpx, Xph: Xph,
);

#[cfg(test)]
mod tests {
    use super::Projection;

    /// Every Table 13 code, with `PV2_m` values each parameterized
    /// projection accepts. The compiler does not enumerate the codes
    /// for us, so the assertion below checks that the list covers
    /// every variant.
    const ALL_CODES: &[(&str, &[f64])] = &[
        ("AZP", &[0.0, 2.0, 15.0]),
        ("SZP", &[0.0, 2.0, 180.0, 45.0]),
        ("TAN", &[]),
        ("STG", &[]),
        ("SIN", &[0.0, 0.0, 0.0]),
        ("ARC", &[]),
        ("ZPN", &[0.0, 1.0, 0.0, 2.0e-4]),
        ("ZEA", &[]),
        ("AIR", &[0.0, 45.0]),
        ("CYP", &[0.0, 1.0, 1.0]),
        ("CEA", &[0.0, 1.0]),
        ("CAR", &[]),
        ("MER", &[]),
        ("SFL", &[]),
        ("PAR", &[]),
        ("MOL", &[]),
        ("AIT", &[]),
        ("COP", &[0.0, 45.0, 25.0]),
        ("COE", &[0.0, 45.0, 25.0]),
        ("COD", &[0.0, 45.0, 25.0]),
        ("COO", &[0.0, 45.0, 25.0]),
        ("BON", &[0.0, 30.0]),
        ("PCO", &[]),
        ("TSC", &[]),
        ("CSC", &[]),
        ("QSC", &[]),
        ("HPX", &[0.0, 4.0, 3.0]),
        ("XPH", &[]),
    ];

    /// `from_code` -> `code()` is the identity for every Table 13
    /// code, case-insensitively, and the list covers every variant.
    ///
    /// The second half re-parses each projection from its own
    /// `code()` and `pv2()` output and requires the same `pv2()`
    /// back. A projection that reported the wrong pairs would
    /// serialize to a header that re-parses with different
    /// parameters; this is the check the `pv2` documentation names.
    #[test]
    fn from_code_round_trips_every_code() {
        let mut seen = Vec::new();
        for &(code, pv2) in ALL_CODES {
            let p = Projection::from_code(code, pv2)
                .unwrap_or_else(|e| panic!("from_code({code:?}) failed: {e}"));
            assert_eq!(p.code(), code, "code() disagrees for {code}");
            let lower = Projection::from_code(&code.to_lowercase(), pv2).unwrap();
            assert_eq!(lower.code(), code);

            // Rebuild from the projection's own serialization pairs.
            let pairs = p.pv2();
            let n = pairs
                .iter()
                .map(|&(m, _)| m as usize + 1)
                .max()
                .unwrap_or(0);
            let mut table = vec![0.0_f64; n];
            for &(m, v) in &pairs {
                table[m as usize] = v;
            }
            let rebuilt = Projection::from_code(p.code(), &table)
                .unwrap_or_else(|e| panic!("{code}: rebuild from pv2() failed: {e}"));
            assert_eq!(
                rebuilt.pv2(),
                pairs,
                "{code}: pv2() does not survive a rebuild"
            );

            let d = std::mem::discriminant(&p);
            if !seen.contains(&d) {
                seen.push(d);
            }
        }
        assert_eq!(
            seen.len(),
            28,
            "a Table 13 variant is missing from ALL_CODES"
        );
    }

    #[test]
    fn unknown_code_is_rejected() {
        assert!(Projection::from_code("XYZ", &[]).is_err());
    }

    /// A dense sweep over *every* registered projection, built the way
    /// a header would build it, asserting that whatever `s2x` accepts
    /// `x2s` inverts.
    ///
    /// The family modules test each projection against the parameters
    /// its own formulation makes interesting. This one tests the
    /// *table*: it walks `ALL_CODES` so that a projection cannot be
    /// added to the enum and left without a round-trip check.
    #[test]
    fn every_registered_projection_inverts_what_it_accepts() {
        // `CSC` is a polynomial *approximation* (Paper II Sec.5.6.2);
        // its own paper quotes an error near an arcminute, so it gets
        // a matching tolerance rather than a machine-precision one.
        // `SIN`'s limb inverts through `acos`, which loses half the
        // mantissa exactly at `theta = 0`.
        let tol = |code: &str| match code {
            "CSC" => 5e-2,
            "SIN" => 1e-6,
            _ => 1e-9,
        };
        for &(code, pv2) in ALL_CODES {
            let p = Projection::from_code(code, pv2)
                .unwrap_or_else(|e| panic!("{code} failed to build: {e}"));
            super::testing::round_trip_tol(&p, code, tol(code));
        }
    }
}
