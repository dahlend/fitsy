//! Projection trait and dispatcher (Paper II Sec.8.3).
//!
//! A [`Projection`] maps native spherical coordinates `(phi, theta)` (in
//! degrees) to projection-plane coordinates `(x, y)` (also in degrees,
//! per Paper I) and back. Each projection has a *reference native
//! latitude* `theta0` used by the celestial-rotation defaults
//! (Paper II Sec.2.4).
//!
//! All Paper II projection codes are implemented natively in
//! [`crate::wcs::projections`] (zenithal, cylindrical, pseudo-
//! cylindrical, conic, polyconic, quadrilateralised cube, and
//! `HEALPix`). XPH inverse currently has a known face-disambiguation
//! limitation that is exercised by an `#[ignore]`d test.

use crate::error::{FitsError, Result};

use crate::wcs::projections::{
    Air, Ait, Arc, Azp, Bon, Car, Cea, Cod, Coe, Coo, Cop, Csc, Cyp, Hpx, Mer, Mol, Par, Pco, Qsc,
    Sfl, Sin, Stg, Szp, Tan, Tsc, Xph, Zea, Zpn,
};

/// Three-letter projection code (Paper II Table 13).
///
/// Adding a variant requires updating [`ProjectionKind::code`],
/// [`ProjectionKind::from_code`], and [`build`]. All three are
/// exhaustive matches, so the compiler enforces the invariant
/// directly: omitting any one is a build error, not a runtime panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionKind {
    Azp,
    Szp,
    Tan,
    Stg,
    Sin,
    Arc,
    Zpn,
    Zea,
    Air,
    Cyp,
    Cea,
    Car,
    Mer,
    Sfl,
    Par,
    Mol,
    Ait,
    Cop,
    Coe,
    Cod,
    Coo,
    Bon,
    Pco,
    Tsc,
    Csc,
    Qsc,
    Hpx,
    Xph,
}

impl ProjectionKind {
    pub fn from_code(code: &str) -> Result<Self> {
        Ok(match code.to_uppercase().as_str() {
            "AZP" => Self::Azp,
            "SZP" => Self::Szp,
            "TAN" => Self::Tan,
            "STG" => Self::Stg,
            "SIN" => Self::Sin,
            "ARC" => Self::Arc,
            "ZPN" => Self::Zpn,
            "ZEA" => Self::Zea,
            "AIR" => Self::Air,
            "CYP" => Self::Cyp,
            "CEA" => Self::Cea,
            "CAR" => Self::Car,
            "MER" => Self::Mer,
            "SFL" => Self::Sfl,
            "PAR" => Self::Par,
            "MOL" => Self::Mol,
            "AIT" => Self::Ait,
            "COP" => Self::Cop,
            "COE" => Self::Coe,
            "COD" => Self::Cod,
            "COO" => Self::Coo,
            "BON" => Self::Bon,
            "PCO" => Self::Pco,
            "TSC" => Self::Tsc,
            "CSC" => Self::Csc,
            "QSC" => Self::Qsc,
            "HPX" => Self::Hpx,
            "XPH" => Self::Xph,
            _ => {
                return Err(FitsError::Wcs(format!("unknown projection code `{code}`")));
            }
        })
    }

    /// Three-letter code for this projection (Paper II Table 13).
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::Azp => "AZP",
            Self::Szp => "SZP",
            Self::Tan => "TAN",
            Self::Stg => "STG",
            Self::Sin => "SIN",
            Self::Arc => "ARC",
            Self::Zpn => "ZPN",
            Self::Zea => "ZEA",
            Self::Air => "AIR",
            Self::Cyp => "CYP",
            Self::Cea => "CEA",
            Self::Car => "CAR",
            Self::Mer => "MER",
            Self::Sfl => "SFL",
            Self::Par => "PAR",
            Self::Mol => "MOL",
            Self::Ait => "AIT",
            Self::Cop => "COP",
            Self::Coe => "COE",
            Self::Cod => "COD",
            Self::Coo => "COO",
            Self::Bon => "BON",
            Self::Pco => "PCO",
            Self::Tsc => "TSC",
            Self::Csc => "CSC",
            Self::Qsc => "QSC",
            Self::Hpx => "HPX",
            Self::Xph => "XPH",
        }
    }
}

/// Projection interface: spherical <-> planar.
pub trait Projection: std::fmt::Debug + Send + Sync {
    /// Reference native latitude `theta_0` in degrees (Paper II Sec.2.4).
    fn theta0(&self) -> f64;

    /// Forward: native (phi, theta) -> plane (x, y) in degrees.
    fn s2x(&self, phi_deg: f64, theta_deg: f64) -> Result<(f64, f64)>;

    /// Inverse: plane (x, y) -> native (phi, theta) in degrees.
    fn x2s(&self, x_deg: f64, y_deg: f64) -> Result<(f64, f64)>;
}

/// Construct a projection. `pv2` is the table of `PV2_m` keyword
/// values (`m = 0..`) for the latitude axis. Missing entries are 0.
pub fn build(kind: ProjectionKind, pv2: &[f64]) -> Result<Box<dyn Projection>> {
    use ProjectionKind as K;
    Ok(match kind {
        K::Azp => Box::new(Azp::from_pv(pv2)?),
        K::Szp => Box::new(Szp::from_pv(pv2)?),
        K::Tan => Box::new(Tan),
        K::Stg => Box::new(Stg),
        K::Sin => Box::new(Sin::from_pv(pv2)?),
        K::Arc => Box::new(Arc),
        K::Zpn => Box::new(Zpn::from_pv(pv2)?),
        K::Zea => Box::new(Zea),
        K::Air => Box::new(Air::from_pv(pv2)?),
        K::Cyp => Box::new(Cyp::from_pv(pv2)?),
        K::Cea => Box::new(Cea::from_pv(pv2)?),
        K::Car => Box::new(Car),
        K::Mer => Box::new(Mer),
        K::Sfl => Box::new(Sfl),
        K::Par => Box::new(Par),
        K::Mol => Box::new(Mol),
        K::Ait => Box::new(Ait),
        K::Cop => Box::new(Cop::from_pv(pv2)?),
        K::Coe => Box::new(Coe::from_pv(pv2)?),
        K::Cod => Box::new(Cod::from_pv(pv2)?),
        K::Coo => Box::new(Coo::from_pv(pv2)?),
        K::Bon => Box::new(Bon::from_pv(pv2)?),
        K::Pco => Box::new(Pco),
        K::Tsc => Box::new(Tsc),
        K::Csc => Box::new(Csc),
        K::Qsc => Box::new(Qsc),
        K::Hpx => Box::new(Hpx::from_pv(pv2)?),
        K::Xph => Box::new(Xph),
    })
}

#[cfg(test)]
mod tests {
    use super::ProjectionKind;

    /// All currently-defined `ProjectionKind` variants. Used by the
    /// round-trip test below; the compiler does not enumerate enum
    /// variants for us, so this list has to be kept in sync with the
    /// enum manually. Adding a variant without updating this list
    /// only weakens the test, not the compile-time exhaustiveness
    /// of the three matches above.
    const ALL_KINDS: &[ProjectionKind] = &[
        ProjectionKind::Azp,
        ProjectionKind::Szp,
        ProjectionKind::Tan,
        ProjectionKind::Stg,
        ProjectionKind::Sin,
        ProjectionKind::Arc,
        ProjectionKind::Zpn,
        ProjectionKind::Zea,
        ProjectionKind::Air,
        ProjectionKind::Cyp,
        ProjectionKind::Cea,
        ProjectionKind::Car,
        ProjectionKind::Mer,
        ProjectionKind::Sfl,
        ProjectionKind::Par,
        ProjectionKind::Mol,
        ProjectionKind::Ait,
        ProjectionKind::Cop,
        ProjectionKind::Coe,
        ProjectionKind::Cod,
        ProjectionKind::Coo,
        ProjectionKind::Bon,
        ProjectionKind::Pco,
        ProjectionKind::Tsc,
        ProjectionKind::Csc,
        ProjectionKind::Qsc,
        ProjectionKind::Hpx,
        ProjectionKind::Xph,
    ];

    /// Round-trip every variant through `code()` -> `from_code()`,
    /// pinning the inverse-pair invariant. Exhaustiveness of the
    /// individual matches is already a compile-time guarantee.
    #[test]
    fn projection_code_round_trips() {
        for &kind in ALL_KINDS {
            let code = kind.code();
            let parsed = ProjectionKind::from_code(code).unwrap_or_else(|e| {
                panic!("from_code({code:?}) failed: {e}");
            });
            assert_eq!(parsed, kind, "from_code({code:?}) returned wrong variant");
        }
    }
}
