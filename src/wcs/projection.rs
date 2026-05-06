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

use crate::wcs::projections;

/// Three-letter projection code (Paper II Table 13).
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
        Ok(match code {
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
    use projections::{
        Air, Ait, Arc, Azp, Bon, Car, Cea, Cod, Coe, Coo, Cop, Csc, Cyp, Hpx, Mer, Mol, Par, Pco,
        Qsc, Sfl, Sin, Stg, Szp, Tan, Tsc, Xph, Zea, Zpn,
    };
    Ok(match kind {
        ProjectionKind::Tan => Box::new(Tan),
        ProjectionKind::Stg => Box::new(Stg),
        ProjectionKind::Sin => Box::new(Sin::from_pv(pv2)?),
        ProjectionKind::Arc => Box::new(Arc),
        ProjectionKind::Zea => Box::new(Zea),
        ProjectionKind::Zpn => Box::new(Zpn::from_pv(pv2)?),
        ProjectionKind::Azp => Box::new(Azp::from_pv(pv2)?),
        ProjectionKind::Car => Box::new(Car),
        ProjectionKind::Cea => Box::new(Cea::from_pv(pv2)?),
        ProjectionKind::Mer => Box::new(Mer),
        ProjectionKind::Cyp => Box::new(Cyp::from_pv(pv2)?),
        ProjectionKind::Sfl => Box::new(Sfl),
        ProjectionKind::Par => Box::new(Par),
        ProjectionKind::Mol => Box::new(Mol),
        ProjectionKind::Ait => Box::new(Ait),
        ProjectionKind::Cop => Box::new(Cop::from_pv(pv2)?),
        ProjectionKind::Coe => Box::new(Coe::from_pv(pv2)?),
        ProjectionKind::Cod => Box::new(Cod::from_pv(pv2)?),
        ProjectionKind::Coo => Box::new(Coo::from_pv(pv2)?),
        ProjectionKind::Bon => Box::new(Bon::from_pv(pv2)?),
        ProjectionKind::Szp => Box::new(Szp::from_pv(pv2)?),
        ProjectionKind::Air => Box::new(Air::from_pv(pv2)?),
        ProjectionKind::Pco => Box::new(Pco),
        ProjectionKind::Hpx => Box::new(Hpx::from_pv(pv2)?),
        ProjectionKind::Xph => Box::new(Xph),
        ProjectionKind::Tsc => Box::new(Tsc),
        ProjectionKind::Csc => Box::new(Csc),
        ProjectionKind::Qsc => Box::new(Qsc),
    })
}
