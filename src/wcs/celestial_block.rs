//! [`CelestialBlock`]: bundles every WCS field that exists if and
//! only if the header carries a celestial axis pair.
//!
//! Splitting these out of [`Wcs`](super::Wcs) lets the type system
//! enforce the all-or-nothing rule. The original layout had five
//! independent `Option` fields (`celestial`, `projection`, `rotation`,
//! `sip`, `tpv`); the parser always populated them as a unit, but the
//! struct allowed e.g. `celestial = Some` and `projection = None`,
//! which the math paths silently treated as "no celestial axes". The
//! grouping makes that state unrepresentable.

use crate::wcs::celestial::{CelestialFrame, CelestialRotation};
use crate::wcs::projection::Projection;
use crate::wcs::sip::Sip;
use crate::wcs::tnx::Tnx;
use crate::wcs::tpv::Tpv;

/// Indices (zero-based) of the celestial-longitude and -latitude axes,
/// plus the frame inferred from their `CTYPE` prefix.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CelestialPair {
    pub lon: usize,
    pub lat: usize,
    pub frame: CelestialFrame,
}

/// Everything that exists if and only if a WCS has a celestial axis
/// pair (Paper II Sec.2). Constructed atomically by the parser.
#[derive(Debug)]
#[non_exhaustive]
pub struct CelestialBlock {
    /// Indices of the celestial axis pair.
    pub pair: CelestialPair,
    /// Projection on the tangent plane (TAN, SIN, ZPN, ...).
    pub projection: Box<dyn Projection>,
    /// Native <-> celestial rotation (LONPOLE/LATPOLE machinery).
    pub rotation: CelestialRotation,
    /// Optional SIP pixel-space distortion (CTYPE suffix `-SIP`).
    pub sip: Option<Sip>,
    /// Optional TPV polynomial in intermediate world coordinates
    /// (CTYPE projection code `TPV`).
    pub tpv: Option<Tpv>,
    /// Optional IRAF TNX/ZPX polynomial distortion in intermediate
    /// world coordinates (CTYPE projection codes `TNX` / `ZPX`,
    /// encoded in the `WAT1_xxx`/`WAT2_xxx` records).
    pub tnx: Option<Tnx>,
}
