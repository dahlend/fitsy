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
use crate::wcs::distortion::sip::Sip;
use crate::wcs::distortion::tnx::Tnx;
use crate::wcs::distortion::tpv::Tpv;
use crate::wcs::projection::Projection;

/// Indices (zero-based) of the celestial-longitude and -latitude axes,
/// plus the frame inferred from their `CTYPE` prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CelestialPair {
    /// Zero-based index of the longitude axis.
    pub lon: usize,
    /// Zero-based index of the latitude axis.
    pub lat: usize,
    /// Frame inferred from the axis-prefix pair.
    pub frame: CelestialFrame,
}

/// Everything that exists if and only if a WCS has a celestial axis
/// pair (Paper II Sec.2). Constructed atomically by the parser.
#[derive(Debug, Clone)]
pub struct CelestialBlock {
    /// Indices of the celestial axis pair.
    pub pair: CelestialPair,
    /// Projection on the tangent plane (TAN, SIN, ZPN, ...).
    pub projection: Projection,
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
    /// Factors converting the longitude and latitude `CUNIT` to
    /// degrees, resolved once at parse time.
    ///
    /// Sec.8.1 requires celestial units to *be* degrees, but headers
    /// do carry `arcsec` and `rad`, so the projection layer needs the
    /// conversion. Resolving it here keeps the Sec.4.3 unit parse off
    /// the per-point transform path.
    pub cunit_to_deg: (f64, f64),
    /// Projection-plane coordinates `(x0, y0)` of the fiducial point,
    /// degrees.
    ///
    /// Normally `(0, 0)`: the fiducial point sits at the projection's
    /// origin. `PVi_1`/`PVi_2` on the longitude axis (Sec.8.2) move it
    /// off that origin, and intermediate coordinates are zero at the
    /// reference point, so they need this offset before projecting.
    pub fiducial_offset: (f64, f64),
}
