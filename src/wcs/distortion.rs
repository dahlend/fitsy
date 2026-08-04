//! The four distortion conventions outside the `PVi_m` family.
//!
//! Papers I-III define a pixel-to-world chain of three stages. The
//! stages are the linear `CRPIX`/`PCi_j`/`CDELT` step, a projection
//! from Table 13, and a spherical rotation. An instrument adds a
//! residual that the three stages do not describe. Four conventions
//! carry that residual. This module holds all four.
//!
//! # Pipeline position
//!
//! Each convention acts at a different point in the chain. Read this
//! before you change any of them.
//!
//! - [`Sip`] corrects in pixel space (Shupe et al. 2005). It applies
//!   to the `CRPIX` offsets, before the linear matrix.
//! - [`Tpv`] and [`Tnx`] correct in intermediate world space, in
//!   degrees. Both apply after the linear matrix and before the
//!   projection. They share one slot. A header that carries both
//!   applies TPV first.
//! - [`Dss`] replaces the linear stage and the projection stage. It
//!   maps raw pixels to world coordinates on its own. It is a plate
//!   solution, not a correction on the chain.
//!
//! # Design constraints
//!
//! There is no `Distortion` trait. The four take different inputs in
//! different spaces. `Sip` takes pixel offsets. `Tpv` and `Tnx` take
//! intermediate world coordinates. `Dss` takes raw pixels. One
//! interface cannot describe all four correctly.
//!
//! The shared code is the implementation, not the signature. The
//! `poly` triangular evaluator and the `newton` inverse solver are
//! private to this module. All four conventions use them, and nothing
//! outside this module does.

mod newton;
mod poly;

pub mod dss;
pub mod sip;
pub mod tnx;
pub mod tpv;

/// `WAT` keyword reassembly.
///
/// The header parser joins the `WATi_nnn` cards into one string before
/// it builds a [`Tnx`].
pub(crate) mod wat;

pub use dss::Dss;
pub use sip::Sip;
pub use tnx::Tnx;
pub use tpv::Tpv;
