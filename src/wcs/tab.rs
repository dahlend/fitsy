//! Tabular WCS axes (Greisen et al. 2006, Paper III Sec.6).
//!
//! A `-TAB` axis encodes its world coordinates in a separate
//! `BINTABLE` extension instead of via a closed-form algorithm. The
//! header carries:
//!
//! - `CTYPE<i><a> = '<kind>-TAB'` -- flags the axis as tabular.
//! - `PS<i>_0<a>` -- `EXTNAME` of the binary table extension that
//!   holds the lookup arrays (required).
//! - `PS<i>_1<a>` -- `TTYPE` of the column carrying the **coordinate
//!   array** (required).
//! - `PS<i>_2<a>` -- `TTYPE` of the column carrying the optional
//!   **index array** (a 1-D mapping from intermediate world
//!   coordinate to a fractional row index). Absent -> identity.
//! - `PV<i>_1<a>` -- `EXTVER` of the binary table (default 1).
//! - `PV<i>_2<a>` -- `EXTLEVEL` (default 1; not used here).
//! - `PV<i>_3<a>` -- which axis of a multi-dimensional coordinate
//!   array this WCS axis indexes (1-based, default 1). For the
//!   single-axis-per-table case this is always 1.
//!
//! ## Scope
//!
//! This implementation supports the **single-axis 1-D `-TAB`** case:
//! one WCS axis per binary table, coordinate array stored as a
//! single 1-D column of length *K*. That is by far the most common
//! real-world use (irregular wavelength / frequency / time grids in
//! spectra, time-series cubes), and matches what astropy's
//! `WCS.from_header` accepts without warnings for non-celestial
//! tabular axes.
//!
//! The full multi-dimensional `-TAB` algorithm (one binary table
//! shared by several WCS axes, *N*-D coordinate array, intervening
//! `TTYPE` for the per-axis index) is **not** implemented. Such
//! headers are detected and rejected with a clear error rather than
//! silently mis-interpreted.

use crate::error::{FitsError, Result};

/// Header-level description of a `-TAB` axis. Populated by the
/// parser; resolved into a [`TabAxis`] once the binary table is
/// actually loaded.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TabSpec {
    /// Zero-based axis index in the WCS pipeline.
    pub axis: usize,
    /// `PS<i>_0<a>` -- `EXTNAME` of the binary table.
    pub extname: String,
    /// `PS<i>_1<a>` -- column with the coordinate array.
    pub coord_column: String,
    /// `PS<i>_2<a>` -- optional column with the index array.
    pub index_column: Option<String>,
    /// `PV<i>_1<a>` -- `EXTVER` (default 1).
    pub extver: i64,
    /// `PV<i>_3<a>` -- axis number within a multi-D coordinate
    /// array (1-based; default 1). For single-axis tables this is 1.
    pub coord_axis: u32,
}

/// A resolved tabular WCS axis: parsed metadata plus the actual
/// lookup arrays loaded from the referenced binary table.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TabAxis {
    /// Zero-based axis index in the WCS pipeline.
    pub axis: usize,
    /// Coordinate array (length *K*, monotonic-by-construction or
    /// not -- we don't enforce monotonicity for the forward map; the
    /// inverse requires a strictly monotonic array and errors
    /// otherwise).
    pub coord: Vec<f64>,
    /// Optional index array (length *K*). When present, an
    /// intermediate-world value `psi` is first mapped to a
    /// fractional array index via `index -> 0..K-1` linear lookup;
    /// when absent, the intermediate value itself **is** the
    /// 1-based array index per Paper III Sec.6 eq. 6.
    pub index: Option<Vec<f64>>,
}

impl TabAxis {
    /// Forward map: intermediate world coordinate -> world coordinate.
    /// `psi` is the **un-shifted** intermediate value (i.e. before
    /// adding `CRVAL`); per Paper III the table lookup is performed
    /// on the linear-pipeline output and the result is the final
    /// world coordinate (no further `CRVAL` addition).
    ///
    /// Out-of-range values are linearly extrapolated from the two
    /// nearest table samples. Astropy's behaviour here is to
    /// extrapolate; we match it rather than erroring, because
    /// pixels near the edge of an image routinely sit a fraction
    /// of a row off the tabulated range.
    pub fn forward(&self, psi: f64) -> Result<f64> {
        if self.coord.is_empty() {
            return Err(FitsError::Wcs(format!(
                "TAB axis {}: coordinate array is empty",
                self.axis,
            )));
        }
        // Step 1: psi -> fractional row index `c` in 0..K-1.
        let c = match &self.index {
            // Index array gives `index_array[c] = psi` for some c.
            Some(idx) => interp_inverse(idx, psi)?,
            // No index array: per Paper III eq. 6, the intermediate
            // coordinate **is** the (1-based) array index. Convert
            // to our 0-based representation.
            None => psi - 1.0,
        };
        // Step 2: linear interpolation in the coordinate array.
        Ok(interp_lookup(&self.coord, c))
    }

    /// Inverse map: world -> intermediate world. Requires the
    /// coordinate array to be strictly monotonic; binary search is
    /// then bracketed and finished with one linear interpolation.
    pub fn inverse(&self, world: f64) -> Result<f64> {
        if self.coord.len() < 2 {
            return Err(FitsError::Wcs(format!(
                "TAB axis {}: cannot invert with fewer than 2 samples",
                self.axis,
            )));
        }
        let c = interp_inverse(&self.coord, world)?;
        // c is now a fractional index into the coordinate array.
        // Map back through the index array if present, else add 1
        // to recover the 1-based intermediate coordinate.
        match &self.index {
            Some(idx) => Ok(interp_lookup(idx, c)),
            None => Ok(c + 1.0),
        }
    }
}

/// Linear interpolation: given an array `a` of length K, return the
/// value at fractional 0-based index `c`. Out-of-range `c` is
/// linearly extrapolated from the two endpoint samples.
fn interp_lookup(a: &[f64], c: f64) -> f64 {
    let k = a.len();
    if k == 1 {
        return a[0];
    }
    // Clamp the *integer* anchor to the interior so the slope is
    // well-defined; the fractional offset can run negative or > 1
    // (extrapolation).
    let i_floor = c.floor();
    let mut i = i_floor as isize;
    if i < 0 {
        i = 0;
    } else if i >= (k as isize) - 1 {
        i = (k as isize) - 2;
    }
    let frac = c - (i as f64);
    let lo = a[i as usize];
    let hi = a[i as usize + 1];
    lo + frac * (hi - lo)
}

/// Inverse of [`interp_lookup`]: given a strictly-monotonic array
/// `a` and a target value `v`, return the fractional 0-based index
/// `c` such that `interp_lookup(a, c) ~= v`. Bracketing uses binary
/// search on the monotone direction. Out-of-range `v` extrapolates
/// from the nearest segment.
fn interp_inverse(a: &[f64], v: f64) -> Result<f64> {
    let k = a.len();
    if k < 2 {
        return Err(FitsError::Wcs(
            "TAB inverse: lookup array needs at least 2 samples".into(),
        ));
    }
    let ascending = a[k - 1] >= a[0];
    // Detect non-monotonicity early -- otherwise binary search
    // returns silently wrong answers on a wiggly array.
    for w in a.windows(2) {
        let increasing = w[1] >= w[0];
        if increasing != ascending {
            return Err(FitsError::Wcs(
                "TAB inverse: coordinate / index array is not monotonic".into(),
            ));
        }
    }
    // Binary search for the segment containing v.
    let mut lo = 0_usize;
    let mut hi = k - 1;
    while hi - lo > 1 {
        let mid = usize::midpoint(lo, hi);
        let in_lower = if ascending { v < a[mid] } else { v > a[mid] };
        if in_lower {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    // Linear interpolation within the bracketed segment. If v lies
    // outside [a[0], a[k-1]] we still extrapolate from the nearest
    // segment, matching astropy.
    let denom = a[hi] - a[lo];
    if denom == 0.0 {
        return Err(FitsError::Wcs(
            "TAB inverse: degenerate (zero-width) segment".into(),
        ));
    }
    Ok(lo as f64 + (v - a[lo]) / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn forward_no_index_uses_one_based_pixel_index() {
        let tab = TabAxis {
            axis: 0,
            coord: vec![1.0, 2.0, 4.0, 8.0, 16.0],
            index: None,
        };
        // psi = 1 -> 0-based idx 0 -> coord 1.0
        assert!(near(tab.forward(1.0).unwrap(), 1.0, 1e-12));
        // psi = 3.5 -> 0-based idx 2.5 -> midway between 4.0 and 8.0
        assert!(near(tab.forward(3.5).unwrap(), 6.0, 1e-12));
        // psi = 5 -> idx 4 -> coord 16
        assert!(near(tab.forward(5.0).unwrap(), 16.0, 1e-12));
    }

    #[test]
    fn round_trip_no_index() {
        let tab = TabAxis {
            axis: 0,
            coord: vec![100.0, 110.0, 125.0, 150.0, 200.0],
            index: None,
        };
        for psi in [1.0, 1.5, 2.7, 3.0, 4.99] {
            let w = tab.forward(psi).unwrap();
            let back = tab.inverse(w).unwrap();
            assert!(near(back, psi, 1e-9), "psi {psi} -> {w} -> {back}");
        }
    }

    #[test]
    fn round_trip_with_index() {
        // Irregular wavelength grid keyed by integer pixel index.
        let tab = TabAxis {
            axis: 0,
            coord: vec![4000.0, 4500.0, 5500.0, 7000.0, 9000.0],
            index: Some(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        };
        for psi in [1.0, 1.25, 2.5, 3.9, 5.0] {
            let w = tab.forward(psi).unwrap();
            let back = tab.inverse(w).unwrap();
            assert!(near(back, psi, 1e-9), "psi {psi} -> {w} -> {back}");
        }
    }

    #[test]
    fn descending_array_is_monotonic_too() {
        let tab = TabAxis {
            axis: 0,
            coord: vec![9000.0, 7000.0, 5500.0, 4500.0, 4000.0],
            index: None,
        };
        let w = tab.forward(3.0).unwrap();
        assert!(near(w, 5500.0, 1e-12));
        let back = tab.inverse(5500.0).unwrap();
        assert!(near(back, 3.0, 1e-12));
    }

    #[test]
    fn non_monotonic_inverse_errors() {
        let tab = TabAxis {
            axis: 0,
            coord: vec![1.0, 5.0, 2.0, 8.0],
            index: None,
        };
        assert!(tab.inverse(3.0).is_err());
    }

    #[test]
    fn extrapolates_off_the_end() {
        let tab = TabAxis {
            axis: 0,
            coord: vec![10.0, 20.0, 30.0],
            index: None,
        };
        // psi = 0.5 -> 0-based idx -0.5 -> 10 - 5 = 5
        assert!(near(tab.forward(0.5).unwrap(), 5.0, 1e-12));
        // psi = 4 -> idx 3 -> 40 (extrapolated from segment 1..2)
        assert!(near(tab.forward(4.0).unwrap(), 40.0, 1e-12));
    }
}
