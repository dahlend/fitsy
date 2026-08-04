//! Tabular WCS axes (Greisen et al. 2006, Paper III Sec.6).
//!
//! A `-TAB` axis takes its world coordinates from a lookup table in a
//! separate `BINTABLE` extension instead of a closed-form algorithm.
//! The header carries:
//!
//! - `CTYPE<i><a> = '<kind>-TAB'` -- flags the axis as tabular.
//! - `PS<i>_0<a>` -- `EXTNAME` of the binary table (required).
//! - `PS<i>_1<a>` -- `TTYPE` of the coordinate-array column. This is
//!   required, and every axis of a group carries the same value.
//! - `PS<i>_2<a>` -- `TTYPE` of the index-vector column of this axis.
//!   When it is absent, the index runs `1, 2, ... K`.
//! - `PV<i>_1<a>` / `PV<i>_2<a>` -- `EXTVER` / `EXTLEVEL` (default 1).
//! - `PV<i>_3<a>` -- which axis `m` of the coordinate array this WCS
//!   axis indexes (1-based, default 1).
//!
//! ## Separable and non-separable axes
//!
//! The simple case is one WCS axis per table: a coordinate array of
//! length *K*, optionally addressed through an index vector. An
//! irregular wavelength or time grid.
//!
//! Sec.6.1.1 then generalizes to `M` non-separable axes. Those are
//! axes whose coordinates cannot be computed independently of one
//! another. A celestial pair whose longitude depends on both pixel
//! axes is such a case. They share a single `(1 + M)`-dimensional
//! coordinate array of shape `(M, K_1, ..., K_M)`, one index vector
//! each, and are interpolated together: `M`-linear interpolation over
//! `2^M` corners, not *M* separate lookups.
//!
//! Both are handled here by [`TabGroup`], which owns every axis
//! sharing one coordinate array. `M = 1` is simply the one-axis group.
//!
//! ## Array layout
//!
//! `TDIM` is written fastest-axis-first, so `(M, K_1, ..., K_M)` puts
//! the coordinate index `m` innermost:
//!
//! ```text
//! offset(m, i_1, ..., i_M) = m + M * (i_1 + K_1 * (i_2 + K_2 * (...)))
//! ```

#![allow(
    clippy::needless_range_loop,
    reason = "the multilinear blend and its Jacobian index several parallel arrays by axis; named indices read closer to the equations than zipped iterators"
)]

use crate::error::{FitsError, Result};

/// Header-level description of one `-TAB` axis. Populated by the
/// parser; several of these resolve into one [`TabGroup`] once the
/// binary table is loaded.
#[derive(Debug, Clone)]
pub struct TabSpec {
    /// Zero-based axis index in the WCS pipeline.
    pub axis: usize,
    /// `PS<i>_0<a>` -- `EXTNAME` of the binary table.
    pub extname: String,
    /// `PS<i>_1<a>` -- column with the coordinate array.
    pub coord_column: String,
    /// `PS<i>_2<a>` -- optional column with this axis's index vector.
    pub index_column: Option<String>,
    /// `PV<i>_1<a>` -- `EXTVER` (default 1).
    pub extver: i64,
    /// `PV<i>_2<a>` -- `EXTLEVEL` (default 1).
    pub extlevel: i64,
    /// `PV<i>_3<a>` -- this axis's slot `m` in the coordinate array
    /// (1-based; default 1).
    pub coord_axis: u32,
}

impl TabSpec {
    /// The table this axis reads from. Axes sharing one are one group
    /// (Sec.6.2: the coordinate array column must be the same for all
    /// *M* axes).
    pub(crate) fn group_key(&self) -> (String, i64, i64, String) {
        (
            self.extname.clone(),
            self.extver,
            self.extlevel,
            self.coord_column.clone(),
        )
    }
}

/// A resolved lookup table plus the WCS axes it drives.
#[derive(Debug, Clone)]
pub struct TabGroup {
    /// WCS axis index for each coordinate-array slot: `axes[m]` is the
    /// axis whose `PVi_3` is `m + 1`. Length *M*.
    pub axes: Vec<usize>,
    /// `K_1 .. K_M`, the extent of each index axis.
    pub dims: Vec<usize>,
    /// Index vector per slot, each of length `K_m`. `None` means the
    /// implicit `1, 2, ... K_m`, where the intermediate coordinate is
    /// itself the (1-based) array index.
    pub index: Vec<Option<Vec<f64>>>,
    /// The coordinate array, flat, `M * K_1 * ... * K_M` long, laid
    /// out as described in the module docs.
    pub coord: Vec<f64>,
}

impl TabGroup {
    /// Number of coordinate axes sharing this table.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Check the pieces agree before anything relies on them.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the group has no axes, when the axis
    /// list and the index list disagree with the rank, when the
    /// coordinate array length does not match the declared dimensions,
    /// or when an index vector has the wrong length.
    pub(crate) fn validate(&self) -> Result<()> {
        let m = self.dims.len();
        if m == 0 {
            return Err(FitsError::Wcs("-TAB: empty coordinate group".into()));
        }
        if self.axes.len() != m || self.index.len() != m {
            return Err(FitsError::Wcs(format!(
                "-TAB: {m} index axes but {} WCS axes and {} index vectors",
                self.axes.len(),
                self.index.len(),
            )));
        }
        let expect: usize = m * self.dims.iter().product::<usize>();
        if self.coord.len() != expect {
            return Err(FitsError::Wcs(format!(
                "-TAB: coordinate array has {} elements, but TDIM implies {expect} \
                 ({m} x {:?})",
                self.coord.len(),
                self.dims,
            )));
        }
        // Sec.6.1.1 forbids degenerate axes once M > 1; K = 1 stays
        // legal for a lone separable axis, where Sec.6.1.2 spells out
        // the extrapolation rule for it.
        for (d, &k) in self.dims.iter().enumerate() {
            if k == 0 || (m > 1 && k < 2) {
                return Err(FitsError::Wcs(format!(
                    "-TAB: index axis {} has K = {k}; Sec.6.1.1 forbids degenerate \
                     axes when several coordinates share a table",
                    d + 1,
                )));
            }
        }
        for (d, idx) in self.index.iter().enumerate() {
            if let Some(v) = idx
                && v.len() != self.dims[d]
            {
                return Err(FitsError::Wcs(format!(
                    "-TAB: index vector {} has {} entries, but its axis has K = {}",
                    d + 1,
                    v.len(),
                    self.dims[d],
                )));
            }
        }
        Ok(())
    }

    /// Forward: intermediate world coordinates -> world coordinates.
    ///
    /// `psi[d]` is the full intermediate value `CRVAL + x` for the axis
    /// in slot `d`; the returned vector holds `C_m` in the same order.
    ///
    /// # Domain
    ///
    /// Sec.6.1.2 defines each axis over
    /// `0.5 <= Upsilon_m <= K_m + 0.5`. That is the table plus half a
    /// sample step at each end, which covers the outer halves of the
    /// boundary pixels. The coordinate is undefined beyond that range,
    /// so this function reports an error there rather than
    /// extrapolating.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `psi.len()` does not equal the rank of
    /// the group, or when a coordinate falls outside the domain above.
    pub fn forward(&self, psi: &[f64]) -> Result<Vec<f64>> {
        let m = self.rank();
        if psi.len() != m {
            return Err(FitsError::Wcs(format!(
                "-TAB: expected {m} intermediate coordinates, got {}",
                psi.len()
            )));
        }
        if m == 1 {
            return Ok(vec![self.forward_scalar(psi[0])?]);
        }
        let mut upsilon = Vec::with_capacity(m);
        for (d, &p) in psi.iter().enumerate() {
            upsilon.push(self.checked_upsilon(d, p)?);
        }
        Ok(self.interpolate_value(&upsilon))
    }

    /// [`Self::forward`] for a separable group, where `M = 1`. This
    /// form allocates nothing per point.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the group has rank other than 1, or
    /// when `psi` falls outside the domain that [`Self::forward`]
    /// describes.
    pub fn forward_scalar(&self, psi: f64) -> Result<f64> {
        self.require_rank_1("forward_scalar")?;
        let c = self.checked_upsilon(0, psi)?;
        // With M = 1 the coordinate array is the plain length-K
        // vector, so the blend is one segment lookup.
        Ok(interp_lookup(&self.coord, c))
    }

    /// `psi -> Upsilon` for slot `d`, enforcing the Sec.6.1.2 domain
    /// `0.5 <= Upsilon <= K + 0.5` (0-based here: `-0.5..=K - 0.5`).
    fn checked_upsilon(&self, d: usize, psi: f64) -> Result<f64> {
        // Sec.6.1.2's degenerate rule: a single-sample axis holds its
        // one tabulated value for *every* intermediate coordinate, so
        // there is no domain to fall out of and nothing to look up --
        // which also keeps a lone index entry away from
        // `interp_inverse`, whose bracketing needs two samples.
        if self.dims[d] == 1 {
            return Ok(0.0);
        }
        let c = self.psi_to_upsilon(d, psi)?;
        let k = self.dims[d] as f64;
        if !(c >= -0.5 && c <= k - 0.5) {
            return Err(FitsError::Wcs(format!(
                "-TAB axis {}: intermediate coordinate {psi} maps to index {:.6}, \
                 outside the permitted 0.5..={} range (Paper III Sec.6.1.2)",
                self.axes[d] + 1,
                c + 1.0,
                k + 0.5,
            )));
        }
        Ok(c)
    }

    fn require_rank_1(&self, what: &str) -> Result<()> {
        if self.rank() == 1 {
            return Ok(());
        }
        Err(FitsError::Wcs(format!(
            "-TAB: {what} on a group of {} non-separable axes",
            self.rank()
        )))
    }

    /// Inverse: world coordinates -> intermediate world coordinates.
    ///
    /// # Asymmetry with [`Self::forward`]
    ///
    /// This extrapolates without limit, while [`Self::forward`]
    /// refuses to go past the Sec.6.1.2 margin. A world value off the
    /// end of the table therefore yields an index that
    /// [`Self::forward`] would reject.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `world.len()` does not equal the rank
    /// of the group, when the coordinate array is not monotonic along
    /// an axis, or when the multi-dimensional inverse does not
    /// converge.
    // Deliberate. A caller may supply only the celestial pair and let
    // the rest default to `CRVAL`. That value routinely falls outside
    // a `-TAB` axis's tabulated range. This reports the error rather
    // than extrapolating.
    pub fn inverse(&self, world: &[f64]) -> Result<Vec<f64>> {
        let m = self.rank();
        if world.len() != m {
            return Err(FitsError::Wcs(format!(
                "-TAB: expected {m} world coordinates, got {}",
                world.len()
            )));
        }
        if m == 1 {
            return Ok(vec![self.inverse_scalar(world[0])?]);
        }
        let upsilon = self.inverse_multi(world)?;
        Ok((0..m).map(|d| self.upsilon_to_psi(d, upsilon[d])).collect())
    }

    /// [`Self::inverse`] for a separable group.
    ///
    /// This runs an exact monotonic search rather than a Newton
    /// iteration, so it reports a non-monotonic table rather than
    /// resolving it to one of several answers. It extrapolates without
    /// limit, as [`Self::inverse`] describes.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when the group has rank other than 1, or
    /// when the coordinate array is not monotonic.
    pub fn inverse_scalar(&self, world: f64) -> Result<f64> {
        self.require_rank_1("inverse_scalar")?;
        // A single-sample axis is the constant `coord[0]`; every world
        // value maps back to the one tabulated position (Sec.6.1.2),
        // consistent with this function's accept-anything extrapolation
        // on longer tables.
        if self.coord.len() == 1 {
            return Ok(self.upsilon_to_psi(0, 0.0));
        }
        Ok(self.upsilon_to_psi(0, interp_inverse(&self.coord, world)?))
    }

    /// `psi_m -> Upsilon_m` for one slot: interpolate in the index
    /// vector, or take the intermediate coordinate as the 1-based index
    /// directly (Sec.6.1.1). Returned 0-based.
    fn psi_to_upsilon(&self, d: usize, psi: f64) -> Result<f64> {
        match &self.index[d] {
            Some(v) => interp_inverse(v, psi),
            None => Ok(psi - 1.0),
        }
    }

    /// The inverse of [`Self::psi_to_upsilon`].
    fn upsilon_to_psi(&self, d: usize, upsilon: f64) -> f64 {
        match &self.index[d] {
            Some(v) => interp_lookup(v, upsilon),
            None => upsilon + 1.0,
        }
    }

    /// Flat offset of coordinate `m` at grid point `corner`.
    fn offset(&self, corner: &[usize], m: usize) -> usize {
        let mut linear = 0_usize;
        for d in (0..self.rank()).rev() {
            linear = linear * self.dims[d] + corner[d];
        }
        m + self.rank() * linear
    }

    /// Anchor cell and fractional offsets for the blend at `upsilon`.
    ///
    /// The anchor is clamped to the interior so the gradient stays
    /// defined; the fractional offsets may leave `[0, 1]`, which is
    /// what makes the half-step extrapolation of Sec.6.1.2 work.
    fn anchor(&self, upsilon: &[f64], base: &mut [usize], t: &mut [f64]) {
        for d in 0..self.rank() {
            if self.dims[d] == 1 {
                base[d] = 0;
                t[d] = 0.0;
                continue;
            }
            let a = (upsilon[d].floor() as isize).clamp(0, self.dims[d] as isize - 2);
            base[d] = a as usize;
            t[d] = upsilon[d] - a as f64;
        }
    }

    /// The `M`-linear blend of [`Self::interpolate`] without its
    /// Jacobian, for the forward transform, which has no use for one.
    fn interpolate_value(&self, upsilon: &[f64]) -> Vec<f64> {
        let m = self.rank();
        let mut base = vec![0_usize; m];
        let mut t = vec![0.0_f64; m];
        self.anchor(upsilon, &mut base, &mut t);
        let mut value = vec![0.0_f64; m];
        let mut corner = vec![0_usize; m];
        for mask in 0..(1_usize << m) {
            // A degenerate axis has only the lower corner.
            if (0..m).any(|d| self.dims[d] == 1 && (mask >> d) & 1 == 1) {
                continue;
            }
            let mut weight = 1.0_f64;
            for d in 0..m {
                corner[d] = if self.dims[d] == 1 || (mask >> d) & 1 == 0 {
                    base[d]
                } else {
                    base[d] + 1
                };
                weight *= if self.dims[d] == 1 {
                    1.0
                } else if (mask >> d) & 1 == 1 {
                    t[d]
                } else {
                    1.0 - t[d]
                };
            }
            for mm in 0..m {
                value[mm] += weight * self.coord[self.offset(&corner, mm)];
            }
        }
        value
    }

    /// `M`-linear interpolation at `upsilon`, with the Jacobian
    /// `d C_m / d Upsilon_d` alongside it (row-major `M x M`), for the
    /// Newton iteration of [`Self::inverse_multi`].
    fn interpolate(&self, upsilon: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let m = self.rank();
        let mut base = vec![0_usize; m];
        let mut t = vec![0.0_f64; m];
        self.anchor(upsilon, &mut base, &mut t);

        let mut value = vec![0.0_f64; m];
        let mut jacobian = vec![0.0_f64; m * m];
        let mut corner = vec![0_usize; m];
        let mut dweight = vec![0.0_f64; m];
        for mask in 0..(1_usize << m) {
            // A degenerate axis has only the lower corner.
            if (0..m).any(|d| self.dims[d] == 1 && (mask >> d) & 1 == 1) {
                continue;
            }
            // This corner's blend factor per axis: `t` on the upper
            // side, `1 - t` on the lower, and 1 on a degenerate axis
            // that has no upper side to blend towards.
            let factor = |d: usize| {
                if self.dims[d] == 1 {
                    1.0
                } else if (mask >> d) & 1 == 1 {
                    t[d]
                } else {
                    1.0 - t[d]
                }
            };
            let mut weight = 1.0_f64;
            for d in 0..m {
                corner[d] = if self.dims[d] == 1 || (mask >> d) & 1 == 0 {
                    base[d]
                } else {
                    base[d] + 1
                };
                weight *= factor(d);
            }
            // d(weight)/d(upsilon_d) swaps that axis's factor for its
            // derivative, which is +1 on the upper corner and -1 on the
            // lower. Written as the product over the *other* axes
            // rather than `weight / factor(d)`, which is undefined when
            // this corner sits exactly on a grid line. It does not
            // depend on the output coordinate, so it is computed once
            // per corner rather than once per coordinate.
            for d in 0..m {
                if self.dims[d] == 1 {
                    dweight[d] = 0.0;
                    continue;
                }
                let others: f64 = (0..m).filter(|&e| e != d).map(factor).product();
                dweight[d] = if (mask >> d) & 1 == 1 {
                    others
                } else {
                    -others
                };
            }
            for mm in 0..m {
                let c = self.coord[self.offset(&corner, mm)];
                value[mm] += weight * c;
                for d in 0..m {
                    jacobian[mm * m + d] += dweight[d] * c;
                }
            }
        }
        (value, jacobian)
    }

    /// Solve `C(Upsilon) = world` for `M > 1`.
    ///
    /// There is no closed form: the coordinate array is an arbitrary
    /// tabulated map, so this seeds from the nearest grid point and
    /// runs Newton on the `M`-linear interpolant. Tables are small
    /// (Sec.6.2 puts them in a single table cell), so the seed search
    /// is a plain scan.
    fn inverse_multi(&self, world: &[f64]) -> Result<Vec<f64>> {
        let m = self.rank();
        let total: usize = self.dims.iter().product();
        let mut corner = vec![0_usize; m];
        let mut best = vec![0.0_f64; m];
        let mut best_dist = f64::INFINITY;
        for flat in 0..total {
            let mut rest = flat;
            for d in 0..m {
                corner[d] = rest % self.dims[d];
                rest /= self.dims[d];
            }
            let dist: f64 = (0..m)
                .map(|mm| {
                    let diff = self.coord[self.offset(&corner, mm)] - world[mm];
                    diff * diff
                })
                .sum();
            if dist < best_dist {
                best_dist = dist;
                for d in 0..m {
                    best[d] = corner[d] as f64;
                }
            }
        }

        let mut upsilon = best;
        let mut converged = false;
        for _ in 0..100 {
            let (value, jacobian) = self.interpolate(&upsilon);
            let mut residual: Vec<f64> = (0..m).map(|mm| world[mm] - value[mm]).collect();
            let mut a = jacobian;
            if solve_in_place(&mut a, &mut residual, m).is_none() {
                return Err(FitsError::Wcs(
                    "-TAB inverse: the coordinate array is singular here, so the world \
                     coordinate does not determine a unique index"
                        .into(),
                ));
            }
            let mut step = 0.0_f64;
            for d in 0..m {
                upsilon[d] += residual[d];
                step = step.max(residual[d].abs());
            }
            if step < 1e-12 {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(FitsError::Wcs(
                "-TAB inverse: Newton iteration did not converge; the world coordinate \
                 is probably outside the tabulated region"
                    .into(),
            ));
        }
        Ok(upsilon)
    }
}

/// Gauss-Jordan with partial pivoting, in place. `a` is row-major
/// `n x n` and `b` length `n`; on success `b` holds the solution.
/// `None` if the matrix is singular to working precision.
fn solve_in_place(a: &mut [f64], b: &mut [f64], n: usize) -> Option<()> {
    for col in 0..n {
        let (pivot_row, pivot) = (col..n)
            .map(|r| (r, a[r * n + col].abs()))
            .max_by(|x, y| x.1.total_cmp(&y.1))?;
        if pivot < 1e-300 {
            return None;
        }
        if pivot_row != col {
            for c in 0..n {
                a.swap(col * n + c, pivot_row * n + c);
            }
            b.swap(col, pivot_row);
        }
        let diag = a[col * n + col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r * n + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= factor * a[col * n + c];
            }
            b[r] -= factor * b[col];
        }
    }
    for r in 0..n {
        b[r] /= a[r * n + r];
    }
    Some(())
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
    //
    // Equal neighbors are *not* a break: Paper III Sec.6.1.1 permits
    // "two adjacent index values in the vector [to] have the same
    // value", which is how the convention encodes a discontinuity.
    // Testing `w[1] >= w[0]` against `ascending` rejected exactly that
    // on a descending vector, since a repeat reads as increasing there.
    for w in a.windows(2) {
        if w[1] == w[0] {
            continue;
        }
        if (w[1] > w[0]) != ascending {
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

    /// One separable axis: the `M = 1` group.
    fn one_d(coord: Vec<f64>, index: Option<Vec<f64>>) -> TabGroup {
        let k = coord.len();
        TabGroup {
            axes: vec![0],
            dims: vec![k],
            index: vec![index],
            coord,
        }
    }

    #[test]
    fn forward_no_index_uses_one_based_pixel_index() {
        let tab = one_d(vec![1.0, 2.0, 4.0, 8.0, 16.0], None);
        assert!(near(tab.forward(&[1.0]).unwrap()[0], 1.0, 1e-12));
        assert!(near(tab.forward(&[3.5]).unwrap()[0], 6.0, 1e-12));
        assert!(near(tab.forward(&[5.0]).unwrap()[0], 16.0, 1e-12));
    }

    #[test]
    fn round_trip_no_index() {
        let tab = one_d(vec![100.0, 110.0, 125.0, 150.0, 200.0], None);
        for psi in [1.0, 1.5, 2.7, 3.0, 4.99] {
            let w = tab.forward(&[psi]).unwrap();
            let back = tab.inverse(&w).unwrap()[0];
            assert!(near(back, psi, 1e-9), "psi {psi} -> {w:?} -> {back}");
        }
    }

    #[test]
    fn round_trip_with_index() {
        let tab = one_d(
            vec![4000.0, 4500.0, 5500.0, 7000.0, 9000.0],
            Some(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        );
        for psi in [1.0, 1.25, 2.5, 3.9, 5.0] {
            let w = tab.forward(&[psi]).unwrap();
            let back = tab.inverse(&w).unwrap()[0];
            assert!(near(back, psi, 1e-9), "psi {psi} -> {w:?} -> {back}");
        }
    }

    #[test]
    fn descending_array_is_monotonic_too() {
        let tab = one_d(vec![9000.0, 7000.0, 5500.0, 4500.0, 4000.0], None);
        assert!(near(tab.forward(&[3.0]).unwrap()[0], 5500.0, 1e-12));
        assert!(near(tab.inverse(&[5500.0]).unwrap()[0], 3.0, 1e-12));
    }

    #[test]
    fn non_monotonic_inverse_errors() {
        let tab = one_d(vec![1.0, 5.0, 2.0, 8.0], None);
        assert!(tab.inverse(&[3.0]).is_err());
    }

    /// Sec.6.1.2's degenerate rule: a lone `K = 1` axis holds its one
    /// tabulated value for every intermediate coordinate, and every
    /// world value maps back to the single tabulated position.
    #[test]
    fn single_sample_axis_is_a_constant() {
        let tab = one_d(vec![5000.0], None);
        for psi in [-50.0, 0.0, 1.0, 1.5, 100.0] {
            assert!(
                near(tab.forward(&[psi]).unwrap()[0], 5000.0, 1e-12),
                "psi {psi}"
            );
            assert!(near(tab.forward_scalar(psi).unwrap(), 5000.0, 1e-12));
        }
        assert!(near(tab.inverse(&[5000.0]).unwrap()[0], 1.0, 1e-12));
        assert!(near(tab.inverse_scalar(5000.0).unwrap(), 1.0, 1e-12));
        // With an index vector, the tabulated position is its single
        // entry -- and the lookup must not reach `interp_inverse`,
        // whose bracketing needs two samples.
        let indexed = one_d(vec![5000.0], Some(vec![3.0]));
        assert!(near(indexed.forward(&[7.0]).unwrap()[0], 5000.0, 1e-12));
        assert!(near(indexed.inverse(&[5000.0]).unwrap()[0], 3.0, 1e-12));
    }

    /// Paper III Sec.6.1.2 allows half a sample step past each end
    /// (`0.5 <= Upsilon_m <= K + 0.5`) and leaves the coordinate
    /// undefined beyond that.
    #[test]
    fn extrapolates_only_to_the_half_step_limit() {
        let tab = one_d(vec![10.0, 20.0, 30.0], None);
        assert!(near(tab.forward(&[1.0]).unwrap()[0], 10.0, 1e-12));
        assert!(near(tab.forward(&[2.5]).unwrap()[0], 25.0, 1e-12));
        assert!(near(tab.forward(&[3.0]).unwrap()[0], 30.0, 1e-12));
        assert!(near(tab.forward(&[0.5]).unwrap()[0], 5.0, 1e-12));
        assert!(near(tab.forward(&[3.5]).unwrap()[0], 35.0, 1e-12));
        assert!(tab.forward(&[0.4999]).is_err(), "below the lower limit");
        assert!(tab.forward(&[3.5001]).is_err(), "above the upper limit");
        assert!(tab.forward(&[4.0]).is_err(), "a whole step past the end");
        assert!(tab.forward(&[-2.0]).is_err(), "far outside");
    }

    /// Paper III Sec.6.1.1 permits two adjacent index values to be
    /// equal, in either direction.
    #[test]
    fn index_vector_may_repeat_a_value_in_either_direction() {
        let ascending = TabGroup {
            axes: vec![0],
            dims: vec![4],
            index: vec![Some(vec![1.0, 2.0, 2.0, 3.0])],
            coord: vec![10.0, 20.0, 30.0, 40.0],
        };
        assert!(ascending.forward(&[1.5]).is_ok(), "ascending with a repeat");

        let descending = TabGroup {
            axes: vec![0],
            dims: vec![4],
            index: vec![Some(vec![3.0, 2.0, 2.0, 1.0])],
            coord: vec![40.0, 30.0, 20.0, 10.0],
        };
        assert!(
            descending.forward(&[2.5]).unwrap()[0].is_finite(),
            "descending with a legal repeat must be accepted"
        );

        let wiggly = TabGroup {
            axes: vec![0],
            dims: vec![4],
            index: vec![Some(vec![1.0, 3.0, 2.0, 4.0])],
            coord: vec![10.0, 20.0, 30.0, 40.0],
        };
        assert!(
            wiggly.forward(&[2.5]).is_err(),
            "non-monotonic must be refused"
        );
    }

    /// The allocation-free scalar paths must agree exactly with the
    /// general form, including at the extrapolation margins.
    #[test]
    fn scalar_paths_match_the_general_form() {
        let tab = one_d(
            vec![100.0, 110.0, 125.0, 150.0, 200.0],
            Some(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        );
        for psi in [0.5, 1.0, 2.7, 5.0, 5.5] {
            assert_eq!(
                tab.forward(&[psi]).unwrap()[0],
                tab.forward_scalar(psi).unwrap(),
                "psi {psi}"
            );
        }
        assert!(tab.forward_scalar(0.4).is_err(), "same domain limit");
        for w in [100.0, 117.0, 200.0, 260.0] {
            assert_eq!(
                tab.inverse(&[w]).unwrap()[0],
                tab.inverse_scalar(w).unwrap(),
                "world {w}"
            );
        }
        // Both refuse a non-separable group.
        let tab = two_d();
        assert!(tab.forward_scalar(1.0).is_err());
        assert!(tab.inverse_scalar(10.0).is_err());
    }

    // -- multi-dimensional ----------------------------------------------

    /// `(M, K_1, K_2) = (2, 4, 3)`, coordinate `m` innermost. A
    /// deliberately *non-separable* map: each coordinate depends on
    /// both indices, so it cannot be reproduced by two 1-D tables.
    fn two_d() -> TabGroup {
        let (k1, k2, m) = (4_usize, 3_usize, 2_usize);
        let mut coord = vec![0.0; m * k1 * k2];
        for i2 in 0..k2 {
            for i1 in 0..k1 {
                let base = m * (i1 + k1 * i2);
                coord[base] = 10.0 + 0.1 * i1 as f64 + 0.02 * i2 as f64;
                coord[base + 1] = 20.0 + 0.03 * i1 as f64 + 0.1 * i2 as f64;
            }
        }
        TabGroup {
            axes: vec![0, 1],
            dims: vec![k1, k2],
            index: vec![None, None],
            coord,
        }
    }

    #[test]
    fn multi_dimensional_grid_points_read_back_exactly() {
        let tab = two_d();
        tab.validate().unwrap();
        // psi is the 1-based index when there is no index vector.
        assert_eq!(tab.forward(&[1.0, 1.0]).unwrap(), vec![10.0, 20.0]);
        let got = tab.forward(&[4.0, 3.0]).unwrap();
        assert!(near(got[0], 10.34, 1e-12), "{got:?}");
        assert!(near(got[1], 20.29, 1e-12), "{got:?}");
    }

    /// The defining property of the non-separable case: the value
    /// between grid points is the `2^M`-corner blend, so each
    /// coordinate moves with *both* indices.
    #[test]
    fn multi_dimensional_interpolates_bilinearly() {
        let tab = two_d();
        let got = tab.forward(&[1.5, 1.5]).unwrap();
        assert!(near(got[0], 10.0 + 0.05 + 0.01, 1e-12), "{got:?}");
        assert!(near(got[1], 20.0 + 0.015 + 0.05, 1e-12), "{got:?}");
        // Moving only the second index still changes the first
        // coordinate -- that is what "non-separable" means.
        let a = tab.forward(&[2.0, 1.0]).unwrap()[0];
        let b = tab.forward(&[2.0, 3.0]).unwrap()[0];
        assert!((a - b).abs() > 1e-6, "{a} vs {b}");
    }

    #[test]
    fn multi_dimensional_round_trips() {
        let tab = two_d();
        for psi in [[1.0, 1.0], [2.5, 1.75], [3.25, 2.5], [4.0, 3.0]] {
            let w = tab.forward(&psi).unwrap();
            let back = tab.inverse(&w).unwrap();
            assert!(
                near(back[0], psi[0], 1e-8) && near(back[1], psi[1], 1e-8),
                "{psi:?} -> {w:?} -> {back:?}"
            );
        }
    }

    /// Index vectors apply per axis in the multi-dimensional case too.
    #[test]
    fn multi_dimensional_honors_index_vectors() {
        let mut tab = two_d();
        // Second axis sampled at 1, 3, 5 rather than 1, 2, 3.
        tab.index = vec![None, Some(vec![1.0, 3.0, 5.0])];
        tab.validate().unwrap();
        // psi = 3 on the second axis is its *middle* sample now.
        let got = tab.forward(&[1.0, 3.0]).unwrap();
        assert!(near(got[0], 10.02, 1e-12), "{got:?}");
        assert!(near(got[1], 20.1, 1e-12), "{got:?}");
        let back = tab.inverse(&got).unwrap();
        assert!(near(back[1], 3.0, 1e-8), "{back:?}");
    }

    #[test]
    fn validate_catches_a_malformed_group() {
        let mut tab = two_d();
        tab.dims = vec![4, 4];
        assert!(
            tab.validate().is_err(),
            "TDIM product disagrees with the data"
        );

        let mut tab = two_d();
        tab.index = vec![None];
        assert!(tab.validate().is_err(), "one index vector for two axes");

        let mut tab = two_d();
        tab.dims = vec![4, 1];
        tab.coord.truncate(2 * 4);
        assert!(
            tab.validate().is_err(),
            "Sec.6.1.1 forbids a degenerate axis"
        );
    }

    /// The Jacobian `interpolate` hands to Newton must be the true
    /// derivative of the value it returns alongside it -- including
    /// *on* a grid line, where one axis's blend factor is exactly zero
    /// and the naive `weight / factor` form is 0/0.
    #[test]
    fn interpolate_jacobian_matches_finite_differences() {
        let tab = two_d();
        let h = 1e-6;
        for upsilon in [[0.0, 0.0], [1.0, 1.0], [0.4, 1.7], [2.0, 0.0]] {
            let (_, jac) = tab.interpolate(&upsilon);
            for d in 0..2 {
                let (mut lo, mut hi) = (upsilon, upsilon);
                lo[d] -= h;
                hi[d] += h;
                let (v_lo, _) = tab.interpolate(&lo);
                let (v_hi, _) = tab.interpolate(&hi);
                for mm in 0..2 {
                    let fd = (v_hi[mm] - v_lo[mm]) / (2.0 * h);
                    assert!(
                        near(jac[mm * 2 + d], fd, 1e-8),
                        "d C_{mm} / d U_{d} at {upsilon:?}: {} vs {fd}",
                        jac[mm * 2 + d],
                    );
                }
            }
        }
    }

    #[test]
    fn small_linear_solve() {
        // [2 1; 1 3] x = [5; 10]  ->  x = [1; 3]
        let mut a = vec![2.0, 1.0, 1.0, 3.0];
        let mut b = vec![5.0, 10.0];
        solve_in_place(&mut a, &mut b, 2).unwrap();
        assert!(near(b[0], 1.0, 1e-12) && near(b[1], 3.0, 1e-12), "{b:?}");

        let mut a = vec![1.0, 2.0, 2.0, 4.0];
        let mut b = vec![1.0, 2.0];
        assert!(solve_in_place(&mut a, &mut b, 2).is_none(), "singular");
    }
}
