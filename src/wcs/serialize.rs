//! Serialize a [`Wcs`] back to a [`Header`].
//!
//! This is the inverse of [`Wcs::from_header`] and is the
//! companion to [`crate::wcs::fit_celestial_wcs`]: after fitting,
//! call [`Wcs::to_header`] to obtain a `Header` that can be merged
//! into an HDU and written to disk.
//!
//! Everything the parser understands is written back:
//!
//! * the linear pipeline -- `CTYPE`, `CUNIT`, `CRVAL`, `CRPIX`, `CDi_j`
//! * `LONPOLE` / `LATPOLE` and the projection's own `PVi_m` table
//! * `RADESYS`, `EQUINOX`, `MJD-OBS`, `WCSNAME`
//! * SIP (`A_`/`B_`/`AP_`/`BP_`), TPV (`PVi_m`), TNX and ZPX
//!   (`WATi_nnn`), and DSS plate solutions
//! * spectral axes (`RESTFRQ` / `RESTWAV`; the rest is CTYPE + CRVAL)
//! * time axes (`TIMESYS`, `TREFPOS`, `TREFDIR`, `PLEPHEM`, and the
//!   per-axis `CZPHS`/`CPERI`/`CRDER`/`CSYER`)
//! * `-TAB` pointer cards (`PSi_m` / `PVi_m`)
//!
//! `tests/wcs.rs::to_header_round_trips_every_projection` pins the
//! contract: parse -> `to_header` -> parse must reproduce the same
//! sky positions for every projection and pole convention.
//!
//! Two things a `Header` cannot carry on its own:
//!
//! * `NAXISi`. No image dimensions are known here, so this emits
//!   zeros. Merge the result into a header that carries the real
//!   values.
//! * The `-TAB` lookup table, which Paper III puts in a separate
//!   `BINTABLE`. The pointer cards name that extension, and the caller
//!   carries it along.
//!
//! `RESTFRQa` and `RESTWAVa` carry the alternate code, as Table 22
//! specifies. An alternate description therefore keeps its own rest
//! quantity.
//!
//! The output is the interpretation, not a reproduction of the source
//! header. The layering note in [`crate::wcs`] explains why. A keyword
//! that the parse dropped as meaningless in context, such as `EQUINOX`
//! under ICRS or a spectral frame with no spectral axis, is not
//! written, because it was never retained. The contract is
//! `from_header(to_header(w)) == w`.
//!
//! Values go out resolved rather than as found, because the Paper II
//! Sec.2.4 defaults depend on the `theta0` of the projection.

use std::fmt::Write as _;

use crate::error::{FitsError, Result};
use crate::header::{Header, Value};
use crate::wcs::celestial::{CelestialFrame, RadeSys};
use crate::wcs::dss::Dss;
use crate::wcs::sip::{Sip, SipPoly};
use crate::wcs::tnx::{Tnx, TnxCrossTerm, TnxFunction, TnxSurface};
use crate::wcs::tpv::Tpv;
use crate::wcs::{Wcs, alt_suffix};

impl Wcs {
    /// Build a fresh [`Header`] holding the WCS keywords of this
    /// object under descriptor `alt`.
    ///
    /// Pass `' '` for `alt` to select the primary description, or a
    /// letter from `A` to `Z` for an alternate.
    ///
    /// Parsing the result reproduces this `Wcs`. The module
    /// documentation names the two pieces that a bare [`Header`]
    /// cannot carry: the `NAXISi` cards, and the `-TAB` lookup
    /// extension.
    ///
    /// # Errors
    ///
    /// [`FitsError::Wcs`] when `alt` is neither a space nor a letter
    /// from `A` to `Z`.
    /// [`crate::FitsError::Header`] when a generated keyword is not a
    /// legal FITS keyword, which happens
    /// when the axis count pushes an indexed keyword past eight
    /// characters.
    pub fn to_header(&self, alt: char) -> Result<Header> {
        let suffix = alt_suffix(alt)?;
        let n = self.naxis();
        let mut h = Header::empty();

        // Inline comments mirror the curated set astropy/wcslib
        // emits from `WCS.to_header()` so the resulting header is
        // self-documenting when inspected by humans or other tools.

        // NAXIS is required by the parser even though it has no
        // alternate-WCS variant. We don't know image dimensions
        // here, so emit zeros for NAXISi -- callers usually merge
        // this header into one that already carries the real
        // values.
        h.push(
            "NAXIS",
            Value::Integer(n as i64),
            Some("Number of coordinate axes"),
        )?;
        for i in 0..n {
            h.push(
                format!("NAXIS{}", i + 1),
                Value::Integer(0),
                Some("length of data axis (placeholder)"),
            )?;
        }

        // CTYPEi / CUNITi / CRVALi. Kept as three passes so the cards
        // stay grouped by keyword, which is how astropy orders them.
        for (i, ax) in self.axes().iter().enumerate() {
            h.push(
                format!("CTYPE{}{}", i + 1, suffix),
                Value::String(ax.ctype.clone()),
                Some(ctype_comment(&ax.ctype)),
            )?;
        }
        for (i, ax) in self.axes().iter().enumerate() {
            if !ax.cunit.is_empty() {
                h.push(
                    format!("CUNIT{}{}", i + 1, suffix),
                    Value::String(ax.cunit.clone()),
                    Some("Units of coordinate increment and value"),
                )?;
            }
        }
        for (i, &v) in self.linear.crval().iter().enumerate().take(n) {
            let unit = self.cunit(i);
            let comment = if unit.is_empty() {
                "Coordinate value at reference point".to_string()
            } else {
                format!("[{unit}] Coordinate value at reference point")
            };
            h.push(
                format!("CRVAL{}{}", i + 1, suffix),
                Value::Real(v),
                Some(&comment),
            )?;
        }

        // CRPIXi.
        let crpix = self.linear.crpix();
        for (i, &v) in crpix.iter().enumerate().take(n) {
            h.push(
                format!("CRPIX{}{}", i + 1, suffix),
                Value::Real(v),
                Some("Pixel coordinate of reference point"),
            )?;
        }

        // CDi_j (preferred over PC + CDELT). Always emit every
        // entry, including zeros -- the parser's defaults differ
        // for "missing" vs "explicitly zero" off-diagonals.
        let m = self.linear.matrix_row_major();
        for i in 0..n {
            for j in 0..n {
                h.push(
                    format!("CD{}_{}{}", i + 1, j + 1, suffix),
                    Value::Real(m[i * n + j]),
                    Some("Coordinate transformation matrix element"),
                )?;
            }
        }

        // LONPOLE / LATPOLE and the projection's own PVi_m table.
        //
        // Written unconditionally, with resolved values: the Sec.2.4
        // defaults depend on the projection's `theta0`, so echoing
        // what we computed avoids re-deriving a default differently.
        //
        // `theta_p` is LATPOLE with the eq. (9) branch already picked,
        // so the reader re-selects the same root. Where `theta0 = 0`
        // both roots are usually valid, and the wrong one moves the
        // sky by tens of degrees.
        if let Some(cb) = &self.celestial {
            let lon_axis = cb.pair.lon + 1;
            let lat_axis = cb.pair.lat + 1;
            h.push(
                format!("LONPOLE{suffix}"),
                Value::Real(cb.rotation.phi_p),
                Some("[deg] Native longitude of celestial pole"),
            )?;
            h.push(
                format!("LATPOLE{suffix}"),
                Value::Real(cb.rotation.theta_p()),
                Some("[deg] Celestial latitude of native pole"),
            )?;
            // Projection parameters ride on the latitude axis
            // (Paper II Sec.2.5), except TPV, which uses both axes.
            for (m, v) in cb.projection.pv2() {
                h.push(
                    format!("PV{lat_axis}_{m}{suffix}"),
                    Value::Real(v),
                    Some("Projection parameter"),
                )?;
            }
            // A moved fiducial point rides on the *longitude* axis
            // (Sec.8.2). Only written when it differs from the
            // projection's origin: TPV and TNX/ZPX claim `PV<lon>_m`
            // for coefficients, which a defaulted card would corrupt.
            if cb.tpv.is_none() && cb.tnx.is_none() {
                if cb.rotation.phi0 != 0.0 {
                    h.push(
                        format!("PV{lon_axis}_1{suffix}"),
                        Value::Real(cb.rotation.phi0),
                        Some("[deg] Native longitude of fiducial point"),
                    )?;
                }
                if cb.rotation.theta0 != cb.projection.theta0() {
                    h.push(
                        format!("PV{lon_axis}_2{suffix}"),
                        Value::Real(cb.rotation.theta0),
                        Some("[deg] Native latitude of fiducial point"),
                    )?;
                }
            }
            if let Some(tpv) = &cb.tpv {
                write_tpv(&mut h, tpv, lon_axis, lat_axis, &suffix)?;
            }
            if let Some(tnx) = &cb.tnx {
                // TNX and ZPX share the `WATi_nnn` carrier and differ
                // only in the underlying projection, so the record
                // has to name whichever one the CTYPE does.
                let wtype = projection_code_of(self.ctype(cb.pair.lat));
                write_tnx(&mut h, tnx, &wtype, lon_axis, lat_axis)?;
            }
        }

        // RADESYS / EQUINOX / MJD-OBS / WCSNAME.
        if let Some(cb) = &self.celestial
            && cb.pair.frame == CelestialFrame::Equatorial
        {
            let rs = match self.radesys {
                RadeSys::Icrs => "ICRS",
                RadeSys::Fk5 => "FK5",
                RadeSys::Fk4 => "FK4",
                RadeSys::Fk4NoE => "FK4-NO-E",
                RadeSys::Gappt => "GAPPT",
                RadeSys::Other => "",
            };
            if !rs.is_empty() {
                h.push(
                    format!("RADESYS{suffix}"),
                    Value::String(rs.into()),
                    Some("Equatorial coordinate system"),
                )?;
            }
        }
        if let Some(eq) = self.equinox {
            h.push(
                format!("EQUINOX{suffix}"),
                Value::Real(eq),
                Some("[yr] Equinox of equatorial coordinates"),
            )?;
        }
        if let Some(mjd) = self.mjd_obs {
            // MJD-OBS has no per-alt suffix in the standard.
            if alt == ' ' {
                h.push("MJD-OBS", Value::Real(mjd), Some("[d] MJD of observation"))?;
            }
        }
        if let Some(name) = &self.wcsname {
            h.push(
                format!("WCSNAME{suffix}"),
                Value::String(name.clone()),
                Some("Coordinate system title"),
            )?;
        }

        // Spectral axes need only their rest quantity; CTYPE, CUNIT
        // and CRVAL are written above, and those four rebuild a
        // `SpectralAxis`.
        //
        // There is one rest-quantity slot however many spectral axes
        // a description declares, so two axes disagreeing is an error:
        // silently keeping the first would re-parse into a different
        // transform.
        if let Some(sx) = self.spectral.first() {
            if let Some(other) = self
                .spectral
                .iter()
                .find(|o| o.restfrq != sx.restfrq || o.restwav != sx.restwav)
            {
                return Err(FitsError::Wcs(format!(
                    "Wcs::to_header: spectral axes {} and {} disagree on the rest \
                     quantity, but one WCS description has a single RESTFRQ/RESTWAV \
                     slot -- it cannot carry both",
                    sx.axis + 1,
                    other.axis + 1,
                )));
            }
            if let Some(f) = sx.restfrq {
                h.push(
                    format!("RESTFRQ{suffix}"),
                    Value::Real(f),
                    Some("[Hz] Line rest frequency"),
                )?;
            }
            if let Some(w) = sx.restwav {
                h.push(
                    format!("RESTWAV{suffix}"),
                    Value::Real(w),
                    Some("[m] Line rest wavelength"),
                )?;
            }
        }

        // Spectral reference frames (Sec.8.4.3). Stored rather than
        // applied, so writing them back is the whole of their
        // contract: dropping one would silently change what frame the
        // coordinates are declared to be in. Parse-time gating means a
        // frame here always has a spectral axis to describe.
        if let Some(frame) = &self.spectral_frame {
            for (key, value, comment) in [
                (
                    "SPECSYS",
                    frame.specsys.as_ref(),
                    "Spectral reference frame",
                ),
                (
                    "SSYSOBS",
                    frame.ssysobs.as_ref(),
                    "Spectral frame of observation",
                ),
                (
                    "SSYSSRC",
                    frame.source.as_ref().and_then(|s| s.ssyssrc.as_ref()),
                    "Reference frame for ZSOURCE",
                ),
            ] {
                if let Some(v) = value {
                    h.push(
                        format!("{key}{suffix}"),
                        Value::String(v.clone()),
                        Some(comment),
                    )?;
                }
            }
            for (key, value, comment) in [
                (
                    "VELOSYS",
                    frame.velosys,
                    "[m/s] Velocity towards the reference frame",
                ),
                (
                    "ZSOURCE",
                    frame.source.as_ref().map(|s| s.zsource),
                    "Redshift of the source",
                ),
                (
                    "VELANGL",
                    frame.source.as_ref().and_then(|s| s.velangl),
                    "[deg] Angle of the true velocity vector",
                ),
            ] {
                if let Some(v) = value {
                    h.push(format!("{key}{suffix}"), Value::Real(v), Some(comment))?;
                }
            }
        }

        // Time-axis keywords. All four are global -- the standard
        // defines no alternate-suffixed forms -- so like `MJD-OBS`
        // they are written from the primary description only; a caller
        // merging `to_header(' ')` with `to_header('A')` must not end
        // up with duplicate cards.
        if let Some(t) = &self.time
            && alt == ' '
        {
            // `TIMESYS` is written only for an axis that *defers* to it
            // (`CTYPE = 'TIME'`), where the scale has nowhere else to
            // live and would otherwise re-parse as the UTC default. A
            // CTYPE naming its own scale (Sec.9.2.1) round-trips
            // through CTYPE alone; copying that scale into TIMESYS
            // would silently change the declared scale of the header's
            // *other* time keywords (`DATE-OBS`, `MJD-OBS`), which
            // TIMESYS, not CTYPE, governs.
            if self.ctype(t.axis).trim().eq_ignore_ascii_case("TIME") {
                let value = match &t.realization {
                    Some(r) => format!("{}({r})", t.scale),
                    None => t.scale.clone(),
                };
                h.push("TIMESYS", Value::String(value), Some("Time scale"))?;
            }
            // Reference frame trio (Sec.9.2.3-9.2.5).
            for (key, value, comment) in [
                ("TREFPOS", t.trefpos.as_ref(), "Time reference position"),
                ("TREFDIR", t.trefdir.as_ref(), "Time reference direction"),
                ("PLEPHEM", t.plephem.as_ref(), "Solar system ephemeris"),
            ] {
                if let Some(v) = value {
                    h.push(key, Value::String(v.clone()), Some(comment))?;
                }
            }
        }
        // Phase axes (Sec.9.6).
        for p in &self.phase {
            let i = p.axis + 1;
            if let Some(v) = p.czphs {
                h.push(
                    format!("CZPHS{i}{suffix}"),
                    Value::Real(v),
                    Some("[s] Phase axis zero point"),
                )?;
            }
            if let Some(v) = p.cperi {
                h.push(
                    format!("CPERI{i}{suffix}"),
                    Value::Real(v),
                    Some("[s] Phase axis period"),
                )?;
            }
        }
        // Per-axis metadata (Sec.8.2 Table 22).
        for (i, m) in self.axes.iter().enumerate() {
            if let Some(v) = &m.cname {
                h.push(
                    format!("CNAME{}{suffix}", i + 1),
                    Value::String(v.clone()),
                    Some("Coordinate axis name"),
                )?;
            }
            if let Some(v) = m.crder {
                h.push(
                    format!("CRDER{}{suffix}", i + 1),
                    Value::Real(v),
                    Some("Random error in coordinate"),
                )?;
            }
            if let Some(v) = m.csyer {
                h.push(
                    format!("CSYER{}{suffix}", i + 1),
                    Value::Real(v),
                    Some("Systematic error in coordinate"),
                )?;
            }
        }

        // Grism dispersers: `PVk_0..PVk_6` on the spectral axis
        // (Paper III Table 7). Unlike the rest quantity these are
        // per-axis, so every grism axis writes its own set. All seven
        // are written rather than only the non-defaults -- a reader
        // cannot tell an omitted `PVk_3` (which defaults to 1) from a
        // dropped one.
        for sx in &self.spectral {
            let Some(g) = sx.grism else { continue };
            let k = sx.axis + 1;
            for (m, value, comment) in [
                (0, g.density, "[m**-1] Grating ruling density"),
                (1, g.order, "Interference order"),
                (2, g.alpha, "[deg] Angle of incidence"),
                (3, g.n_r, "Refractive index at the reference wavelength"),
                (4, g.n_r_prime, "[m**-1] dn/dlambda at the reference"),
                (5, g.epsilon, "[deg] Grating normal out of dispersion plane"),
                (6, g.theta, "[deg] Reference ray to camera axis"),
            ] {
                h.push(
                    format!("PV{k}_{m}{suffix}"),
                    Value::Real(value),
                    Some(comment),
                )?;
            }
        }

        // -TAB lookup axes: write the PSi_/PVi_ cards that point at
        // the companion BINTABLE. The table itself is a separate HDU
        // and cannot travel in a `Header`; a caller writing this out
        // must carry that extension along, exactly as the source file
        // did. See the module docs.
        for spec in &self.tab_specs {
            let i = spec.axis + 1;
            h.push(
                format!("PS{i}_0{suffix}"),
                Value::String(spec.extname.clone()),
                Some("Coordinate table extension name"),
            )?;
            h.push(
                format!("PS{i}_1{suffix}"),
                Value::String(spec.coord_column.clone()),
                Some("Coordinate table column name"),
            )?;
            if let Some(idx) = &spec.index_column {
                h.push(
                    format!("PS{i}_2{suffix}"),
                    Value::String(idx.clone()),
                    Some("Index table column name"),
                )?;
            }
            h.push(
                format!("PV{i}_1{suffix}"),
                Value::Integer(spec.extver),
                Some("Coordinate table extension version"),
            )?;
            // EXTLEVEL is checked against the table HDU when the spec
            // resolves, so dropping it here would break the round trip
            // for any lookup table carrying EXTLEVEL != 1.
            h.push(
                format!("PV{i}_2{suffix}"),
                Value::Integer(spec.extlevel),
                Some("Coordinate table extension level"),
            )?;
            h.push(
                format!("PV{i}_3{suffix}"),
                Value::Integer(i64::from(spec.coord_axis)),
                Some("Axis number in the coordinate array"),
            )?;
        }

        // DSS plate solution. Bypasses the standard pipeline entirely
        // on read, so it is written as its own keyword family.
        if let Some(dss) = &self.dss {
            write_dss(&mut h, dss)?;
        }

        // SIP A_/B_/AP_/BP_ -- only meaningful on alt=' '. The SIP
        // convention does not define alternate-description suffixes.
        if let Some(cb) = &self.celestial
            && let Some(sip) = &cb.sip
            && alt == ' '
        {
            write_sip(&mut h, sip)?;
        }

        Ok(h)
    }
}

/// TPV polynomial: `PV<lon>_m` and `PV<lat>_m`.
///
/// `PVi_1` is always emitted, even when zero: its default is `1.0`,
/// not `0.0` (TPV registry), so omitting a deliberately-zeroed linear
/// term would restore identity scaling on the way back in.
///
/// Emitting `PVi_1` on both axes also removes an ambiguity. A header
/// that carries `PV2_*` alone leaves `PV1_1` to a default that readers
/// resolve differently, and the two readings can differ by several
/// arcseconds. Writing both axes fixes the reading to within 1e-10
/// arcseconds.
fn write_tpv(
    h: &mut Header,
    tpv: &Tpv,
    lon_axis: usize,
    lat_axis: usize,
    suffix: &str,
) -> Result<()> {
    for (axis, table) in [(lon_axis, &tpv.pv1), (lat_axis, &tpv.pv2)] {
        for (m, &c) in table.coeffs.iter().enumerate() {
            if c != 0.0 || m == 1 {
                h.push(
                    format!("PV{axis}_{m}{suffix}"),
                    Value::Real(c),
                    Some("TPV distortion coefficient"),
                )?;
            }
        }
    }
    Ok(())
}

/// The lower-case projection code of a CTYPE (`'DEC--ZPX'` ->
/// `"zpx"`), which is the spelling the IRAF `wtype=` token uses.
/// Falls back to `"tnx"` for a CTYPE with no recognizable code,
/// which cannot happen for a WCS carrying TNX surfaces but keeps
/// this total.
fn projection_code_of(ctype: &str) -> String {
    ctype
        .trim()
        .split('-')
        .next_back()
        .filter(|s| !s.is_empty())
        .map_or_else(|| "tnx".to_string(), str::to_ascii_lowercase)
}

/// TNX/ZPX correction surfaces, as IRAF `WATi_nnn` record fragments.
///
/// The records carry no alternate-description suffix: `WATi_nnn` is an
/// IRAF convention that predates alternate WCS keywords and has no
/// suffixed form.
///
/// `wtype` comes from the CTYPE code rather than being hardcoded.
/// Our reader ignores it, so a wrong value round-trips undetected
/// while contradicting the CTYPE for every reader that does read it.
fn write_tnx(
    h: &mut Header,
    tnx: &Tnx,
    wtype: &str,
    lon_axis: usize,
    lat_axis: usize,
) -> Result<()> {
    // `WAT0_001` carries the WCS's overall coordinate system and is
    // part of the convention every IRAF-lineage reader expects to
    // find; a plain image WCS is `system=image`.
    h.push(
        "WAT0_001",
        Value::String("system=image".into()),
        Some("IRAF WCS system"),
    )?;
    for (axis, axtype, key, surface) in [
        (lon_axis, "ra", "lngcor", tnx.lngcor.as_ref()),
        (lat_axis, "dec", "latcor", tnx.latcor.as_ref()),
    ] {
        let Some(s) = surface else { continue };
        let record = format!(
            "wtype={wtype} axtype={axtype} {key} = \"{}\"",
            encode_tnx_surface(s)
        );
        write_wat_record(h, axis, &record)?;
    }
    Ok(())
}

/// Serialize one surface body: `ft ni nj xterms xi_min xi_max
/// eta_min eta_max c0 c1 ...`, the token order `TnxSurface::parse`
/// consumes.
fn encode_tnx_surface(s: &TnxSurface) -> String {
    let ft = match s.function {
        TnxFunction::Chebyshev => 1,
        TnxFunction::Legendre => 2,
        TnxFunction::Polynomial => 3,
    };
    let xt = match s.cross {
        TnxCrossTerm::None => 0,
        TnxCrossTerm::Full => 1,
        TnxCrossTerm::Half => 2,
    };
    let mut out = format!(
        "{}. {}. {}. {}. {:?} {:?} {:?} {:?}",
        ft, s.ni, s.nj, xt, s.xi_min, s.xi_max, s.eta_min, s.eta_max
    );
    for c in &s.coeffs {
        // `{:?}` on f64 emits the shortest representation that
        // round-trips exactly, which is what keeps the coefficients
        // bit-identical across a write/read cycle.
        let _ = write!(out, " {c:?}");
    }
    out
}

/// Split `record` across `WAT<axis>_001`, `WAT<axis>_002`, ... cards.
///
/// # Why every continuation fragment starts with a space
///
/// Readers disagree on how to rejoin the fragments. IRAF concatenates
/// them, relying on its own trailing blanks; ours joins with a single
/// space, because Sec.4.2.1.1 makes trailing blanks insignificant so a
/// conforming parser cannot see IRAF's padding.
///
/// Splitting on a token boundary and dropping the space suits only the
/// second -- a concatenating reader glues `1.0` and `0.00015` into
/// `1.00.00015`. Leading blanks *are* significant, so putting the
/// separator at the head of the next fragment suits both: concatenation
/// is exact, and joining with a space merely doubles one, which the
/// whitespace-tokenizing parser cannot notice.
///
/// Fragments stay short enough for one 80-byte card, since the
/// `CONTINUE` convention would not help an IRAF reader here.
fn write_wat_record(h: &mut Header, axis: usize, record: &str) -> Result<()> {
    const MAX: usize = 60;
    let mut fragment = String::new();
    let mut n = 1;
    let mut flush = |frag: &mut String, n: &mut usize| -> Result<()> {
        if frag.is_empty() {
            return Ok(());
        }
        h.push(
            format!("WAT{axis}_{:03}", *n),
            Value::String(std::mem::take(frag)),
            None,
        )?;
        *n += 1;
        Ok(())
    };
    for token in record.split(' ') {
        if !fragment.is_empty() && fragment.len() + 1 + token.len() > MAX {
            flush(&mut fragment, &mut n)?;
            // Not `is_empty()` below: the separator this token needs
            // has to ride at the *head* of the new fragment, where
            // FITS will preserve it.
            fragment.push(' ');
        } else if !fragment.is_empty() {
            fragment.push(' ');
        }
        fragment.push_str(token);
    }
    flush(&mut fragment, &mut n)
}

/// DSS plate solution: the keyword family `Dss::from_header` reads.
///
/// `PLTRAH/M/S` and `PLTDECSN/D/M/S` are written back in sexagesimal
/// because that is the only form the reader accepts.
fn write_dss(h: &mut Header, dss: &Dss) -> Result<()> {
    let ra_hours = dss.plate_ra / 15.0;
    let rah = ra_hours.trunc();
    let ram = ((ra_hours - rah) * 60.0).trunc();
    let ras = (ra_hours - rah - ram / 60.0) * 3600.0;
    h.push("PLTRAH", Value::Real(rah), Some("[h] Plate center RA"))?;
    h.push("PLTRAM", Value::Real(ram), Some("[min] Plate center RA"))?;
    h.push("PLTRAS", Value::Real(ras), Some("[s] Plate center RA"))?;

    let sign = if dss.plate_dec < 0.0 { "-" } else { "+" };
    let ad = dss.plate_dec.abs();
    let dd = ad.trunc();
    let dm = ((ad - dd) * 60.0).trunc();
    let ds = (ad - dd - dm / 60.0) * 3600.0;
    h.push(
        "PLTDECSN",
        Value::String(sign.into()),
        Some("Plate center Dec sign"),
    )?;
    h.push("PLTDECD", Value::Real(dd), Some("[deg] Plate center Dec"))?;
    h.push(
        "PLTDECM",
        Value::Real(dm),
        Some("[arcmin] Plate center Dec"),
    )?;
    h.push(
        "PLTDECS",
        Value::Real(ds),
        Some("[arcsec] Plate center Dec"),
    )?;

    h.push("PPO3", Value::Real(dss.ppo3), Some("[um] Plate center x"))?;
    h.push("PPO6", Value::Real(dss.ppo6), Some("[um] Plate center y"))?;
    h.push(
        "XPIXELSZ",
        Value::Real(dss.xpixelsz),
        Some("[um] Pixel size in x"),
    )?;
    h.push(
        "YPIXELSZ",
        Value::Real(dss.ypixelsz),
        Some("[um] Pixel size in y"),
    )?;
    h.push(
        "CNPIX1",
        Value::Real(dss.cnpix1),
        Some("Subimage x offset on the plate"),
    )?;
    h.push(
        "CNPIX2",
        Value::Real(dss.cnpix2),
        Some("Subimage y offset on the plate"),
    )?;
    // AMDX1 / AMDY1 gate detection in `Dss::from_header`, so both are
    // emitted unconditionally; the rest only when non-zero.
    for (i, &v) in dss.amdx.iter().enumerate() {
        if v != 0.0 || i == 0 {
            h.push(
                format!("AMDX{}", i + 1),
                Value::Real(v),
                Some("Plate solution x coefficient"),
            )?;
        }
    }
    for (i, &v) in dss.amdy.iter().enumerate() {
        if v != 0.0 || i == 0 {
            h.push(
                format!("AMDY{}", i + 1),
                Value::Real(v),
                Some("Plate solution y coefficient"),
            )?;
        }
    }
    Ok(())
}

fn write_sip(h: &mut Header, sip: &Sip) -> Result<()> {
    h.push(
        "A_ORDER",
        Value::Integer(i64::from(sip.a.order)),
        Some("SIP polynomial order, axis 1, detector to sky"),
    )?;
    h.push(
        "B_ORDER",
        Value::Integer(i64::from(sip.b.order)),
        Some("SIP polynomial order, axis 2, detector to sky"),
    )?;
    write_sip_poly(h, "A", &sip.a)?;
    write_sip_poly(h, "B", &sip.b)?;
    if let (Some(ap), Some(bp)) = (&sip.ap, &sip.bp) {
        h.push(
            "AP_ORDER",
            Value::Integer(i64::from(ap.order)),
            Some("SIP polynomial order, axis 1, sky to detector"),
        )?;
        h.push(
            "BP_ORDER",
            Value::Integer(i64::from(bp.order)),
            Some("SIP polynomial order, axis 2, sky to detector"),
        )?;
        write_sip_poly(h, "AP", ap)?;
        write_sip_poly(h, "BP", bp)?;
    }
    Ok(())
}

fn write_sip_poly(h: &mut Header, prefix: &str, poly: &SipPoly) -> Result<()> {
    let n = (poly.order as usize) + 1;
    for p in 0..n {
        for q in 0..n {
            if p + q > poly.order as usize {
                continue;
            }
            let c = poly.coeffs[p * n + q];
            if c == 0.0 {
                continue;
            }
            h.push(
                format!("{prefix}_{p}_{q}"),
                Value::Real(c),
                Some("SIP distortion coefficient"),
            )?;
        }
    }
    Ok(())
}

/// Human-readable comment for a `CTYPE` value, such as
/// "TAN (gnomonic) projection + SIP distortions". This makes a
/// generated header self-documenting. An unrecognized `CTYPE` yields
/// an empty comment.
fn ctype_comment(ctype: &str) -> &'static str {
    let upper = ctype.trim();
    let has_sip = upper.ends_with("-SIP");
    let core = upper
        .split('-')
        .next_back()
        .unwrap_or("")
        .trim()
        .to_ascii_uppercase();
    match core.as_str() {
        _ if has_sip => "TAN (gnomonic) projection + SIP distortions",
        "TAN" => "TAN (gnomonic) projection",
        "SIN" => "SIN (orthographic) projection",
        "ARC" => "ARC (zenithal equidistant) projection",
        "STG" => "STG (stereographic) projection",
        "ZEA" => "ZEA (zenithal equal-area) projection",
        "AIR" => "AIR (Airy) projection",
        "CAR" => "CAR (plate carree) projection",
        "MER" => "MER (Mercator) projection",
        "AIT" => "AIT (Hammer-Aitoff) projection",
        "MOL" => "MOL (Mollweide) projection",
        "CEA" => "CEA (cylindrical equal-area) projection",
        "TPV" => "TPV (TAN with PV distortions) projection",
        "TNX" | "ZPX" => "IRAF TNX/ZPX projection",
        "TAB" => "Tabular axis (Paper III)",
        _ => "Coordinate axis type",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wcs::WcsFitOptions;
    use crate::wcs::fit_celestial_wcs;

    /// Sky position of a celestial pixel, as `(lon, lat)`.
    ///
    /// The public API takes one pixel value per axis and returns one
    /// world value per axis, both in axis order. This supplies the
    /// celestial pair and holds every other axis at its reference
    /// pixel, which is what these round-trip comparisons assume.
    fn sky_at(wcs: &Wcs, px: f64, py: f64) -> (f64, f64) {
        let (lon, lat) = wcs.celestial_axes().expect("celestial pair");
        // `CRPIX` is 1-based and the API is 0-based, hence the shift.
        let mut point: Vec<f64> = wcs.linear.crpix().iter().map(|c| c - 1.0).collect();
        point[lon] = px;
        point[lat] = py;
        let w = wcs.pixel_to_world(&point).expect("pixel_to_world");
        (w[lon], w[lat])
    }

    fn build_truth(crpix: (f64, f64), crval: (f64, f64), cd: [f64; 4]) -> Wcs {
        // Round-trip through the fitter to obtain a Wcs we know
        // serializes cleanly. Use a tight grid so the fit is
        // numerically perfect and the comparison is well-defined.
        let projection =
            crate::wcs::projection::build(crate::wcs::projection::ProjectionKind::Tan, &[])
                .unwrap();
        let theta0 = projection.theta0();
        let rotation = crate::wcs::celestial::CelestialRotation::new(
            crval.0, crval.1, None, None, 0.0, theta0,
        )
        .unwrap();
        let _ = (theta0, rotation, projection);
        // Use the fitter's `synthesize` path indirectly by building
        // a header and parsing.
        let mut h = Header::empty();
        h.push("NAXIS", Value::Integer(2), None).unwrap();
        h.push("CTYPE1", Value::String("RA---TAN".into()), None)
            .unwrap();
        h.push("CTYPE2", Value::String("DEC--TAN".into()), None)
            .unwrap();
        h.push("CRPIX1", Value::Real(crpix.0), None).unwrap();
        h.push("CRPIX2", Value::Real(crpix.1), None).unwrap();
        h.push("CRVAL1", Value::Real(crval.0), None).unwrap();
        h.push("CRVAL2", Value::Real(crval.1), None).unwrap();
        h.push("CD1_1", Value::Real(cd[0]), None).unwrap();
        h.push("CD1_2", Value::Real(cd[1]), None).unwrap();
        h.push("CD2_1", Value::Real(cd[2]), None).unwrap();
        h.push("CD2_2", Value::Real(cd[3]), None).unwrap();
        Wcs::from_header(&h, ' ').unwrap().unwrap()
    }

    #[test]
    fn round_trip_celestial_only() {
        let crpix = (123.0_f64, 456.0_f64);
        let crval = (45.0_f64, 30.0_f64);
        let scale = 0.5 / 3600.0;
        let cd = [-scale, 0.0, 0.0, scale];
        let truth = build_truth(crpix, crval, cd);
        let header = truth.to_header(' ').unwrap();
        let round = Wcs::from_header(&header, ' ').unwrap().unwrap();
        for (a, b) in [(50.0, 50.0), (200.0, 100.0), (300.0, 600.0)] {
            let (ra1, de1) = sky_at(&truth, a, b);
            let (ra2, de2) = sky_at(&round, a, b);
            assert!(
                (ra1 - ra2).abs() < 1e-12,
                "RA differs at ({a},{b}): {ra1} vs {ra2}"
            );
            assert!(
                (de1 - de2).abs() < 1e-12,
                "Dec differs at ({a},{b}): {de1} vs {de2}"
            );
        }
    }

    /// A minimal celestial header for `code`, with `extra` cards.
    fn distorted_header(code: &str, extra: &[(&str, Value)]) -> Header {
        let mut h = Header::empty();
        h.push("NAXIS", Value::Integer(2), None).unwrap();
        h.push("CTYPE1", Value::String(format!("RA---{code}")), None)
            .unwrap();
        h.push("CTYPE2", Value::String(format!("DEC--{code}")), None)
            .unwrap();
        h.push("CRPIX1", Value::Real(512.0), None).unwrap();
        h.push("CRPIX2", Value::Real(512.0), None).unwrap();
        h.push("CRVAL1", Value::Real(180.0), None).unwrap();
        h.push("CRVAL2", Value::Real(20.0), None).unwrap();
        h.push("CD1_1", Value::Real(-2.8e-4), None).unwrap();
        h.push("CD1_2", Value::Real(0.0), None).unwrap();
        h.push("CD2_1", Value::Real(0.0), None).unwrap();
        h.push("CD2_2", Value::Real(2.8e-4), None).unwrap();
        for (k, v) in extra {
            h.push(*k, v.clone(), None).unwrap();
        }
        h
    }

    const TNX_LNG: &str = "wtype=REPLACE axtype=ra lngcor = \
         \"3. 3. 3. 2. -1. 1. -1. 1. 1.5e-4 -8.2e-5 4.1e-5 2.3e-5 -1.1e-5 6.0e-6 \"";
    const TNX_LAT: &str = "wtype=REPLACE axtype=dec latcor = \
         \"3. 3. 3. 2. -1. 1. -1. 1. -9.0e-5 5.5e-5 -3.3e-5 1.7e-5 8.0e-6 -4.0e-6 \"";

    /// The `WATi_nnn` cards a serialized WCS carries, in order.
    fn wat_fragments(h: &Header, axis: usize) -> Vec<String> {
        let prefix = format!("WAT{axis}_");
        (1..=999)
            .map_while(|n| match h.first(&format!("{prefix}{n:03}")) {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect()
    }

    /// `wtype=` must name the projection the CTYPE names.
    ///
    /// Regression: it was hardcoded to `tnx`, so a `RA---ZPX` WCS
    /// wrote `WAT` records claiming TNX. Our reader ignores `wtype`,
    /// so the round trip passed while the output was wrong for every
    /// reader that does not.
    #[test]
    fn wat_wtype_follows_the_ctype() {
        for (code, extra) in [
            ("TNX", Vec::new()),
            (
                "ZPX",
                vec![("PV2_1", Value::Real(1.0)), ("PV2_3", Value::Real(337.6))],
            ),
        ] {
            let mut cards = extra;
            cards.push((
                "WAT1_001",
                Value::String(TNX_LNG.replace("REPLACE", &code.to_lowercase())),
            ));
            cards.push((
                "WAT2_001",
                Value::String(TNX_LAT.replace("REPLACE", &code.to_lowercase())),
            ));
            let wcs = Wcs::from_header(&distorted_header(code, &cards), ' ')
                .unwrap()
                .unwrap();
            let out = wcs.to_header(' ').unwrap();
            let want = format!("wtype={}", code.to_lowercase());
            for axis in [1, 2] {
                let joined = wat_fragments(&out, axis).join(" ");
                assert!(
                    joined.contains(&want),
                    "{code} WAT{axis} says `{joined}`, expected `{want}`"
                );
            }
            assert!(
                matches!(out.first("WAT0_001"), Some(Value::String(s)) if s == "system=image"),
                "{code}: WAT0_001 missing from the serialized header"
            );
        }
    }

    /// The `WATi_nnn` split must survive both rejoin conventions:
    /// IRAF concatenates, our reader joins with a space. Leading the
    /// continuation fragment with the separator suits both, since FITS
    /// keeps leading blanks but not trailing ones.
    ///
    /// Regression: the split dropped the separator, so a concatenating
    /// reader turned `1.0` + `0.00015` into `1.00.00015`.
    #[test]
    fn wat_fragments_rejoin_under_both_conventions() {
        let cards = vec![
            ("WAT1_001", Value::String(TNX_LNG.replace("REPLACE", "tnx"))),
            ("WAT2_001", Value::String(TNX_LAT.replace("REPLACE", "tnx"))),
        ];
        let wcs = Wcs::from_header(&distorted_header("TNX", &cards), ' ')
            .unwrap()
            .unwrap();
        let out = wcs.to_header(' ').unwrap();
        for axis in [1, 2] {
            let frags = wat_fragments(&out, axis);
            assert!(
                frags.len() > 1,
                "WAT{axis} fits in one card; the split is not exercised"
            );
            let concatenated = frags.concat();
            let space_joined = frags.join(" ");
            // Both readings must recover the same token stream --
            // that is the property the coefficients depend on.
            let a: Vec<&str> = concatenated.split_ascii_whitespace().collect();
            let b: Vec<&str> = space_joined.split_ascii_whitespace().collect();
            assert_eq!(
                a, b,
                "WAT{axis} rejoins differently under the two conventions:\n\
                 concatenated: {concatenated}\nspace-joined: {space_joined}"
            );
        }
    }

    /// TPV and TNX/ZPX must survive a `to_header` -> re-parse cycle.
    ///
    /// Regression: both used to serialize silently *without* their
    /// coefficient cards, leaving a `RA---TPV` CTYPE that re-parsed
    /// as an undistorted TAN and shifted every coordinate.
    #[test]
    fn round_trip_preserves_tpv_and_tnx() {
        for (ctype, extra) in [
            (
                "TPV",
                vec![
                    ("PV1_1", Value::Real(1.0)),
                    ("PV1_4", Value::Real(2.1e-3)),
                    ("PV1_7", Value::Real(-1.2e-2)),
                    ("PV2_1", Value::Real(1.0)),
                    ("PV2_4", Value::Real(-1.7e-3)),
                    ("PV2_7", Value::Real(9.4e-3)),
                ],
            ),
            (
                "TNX",
                vec![
                    ("WAT0_001", Value::String("system=image".into())),
                    (
                        "WAT1_001",
                        Value::String(
                            "wtype=tnx axtype=ra lngcor = \"3. 3. 3. 2. -1. 1. -1. 1. \
                             1.5e-4 -8.2e-5 4.1e-5 2.3e-5 -1.1e-5 6.0e-6 \""
                                .into(),
                        ),
                    ),
                    (
                        "WAT2_001",
                        Value::String(
                            "wtype=tnx axtype=dec latcor = \"3. 3. 3. 2. -1. 1. -1. 1. \
                             -9.0e-5 5.5e-5 -3.3e-5 1.7e-5 8.0e-6 -4.0e-6 \""
                                .into(),
                        ),
                    ),
                ],
            ),
        ] {
            let mut h = Header::empty();
            h.push("NAXIS", Value::Integer(2), None).unwrap();
            h.push("CTYPE1", Value::String(format!("RA---{ctype}")), None)
                .unwrap();
            h.push("CTYPE2", Value::String(format!("DEC--{ctype}")), None)
                .unwrap();
            h.push("CRPIX1", Value::Real(512.0), None).unwrap();
            h.push("CRPIX2", Value::Real(512.0), None).unwrap();
            h.push("CRVAL1", Value::Real(180.0), None).unwrap();
            h.push("CRVAL2", Value::Real(20.0), None).unwrap();
            h.push("CD1_1", Value::Real(-2.8e-4), None).unwrap();
            h.push("CD1_2", Value::Real(0.0), None).unwrap();
            h.push("CD2_1", Value::Real(0.0), None).unwrap();
            h.push("CD2_2", Value::Real(2.8e-4), None).unwrap();
            for (k, v) in extra {
                h.push(k, v, None).unwrap();
            }
            let truth = Wcs::from_header(&h, ' ').unwrap().unwrap();

            // Guard the guard: confirm the distortion actually moves
            // coordinates, so a silent drop would be detectable.
            let mut plain = Header::empty();
            for e in h.entries() {
                let key = match e.keyword.as_str() {
                    "CTYPE1" | "CTYPE2" if ctype == "TNX" => e.keyword.as_str(),
                    k if k.starts_with("PV") || k.starts_with("WAT") => continue,
                    k => k,
                };
                if let Some(v) = &e.value {
                    plain.push(key, v.clone(), None).unwrap();
                }
            }
            if ctype == "TNX" {
                // Without WAT records a TNX CTYPE is just a TAN.
                let undistorted = Wcs::from_header(&plain, ' ').unwrap().unwrap();
                let (ra0, de0) = sky_at(&truth, 100.0, 100.0);
                let (ra1, de1) = sky_at(&undistorted, 100.0, 100.0);
                assert!(
                    (ra0 - ra1).abs() > 1e-6 || (de0 - de1).abs() > 1e-6,
                    "{ctype} fixture has no measurable distortion; the test cannot fail"
                );
            }

            let round = Wcs::from_header(&truth.to_header(' ').unwrap(), ' ')
                .unwrap()
                .unwrap();
            for (px, py) in [
                (10.0, 10.0),
                (512.0, 512.0),
                (900.0, 300.0),
                (1000.0, 1000.0),
            ] {
                let (ra1, de1) = sky_at(&truth, px, py);
                let (ra2, de2) = sky_at(&round, px, py);
                assert!(
                    (ra1 - ra2).abs() < 1e-11 && (de1 - de2).abs() < 1e-11,
                    "{ctype} round-trip drifted at ({px},{py}): \
                     ({ra1},{de1}) vs ({ra2},{de2})"
                );
            }
        }
    }

    #[test]
    fn round_trip_with_sip_from_fit() {
        // Build a known TAN+SIP truth, fit it, serialize, re-parse,
        // and check the re-parsed model agrees with the fit.
        let crpix = (100.0_f64, 100.0_f64);
        let crval = (10.0_f64, 20.0_f64);
        let scale = 0.3 / 3600.0;
        let cd = [scale, 0.0, 0.0, scale];
        let truth = build_truth(crpix, crval, cd);
        // Sample, then fit with SIP.
        let mut pixels = Vec::new();
        let mut sky = Vec::new();
        for i in 0..8 {
            for j in 0..8 {
                let px = 10.0 + 22.0 * f64::from(i);
                let py = 10.0 + 22.0 * f64::from(j);
                let (ra, dec) = sky_at(&truth, px, py);
                pixels.push((px, py));
                sky.push((ra, dec));
            }
        }
        let opts = WcsFitOptions {
            sip_order: Some(3),
            ..Default::default()
        };
        let fit = fit_celestial_wcs(&pixels, &sky, &opts).unwrap();
        let header = fit.wcs.to_header(' ').unwrap();
        let round = Wcs::from_header(&header, ' ').unwrap().unwrap();
        // Compare round-tripped vs fitted at every input pixel.
        for &(px, py) in &pixels {
            let (ra1, de1) = sky_at(&fit.wcs, px, py);
            let (ra2, de2) = sky_at(&round, px, py);
            assert!((ra1 - ra2).abs() < 1e-10);
            assert!((de1 - de2).abs() < 1e-10);
        }
    }
}
