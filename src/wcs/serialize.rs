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
//! * `-TAB` pointer cards (`PSi_m` / `PVi_m`)
//!
//! `tests/wcs.rs::to_header_round_trips_every_projection` pins the
//! contract: parse -> `to_header` -> parse must reproduce the same
//! sky positions for every projection and pole convention.
//!
//! Two things a `Header` cannot carry on its own:
//!
//! * **`NAXISi`.** No image dimensions are known here, so zeros are
//!   emitted; merge into a header that has the real values.
//! * **The `-TAB` lookup table**, which Paper III puts in a separate
//!   BINTABLE. The pointer cards name it, but the caller has to carry
//!   that extension along.
//!
//! `RESTFRQ` / `RESTWAV` are written unsuffixed even for an alternate
//! description. The Standard defines `RESTFRQa` / `RESTWAVa`, but this
//! crate's parser reads only the bare spelling, so a suffixed card
//! would not round-trip -- at the cost that wcslib and astropy read
//! ours as belonging to the primary description.
//!
//! Values are written **resolved** rather than as-found, since the
//! Paper II Sec.2.4 defaults depend on the projection's `theta0`.

use std::fmt::Write as _;

use crate::error::{FitsError, Result};
use crate::header::{Header, Value};
use crate::wcs::Wcs;
use crate::wcs::celestial::{CelestialFrame, RadeSys};
use crate::wcs::dss::Dss;
use crate::wcs::sip::{Sip, SipPoly};
use crate::wcs::tnx::{Tnx, TnxCrossTerm, TnxFunction, TnxSurface};
use crate::wcs::tpv::Tpv;

impl Wcs {
    /// Build a fresh [`Header`] holding the WCS keywords for this
    /// object under the chosen `alt` (`' '` for the primary
    /// description, `'A'..'Z'` for an alternate).
    ///
    /// Re-parsing the result reproduces this `Wcs`. See the
    /// module-level note for the two pieces a bare `Header` cannot
    /// carry (`NAXISi`, and the `-TAB` lookup extension).
    pub fn to_header(&self, alt: char) -> Result<Header> {
        let suffix = alt_suffix(alt)?;
        let n = self.naxis;
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

        // CTYPEi / CUNITi / CRVALi.
        for i in 0..n {
            h.push(
                format!("CTYPE{}{}", i + 1, suffix),
                Value::String(self.ctype[i].clone()),
                Some(ctype_comment(&self.ctype[i])),
            )?;
        }
        for i in 0..n {
            if !self.cunit[i].is_empty() {
                h.push(
                    format!("CUNIT{}{}", i + 1, suffix),
                    Value::String(self.cunit[i].clone()),
                    Some("Units of coordinate increment and value"),
                )?;
            }
        }
        for i in 0..n {
            let v = self.crval[i];
            let unit = self.cunit.get(i).map_or("", String::as_str);
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
                Value::Real(cb.rotation.theta_p),
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
                let wtype = projection_code_of(&self.ctype[cb.pair.lat]);
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
        //
        // Written unsuffixed even for an alternate description; see
        // the module note.
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
                h.push("RESTFRQ", Value::Real(f), Some("[Hz] Line rest frequency"))?;
            }
            if let Some(w) = sx.restwav {
                h.push("RESTWAV", Value::Real(w), Some("[m] Line rest wavelength"))?;
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

fn alt_suffix(alt: char) -> Result<String> {
    if alt == ' ' {
        return Ok(String::new());
    }
    if !alt.is_ascii_uppercase() {
        return Err(FitsError::Wcs(format!(
            "alt must be ' ' or 'A'..'Z' (got {alt:?})"
        )));
    }
    Ok(alt.to_string())
}

/// TPV polynomial: `PV<lon>_m` and `PV<lat>_m`.
///
/// `PVi_1` is always emitted, even when zero: its default is `1.0`,
/// not `0.0` (TPV registry), so omitting a deliberately-zeroed linear
/// term would restore identity scaling on the way back in.
///
/// It also settles a disagreement -- wcslib skips that default when an
/// axis has no `PVi_m` cards at all, so a header with only `PV2_*`
/// reads 3.5 arcsec differently in astropy. Emitting both axes brings
/// the two within 1e-10 arcsec.
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
    h.push("PLTRAH", Value::Real(rah), Some("[h] Plate centre RA"))?;
    h.push("PLTRAM", Value::Real(ram), Some("[min] Plate centre RA"))?;
    h.push("PLTRAS", Value::Real(ras), Some("[s] Plate centre RA"))?;

    let sign = if dss.plate_dec < 0.0 { "-" } else { "+" };
    let ad = dss.plate_dec.abs();
    let dd = ad.trunc();
    let dm = ((ad - dd) * 60.0).trunc();
    let ds = (ad - dd - dm / 60.0) * 3600.0;
    h.push(
        "PLTDECSN",
        Value::String(sign.into()),
        Some("Plate centre Dec sign"),
    )?;
    h.push("PLTDECD", Value::Real(dd), Some("[deg] Plate centre Dec"))?;
    h.push(
        "PLTDECM",
        Value::Real(dm),
        Some("[arcmin] Plate centre Dec"),
    )?;
    h.push(
        "PLTDECS",
        Value::Real(ds),
        Some("[arcsec] Plate centre Dec"),
    )?;

    h.push("PPO3", Value::Real(dss.ppo3), Some("[um] Plate centre x"))?;
    h.push("PPO6", Value::Real(dss.ppo6), Some("[um] Plate centre y"))?;
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

/// Best-effort human-readable comment for a `CTYPE` value. Mirrors
/// the strings wcslib emits (e.g. "TAN (gnomonic) projection +
/// SIP distortions") so produced headers are self-documenting.
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
            let (ra1, de1) = truth.pixel_to_celestial(a, b).unwrap();
            let (ra2, de2) = round.pixel_to_celestial(a, b).unwrap();
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
                let (ra0, de0) = truth.pixel_to_celestial(100.0, 100.0).unwrap();
                let (ra1, de1) = undistorted.pixel_to_celestial(100.0, 100.0).unwrap();
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
                let (ra1, de1) = truth.pixel_to_celestial(px, py).unwrap();
                let (ra2, de2) = round.pixel_to_celestial(px, py).unwrap();
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
                let (ra, dec) = truth.pixel_to_celestial(px, py).unwrap();
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
            let (ra1, de1) = fit.wcs.pixel_to_celestial(px, py).unwrap();
            let (ra2, de2) = round.pixel_to_celestial(px, py).unwrap();
            assert!((ra1 - ra2).abs() < 1e-10);
            assert!((de1 - de2).abs() < 1e-10);
        }
    }
}
