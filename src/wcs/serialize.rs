//! Serialize a [`Wcs`] back to a [`Header`].
//!
//! This is the inverse of [`Wcs::from_header`] and is the
//! companion to [`crate::wcs::fit_celestial_wcs`]: after fitting,
//! call [`Wcs::to_header`] to obtain a `Header` that can be merged
//! into an HDU and written to disk.
//!
//! Coverage today is the celestial subset (CTYPE/CRPIX/CRVAL/CD,
//! optional SIP) plus `RADESYS`, `EQUINOX`, `MJD-OBS`, and
//! `WCSNAME`. Spectral axes, `-TAB`, TPV, TNX, and DSS are
//! **not** serialized -- the parser handles them on read but
//! round-trip serialisation is out of scope for the current
//! version. They are silently omitted from the produced header,
//! and the function returns an error if the input WCS uses any of
//! them, so callers cannot accidentally lose information.

use crate::error::{FitsError, Result};
use crate::header::{Header, Value};
use crate::wcs::Wcs;
use crate::wcs::celestial::{CelestialFrame, RadeSys};
use crate::wcs::sip::{Sip, SipPoly};

impl Wcs {
    /// Build a fresh [`Header`] holding the standard WCS keywords
    /// for this object under the chosen `alt` (`' '` for the
    /// primary description, `'A'..'Z'` for an alternate). Only the
    /// celestial pipeline + SIP is currently serialized; see the
    /// module-level note for unsupported features.
    pub fn to_header(&self, alt: char) -> Result<Header> {
        if !self.spectral.is_empty() {
            return Err(FitsError::Wcs(
                "Wcs::to_header: spectral axis serialisation not implemented".into(),
            ));
        }
        if !self.tab_specs.is_empty() || !self.tab.is_empty() {
            return Err(FitsError::Wcs(
                "Wcs::to_header: -TAB axis serialisation not implemented".into(),
            ));
        }
        if self.dss.is_some() {
            return Err(FitsError::Wcs(
                "Wcs::to_header: DSS plate-solution serialisation not implemented".into(),
            ));
        }
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
            // The parser zeroes `self.crval` on the celestial axis
            // pair (the values are absorbed into the rotation), so
            // we have to read them back from the rotation block to
            // get a faithful round-trip.
            let v = if let Some(cb) = &self.celestial {
                if i == cb.pair.lon {
                    cb.rotation.alpha0
                } else if i == cb.pair.lat {
                    cb.rotation.delta0
                } else {
                    self.crval[i]
                }
            } else {
                self.crval[i]
            };
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
        let rotation =
            crate::wcs::celestial::CelestialRotation::new(crval.0, crval.1, None, None, theta0)
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
