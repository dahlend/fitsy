//! Parser: build a [`Wcs`] from a [`Header`] for a chosen alternate
//! code (Standard Sec.8.2.6: `' '` is the primary description, `'A'`
//! through `'Z'` are alternates).

use crate::error::{FitsError, Result};
use crate::header::Header;
use crate::header::value::Value;
use crate::wcs::Wcs;
use crate::wcs::celestial::{CelestialFrame, CelestialRotation, RadeSys};
use crate::wcs::celestial_block::{CelestialBlock, CelestialPair};
use crate::wcs::dss::Dss;
use crate::wcs::linear::LinearTransform;
use crate::wcs::projection::{self, Projection, ProjectionKind};
use crate::wcs::sip::{Sip, SipPoly};
use crate::wcs::spectral::{SpectralAlgorithm, SpectralAxis, SpectralKind};
use crate::wcs::tab::TabSpec;
use crate::wcs::tnx::Tnx;
use crate::wcs::tpv::{Tpv, TpvAxis};
use crate::wcs::wat;

impl Wcs {
    /// Parse the WCS for alternate `alt` (use `' '` for the primary).
    /// Returns `Ok(None)` if the header carries no recognizable WCS
    /// for that alternate.
    pub fn from_header(header: &Header, alt: char) -> Result<Option<Self>> {
        if !alt.is_ascii() {
            return Err(FitsError::Header(format!(
                "WCS alt must be an ASCII character (got {alt:?})"
            )));
        }
        let header_naxis = header.naxis()?;
        if header_naxis == 0 {
            return Ok(None);
        }
        let alt_suffix: String = if alt == ' ' {
            String::new()
        } else {
            alt.to_string()
        };

        // WCSAXES (Standard Sec.8.2): may declare more (or fewer) WCS
        // axes than NAXIS. When present it overrides NAXIS for the
        // dimensionality of the WCS pipeline.
        let naxis = match header.first(&format!("WCSAXES{alt_suffix}")) {
            Some(Value::Integer(n)) if *n > 0 => *n as usize,
            _ => header_naxis,
        };

        // Required: CRPIX, CRVAL, CTYPE per axis.
        let mut crpix = Vec::with_capacity(naxis);
        let mut crval = Vec::with_capacity(naxis);
        let mut ctype = Vec::with_capacity(naxis);
        let mut cunit = Vec::with_capacity(naxis);
        let mut cdelt = Vec::with_capacity(naxis);
        let mut have_any_wcs_keyword = false;

        for i in 1..=naxis {
            crpix.push(read_real(
                header,
                &format!("CRPIX{i}{alt_suffix}"),
                0.0,
                &mut have_any_wcs_keyword,
            ));
            crval.push(read_real(
                header,
                &format!("CRVAL{i}{alt_suffix}"),
                0.0,
                &mut have_any_wcs_keyword,
            ));
            cdelt.push(read_real(
                header,
                &format!("CDELT{i}{alt_suffix}"),
                1.0,
                &mut have_any_wcs_keyword,
            ));
            ctype.push(read_string(
                header,
                &format!("CTYPE{i}{alt_suffix}"),
                "",
                &mut have_any_wcs_keyword,
            ));
            cunit.push(read_string(
                header,
                &format!("CUNIT{i}{alt_suffix}"),
                "",
                &mut have_any_wcs_keyword,
            ));
        }

        if !have_any_wcs_keyword {
            return Ok(None);
        }

        // Build linear transform: prefer CDi_j, then PCi_j+CDELT, then
        // CROTAi (legacy), else identity*CDELT.
        let cd = read_matrix(header, "CD", &alt_suffix, naxis)?;
        let pc = read_matrix(header, "PC", &alt_suffix, naxis)?;
        if cd.is_some() && pc.is_some() {
            return Err(FitsError::Wcs(
                "header specifies both CDi_j and PCi_j (mutually exclusive, Sec.8.2.1)".into(),
            ));
        }
        let linear = if let Some(cd) = cd {
            LinearTransform::from_cd(crpix.clone(), cd)?
        } else if let Some(pc) = pc {
            LinearTransform::from_pc(crpix.clone(), cdelt.clone(), pc)?
        } else if naxis == 2
            && let Some(crota) = header.first("CROTA2").and_then(|v| match v {
                Value::Integer(i) => Some(*i as f64),
                Value::Real(r) => Some(*r),
                _ => None,
            })
        {
            LinearTransform::from_crota([crpix[0], crpix[1]], [cdelt[0], cdelt[1]], crota)?
        } else {
            // Identity PC with CDELT scaling.
            let mut id = vec![0.0; naxis * naxis];
            for i in 0..naxis {
                id[i * naxis + i] = 1.0;
            }
            LinearTransform::from_pc(crpix.clone(), cdelt.clone(), id)?
        };

        // IRAF subimage convention (`LTVn`, `LTMi_j`): the WCS as
        // written refers to original (physical) detector pixels, but
        // the array on disk is a subimage in logical pixels with
        // `phys = LTM*log + LTV`. Absorb into the linear pipeline so
        // downstream code keeps working in logical pixel space. Only
        // applied when at least one LTV/LTM keyword is present
        // (otherwise `LTM` defaults to identity and `LTV` to zero,
        // which is a no-op).
        let (ltv, ltm, ltv_ltm_present) = read_iraf_subimage(header, naxis)?;
        let linear = if ltv_ltm_present {
            linear.compose_with_input_affine(&ltm, &ltv)?
        } else {
            linear
        };

        // Identify celestial axis pair from CTYPE prefixes. Then,
        // if present, build every dependent piece (projection,
        // rotation, optional SIP, optional TPV) as a single
        // `CelestialBlock` so the type system enforces the
        // all-or-nothing rule (Paper II Sec.2).
        let celestial_pair = identify_celestial_pair(&ctype);
        let celestial = if let Some(pair) = celestial_pair {
            Some(build_celestial_block(
                header,
                &alt_suffix,
                pair,
                &ctype,
                &crval,
            )?)
        } else {
            None
        };

        // Zero out CRVAL on the celestial axes -- those are absorbed
        // into the rotation; intermediate world for the celestial pair
        // is already in degrees on the projection plane (no offset).
        let mut crval_for_struct = crval.clone();
        if let Some(c) = celestial.as_ref() {
            crval_for_struct[c.pair.lon] = 0.0;
            crval_for_struct[c.pair.lat] = 0.0;
        }

        // Frame metadata (Paper II Sec.3.1).
        // EPOCH is a legacy alias for EQUINOX.
        let equinox = read_optional_real(header, &format!("EQUINOX{alt_suffix}"))
            .or_else(|| read_optional_real(header, "EPOCH"));
        let radesys_kw = match header.first(&format!("RADESYS{alt_suffix}")) {
            Some(Value::String(s)) => Some(s.clone()),
            _ => match header.first("RADECSYS") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            },
        };
        let radesys = radesys_kw.as_deref().map_or_else(
            || RadeSys::default_for_equinox(equinox),
            RadeSys::from_keyword,
        );
        let mjd_obs = read_optional_real(header, "MJD-OBS");

        // WCSNAME (Standard Sec.8.2.6): free-form name for this alternate.
        let wcsname = read_optional_string(header, &format!("WCSNAME{alt_suffix}"));
        // Spectral reference frame keywords (Paper III Sec.7). Stored
        // verbatim -- we do not currently transform between frames.
        let specsys = read_optional_string(header, &format!("SPECSYS{alt_suffix}"));
        let ssysobs = read_optional_string(header, &format!("SSYSOBS{alt_suffix}"));
        let velosys = read_optional_real(header, &format!("VELOSYS{alt_suffix}"));

        // DSS plate solution (non-standard): only meaningful for the
        // primary alternate. When present it replaces the standard
        // celestial pipeline.
        let dss = if alt == ' ' {
            Dss::from_header(header)?
        } else {
            None
        };

        // Spectral axes (Paper III). RESTFRQ/RESTWAV are global to
        // the HDU (no per-alternate variant in the standard).
        let restfrq = read_optional_real(header, "RESTFRQ")
            .or_else(|| read_optional_real(header, "RESTFREQ"));
        let restwav = read_optional_real(header, "RESTWAV");
        let mut spectral: Vec<SpectralAxis> = Vec::new();
        for (i, ct) in ctype.iter().enumerate() {
            if let Some(c) = celestial.as_ref()
                && (i == c.pair.lon || i == c.pair.lat)
            {
                continue;
            }
            let Some((kind, algo)) = parse_spectral_ctype(ct) else {
                continue;
            };
            let sx = SpectralAxis::new(i, kind, algo, crval[i], &cunit[i], restfrq, restwav)?;
            spectral.push(sx);
        }

        // Zero out CRVAL on spectral axes too -- the spectral module
        // reconstructs S from the intermediate coordinate around
        // CRVAL internally, so the linear pass must add 0.
        for sx in &spectral {
            crval_for_struct[sx.axis] = 0.0;
        }

        // Tabular `-TAB` axes (Paper III Sec.6). Parse the metadata
        // here; the actual coordinate / index arrays live in a
        // separate BINTABLE extension and must be loaded by the
        // caller via `Wcs::resolve_tab` (or transparently through
        // `FitsFile::wcs`). The CRVAL contribution is zeroed: TAB
        // returns the world coordinate directly from the lookup.
        let mut tab_specs: Vec<TabSpec> = Vec::new();
        for (i, ct) in ctype.iter().enumerate() {
            if !is_tab_ctype(ct) {
                continue;
            }
            // -TAB on a celestial axis would require the full
            // multi-D algorithm (the longitude/latitude pair shares
            // a 2-D coordinate array). Reject rather than silently
            // doing the wrong thing.
            if let Some(c) = celestial.as_ref()
                && (i == c.pair.lon || i == c.pair.lat)
            {
                return Err(FitsError::Wcs(
                    "celestial -TAB axes (multi-dimensional lookup) are not supported".into(),
                ));
            }
            tab_specs.push(read_tab_spec(header, &alt_suffix, i)?);
            // Note: unlike spectral axes, we do NOT zero CRVAL for
            // -TAB axes. Paper III Sec.6 specifies that the lookup
            // operates on the full intermediate world coordinate
            // `xi = CRVAL + (PC * (p - CRPIX)) * CDELT`, so CRVAL
            // must remain in the linear pipeline output.
        }

        Ok(Some(Self {
            naxis,
            linear,
            ctype,
            cunit,
            crval: crval_for_struct,
            celestial,
            spectral,
            radesys,
            equinox,
            mjd_obs,
            wcsname,
            specsys,
            ssysobs,
            velosys,
            dss,
            tab_specs,
            tab: Vec::new(),
        }))
    }
}

/// Assemble the [`CelestialBlock`] for a header that has a celestial
/// axis pair. Pulled out of [`Wcs::from_header`] both to compress that
/// function and to make the all-or-nothing invariant obvious: this
/// helper either returns a fully-populated block or an error.
fn build_celestial_block(
    header: &Header,
    alt_suffix: &str,
    pair: CelestialPair,
    ctype: &[String],
    crval: &[f64],
) -> Result<CelestialBlock> {
    let lat_ctype = &ctype[pair.lat];
    let lon_ctype = &ctype[pair.lon];
    let proj_code = projection_code(lat_ctype)?;
    // Validate that lon/lat CTYPE projection codes agree.
    let lon_code = projection_code(lon_ctype)?;
    if !lon_code.eq_ignore_ascii_case(proj_code) {
        return Err(FitsError::Wcs(format!(
            "celestial CTYPE pair has mismatched projection codes: `{lon_ctype}` vs `{lat_ctype}`"
        )));
    }
    // TPV is signalled by projection code; underlying maths is TAN
    // with polynomial pre-warp on intermediate coords. TNX uses the
    // same slot on TAN; ZPX uses it on ZPN.
    let (kind, is_tpv, is_tnx, is_zpx) = if proj_code.eq_ignore_ascii_case("TPV") {
        (ProjectionKind::Tan, true, false, false)
    } else if proj_code.eq_ignore_ascii_case("TNX") {
        (ProjectionKind::Tan, false, true, false)
    } else if proj_code.eq_ignore_ascii_case("ZPX") {
        (ProjectionKind::Zpn, false, false, true)
    } else {
        (ProjectionKind::from_code(proj_code)?, false, false, false)
    };
    // Collect PV2_m. TPV needs up to PV2_39; standard projections
    // only consume PV2_0..PV2_19.
    let pv_count = if is_tpv { 40 } else { 20 };
    let pv2 = collect_pv(header, pair.lat + 1, alt_suffix, pv_count);
    let projection: Box<dyn Projection> = if is_tpv || is_tnx {
        // TAN takes no PV parameters.
        projection::build(kind, &[])?
    } else {
        projection::build(kind, &pv2[..pv_count.min(20)])?
    };
    let lonpole = read_optional_real(header, &format!("LONPOLE{alt_suffix}"));
    let latpole = read_optional_real(header, &format!("LATPOLE{alt_suffix}"));
    let rotation = CelestialRotation::new(
        crval[pair.lon],
        crval[pair.lat],
        lonpole,
        latpole,
        projection.theta0(),
    )?;
    let tpv = if is_tpv {
        let pv1_pairs = collect_pv_pairs(header, pair.lon + 1, alt_suffix);
        let pv2_pairs = collect_pv_pairs(header, pair.lat + 1, alt_suffix);
        let pv1 = TpvAxis::from_pv_pairs(1, &pv1_pairs)?;
        let pv2 = TpvAxis::from_pv_pairs(2, &pv2_pairs)?;
        Some(Tpv { pv1, pv2 })
    } else {
        None
    };
    // TNX/ZPX polynomial distortion in WAT1_xxx / WAT2_xxx.
    // The IRAF convention writes the longitude axis surface in
    // `WAT<lon+1>_xxx` and the latitude surface in `WAT<lat+1>_xxx`.
    let tnx = if is_tnx || is_zpx {
        let lon_prefix = format!("WAT{}_", pair.lon + 1);
        let lat_prefix = format!("WAT{}_", pair.lat + 1);
        let wat_lon = wat::reassemble(header, &lon_prefix);
        let wat_lat = wat::reassemble(header, &lat_prefix);
        Tnx::from_wat_strings(wat_lon.as_deref(), wat_lat.as_deref())?
    } else {
        None
    };
    // SIP: detected by `-SIP` suffix on the full celestial CTYPE
    // (e.g. `RA---TAN-SIP`). The `projection_code` helper only
    // returns the 3-char code; SIP lives in chars 8+.
    let ct_lat = lat_ctype.to_ascii_uppercase();
    let sip = if ct_lat.len() > 8 && ct_lat[8..].contains("SIP") {
        Some(read_sip(header, alt_suffix)?)
    } else {
        None
    };
    Ok(CelestialBlock {
        pair,
        projection,
        rotation,
        sip,
        tpv,
        tnx,
    })
}

fn read_real(header: &Header, key: &str, default: f64, hit: &mut bool) -> f64 {
    match header.first(key) {
        Some(Value::Integer(i)) => {
            *hit = true;
            *i as f64
        }
        Some(Value::Real(r)) => {
            *hit = true;
            *r
        }
        _ => default,
    }
}

fn read_optional_real(header: &Header, key: &str) -> Option<f64> {
    match header.first(key)? {
        Value::Integer(i) => Some(*i as f64),
        Value::Real(r) => Some(*r),
        _ => None,
    }
}

fn read_optional_string(header: &Header, key: &str) -> Option<String> {
    match header.first(key)? {
        Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        }
        _ => None,
    }
}

/// Read the IRAF subimage convention (`LTVn`, `LTMi_j`).
///
/// Returns `(ltv, ltm, present)` where `ltm` is row-major `nxn`
/// (defaulting to identity) and `ltv` is length `n` (defaulting to
/// zero). `present` is `true` iff the header contains at least one
/// `LTVn` or `LTMi_j` keyword (i.e. the convention is in effect).
fn read_iraf_subimage(header: &Header, n: usize) -> Result<(Vec<f64>, Vec<f64>, bool)> {
    let mut ltv = vec![0.0; n];
    let mut ltm = vec![0.0; n * n];
    for i in 0..n {
        ltm[i * n + i] = 1.0;
    }
    let mut any = false;
    for i in 1..=n {
        let key = format!("LTV{i}");
        if let Some(v) = header.first(&key) {
            any = true;
            ltv[i - 1] = match v {
                Value::Integer(k) => *k as f64,
                Value::Real(r) => *r,
                _ => return Err(FitsError::Wcs(format!("{key} must be numeric"))),
            };
        }
    }
    for i in 1..=n {
        for j in 1..=n {
            let key = format!("LTM{i}_{j}");
            if let Some(v) = header.first(&key) {
                any = true;
                ltm[(i - 1) * n + (j - 1)] = match v {
                    Value::Integer(k) => *k as f64,
                    Value::Real(r) => *r,
                    _ => return Err(FitsError::Wcs(format!("{key} must be numeric"))),
                };
            }
        }
    }
    Ok((ltv, ltm, any))
}

fn read_string(header: &Header, key: &str, default: &str, hit: &mut bool) -> String {
    match header.first(key) {
        Some(Value::String(s)) => {
            *hit = true;
            s.clone()
        }
        _ => default.to_string(),
    }
}

/// Read an `<name>i_j<alt>` matrix; returns `Ok(None)` if no entries
/// are present. Missing entries default per Sec.8.2.1: PC defaults to
/// the identity, CD defaults to zero (so a CD matrix with any entry
/// present is taken as fully specified -- missing off-diagonals are 0).
fn read_matrix(header: &Header, name: &str, alt: &str, n: usize) -> Result<Option<Vec<f64>>> {
    let mut any = false;
    let mut m = if name == "PC" {
        let mut id = vec![0.0; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        id
    } else {
        vec![0.0; n * n]
    };
    for i in 1..=n {
        for j in 1..=n {
            let key = format!("{name}{i}_{j}{alt}");
            if let Some(v) = header.first(&key) {
                any = true;
                let r = match v {
                    Value::Integer(i) => *i as f64,
                    Value::Real(r) => *r,
                    _ => {
                        return Err(FitsError::Wcs(format!("{key} must be numeric")));
                    }
                };
                m[(i - 1) * n + (j - 1)] = r;
            }
        }
    }
    Ok(if any { Some(m) } else { None })
}

fn collect_pv(header: &Header, axis: usize, alt: &str, count: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(count);
    for m in 0..count {
        let key = format!("PV{axis}_{m}{alt}");
        let v = match header.first(&key) {
            Some(Value::Integer(i)) => *i as f64,
            Some(Value::Real(r)) => *r,
            _ => 0.0,
        };
        out.push(v);
    }
    out
}

/// Collect every present `PV<axis>_<m><alt>` card as `(m, value)`
/// pairs, scanning `m = 0..40`. Used by TPV which needs to know
/// which terms were actually specified (vs left at default 0).
fn collect_pv_pairs(header: &Header, axis: usize, alt: &str) -> Vec<(u32, f64)> {
    let mut out = Vec::new();
    for m in 0..40_u32 {
        let key = format!("PV{axis}_{m}{alt}");
        if let Some(v) = header.first(&key) {
            let val = match v {
                Value::Integer(i) => *i as f64,
                Value::Real(r) => *r,
                _ => continue,
            };
            out.push((m, val));
        }
    }
    out
}

/// Build a [`Sip`] from `A_ORDER`, `A_p_q`, `B_ORDER`, `B_p_q`, plus
/// optional `AP_ORDER`/`AP_p_q`/`BP_ORDER`/`BP_p_q` cards. The SIP
/// convention does not use the per-alternate suffix on these
/// keywords (they live outside the standard WCS namespace).
fn read_sip(header: &Header, _alt: &str) -> Result<Sip> {
    let a_order = read_required_uint(header, "A_ORDER")?;
    let b_order = read_required_uint(header, "B_ORDER")?;
    let a = collect_sip_poly(header, "A", a_order)?;
    let b = collect_sip_poly(header, "B", b_order)?;
    let ap_order = read_optional_uint(header, "AP_ORDER");
    let bp_order = read_optional_uint(header, "BP_ORDER");
    // SIP defines `AP_*` (inverse of `A_*`) and `BP_*` (inverse of
    // `B_*`) as a paired set: either both inverses are tabulated or
    // neither is. A header with only one is malformed -- silently
    // dropping both would force the slow Newton fallback while the
    // user thinks the lookup is being used. Reject loudly.
    let (ap, bp) = match (ap_order, bp_order) {
        (Some(ao), Some(bo)) => (
            Some(collect_sip_poly(header, "AP", ao)?),
            Some(collect_sip_poly(header, "BP", bo)?),
        ),
        (None, None) => (None, None),
        (Some(_), None) => {
            return Err(FitsError::Wcs(
                "SIP: AP_ORDER present without BP_ORDER (Shupe et al. 2005 Sec.3)".into(),
            ));
        }
        (None, Some(_)) => {
            return Err(FitsError::Wcs(
                "SIP: BP_ORDER present without AP_ORDER (Shupe et al. 2005 Sec.3)".into(),
            ));
        }
    };
    Ok(Sip { a, b, ap, bp })
}

fn collect_sip_poly(header: &Header, prefix: &str, order: u32) -> Result<SipPoly> {
    let mut terms = Vec::new();
    for p in 0..=order {
        for q in 0..=(order - p) {
            let key = format!("{prefix}_{p}_{q}");
            if let Some(v) = header.first(&key) {
                let val = match v {
                    Value::Integer(i) => *i as f64,
                    Value::Real(r) => *r,
                    _ => {
                        return Err(FitsError::Wcs(format!("{key} must be numeric")));
                    }
                };
                terms.push((p, q, val));
            }
        }
    }
    SipPoly::from_terms(order, &terms)
}

fn read_required_uint(header: &Header, key: &str) -> Result<u32> {
    match header.first(key) {
        Some(Value::Integer(i)) if *i >= 0 => Ok(*i as u32),
        Some(_) => Err(FitsError::Wcs(format!(
            "{key} must be a non-negative integer"
        ))),
        None => Err(FitsError::Wcs(format!("SIP requires {key}"))),
    }
}

fn read_optional_uint(header: &Header, key: &str) -> Option<u32> {
    match header.first(key) {
        Some(Value::Integer(i)) if *i >= 0 => Some(*i as u32),
        _ => None,
    }
}

fn identify_celestial_pair(ctype: &[String]) -> Option<CelestialPair> {
    // Find a longitude axis and a matching latitude axis. Per Sec.8.3
    // both CTYPE values share the same projection code in chars 5-8.
    for (i, ct) in ctype.iter().enumerate() {
        let p = first4(ct);
        // Skip non-longitude axes (e.g. FREQ, WAVE, linear): they are
        // not the celestial pair. A bare `?` here would have the
        // disastrous effect of aborting the whole search the moment
        // any axis fails the longitude prefix test.
        let Some((frame, _, lat_pref)) =
            CelestialFrame::named_with_prefixes().find(|(_, lon, _)| *lon == p)
        else {
            continue;
        };
        for (j, ct2) in ctype.iter().enumerate() {
            if i == j {
                continue;
            }
            if first4(ct2) == lat_pref {
                return Some(CelestialPair {
                    lon: i,
                    lat: j,
                    frame,
                });
            }
        }
    }
    None
}

fn first4(s: &str) -> &str {
    if s.len() < 4 { s } else { &s[..4] }
}

fn projection_code(ctype: &str) -> Result<&str> {
    if ctype.len() < 8 {
        return Err(FitsError::Wcs(format!(
            "CTYPE `{ctype}` is shorter than 8 chars; cannot extract projection code"
        )));
    }
    // Chars 5-8 are `-CCC` where `CCC` is the projection code.
    let tail = &ctype[4..8];
    if !tail.starts_with('-') {
        return Err(FitsError::Wcs(format!(
            "CTYPE `{ctype}` missing `-` separator before projection code"
        )));
    }
    Ok(&tail[1..])
}

/// Parse a spectral CTYPE per Paper III Sec.3.3. Returns
/// `Some((kind, algorithm))` if the leading 4 chars match a spectral
/// code, with the optional 3-char algorithm parsed from chars 6-8.
fn parse_spectral_ctype(ctype: &str) -> Option<(SpectralKind, Option<SpectralAlgorithm>)> {
    let ct = ctype.trim();
    if ct.len() < 4 {
        return None;
    }
    let head = &ct[..4];
    let kind = SpectralKind::from_code(head)?;
    // Linear: bare 4-char code, optionally padded with spaces or
    // trailing dashes.
    if ct.len() <= 4 || ct[4..].chars().all(|c| c == ' ' || c == '-') {
        return Some((kind, None));
    }
    // Non-linear: chars 5..8 should be "-XXX".
    if ct.len() < 8 {
        return None;
    }
    let tail = &ct[4..8];
    if !tail.starts_with('-') {
        return None;
    }
    let algo = SpectralAlgorithm::from_code(&tail[1..])?;
    Some((kind, Some(algo)))
}

/// True iff `ctype` ends in the `-TAB` algorithm code.
fn is_tab_ctype(ctype: &str) -> bool {
    let ct = ctype.trim();
    ct.len() >= 8 && ct[4..8].eq_ignore_ascii_case("-TAB")
}

/// Parse the `PSi_*<a>` / `PVi_*<a>` keywords describing a `-TAB`
/// axis. `axis` is zero-based; FITS keywords are 1-based.
fn read_tab_spec(header: &Header, alt_suffix: &str, axis: usize) -> Result<TabSpec> {
    let i = axis + 1;
    let extname = match header.first(&format!("PS{i}_0{alt_suffix}")) {
        Some(Value::String(s)) => s.trim().to_string(),
        _ => {
            return Err(FitsError::Wcs(format!(
                "-TAB axis {i}: missing PS{i}_0{alt_suffix} (binary table EXTNAME)"
            )));
        }
    };
    let coord_column = match header.first(&format!("PS{i}_1{alt_suffix}")) {
        Some(Value::String(s)) => s.trim().to_string(),
        _ => {
            return Err(FitsError::Wcs(format!(
                "-TAB axis {i}: missing PS{i}_1{alt_suffix} (coordinate column TTYPE)"
            )));
        }
    };
    let index_column = match header.first(&format!("PS{i}_2{alt_suffix}")) {
        Some(Value::String(s)) => {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        }
        _ => None,
    };
    let extver = match header.first(&format!("PV{i}_1{alt_suffix}")) {
        Some(Value::Integer(v)) => *v,
        Some(Value::Real(r)) => *r as i64,
        _ => 1,
    };
    let coord_axis = match header.first(&format!("PV{i}_3{alt_suffix}")) {
        Some(Value::Integer(v)) if *v > 0 => *v as u32,
        Some(Value::Real(r)) if *r > 0.0 => *r as u32,
        _ => 1,
    };
    Ok(TabSpec {
        axis,
        extname,
        coord_column,
        index_column,
        extver,
        coord_axis,
    })
}
