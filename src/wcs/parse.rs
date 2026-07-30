//! Parser: build a [`Wcs`] from a [`Header`] for a chosen alternate
//! code (Standard Sec.8.2.6: `' '` is the primary description, `'A'`
//! through `'Z'` are alternates).

use std::sync::Arc;

use crate::error::{FitsError, Result};
use crate::header::Header;
use crate::header::value::Value;
use crate::units::{dimensions, factor_to};
use crate::wcs::Wcs;
use crate::wcs::celestial::{CelestialFrame, CelestialRotation, RadeSys};
use crate::wcs::celestial_block::{CelestialBlock, CelestialPair};
use crate::wcs::dss::Dss;
use crate::wcs::linear::LinearTransform;
use crate::wcs::projection::{self, Projection, ProjectionKind};
use crate::wcs::sip::{Sip, SipPoly};
use crate::wcs::spectral::{
    Grism, SourceFrame, SpectralAlgorithm, SpectralAxis, SpectralFrame, SpectralKind,
};
use crate::wcs::tab::TabSpec;
use crate::wcs::time::{PhaseAxis, TimeAxis, is_phase_ctype};
use crate::wcs::tnx::Tnx;
use crate::wcs::tpv::{Tpv, TpvAxis};
use crate::wcs::wat;
use crate::wcs::{Axis, WcsParts};

impl Wcs {
    /// Parse the WCS for alternate `alt` (use `' '` for the primary).
    /// Returns `Ok(None)` if the header carries no recognizable WCS
    /// for that alternate.
    pub fn from_header(header: &Header, alt: char) -> Result<Option<Self>> {
        let alt_suffix = crate::wcs::alt_suffix(alt)?;
        let header_naxis = header.naxis()?;
        if header_naxis == 0 {
            return Ok(None);
        }

        // WCSAXES (Standard Sec.8.2): may declare more (or fewer) WCS
        // axes than NAXIS. When present it overrides NAXIS for the
        // dimensionality of the WCS pipeline.
        let naxis = match header
            .first(&format!("WCSAXES{alt_suffix}"))
            .or_else(|| header.first(&format!("WCSDIM{alt_suffix}")))
        {
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

        // Needed before the linear transform: the legacy CROTAi form
        // below attaches its rotation to the latitude axis.
        let celestial_pair = identify_celestial_pair(&ctype);

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
            LinearTransform::from_cd(crpix.clone(), crval.clone(), cd)?
        } else if let Some(pc) = pc {
            LinearTransform::from_pc(crpix.clone(), crval.clone(), cdelt.clone(), pc)?
        } else if let Some((crota, lon, lat)) =
            read_crota(header, &alt_suffix, celestial_pair, naxis)
        {
            LinearTransform::from_crota(
                crpix.clone(),
                crval.clone(),
                cdelt.clone(),
                crota,
                lon,
                lat,
            )?
        } else {
            // Identity PC with CDELT scaling.
            let mut id = vec![0.0; naxis * naxis];
            for i in 0..naxis {
                id[i * naxis + i] = 1.0;
            }
            LinearTransform::from_pc(crpix.clone(), crval.clone(), cdelt.clone(), id)?
        };

        // IRAF subimage convention (`LTVn`, `LTMi_j`): the WCS refers
        // to original detector pixels, but the array on disk is a
        // subimage with `phys = LTM*log + LTV`. Fold it into the
        // linear pipeline so the rest of the code stays in subimage
        // pixels. Skipped when neither keyword is present, where the
        // defaults make it a no-op.
        let (ltv, ltm, ltv_ltm_present) = read_iraf_subimage(header, naxis)?;
        let linear = if ltv_ltm_present {
            linear.compose_with_input_affine(&ltm, &ltv)?
        } else {
            linear
        };

        // With the celestial axis pair known, build every dependent
        // piece (projection, rotation, optional SIP, optional TPV) as
        // a single `CelestialBlock` so the type system enforces the
        // all-or-nothing rule (Paper II Sec.2).
        // A celestial `-TAB` pair carries no projection at all: its
        // longitude and latitude come straight from a shared
        // coordinate array (Paper III Sec.6.1.1). There is no
        // `CelestialBlock` to build -- `build_celestial_block` would
        // reach the projection lookup and report `TAB` as an unknown
        // projection code -- so the pair is recorded on its own and the
        // `-TAB` machinery supplies the coordinates.
        //
        // Either both members are `-TAB` or neither: half a pair has
        // no defined transform -- the tabular side needs the shared
        // array, the projected side needs its partner in the spherical
        // rotation -- and Paper III Sec.6.1.1 declares unmet `-TAB`
        // group conditions undefined. `wcslib` rejects the same header
        // as "unmatched celestial axes"; transforming it anyway put
        // the projected axis on the bare linear pipeline, a wrong
        // answer disguised as a right one.
        let celestial_is_tabular = match celestial_pair {
            Some(p) => {
                let lon_tab = is_tab_ctype(&ctype[p.lon]);
                let lat_tab = is_tab_ctype(&ctype[p.lat]);
                if lon_tab != lat_tab {
                    return Err(FitsError::Wcs(format!(
                        "CTYPE{}{alt_suffix} = `{}` / CTYPE{}{alt_suffix} = `{}`: a \
                         celestial pair must be -TAB on both axes or neither \
                         (Paper III Sec.6.1.1)",
                        p.lon + 1,
                        ctype[p.lon],
                        p.lat + 1,
                        ctype[p.lat],
                    )));
                }
                lon_tab && lat_tab
            }
            None => false,
        };
        let celestial = if let Some(pair) = celestial_pair.filter(|_| !celestial_is_tabular) {
            Some(build_celestial_block(
                header,
                &alt_suffix,
                pair,
                &ctype,
                &crval,
                &cunit,
            )?)
        } else {
            None
        };

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
        // Retention gates (see the module-level layering note in
        // `wcs`): EQUINOX had to be *read* above because Sec.8.3
        // resolves the frame from it when RADESYS is absent, but it is
        // *kept* only where it means something -- a celestial pair
        // exists, and the resolved frame is not one that defines the
        // equinox away (Paper II Sec.3.1: "not applicable to ICRS or
        // GAPPT"). An unrecognized frame keeps it: dropping data
        // because the RADESYS string was not understood would be
        // strictness against ourselves, not the header.
        let equinox = equinox.filter(|_| {
            celestial_pair.is_some() && !matches!(radesys, RadeSys::Icrs | RadeSys::Gappt)
        });
        let radesys = if celestial_pair.is_some() {
            radesys
        } else {
            RadeSys::default()
        };
        let mjd_obs = header.mjd_obs_utc();

        // Time axis (Sec.9.5.3). Sec.9.2.1 lets a CTYPE name its own
        // scale, otherwise it defers to TIMESYS, which defaults to UTC.
        // Purely descriptive -- the axis stays on the linear pipeline,
        // which is what Sec.9.5.3 defines its transform to be.
        let timesys = header.time_sys();
        let mut time = ctype
            .iter()
            .enumerate()
            .find_map(|(i, ct)| TimeAxis::recognize(i, ct, &timesys));
        // The time reference frame trio (Sec.9.2.3-9.2.5) rides on the
        // axis: with no time axis there is no time for it to locate,
        // so the keywords stay in the source `Header` only.
        if let Some(t) = time.as_mut() {
            t.trefpos = read_optional_string(header, "TREFPOS");
            t.trefdir = read_optional_string(header, "TREFDIR");
            t.plephem = read_optional_string(header, "PLEPHEM");
        }
        // Phase axes (Sec.9.6): `CZPHSia`/`CPERIia` describe the zero
        // point and period of a `'PHASE'` axis and are retained on no
        // other kind.
        let phase: Vec<PhaseAxis> = ctype
            .iter()
            .enumerate()
            .filter(|(_, ct)| is_phase_ctype(ct))
            .map(|(i, _)| PhaseAxis {
                axis: i,
                czphs: read_optional_real(header, &format!("CZPHS{}{alt_suffix}", i + 1)),
                cperi: read_optional_real(header, &format!("CPERI{}{alt_suffix}", i + 1)),
            })
            .collect();
        // WCSNAME (Standard Sec.8.2.6): free-form name for this alternate.
        let wcsname = read_optional_string(header, &format!("WCSNAME{alt_suffix}"));
        // Spectral reference frames (Paper III Sec.7, Standard
        // Sec.8.4.3). Stored, not applied -- and retained only when
        // there is a spectral axis for them to describe. The gate is
        // the CTYPE *type* code, so a bare linear `FREQ` and a
        // `WAVE-TAB` count even though `parse_spectral_ctype` routes
        // both elsewhere: the frame applies to them equally.
        let has_spectral_ctype = ctype
            .iter()
            .any(|ct| SpectralKind::from_code(first4(ct.trim())).is_some());
        let spectral_frame = has_spectral_ctype
            .then(|| {
                // ZSOURCE is the parent of its trio: SSYSSRC names the
                // frame the redshift is expressed in and VELANGL
                // orients the velocity vector, so neither is retained
                // without it (Sec.8.4.3).
                let source =
                    read_optional_real(header, &format!("ZSOURCE{alt_suffix}")).map(|zsource| {
                        SourceFrame {
                            zsource,
                            ssyssrc: read_optional_string(header, &format!("SSYSSRC{alt_suffix}")),
                            velangl: read_optional_real(header, &format!("VELANGL{alt_suffix}")),
                        }
                    });
                SpectralFrame {
                    specsys: read_optional_string(header, &format!("SPECSYS{alt_suffix}")),
                    ssysobs: read_optional_string(header, &format!("SSYSOBS{alt_suffix}")),
                    velosys: read_optional_real(header, &format!("VELOSYS{alt_suffix}")),
                    source,
                }
            })
            .filter(|f| *f != SpectralFrame::default());

        // DSS plate solution (non-standard): only meaningful for the
        // primary alternate. When present it replaces the standard
        // celestial pipeline.
        let dss = if alt == ' ' {
            Dss::from_header(header)?
        } else {
            None
        };

        // Spectral axes (Paper III). `RESTFRQ` and `RESTWAV` take the
        // alternate code like any other WCS keyword -- Table 22 marks
        // both with the suffix, and its footnote 4 spells out that the
        // 7-character forms "can include an alternate version code
        // letter at the end". Without this an alternate description
        // could not carry its own rest quantity, which is exactly the
        // frequency-alongside-wavelength case Sec.8.2.1 exists for.
        //
        // Falling back to the unsuffixed keyword is leniency: Sec.8.2.1
        // says an alternate must repeat every coordinate keyword, but
        // writers routinely give `RESTFRQ` once and add alternates
        // around it, and the primary's value is the only sensible
        // reading of such a header.
        //
        // `RESTFREQ` is the deprecated 8-character spelling, so it has
        // no room for a suffix and is global by construction.
        let restfrq = read_optional_real(header, &format!("RESTFRQ{alt_suffix}"))
            .or_else(|| read_optional_real(header, "RESTFRQ"))
            .or_else(|| read_optional_real(header, "RESTFREQ"));
        let restwav = read_optional_real(header, &format!("RESTWAV{alt_suffix}"))
            .or_else(|| read_optional_real(header, "RESTWAV"));
        let mut spectral: Vec<SpectralAxis> = Vec::new();
        for (i, ct) in ctype.iter().enumerate() {
            if let Some(c) = celestial.as_ref()
                && (i == c.pair.lon || i == c.pair.lat)
            {
                continue;
            }
            let (kind, algo) = match parse_spectral_ctype(ct) {
                SpectralCtype::Recognized(kind, algo) => (kind, algo),
                SpectralCtype::NotSpectral => continue,
                // Falling through to the linear pipeline here would
                // hand back `CRVAL + x` for an axis the standard says
                // is non-linear -- a confidently wrong coordinate, off
                // by orders of magnitude, with no way for the caller
                // to notice. Refuse instead.
                SpectralCtype::Unsupported(code) => {
                    return Err(FitsError::Wcs(format!(
                        "CTYPE{}{alt_suffix} = `{ct}`: `{code}` is not a spectral algorithm \
                         code (Standard Sec.8.4 Table 26)",
                        i + 1,
                    )));
                }
                SpectralCtype::WrongAssociate { code, expected } => {
                    return Err(FitsError::Wcs(format!(
                        "CTYPE{}{alt_suffix} = `{ct}`: algorithm `{code}` names associate \
                         variable `{}`, but this type is associated with `{expected}` \
                         (Paper III Sec.3.3.1)",
                        i + 1,
                        &code[2..],
                    )));
                }
            };
            // Grism dispersers take their parameters from `PVk_m` on
            // the *spectral* axis (Paper III Table 7). Every other
            // algorithm leaves `PVk_m` unused.
            let grism = matches!(algo, Some(SpectralAlgorithm::Grism { .. }))
                .then(|| read_grism(header, &alt_suffix, i));
            let sx =
                SpectralAxis::new(i, kind, algo, crval[i], &cunit[i], restfrq, restwav, grism)?;
            spectral.push(sx);
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
            tab_specs.push(read_tab_spec(header, &alt_suffix, i)?);
            // Unlike the celestial and spectral axes above, the
            // pipeline really does read `crval` for a -TAB axis: the
            // lookup takes the full intermediate world coordinate
            // `xi = CRVAL + (PC * (p - CRPIX)) * CDELT`
            // (Paper III Sec.6).
        }

        // Built last: every `ctype` / `cunit` read above still needs the
        // flat vectors, so they are consumed into the per-axis records
        // only once nothing else wants them. The Sec.8.2 Table 22
        // per-axis keywords (name, coordinate error pair -- the latter
        // the per-axis override of TIMSYER/TIMRDER per Sec.9.4.3) are
        // picked up in the same pass.
        let axes: Vec<Axis> = ctype
            .into_iter()
            .zip(cunit)
            .enumerate()
            .map(|(k, (ctype, cunit))| {
                let i = k + 1;
                Axis {
                    ctype,
                    cunit,
                    cname: read_optional_string(header, &format!("CNAME{i}{alt_suffix}")),
                    crder: read_optional_real(header, &format!("CRDER{i}{alt_suffix}")),
                    csyer: read_optional_real(header, &format!("CSYER{i}{alt_suffix}")),
                }
            })
            .collect();

        Ok(Some(Self::new(
            axes,
            linear,
            WcsParts {
                celestial_pair,
                time,
                phase,
                celestial,
                spectral,
                spectral_frame,
                radesys,
                equinox,
                mjd_obs,
                wcsname,
                dss,
                tab_specs,
                tab: Vec::new(),
                // Snapshot of the image extent, purely for callers; the
                // pipeline above never consults it. `axes()` yields the
                // NAXISn cards, so this is `None` only for a header
                // that has none (NAXIS = 0 is rejected earlier).
                pixel_shape: header.axes().ok().filter(|a| !a.is_empty()),
            },
        )?))
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
    cunit: &[String],
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
    // Collect PV2_m. Sec.8.2 puts `m` in the range 0..=99; ZPN is the
    // projection that reaches highest, with a polynomial in
    // `PV2_0..PV2_20`. Collecting only 20 silently dropped its top
    // term. TPV reuses the same slot for its own 0..=39.
    let pv_count = if is_tpv { 40 } else { PV_MAX + 1 };
    let pv2 = collect_pv(header, pair.lat + 1, alt_suffix, pv_count);
    let projection: Arc<dyn Projection> = if is_tpv || is_tnx {
        // TAN takes no PV parameters.
        projection::build(kind, &[])?
    } else {
        projection::build(kind, &pv2)?
    };
    // Sec.8.2 puts four parameters on the *longitude* axis: `PVi_1`
    // and `PVi_2` are the fiducial point, `PVi_3`/`PVi_4` spell
    // LONPOLE/LATPOLE. TPV, TNX and ZPX reuse `PV<lon>_m` for
    // polynomial coefficients, so skip them there.
    let lon_pv = if is_tpv || is_tnx || is_zpx {
        [None; 4]
    } else {
        let lon_axis = pair.lon + 1;
        std::array::from_fn(|k| {
            read_optional_real(header, &format!("PV{lon_axis}_{}{alt_suffix}", k + 1))
        })
    };
    let phi0 = lon_pv[0].unwrap_or(0.0);
    let theta0 = lon_pv[1].unwrap_or_else(|| projection.theta0());
    let lonpole = read_optional_real(header, &format!("LONPOLE{alt_suffix}")).or(lon_pv[2]);
    let latpole = read_optional_real(header, &format!("LATPOLE{alt_suffix}")).or(lon_pv[3]);
    let rotation = CelestialRotation::new(
        crval[pair.lon],
        crval[pair.lat],
        lonpole,
        latpole,
        phi0,
        theta0,
    )?;
    // Intermediate coordinates are zero at the reference point but
    // the projection measures from its own origin, so a moved
    // fiducial point leaves a constant offset between them.
    let fiducial_offset = if lon_pv[0].is_some() || lon_pv[1].is_some() {
        projection.s2x(phi0, theta0)?
    } else {
        (0.0, 0.0)
    };
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
    // Sec.8.1 requires degrees on a celestial axis, but headers do
    // carry `arcsec` and `rad`, so the declared CUNIT has to be
    // converted -- and checked, since a non-angle here is a broken
    // header rather than something to rescale. Resolved once, not on
    // every transformed point.
    let to_deg = |axis: usize| {
        factor_to(
            cunit.get(axis).map_or("", String::as_str),
            dimensions::ANGLE,
        )
    };
    let cunit_to_deg = (to_deg(pair.lon)?, to_deg(pair.lat)?);
    Ok(CelestialBlock {
        pair,
        projection,
        rotation,
        sip,
        tpv,
        tnx,
        cunit_to_deg,
        fiducial_offset,
    })
}

/// Read the deprecated `CROTAi` rotation (Sec.8.2, Table 22) and the
/// two axes it rotates.
///
/// `CROTAi` is indexed and attached to the latitude axis, so it
/// rotates the celestial pair -- not necessarily axes 1 and 2, and not
/// only in 2-D images. Falls back to `CROTA2` on the first two axes
/// when the header has no celestial pair.
fn read_crota(
    header: &Header,
    alt_suffix: &str,
    pair: Option<CelestialPair>,
    naxis: usize,
) -> Option<(f64, usize, usize)> {
    let (lon, lat) = match pair {
        Some(p) if p.lon < naxis && p.lat < naxis => (p.lon, p.lat),
        _ if naxis >= 2 => (0, 1),
        _ => return None,
    };
    let value = read_optional_real(header, &format!("CROTA{}{alt_suffix}", lat + 1))
        // Some writers put the rotation on the longitude axis instead.
        .or_else(|| read_optional_real(header, &format!("CROTA{}{alt_suffix}", lon + 1)))?;
    Some((value, lon, lat))
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

/// Highest `PVi_m` index any projection in Paper II consumes: ZPN's
/// polynomial runs `PV2_0..PV2_20`. Sec.8.2 permits `m` up to 99, but
/// no registered projection uses more than this.
const PV_MAX: usize = 20;

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
    //
    // Registered frames first, then the generic Sec.8.4 forms:
    // otherwise `GLON`/`GLAT` matches the generic rule and reports
    // `CelestialFrame::Other`.
    for named_only in [true, false] {
        for (i, ct) in ctype.iter().enumerate() {
            let p = first4(ct);
            // Skip non-longitude axes (FREQ, WAVE, linear): a `?` here
            // would abort the search on the first one instead.
            let Some((frame, lat_pref)) = CelestialFrame::lat_prefix_for(p) else {
                continue;
            };
            if named_only != (frame != CelestialFrame::Other) {
                continue;
            }
            for (j, ct2) in ctype.iter().enumerate() {
                if i == j {
                    continue;
                }
                if first4(ct2).eq_ignore_ascii_case(&lat_pref) {
                    return Some(CelestialPair {
                        lon: i,
                        lat: j,
                        frame,
                    });
                }
            }
        }
    }
    None
}

/// The leading 4 characters, for matching a CTYPE against the
/// celestial axis prefixes.
///
/// `get`, not slicing: a CTYPE whose fourth byte falls inside a
/// multi-byte character would otherwise panic. The card reader keeps
/// non-ASCII out of a value read from a file, but a `Header` built
/// programmatically can hold anything, and a parser should refuse
/// input rather than abort on it.
fn first4(s: &str) -> &str {
    if s.len() < 4 {
        s
    } else {
        s.get(..4).unwrap_or(s)
    }
}

fn projection_code(ctype: &str) -> Result<&str> {
    if ctype.len() < 8 {
        return Err(FitsError::Wcs(format!(
            "CTYPE `{ctype}` is shorter than 8 chars; cannot extract projection code"
        )));
    }
    // Chars 5-8 are `-CCC` where `CCC` is the projection code. `get`
    // rather than slicing, for the same reason as `first4`.
    let Some(tail) = ctype.get(4..8) else {
        return Err(FitsError::Wcs(format!(
            "CTYPE `{ctype}` is not ASCII; cannot extract a projection code"
        )));
    };
    if !tail.starts_with('-') {
        return Err(FitsError::Wcs(format!(
            "CTYPE `{ctype}` missing `-` separator before projection code"
        )));
    }
    Ok(&tail[1..])
}

/// Read the grism disperser parameters `PV<k>_0..PV<k>_6` for
/// zero-based spectral axis `axis` (Paper III Sec.5.1.3 Table 7).
///
/// Absent keywords take the table's defaults, which are all zero
/// except `n_r = 1` -- so this cannot use [`collect_pv`], which
/// zero-fills. Units are fixed by the table (SI, angles in degrees)
/// and are deliberately independent of `CUNIT`.
fn read_grism(header: &Header, alt_suffix: &str, axis: usize) -> Grism {
    let k = axis + 1;
    let pv = |m: u32, default: f64| {
        read_optional_real(header, &format!("PV{k}_{m}{alt_suffix}")).unwrap_or(default)
    };
    let d = Grism::default();
    Grism {
        density: pv(0, d.density),
        order: pv(1, d.order),
        alpha: pv(2, d.alpha),
        n_r: pv(3, d.n_r),
        n_r_prime: pv(4, d.n_r_prime),
        epsilon: pv(5, d.epsilon),
        theta: pv(6, d.theta),
    }
}

/// Outcome of matching a `CTYPE` against Paper III Sec.3.3.
enum SpectralCtype {
    /// Not a spectral axis; leave it to the linear pipeline.
    NotSpectral,
    /// Recognized. `None` algorithm means linear in pixel.
    Recognized(SpectralKind, Option<SpectralAlgorithm>),
    /// A spectral type code carrying an algorithm code that is not in
    /// Table 26. Carries the code for the diagnostic.
    Unsupported(String),
    /// A Table 26 code whose associate variable disagrees with the
    /// type's (Paper III Sec.3.3.1).
    WrongAssociate { code: String, expected: char },
}

/// Parse a spectral CTYPE per Paper III Sec.3.3: the leading 4 chars
/// are the coordinate type and chars 6-8 an optional algorithm code.
///
/// Algorithm codes shorter than three characters are blank-padded
/// (Standard Sec.8.2), and string values lose trailing blanks, so the
/// code is matched after trimming rather than at a fixed width.
///
/// `-TAB` reports [`SpectralCtype::NotSpectral`]: those axes are
/// driven by [`crate::wcs::tab`], not by the algorithms here.
fn parse_spectral_ctype(ctype: &str) -> SpectralCtype {
    let ct = ctype.trim();
    // `get`, not slicing: a non-ASCII CTYPE would otherwise panic on a
    // char boundary rather than being reported as non-spectral.
    let (Some(head), Some(rest)) = (ct.get(..4), ct.get(4..)) else {
        return SpectralCtype::NotSpectral;
    };
    let Some(kind) = SpectralKind::from_code(head) else {
        return SpectralCtype::NotSpectral;
    };
    // Linear: bare 4-char code, optionally padded with spaces or
    // trailing dashes.
    let tail = rest.trim_end();
    if tail.is_empty() || tail.chars().all(|c| c == '-') {
        return SpectralCtype::Recognized(kind, None);
    }
    // Non-linear: the remainder is `-XXX`.
    let Some(code) = tail.strip_prefix('-') else {
        return SpectralCtype::NotSpectral;
    };
    let code = code.trim();
    if code.eq_ignore_ascii_case("TAB") {
        return SpectralCtype::NotSpectral;
    }
    match SpectralAlgorithm::from_code(code) {
        Some(algo) => {
            // Paper III Sec.3.3.1: in an `X2P` code the third letter is
            // the type's *associate variable*, which the type already
            // determines. A mismatch is not a different transform, it
            // is a code the paper says does not exist -- `ZOPT-F2V` is
            // its own example, since z goes with lambda, not v.
            if let Some(p) = code
                .as_bytes()
                .get(2)
                .map(|b| b.to_ascii_uppercase() as char)
                && matches!(algo, SpectralAlgorithm::Linear(_))
                && p != kind.associate_letter()
            {
                return SpectralCtype::WrongAssociate {
                    code: code.to_ascii_uppercase(),
                    expected: kind.associate_letter(),
                };
            }
            SpectralCtype::Recognized(kind, Some(algo))
        }
        None => SpectralCtype::Unsupported(code.to_ascii_uppercase()),
    }
}

/// True iff `ctype` ends in the `-TAB` algorithm code.
fn is_tab_ctype(ctype: &str) -> bool {
    let ct = ctype.trim();
    // `get`, not slicing: see `first4`.
    ct.get(4..8)
        .is_some_and(|tail| tail.eq_ignore_ascii_case("-TAB"))
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
    let extlevel = match header.first(&format!("PV{i}_2{alt_suffix}")) {
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
        extlevel,
        coord_axis,
    })
}
