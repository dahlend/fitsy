//! Concrete projections (Paper II Sec.8.3), grouped by family.
//!
//! Each submodule holds one family:
//!
//! - `zenithal` -- TAN, STG, SIN, ZPN, AZP, ARC, ZEA, SZP and AIR
//!   (Paper II Sec.5.1).
//! - `cylindrical` -- CAR, CEA, MER and CYP (Sec.5.2).
//! - `pseudocyl` -- SFL, PAR, MOL and AIT (Sec.5.3).
//! - `conic` -- COP, COE, COD and COO (Sec.5.4).
//! - `polyconic` -- BON and PCO (Sec.5.5).
//! - `quadcube` -- TSC, CSC and QSC (Sec.5.6).
//! - `healpix` -- HPX and XPH (Calabretta & Roukema 2007).

mod conic;
mod cylindrical;
mod healpix;
mod polyconic;
mod pseudocyl;
mod quadcube;
mod zenithal;

pub use conic::{Cod, Coe, Coo, Cop};
pub use cylindrical::{Car, Cea, Cyp, Mer};
pub use healpix::{Hpx, Xph};
pub use polyconic::{Bon, Pco};
pub use pseudocyl::{Ait, Mol, Par, Sfl};
pub use quadcube::{Csc, Qsc, Tsc};
pub use zenithal::{Air, Arc, Azp, Sin, Stg, Szp, Tan, Zea, Zpn};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wcs::R2D;
    use crate::wcs::projection::Projection;

    fn check_round_trip<P: Projection>(p: &P, name: &str) {
        for &phi in &[-170.0_f64, -90.0, -10.0, 0.0, 25.0, 100.0, 170.0] {
            for &theta in &[-80.0_f64, -45.0, -5.0, 0.0, 5.0, 45.0, 80.0] {
                if name == "TAN" && theta <= 0.0 {
                    continue;
                }
                if name == "SIN" && theta < 0.0 {
                    continue;
                }
                let xy = p.s2x(phi, theta);
                let Ok((x, y)) = xy else { continue };
                let (phi2, theta2) = p
                    .x2s(x, y)
                    .unwrap_or_else(|e| panic!("{name}: x2s failed at ({phi},{theta}): {e}"));
                assert!(
                    (theta - theta2).abs() < 1e-8,
                    "{name}: theta {theta} -> {theta2}"
                );
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                if theta.abs() < 89.0 {
                    assert!(
                        dphi.abs() < 1e-8,
                        "{name}: phi {phi} -> {phi2} (delta={dphi})"
                    );
                }
            }
        }
    }

    #[test]
    fn tan_round_trip() {
        check_round_trip(&Tan, "TAN");
    }
    #[test]
    fn stg_round_trip() {
        check_round_trip(&Stg, "STG");
    }
    #[test]
    fn sin_round_trip() {
        check_round_trip(&Sin { xi: 0.0, eta: 0.0 }, "SIN");
    }
    #[test]
    fn arc_round_trip() {
        check_round_trip(&Arc, "ARC");
    }
    #[test]
    fn zea_round_trip() {
        check_round_trip(&Zea, "ZEA");
    }
    #[test]
    fn car_round_trip() {
        check_round_trip(&Car, "CAR");
    }
    #[test]
    fn cea_round_trip() {
        check_round_trip(&Cea { lambda: 1.0 }, "CEA");
    }
    #[test]
    fn mer_round_trip() {
        check_round_trip(&Mer, "MER");
    }
    #[test]
    fn cyp_round_trip() {
        check_round_trip(
            &Cyp {
                mu: 1.0,
                lambda: std::f64::consts::FRAC_1_SQRT_2,
            },
            "CYP",
        );
    }
    #[test]
    fn sfl_round_trip() {
        check_round_trip(&Sfl, "SFL");
    }
    #[test]
    fn par_round_trip() {
        check_round_trip(&Par, "PAR");
    }
    #[test]
    fn mol_round_trip() {
        check_round_trip(&Mol, "MOL");
    }
    #[test]
    fn ait_round_trip() {
        check_round_trip(&Ait, "AIT");
    }

    #[test]
    fn sin_slant_round_trip() {
        let p = Sin {
            xi: 0.05,
            eta: -0.03,
        };
        for &phi in &[-150.0_f64, -45.0, 0.0, 45.0, 150.0] {
            for &theta in &[10.0_f64, 30.0, 60.0, 80.0] {
                let (x, y) = p.s2x(phi, theta).unwrap();
                let (phi2, theta2) = p.x2s(x, y).unwrap();
                assert!((theta - theta2).abs() < 1e-8, "theta {theta} -> {theta2}");
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                assert!(dphi.abs() < 1e-8, "phi {phi} -> {phi2}");
            }
        }
    }

    #[test]
    fn sin_slant_zero_matches_simple() {
        let slant = Sin { xi: 0.0, eta: 0.0 };
        let (x, y) = slant.s2x(30.0, 50.0).unwrap();
        let t = 50.0_f64.to_radians();
        let p = 30.0_f64.to_radians();
        let r = R2D * t.cos();
        assert!((x - r * p.sin()).abs() < 1e-10 && (y - (-r * p.cos())).abs() < 1e-10);
    }

    #[test]
    fn zpn_matches_arc_with_p1_only() {
        let p = Zpn::from_pv(&[0.0, 1.0]).unwrap();
        for &theta in &[-50.0_f64, 0.0, 30.0, 75.0] {
            let (x, y) = p.s2x(45.0, theta).unwrap();
            let (xr, yr) = Arc.s2x(45.0, theta).unwrap();
            assert!((x - xr).abs() < 1e-9 && (y - yr).abs() < 1e-9);
        }
    }

    #[test]
    fn zpn_round_trip_monotonic_polynomial() {
        let p = Zpn::from_pv(&[0.0, 1.0, 0.0, 0.05]).unwrap();
        for &phi in &[-150.0_f64, -45.0, 0.0, 45.0, 150.0] {
            for &theta in &[-30.0_f64, 0.0, 30.0, 60.0, 85.0] {
                let (x, y) = p.s2x(phi, theta).unwrap();
                let (phi2, theta2) = p.x2s(x, y).unwrap();
                assert!((theta - theta2).abs() < 1e-7, "theta {theta} -> {theta2}");
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                if theta.abs() < 89.0 {
                    assert!(dphi.abs() < 1e-7, "phi {phi} -> {phi2}");
                }
            }
        }
    }

    #[test]
    fn azp_round_trip_modest_slant() {
        let p = Azp::from_pv(&[0.0, 2.0, 30.0]).unwrap();
        for &phi in &[-150.0_f64, -45.0, 0.0, 45.0, 150.0] {
            for &theta in &[40.0_f64, 60.0, 80.0] {
                let (x, y) = p.s2x(phi, theta).unwrap();
                let (phi2, theta2) = p.x2s(x, y).unwrap();
                assert!((theta - theta2).abs() < 1e-7, "theta {theta} -> {theta2}");
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                assert!(dphi.abs() < 1e-7, "phi {phi} -> {phi2}");
            }
        }
    }

    #[test]
    fn azp_zero_params_matches_tan() {
        let p = Azp::from_pv(&[0.0, 0.0, 0.0]).unwrap();
        let (x1, y1) = p.s2x(40.0, 60.0).unwrap();
        let (x2, y2) = Tan.s2x(40.0, 60.0).unwrap();
        assert!((x1 - x2).abs() < 1e-10 && (y1 - y2).abs() < 1e-10);
    }

    fn conic_round_trip<P: Projection>(p: &P, theta_a: f64, name: &str) {
        let lats: Vec<f64> = if theta_a > 0.0 {
            vec![
                theta_a - 30.0,
                theta_a - 10.0,
                theta_a,
                theta_a + 10.0,
                theta_a + 30.0,
            ]
        } else {
            vec![
                theta_a + 30.0,
                theta_a + 10.0,
                theta_a,
                theta_a - 10.0,
                theta_a - 30.0,
            ]
        };
        for &phi in &[-90.0_f64, -30.0, 0.0, 30.0, 90.0] {
            for &theta in &lats {
                if !(-89.0..=89.0).contains(&theta) {
                    continue;
                }
                let Ok((x, y)) = p.s2x(phi, theta) else {
                    continue;
                };
                let (phi2, theta2) = p
                    .x2s(x, y)
                    .unwrap_or_else(|e| panic!("{name}: x2s failed at ({phi},{theta}): {e}"));
                assert!(
                    (theta - theta2).abs() < 1e-7,
                    "{name}: theta {theta} -> {theta2}"
                );
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                assert!(dphi.abs() < 1e-7, "{name}: phi {phi} -> {phi2}");
            }
        }
    }

    #[test]
    fn cop_round_trip_north() {
        conic_round_trip(&Cop::from_pv(&[0.0, 45.0, 15.0]).unwrap(), 45.0, "COP");
    }
    #[test]
    fn cop_round_trip_south() {
        conic_round_trip(&Cop::from_pv(&[0.0, -30.0, 10.0]).unwrap(), -30.0, "COP-S");
    }
    #[test]
    fn coe_round_trip() {
        conic_round_trip(&Coe::from_pv(&[0.0, 45.0, 15.0]).unwrap(), 45.0, "COE");
    }
    #[test]
    fn cod_round_trip_with_eta() {
        conic_round_trip(&Cod::from_pv(&[0.0, 45.0, 15.0]).unwrap(), 45.0, "COD");
    }
    #[test]
    fn cod_round_trip_no_eta() {
        conic_round_trip(&Cod::from_pv(&[0.0, 60.0, 0.0]).unwrap(), 60.0, "COD-eta0");
    }
    #[test]
    fn coo_round_trip() {
        conic_round_trip(&Coo::from_pv(&[0.0, 45.0, 15.0]).unwrap(), 45.0, "COO");
    }
    #[test]
    fn coo_round_trip_no_eta() {
        conic_round_trip(&Coo::from_pv(&[0.0, 30.0, 0.0]).unwrap(), 30.0, "COO-eta0");
    }

    #[test]
    fn bon_round_trip() {
        let p = Bon::from_pv(&[0.0, 45.0]).unwrap();
        for &phi in &[-90.0_f64, -30.0, 0.0, 30.0, 90.0] {
            for &theta in &[-30.0_f64, 0.0, 30.0, 60.0, 80.0] {
                let (x, y) = p.s2x(phi, theta).unwrap();
                let (phi2, theta2) = p.x2s(x, y).unwrap();
                assert!((theta - theta2).abs() < 1e-7, "theta {theta} -> {theta2}");
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                if theta.abs() < 89.0 {
                    assert!(dphi.abs() < 1e-7, "phi {phi} -> {phi2}");
                }
            }
        }
    }

    #[test]
    fn szp_zero_params_matches_tan() {
        let szp = Szp::from_pv(&[0.0, 0.0, 0.0, 90.0]).unwrap();
        for &(phi, theta) in &[
            (0.0_f64, 90.0_f64),
            (45.0, 60.0),
            (-90.0, 30.0),
            (170.0, 5.0),
        ] {
            let (xs, ys) = szp.s2x(phi, theta).unwrap();
            let (xn, yn) = Tan.s2x(phi, theta).unwrap();
            assert!(
                (xs - xn).abs() < 1e-9 && (ys - yn).abs() < 1e-9,
                "SZP(mu=0) != TAN at ({phi},{theta})"
            );
        }
    }

    #[test]
    fn szp_round_trip() {
        let p = Szp::from_pv(&[0.0, 2.0, 30.0, 60.0]).unwrap();
        for &phi in &[-150.0_f64, -45.0, 0.0, 45.0, 150.0] {
            for &theta in &[20.0_f64, 45.0, 70.0] {
                let Ok((x, y)) = p.s2x(phi, theta) else {
                    continue;
                };
                let (phi2, theta2) = p.x2s(x, y).unwrap();
                assert!(
                    (theta - theta2).abs() < 1e-7,
                    "SZP theta {theta} -> {theta2}"
                );
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                assert!(dphi.abs() < 1e-7, "SZP phi {phi} -> {phi2}");
            }
        }
    }

    #[test]
    fn air_default_round_trip() {
        let p = Air::from_pv(&[0.0]).unwrap();
        for &phi in &[-150.0_f64, 0.0, 90.0] {
            for &theta in &[5.0_f64, 30.0, 60.0, 89.0, -30.0, -60.0] {
                let (x, y) = p.s2x(phi, theta).unwrap();
                let (phi2, theta2) = p.x2s(x, y).unwrap();
                assert!(
                    (theta - theta2).abs() < 1e-7,
                    "AIR theta {theta} -> {theta2}"
                );
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                if theta > -89.0 {
                    assert!(dphi.abs() < 1e-7, "AIR phi {phi} -> {phi2}");
                }
            }
        }
    }

    #[test]
    fn air_with_theta_b_round_trip() {
        let p = Air::from_pv(&[0.0, 45.0]).unwrap();
        for &phi in &[-90.0_f64, 0.0, 90.0] {
            for &theta in &[10.0_f64, 30.0, 60.0, 80.0] {
                let (x, y) = p.s2x(phi, theta).unwrap();
                let (phi2, theta2) = p.x2s(x, y).unwrap();
                assert!(
                    (theta - theta2).abs() < 1e-6,
                    "AIR theta {theta} -> {theta2}"
                );
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                assert!(dphi.abs() < 1e-6, "AIR phi {phi} -> {phi2}");
            }
        }
    }

    #[test]
    fn pco_round_trip() {
        let p = Pco;
        for &phi in &[-150.0_f64, -45.0, 0.0, 45.0, 150.0] {
            for &theta in &[-60.0_f64, -30.0, -5.0, 0.0, 5.0, 30.0, 60.0] {
                let (x, y) = p.s2x(phi, theta).unwrap();
                let (phi2, theta2) = p.x2s(x, y).unwrap();
                assert!(
                    (theta - theta2).abs() < 1e-7,
                    "PCO theta {theta} -> {theta2}"
                );
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                assert!(dphi.abs() < 1e-6, "PCO phi {phi} -> {phi2}");
            }
        }
    }

    #[test]
    fn pco_equator_is_straight() {
        for &phi in &[-170.0_f64, -50.0, 0.0, 50.0, 170.0] {
            let (x, y) = Pco.s2x(phi, 0.0).unwrap();
            assert!((x - phi).abs() < 1e-12 && y.abs() < 1e-12);
        }
    }

    #[test]
    fn hpx_equatorial_round_trip() {
        let p = Hpx::from_pv(&[0.0, 4.0, 3.0]).unwrap();
        for &phi in &[-170.0_f64, -90.0, 0.0, 90.0, 170.0] {
            for &theta in &[-40.0_f64, -10.0, 0.0, 10.0, 40.0] {
                let (x, y) = p.s2x(phi, theta).unwrap();
                let (phi2, theta2) = p.x2s(x, y).unwrap();
                assert!(
                    (theta - theta2).abs() < 1e-9,
                    "HPX-eq theta {theta} -> {theta2}"
                );
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                assert!(dphi.abs() < 1e-9, "HPX-eq phi {phi} -> {phi2}");
            }
        }
    }

    /// `phi = +180` is the closed end of Paper II's `arg` range
    /// `(-180, 180]`: the facet index `floor((phi + 180) H / 360)`
    /// evaluates to `H` there and walked off the ring before the
    /// facet-center clamp -- at `H = 4`, `theta = 88` the projected
    /// `x` came out 223.08 instead of 136.92 (the CHANGELOG's worked
    /// example).
    #[test]
    fn hpx_polar_facet_holds_at_phi_180() {
        let p = Hpx::from_pv(&[0.0, 4.0, 3.0]).unwrap();
        for &theta in &[60.0_f64, 88.0, -60.0, -88.0] {
            let (x, y) = p.s2x(180.0, theta).unwrap();
            assert!(
                (90.0..=180.0).contains(&x),
                "theta {theta}: x = {x} left the last facet"
            );
            let (phi2, theta2) = p.x2s(x, y).unwrap();
            let dphi = ((180.0 - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
            assert!(dphi.abs() < 1e-9, "HPX phi 180 -> {phi2} (theta {theta})");
            assert!((theta - theta2).abs() < 1e-9, "theta {theta} -> {theta2}");
        }
        let (x, _) = p.s2x(180.0, 88.0).unwrap();
        assert!((x - 136.92).abs() < 0.01, "x = {x}");
    }

    #[test]
    fn tsc_face_centers_round_trip() {
        let p = Tsc;
        for &(phi, theta) in &[
            (0.0_f64, 0.0_f64),
            (45.0, 0.0),
            (-30.0, 20.0),
            (170.0, -25.0),
            (0.0, 80.0),
            (0.0, -80.0),
        ] {
            let (x, y) = p.s2x(phi, theta).unwrap();
            let (phi2, theta2) = p.x2s(x, y).unwrap();
            assert!(
                (theta - theta2).abs() < 1e-9,
                "TSC theta ({phi},{theta}) -> {theta2}"
            );
            if theta.abs() < 89.0 {
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                assert!(dphi.abs() < 1e-9, "TSC phi ({phi},{theta}) -> {phi2}");
            }
        }
    }

    #[test]
    fn xph_round_trip_off_pole() {
        let p = Xph;
        for &phi in &[-150.0_f64, -60.0, 0.0, 60.0, 150.0] {
            for &theta in &[10.0_f64, 30.0, 60.0, 80.0] {
                let (x, y) = p.s2x(phi, theta).unwrap();
                let (phi2, theta2) = p.x2s(x, y).unwrap();
                assert!(
                    (theta - theta2).abs() < 1e-7,
                    "XPH theta {theta} -> {theta2}"
                );
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                if theta < 89.0 {
                    assert!(dphi.abs() < 1e-7, "XPH phi {phi} -> {phi2} (delta={dphi})");
                }
            }
        }
    }

    #[test]
    fn csc_round_trip() {
        let csc = Csc;
        for &phi in &[-160.0_f64, -90.0, -10.0, 0.0, 25.0, 100.0, 170.0] {
            for &theta in &[-80.0_f64, -30.0, 0.0, 30.0, 80.0] {
                let (x, y) = csc.s2x(phi, theta).unwrap();
                let (phi2, theta2) = csc.x2s(x, y).unwrap();
                assert!(
                    (theta - theta2).abs() < 1e-2,
                    "CSC theta {theta} -> {theta2}"
                );
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                if theta.abs() < 89.0 {
                    assert!(
                        dphi.abs() * theta.to_radians().cos() < 1e-2,
                        "CSC phi {phi} -> {phi2} (delta={dphi})"
                    );
                }
            }
        }
    }

    #[test]
    fn qsc_round_trip() {
        let qsc = Qsc;
        for &phi in &[-160.0_f64, -90.0, -10.0, 0.0, 25.0, 100.0, 170.0] {
            for &theta in &[-70.0_f64, -30.0, 0.0, 30.0, 70.0] {
                let (x, y) = qsc.s2x(phi, theta).unwrap();
                let (phi2, theta2) = qsc.x2s(x, y).unwrap();
                assert!(
                    (theta - theta2).abs() < 1e-6,
                    "QSC theta {theta} -> {theta2} (x={x},y={y})"
                );
                let dphi = ((phi - phi2 + 540.0).rem_euclid(360.0)) - 180.0;
                if theta.abs() < 89.0 {
                    assert!(
                        dphi.abs() * theta.to_radians().cos() < 1e-6,
                        "QSC phi {phi} -> {phi2} (delta={dphi})"
                    );
                }
            }
        }
    }

    #[test]
    fn tan_pole_is_origin() {
        let (x, y) = Tan.s2x(0.0, 90.0).unwrap();
        assert!(x.abs() < 1e-12 && y.abs() < 1e-12);
    }

    #[test]
    fn car_identity() {
        let (x, y) = Car.s2x(42.0, -17.5).unwrap();
        assert_eq!((x, y), (42.0, -17.5));
    }

    /// A dense sweep over *every* registered projection, built the way
    /// a header would build it, asserting that whatever `s2x` accepts
    /// `x2s` inverts.
    ///
    /// The per-projection tests above walk a 7x7 grid; this walks
    /// 45x52 and so reaches the domain edges where the fold bugs in
    /// `AZP`, `SZP` and `COP` lived -- each of those returned a plane
    /// coordinate belonging to a different latitude.
    #[test]
    fn every_registered_projection_inverts_what_it_accepts() {
        use crate::wcs::D2R;
        use crate::wcs::projection::{ProjectionKind, build};

        // Representative parameters for the parameterized codes.
        // One arm per code, mirroring Paper II's parameter tables, even
        // where two codes happen to take the same numbers.
        #[allow(
            clippy::match_same_arms,
            reason = "one arm per projection documents what each PV2_m means"
        )]
        let params = |code: &str| -> Vec<f64> {
            match code {
                "AZP" => vec![0.0, 2.0, 30.0],
                "SZP" => vec![0.0, 2.0, 180.0, 60.0],
                "SIN" => vec![0.0, 0.0, 0.0],
                "ZPN" => vec![0.0, 1.0],
                "AIR" => vec![0.0, 45.0],
                "CYP" => vec![0.0, 1.0, std::f64::consts::FRAC_1_SQRT_2],
                "CEA" => vec![0.0, 1.0],
                "COP" | "COE" | "COD" | "COO" => vec![0.0, 45.0, 25.0],
                "BON" => vec![0.0, 45.0],
                _ => vec![],
            }
        };
        // `CSC` is a polynomial *approximation* (Paper II Sec.5.6.2);
        // its own paper quotes an error near an arcminute, so it gets
        // a matching tolerance rather than a machine-precision one.
        // `SIN`'s limb inverts through `acos`, which loses half the
        // mantissa exactly at `theta = 0`.
        let tol = |code: &str| match code {
            "CSC" => 5e-2,
            "SIN" => 1e-6,
            _ => 1e-9,
        };

        let codes = [
            "AZP", "SZP", "TAN", "STG", "SIN", "ARC", "ZPN", "ZEA", "AIR", "CYP", "CEA", "CAR",
            "MER", "SFL", "PAR", "MOL", "AIT", "COP", "COE", "COD", "COO", "BON", "PCO", "TSC",
            "CSC", "QSC", "HPX", "XPH",
        ];
        for code in codes {
            let kind = ProjectionKind::from_code(code)
                .unwrap_or_else(|e| panic!("{code} is not a registered code: {e}"));
            let p = build(kind, &params(code))
                .unwrap_or_else(|e| panic!("{code} failed to build: {e}"));
            let (mut checked, t) = (0_usize, tol(code));
            let mut theta = -88.0_f64;
            while theta <= 88.0 {
                let mut phi = -179.0_f64;
                while phi <= 179.0 {
                    if let Ok((x, y)) = p.s2x(phi, theta)
                        && x.is_finite()
                        && y.is_finite()
                    {
                        let (p2, t2) = p.x2s(x, y).unwrap_or_else(|e| {
                            panic!("{code}: x2s failed after s2x accepted ({phi}, {theta}): {e}")
                        });
                        // Unit-vector separation: phi is degenerate at
                        // the poles and wraps at +-180, so comparing the
                        // angles directly would flag both as errors.
                        let v = |a: f64, b: f64| {
                            let (c, s) = ((b * D2R).cos(), (b * D2R).sin());
                            [c * (a * D2R).cos(), c * (a * D2R).sin(), s]
                        };
                        let (u, w) = (v(phi, theta), v(p2, t2));
                        let dot = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
                        let cr = [
                            u[1] * w[2] - u[2] * w[1],
                            u[2] * w[0] - u[0] * w[2],
                            u[0] * w[1] - u[1] * w[0],
                        ];
                        let sep = (cr[0].powi(2) + cr[1].powi(2) + cr[2].powi(2))
                            .sqrt()
                            .atan2(dot)
                            / D2R;
                        assert!(
                            sep < t,
                            "{code}: ({phi}, {theta}) -> ({x}, {y}) -> ({p2}, {t2}), \
                             off by {sep:.3e} deg (tol {t:.0e})"
                        );
                        checked += 1;
                    }
                    phi += 7.0;
                }
                theta += 4.0;
            }
            assert!(checked > 100, "{code}: only {checked} points accepted");
        }
    }
}
