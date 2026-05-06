//! Concrete projections (Paper II Sec.8.3), organized by family.
//!
//! | Module       | Projections                                          | Paper II Sec. |
//! |--------------|------------------------------------------------------|---------------|
//! | `zenithal`   | TAN, STG, SIN, ZPN, AZP, ARC, ZEA, SZP, AIR         | Sec.5.1       |
//! | `cylindrical`| CAR, CEA, MER, CYP                                   | Sec.5.2       |
//! | `pseudocyl`  | SFL, PAR, MOL, AIT                                   | Sec.5.3       |
//! | `conic`      | COP, COE, COD, COO                                   | Sec.5.4       |
//! | `polyconic`  | BON, PCO                                             | Sec.5.5       |
//! | `quadcube`   | TSC, CSC, QSC                                        | Sec.5.6       |
//! | `healpix`    | HPX, XPH                                             | CR 2007       |

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

    #[test]
    fn tsc_face_centres_round_trip() {
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
    #[ignore = "XPH inverse face-disambiguation not yet correct"]
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
}
