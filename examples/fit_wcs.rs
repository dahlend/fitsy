//! Fit a celestial WCS from pixel and sky pairs.
//!
//! This shows `fit_celestial_wcs` solving for a `TAN` description from
//! four reference points, and the residuals it reports.
//!
//! Run from the repository root:
//!
//!     cargo run --example fit_wcs

use fitsy::wcs::{WcsFitOptions, fit_celestial_wcs};

fn main() -> Result<(), fitsy::FitsError> {
    // Four corner correspondences: pixel (0-based) and sky in degrees.
    let pixels = vec![
        (100.0_f64, 100.0),
        (200.0, 100.0),
        (100.0, 200.0),
        (200.0, 200.0),
    ];
    let sky = vec![
        (10.00_f64, -5.00),
        (10.05, -5.00),
        (10.00, -4.95),
        (10.05, -4.95),
    ];

    // Default: TAN projection, CRPIX solved as a free parameter, no SIP.
    let opts = WcsFitOptions::default();
    let fit = fit_celestial_wcs(&pixels, &sky, &opts)?;

    println!(
        "rms = {:.3}\"  max = {:.3}\"",
        fit.rms_arcsec, fit.max_arcsec
    );
    for (i, (dx, dy)) in fit.residuals_arcsec.iter().enumerate() {
        println!("  point {i}: ({dx:+.3}\", {dy:+.3}\")");
    }

    // The fitted Wcs transforms in both directions.
    // `fit_wcs` always emits RA on axis 1 and Dec on axis 2.
    let world = fit.wcs.pixel_to_world(&[150.0, 150.0])?;
    println!("center: RA={:.4}  Dec={:.4}", world[0], world[1]);

    Ok(())
}
