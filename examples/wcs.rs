//! WCS pixel-to-sky coordinate transforms using a real FITS image.
//!
//! Run from the repo root with:
//!
//!     cargo run --example wcs

use fitsy::{FitsFile, Hdu, Wcs};

fn main() -> Result<(), fitsy::FitsError> {
    let f = FitsFile::open("examples/data/ngc2403.fits.gz")?;

    // FitsFile::wcs(hdu_index, alt_char) resolves -TAB axes automatically.
    // Use ' ' (space) for the primary WCS; 'A'..'Z' for alternates.
    let wcs: Wcs = f.wcs(0, ' ')?.expect("no WCS in HDU 0");

    // Single pixel -> sky (0-based pixel coordinates).
    // The center of the first pixel is (0.0, 0.0).
    let (ra, dec) = wcs.pixel_to_celestial(724.0, 1086.0)?;
    println!("center:     RA={ra:.4}  Dec={dec:.4}");
    // center:     RA=114.2089  Dec=65.5917

    // Sky -> pixel (round-trip).
    let (px, py) = wcs.celestial_to_pixel(ra, dec)?;
    println!("round-trip: ({px:.2}, {py:.2})");
    // round-trip: (724.00, 1086.00)

    // Batch transform: corners + center -> sky.
    let pairs = vec![(0.0_f64, 0.0_f64), (1447.0, 2171.0), (724.0, 1086.0)];
    let sky = wcs.pixel_to_celestial_many(&pairs)?;
    println!("corners + center:");
    for ((px, py), (ra, dec)) in pairs.iter().zip(&sky) {
        println!("  ({px:.0}, {py:.0}) -> RA={ra:.4}  Dec={dec:.4}");
    }

    // Local pixel scale at the center (arcseconds per pixel, each axis).
    let (sx, sy) = wcs.pixel_scale_at(724.0, 1086.0)?;
    println!("pixel scale: {sx:.4}\" x {sy:.4}\"/px");

    // Full N-axis pixel_to_world / world_to_pixel (useful when the
    // image has non-celestial axes, e.g. spectral).
    let world = wcs.pixel_to_world(&[724.0, 1086.0])?;
    println!("world:  {world:?}");

    // Parsing directly from a Header skips -TAB resolution and is
    // lighter-weight when you know the image has no tabular axes.
    if let Hdu::Image(img) = f.hdu(0)? {
        let _wcs2 = Wcs::from_header(img.header(), ' ')?.expect("no WCS");
    }

    Ok(())
}
