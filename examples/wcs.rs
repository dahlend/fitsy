//! Transform pixel coordinates to sky coordinates.
//!
//! This reads the bundled NGC 2403 plate scan, which carries a `TAN`
//! WCS with SIP distortion. It shows the single-point and batch
//! transforms, the inverse, and the local pixel scale.
//!
//! Run from the repository root:
//!
//!     cargo run --example wcs

use fitsy::{AxisKind, FitsFile, Hdu, Wcs};

fn main() -> Result<(), fitsy::FitsError> {
    let f = FitsFile::open("examples/data/ngc2403.fits.gz")?;

    // FitsFile::wcs(hdu_index, alt_char) resolves -TAB axes automatically.
    // Use ' ' (space) for the primary WCS; 'A'..'Z' for alternates.
    let wcs: Wcs = f.wcs(0, ' ')?.expect("no WCS in HDU 0");

    // `pixel_to_world` returns one value per axis, in axis order.
    // `axis_kinds` says which value is which, so a caller finds an axis
    // by meaning rather than by position -- FITS permits DEC on axis 1.
    let kinds = wcs.axis_kinds();
    let lon = kinds
        .iter()
        .position(|k| *k == AxisKind::Longitude)
        .unwrap();
    let lat = kinds.iter().position(|k| *k == AxisKind::Latitude).unwrap();

    // Single pixel -> sky (0-based pixel coordinates).
    // The center of the first pixel is (0.0, 0.0).
    let world = wcs.pixel_to_world(&[724.0, 1086.0])?;
    let (ra, dec) = (world[lon], world[lat]);
    println!("center:     RA={ra:.4}  Dec={dec:.4}");
    // center:     RA=114.2089  Dec=65.5917

    // Sky -> pixel (round-trip). The result needs no `lon`/`lat`
    // lookup. A pixel coordinate belongs to an axis by position, so
    // entry 0 is axis 1. Only the world side carries a coordinate kind.
    let back = wcs.world_to_pixel(&world)?;
    let (px, py) = (back[0], back[1]);
    println!("round-trip: ({px:.2}, {py:.2})");
    // round-trip: (724.00, 1086.00)

    // Batch transform: corners + center -> sky. Points go in flat,
    // NAXIS values each, and come back in the same layout.
    let pairs = [(0.0_f64, 0.0_f64), (1447.0, 2171.0), (724.0, 1086.0)];
    let flat: Vec<f64> = pairs.iter().flat_map(|&(x, y)| [x, y]).collect();
    let out = wcs.pixel_to_world_many(&flat)?;
    let sky: Vec<(f64, f64)> = out.chunks_exact(2).map(|c| (c[lon], c[lat])).collect();
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

    // The batch form takes the points flat: NAXIS values per point,
    // end to end, and returns the same layout. It builds its working
    // buffers once, so it beats calling the single-point form in a
    // loop. A point outside the projection becomes NaN rather than
    // failing the whole call.
    // Each point comes back in axis order, so `lon` and `lat` from
    // `axis_kinds` above still say which value is which -- the batch
    // form changes the layout, not the meaning of a slot.
    let flat = [0.0, 0.0, 724.0, 1086.0, 1447.0, 2171.0];
    let many = wcs.pixel_to_world_many(&flat)?;
    for point in many.chunks_exact(2) {
        println!("batch:  lon={:.4} lat={:.4}", point[lon], point[lat]);
    }

    // Parsing straight from a Header skips -TAB resolution. It costs
    // less when the image carries no tabular axis.
    if let Hdu::Image(img) = f.hdu(0)? {
        let _wcs2 = Wcs::from_header(img.header(), ' ')?.expect("no WCS");
    }

    Ok(())
}
