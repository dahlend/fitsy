//! Read an image FITS file: header, dtype, pixel data.
//!
//! Run from the repo root:
//!
//!     cargo run --example read_image

use fitsy::{FitsFile, Hdu};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = FitsFile::open("examples/data/ngc2403.fits.gz")?;
    println!("HDU count: {}", f.len());

    let Hdu::Image(img) = f.hdu(0)? else {
        return Err("HDU 0 is not an image".into());
    };

    println!("BITPIX: {:?}", img.bitpix());
    println!("axes (NAXIS1, NAXIS2): {:?}", img.axes());

    // A few common header keywords.
    let hdr = img.header();
    if let Some(date) = hdr.first("DATE-OBS") {
        println!("DATE-OBS: {date:?}");
    }
    if let Some(crval1) = hdr.first("CRVAL1") {
        println!("CRVAL1: {crval1:?}");
    }

    // Decode pixels in physical units (BZERO/BSCALE applied -> f64).
    let pixels = img.read_physical()?;
    let data = pixels.as_slice();
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let max = data.iter().copied().fold(f64::MIN, f64::max);
    println!("{n} pixels  mean={mean:.1}  max={max:.1}");

    Ok(())
}
