//! Build and write a 2D image FITS file from scratch.
//!
//! Run from the repo root:
//!
//!     cargo run --example write_image

use fitsy::{FitsWriter, ImageBuilder, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("fitsy_example_image.fits");

    // Synthesize a 64x48 f32 ramp.
    let (nx, ny) = (64_u64, 48_u64);
    let mut data = Vec::with_capacity((nx * ny) as usize);
    for y in 0..ny {
        for x in 0..nx {
            data.push(x as f32 + 0.01 * y as f32);
        }
    }

    // Builder: axes are FITS-order (NAXIS1 first).
    let (header, bytes) = ImageBuilder::new(vec![nx, ny], data)?
        .primary(true)
        .card("OBJECT", Value::from("synthetic ramp"), Some("test image"))
        .card("BUNIT", Value::from("counts"), None)
        .build()?;

    // Write.
    let mut out = std::fs::File::create(&path)?;
    let mut w = FitsWriter::new(&mut out);
    w.write_hdu(&header, &bytes)?;
    w.finish()?;

    println!(
        "wrote {} ({} bytes)",
        path.display(),
        std::fs::metadata(&path)?.len()
    );
    Ok(())
}
