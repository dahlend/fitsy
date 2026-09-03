//! Move image pixels and a WCS into `nalgebra` and back.
//!
//! The `nalgebra` feature adds conversions between fitsy's types and
//! `DMatrix`: pixels become a matrix, a matrix becomes an image, the
//! WCS linear transform becomes a matrix, and pixel/sky transforms
//! take a batch of points as columns.
//!
//! Run from the repository root:
//!
//!     cargo run --example nalgebra_interop --features nalgebra

use fitsy::{FitsFile, FitsWriter, ImageBuilder};
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = FitsFile::open("examples/data/ngc2403.fits.gz")?;
    // `image` accepts a plain or a tile-compressed HDU and returns the
    // same type for both, so this example runs against a `.fz` too.
    let img = f.image(0)?;

    // Pixels as a matrix: rows are NAXIS2, columns are NAXIS1.
    let pixels = img.read_physical()?;
    let m: DMatrix<f64> = pixels.to_dmatrix()?;
    println!("image as a matrix: {} rows x {} cols", m.nrows(), m.ncols());
    println!("mean={:.1}  max={:.1}", m.mean(), m.max());

    // Now it is an ordinary matrix, so linear algebra applies. Take a
    // 256x256 corner and subtract its column means.
    let corner = m.view((0, 0), (256, 256)).into_owned();
    let means = corner.row_mean();
    let centered = DMatrix::from_fn(corner.nrows(), corner.ncols(), |r, c| {
        corner[(r, c)] - means[c]
    });
    println!(
        "centered corner: mean={:.3e}  largest residual={:.1}",
        centered.mean(),
        centered.abs().max()
    );

    // The WCS linear transform is a matrix too. `M` maps
    // (pixel - CRPIX) to intermediate world coordinates.
    if let Some(wcs) = f.wcs(0, ' ')? {
        let lin = wcs.linear();
        let matrix = lin.matrix_na();
        println!("linear matrix:\n{matrix:.3e}");
        // `M` and its inverse compose to the identity, up to rounding.
        let round_trip = &matrix * lin.inverse_na();
        println!(
            "M * M^-1 deviates from I by {:.2e}",
            (round_trip - DMatrix::identity(matrix.nrows(), matrix.ncols()))
                .abs()
                .max()
        );

        // Batch transforms take one point per column.
        let pix = DMatrix::from_column_slice(2, 3, &[1.0, 1.0, 512.0, 512.0, 1024.0, 2048.0]);
        let world = wcs.pixel_to_world_na(&pix)?;
        for c in 0..world.ncols() {
            println!(
                "pixel ({:7.1}, {:7.1}) -> ra={:9.5} dec={:9.5}",
                pix[(0, c)],
                pix[(1, c)],
                world[(0, c)],
                world[(1, c)]
            );
        }
    }

    // A matrix goes back to an image the same way. Write the centered
    // corner out as its own file.
    let (header, bytes) = ImageBuilder::from_dmatrix(&centered)?
        .primary(true)
        .build()?
        .into_parts();
    let path = std::env::temp_dir().join("fitsy_nalgebra_corner.fits");
    let mut out = std::fs::File::create(&path)?;
    let mut w = FitsWriter::new(&mut out);
    w.write_hdu_parts(&header, &bytes)?;
    w.finish()?;
    println!("wrote {}", path.display());

    Ok(())
}
