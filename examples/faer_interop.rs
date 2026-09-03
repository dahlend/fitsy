//! Move image pixels and a WCS into `faer` and back.
//!
//! The `faer` feature adds conversions between fitsy's types and
//! `Mat`: pixels become a matrix, a matrix becomes an image, the WCS
//! linear transform becomes a matrix, and pixel/sky transforms take a
//! batch of points as columns.
//!
//! Run from the repository root:
//!
//!     cargo run --example faer_interop --features faer

use faer::Mat;
use fitsy::{FitsFile, FitsWriter, ImageBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = FitsFile::open("examples/data/ngc2403.fits.gz")?;
    // `image` accepts a plain or a tile-compressed HDU and returns the
    // same type for both, so this example runs against a `.fz` too.
    let img = f.image(0)?;

    // Pixels as a matrix: rows are NAXIS2, columns are NAXIS1.
    let pixels = img.read_physical()?;
    let m: Mat<f64> = pixels.to_faer()?;
    println!("image as a matrix: {} rows x {} cols", m.nrows(), m.ncols());

    // Now it is an ordinary matrix, so linear algebra applies. Take a
    // 256x256 corner and subtract its column means.
    let corner = Mat::from_fn(256, 256, |r, c| m[(r, c)]);
    let means: Vec<f64> = (0..corner.ncols())
        .map(|c| (0..corner.nrows()).map(|r| corner[(r, c)]).sum::<f64>() / corner.nrows() as f64)
        .collect();
    let centered = Mat::from_fn(corner.nrows(), corner.ncols(), |r, c| {
        corner[(r, c)] - means[c]
    });
    let residual = (0..centered.nrows())
        .flat_map(|r| (0..centered.ncols()).map(move |c| (r, c)))
        .fold(0.0_f64, |acc, (r, c)| acc.max(centered[(r, c)].abs()));
    println!("centered corner: largest residual={residual:.1}");

    // The WCS linear transform is a matrix too. `M` maps
    // (pixel - CRPIX) to intermediate world coordinates.
    if let Some(wcs) = f.wcs(0, ' ')? {
        let lin = wcs.linear();
        let matrix = lin.matrix_faer();
        let inverse = lin.inverse_faer();
        println!(
            "linear matrix: [[{:.3e}, {:.3e}], [{:.3e}, {:.3e}]]",
            matrix[(0, 0)],
            matrix[(0, 1)],
            matrix[(1, 0)],
            matrix[(1, 1)]
        );
        // `M` and its inverse compose to the identity, up to rounding.
        let product = &matrix * &inverse;
        let n = matrix.nrows();
        let deviation =
            (0..n)
                .flat_map(|r| (0..n).map(move |c| (r, c)))
                .fold(0.0_f64, |acc, (r, c)| {
                    let want = if r == c { 1.0 } else { 0.0 };
                    acc.max((product[(r, c)] - want).abs())
                });
        println!("M * M^-1 deviates from I by {deviation:.2e}");

        // Batch transforms take one point per column.
        let pix = Mat::from_fn(2, 3, |r, c| {
            [[1.0, 512.0, 1024.0], [1.0, 512.0, 2048.0]][r][c]
        });
        let world = wcs.pixel_to_world_faer(&pix)?;
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
    let (header, bytes) = ImageBuilder::from_faer(&centered)?
        .primary(true)
        .build()?
        .into_parts();
    let path = std::env::temp_dir().join("fitsy_faer_corner.fits");
    let mut out = std::fs::File::create(&path)?;
    let mut w = FitsWriter::new(&mut out);
    w.write_hdu_parts(&header, &bytes)?;
    w.finish()?;
    println!("wrote {}", path.display());

    Ok(())
}
